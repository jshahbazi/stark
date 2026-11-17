import { createHash, randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { lstat, mkdir, open, readdir, rename, rm, rmdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { parseJson, PilotError } from './config.mjs';

const OWNER = 'com.starkengine.pi-default/v1';
const RECEIPT = '.stark-connection.json';
const JOURNAL = '.stark-setup-transaction.json';
const NAMES = ['models.json', 'settings.json', RECEIPT];
const MAX_BYTES = 256 * 1024;
const fail = (code) => { throw new PilotError(code); };
const object = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const digest = (bytes) => bytes === null ? null : createHash('sha256').update(bytes).digest('hex');
const encode = (value) => Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
const quote = (value) => `'${value.replaceAll("'", "'\\''")}'`;
const validPath = (value) => typeof value === 'string' && path.isAbsolute(value) && !/[\x00-\x1f\x7f]/.test(value);

async function safeDirectory(directory, privateMode = false) {
  if (!validPath(directory)) fail('unsafe_default_profile');
  let current = path.parse(directory).root;
  for (const part of path.resolve(directory).split(path.sep).filter(Boolean)) {
    current = path.join(current, part);
    try { await mkdir(current, { mode: 0o700 }); }
    catch (error) { if (error.code !== 'EEXIST') throw error; }
    const info = await lstat(current);
    if (!info.isDirectory() || info.isSymbolicLink()) fail('unsafe_default_profile');
  }
  const info = await lstat(directory);
  if (info.uid !== process.getuid() || (info.mode & (privateMode ? 0o077 : 0o022))) fail('unsafe_default_profile');
}

async function bounded(file) {
  let handle;
  try {
    handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
    const before = await handle.stat();
    if (!before.isFile() || before.nlink !== 1 || before.uid !== process.getuid() ||
        (before.mode & 0o022) || before.size > MAX_BYTES) fail('invalid_default_configuration');
    const buffer = Buffer.alloc(MAX_BYTES + 1);
    let length = 0;
    while (length < buffer.length) {
      const { bytesRead } = await handle.read(buffer, length, buffer.length - length, null);
      if (!bytesRead) break;
      length += bytesRead;
    }
    const after = await handle.stat();
    if (length > MAX_BYTES || length !== before.size || after.mtimeMs !== before.mtimeMs ||
        after.ctimeMs !== before.ctimeMs || after.size !== before.size) fail('configuration_changed');
    return buffer.subarray(0, length);
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    if (error instanceof PilotError) throw error;
    fail('invalid_default_configuration');
  } finally { await handle?.close(); }
}

const decode = (bytes) => bytes === null ? {} : parseJson(bytes, 'invalid_default_configuration');
async function atomic(file, bytes) {
  if (bytes === null) { await rm(file); return; }
  const temporary = `${file}.stark-${randomUUID()}`;
  try {
    await writeFile(temporary, bytes, { mode: 0o600, flag: 'wx' });
    await rename(temporary, file);
  } finally { await rm(temporary, { force: true }); }
}

async function acquireLock(file) {
  try { await mkdir(file, { mode: 0o700 }); }
  catch (error) {
    if (error.code !== 'EEXIST') fail('configuration_busy');
    // Recover only our own abandoned lock. Never remove Pi's lock or an
    // unfamiliar file/directory. A live or reused PID remains a conflict.
    const previous = decode(await bounded(path.join(file, 'stark-owner.json')));
    if (previous.owner !== OWNER || !Number.isSafeInteger(previous.pid) || previous.pid <= 0) fail('configuration_busy');
    try { process.kill(previous.pid, 0); fail('configuration_busy'); }
    catch (probe) { if (probe.code !== 'ESRCH') fail('configuration_busy'); }
    const info = await lstat(file);
    if (!info.isDirectory() || info.isSymbolicLink() || info.uid !== process.getuid() ||
        (await readdir(file)).join() !== 'stark-owner.json') fail('configuration_busy');
    await rm(path.join(file, 'stark-owner.json'));
    await rmdir(file);
    await mkdir(file, { mode: 0o700 });
  }
  const info = await lstat(file);
  const token = randomUUID();
  const marker = path.join(file, 'stark-owner.json');
  let handle, created;
  try {
    handle = await open(marker, 'wx', 0o600);
    created = await handle.stat();
    await handle.writeFile(encode({ owner: OWNER, pid: process.pid, token }));
  } catch {
    // Failed marker creation must not strand an empty lock. Remove only the
    // marker/directory whose identities this invocation actually created.
    try {
      const current = await lstat(file);
      if (current.dev === info.dev && current.ino === info.ino) {
        if (created) {
          const candidate = await lstat(marker);
          if (candidate.dev === created.dev && candidate.ino === created.ino) await rm(marker);
        }
        if ((await readdir(file)).length === 0) await rmdir(file);
      }
    } catch {}
    fail('configuration_busy');
  } finally { await handle?.close(); }
  return { file, device: info.dev, inode: info.ino, token };
}

async function releaseLock(lock) {
  const info = await lstat(lock.file);
  const owner = decode(await bounded(path.join(lock.file, 'stark-owner.json')));
  if (info.dev !== lock.device || info.ino !== lock.inode || owner.token !== lock.token) fail('configuration_busy');
  await rm(path.join(lock.file, 'stark-owner.json'));
  await rmdir(lock.file);
}

async function recover(directory) {
  const bytes = await bounded(path.join(directory, JOURNAL));
  if (bytes === null) return;
  const journal = decode(bytes);
  if (journal.owner !== OWNER || !/^\.stark-backup-[0-9a-f-]{36}$/.test(journal.backup ?? '') ||
      !object(journal.files) || Object.keys(journal.files).sort().join() !== NAMES.toSorted().join()) fail('configuration_recovery_required');
  const backup = path.join(directory, journal.backup);
  await safeDirectory(backup, true);
  const originals = {};
  // Validate the complete recovery before changing any user file.
  for (const name of NAMES) {
    const state = journal.files[name];
    if (!object(state) || (state.before !== null && !/^[0-9a-f]{64}$/.test(state.before ?? '')) ||
        !/^[0-9a-f]{64}$/.test(state.after ?? '')) fail('configuration_recovery_required');
    const before = await bounded(path.join(backup, name));
    if (digest(before) !== state.before) fail('configuration_recovery_required');
    const current = digest(await bounded(path.join(directory, name)));
    if (current !== state.before && current !== state.after) fail('configuration_recovery_required');
    originals[name] = before;
  }
  for (const name of NAMES) {
    if (digest(await bounded(path.join(directory, name))) !== digest(originals[name])) {
      const current = digest(await bounded(path.join(directory, name)));
      if (current !== journal.files[name].before && current !== journal.files[name].after) fail('configuration_recovery_required');
      await atomic(path.join(directory, name), originals[name]);
    }
  }
  await rm(path.join(directory, JOURNAL));
}

/** Save only Stark's provider/extension/default fields. Never change auth.json. */
export async function installDefaultProfile({ directory, profile, provider, keychainService, beforeCommit }) {
  if (!validPath(profile) || !/^com\.starkengine\.menubar\.local-api(?:\.[A-Za-z0-9_.-]+)?$/.test(keychainService ?? '')) {
    fail('invalid_default_configuration');
  }
  await safeDirectory(directory);
  const locks = [];
  try {
    // Pi's settings manager uses proper-lockfile's default <file>.lock directory.
    // Refuse an active/stale lock instead of stealing it from a running client.
    for (const name of NAMES) {
      locks.push(await acquireLock(path.join(directory, `${name}.lock`)));
    }
    await recover(directory);
    const original = Object.fromEntries(await Promise.all(NAMES.map(async (name) => [name, await bounded(path.join(directory, name))])));
    // Pi gives stored auth precedence over a models.json command. Inspect only
    // for this collision; never return, copy, log or modify the auth store.
    const authentication = decode(await bounded(path.join(directory, 'auth.json')));
    if (!object(authentication)) fail('invalid_default_configuration');
    if (Object.hasOwn(authentication, 'stark-local')) fail('credential_conflict');
    const models = decode(original['models.json']), settings = decode(original['settings.json']);
    const receipt = decode(original[RECEIPT]);
    if (!object(models) || !object(settings) || !object(receipt) ||
        (models.providers !== undefined && !object(models.providers)) ||
        (settings.extensions !== undefined && (!Array.isArray(settings.extensions) || settings.extensions.some((v) => typeof v !== 'string')))) {
      fail('invalid_default_configuration');
    }
    const prior = models.providers?.['stark-local'];
    if (original[RECEIPT] !== null && (receipt.owner !== OWNER || receipt.directory !== directory ||
        !object(receipt.provider) || !validPath(receipt.extension))) fail('invalid_default_configuration');
    if (prior !== undefined && (receipt.owner !== OWNER || JSON.stringify(prior) !== JSON.stringify(receipt.provider))) fail('provider_conflict');
    const extension = path.join(profile, 'extensions/stark-stream-guard.mjs');
    const persistent = { ...provider,
      apiKey: `!/usr/bin/security find-generic-password -s ${quote(keychainService)} -a 'bearer-key-v1' -w 2>/dev/null` };
    const extensions = (settings.extensions ?? []).filter((entry) => entry !== receipt.extension && entry !== extension);
    const next = {
      'models.json': encode({ ...models, providers: { ...models.providers, 'stark-local': persistent } }),
      'settings.json': encode({ ...settings, defaultProvider: 'stark-local', defaultModel: provider.models[0].id,
        extensions: [...extensions, extension] }),
      [RECEIPT]: encode({ owner: OWNER, directory, provider: persistent, extension }),
    };
    if (Object.values(next).some((bytes) => bytes.length > MAX_BYTES)) fail('invalid_default_configuration');
    const backup = `.stark-backup-${randomUUID()}`;
    await mkdir(path.join(directory, backup), { mode: 0o700 });
    for (const name of NAMES) {
      if (original[name] !== null) await writeFile(path.join(directory, backup, name), original[name], { flag: 'wx', mode: 0o600 });
    }
    const journal = { owner: OWNER, backup, files: Object.fromEntries(NAMES.map((name) => [name,
      { before: digest(original[name]), after: digest(next[name]) }])) };
    await beforeCommit?.();
    for (const name of NAMES) {
      if (digest(await bounded(path.join(directory, name))) !== digest(original[name])) fail('configuration_changed');
    }
    await writeFile(path.join(directory, JOURNAL), encode(journal), { mode: 0o600, flag: 'wx' });
    try {
      for (const name of NAMES) {
        if (digest(await bounded(path.join(directory, name))) !== digest(original[name])) fail('configuration_changed');
        await atomic(path.join(directory, name), next[name]);
      }
      await rm(path.join(directory, JOURNAL));
    } catch (error) {
      try { await recover(directory); } catch { fail('configuration_recovery_required'); }
      throw error;
    }
    return { commandLineConfigured: true };
  } finally {
    for (const lock of locks.reverse()) await releaseLock(lock);
  }
}
