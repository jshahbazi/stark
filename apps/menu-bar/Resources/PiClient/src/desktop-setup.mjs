import { execFile } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { constants, realpathSync } from 'node:fs';
import { access, lstat, mkdir, mkdtemp, open, realpath, rm, stat, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs, promisify } from 'node:util';
import { discoverProvider, PilotError, validateBaseUrl, writeConfiguration } from './config.mjs';

export const PI_VERSION = '1.5.2';
const OWNER = 'com.starkengine.pi-profile/v1';
const sourceRoot = fileURLToPath(new URL('../', import.meta.url));
const SOURCES = ['package.json', 'LICENSE', 'src/config.mjs', 'src/guarded-fetch.mjs',
  'src/desktop-setup.mjs', 'src/desktop-launch.mjs', 'extensions/stark-stream-guard.mjs'];
const PROFILE_FILES = [...SOURCES, 'agent/models.json', 'agent/settings.json', 'terminal.zsh'];
const fail = (code) => { throw new PilotError(code); };
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const validPath = (value) => typeof value === 'string' && path.isAbsolute(value) && !/[\x00-\x1f\x7f]/.test(value);
const quote = (value) => `'${value.replaceAll("'", "'\\''")}'`;

export function privateEnvironment(piPath) {
  const clientBins = piPath ? [path.dirname(piPath), path.dirname(realpathSync(piPath))] : [];
  return {
    HOME: homedir(), TMPDIR: tmpdir(), LANG: 'en_US.UTF-8', SHELL: '/bin/zsh',
    TERM: /^[\w.+-]{1,64}$/.test(process.env.TERM ?? '') ? process.env.TERM : 'xterm-256color',
    PATH: [...new Set([path.dirname(process.execPath), ...clientBins, path.join(homedir(), '.local/bin'), '/opt/homebrew/bin',
      '/usr/local/bin', '/usr/bin', '/bin', '/usr/sbin', '/sbin'])].join(':'),
  };
}

export async function probePi(piPath) {
  if (!validPath(piPath)) fail('pi_not_found');
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (major < 22 || (major === 22 && minor < 19)) fail('node_version_unsupported');
  const probe = await mkdtemp(path.join(await realpath(tmpdir()), 'stark-pi-version-'));
  try {
    // Pi reads settings before handling --version. Keep even this inspection out
    // of the default agent profile and disable startup network/telemetry.
    const { stdout } = await promisify(execFile)(piPath, ['--offline', '--version'], {
      env: { ...privateEnvironment(piPath), PI_CODING_AGENT_DIR: probe, PI_OFFLINE: '1', PI_TELEMETRY: '0' },
      timeout: 5000, maxBuffer: 4096, windowsHide: true,
    });
    if (stdout.trim() !== PI_VERSION) fail('pi_version_unsupported');
  } catch (error) {
    if (error instanceof PilotError) throw error;
    fail('pi_not_found');
  } finally { await rm(probe, { recursive: true, force: true }); }
  return PI_VERSION;
}

async function privateDirectory(directory, create = false) {
  if (!validPath(directory)) fail('unsafe_setup_location');
  // Do not follow a redirected settings path. All created content is public,
  // but setup must never write through a symlink into an unrelated directory.
  let current = path.parse(directory).root;
  for (const component of path.resolve(directory).split(path.sep).filter(Boolean)) {
    current = path.join(current, component);
    try {
      const info = await lstat(current);
      if (!info.isDirectory() || info.isSymbolicLink()) fail('unsafe_setup_location');
    } catch (error) {
      if (!create || error.code !== 'ENOENT') throw error;
      await mkdir(current, { mode: 0o700 });
    }
  }
  const info = await lstat(directory);
  if (info.uid !== process.getuid() || (info.mode & 0o077)) fail('unsafe_setup_location');
}

async function boundedFile(file, publicResource = false) {
  const handle = await open(file, constants.O_RDONLY | constants.O_NONBLOCK | constants.O_NOFOLLOW);
  try {
    const info = await handle.stat();
    if (!info.isFile() || info.nlink !== 1 ||
        (publicResource ? ![0, process.getuid()].includes(info.uid) || (info.mode & 0o022) : info.uid !== process.getuid()) ||
        info.size > 256 * 1024) fail('invalid_setup');
    const bytes = Buffer.alloc(256 * 1024 + 1);
    let length = 0;
    while (length < bytes.length) {
      const { bytesRead } = await handle.read(bytes, length, bytes.length - length, null);
      if (!bytesRead) break;
      length += bytesRead;
    }
    const after = await handle.stat();
    if (length > 256 * 1024 || length !== info.size || after.size !== info.size ||
        after.mtimeMs !== info.mtimeMs || after.ctimeMs !== info.ctimeMs) fail('invalid_setup');
    return bytes.subarray(0, length);
  } finally { await handle.close(); }
}

function terminalScript(profile, nodePath) {
  return `#!/bin/zsh -f
unsetopt XTRACE VERBOSE
print -r -- 'Stark local model · Pi ${PI_VERSION}'
print -r -- 'Paste the API key copied by Stark. Input is hidden.'
print -r -- 'Pi can read and edit files and run commands in this folder.'
[[ -t 0 ]] || exit 1
read -rs 'STARK_API_KEY?API key: ' || exit 1
print
export STARK_API_KEY
${quote(nodePath)} ${quote(path.join(profile, 'src/desktop-launch.mjs'))} --profile ${quote(profile)} --workspace "$1"
pi_result=$?
unset STARK_API_KEY
if (( pi_result != 0 )); then
  print -r -- 'Pi could not continue. Return to Stark Connections and update setup.'
  read -r '?Press Return to close.'
fi
exit $pi_result
`;
}

export async function setupDesktop({ root, piPath, baseUrl, modelId, key }) {
  await probePi(piPath);
  const provider = await discoverProvider(baseUrl, modelId, key);
  // Match the qualified manual profile's reply budget. API context is kept exact.
  provider.models[0].maxTokens = Math.min(1024, provider.models[0].maxTokens);
  await privateDirectory(root, true);
  const profile = await mkdtemp(path.join(root, 'profile-'));
  try {
    for (const directory of ['agent', 'src', 'extensions']) await mkdir(path.join(profile, directory), { mode: 0o700 });
    for (const file of SOURCES) {
      await writeFile(path.join(profile, file), await boundedFile(path.join(sourceRoot, file), true), { flag: 'wx', mode: 0o600 });
    }
    await writeConfiguration(provider, path.join(profile, 'agent/models.json'));
    await writeFile(path.join(profile, 'agent/settings.json'), `${JSON.stringify({
      retry: { enabled: false, provider: { maxRetries: 0 } }, compaction: { enabled: false },
    }, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
    await writeFile(path.join(profile, 'terminal.zsh'), terminalScript(profile, process.execPath), { flag: 'wx', mode: 0o600 });
    const files = {};
    for (const file of PROFILE_FILES) files[file] = digest(await boundedFile(path.join(profile, file)));
    const receipt = { owner: OWNER, schemaVersion: 1, piVersion: PI_VERSION, piPath,
      nodePath: process.execPath, baseUrl: validateBaseUrl(baseUrl), modelId, files };
    await writeFile(path.join(profile, 'profile.json'), `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
    return { profile, baseUrl: receipt.baseUrl, modelId, version: PI_VERSION };
  } catch (error) {
    await rm(profile, { recursive: true, force: true });
    throw error;
  }
}

export async function readProfile(profile) {
  await privateDirectory(profile);
  const receipt = JSON.parse(await boundedFile(path.join(profile, 'profile.json')));
  if (receipt.owner !== OWNER || receipt.schemaVersion !== 1 || receipt.piVersion !== PI_VERSION ||
      !validPath(receipt.piPath) || !validPath(receipt.nodePath) ||
      typeof receipt.modelId !== 'string' || !receipt.modelId ||
      !receipt.files || Object.keys(receipt.files).sort().join('\n') !== PROFILE_FILES.toSorted().join('\n')) fail('invalid_setup');
  validateBaseUrl(receipt.baseUrl);
  for (const executable of [receipt.nodePath, receipt.piPath]) {
    try {
      if (!(await stat(executable)).isFile()) fail('setup_stale');
      await access(executable, constants.X_OK);
    } catch { fail('setup_stale'); }
  }
  for (const directory of ['agent', 'src', 'extensions']) await privateDirectory(path.join(profile, directory));
  for (const file of PROFILE_FILES) {
    if (digest(await boundedFile(path.join(profile, file))) !== receipt.files[file]) fail('setup_changed');
  }
  // An app update must not silently reopen an older self-contained guard/launcher.
  for (const file of SOURCES) {
    if (digest(await boundedFile(path.join(sourceRoot, file), true)) !== receipt.files[file]) fail('setup_stale');
  }
  return receipt;
}

async function resolveWorkspace(profile, workspace) {
  if (workspace === undefined) {
    const automatic = path.join(path.dirname(profile), 'Workspace');
    await privateDirectory(automatic, true);
    try { return await realpath(automatic); }
    catch { fail('workspace_unavailable'); }
  }
  if (!validPath(workspace)) fail('workspace_unavailable');
  try {
    const resolved = await realpath(workspace);
    if (!(await stat(resolved)).isDirectory()) fail('workspace_unavailable');
    return resolved;
  } catch (error) {
    if (error instanceof PilotError) throw error;
    fail('workspace_unavailable');
  }
}

async function prepareLaunchDetails(profile, workspace) {
  const receipt = await readProfile(profile);
  await probePi(receipt.piPath);
  const resolvedWorkspace = await resolveWorkspace(profile, workspace);
  const environment = privateEnvironment(receipt.piPath);
  const assignments = Object.entries(environment).map(([name, value]) => quote(`${name}=${value}`)).join(' ');
  // env receives only public values here. The API key is read later by terminal.zsh,
  // so it never enters a command line, .command file, config or shell history.
  const script = `#!/bin/zsh -f\nexec /usr/bin/env -i ${assignments} /bin/zsh -f ${quote(path.join(profile, 'terminal.zsh'))} ${quote(resolvedWorkspace)}\n`;
  const command = path.join(profile, `Open Pi-${randomUUID()}.command`);
  await writeFile(command, script, { flag: 'wx', mode: 0o700 });
  return { command, workspace: resolvedWorkspace };
}

export async function prepareLaunch(profile, workspace) {
  return (await prepareLaunchDetails(profile, workspace)).command;
}

async function main() {
  try {
    const { values } = parseArgs({ options: {
      action: { type: 'string' }, pi: { type: 'string' }, root: { type: 'string' },
      'base-url': { type: 'string' }, model: { type: 'string' },
      profile: { type: 'string' }, workspace: { type: 'string' },
    } });
    let result;
    if (values.action === 'inspect') {
      const version = await probePi(values.pi);
      const receipt = values.profile ? await readProfile(values.profile) : null;
      result = { version, ...(receipt ? { profile: values.profile, baseUrl: receipt.baseUrl, modelId: receipt.modelId } : {}) };
    } else if (values.action === 'setup') {
      result = await setupDesktop({ root: values.root, piPath: values.pi, baseUrl: values['base-url'],
        modelId: values.model, key: process.env.STARK_API_KEY });
    } else if (values.action === 'prepare-launch') {
      result = await prepareLaunchDetails(values.profile, values.workspace);
    } else fail('invalid_arguments');
    process.stdout.write(`${JSON.stringify({ ok: true, ...result })}\n`);
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ ok: false, code: error instanceof PilotError ? error.code : 'setup_failed' })}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
