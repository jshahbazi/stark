import http from 'node:http';
import { constants } from 'node:fs';
import { open, writeFile } from 'node:fs/promises';

export class PilotError extends Error {
  constructor(code) {
    super(code);
    this.code = code;
  }
}

const fail = (code) => { throw new PilotError(code); };
const record = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const positiveInteger = (value) => Number.isSafeInteger(value) && value > 0;
const MAX_BYTES = 256 * 1024;
export const GUARDED_API = 'stark-guarded-chat-completions';

function parseJson(bytes, code) {
  try {
    const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    const value = JSON.parse(text, (_key, item) => {
      if (typeof item === 'number' && !Number.isFinite(item)) fail(code);
      return item;
    });
    // JSON.parse silently discards duplicate fields. Scan valid JSON so escaped
    // spellings of a key are compared too, before a candidate can lose settings.
    const containers = [];
    for (const match of text.matchAll(/"(?:[^"\\]|\\.)*"|[{}\[\]]/g)) {
      const token = match[0];
      if (token === '{') containers.push(new Set());
      else if (token === '[') containers.push(null);
      else if (token === '}' || token === ']') containers.pop();
      else {
        let next = match.index + token.length;
        while (/[ \t\r\n]/.test(text[next] ?? '')) next++;
        if (text[next] !== ':') continue;
        const keys = containers.at(-1);
        const key = JSON.parse(token);
        if (keys.has(key)) fail(code);
        keys.add(key);
      }
    }
    return value;
  } catch { fail(code); }
}

async function readConfiguration(path) {
  let file;
  try {
    file = await open(path, constants.O_RDONLY | constants.O_NONBLOCK);
    const info = await file.stat();
    if (!info.isFile() || info.size > MAX_BYTES) fail('invalid_existing_configuration');
    // Keep the read bounded even if the regular file grows after stat().
    const bytes = Buffer.alloc(MAX_BYTES + 1);
    let length = 0;
    while (length < bytes.length) {
      const read = await file.read(bytes, length, bytes.length - length, null);
      if (read.bytesRead === 0) break;
      length += read.bytesRead;
    }
    if (length > MAX_BYTES) fail('invalid_existing_configuration');
    return parseJson(bytes.subarray(0, length), 'invalid_existing_configuration');
  } catch { fail('invalid_existing_configuration'); }
  finally { if (file) await file.close(); }
}

export function validateBaseUrl(value) {
  // Accept only literal loopback addresses, avoiding DNS and proxy resolution.
  if (typeof value !== 'string' || !/^http:\/\/(127\.0\.0\.1|\[::1\]):[0-9]+\/v1\/?$/.test(value)) {
    fail('invalid_base_url');
  }
  let url;
  try { url = new URL(value); } catch { fail('invalid_base_url'); }
  if (Number(url.port || 80) < 1) fail('invalid_base_url');
  // WHATWG URL drops the default HTTP port. Keep it explicit in public config
  // so the same value remains valid on profile/guard revalidation.
  return `${url.protocol}//${url.hostname}:${url.port || 80}${url.pathname.replace(/\/$/, '')}`;
}

export function validateKey(value) {
  if (typeof value !== 'string' || !/^[\x21-\x7e]{1,4096}$/.test(value)) fail('missing_or_invalid_key');
  return value;
}

async function getJson(url, key) {
  return new Promise((resolve, reject) => {
    const rejectCode = (code) => reject(new PilotError(code));
    // No redirect, proxy, retry or shared connection. Never include payloads in errors.
    const request = http.get(url, {
      agent: false,
      headers: { Authorization: `Bearer ${key}`, Accept: 'application/json' },
    }, (response) => {
      if (response.statusCode !== 200) {
        response.destroy();
        rejectCode(response.statusCode === 401 ? 'authentication_failed' : 'discovery_http_error');
        return;
      }
      if (!/^application\/json(?:;|$)/i.test(response.headers['content-type'] ?? '')) {
        response.destroy();
        rejectCode('invalid_discovery');
        return;
      }
      let length = 0;
      const chunks = [];
      response.on('data', (chunk) => {
        length += chunk.length;
        if (length > MAX_BYTES) {
          response.destroy();
          rejectCode('discovery_too_large');
        } else chunks.push(chunk);
      });
      response.on('error', () => rejectCode('discovery_transport_error'));
      response.on('end', () => {
        try {
          resolve(parseJson(Buffer.concat(chunks), 'invalid_discovery'));
        } catch { rejectCode('invalid_discovery'); }
      });
    });
    const timer = setTimeout(() => {
      request.destroy();
      rejectCode('discovery_timeout');
    }, 5000);
    request.on('close', () => clearTimeout(timer));
    request.on('error', () => rejectCode('discovery_transport_error'));
  });
}

function selectedModel(models, modelId) {
  if (!Array.isArray(models) || models.length !== 1 || !record(models[0]) || models[0].id !== modelId) {
    fail('model_unavailable_or_changed');
  }
  const model = models[0];
  if (model.object !== 'model' || model.owned_by !== 'stark' ||
      model.context_capacity_known !== true || !positiveInteger(model.context_length) ||
      !positiveInteger(model.max_output_tokens) || model.max_output_tokens >= model.context_length) {
    fail('unknown_or_invalid_limits');
  }
  if (model.tool_calling !== true) fail('tools_unsupported');
  return model;
}

function modelSnapshot(model) {
  return JSON.stringify([
    model.id, model.context_length, model.max_output_tokens, model.tool_calling,
    model.capabilities?.mtp, model.defaults?.mtp,
  ]);
}

export function createProvider(baseUrl, modelId, listing, status, finalListing = listing) {
  baseUrl = validateBaseUrl(baseUrl);
  if (typeof modelId !== 'string' || !modelId.trim() || modelId.length > 1024 || /[\x00-\x1f\x7f]/.test(modelId)) {
    fail('invalid_model_id');
  }
  if (listing?.object !== 'list' || finalListing?.object !== 'list') fail('invalid_discovery');
  const model = selectedModel(listing.data, modelId);
  const statusModel = selectedModel(status?.models, modelId);
  const finalModel = selectedModel(finalListing.data, modelId);
  if (modelSnapshot(model) !== modelSnapshot(statusModel) || modelSnapshot(model) !== modelSnapshot(finalModel)) {
    fail('model_unavailable_or_changed');
  }
  const caps = status?.capabilities;
  if (status?.status !== 'ok' || status.model_loaded !== true || !record(caps) ||
      caps.chat_completions !== true || caps.streaming !== true || caps.cancellation !== true) {
    fail('required_capability_missing');
  }
  if (caps.tool_calling !== true || typeof caps.tool_parser !== 'string' || !caps.tool_parser ||
      !Array.isArray(caps.tool_choices) || !['auto', 'none', 'required', 'function'].every((v) => caps.tool_choices.includes(v))) {
    fail('tools_unsupported');
  }
  return {
    name: 'Stark local',
    baseUrl,
    api: GUARDED_API,
    apiKey: '${STARK_API_KEY}',
    authHeader: true,
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      supportsUsageInStreaming: true,
      supportsFinishReason: true,
      supportsLongCacheRetention: false,
      promptCacheKeyMode: 'disabled',
      maxTokensField: 'max_tokens',
    },
    models: [{
      id: model.id,
      name: 'Stark running model',
      reasoning: false,
      input: ['text'],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: model.context_length,
      maxTokens: model.max_output_tokens,
    }],
  };
}

export async function discoverProvider(baseUrl, modelId, key) {
  baseUrl = validateBaseUrl(baseUrl);
  key = validateKey(key);
  const listing = await getJson(`${baseUrl}/models`, key);
  const status = await getJson(`${baseUrl}/capabilities`, key);
  const finalListing = await getJson(`${baseUrl}/models`, key);
  return createProvider(baseUrl, modelId, listing, status, finalListing);
}

export async function writeConfiguration(provider, outputPath, mergePath) {
  let existing = {};
  if (mergePath) existing = await readConfiguration(mergePath);
  if (!record(existing) || (existing.providers !== undefined && !record(existing.providers))) {
    fail('invalid_existing_configuration');
  }
  if (Object.hasOwn(existing.providers ?? {}, 'stark-local')) fail('provider_conflict');
  const config = {
    ...existing,
    $schema: existing.$schema ?? 'https://pi.dev/schemas/models.schema.json',
    providers: { ...existing.providers, 'stark-local': provider },
  };
  try {
    // Exclusive creation prevents overwriting a profile, symlink or concurrent writer.
    await writeFile(outputPath, `${JSON.stringify(config, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
  } catch { fail('output_not_created'); }
}
