import http from 'node:http';
import { Readable } from 'node:stream';
import { validateBaseUrl } from './config.mjs';

const MAX_EVENT_CHARS = 1024 * 1024;
const encode = new TextEncoder();
const object = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const count = (v) => Number.isSafeInteger(v) && v >= 0;

export class StreamContractError extends Error {
  constructor(code) { super(`Stark stream contract: ${code}`); this.name = 'StreamContractError'; }
}
const fail = (code) => { throw new StreamContractError(code); };

// This transport serves only the SDK's serialized JSON POST. Node http bypasses
// proxy environment variables and does not follow redirects.
async function directFetch(url, init) {
  const body = init?.body;
  if (typeof body !== 'string' && !(body instanceof Uint8Array)) fail('unsupported_request_body');
  const headers = new Headers(init.headers);
  headers.set('content-length', String(Buffer.byteLength(body)));
  return new Promise((resolve, reject) => {
    const request = http.request(url, {
      method: 'POST', agent: false, signal: init.signal,
      headers: Object.fromEntries(headers),
    }, (response) => {
      try {
        const responseHeaders = new Headers();
        for (const [name, value] of Object.entries(response.headers)) {
          if (value !== undefined) responseHeaders.set(name, Array.isArray(value) ? value.join(', ') : value);
        }
        const empty = [204, 205, 304].includes(response.statusCode);
        resolve(new Response(empty ? null : Readable.toWeb(response), { status: response.statusCode, headers: responseHeaders }));
        if (empty) response.destroy();
      } catch {
        response.destroy();
        reject(new StreamContractError('invalid_response'));
      }
    });
    request.on('error', () => reject(new StreamContractError('transport_failed')));
    request.end(body);
  });
}

function validUsage(usage) {
  if (!object(usage) || !count(usage.prompt_tokens) || !count(usage.completion_tokens) ||
      !count(usage.total_tokens) || usage.total_tokens !== usage.prompt_tokens + usage.completion_tokens) return false;
  const cached = usage.prompt_tokens_details?.cached_tokens;
  return cached === undefined || (count(cached) && cached <= usage.prompt_tokens);
}

export function guardedBody(body, modelId) {
  if (!body) fail('missing_body');
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let pending = '', event = [], eventChars = 0;
  let eof = false, finished = false, usage = false, done = false;

  function validate(lines) {
    const data = [];
    for (const line of lines) {
      if (!line || line.startsWith(':')) continue;
      const colon = line.indexOf(':');
      const field = colon < 0 ? line : line.slice(0, colon);
      let value = colon < 0 ? '' : line.slice(colon + 1);
      if (value.startsWith(' ')) value = value.slice(1);
      if (field === 'data') data.push(value);
      else if (field === 'event' && value === 'error') fail('upstream_error');
    }
    if (!data.length) return null;
    const text = data.join('\n');
    if (text === '[DONE]') {
      if (!finished) fail('missing_finish_reason');
      if (!usage) fail('missing_usage');
      done = true;
      return 'data: [DONE]\n\n';
    }
    let chunk;
    try { chunk = JSON.parse(text); } catch { fail('invalid_json'); }
    if (chunk?.error) fail('upstream_error');
    if (!object(chunk) || chunk.model !== modelId || !Array.isArray(chunk.choices) || chunk.choices.length > 1) {
      fail('invalid_chunk');
    }
    if (chunk.choices.length) {
      if (finished) fail('data_after_finish');
      const choice = chunk.choices[0];
      if (!object(choice) || choice.index !== 0 || !object(choice.delta)) fail('invalid_choice');
      if (choice.finish_reason != null) {
        if (!['stop', 'length', 'tool_calls'].includes(choice.finish_reason)) fail('invalid_finish_reason');
        finished = true;
      }
    }
    if (chunk.usage != null) {
      if (!finished || usage || !validUsage(chunk.usage)) fail('invalid_usage');
      usage = true;
    }
    return `data: ${text}\n\n`;
  }

  async function cleanup() {
    try { await reader.cancel(); } catch { /* Never expose a transport exception. */ }
  }

  return new ReadableStream({
    async pull(controller) {
      try {
        for (;;) {
          const next = pending.search(/[\r\n]/);
          if (next >= 0 && (pending[next] !== '\r' || next + 1 < pending.length || eof)) {
            const line = pending.slice(0, next);
            const width = pending[next] === '\r' && pending[next + 1] === '\n' ? 2 : 1;
            pending = pending.slice(next + width);
            if (line) {
              eventChars += line.length + 1;
              if (eventChars > MAX_EVENT_CHARS) fail('event_too_large');
              event.push(line);
            } else {
              const frame = validate(event);
              event = []; eventChars = 0;
              if (frame) {
                controller.enqueue(encode.encode(frame));
                if (done) { controller.close(); await cleanup(); }
                return;
              }
            }
            continue;
          }
          if (pending.length + eventChars > MAX_EVENT_CHARS) fail('event_too_large');
          if (eof) fail(!finished ? 'missing_finish_reason' : !usage ? 'missing_usage' : 'missing_done');
          const part = await reader.read();
          eof = part.done;
          pending += decoder.decode(part.value, { stream: !eof });
        }
      } catch (error) {
        controller.error(error instanceof StreamContractError ? error : new StreamContractError('transport_or_encoding_failed'));
        await cleanup();
      }
    },
    async cancel() { await cleanup(); },
  }, { highWaterMark: 0 });
}

export function createGuardedFetch({ baseUrl, modelId, fetchImpl = directFetch }) {
  return async (input, init = {}) => {
    let response;
    try {
      const expected = `${validateBaseUrl(baseUrl)}/chat/completions`;
      const url = String(input instanceof Request ? input.url : input);
      // SDKs serialize :80 without a port. Compare the actual endpoint after
      // URL normalization while keeping the configured address explicit.
      if (new URL(url).href !== new URL(expected).href || init.method?.toUpperCase() !== 'POST') fail('invalid_request_route');
      response = await fetchImpl(input, { ...init, redirect: 'error' });
    } catch (error) {
      throw error instanceof StreamContractError ? error : new StreamContractError('transport_failed');
    }
    if (!response.ok) {
      try { await response.body?.cancel(); } catch { /* Scrub raw upstream failures. */ }
      const status = response.status >= 400 && response.status <= 599 ? response.status : 502;
      const headers = new Headers({ 'content-type': 'application/json' });
      const retry = response.headers.get('retry-after');
      if (/^\d{1,2}$/.test(retry ?? '') && Number(retry) <= 60) headers.set('retry-after', retry);
      return new Response(JSON.stringify({ error: {
        message: 'Stark request failed.', type: 'stark_api_error', code: `stark_http_${status}`,
      } }), { status, headers });
    }
    if (!/^text\/event-stream(?:;|$)/i.test(response.headers.get('content-type') ?? '')) {
      try { await response.body?.cancel(); } catch { /* No payload diagnostics. */ }
      fail('invalid_content_type');
    }
    const headers = new Headers(response.headers);
    headers.delete('content-length');
    headers.delete('content-encoding');
    return new Response(guardedBody(response.body, modelId), { status: response.status, headers });
  };
}
