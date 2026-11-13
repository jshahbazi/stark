import { openAICompletionsApi } from '@earendil-works/pi-ai/compat';
import { VERSION } from '@earendil-works/pi-coding-agent';
import { createGuardedFetch } from '../src/guarded-fetch.mjs';
import { GUARDED_API } from '../src/config.mjs';

export default function starkStreamGuard(pi) {
  if (VERSION !== '1.5.2') throw new Error('Stark requires Pi 1.5.2. Update setup in Stark Connections.');
  const api = openAICompletionsApi();
  pi.registerProvider('stark-local', {
    api: GUARDED_API,
    streamSimple(model, transcript, options = {}) {
      return api.streamSimple(model, transcript, {
        ...options,
        maxRetries: 0,
        fetch: createGuardedFetch({ baseUrl: model.baseUrl, modelId: model.id, fetchImpl: options.fetch }),
      });
    },
  });
}
