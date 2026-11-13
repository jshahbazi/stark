# OpenCode guided model discovery — 2026-11-11

The local reproduction used official OpenCode Desktop 2.2.7 and Stark's existing Lily Qwen3.8 Flash Next artifact. No inference request was sent and no prompt or response data was collected.

## Cause and correction

The guided API uses a 32,768-token context. The desktop ready summary previously exposed only the model's native output ceiling through `max_output_tokens`; Swift reused it in managed-client metadata. For the larger native context this ceiling exceeds 32,768. The pinned connector correctly rejects model metadata whose output ceiling exceeds its context, so the plugin was loaded and authorized while its Stark provider was empty.

The summary now adds `api_max_output_tokens` from the existing API-specific limit. Swift uses it for managed discovery and bounds the compatibility fallback to API context minus one, preserving the native model and playground limits separately. Both managed-client discovery observations reject zero/oversized output caps. A fresh valid OpenCode observation also clears the obsolete stopped-engine diagnostic.

## Live verification

Installed, signed local app source revision: `6b60af48` (runtime fix). The later controller-fixture update changes tests only.

Before the correction, OpenCode's plugin list contained the managed Stark connector, but its provider registry contained one provider and its model registry contained 12 free models, with no Stark entry. After rebuilding and loading the same model through the guide, the registries contained two providers and 13 models, including Stark. The Stark entry advertised context 32,768 and output 32,767. Closing and reopening the already-open model dialog displayed the Stark group and the running Lily model. The picker was left open; model selection and inference were not performed.

This verifies metadata publication and visible model discovery for that app/version/artifact. It does not measure inference quality or validate a new model/backend.

## Regression checks

- 117 guided-model tests passed, including native 262,144-token capacity with a separate 32,768-token guided API context.
- 147 native controller checks passed, including explicit API output metadata and the older-summary fallback. The first full run exposed ten stale fixture requests from the earlier task/preference UI; those fixtures now use General Work/Coding and Speed/Quality.
- Native desktop connection scenarios passed, including rejection of zero and oversized output caps.
- Targeted backend output-limit test passed.
- The bundled connector rejected synthetic old metadata (32,768 context / 262,143 output) and accepted the corrected 32,767 output cap.
- Repository policy, changed Python lint and diff whitespace checks passed.
