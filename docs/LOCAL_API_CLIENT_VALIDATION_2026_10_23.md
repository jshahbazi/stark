# Local API client validation: 2026-10-23

This report records aggregate results from a scoped client compatibility check. It contains no prompts, responses, credentials, token vectors, private paths or model payloads.

## Runtime and scope

The tested server was the unmodified published Stark 0.3.0 desktop bridge and native worker. The bridge was launched through its actual private control protocol with isolated data. The bridge source revision was `4798f6e98239e0a2afd393f1a8ce8cde14dfa613`; `local_api.py`, `api_request.py` and `desktop.py` matched main at `6c96520d81af2990b5761a62cc116742d5fc87c3`. The host was Apple silicon macOS 26.7.1. This validates the bridge and worker path; it does not qualify the complete menu-bar GUI lifecycle.

The loaded model was the local `qwen3-dense` Stark LanguageModelConfig/Safetensors fixture. Its upstream checkpoint revision was not independently requalified for this check; the hashes below identify the exact artifacts used.

| Artifact | SHA-256 |
| --- | --- |
| Stark 0.3.0 macOS arm64 release archive | `42b9485d410b3c0853170cd27109ca9fb704585fa5ddc98095bb8a356ac951cb` |
| AIChat 0.30.0 macOS arm64 release archive | `680d36ed7f7ba8a2c2490a099df26c9da54495903c27956109a89c87eeb9edb3` |
| Cherry Studio 2.1.7 macOS arm64 release ZIP | `dd242e453735951dbea6a7755903da4ec2d5e19945fad0214e4f8a0db8622ff7` |
| Model configuration | `dfa5763dc220e871b86a6119536f5c1c2862a5137a473eaf9d812be3cc53924e` |
| Tokenizer | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| Weights | `87c3344a80b7376392cc5c44d5369680b68936e6930bb8ac476ca65a136f64c7` |

## Client results

| Client | Version and configuration | Result | Qualification |
| --- | --- | --- | --- |
| `llm` | 0.36; custom OpenAI model; key supplied through an environment variable; `--no-log`; buffered and default streaming invocations | Both passed with non-empty output. Three concurrent requests completed. A later short request after cancellation also passed. | No process restart or model restart was used for the post-cancellation reuse check. |
| `aichat` | 0.30.0; custom OpenAI endpoint; buffered and streaming invocations | Both passed with non-empty output. | Scoped text completion check. |
| Cherry Studio | 2.1.7, macOS arm64; custom OpenAI endpoint; Chat mode; streaming initially enabled; temperature and max-token overrides disabled | Model discovery, provider verification, fresh streaming chat and fresh buffered chat passed. | Follow-up after either reply returned HTTP 400 for unsupported message fields. Work, tools and RAG were not tested. |
| Codex in-app browser | Cross-origin fetch POST with JSON and bearer authentication | Blocked by browser CORS policy. | Expected unsupported path; one browser probe only. |

The client commands used isolated configuration data. The `llm` invocation used the actual CLI with no prompt or response logging and did not put the API key in process arguments. Results are represented by aggregate status and output hashes only.

## Reproduce the client configurations

Run a compatible model with Stark 0.3.0 and use the Base URL and Model ID from Connection Details. Keep keys out of shell arguments and saved test reports. The acquisition used Python 3.13.13 for `llm` 0.36 and the official arm64 AIChat and Cherry Studio release artifacts identified above.

- **LLM:** use a disposable `LLM_USER_PATH` with an `extra-openai-models.yaml` entry setting `model_id` to a local alias, `model_name` to Stark's Model ID, `api_base` to the displayed `/v1` URL, and `api_key_name: openai`. Supply the key through `OPENAI_API_KEY`. Run `llm prompt -m <alias> --no-log -o max_tokens 128 -o temperature 0`; default streaming and `--no-stream` were tested. Synthetic input and output were handled through private pipes. The overlap check launched three such streaming CLI processes against one loaded worker.
- **AIChat:** use a disposable `AICHAT_CONFIG_DIR`/`AICHAT_CONFIG_FILE`, an `openai-compatible` client named `stark`, the displayed `/v1` URL and exact Model ID, and a model output limit of 128. Set `save: false`, `function_calling: false`, `use_tools: null` and temperature 0. Supply the key through `STARK_API_KEY`; route message/log paths to the null device. Default streaming and `--no-stream` were tested.
- **Cherry Studio:** add a custom provider with the displayed API key and `/v1` URL in its OpenAI endpoint field. Sync and select the discovered model, then use **Chat**, rather than Work/agent mode. The initial **Stream output** setting was on; temperature and max-token overrides were off. Test a fresh conversation with streaming on, then another with streaming off. Both first replies succeeded. A second turn in each conversation failed with HTTP 400, `Unsupported message fields.` No particular extra field or workaround was qualified by this acquisition.

The GUI result was checked through aggregate completion/error controls without retaining conversation text in the validation record. Temporary client credentials were invalidated by stopping the isolated server after testing.

## Boundary checks

The browser probe displayed a CORS failure. Source inspection identifies the relevant server boundaries: `OPTIONS` has no handler and returns HTTP 501, while GET/POST requests containing `Origin` return HTTP 403. This acquisition did not separately measure those HTTP statuses or an oversized request body. Remote-device access and container deployments are unsupported.

Cancellation was observed during streaming, and a later short `llm` request completed with output hash `61f456b6557710d2d875aec6232cc3186cc1931dc23e3fb9b1596766e4a97dc0`. This is the real post-cancellation reuse check. It is distinct from an idle capability check and does not establish behavior after a runtime or model restart.

A preliminary cancellation harness attempt produced no valid runtime evidence. The completed cancellation probe and subsequent real client request supply the qualification above; the preliminary attempt is retained separately in private aggregate artifacts.

## Interpretation

The results support basic single-turn buffered and streaming text interoperability for the tested `llm` and `aichat` configurations, plus fresh Cherry Studio Chat requests. Cherry Studio multi-turn chat failed and is not qualified. The three overlapping CLI requests did not exercise queue saturation or the 30-second expiry. Key rotation, client reconnection after app restart, other models, larger workloads, tools/RAG, remote access and the full Stark GUI lifecycle were not tested. These checks do not establish model quality or performance equivalence.
