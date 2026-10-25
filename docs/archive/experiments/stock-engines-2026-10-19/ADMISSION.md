# Stock admission history

These setup and driver attempts precede or supplement the measured runs. Their
timings do not contribute to the performance tables.

| Attempt | Outcome and disposition | Evidence |
| --- | --- | --- |
| Initial Qwen admission, all six configurations | Stark, MLX-LM, llama.cpp and both TensorFold modes completed. vLLM's empty successful health response exposed a driver's JSON parsing assumption. Driver failure, not model rejection. | `generation-admission.json` |
| vLLM Qwen admission after status-only health check | All ten fixtures completed through stock generation, including returned token IDs and exact performance output counts. | `vllm-generation-admission.json` |
| JevK5 scoring admission | Stark and llama.cpp completed four sampled sealed cases. MLX-LM 0.32.2 failed unchanged model loading. | `scoring-admission.json` |
| TensorFold 1.1.1 unchanged JevK5 BF16 source | Server exited before ready. No model metadata rewrite or custom port was applied. This proves failure of this stock invocation, without assigning an unobserved internal cause. | `tensorfold-jevk5-admission.json` |
| vLLM Metal unchanged JevK5 BF16 source | Stock MLX loader could not import the flat `qwen3_5_text` model module; initialization failed. No alias shim or model port was applied. | Final entry in `vllm-setup-attempts.json` |
| MetalRT exact Qwen3.5-2B | Stock model eligibility was not established; not installed or measured. The identified Qwen3.5-2B bundle targets Qualcomm QHexRT. | [Model bundle](https://huggingface.co/runanywhere/qwen3_5_2b_HNPU), [official macOS distribution](https://github.com/RunanywhereAI/metalrt-binaries) |

The vLLM setup history also preserves an unsupported logging flag, two startup
attempts with long temporary socket paths, a successful request after moving RPC
files into an owned short temporary directory, and the JevK5 failure. These are
setup failures; they are not throughput measurements or evidence that Qwen is
unsupported. The stock final command omits the unsupported flag, disables
prefix caching, and confines caches to the disposable task root. Request
logging was not enabled. Raw diagnostics and request/response payloads were not
persisted.

The original generation run's llama.cpp adapter used a legacy termination field.
The original report is preserved without retroactive eligibility changes. The
protocol amendment, schema test and separate corrected repeat record the repair.
No engine executable or model representation changed for that repeat.

Only aggregate observations, safe diagnostic categories, exception class names,
counts and hashes are retained. Private paths and full server command arguments
from setup receipts are replaced by the separate configuration templates.
