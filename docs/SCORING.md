# Probability scoring

Build `stark-score` using the [scoring setup commands](TECHNICAL_OVERVIEW.md#run-the-scoring-worker) and provide a converted model directory through `--score-model-path`. The worker emits readiness followed by one JSON response per request. Readiness identifies Stark, the original Uzu revision, worker mode, and cache policy. The build receipt records selected optimization features.

Operations are `tokenize`, `score`, and `reset_cache`. A score request supplies `input_token_ids`, distinct `candidate_token_ids`, `context_tokens`, and a positive finite `temperature`. Context is capped at 16,384 tokens and each input line at 1 MiB. Unknown fields and invalid arguments are rejected with fixed error codes.

`logits` contain raw candidate scores. `log_probabilities` use log-sum-exp normalization over the complete vocabulary. `candidate_probabilities` use softmax over the requested candidate logits divided by the supplied calibration temperature. This candidate distribution is the calibrated decision readout. Candidate order is preserved. The typed JevK5 decision rule remains in the caller.

## Consecutive cache

Add `--consecutive-cache` after the model path to enable one complete raw f32 vocabulary-logit entry. Its key is the actual token sequence and context size for the immutable loaded model. A hit recomputes normalization and calibration using the current candidates and temperature. Candidate selection is therefore not part of the key.

Only the immediately previous successful score can be reused. A/B/A sequences score A twice. Errors and explicit reset clear the entry. Process restart clears all state. A successful tokenization helper request preserves it. Entries become reusable only after the response is written and flushed successfully. Telemetry reports hits and actual native scoring calls.

The worker suppresses native stdout/stderr diagnostics, panic details, and both Stark and historical Uzu capture variables. The JSON protocol itself contains sensitive material and must be handled through private channels. It does not expose an HTTP listener.

`stark-score-digest` runs the same scorer/cache path and returns a SHA256 digest of the complete raw f32 vocabulary logits for parity testing. It is a validation tool.
