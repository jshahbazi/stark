# Security policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/jshahbazi/stark/security/advisories/new). Include the affected commit, platform, impact, and a minimal reproduction using synthetic inputs. Keep credentials, model weights, real prompts, responses, token vectors, decision probabilities, exception dumps, and private evaluation artifacts out of reports and public issues. Coordinate disclosure through the private report.

## Supported code

Stark is an experimental, standalone Rust/Metal inference engine derived from Uzu. Security fixes target the current `main` branch. There are no supported Stark binary releases or published binding distributions. The qualified execution path is the Apple silicon `stark-score` worker and its native engine dependencies. Inherited bindings, tools, and examples remain in the repository and can still contain reportable vulnerabilities; they have not been qualified as Stark products.

## Trust boundaries

The scorer loads a local model directory selected by its caller and communicates through private JSON-lines pipes. The caller owns authentication, authorization, process isolation, and protection of request and response data. The worker does not provide an HTTP service or an operating-system sandbox. Use models and executables from trusted sources; model parsing and native code are still security-relevant attack surfaces. A malicious repository contributor can also supply build scripts or workflow changes.

Requests, model configuration, tokenizer files, model tensors, dependency packages, and pull-request contents are inputs whose validation matters. Local paths and environments must not grant a worker more access than its caller intends. The process owner manages one declared process group; workers that deliberately escape that group are not contained by it.

## Security invariants

- Scoring requests reject invalid fields, types, lengths, context sizes, token IDs, and non-finite numerical parameters before unsafe native operations.
- A complete decision is serialized against one loaded model. The optional in-memory cache serves only identical immediately consecutive token sequences with the same context size, clears on failures and restart, and is never persisted.
- Native diagnostics, panic details, and capture variables must not expose prompts, responses, token vectors, probabilities, credentials, or exception dumps. Protocol messages themselves require private handling.
- Model and tensor parsing must not cause out-of-bounds access, memory corruption, unintended file access, or uncontrolled resource consumption.
- Public CI uses synthetic, model-free tests on disposable hosted runners. It must not download evaluation models, expose repository secrets to contribution code, publish validation artifacts, or activate inherited cloud, signing, or release infrastructure.
- Dependencies remain locked, external Actions remain pinned to immutable commits, and upstream licenses and attribution remain intact.

## Findings and limitations

Report memory safety flaws, unsafe model parsing, sensitive-data disclosure, protocol/cache isolation failures, unintended process or file access, dependency vulnerabilities, and exploitable workflow changes. Establish a reachable input and concrete impact; severity depends on the caller's deployment and privileges. The experimental status and inherited origin do not exempt code from security review.

CI and CodeQL provide automated checks, not proof of runtime safety. CodeQL does not validate Metal kernels or private model-dependent numerical results. GPU parity, performance, converted-model compatibility, and host application integration require separate validation. No dependency vulnerability is accepted merely because an upstream configuration ignored it.
