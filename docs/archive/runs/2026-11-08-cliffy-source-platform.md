# Cliffy source platform qualification — 2026-11-08

## Observed result

A controlled stock oracle for `cliffy-config-file-parsing` passed all 38
feature checks and 451 regression checks, with reward 1. Both source and
verifier images were observed as `linux/amd64`. The intervention selected that
platform in a deep copy of the Trial environment config. Original caller
config/env/kwargs and global environment stayed unchanged.

The preceding default-platform attempt failed during source environment setup
after 55.35 seconds, before image capture, agent execution or verification.
It remains a separate unscored infrastructure attempt. Its saved exception
type was `RuntimeError`; message and traceback fields were suppressed. No raw
logs, exception dumps, prompts, responses or generated patch content were
opened to diagnose it.

## Source-bound architecture evidence

The pinned source Dockerfile uses a public AMD64/ARM64 base, downloads the
`deno-x86_64-unknown-linux-gnu.zip` release, and then executes Deno during the
build. The host daemon reports `aarch64`. The task's declared prebuilt image
publishes only AMD64. Public manifest reads succeeded for both references.
The source recipe therefore has a concrete x86-only executable requirement.

The successful intervention supports source architecture as the explanation
for the default failure. The original inner error and failed command were
not retained, so this record does not assert their exact contents. No source
Dockerfile reference names any removed local benchmark image. The controlled
replay completed its source build and grading without registry or credential
changes, and produced no package-network failure classification.

The task Dockerfile and task ID are guarded exactly. The other released task
recipes retain their existing platform selection; no general architecture
guessing or global x86 default is introduced.

## Bound inputs

- Caller: `a153adf243f542724ffdb21078ab6b016b5dc0e5`.
- MaxxBench: `1a0df5e9c5d4bccbae131b01830d478088fea947`, 89 released tasks.
- Upstream stock handoff SHA-256:
  `0a9ac622f135fbfa3953e9edc46031ce0e731b55cf498614b718a8d2ec887bb1`.
- Cliffy source Dockerfile SHA-256:
  `2575713dcc93a27bc1b3b052aa02129f13d2dba753f7ca7fe97a796397d9bef1`.
- Cliffy task metadata SHA-256:
  `c21651d7a9fd6b3dbe5562e804153c51775cf1d4ed112f775229d11560b94ba4`.
- Controlled observer SHA-256:
  `3b1e24db758d363c569c85ab6d6fae70271aff5a8fea2c741c2d291b1486a488`.
- Controlled runner SHA-256:
  `55339e4b24a0e0092074df3ee72686630c2d3aebe5833d878b1c63ca23d13937`.
- Controlled verifier helper SHA-256:
  `fa3d7472c339b9ba0e2affc43ccef4e58f084b9e595425ca7b8d00fbd60bc0b4`.
- Implementation source/verifier helper SHA-256:
  `51029198898a8085b151776724e5aacc6ec9fbe875b0fb3445f6f537e501b30f`.
- Resource helper SHA-256:
  `88f2b8b4b84c49bc7b48c5a06f3f3adf6352a95fe82bb9f75db7645445b0479e`.
- Locked runner dependency file SHA-256:
  `bb91ad35bb41daca4f409d839a1ee67c947d9b16d84be25a740998182c6fd54e`.
- Existing native worker SHA-256:
  `a589fb72b94ffd8381642ab898f348d53bc3d78d5e59c36ddf294e023ebb2d79`.
- Existing template-helper SHA-256:
  `1765a2c28309db99b8095da015035b89acc2cf4759752fba931079121421eb97`.

The oracle reused Python 3.12.13 and locked Pier 0.3.1. Native binaries were
hashed for provenance and were not executed. Root source/helper/task/native
hashes stayed unchanged throughout the controlled replay. No model ran.

## Replay and resulting implementation

The source build was admitted with 35,058,065,408 bytes free, no native model
worker and the original shared benchmark lock held. The Trial ran from
`12:43:39.448258` to `12:46:25.186521` UTC. Captured source image ID:
`sha256:b8222bfb2f95b9c15014b3694f9614b7086bfda215178ae5e24d5eabc4c55b30`.
Cleanup succeeded: one verified source image and 41 eligible new cache IDs
were removed, 61 protected volumes remained, and the lock was released. The
immediate cleanup free-space reading was 26,603,122,688 bytes; later disk
reclamation is measured separately rather than inferred from deleted IDs.

The implementation wraps source Trial creation for this one task, validates
the recipe hash and metadata ID, and copies the config before setting the
platform. Conflicting process/trial/task/persistent-env settings fail closed.
All other tasks pass through with the original config. The existing Anko
verifier handoff behavior is preserved.

Receipts bind the selected platform, observed source platform, recipe hash and
immutable image ID. The platform observation shares the original five-second
image capture budget. Missing or mismatched Cliffy platform evidence stops
the attempt before agent execution. Only fixed platform fields reach progress
or aggregate results; image config and labels stay in memory.

Validation passed 129 model-free runner, resource, verifier, bridge and source
policy tests, including actual Trial-instance inheritance, unselected-task
capture/progress, exact recipe guards, conflict handling, config immutability
and the shared inspection deadline. Ruff, repository policy and diff checks
passed. No native build or model inference was needed for the implementation.

The prior 15 accepted source-bound oracle passes remain historical controls.
A fresh strict 89-task oracle cohort is required after the new runner/helper
freeze. Earlier receipts are not edited or combined across that source change.
Independent force-built image IDs do not establish byte equivalence between
rebuilds. This record qualifies environment handling and carries no model score.
