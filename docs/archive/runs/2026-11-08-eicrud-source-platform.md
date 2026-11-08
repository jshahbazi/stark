# Eicrud source platform qualification — 2026-11-08

## Result and diagnosis

A controlled stock oracle for `eicrud-keyset-pagination-cursor` passed all
25 feature checks and 164 regression checks, with reward 1. Its source and
verifier images were both observed as `linux/amd64`. No model ran.

Three preceding default-platform attempts failed during source environment
setup and remain separate unscored infrastructure attempts. The original
cohort stopped after 21 passing controls. Two isolated diagnostic attempts
reproduced the setup failure. The second identified APT exit 100 and
`UnableToLocate` for the source-declared `mongodb-org` package in the Dockerfile
RUN instruction beginning at line 19. It found no dependency, signature,
HTTP 403/404 or network-failure classification. No source image was available
after these failures; their actual platform was not directly observed.

The recipe uses the pinned multi-platform Mars base below. Its OCI history
identifies Debian Bookworm. The MongoDB RUN instruction constructs a Debian
Bookworm MongoDB 7 repository with an `amd64,arm64` architecture filter, then
runs `apt-get update` before `apt-get install -y mongodb-org` at line 27.
The [official MongoDB 7 Debian instructions](https://www.mongodb.com/docs/v7.0/tutorial/install-mongodb-on-debian/)
support Debian 11/12 on x86-64. The source filter alone does not establish
availability of a Debian ARM64 package. Aikido's events command returned
success with no output; no registry, credential, signature or TLS settings
were changed.

The successful single-platform intervention supports a source-platform
availability problem as the explanation for package lookup failure. It does
not establish the failed images' architecture. No task source, Dockerfile,
commands, solution, tests or verifier logic changed during the control.

## Controlled intervention and implementation

The observer deep-copied the per-Trial environment config and persistent-env
mapping, selecting `DOCKER_DEFAULT_PLATFORM=linux/amd64` only in that copy.
Original config and process environment remained unchanged. Conflicting
explicit platforms were rejected. The complete stock oracle then passed;
this diagnostic is retained separately and is not imported into a cohort.

The public helper extends the existing Cliffy allowlist with Eicrud's exact
task ID and source Dockerfile SHA-256. Both task metadata and recipe bytes
must match. It reuses the existing copied-config/conflict checks and actual
source-platform capture before agent execution. Aggregate `source_platform`
fields retain the existing `selected`, `observed`, `dockerfile_sha256` schema.
Cliffy's policy is preserved; the other 87 tasks retain their original config
and null source-policy fields. The Anko verifier compatibility is unchanged.

The control ran from 23:38:43 to 23:44:15 UTC under the shared benchmark lock,
with zero model loads. Minimum observed free space was 38,666,010,624 bytes.
Cleanup succeeded, removing one verified benchmark image and 55 eligible new
cache records while preserving 61 protected volumes. The owned sleep
assertion was reaped and the lock released. Deleted logical bytes are not
treated as an exact physical-space recovery measurement.

## Source and evidence identity

All attempts used caller `a762aca740263a75c639e78845f2bbe35f662b27` and
MaxxBench `1a0df5e9c5d4bccbae131b01830d478088fea947`, with Python 3.12.13 and
locked Pier 0.3.1. Native binaries were hashed but not executed. Rust build
source remains `aae3d17761552e360d8f1497ab853fc45047ba18`; no rebuild was needed.

| Input | SHA-256 |
| --- | --- |
| Source Dockerfile | `6bf8747fe0b1345d38333906a550ffe90e9b8d854e70d06dd1b503bdf455f68b` |
| Task metadata | `a7f4e26c0e61fedb75f15baf6f8b0c41f04e06da525ea904e85f9253260c8dc8` |
| Verifier Dockerfile | `b0966003dfac67c42436887f0cc6dbba57b2c87b26671d7443bec496039f061f` |
| Base image manifest | `91db850db926024eed328c4bf519d54986bc10aad75302cbb074f8e9d79b4c46` |
| Upstream stock handoff | `0a9ac622f135fbfa3953e9edc46031ce0e731b55cf498614b718a8d2ec887bb1` |
| Runner | `afdb65f875ab493a4e2aa0295cb3b84b10a0b5839689a9364f7ccaec0c04444b` |
| Controlled source/verifier helper | `51029198898a8085b151776724e5aacc6ec9fbe875b0fb3445f6f537e501b30f` |
| Implemented source/verifier helper | `0cc798ce950d3dc180ee209ff79a9d45a2a9c9aab4a242623e217a0429e6aaee` |
| Resource helper | `88f2b8b4b84c49bc7b48c5a06f3f3adf6352a95fe82bb9f75db7645445b0479e` |
| Runner dependency lock | `bb91ad35bb41daca4f409d839a1ee67c947d9b16d84be25a740998182c6fd54e` |
| Full-context production matrix | `76fa913490615f0bb437f585aea653ecf35c9e94dcf3a21bc52f0caf806e0ca0` |
| Existing native worker | `c5514016280ed85c69dac972373a688db85d10ab5810d134a5b3fe2ed8449415` |
| Existing template helper | `fdbf8a879cd15f6bea2c84e03d6afb6f16e6b49c828fce8a656d3b403a8cc588` |

| Private attempt | Observer SHA-256 | Diagnostic receipt SHA-256 |
| --- | --- | --- |
| Default 001 | `0b5c051188444d19b4d088440d951d27204060dd1d753faf34711983780db56d` | `70c0180f9499071f32d2908eada5cf3aed6a9a60ba90cf54967bc5ee9daf5d46` |
| Default 002 | `ac44eeb202b35c8eb19e440e926739fd8701bf67c35c76855bfeb4bb43e8923b` | `c87cb1b60ada9123ff691467a5e8e78ead84d44d38834359edd301550b5f3491` |
| AMD64 003 | `53ed1187252e4c73d0a9f5472e4ea7c12d9474949a5d0f370277e9a7b7c81c2a` | `f2e333713c62512cedaaea74b449c85386ab85013031508fde1e5bdd8fc406ab` |

| Attempt | Oracle aggregate SHA-256 | Cleanup receipt SHA-256 |
| --- | --- | --- |
| Original cohort failure | `d1f1e78a44e95bd6a7ac8c3b9d72cab0aedf7cdf63941e0a392d6b9bf87ec2bf` | `54fddd292777d8422a08d18eb969c3db599ed7b2cacf755a82b64ba7a4aab3e5` |
| Default 001 | `f73d0886d9d18b5d44919fa519724f1f07a76d37f79f066e91a4c7d0be62c17e` | `bc44f48db2745d2b3c6159ac71643867492012c9e3d30cdd4473870b9f5692b0` |
| Default 002 | `0a1a6116eb9928e743f3f2b96aa963ab19a14cf0026eefef4adc1e260302973d` | `20cfa4bb13104cc758c9e310dfc6f09178089317a4378e2a00b0a48b113621cd` |
| AMD64 003 | `016e7b94e52a0b64a1a37166a6a9b35a42176a09bbe5aafc4b7138f5314685a1` | `2631ceb266a440738d5c68269e9fecebaa016a5bd216663141d1d25046b25cc1` |

Controlled source image: `sha256:579e0d30f152fc03b60e93f862e3f6a62637572f74676fbf724f91929fa01717`.
Controlled verifier image: `sha256:73a68381d0ee35d8c41f5df7e9164591946602e9cd3730ea572dc8db5d2f8a1d`.
Force-built images remain independent per attempt; their identities do not
assert byte equivalence across rebuilds. Evidence remains in ignored private
runtime receipts; only fixed classifications, counts and hashes are published.

## Validation and remaining work

The source-platform, runner and verifier-compatibility suites passed 126
model-free tests on the durable Python 3.12.13 runner. Tests cover both exact
recipes, wrong task/hash pairs, borrowed recipes, conflict rejection, copied
config/file immutability, actual platform validation and unselected tasks.
Pytest used only locked pure-Python devtool modules from the converter via an
ignored shim, with source paths first and plugin autoload disabled; no managed
environment was modified. Shim identity SHA-256:
`7403f176f35925715bfc029fb4dc75b1764cf170100a14e33f1c4a7ce3e591d9`.
Ruff, repository policy and diff checks passed. No native build or model ran.

The preceding 21 source-bound oracle passes and all failures remain immutable
historical controls. A fresh strict 89-task cohort is required on the new
helper before the 623 model/task attempts. The full native context matrix is
unchanged: 40,960 for Qwen3 0.6B/4B, 262,144 for the advanced Qwen/Flint
artifacts, and 1,048,576 for GLM. No context cap, history trim or model inference
policy changed. This record qualifies environment handling and carries no
model score.
