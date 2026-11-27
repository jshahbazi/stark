# HY4STQ-NATIVE-02 · STQ catalog installation and runtime defaults

This change promotes the exact `beamster/Hy4-Flash-Sushi-STQ-4bpw`
artifact at `61528ba635a623f127c5a7b1e1f4ac2c2b0d4885`. It follows
the native Metal and [16K and 27K OpenCode evidence](2026-11-27-hy4-stq-kv-width-dense-io.md)
on the 128 GiB M5 Max. It incorporates main's [generation-context follow-up](2026-11-26-context-sizing-follow-up.md).

The native catalog offers this single Hy4-flash source. Its install
allows exactly the 99 pinned files, including 89 Safetensors shards,
the original configuration, image processor, tokenizer, chat template
and upstream license. The downloader verifies configuration and
Safetensors metadata plus an aggregate file-content identity. An
invalid pack stays in its owned staging directory. The worker checks
the immutable format, indexed header inventory and file fingerprints
before mapping the original weights.

The standard product path loads without an STQ activation flag. It
uses F16 KV, 4,096-row prefill, prepared mapped residency, Engram page
caching, the measured packed and register prefill paths, and the tuned
greedy verifier. Text defaults to five DFly2 proposals. Users can
disable MTP. Stochastic and image requests use target-only generation.
The original prepared AngelSlim folders retain their local load route.

The fit reports loaded capacity separately from the checkpoint and
backend limits. Text requests can select smaller windows. Prefix
restore uses the requested context, scope and draft policy, with
4,096-position checkpoints under the default settings. Images
invalidate cached text and preserve the next text turn's draft policy.
The original image-loaded quality and performance measurements
remain in their own records.

## Local catalog validation

The pinned Hugging Face revision returned a public 99-file manifest
with 208,397,806,219 total bytes. The full payload already on the
test host entered the actual native installer through the Hub
fetch boundary using copy-on-write clones. The installer verified
the full contents and published the original format in a private
Stark library. Native load and the desktop's local image checks
passed before the catalog registered the entry. This run exercised
the pinned download selection and install boundary with an existing
payload. No second 208 GB transfer was needed.

## Uncapped OpenCode and lifecycle

A release Rust 1.98.1 Metal worker loaded the private catalog
install with no Hy4 environment overrides. It reported ready
in 113.80 s with 249,856 loaded positions and no user cap. The
checkpoint declares 1,048,576 positions. The live worker
advertised the measured tuning options, tool parsing, image
support and five default greedy drafts.

| OpenCode fixture | Reused positions | Prefill seconds | Useful decode tok/s |
| --- | ---: | ---: | ---: |
| Fresh 16K | 0 | 19.61 | 52.17 |
| Continued 27K | 12,288 | 17.66 | 52.20 |
| Fresh 27K, first | 0 | 38.35 | 45.73 |
| Fresh 27K, repeat | 0 | 35.49 | 51.25 |

The rendered input and output hashes, draft counts and prefix
reuse matched the earlier 49,152-position fixture. The two
uncapped 27K measurements had a 36.92 s prefill median and
48.49 tok/s decode median. One observation fell below 48 tok/s.
The earlier 31.38 s and 51.66 tok/s medians measured the tuned
implementation under a 49,152-position cap. The different
loads and their timing variation do not isolate the effect of
the larger capacity.

On the same uncapped worker, windows of 8,192, 8,192 and 12,288
reused 0, 4,096 and 0 positions with identical outputs. The 27K
continuation produced the same output when replayed from zero
reuse. An actual image turn disabled drafts and cleared text
reuse. The next text turn reproduced the original output.

The catalog publisher was also rerun on all 99 files after moving
the ownership marker into the staging metadata directory. This
keeps the original pack inventory loadable across an interruption
after publication.

| Verified artifact or aggregate receipt | SHA-256 |
| --- | --- |
| Release worker | `1f1b2c65bedca32b031bc4ef5f16ebeaa605b6c33fe8e13a88762093dd3b2b38` |
| Catalog install with local image checks | `d703477d412420b70b8991c0b2593c5ec58ae9302805ab88fb03540a32740d02` |
| Final native publisher check | `92a5b349b5506f730f7345cfb4ab3d0067d042598aecf3898cbad3e4bad5cd57` |
| OpenCode defaults and lifecycle | `1a536048d63b6f32b14b0292b95b7010f7dc9b63ac9f4c932dbc97eb81e34eb9` |

Release Clippy, the native inspection example and Rust formatting
passed. The generation suite passed 188 tests with six ignored.
The engine's Hy4 nightly suite passed six tests with two private-pack
tests ignored. The converter and desktop suite passed 879 with one
skip. The 52 focused tests also passed after the staging-marker fix.
Ruff and repository policy checks passed.
