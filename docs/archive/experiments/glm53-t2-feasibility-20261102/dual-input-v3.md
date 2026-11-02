# GLM-5.3 T2 Exact Dual-Input Candidate and ABBA Studies v3

Date: 2026-11-02. Status: **exact candidate proof passed; both ABBA performance attempts unqualified**. The dual-input selector is test-only and defaults off. These records measure a bounded two-token component; they do not establish catch-up, rollback, request-level gain, or production admission.

## Exact dual-input candidate

The guarded candidate passed the run's complete hidden-state and logit comparison gates. It compared 192 complete rows before timing and 192 compact rows after timing, with 36 measured serial pairs and 36 dual-input pairs over four rounds (one warmup). Eight ordinary serial alignment rows, four pooled pairs over four pool residues, and zero rollback calls were recorded. No full-logit readback rows were timed.

| Median component time | Two serial steps | Layer-major two-token graph |
| --- | ---: | ---: |
| Wall time (ms) | 68.735271 | 63.683625 |
| GPU time (ms) | 62.68943750183098 | 56.286624996573664 |

The medians and totals are a single guarded candidate run, separate from the two later ABBA attempts. Dual-input coverage recorded 100 pair graphs, 68 Q5 and 68 Q6 calls per graph, and 6,800 dispatches of each kind. The precision policy remained `existing_safe_f32_legacy_projections`. Timed work includes compact greedy and hidden readback; rollback and head catch-up were not run.

Graph scratch high-water was 80,740,352 bytes. The guarded required peak was 108,692,065,600 bytes under a 109,676,289,225-byte cap; sampled Metal peak was 105,772,580,864 bytes. These are probe observations, not production memory admission.

## ABBA attempt 1: unscreened

The unscreened A-B-B-A attempt completed all four runs and passed source, binary, model-artifact, and weight guards. It had no whole-run host screen and only two runs per mode. The serial baseline drifted substantially between the first and last A runs, and mean- and median-normalized results diverged. The analysis receipt marks performance unqualified.

| Run | Pair wall median (ms) | Serial wall median (ms) | Pair GPU median (ms) | Serial GPU median (ms) |
| --- | ---: | ---: | ---: | ---: |
| A1 legacy | 70.197229 | 73.73179200000001 | 62.04795834491961 | 66.47852083551697 |
| B1 dual | 74.4696255 | 79.9593955 | 66.87618751311675 | 73.45056251506321 |
| B2 dual | 101.0903745 | 114.4025005 | 92.93793748656753 | 107.31760419730563 |
| A2 legacy | 90.03468749999999 | 91.84425050000002 | 82.22758334886748 | 85.75372918858193 |

The normalized B-versus-A comparison favored B by 2.33% to 6.12% across the reported mean- and median-based wall and GPU ratios. Raw paired wall times moved in the other direction: B/A was +1.31% by run means and +1.70% by the mean of run medians. Raw paired GPU time was +0.88% by run means and +1.28% by the mean of run medians. The control drift and differing normalized/raw comparisons leave this attempt unqualified.

## ABBA attempt 2: whole-run screened

The second attempt screened the full run, including load and validation. All four A/B runs were host-ineligible. External CPU maxima ranged from 155.18% to 254.88%; pageouts ranged from 292 to 435 per run. Both B2 and A2 recorded a thermal maximum of 1. The serial-control max/min ratios were 1.116007 wall mean, 1.114599 wall median, 1.116675 GPU mean, and 1.108907 GPU median against a 1.05 limit. The summary marks performance controls unqualified.

| Run | Pair wall median (ms) | Serial wall median (ms) | Pair GPU median (ms) | Serial GPU median (ms) | External CPU max (%) | Pageouts | Host eligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 legacy | 65.5086665 | 67.51597950000001 | 58.31741665315349 | 61.58406249596737 | 155.18223818970836 | 435 | No |
| B1 dual | 69.16191649999999 | 75.25327100000001 | 61.468625004636124 | 68.29100000322796 | 176.1932515786721 | 307 | No |
| B2 dual | 64.58112450000002 | 69.2953955 | 57.36875000002328 | 63.107395835686475 | 254.88491491904017 | 292 | No |
| A2 legacy | 65.997063 | 68.3104795 | 59.017437510192394 | 62.2788124892395 | 160.2310967706264 | 349 | No |

The normalized B-versus-A ratios favored B by 4.41% to 5.62% across the four summary views. Raw paired wall times showed B/A of +1.31% by run means and +1.70% by the mean of run medians; raw paired GPU times were +0.88% and +1.28%, respectively. All four whole-run host screens were ineligible, and serial-control drift exceeded the 1.05 guard. Treat these values as unqualified observations, not a performance result.

## Source and receipt provenance

The 13-file source freeze is based on `b4f1836d97c9fcfb154d803ba93f96b8255d18e4`, with SHA-256 `3b882153ddd85fbfce058eb92f03551a39a4ce986ac253701c78c7f0e11367c8`. [The exact source patch](dual-input-v3.patch) has SHA-256 `37898062142fdf04f2cda150a3c96d7652fb048e9cc30cd4d7ec8724400e4449` (92,042 bytes); default full-context application from the pinned base reconstructed all 13 frozen file hashes, and reverse application restored the base. The private source archive hash is `717103238c6e7127f6eec824758d25a39c7926af65f149782d34456d7f3bb0ba`.

The candidate result and guard receipt hashes are `97feb3a9e875e2470d2343791d5ec2b8aa8db6a6139054bedfc29f81562ac5db` and `5a57252ca14c4ed19a57a0fd1ba7c2f73c793de741dc10e64d896a014a7842e2`. The release no-run build receipt is `a064258c471fd9fdf4b0d4b87a189a11c3c7f7d74c14b26a76e6298206c1942a`; the model-free parser receipt records 28 passing cases and has SHA-256 `d5b0190a328c662575105cc1462931d8fa0f84ae060ce31d5579e4fc2a9bbdb7`. No native tests were invoked by that build receipt.

The two ABBA summary hashes are `c2e38e036ba6dcd6dfc6f115ac215a5a685ea2dc6179377d8ca776f3692e4641` (unscreened) and `fbdfea3dcc366de22875b81f75ac75dd79159dbdfda9a0f1f97ba9c20ba76445` (whole-run screened). Per-run guard hashes and exact aggregates are in [dual-input-v3.json](dual-input-v3.json). The retained result and guard files are aggregate summaries. Raw prompt/response payloads, token IDs, vectors, logits, probabilities, raw host rows, process names/PIDs, and native process output were not persisted. The public source patch contains authored fixed test fixtures only.

V1 and v2 records remain separate and unchanged. The candidate and both ABBA studies are test-only evidence; none promotes the dual-input path or qualifies a performance gain.
