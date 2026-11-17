# Lily startup mode and memory admission

Registered and executed on 2026-11-17. Owner: Codex.

## Attempts and decisions

| Attempt | Scope | Result | Disposition |
| --- | --- | --- | --- |
| `20261117-LILYSTART-01` | Installed 0.22.1 mode admission | Saved Decision mode rejected the installed vision-language Lily artifact before native loading | Reproduced; fix default model starts to select Chat, with native Decider selecting Decision |
| `20261117-LILYSTART-02` | Installed Python memory preflight | Full 88.73 GB estimate rejected an approximately 85 GB available-memory observation | False rejection for the tested startup; diagnostic-only bypass reached native readiness with images enabled |
| `20261117-LILYSTART-03` | Exact-artifact headroom policy and fresh image checks | Candidate reached native readiness and passed fresh automatic image execution checks | Adopted on branch for the exact pinned artifact; installed-app verification recorded below |

These attempts change host admission and error handling. They introduce no
native kernel, model state, precision, MTP or prefix-cache changes. Prior model
qualification records remain bound to their original sources and artifacts.

## Artifact and environment

| Field | Identity |
| --- | --- |
| Checkpoint | `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` |
| Upstream revision | `0f94842e3682b49acace4daeb4cf2e65a6f94f83` |
| Retained image profile | `qwen38-lily-retained-bf16-vision-v1`, version 1, schema 6 |
| Retained artifact digest | `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4` |
| Payload | 55 files, 106,905,155,192 bytes; existing Lily mixed quantization and retained image/MTP weights |
| Generator SHA-256 | `9a3a4fac1f5c15124b913667260df7c33f9b35508ca988a65b6386e08c3a4707` |
| Installed input bundle | 0.22.1; bundle source `18addd2d10ac39b931ef1c472afb1a84d21a1bba` |
| Input app manifest SHA-256 | `8ca38dcb22fb04da6911091c2dfde9d105f1b299714f5a68065427d7a19c7497` |
| Input runtime manifest SHA-256 | `f1c8a18f97a35c5995976c83ba83a853557e8129f06f9f3ffbbd819cc5c9b7f7` |
| Candidate source base | `fb7b6771576a89d11f5409f0f23745064f5cc3b2` plus this branch's host changes |
| Host | Apple silicon; 137,438,953,472 bytes physical RAM; macOS 26.7.2 (`25G332`) |
| Python/dependencies | Python 3.13.13; locked converter dependencies; lock SHA-256 `df7711726b1ed7e235ded21b9e6e1cbe130924be4d2c580d5d6bd9005248f8d5` |

Native executables are reused from the signed installed bundle. No native
rebuild or new numerical equivalence claim is part of this change.

## Policy and bounded live controls

Default starts choose Chat for non-Decider models, including saved Decision
preferences and transitions from Decider. Explicit Decision selection remains
available for loaded text scoring models. Vision-language artifacts disable
Decision. A stopped model resumes its previous compatible mode after selecting
the actual restart target. Expected startup failures retain their content-free
recovery guidance and close only the failing worker.

The full Lily planning estimate remains 88,728,893,312 bytes. The exact pinned
artifact uses 80,138,958,720 bytes of observed available RAM for selected weights,
states and workspace, and retains the full estimate as an installed-RAM floor.
This separates the 8 GiB planning allowance from the available-memory cutoff;
it does not reserve 8 GiB of free driver/host memory. Identity drift and generic
artifacts retain their original observed-memory requirement. Native allocation
limits, full payload admission, and automatic image execution checks remain.

Attempt 02 bypassed only the Python preflight in one disposable diagnostic
process. Native guards stayed active. Readiness reported `text-generation-v1`
with images enabled; no generation output was recorded. That process closed.

Attempt 03 used the candidate preflight without a bypass and an empty temporary
image-check receipt directory, forcing fresh automatic checks. Available RAM
before launch was 91,805,941,760 bytes. Native readiness passed, images were
enabled, and `image_check_cached` was false. Sampled worker RSS peaked at
72,802,877,440 bytes; reported GPU telemetry peaked at 75,871,682,560 bytes.
RSS sampling ran every 100 ms after worker construction and is not a complete
allocation high-water mark. These quantities are not interchangeable or added.
The worker closed and the temporary receipt directory was removed.

The fresh diagnostic used `image_checks.py` SHA-256
`06ec9a876f2ae48f067733b28a7b7a27ddd4040510ff105df18d02b7feba314c`
and `hybrid_image_memory.py` SHA-256
`2291dbe4d9e45542ae03c130ad1e7dafe8ab8c9964093a08d457ba623f62d4a6`.
Later edits format the same arithmetic, clarify the non-reservation wording,
and distinguish installed-RAM failures. Final bundle validation is separate.

## Validation and limits

- Repository policy, Ruff lint/format and diff checks passed.
- Native controller harness: 167 passed, including default starts, retry,
  explicit text Decision selection, Decider images and stale-selection resume.
  An existing expected family list omitted Decider; the baseline failure was
  reproduced before updating that exact sorted expectation.
- Native window checks: 14 passed.
- Desktop lifecycle: 372 passed, one existing skip.
- Vision installation, family checks and native Decider backend: 67 passed.
- Memory and image-check controls: 128 passed, including an 85 GB available /
  128 GiB physical observation, both cutoff boundaries, malformed capacity and
  identity drift. Tests require the existing native template helper on PATH;
  an initial run without it failed setup and was rerun correctly.

No throughput, task quality, numerical parity, larger-image peak memory or
long-context performance was measured. Existing native precision/cache/MTP
qualification is unchanged. Success is limited to this artifact and the bounded
startup/automatic-check workloads above; it does not guarantee allocation under
arbitrary competing memory use.

## Final signed installation

Installed Developer ID signed development build `20261117152352983526` into
`~/Applications/Stark.app`. Its source manifest records clean implementation
commit `6e15e34de89316ac13f24a18ecf6a6c6154efc5a`. The installed host files match
the implementation source, and deep/strict code-signature verification passed.
Packaging re-signs the reused native executables; the final generator SHA-256 is
`b120c1ab4a3a3678859e6262cd63894133c691e40b22e039fc1665021779fdd8`.
The installed app manifest SHA-256 is
`e10c8d4e560ce36ba33de0b7cb7487be7ed61f627b52f2b3c19d94ea71791f83`.

The final packaged Python/controller and re-signed generator passed another
isolated full Desktop startup with an empty temporary receipt directory:
Chat readiness, images enabled, fresh checks (`image_check_cached=false`) and
Ready status. That worker and its temporary directory were closed afterward.

Installed GUI control then loaded Decider to Ready with Decision selected,
selected Lily through Manage Models, and observed automatic Chat selection
followed by Ready with image attachment enabled. Lily remained running in Chat.
No user conversation or generated content was used or recorded.
