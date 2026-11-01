# Window presentation and agent text streaming follow-up

Date: 2026-10-31. Implementation commit:
`69c9cffc491e1292d4118b9ee6b4ae20232a6bcb`.

## Findings and changes

The app's reopen delegate previously opened Playground when models existed and
returned true, allowing AppKit's default reopening behavior. Reopen now shows
the main menu when needed and returns false. Windows disable restoration and
snapshot restoration; explicit presentation selects the requested window before
activation. The original external event behind the reported spontaneous windows
was not captured, so this establishes removal of the known implicit paths.

The main menu reuses Playground's GPU memory indicator, including its existing
thresholds, unavailable state and worker-only allocation semantics.

`Desktop.api_complete` previously disabled native deltas for every request with
tools or tool history, withholding even plain text until generation completed.
Tool-enabled text now streams before the first potential tool marker. Marker
prefixes, separators and arguments remain private until terminal validation.
Executable calls are still emitted exactly once only after strict parsing;
malformed calls fail after any provisional text without emitting tool deltas.
The parser/filter bounds all tool-enabled output to 512 KiB and retains bounded,
linear incremental work. Buffered and streamed content agree, including the
trimmed separator before calls. Cancellation still drains the native stream.

## Verification

| Check | Result |
| --- | --- |
| Locked offline converter suite | 5,177 passed, 88 skipped in 173.41 seconds |
| Focused tool/API/backend suite | 206 passed |
| Native connection, isolated credential, installer, controller, builder and window runners | 151 tests passed; broker and installer runners include their 17/25 scenarios |
| Independent synthetic chunk-equivalence exercise | 600 cases passed |
| Swift typecheck, repository policy, Ruff, formatting, whitespace | Passed |

The new native window harness uses actual delegate/view code, hidden windows,
isolated preferences and credential storage that fails if accessed. It exercises
reopen handling, non-restorable retained windows and 20 background state/GPU
updates. Its status item is absent; installed UI evidence is separate below.
The HTTP regression blocks the worker after its first text delta and requires
the real HTTP client to receive that text before allowing generation to finish.
Additional tests cover every marker split, whitespace/Unicode equivalence,
bounded output, malformed/truncated/unknown calls, disconnects and clean retries.
A distinct review found no blocking source findings before installation/push.
The full converter suite and all 151 app checks were repeated from a clean
detached snapshot of the implementation commit, excluding concurrent drafts.

Rust source is unchanged by this patch. The app reuses the previous Rust 1.98.1
locked native builds; both workers' pre-signing bytes match the 0.11.13 build.
Concurrent timing-display/Rust edits appeared later in the shared checkout and
were preserved. The built package's 98 relevant committed source-file hashes
were checked against the implementation commit with zero mismatches.

## Bounded latency measurement

The synthetic driver used one loaded `mlx-community/Qwen3-4B-4bit` worker,
MLX affine 4-bit/group-64, revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`. It requested 256 output tokens with
greedy sampling and no tool execution. A large synthetic tool description
provided an agent-sized input. The buffered control disables deltas before
native generation, reproducing the prior delivery behavior. The four large
input runs use buffered/streamed/streamed/buffered order and produced identical
final output and usage. No rendered request, generated output, token vector or
credential was saved; synthetic request definitions remain in the ignored driver
source.

| Input | Delivery | Input/output tokens | First visible text | Completion |
| --- | --- | --- | ---: | ---: |
| Short chat | Streamed | 31 / 256 | 0.013 s | 1.477 s |
| Short tools | Streamed | 162 / 256 | 0.039 s | 1.522 s |
| Agent-sized, first control | Buffered | 15,359 / 256 | 8.008 s | 8.008 s |
| Agent-sized, first stream | Streamed | 15,359 / 256 | 5.606 s | 8.960 s |
| Agent-sized, second stream | Streamed | 15,359 / 256 | 5.746 s | 9.180 s |
| Agent-sized, second control | Buffered | 15,359 / 256 | 9.060 s | 9.060 s |

The streamed large-input requests measured native first-token processing at
5.582/5.713 seconds and subsequent decoding at 3.340/3.420 seconds. Text delivery
now starts after input processing instead of waiting through decoding as well.
This does not make large agent requests equivalent to short Playground inputs.

This is a backend delivery/stage check, not an exclusive-host throughput
benchmark or a new OpenWork end-to-end latency measurement. Model-free tests
were running on the host, and the synthetic description is not OpenWork's
actual request. The prior real-client validation recorded about 19,000 input
tokens per turn; client instructions, tools and history contribute real prefill
work. No trimming, prompt cache, client authentication change or model task
competence claim is introduced here. See the preceding
[managed connection run](2026-10-31-openwork-managed-connection.md) for the
official OpenWork 0.18.63 / OpenCode 2.0.26 execution evidence.

## Installed package and UI

The existing builder fetched the locked connector revision
`1ca4c4586de8052c5a3bcb95455ab67e0b1fd95c` and verified its six file hashes,
without a development override. The existing signing workflow signed 37 runtime
Mach-O images and the canonical `com.starkengine.menubar` app with Developer ID
team `S5BT2PRN5B`, hardened runtime and secure timestamps. No validation profile,
helper or instrumentation was bundled. This local package is not notarized.

Installed to `/Applications/Stark.app`, version **0.11.14**. Strict deep
signature verification and installed host hash verification passed.

| Delivered executable | SHA-256 |
| --- | --- |
| Swift host | `3ff648d6f063a4d9c48aa3100cde45db76b2c680019da6edd9292b31198df88a` |
| Generation worker | `9da7b12555316ec983e57ad61ce35db2a9649853f2b2c407c7c243b2f8972639` |
| Scoring worker | `d4dd1c14f9b253fbca3fce7a7bfe5adbeec3490a7bd8f46b927da37a9715a68d` |

Native UI checks observed explicit app reopen showing the main menu without
Playground or Connection Details. Their named controls still open both windows.
After loading Qwen3-4B, the main menu and Playground showed GPU memory 1%, with
2.11 GB worker allocation. Both windows were closed and remained closed during
subsequent background updates. The model remains loaded for the user; the main
menu was dismissed. Connection Details reported API ready on saved port 8000
and existing OpenWork approval. The key was never shown or copied. This is an
ordinary canonical-identity app update, without a byte-for-byte credential
comparison or a notarized-upgrade claim.
