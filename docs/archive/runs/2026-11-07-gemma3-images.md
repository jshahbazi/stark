# Gemma 3 image validation, 2026-11-07

Native numerical replay and the clean worker/API lifecycle passed for the
original Gemma 3 4B BF16 image artifact with the exact builds recorded below.
Gemma 3 12B and 27B have structural admission only; execution was not tested.

## Bound inputs and builds

| Input | Identity |
| --- | --- |
| Checkpoint | `google/gemma-3-4b-it` |
| Checkpoint revision | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Converted weights SHA-256 | `50b031efe933a6023b4188a718b1f4e2d654d6eac5b5c9336205de59a3426805` |
| Converted config SHA-256 | `69bd51786943ca8320926a2359947e618fc11a8f8cf09f1640329f48b75bcc98` |
| Source reference manifest SHA-256 | `d4ab18823153f25fdfbc1cd8d8f90a95262ccfe482a0b996b765ae7f6067b726` |
| Reference framework | Torch 2.14.1, Transformers 5.15.1, Pillow 13.0.0; CPU eager BF16 |
| Native replay source | `af3569191845c566d9e52deb076063348fc9301a` |
| Native replay source tree | `955379e39b9315e710532d425036e59414ebcac1` |
| Native replay executable SHA-256 | `1df910fb8508e7df8933bc2f63ff4f64837382c22874e121edf7e0dc23bcce03` |
| Passing native metrics SHA-256 | `ec8e6def407da88ef3af0b14bfc2d2f0dab6497fc2fdb47d084b1af5a0626485` |
| Release worker source | `9ce9cf93d92bb6f4d82e0770d0889e5dae9de0df` |
| Release worker SHA-256 | `2977f20ea75742215ba23b092a6011c17f75a90ec594f26eb87c15ccdb3c806c` |
| Template helper SHA-256 | `913ec72e2e94ae98cc75d098315f2d42abe02af068718c384662ab75cd236f6f` |
| Worker/API receipt SHA-256 | `5bd96c632f61bcb33d1b4a5d508b8b1cabb376b81034142fafef58aecc60925f` |
| Lifecycle qualifier SHA-256 | `b95775665868391012079cc91ed91194a719a9f6dddb78a28504fc2ae4a360fe` |
| Completed-block cancellation receipt SHA-256 | `94427c47f5149f0fab7ad85c46b878431e9f96b462715d0f1c6ae1beabd642a0` |

The local runtime was Apple M5 Max, 40 GPU cores, Metal, macOS 26.7.2 and
Python 3.13.13. The worker receipt additionally binds the frozen host modules,
dependency versions, image policy, device identity and five artifact hashes.

The preserved canonical reference contains three synthetic PNG controls:
64×48, 48×64 and a nonuniform 111×79 pattern. Each is processed into 896×896
pixels, 4096 patches and 256 image embeddings. The original source processor
expands the fixed image-first prompt from 15 to 273 tokens. The native test
checks the exact expanded sequence independently against that source output.

Completed red/blue trajectories and the pattern feature file survived the
pause unchanged. Resume verified every selected original source payload,
hardlinked completed evidence into a new directory, and computed only the
missing pattern trajectory. A read-only hook in its unchanged full forward
verified that the pattern projector output reproduced the preserved feature
bytes exactly. All 13 original reference-file hashes remained unchanged.

Source implementation hashes are:

- `modeling_gemma3.py`: `0d8d5dff8e72b1207e54edce0bd2b22a34b3194edfa114996b04cef1acf4ef6a`
- `modeling_siglip.py`: `0a0d2b55f4fcb977b74549540211e42a3c3f41d0d446dc474dcc244a925791aa`

## Numerical replay

The native test uses its own image preprocessing, complete SigLIP tower,
pooling, projector and decoder. It compares the full prefill vocabulary
against the original model and runs an independent eight-step greedy
trajectory for each image. Later source logit rows remain private reference
evidence; the logit metrics below measure the prefill row.

Frozen limits were exact BF16 patches; feature relative RMS ≤0.02 and cosine
≥0.999; centered logit relative RMS ≤0.1 and source-to-native KL ≤0.03;
identical greedy trajectories. These limits were unchanged after failures.

| Control | Patches | Feature relative RMS | Feature cosine | Centered logit relative RMS | Logit KL | Greedy |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 64×48 | Exact | 0.0000277791 | 0.999999999614 | 0.0399148 | 0.000869342 | 8/8 exact |
| 48×64 | Exact | 0.0000241230 | 0.999999999709 | 0.0295242 | 0.000709869 | 8/8 exact |
| 111×79 pattern | Exact | 0.0000713012 | 0.999999997458 | 0.0351086 | 0.000000824432 | 8/8 exact |

The nonuniform control exposed two arithmetic defects that solid colors
missed. The pinned CPU processor uses adaptive signed-int16 resize
coefficients; fixed Pillow 22-bit coefficients differed in 799 patch
elements. Once those patches matched, the patch Conv2D still used the dense
32-stream contraction. The original CPU convolution uses four streams, with
bias added before the BF16 cast. Correcting that dispatch reduced the pattern
feature relative RMS from 0.0338022 to 0.0000713012.

The original failed replay, resize-only failed replay and passing replay
remain in distinct ignored directories. A separate original-source first-block
diagnostic captured 17 bounded stage prefixes; the reviewed producer
reproduced all 17 stage hashes. This diagnostic identified the convolution
reduction difference and does not replace the full-model gates above.

## Lifecycle and source checks

The final release worker passed all 14 recorded lifecycle gates. It ran
cold local checks, all three canonical source greedy comparisons,
prepared-image single use and discard, fresh replay, malformed-image-to-text
recovery, and image-request cancellation with same-instance text recovery.
An isolated Desktop instance served the real loopback HTTP API: its image
request returned 200 and usage 273+8=281, and a streaming client disconnect
cancelled the native request, drained the transaction and preserved the same
instance for text. A restarted worker reused the source-bound local-check
receipt and successfully processed another fresh image. Every staged-image
directory was empty after request cleanup; app events contained no prompt,
data URL or generated response. All task-owned workers exited afterward.

The native component control separately cancelled after one known completed
tower block, verified that immutable model weights remained unfenced, and
generated text on the same instance. This provides the completed-block proof;
the timed worker and HTTP checks establish request cancellation and recovery.

The lifecycle receipt binds the qualifier from commit `722569d6`. A later
failure-output guard has a model-free regression verifying that refusing an
existing replay directory preserves its completed and failure receipts.

Converter, Gemma text/MLX and security-boundary checks passed 276 tests after
the mask-policy update. The preserved-reference and output-preservation
helpers passed four tests.
The native nonuniform resize regression passed four independent source patch
hashes covering upsampling, downsampling and either unchanged-size axis.
Repository policy and relevant Rust formatting checks passed.

Global Gemma 3 decoder layers use causal-or-same-image visibility. Local
layers intersect that visibility with the strict sliding lower bound.
New conversion records these policies explicitly. An older admitted local
`same_image` policy remains equivalent because the fixed 256-token image
block is strictly shorter than the 1024-token window.

These controls qualify numerical implementation and request behavior for
this exact artifact and the bound builds. They establish no image-answer
quality or throughput result. Generation prefix reuse remains disabled for
this wrapper: its sliding KV rings cannot restore overwritten earlier rows.
Source images, prompts, tokens, logits, staged files and model weights remain
private and ignored.
