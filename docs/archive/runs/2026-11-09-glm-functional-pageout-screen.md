# 20261109-GLMVIS-10: functional-only GLM VM screen

State: separately versioned functional-only supervisor with source/CPU and
real-child cleanup controls. Independent parent review approved one conditional
short run after the queued Qwen and Gemma slots; no new model run has started.
This record alone does not release the resource slot. Predecessor:
[GLMVIS-09](2026-11-09-glm-bounded-image-execution.md).

## Problem and source evidence

Both strict GLMVIS-09 attempts stopped on growth of the host-global Pageouts
counter before their first text completion. Neither completed an image or
qualified correctness, cache behavior, MTP or performance. Both failed receipts
retain their original disposition. Available memory stayed above 91 GB,
compressor growth and swap stayed zero; those observations do not establish
the cause or show that the counter growth was harmless.

Apple XNU commit `f6217f891ac0bb64f3d375211650a4c1ff8ca1ea`
(`xnu-12377.1.9`) supplies the following primary evidence:

- [`upl_commit_range`](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_upl.c)
  increments `vm_statistics_pageouts` when cleaned, non-hibernation pages carry
  `UPL_PAGEOUT`.
- The [`vnode_pageout` path](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/bsd/vm/vnode_pager.c)
  handles real-file pageouts and msync dirty-page runs through `VNOP_PAGEOUT`.
- [`vm_compressor_backing_store.c`](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/vm/vm_compressor_backing_store.c)
  separately increments `vm_statistics_swapouts` for compressor-segment pages
  written successfully to swap.
- [`host.c`](https://github.com/apple-oss-distributions/xnu/blob/f6217f891ac0bb64f3d375211650a4c1ff8ca1ea/osfmk/kern/host.c)
  defines and exports the Pageouts and Swapouts counters separately.

This establishes that Pageouts is not an exclusive compressor-swap indicator.
It does not attribute the earlier global increments to any process or file.
The public revision is from the same XNU release family; it is not the host's
exact `xnu-12377.161.15.701.48~2` build. Source references use symbols and pinned
file hashes, avoiding mismatched rendered/raw line anchors. The private source
evidence receipt is
`275cf7cfa4c9d2c69ce94509320fc71c187ae4bfc213b8294d68866647b3a468`.

A separate owned 16 MiB CPU file experiment covered idle, create/fsync,
buffered read, streaming hash, read-only mapping and dirty mapping/flush.
Every phase had zero Pageouts, Swapouts and swap-used growth. The owned fixture
was removed. This small experiment did not reproduce the GLM interruption or
establish its cause. Receipt:
`52bd7eb0783ff0c0ab08820065c72e94c6a351fb507c8d631a45343b9cd5d82d`.

## Proposed functional-only policy

Keep the strict version-09 supervisor unchanged for the timing gate. A new
private version-10 supervisor removes global Pageouts growth from fatal
functional controls and records it as a diagnostic. Every functional-10 run
is ineligible for performance claims; any Pageouts growth also fails the
separately reported zero-growth timing screen. This does not change production
request admission or qualify either earlier failed attempt.

All other limits remain fixed:

| Screen | Limit |
| --- | ---: |
| Entry available bytes | At least 64 GiB |
| Runtime available bytes | At least 32 GiB |
| Worker RSS / physical footprint | At most 32 GiB each |
| Host wired growth | At most 32 GiB |
| Host compressor growth | Less than 1 GiB |
| Swap used / Swapouts growth | Zero |

Counter discontinuity, unavailable telemetry and a foreign inference worker
or installed Stark process remain fatal. Cleanup owns one Popen handle, closes
its cancellation pipe, terminates only that child when needed and reaps it.
Success requires positive process samples, no fatal violation, a drained child
and joined supervisor. Every available host sample carries wall and monotonic
timestamps and all counter values. The report retains the exact first Pageouts
growth and fatal breach samples, including a final post-drain host sample.

The new supervisor SHA256 is
`9860befa761a1b7fabc563ddb873e6854fcbeb072e52b7fe6755bb83bb096b6e`;
its immutable diff against strict version 09 is
`a498b1d4d3320d0fd77cdfba279581d256cc3830bb7cdcfeb55296e03c3728f8`.
Fourteen injected source controls pass, covering retained hard limits,
diagnostic Pageouts, counter discontinuity, exact samples and performance
ineligibility. Receipt:
`0ec47c3548a0bc3ddfa82a3b29b0892080300be94b4988fd842fd1007e6ea3f1`.
The exact-source real tiny-child smoke also passes five controls: natural
exit, telemetry-failure cancellation, ignored cancellation followed by owned
termination, constructor/base-close failure and early exit. All children were
reaped and supervisors joined. Receipt:
`a40ec46f2ec143b92e55e6ffd6e7fb2c868c8263a0c30c0ce02cfbc7a9782983`.
It ran in a separately granted quiet gap; the gap was released immediately.

## Fixed next recipe and remaining gates

The registered recipe receipt is
`fd383670405e669c8a9637796f084957474404e705418ab7724bbca66789c771`.
It retains the compiled GLMVIS-09 source graph
`a2c9fad27a697206d70f0661b2116675d14f4fec4d1648d78729360b819d81f6`,
worker `c8e528dc017809edb9c717b55c6c2dce6533e34259a4175988e9c8d4aa506f3d`
and helper `7d8f92e9552a9fd4e898113d674dd3ad235f423f7ce8a03d1ed03a2f6380164d`.
The new driver SHA256 is
`e6b1ef48893df7613bdf54ceb90f8bec3086f5bc53efae3d9cea6f456da99fd4`.

The parent independently reviewed the immutable supervisor diff and source,
then accepted the exact-source five-child smoke. The private pre-execution
checkpoint is
`8ee39e269a5ce3bb09f4411f5319c146115ffbff46772b8b635768b7a1a867af`.
It verifies all 1,262 build-source files and binds the exact next command,
driver, supervisor, worker/helper, artifact manifest, fixed environment, prior
failed receipts and remaining gate. The gate requires the queued slot release
and a fresh available-memory/swap/foreign-model/Metal/owned-compiler screen.
One attempt is allowed; a hard violation ends the grant without a retry.

After that resource gate, the only authorized model sequence remains
text-only eight tokens, image-loaded text eight tokens with exact output-hash
agreement, an actual PNG request with one output token and zero MTP, then
eight-token text recovery with the original output hash. Use
`STARK_GLM53_BOUNDED_GRAPH=1` and `STARK_GLM53_BANK_GROUP_LAYERS=1`; all other
GLM selectors remain unset. The generic divisor-four request guard is intact.

There is no new model result. The full resident recipe remains held by its
complete memory calculation. Image/cache/MTP qualification and matched quiet
performance controls remain open. Bounded functional parity would not qualify
normal-text speed against the earlier fast resident recipe.
