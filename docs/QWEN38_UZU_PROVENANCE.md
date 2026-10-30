# Qwen3.8 Uzu execution provenance

Stark's Qwen3.8 architecture blocks use the normal Uzu backend, buffers,
parameter loading, matrix dispatch, generated kernel interfaces, command
encoding, request state and language-model API.

The model equations and some shader arithmetic are adapted from
[lily-qwen3.8-flash-next revision e22d58fa7cc488493fb1b5d810a198bd29730aeb](https://github.com/fabiogreter/lily-qwen3.8-flash-next/tree/e22d58fa7cc488493fb1b5d810a198bd29730aeb).
That source is Apache-2.0, with Copyright 2026 Perplexity AI, Inc. and
Copyright 2026 Fabio Greter. MLX and MLX-LM mathematical/shader adaptations
also retain their Apple MIT notices. See [LICENSE](../LICENSE.qwen38-reference)
and [NOTICE](../NOTICE.qwen38-reference). Existing earlier reference-model
attributions and dated validation records remain intact.

## Adaptation boundary

- HC normalization, low-rank mixing/injection and final reduction are Qwen
  encodable blocks with standard Uzu matrix projections.
- GDN projections, convolution, decay/update and output normalization/gating
  use standard projection dispatch and backend kernels with request-owned
  F32 recurrent state.
- Routed and shared experts use standard full-bank `WeightMatrix` storage,
  gathered matrix dispatch and explicit routing/activation/combine kernels.
- QSA retains causal compressed-indexer selection, rotary positions, Q8
  attention caches and sparse/dense-prefix attention in standard backend
  kernels. Its projections use the common matrix dispatcher.
- PLE retains exact unsigned hashing, bounded paged n-gram reads, standard
  projections and causal convolution/history handling.

The adapted kernels have new Uzu DSL interfaces and matching CPU reference
implementations. No original Lily Tensor, QuantWeights, ComputePass,
MetalContext, model executor, allocator or command queue belongs to this
architecture implementation. Attribution to a mathematical source does not
identify the runtime that executes it.

## Validation scope

This document records provenance and implementation boundaries. It does not
establish numerical parity, complete integration or a performance gain.
[The execution plan](QWEN38_UZU_EXECUTION_PLAN.md) defines the required final
audit and runtime gates. New source-bound validation is recorded separately
from earlier Lily-derived execution results.
