# Flash Next kernel and prefix follow-up

**November 17, 2026 · Lily Q4-XL on an M5 Max with 128 GiB**

| Attempt | Change | Evidence and decision criteria |
| --- | --- | --- |
| 20261117-FLASHSPEED-08 | Keep the 4,096 physical grid and place recurrent snapshots at 2,048-token boundaries. | Assess the opportunities in the existing Pi trace and size the required state. An implementation needs exact capture neutrality and a fresh-versus-resumed comparison with matched MTP trajectories, context, sampling, and admitted cache scope. |
| 20261117-FLASHSPEED-09 | Fuse grouped Q4 expert gate/up and SwiGLU on the existing 4,096 physical grid. | For further qualification, require identical output and MTP work on every pair, median native first-token improvement of at least 5% in both launch orders, and no length with a median first-token regression over 5%. |
| 20261117-FLASHSPEED-10 | Share packed Q8 HC down weights across the verifier rows, as implemented by `STARK_LILY_VERIFY_HC_DOWN_REUSE`. | For further qualification, require identical output and MTP work on every pair and at least 3% median decode improvement in both launch orders. First-token time is a host-variation control because the selector only affects verification. |

## Controls

Screen each kernel separately with one model resident and the Sealed Lily Q4-XL worker from [attempts 02–05](2026-11-17-flashnext-speed-experiments.md). Model `fabiogreter/Qwen3.8-Flash-Next-lily-q4-xl` is pinned at `0f94842e3682b49acace4daeb4cf2e65a6f94f83`, with retained vision and MTP loaded and artifact digest `18c7bcc1e97d1fefdd361472c2daa59d81d89fcd6f3a61983d995a032b44bfb4`. Worker SHA-256 is `bdf851bedf895ab09859cbdc40cab583a9976ddaf06f01d98ad3872dcd8f7dbe`, built from `3e956d960c631beacf573053a6a2fa0ba0c99b3e`. The only runtime difference is the relevant kernel selector.

Use one previously pinned repository-source coding fixture at each of 8,192, 12,288 and 32,768 tokens. Compare launches in off/on/on/off order, with the same fixture order inside each pair and the reverse fixture order in the second pair. Exclude an 8,192-token warmup on each launch. Use no cache scope, 40,960 context, seed 42, temperature 0.6, top-k 20, top-p 0.95 and two exact-rejection MTP drafts. The MoE cap is 96 samples and the HC cap is 192. EOS is valid if a pair matches. Record matched native first-token, decode, and verifier duration, complete wall time and system paging. Treat altered trajectories as correctness evidence; exclude those pairs from matched-work timing. Stop an arm on native failure, worker timeout, artifact change, or geometry or MTP admission mismatch. Restore the installed service after the run.

This is a directional screen with six matched observations per selector and three distinct fixtures. A passing screen earns follow-up qualification. Default promotion would also require broader numerical and lifecycle controls with the exact checkpoint, including cancellation and the image transition.
