# GLM-5.3 MTP Head Profile v4

Date: 2026-11-02. Status: **test-only component profile**. This is a separate timing follow-up to the v3 feasibility result; it retains the same three-case, 24-comparison corpus and 23/24 agreement. It does not execute speculative decoding, T2 verification, or target rollback, and makes no runtime speedup claim. See [profile-v4.json](profile-v4.json) for exact aggregate phase totals, call counts, and receipt hashes.

## Headline timing

Across 24 warm decode head calls, the decoder's `decode_head_wall` averaged **18.569333 ms per call** and reported GPU time averaged **2.574752 ms per call**. The profile's separate `head_total_inclusive` timer averaged 18.568861 ms wall per call, a 0.000472 ms per-call boundary difference. The disjoint `mapped_ffn_compute_wait` phase averaged **15.665695 ms wall / 1.289024 ms reported GPU**. Its nested `ffn_submit_return_inclusive` span averaged **15.527127 ms wall / 1.289024 ms reported GPU**.

The nested span's wall-minus-reported-GPU gap is **14.238102 ms**, about **89.0%** of the inclusive head's wall-minus-reported-GPU gap. The measurements do not isolate submit-to-GPU-start or residency delay from wait wake latency. Wall-minus-GPU time cannot be assigned to a specific CPU or scheduling cause from this profile.

## Phase totals and scope

The table reports aggregate milliseconds over the 24 head calls, except validation observation, which ran 120 times. “Main” phases are disjoint wall spans. “Nested inclusive” spans are separately measured subspans and must not be added to the main phase totals. `head_total_inclusive` is the overall inclusive head boundary.

| Phase | Scope | Calls | Wall ms total | Reported GPU ms total |
| --- | --- | ---: | ---: | ---: |
| Prepare | Main | 24 | 4.562626 | 0 |
| Norm/MLA encode | Main | 24 | 2.306918 | 0 |
| Norm/MLA wait/readback | Main | 24 | 21.708626 | 13.920542 |
| Projection blocking | Main | 24 | 6.034585 | 1.021417 |
| CPU router | Main | 24 | 0.210750 | 0 |
| Selected mapping/admission | Main | 24 | 2.064710 | 0 |
| Mapped FFN compute/wait | Main | 24 | 375.976669 | 30.936583 |
| Final readout encode | Main | 24 | 1.722874 | 0 |
| Final readout wait/readback | Main | 24 | 27.480834 | 15.915500 |
| Validation observation | Main | 120 | 1.983042 | 0 |
| Selected owner release | Main | 24 | 0.921667 | 0 |
| Autorelease drain | Main | 24 | 0.002416 | 0 |
| Projection submit-to-return | Nested inclusive | 24 | 4.433834 | 1.021417 |
| FFN submit-to-return | Nested inclusive | 24 | 372.651040 | 30.936583 |
| Head total | Inclusive | 24 | 445.652665 | 61.794042 |

The sum of the 12 disjoint main wall phases is 444.975717 ms; the inclusive head total leaves a 0.676948 ms wall residual across the 24 calls (0.028206 ms per call), consistent with local destruction and instrumentation gaps outside the named spans. The disjoint GPU totals sum to the inclusive head GPU total. The source profile explicitly keeps the submit-to-return spans nested so they are not double-counted.

Selected mapping/admission averaged 0.086 ms per head call. Validation observation averaged 0.083 ms per head call (120 observations across 24 calls). Both are included in head wall timing; subtracting either is not a production performance estimate.

## Comparison with v3 and interpretation

V3 and v4 have the same corpus hash, target-continuation hash, 23/24 agreement, case split (8/8, 7/8, 8/8), eight-row target-logit parity gate, and eight-row shifted cache-alignment gate. V4's decoder head wall is 0.455% higher and reported GPU time 0.533% higher than v3. The serial target-with-hidden-tap wall time is 3.55% higher in v4; this is unpaired run drift and is not attributed to the profiling hooks. The profile note says the v4 instrumentation does not change model math, precision, cap, mapping lifetime, wait function, sampling, or alignment gate.

The 14.238 ms span gap is diagnostic evidence about timing boundaries, not an estimate of speedup available from a code change. In particular, GPU command start/residency delay and host wake-after-wait latency remain unresolved. No phase subtraction is used to claim improved production cost.

The measured result retains the prior feasibility limits: selected weights returned with zero live bytes; peak registered selected weights were 73,269,248 bytes; the static cache peak was 609 entries; and sampled Metal allocation peak was 106,012,295,168 bytes against a 109,676,289,225-byte cap. The prospective production overlap margin remains a projection, not production admission.

## Source and follow-up

The v4 source freeze is based on Stark `379e5c79d1cfae249e947740732b783b1a6fec82`, with source-freeze SHA-256 `c513dfa73405b60aaea2fdff82811a4cca14b4dfa08189bfece374ddfd0c5f24`. The no-run v4 build receipt SHA-256 is `a53560cfb1c37d0ba29717d0c53690c2744a7b526fdc8559c7cbd4ee0fb3050c`; it records source unchanged during build. The guarded result SHA-256 is `8c318b7fa713a78a3da5184c11bdaaac1b3e0c4d18a355f81e5b91016621990c`, and its guard receipt SHA-256 is `d777bf06b7f64356d6921392001a3a5463f8e25504b0da8a2e2cb2d614b380ec`; the guard confirms source before/after and unchanged identities for all four GGUF shards. The v4 test binary SHA-256 is `ff4a3751e26ce0d2479fa5d3a1509de3ee929113536e7ee11687e56a16959741`; driver SHA-256 is `6e2866ef15b5c40226183eeae96bfda0ab94a9fada2978c0ee91091cff91f584`.

At this v4 checkpoint, the test-only polling-versus-blocking v5 comparison had not been executed. T2 measurements are outside this profile's scope.

The public canonical [v4 source patch](profile-v4.patch) has SHA-256 `231845e605bb864920e4a2df049b5e28fa89cf69726bd2d20251f7502cc5d67a` and is based on `379e5c79d1cfae249e947740732b783b1a6fec82`. It was generated with external diff and text conversion disabled (`--no-ext-diff --no-textconv`), full file hashes, and binary patch output. It declares all nine v4 source paths, including four `new file mode 100644` entries. The sealed archive manifest records that forward application reconstructed all nine freeze hashes and the reverse check passed. Exact paths and proof counts are in [profile-v4.json](profile-v4.json); no source or file was changed from the sealed archive.

The unchanged private evidence archive `mtp-probe-v4-evidence.tgz` has SHA-256 `1750767839fae360df91493012189a84dc9f74ec4065d9214b97a530bbaf3fa9` and size 32,124,359 bytes. Its canonical `prototype.patch` publishes the nine source files, including authored public test cases. The engine binary, model artifacts, private receipts, and raw runtime payloads remain private.
