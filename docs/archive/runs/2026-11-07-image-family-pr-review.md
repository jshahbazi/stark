# Image-family pre-PR review

This is the independent review and documentation follow-up to the
[image-family integration record](2026-11-07-image-family-integration.md).
It retains that record's checkpoint, arithmetic, device and execution limits.
The branch includes the supporting native Gemma 4 and OLMo3 foundation that
was absent from main; image coverage is the principal user-facing addition.

## Review and correction

Two separate reviews examined Qwen/Lily admission, host routing, state reuse,
image MTP exclusion and Gemma conversion/runtime admission. The Qwen/Lily
review found no blocking correctness issue. It found a stale opening status
in the Qwen qualification record, now corrected to reflect the passing small
packed source comparison while retaining the larger decoder parity limit.

The Gemma review reproduced a named-profile admission error: a changed decoder
could retain an official Gemma 4 image profile despite altered layer count or
global RoPE base. Artifact identities remained distinct; the problem was
admission/provenance correctness, not inheritance of another artifact's checks.
Commit `f8293f87db7052da910acc378d21470072b1d4a7` fixes this by hashing and
parsing the same pinned source configuration bytes during named conversion,
then comparing the complete typed native decoder against its canonical profile.
The native check includes geometry, RoPE, precision selectors, PLE and MoE.
Generic text conversion remains separate.

The original E2B artifact predates the embedding materialization selectors.
Its exact pair of false defaults remains an explicit legacy admission contract;
new canonical artifacts use the current pair of true values. Mixed pairs are
rejected. Decoder-only fixtures preserve actual retained E2B/E4B compatibility.
These fixtures contain public configuration metadata, with no weights or
request data, and establish structural compatibility rather than numerical parity.

Main was merged at `b5ef8164eeb9c136483d2e7b413342bb143b6e49`, retaining both
the protocol-stop and image imports and both menu-bar capability checks.
Commit `a612db68` then applied Ruff formatting and import sorting to seven
converter files. The native engine, worker and Lily source trees are identical
between `f8293f87` and `a612db68`.

## Documentation and website

The README now separates BF16 Qwen, quantized Qwen dense/MoE, retained Lily,
Gemma 3 and Gemma 4 coverage, and lists the larger checkpoints without full-model
execution evidence. The native guide includes the retained Lily image upgrade
command and distinguishes default text preparation from explicit image artifacts.
The original-format Lily setup link and generation-protocol anchor were corrected.

Website work starts from the existing public Site source commit
`ff5a64b80af5218fbb400064cb812a7e10144692` (saved version 61), preserving its
newer documentation/navigation layout. Tracked source was synchronized into
`website/` before adding the image update. The Images panel links to a coverage
section with explicit development-build/under-review wording and validation
limits. The installed app can have earlier capabilities. No image-performance
or release-availability claim was added.

Independent documentation review checked four HTML pages and 72 local links
and assets with no missing targets, fragments or duplicate IDs. It also corrected
the app's exact “Connect your own app” label and narrowed the benchmark-copy
claim to the three byte-identical Qwen3.8 JSON records. Historical GitHub targets
whose commits were absent locally were not independently verified online.

Local browser checks covered the technical Images tab, its coverage link,
the documentation section and a 390-pixel mobile viewport. No horizontal
overflow was observed. Static build and JavaScript syntax checks passed.

Sites saved and published version 62 from source commit
`ee148735e6d654c36d5fcb66fdee57e1b0f4f52b`. Deployment
`appgdep_6aefd705675481919669278960b27dbc` succeeded; the deployment archive
SHA256 is `9a2b3bf4945cfdb6dee54f8269e25d8fd87be31350d46e20a3c99763a27ec41c`.
The published documentation at `https://starkengine.app/docs#image-support`
was opened and visually checked. The existing public audience was preserved.

## Validation

The integrated native CPU/Metal test binary is sealed from `f8293f87`, with
SHA256 `f7b3372d90de9e5a3fa5b6097f22720492df965ed450f35adc9031c59bacc93b`.
It uses nightly-2026-10-01, no default features, `cpu,metal`, no debug information
and no incremental compilation. Its engine source tree is
`523554a8eec6e42a03d6c9da837f33d0e5bc83f2`.

Rust 1.98.1 Clippy for all worker targets passed with warnings denied.
Repository policy, nightly formatting, converter Ruff checks/formatting and
whitespace checks passed. The merged menu-bar controller suite passed 134 checks.
Focused Gemma converter/source/runtime checks passed 134 tests before integration.

The integrated CPU/Metal regression passed **931 tests**, with zero failures,
130 ignored and six model/benchmark tests filtered out. It includes all five
canonical Gemma 4 image decoder controls, both retained decoder controls, and
the new mutation-rejection tests. The receipt is
`runtime/qualification-builds/f8293f87/native-regression.json` (ignored locally).

The final converter suite passed **6,604 tests**, with zero failures and 96
skipped. It started at `a612db68`; documentation commit `94697f10` landed while
it ran, without changing converter sources. The converter tree is
`368e7571ce8736faf38a6005d1e377b704a315de`. The live module path and sealed helper
were checked before execution. Rust 1.98.1 built that helper from `f8293f87`;
its SHA256 is `b02fbabb7ffa56ecdea3ccfb73a3088d00a11e1441326501580d83d19491ea27`.
The final capture SHA256 is
`aa13b1a3d312cde51bfe801aaa7d4c123cb083d9547d1b35c8376ab1c93bd7e8`, with
aggregate receipt `runtime/test-receipts/converter-fullsuite-live-a612db68.json`.
Earlier failed invocations using an absent helper path or a helper predating
the merged template code are not counted as successful validation; their
aggregate receipts remain separate.

The final Rust 1.98.1 release worker has SHA256
`a49cb0620f5634772712d9ce44c6fbb2c61f845593d2cb05d9b0cb7fc5296988`.
It was built after the admission fix, with engine and worker trees identical
between `f8293f87` and `a612db68`. Its 76-file worker/host/harness seal is
`87f660dd4fd5d2cce87919a2cdeb781601b29d155f802c4e72342c4bd0b27bb6`.
The checked vision reference selectors are absent from the release executable.

The retained E2B canonical replay passed all three cases with exact greedy
controls, red/blue discrimination, fresh/restart replay, malformed-image
recovery and single-use/staging cleanup. Receipt SHA256 is
`0676f408016e6cfd9480194c1e25ecc8eca2f7d744559bd9fbc67559ad45395d`.
Peak owned RSS was 10,242,211,840 bytes; minimum available memory was
19,671,007,232 bytes. No guard fired. This was a functional check with a
competing worker present, not a quiet-host performance measurement.

The corresponding retained E4B replay also passed all three canonical greedy
cases, red/blue discrimination, fresh/restart replay, malformed-image recovery,
single-use admission and staged cleanup. Its receipt SHA256 is
`a52bbfefe08ee28c15f6ff8fce9272021978484ec80b5d56e6cd1d80fec4141c`.
Peak sampled owned RSS was 15,884,877,824 bytes; minimum available memory was
15,980,675,072 bytes. Both harnesses exited zero and neither guard nor timeout
fired. These retained-artifact greedy controls establish functional replay,
not full logit parity for newly converted canonical artifacts.

## Limits

The original integration record remains authoritative for qualified model
execution and component parity. Full BF16-source Qwen 35B MoE, Gemma 3 12B/27B
and Gemma 4 12B/26B/31B generation remain unverified. Component checks do not
establish those results. Image throughput and the installed UI were not measured.
Opening a PR does not establish hosted CI or Security success, and no merge is
authorized by this record.
