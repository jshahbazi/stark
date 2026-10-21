# Stark presentation and launch notes

## What changed

The public introduction now follows the visitor's decision: understand Stark, inspect the measured result, find a supported setup, then inspect its evidence. The previous README's complete technical content is preserved in [TECHNICAL_OVERVIEW.md](TECHNICAL_OVERVIEW.md), with its relative links rebased.

- **Identity:** the supplied cat logo, consistent charcoal/paper/gold colors and a concise “Native inference on Apple silicon” description.
- **README:** a shorter introduction, capability table, six-workload benchmark, earlier quickstart and direct links to technical references.
- **Website:** a responsive introduction, command tabs, copy control, accessible benchmark table with rate/relative views, visible qualification notes and a source-build path.
- **Evidence:** the 7.36% figure remains tied to pinned Qwen3.5-2B, one M5 Max and accepted pairs. Historical timings, scoring validation and converter measurements keep their separate scopes.
- **Preservation:** no runtime code, dependency lock, model qualification or inherited release infrastructure changes.

The site is an engineering preview. Its creation does not qualify a production release, an HTTP serving comparison or broad model quality.

## Recommended repository metadata

Use this description when updating GitHub's About panel:

> Native Rust/Metal inference, probability scoring and model preparation on Apple silicon. Experimental Uzu-derived engine with reproducible benchmarks.

Suggested topics: `apple-silicon`, `metal`, `rust`, `llm-inference`, `local-ai`, `model-conversion`.

Use the supplied logo for the repository's visual identity. A dedicated wide social-preview card would improve link previews; retain the full cat/wordmark and keep benchmark qualifiers readable if including a speed claim. Set the official homepage only after selecting a public site audience and URL. A private preview should not be advertised as a public homepage.

## Remaining work before a broader launch

1. **Deliver the source people are reading about.** Integrate the reviewed presentation and measured runtime branches into the chosen public branch, and name a reproducible release or tag. This task pushes a reviewable branch; it does not merge or open a PR.
2. **Make first use easier.** Verify the exact fresh-machine build and model-preparation path, then consider signed packages through a separately scoped distribution plan. Current instructions require Rust, Xcode's Metal tools and `uv`.
3. **Strengthen the public performance claim.** Repeat the frozen comparison in a fully accepted campaign, add independent hardware/model replication and investigate quality failures. Benchmark HTTP serving or concurrency separately if advertising those workloads.
4. **Show a small real integration.** Publish a documented generation or scoring example with application-specific acceptance checks. Keep hypothetical examples clearly labelled and private evaluation payloads outside Git.
5. **Finish the public launch configuration.** Choose the site's audience/domain, set GitHub About/topics/social preview, and verify outbound links against the released revision.

These are follow-up recommendations. The existing benchmark remains a preliminary local result and the source-build setup remains the current entry path.

## Verification

The presentation changes use dependency-free static site generation. Verification covers source-bound table values, archived README preservation, local/immutable links, original-logo hashes, JavaScript syntax, repository policy, desktop/mobile browser layout and the implemented controls. Native inference benchmarks are not rerun for this documentation and design change.
