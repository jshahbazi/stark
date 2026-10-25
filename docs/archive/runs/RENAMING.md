# Stark identity and compatibility

Active Rust crates and imports use `stark-engine`, `stark-engine-macros`, and `stark`. The independent scorer binaries are `stark-score` and `stark-score-digest`, version 0.1.0. Source bindings use Stark identity. The root Swift manifest expects a locally built `stark.xcframework`; an upstream Uzu binary cannot satisfy that renamed ABI.

These names remain intentional:

- Uzu Git history, upstream license, revision metadata, and original patch hashes.
- Mirai model registry/provider IDs, API schema names, model downloads, and dependency repository URLs.
- Registry backend IDs `uzu`/`Uzu`, serialized telemetry field `uzu_version`, and fallback key `__uzu_unparsed_arguments`, which preserve external compatibility.
- Historical `UZU_CAPTURE_` environment names, which the scoring worker clears together with `STARK_CAPTURE_` names.
- Mirai attribution and artwork in inherited samples; those samples are not a new Stark app release.

Inherited self-hosted CI, Prefect benchmark deployments, signing, package publishing, and cloud artifact destinations are unsupported and disabled. Building the Rust scoring worker is the qualified local path. Regenerating and qualifying all native bindings and packaging is separate work.
