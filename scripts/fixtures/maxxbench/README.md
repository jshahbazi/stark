# MaxxBench image-handoff test fixture

`run.py` is copied without modification from
[`unlap-labs/maxxbench`](https://github.com/unlap-labs/maxxbench), revision
`1a0df5e9c5d4bccbae131b01830d478088fea947`. Its Apache-2.0 license is preserved
in `LICENSE` beside it. This fixture contains the upstream runner hook, with no
task instructions, reference patches or model outputs.

Model-free tests execute that exact hook against synthetic verifier contexts.
The evaluation runner continues to import the hook from the separately checked
out and validated upstream release; this fixture is not the runtime implementation.
