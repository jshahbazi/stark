From the Stark repository root, build and install the local Python binding:

```sh
cargo tools install python
cargo tools build python
```

The build installs the wheel into `crates/legacy/stark/bindings/python/.venv`.
The fork has no published PyPI package.
