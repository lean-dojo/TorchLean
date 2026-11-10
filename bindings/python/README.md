# Python binding

This optional package lets `torchlean.compile` and `torchlean.verify` call the Lean IR checker
inside Python. It also returns Lean's op table (`_torchlean.op_table`, the JSON that
`verify ir --op-table` writes), which the Python captures read. PyTorch or JAX runs and
differentiates the model itself.

Use a virtual environment with CPython 3.14 or newer and a shared Python library. Nerodia does not
support free-threaded Python. The binding uses TorchLean's pinned Lean toolchain and CPU build.
From the TorchLean checkout, with that environment activated:

```bash
python -m pip install "setuptools>=82" setuptools-lean wheel
TORCHLEAN_PACKAGE_ROOT="$PWD/bindings/python" PYTHON3="$(command -v python)" \
  scripts/lake.sh build +TorchLeanPython:nerodiaExt
python -m pip install --no-build-isolation ./bindings/python
```

Keep using `import torchlean` from the checkout, with `python/` on `PYTHONPATH` or after
`python -m pip install -e python`. It picks up the installed `_torchlean` extension, and a
`checker=` argument or `TORCHLEAN_VERIFY` selects the command-line checker instead. Without the
extension, the package runs the `verify` executable. With it, the extension's errors reach the
caller, with no fallback to the executable.

Both interfaces return the same reports, including failures and unsupported requests. The binding
adds no proof that the frontend translation or the native kernels are correct. Nerodia is an
experimental dependency and stays out of TorchLean's main Lake package.

The package builds from this checkout and its parent source, and is not a standalone source
distribution. A built wheel contains the extension and the Lean runtime libraries it bundles.
