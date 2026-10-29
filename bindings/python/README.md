# Python binding

This optional package lets `torchlean.compile` and `torchlean.verify` call the existing Lean IR
checker inside Python. It also supplies the same PyTorch capture adapter as the command-line
checker. Model execution and differentiation stay with PyTorch or JAX.

Use a CPython 3.14 or newer virtual environment with a shared Python library; free-threaded Python
is not supported by Nerodia. The binding uses TorchLean's pinned Lean toolchain and CPU build.
From the TorchLean checkout, with that environment activated:

```bash
python -m pip install "setuptools>=82" setuptools-lean wheel
TORCHLEAN_PACKAGE_ROOT="$PWD/bindings/python" PYTHON3="$(command -v python)" \
  scripts/lake.sh build +TorchLeanPython:nerodiaExt
python -m pip install --no-build-isolation ./bindings/python
```

Continue using `import torchlean` from the checkout. The installed `_torchlean` extension is
selected automatically. An explicit `checker=` argument or `TORCHLEAN_VERIFY` selects the
command-line checker instead. Without the extension, the ordinary executable interface remains
available. Errors from an installed extension propagate to the caller.

Both interfaces return the same reports, including failures and unsupported requests. The
binding supplies no additional proof of frontend translation or native kernel correctness.
Nerodia remains an experimental dependency, isolated from TorchLean's main Lake package.

The package builds from this checkout and its parent source; it is not a standalone source
distribution. A built wheel contains the extension and its bundled Lean runtime libraries.
