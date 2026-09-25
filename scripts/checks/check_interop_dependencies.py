#!/usr/bin/env python3
"""Require the Python dependencies used by the PyTorch and ONNX runtime checks."""

import argparse
import importlib
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(f"Interop Python: {sys.executable}")
    failed = False
    for name in ("torch", "numpy", "onnx"):
        try:
            module = importlib.import_module(name)
        except Exception as error:
            print(f"{name}: import failed: {error}", file=sys.stderr)
            failed = True
        else:
            print(f"{name}: {module.__version__}")
    if failed:
        print(
            "Install with python3 -m pip install -r scripts/checks/requirements-interop.txt "
            "and use the same Python environment for the Lean executables.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
