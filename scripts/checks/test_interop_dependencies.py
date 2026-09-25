"""Exercise dependency-gate failures without loading native Python packages."""

from contextlib import redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check_interop_dependencies


class InteropDependencyTests(unittest.TestCase):
    def run_check(self, unavailable: str | None = None, error: Exception | None = None):
        def load(name: str):
            if name == unavailable:
                raise error or ModuleNotFoundError(f"No module named '{name}'")
            return SimpleNamespace(__version__="test-version")

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(check_interop_dependencies.importlib, "import_module", side_effect=load):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = check_interop_dependencies.main([])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_success_requires_all_three_imports(self) -> None:
        status, stdout, stderr = self.run_check()
        self.assertEqual(status, 0)
        for name in ("torch", "numpy", "onnx"):
            self.assertIn(f"{name}: test-version", stdout)
        self.assertEqual(stderr, "")

    def test_each_missing_dependency_fails(self) -> None:
        for name in ("torch", "numpy", "onnx"):
            with self.subTest(name=name):
                status, _, stderr = self.run_check(name)
                self.assertEqual(status, 1)
                self.assertIn(f"{name}: import failed", stderr)
                self.assertIn("requirements-interop.txt", stderr)

    def test_broken_native_import_fails(self) -> None:
        status, _, stderr = self.run_check("torch", OSError("shared library unavailable"))
        self.assertEqual(status, 1)
        self.assertIn("shared library unavailable", stderr)


if __name__ == "__main__":
    unittest.main()
