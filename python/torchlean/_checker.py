"""Call the same Lean checker through an optional extension or an executable."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
import subprocess


class Checker:
    """Resolve the checker once.

    The executable is ``checker`` if given, else ``TORCHLEAN_VERIFY`` if set. Without either,
    the native ``_torchlean`` extension is used when installed, and the built
    ``.lake/build/bin/verify`` otherwise.
    """

    def __init__(self, checker: str | Path | Sequence[str] | None):
        self._native = None
        self._command: list[str] = []
        configured = os.environ.get("TORCHLEAN_VERIFY")
        if checker is None and not configured:
            try:
                self._native = importlib.import_module("_torchlean")
            except ModuleNotFoundError as error:
                if error.name != "_torchlean":
                    raise
            if self._native is not None:
                if not all(callable(getattr(self._native, name, None))
                           for name in ("check", "op_table")):
                    raise RuntimeError("the installed _torchlean extension has an incompatible API")
                return
        if isinstance(checker, (str, Path)):
            self._command = [str(checker), "ir"]
        elif checker is not None:
            if (not isinstance(checker, Sequence) or not checker
                    or not all(isinstance(argument, str) for argument in checker)):
                raise TypeError("checker must be a path or a nonempty sequence of command arguments")
            self._command = list(checker)
        elif configured:
            self._command = [configured, "ir"]
        else:
            executable = Path(__file__).resolve().parents[2] / ".lake/build/bin/verify"
            if not executable.is_file():
                raise FileNotFoundError(
                    "Install the optional TorchLean Python binding or build the `verify` "
                    "executable with scripts/lake.sh build verify. You can also supply checker= "
                    "with a command ending in `verify ir`. Verification never starts a build.")
            self._command = [str(executable), "ir"]

    def _run(self, arguments: Sequence[str], *, report: bool = False
             ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(self._command + list(arguments), capture_output=True,
                                text=True, check=False)
        if result.returncode not in ((0, 1) if report else (0,)):
            raise RuntimeError(
                f"TorchLean checker exited with {result.returncode}: {result.stderr.strip()}")
        return result

    def op_table(self, names: Sequence[str], directory: Path) -> str:
        """Return the JSON text of Lean's op table for the dtypes `names`."""
        if self._native is not None:
            return self._native.op_table(json.dumps(list(names)))
        path = directory / "op-table.json"
        self._run(["--op-table", str(path), *names])
        return path.read_text(encoding="utf-8")

    def check(self, source: str, requests: Sequence[str], directory: Path) -> dict:
        """Validate the report identically for native and subprocess calls."""
        status = None
        stderr = ""
        if self._native is None:
            path = directory / "artifact.json"
            path.write_text(source, encoding="utf-8")
            result = self._run([str(path), *requests], report=True)
            output, stderr, status = result.stdout, result.stderr, result.returncode
        else:
            output = self._native.check(source, json.dumps(list(requests)))
        try:
            report = json.loads(output)
        except (json.JSONDecodeError, TypeError) as error:
            raise RuntimeError(f"checker did not produce a JSON report: {stderr.strip()}") from error
        if not isinstance(report, dict) or report.get("format") != "torchlean.verification":
            raise RuntimeError("checker returned an unknown verification report format")
        findings = report.get("checks")
        if (not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings)
                or [item.get("name") for item in findings] != list(requests)
                or type(report.get("ok")) is not bool):
            raise RuntimeError("checker report does not cover the requested checks")
        if status is not None and (status == 0) != report["ok"]:
            raise RuntimeError("checker exit status conflicts with its verification report")
        return report
