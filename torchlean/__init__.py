"""Capture tensor computations and request explicit checks from TorchLean.

``verify(model, example)`` checks the captured graph's declared structure and
types. Add ``checks=("structure", "types", "compare")`` to compare CPU reference
IR execution with the frontend on that example, or ``compare_bitwise`` to compare
NaN words too. Neither result is a proof of frontend equivalence or correctness on
every input.

Capture and translation happen automatically. ``export="Model.lean"`` optionally
keeps that same artifact in a self-contained Lean module for further proofs;
``export="model.json"`` saves the interchange artifact instead.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any


def _command(checker: str | Path | Sequence[str] | None) -> list[str]:
    if isinstance(checker, (str, Path)):
        return [str(checker), "ir"]
    if checker is not None:
        if (not isinstance(checker, Sequence) or not checker
                or not all(isinstance(argument, str) for argument in checker)):
            raise TypeError("checker must be a path or a nonempty sequence of command arguments")
        return list(checker)
    path = os.environ.get("TORCHLEAN_VERIFY")
    if path:
        return [path, "ir"]
    executable = Path(__file__).resolve().parent.parent / ".lake/build/bin/verify"
    if not executable.is_file():
        raise FileNotFoundError(
            "Build the TorchLean `verify` executable first, or supply checker= with "
            "a command ending in `verify ir`. Verification does not start a build implicitly."
        )
    return [str(executable), "ir"]


def _run(command: list[str], arguments: Sequence[str], *, report: bool = False
         ) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command + list(arguments), capture_output=True, text=True, check=False)
    if result.returncode not in ((0, 1) if report else (0,)):
        raise RuntimeError(f"TorchLean checker exited with {result.returncode}: {result.stderr.strip()}")
    return result


def _capture(model: Any, examples: tuple[Any, ...], frontend: str, command: list[str],
             directory: Path, checks: Sequence[str]) -> dict[str, Any]:
    compare = "compare" in checks or "compare_bitwise" in checks
    execute = compare or "execute" in checks
    if not examples or examples[0] is None:
        raise ValueError("capturing a computation requires example tensors")
    if frontend == "jax":
        from NN.Runtime.JAX.jaxpr import Unsupported, encode, export, references

        try:
            artifact = export(model, *examples)
        except Unsupported as error:
            raise NotImplementedError(str(error)) from error
        if execute:
            artifact["arguments"] = [encode(value) for value in examples]
        if compare:
            artifact.update(references(model, *examples))
        return artifact
    if frontend != "pytorch":
        raise ValueError(f"unknown frontend {frontend!r}; use pytorch, jax, or pass an artifact")
    import torch

    if not all(isinstance(value, torch.Tensor) for value in examples):
        raise NotImplementedError("PyTorch capture requires positional tensor arguments")

    path = directory / "capture.py"
    _run(command, ["--export-pytorch", str(path)])
    spec = importlib.util.spec_from_file_location("torchlean_capture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the emitted PyTorch adapter")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    # Capture and ShapeProp may execute code. Keep the caller's model and input
    # intact; arbitrary external effects in user Python remain outside this API.
    if not isinstance(model, torch.nn.Module):
        if not callable(model):
            raise TypeError("PyTorch capture requires a module or callable")
        function = model

        class Function(torch.nn.Module):
            def forward(self, *inputs):
                return function(*inputs)

        model = Function()
    model = copy.deepcopy(model)
    examples = tuple(value.detach().clone().requires_grad_(value.requires_grad)
                     for value in examples)
    artifact = adapter.export_torchlean_graph_json(
        model, examples, str(directory / "graph.json"), require_torch_export=True)
    if execute:
        artifact["arguments"] = [adapter._constant(value) for value in examples]
    if compare:
        with torch.no_grad():
            outputs, _ = torch.utils._pytree.tree_flatten(model(*examples))
        if not all(isinstance(value, torch.Tensor) for value in outputs):
            raise ValueError("sample comparison requires tensor outputs")
        artifact["reference_outputs"] = [adapter._constant(value) for value in outputs]
    return artifact


def _lean_string(value: str) -> str:
    escapes = {'"': '\\"', '\\': '\\\\', '\n': '\\n', '\r': '\\r', '\t': '\\t'}
    return '"' + ''.join(escapes.get(char, f"\\u{ord(char):04x}" if ord(char) < 32 else char)
                         for char in value) + '"'


def _lean_json(value: Any, indent: int = 2) -> str:
    """Quote JSON constructors so Lean can reduce the importer inside proofs."""
    if value is None:
        return ".null"
    if isinstance(value, bool):
        return ".bool " + ("true" if value else "false")
    if isinstance(value, (int, Decimal)):
        number = Decimal(value)
        sign, digits, exponent = number.as_tuple()
        mantissa = int("".join(map(str, digits))) * (-1 if sign else 1)
        if exponent > 0:
            mantissa *= 10 ** exponent
        return f".num ⟨{mantissa}, {max(0, -exponent)}⟩"
    if isinstance(value, str):
        return ".str " + _lean_string(value)
    if isinstance(value, list):
        opening = ".arr #["
        entries = [_lean_json(item, indent + 2) for item in value]
    else:
        opening = "Lean.Json.mkObj ["
        entries = [f"({_lean_string(key)}, {_lean_json(item, indent + 2)})"
                   for key, item in value.items()]
    inline = opening + ", ".join(entries) + "]"
    if "\n" not in inline and len(inline) + indent <= 96:
        return inline
    padding = " " * (indent + 2)
    return opening + "\n" + padding + (",\n" + padding).join(entries) + "\n" + " " * indent + "]"


def _export(source: str, path: Path) -> None:
    if path.suffix == ".json":
        content = source
    else:
        # Json.parse is partial and cannot reduce in the kernel. Quote its data
        # constructors, decoding decimal literals without another float conversion.
        literal = _lean_json(json.loads(source, parse_float=Decimal))
        content = f'''module

public import NN.IR.Properties
public import NN.Runtime.IR
import all Lean.Data.Json.Basic
import all Lean.Data.Name
import all Init.Data.Ord.String
import all Init.Data.Range.Basic
import all Init.Prelude
import all Init.Meta.Defs

/-!
# Captured tensor program

This module embeds the artifact captured by `torchlean.verify`. The constants,
types, casts and input/output order belong to the same capture as the check report.
The import and validation proofs below do not prove frontend or kernel equivalence.
Write further properties about `program` using its tensor semantics.
-/

@[expose] public section

namespace «{path.stem}»

/-- The captured artifact, including its exact element words and operation attributes. -/
def artifact : Lean.Json :=
  {literal}

/-- The typed program obtained from this artifact by the shared importer. -/
def program : NN.IR.Program :=
  (Import.IR.parseProgram artifact).toOption.get (by decide +kernel)

/-- The program is the importer's result for this exact artifact. -/
theorem program_imported : Import.IR.parseProgram artifact = .ok program := by
  have success : (Import.IR.parseProgram artifact).toOption.isSome := by decide +kernel
  cases h : Import.IR.parseProgram artifact with
  | error message => simp [h, Except.toOption] at success
  | ok value => simp [program, h, Except.toOption]

/-- The imported program passes the shared shape, type and interface checks. -/
theorem program_checked : program.check = .ok () :=
  Import.IR.parseProgram_checked program_imported

end «{path.stem}»
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    # An export may become a hand-written proof file. Never replace it on recapture.
    with path.open("x", encoding="utf-8") as output:
        output.write(content)


def verify(model: Any, example: Any = None, *examples: Any, frontend: str = "pytorch",
           checks: Sequence[str] = ("structure", "types"),
           checker: str | Path | Sequence[str] | None = None,
           export: str | Path | None = None) -> dict[str, Any]:
    """Check a captured computation or an existing typed IR artifact.

    ``model`` is a PyTorch module, a JAX function with ``frontend="jax"``, or an
    artifact mapping. Both frontends accept positional tensor arguments, as in
    ``verify(function, x, y)``. ``checks`` may request ``structure``, ``types``,
    ``execute``, ``compare`` or ``compare_bitwise``. ``compare`` lets two NaN results
    agree whatever their sign and payload; ``compare_bitwise`` compares their words.
    Unknown requests produce unsupported entries, not a pass.

    Capture and translation are automatic. Optional ``export="model.json"`` saves
    the exact artifact sent to Lean; ``export="Model.lean"`` embeds it in a Lean
    module exposing ``Model.program`` and its import/validation proofs. Further
    mathematical proofs can refer to this program without recapturing the function.
    Export writes proof source; open or compile the Lean module to check its proofs.
    The Lean filename must be an ASCII identifier. Existing files are never replaced.
    Failed checks or checker errors still leave the capture for inspection; unsupported
    capture creates no file. Invalid graphs cannot pass the generated Lean validation proof.

    ``checker`` is an executable path (the ``ir`` subcommand is appended), or a
    complete argument sequence already ending in the IR checker command. No shell
    is used. Omitting it uses ``TORCHLEAN_VERIFY`` or the local built executable.

    The returned ``ok`` covers only the requested checks. Inspect ``checks`` for
    their scope. In particular, a sample comparison is not an all-input theorem,
    and a mismatch may reflect different reduction or transcendental
    implementations rather than a frontend bug. PyTorch's tanh GELU depends on the
    CPU capability level its kernel runs at. The IR follows the AVX2 and AVX-512
    kernels, which fuse ``x + kappa x^3`` on x86-64; the DEFAULT level rounds that
    sum twice, and other processors and CUDA were not measured.

    For JAX, the reference is ``model(*examples)``, which the artifact records as
    ``reference_execution``: ``"jit"`` for a ``jax.jit`` function, ``"eager"`` otherwise, with
    the platform that computed it and the jaxlib version. When ``jax.jit`` compiles any part of
    the function, the artifact also holds the outputs of evaluating it one primitive at a time.
    Every mismatch fails. When the reference came from XLA's CPU backend and re-running the
    program as XLA's CPU code computes it, as far as that is exactly emulable, reproduces every
    differing element, the failed entry lists those documented behaviours in
    ``consistent_with``; under ``jax.jit`` this also needs the primitive-by-primitive outputs to
    agree with the IR or to be reproduced the same way. The field is advice: an importer error
    with the same effect cannot be ruled out, and some coincide with XLA's own rewrites. The
    checker trusts the recorded execution, platform and outputs, so a forged record can at most
    add this advice to a failure.
    """
    if (isinstance(checks, str) or not isinstance(checks, Sequence)
            or not all(isinstance(name, str) for name in checks)):
        raise TypeError("checks must be a sequence of names, not a single string")
    if not checks:
        raise ValueError("request at least one check")
    if isinstance(model, Mapping) and (example is not None or examples):
        raise ValueError("an artifact already contains its arguments; do not supply example tensors")
    if export is not None and not isinstance(export, (str, Path)):
        raise TypeError("export must be a .json or .lean file path")
    destination = Path(export) if export is not None else None
    if destination is not None:
        if destination.suffix not in (".json", ".lean"):
            raise ValueError("export must end in .json or .lean")
        if destination.suffix == ".lean" and not (
                destination.stem.isascii() and destination.stem.isidentifier()):
            raise ValueError("a Lean export filename must be an ASCII identifier, such as Model.lean")
        if destination.exists():
            raise FileExistsError(f"refusing to replace existing export: {destination}")
    command = _command(checker)
    with tempfile.TemporaryDirectory(prefix="torchlean-verify-") as temporary:
        directory = Path(temporary)
        try:
            artifact = dict(model) if isinstance(model, Mapping) else _capture(
                model, (example, *examples), frontend, command, directory, checks)
        except NotImplementedError as error:
            # No Lean checks ran. Keep every request visible rather than fabricating
            # a successful result for a smaller graph or a different frontend.
            return dict(format="torchlean.verification.v1", ok=False, checker="not_run",
                        checks=[dict(name=name, status="unsupported",
                                     message=f"Capture is unsupported: {error}") for name in checks],
                        outputs=None, conversion_flags=None, frontend_equivalence="not_checked",
                        frontend_device_layout_constraints="not_checked",
                        external_kernels="not_checked")
        path = directory / "artifact.json"
        source = json.dumps(artifact, ensure_ascii=False, allow_nan=False) + "\n"
        path.write_text(source, encoding="utf-8")
        if destination is not None:
            _export(source, destination)
        result = _run(command, [str(path), *checks], report=True)
        try:
            report = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"checker did not produce a JSON report: {result.stderr.strip()}") from error
        if not isinstance(report, dict) or report.get("format") != "torchlean.verification.v1":
            raise RuntimeError("checker returned an unknown verification report format")
        findings = report.get("checks")
        if (not isinstance(findings, list) or not all(isinstance(item, dict) for item in findings)
                or [item.get("name") for item in findings] != list(checks)
                or type(report.get("ok")) is not bool):
            raise RuntimeError("checker report does not cover the requested checks")
        if (result.returncode == 0) != report["ok"]:
            raise RuntimeError("checker exit status conflicts with its verification report")
        return report


__all__ = ["verify"]
