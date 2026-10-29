"""Compile framework graphs to TorchLean IR, or check a concrete capture explicitly.

``@torchlean.compile`` selects PyTorch or JAX on a normal call, lowers the
framework's captured graph, and checks its structure and types. Framework
execution and automatic differentiation are preserved. ``.torchlean`` provides
the captured IR and the check reports.

``verify(model, example)`` checks the captured graph's declared structure and
types. Add ``checks=("structure", "types", "compare")`` to compare CPU reference
IR execution with the frontend on that example, or ``compare_bitwise`` to compare
NaN words too. Neither result is a proof of frontend equivalence or correctness on
every input.

Capture and translation happen automatically. ``export="Model.lean"`` optionally
keeps that same artifact in a self-contained Lean module for further proofs;
``export="model.json"`` saves the interchange artifact instead.

The optional native Python binding calls the same checker in process. Explicit
``checker=`` or ``TORCHLEAN_VERIFY`` settings select an executable instead.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

from ._checker import Checker


_NO_EXAMPLE = object()


def _detect_frontend(model: Any, examples: tuple[Any, ...],
                     kwargs: Mapping[str, Any] | None = None, frontend: str = "auto") -> str:
    """Identify framework values without importing either optional dependency."""
    if frontend not in ("auto", "pytorch", "jax"):
        raise ValueError(f"unknown frontend {frontend!r}; use auto, pytorch, or jax")
    if frontend != "auto":
        return frontend
    found: set[str] = set()
    numpy_input = False

    def inspect(value: Any) -> None:
        nonlocal numpy_input
        modules = {base.__module__.split(".")[0] for base in type(value).__mro__}
        if "torch" in modules:
            found.add("pytorch")
        if modules & {"jax", "jaxlib"}:
            found.add("jax")
        numpy_input |= "numpy" in modules
        if isinstance(value, Mapping):
            for item in value.values():
                inspect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                inspect(item)

    arguments = (examples, dict(kwargs or {}))
    inspect(arguments)
    # Registered pytree nodes can hide all of a call's tensor arguments. Use
    # frameworks the caller already imported, so a plain PyTorch call still
    # neither imports JAX nor depends on its optional installation. Inspect both
    # registries even after finding a tensor to reject mixed-framework trees.
    for module_name, flatten_name in (("torch.utils._pytree", "tree_leaves"),
                                      ("jax.tree_util", "tree_leaves")):
        module = sys.modules.get(module_name)
        flatten = getattr(module, flatten_name, None)
        if flatten is not None:
            for value in flatten(arguments):
                inspect(value)
    inspect(getattr(model, "__self__", model))
    if len(found) > 1:
        raise NotImplementedError("mixed PyTorch and JAX tensor arguments are unsupported")
    if len(found) == 1:
        return found.pop()
    if numpy_input:
        return "jax"
    raise NotImplementedError("cannot detect a frontend; supply PyTorch or JAX example tensors")


@contextmanager
def _preserve_random_state():
    """Tracing and sample checks must not consume the caller's host RNG streams."""
    state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    try:
        yield
    finally:
        random.setstate(state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


def _load_pytorch_adapter(checker: Checker, directory: Path) -> Any:
    path = directory / "capture.py"
    checker.write_adapter(path)
    spec = importlib.util.spec_from_file_location("torchlean_capture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the emitted PyTorch adapter")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def _lower_pytorch_graph(adapter: Any, model: Any, graph: Any, input_names: Sequence[str],
                         directory: Path, *,
                         tensor_values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Reuse the emitted lowerer on an already captured framework graph."""
    adapter._TENSOR_VALUES = dict(tensor_values or {})
    for node in graph.nodes:
        if node.op == "get_attr":
            value = model
            for part in str(node.target).split("."):
                value = getattr(value, part)
            if isinstance(value, adapter.torch.Tensor):
                adapter._TENSOR_VALUES[node.name] = value
    adapter._check_graph_operators(graph, model)
    adapter._capture = lambda *args, **keywords: (graph, list(input_names))
    return adapter.export_torchlean_graph_json(
        model, (), str(directory / "graph.json"), require_torch_export=True)


def _capture(model: Any, examples: tuple[Any, ...], frontend: str, checker: Checker,
             directory: Path, checks: Sequence[str], *,
             kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Lower a framework trace, retaining its flat tensor input order.

    Container structure and scalar leaves are specialized to this call. The input
    tensors remain dynamic IR inputs, including unused tensor arguments.
    """
    compare = "compare" in checks or "compare_bitwise" in checks
    execute = compare or "execute" in checks
    kwargs = dict(kwargs or {})
    frontend = _detect_frontend(model, examples, kwargs, frontend)
    original = getattr(model, "__torchlean_original__", None)
    if original is not None:
        owner = getattr(model, "__self__", None)
        model = original.__get__(owner, type(owner)) if owner is not None else original
    if frontend == "jax":
        import jax
        from NN.Runtime.JAX.jaxpr import Unsupported, encode, lower, references

        leaves, structure = jax.tree.flatten(copy.deepcopy((examples, kwargs)))
        tensors = [value for value in leaves if hasattr(value, "shape")
                   and hasattr(value, "dtype")]
        if not tensors:
            raise NotImplementedError("JAX capture requires at least one tensor argument")
        if any(not (hasattr(value, "shape") and hasattr(value, "dtype"))
               and type(value) not in (type(None), bool, int, float, str) for value in leaves):
            raise NotImplementedError("JAX static arguments must be None, bool, int, float or str")
        dynamic = [hasattr(value, "shape") and hasattr(value, "dtype") for value in leaves]
        function = copy.deepcopy(model)

        def flattened(*inputs):
            inputs = iter(inputs)
            values = [next(inputs) if is_tensor else value
                      for value, is_tensor in zip(leaves, dynamic)]
            args, keywords = jax.tree.unflatten(structure, values)
            return function(*args, **keywords)

        try:
            with _preserve_random_state():
                artifact = lower(jax.make_jaxpr(flattened)(*tensors))
                if execute:
                    artifact["arguments"] = [encode(value) for value in tensors]
                if compare:
                    artifact.update(references(flattened, *tensors))
        except Unsupported as error:
            raise NotImplementedError(str(error)) from error
        return artifact
    import torch
    from ._compiler import _capture_scope

    adapter = _load_pytorch_adapter(checker, directory)
    # Capture may execute code. Keep the caller's model and input
    # intact; arbitrary external effects in user Python remain outside this API.
    if not isinstance(model, torch.nn.Module):
        if not callable(model):
            raise TypeError("PyTorch capture requires a module or callable")

        class Function(torch.nn.Module):
            def __init__(self, function):
                super().__init__()
                self.function = function

            def forward(self, *inputs, **keywords):
                return self.function(*inputs, **keywords)

        model = Function(model)
    memo: dict[int, Any] = {}
    model = copy.deepcopy(model, memo)
    leaves, structure = torch.utils._pytree.tree_flatten((examples, kwargs))
    if any(not isinstance(value, torch.Tensor)
           and type(value) not in (type(None), bool, int, float, str) for value in leaves):
        raise NotImplementedError("PyTorch static arguments must be None, bool, int, float or str")
    for value in leaves:
        if isinstance(value, torch.Tensor) and id(value) not in memo:
            # clone() compacts non-dense views and can change a layout-dependent
            # branch. Tensor's deepcopy preserves strides and shared storage;
            # detaching first also accepts non-leaf arguments without their tape.
            memo[id(value)] = copy.deepcopy(value.detach(), memo).requires_grad_(value.requires_grad)
    tensors = [memo[id(value)] for value in leaves if isinstance(value, torch.Tensor)]
    if not tensors:
        raise NotImplementedError("PyTorch capture requires at least one tensor argument")
    copied = [memo[id(value)] if isinstance(value, torch.Tensor) else value for value in leaves]
    examples, kwargs = torch.utils._pytree.tree_unflatten(copied, structure)
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with _capture_scope(), _preserve_random_state(), torch.random.fork_rng(devices=devices):
        exported = torch.export.export(model, examples, kwargs=kwargs)
        if any(spec.kind.name != "USER_OUTPUT" for spec in exported.graph_signature.output_specs):
            raise NotImplementedError(
                "exported mutation/effects are outside the pure graph contract")
        names = []
        tensor_values = {}
        state = dict(exported.state_dict)
        constants = dict(exported.constants)
        for input_spec in exported.graph_signature.input_specs:
            if input_spec.kind.name == "USER_INPUT":
                if isinstance(input_spec.arg, torch.export.graph_signature.TensorArgument):
                    names.append(input_spec.arg.name)
                elif not isinstance(input_spec.arg, torch.export.graph_signature.ConstantArgument):
                    raise NotImplementedError("exported non-tensor dynamic inputs are unsupported")
            else:
                value = state.get(input_spec.target, constants.get(input_spec.target))
                if isinstance(value, torch.Tensor):
                    tensor_values[input_spec.arg.name] = value
        if len(names) != len(tensors):
            raise NotImplementedError(
                "exported tensor input order does not match the example pytree")
        artifact = _lower_pytorch_graph(adapter, model, exported.graph, names, directory,
                                       tensor_values=tensor_values)
        if execute:
            artifact["arguments"] = [adapter._constant(value) for value in tensors]
        if compare:
            with torch.no_grad():
                outputs, _ = torch.utils._pytree.tree_flatten(model(*examples, **kwargs))
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


def verify(model: Any, example: Any = _NO_EXAMPLE, *examples: Any, frontend: str = "auto",
           checks: Sequence[str] = ("structure", "types"),
           checker: str | Path | Sequence[str] | None = None,
           export: str | Path | None = None,
           example_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Check a captured computation or an existing typed IR artifact.

    ``model`` is a PyTorch module, a JAX function, or an artifact mapping. The
    frontend is detected from the example tensors; ``frontend`` may select it
    explicitly. Both frontends accept tensor pytrees, as in ``verify(function, x, y)``
    or ``verify(function, {"x": x}, example_kwargs={"y": y})``. Scalar arguments
    (None, bool, int, float and str) specialize the capture to their supplied values.
    ``checks`` may request ``structure``, ``types``,
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
    is used. Omitting it uses ``TORCHLEAN_VERIFY``, the optional native extension,
    or the local built executable, in that order. An installed extension's errors
    propagate instead of retrying through a different checker.

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
    if example_kwargs is not None and (
            not isinstance(example_kwargs, Mapping)
            or not all(isinstance(key, str) for key in example_kwargs)):
        raise TypeError("example_kwargs must be a mapping with string argument names")
    if isinstance(model, Mapping) and (
            (example is not _NO_EXAMPLE and example is not None) or examples or example_kwargs):
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
    runner = Checker(checker)
    with tempfile.TemporaryDirectory(prefix="torchlean-verify-") as temporary:
        directory = Path(temporary)
        try:
            arguments = () if example is _NO_EXAMPLE else (example, *examples)
            artifact = dict(model) if isinstance(model, Mapping) else _capture(
                model, arguments, frontend, runner, directory, checks, kwargs=example_kwargs)
        except NotImplementedError as error:
            # No Lean checks ran. Keep every request visible rather than fabricating
            # a successful result for a smaller graph or a different frontend.
            return dict(format="torchlean.verification.v1", ok=False, checker="not_run",
                        checks=[dict(name=name, status="unsupported",
                                     message=f"Capture is unsupported: {error}") for name in checks],
                        outputs=None, conversion_flags=None, frontend_equivalence="not_checked",
                        frontend_device_layout_constraints="not_checked",
                        external_kernels="not_checked")
        source = json.dumps(artifact, ensure_ascii=False, allow_nan=False) + "\n"
        if destination is not None:
            _export(source, destination)
        return runner.check(source, checks, directory)


from ._compiler import compile

__all__ = ["compile", "verify"]
