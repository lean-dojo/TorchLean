"""Lazy framework compilation with a checked TorchLean IR for each captured graph.

PyTorch owns its specialization guards and executes the captured FX graph. JAX
compiles the closed Jaxpr whose lowering we check. Neither path substitutes a
second implementation of the model or of its framework's automatic differentiation.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar
import functools
import inspect
import sys
import tempfile
import threading
import types
import weakref
from pathlib import Path
from typing import Any

_capture_context = threading.local()
_execution: ContextVar[list | None] = ContextVar("torchlean_execution", default=None)
_torch_compile_lock = threading.RLock()
_module_entry_codes: weakref.WeakKeyDictionary[type, types.CodeType] = weakref.WeakKeyDictionary()


@contextmanager
def _capture_scope():
    """Let an explicit TorchLean export inline decorated model calls."""
    previous = getattr(_capture_context, "active", False)
    previous_owner = getattr(_capture_context, "compilation", None)
    previous_exporting = getattr(_capture_context, "exporting", False)
    _capture_context.active = True
    _capture_context.exporting = True
    try:
        yield
    finally:
        _capture_context.active = previous
        _capture_context.compilation = previous_owner
        _capture_context.exporting = previous_exporting


def _torch_backend(graph, inputs):
    """Share the backend identity so instances do not exhaust Dynamo's code cache."""
    from ._checker import Checker
    from ._pytorch import lower

    owner = _capture_context.compilation
    with tempfile.TemporaryDirectory(prefix="torchlean-compile-") as temporary:
        artifact = lower(graph, inputs, Checker(owner._checker), Path(temporary))
    token = object()
    owner._adopt(token, artifact)

    def execute(*values):
        # A guarded graph may be reused by another model instance. Its parameters
        # are live arguments; its check must also be visible to that instance.
        current = _capture_context.compilation
        current._adopt(token, artifact)
        invocation = _execution.get()
        if invocation is not None:
            invocation.append(artifact)
        return graph.forward(*values)

    return execute


@contextmanager
def _torch_guard_scope():
    import torch

    # Dynamo otherwise omits hook guards, so installing a hook after the first
    # call can silently reuse a graph that never runs it. This capture context
    # enables native guards for the complete call, including nested modules.
    # Serialize our temporary config scopes so concurrent calls restore them.
    with _torch_compile_lock, torch._dynamo.config.patch(skip_nnmodule_hook_guards=False):
        yield


def _module_entry(function):
    """Give native module entry methods a traceable frame shared by their class."""
    @functools.wraps(function)
    def invoke(*args, **kwargs):
        return function(*args, **kwargs)

    # A single wrapper code object would put unrelated model classes into the
    # same small Dynamo recompile cache. Copies and instances of one class must
    # still share it, rather than spending one cache entry on each instance.
    model_type = type(function.__self__)
    with _torch_compile_lock:
        code = _module_entry_codes.get(model_type)
        if code is None:
            code = invoke.__code__.replace()
            _module_entry_codes[model_type] = code
    invoke.__code__ = code
    return invoke


class _Compilation:
    """Capture information is separate from parameters and is never a proof of execution."""

    def __init__(self, checker: Any):
        self._checker = checker
        self._report: dict[str, Any] | None = None
        self._artifact: dict[str, Any] | None = None
        self._graphs: list[dict[str, Any]] = []
        self._executors: dict[str, Any] = {}
        self._accepted: set[object] = set()
        self._module: weakref.ReferenceType | None = None
        self._entry = None

    def __deepcopy__(self, memo):
        # Capture copies model state, not compiler closures bound to the original model.
        result = type(self)(self._checker)
        memo[id(self)] = result
        # nn.Module.__getstate__ removes the native compiled entry before
        # deepcopy. Restore it on the already-memoized copied module, or its
        # otherwise normal calls would silently lose TorchLean capture.
        if self._module is not None and (module := self._module()) is not None:
            copied = memo.get(id(module))
            if copied is not None:
                result._bind_module(copied, self._entry)
        return result

    def _bind_module(self, module, entry):
        self._module = weakref.ref(module)
        self._entry = entry
        object.__setattr__(module, "_compiled_call_impl", types.MethodType(entry, module))

    @property
    def report(self) -> dict[str, Any] | None:
        """The latest check report, or None before the first compilation."""
        return copy.deepcopy(self._report)

    @property
    def artifact(self) -> dict[str, Any] | None:
        """The latest captured TorchLean IR; live parameters remain graph inputs."""
        return copy.deepcopy(self._artifact)

    @property
    def graphs(self) -> tuple[dict[str, Any], ...]:
        """Successfully checked graph specializations, in capture order."""
        return tuple(copy.deepcopy(self._graphs))

    def _check(self, artifact: dict[str, Any], frontend: str) -> None:
        from . import verify

        self._artifact = copy.deepcopy(artifact)
        self._report = None
        report = verify(artifact, checker=self._checker)
        report["execution"] = "pytorch" if frontend == "pytorch" else "jax_xla"
        report["autograd"] = "framework"
        self._report = report
        if not report["ok"]:
            details = "; ".join(f"{item['name']}: {item.get('message', item['status'])}"
                                for item in report["checks"] if item["status"] != "checked")
            raise RuntimeError(f"TorchLean compilation failed: {details}")
        self._graphs.append({"artifact": copy.deepcopy(artifact), "report": copy.deepcopy(report)})

    def _adopt(self, token: object, artifact: dict[str, Any]) -> None:
        if token not in self._accepted:
            self._check(artifact, "pytorch")
            self._accepted.add(token)

    def _torch(self, function):
        import torch
        from ._checker import Checker

        Checker(self._checker)
        # Some nn.Module entry methods live in Dynamo's skipped framework files.
        # A user-code entry keeps fullgraph capture active when calling them.
        if getattr(function, "__func__", None) in (
                torch.nn.Module._call_impl, torch.nn.Module.__call__):
            function = _module_entry(function)
        return torch.compile(function, backend=_torch_backend, fullgraph=True, dynamic=False)

    def _jax(self, function):
        import jax
        from jax.extend.core import jaxpr_as_fun
        from NN.Runtime.JAX.jaxpr import lower

        specializations = {}

        def execute(*args, **kwargs):
            leaves, structure = jax.tree.flatten((args, kwargs))
            dynamic = tuple(hasattr(value, "shape") and hasattr(value, "dtype")
                            for value in leaves)
            if not any(dynamic):
                raise NotImplementedError("TorchLean compilation needs a tensor argument")
            static = []
            for value, tensor in zip(leaves, dynamic):
                if tensor:
                    continue
                if type(value) not in (type(None), bool, int, float, str):
                    raise NotImplementedError(
                        "static arguments must be None, bool, int, float or str; "
                        "register structured JAX arguments as pytrees")
                static.append((type(value), value.hex() if type(value) is float else value))
            key = (structure, dynamic, tuple(static))
            if key not in specializations:
                constants = tuple(None if tensor else value
                                  for value, tensor in zip(leaves, dynamic))

                def checked(*inputs):
                    def original(*flat):
                        iterator = iter(flat)
                        values = [next(iterator) if tensor else value
                                  for value, tensor in zip(constants, dynamic)]
                        positional, keywords = jax.tree.unflatten(structure, values)
                        return function(*positional, **keywords)

                    closed, output = jax.make_jaxpr(original, return_shape=True)(*inputs)
                    artifact = lower(closed)
                    # This Jaxpr is executed as a single jit computation, not as
                    # the primitive-by-primitive reference used by verify().
                    artifact["reference_execution"] = "jit"
                    self._check(artifact, "jax")
                    values = jaxpr_as_fun(closed)(*inputs)
                    return jax.tree.unflatten(jax.tree.structure(output), values)

                specializations[key] = jax.jit(checked)
            tensors = (value for value, tensor in zip(leaves, dynamic) if tensor)
            return specializations[key](*tensors)

        return execute

    def __call__(self, function, args, kwargs):
        from . import _detect_frontend

        frontend = _detect_frontend(function, args, kwargs)
        if frontend not in self._executors:
            self._executors[frontend] = (
                self._torch(function) if frontend == "pytorch" else self._jax(function))
        if frontend == "jax":
            return self._executors[frontend](*args, **kwargs)
        invocation = []
        token = _execution.set(invocation)
        previous = getattr(_capture_context, "active", False)
        previous_owner = getattr(_capture_context, "compilation", None)
        _capture_context.active = True
        _capture_context.compilation = self
        try:
            with _torch_guard_scope():
                result = self._executors[frontend](*args, **kwargs)
            if not invocation:
                self._report = self._artifact = None
                raise NotImplementedError("TorchLean did not capture a tensor graph for this call")
            import torch
            leaves, _ = torch.utils._pytree.tree_flatten(result)
            if not leaves or not all(isinstance(value, torch.Tensor) for value in leaves):
                raise NotImplementedError("compiled functions must return tensor leaves")
            return result
        finally:
            _capture_context.active = previous
            _capture_context.compilation = previous_owner
            _execution.reset(token)


def _state(instance, checker):
    state = vars(instance).get("_torchlean_compilation")
    if state is None:
        state = _Compilation(checker)
        object.__setattr__(instance, "_torchlean_compilation", state)
    return state


def _tracing_torch():
    """An outer full-graph capture includes nested decorated callables itself."""
    if getattr(_capture_context, "exporting", False):
        return True
    torch = sys.modules.get("torch")
    if torch is None or not torch.compiler.is_compiling():
        return False
    if not getattr(_capture_context, "active", False):
        raise NotImplementedError(
            "an external torch.compile cannot bypass TorchLean capture; "
            "use torchlean.compile as the outer compiler")
    return True


def compile(function=None, *, checker=None):
    """Lower lazily captured framework graphs to TorchLean IR and check their structure/types.

    Use ``@torchlean.compile`` on a function or a PyTorch module class, or call
    ``torchlean.compile(module)``. Normal calls select the frontend automatically.
    PyTorch's compiler guards specialize shapes, types and Python configuration;
    JAX uses its usual pure-function jit semantics, with Python scalar arguments
    specialized and tensor pytrees kept dynamic. Pass changing JAX parameters as
    arguments, as with ordinary ``jax.jit``.

    ``.torchlean.report`` and ``.torchlean.artifact`` expose the latest capture;
    ``.torchlean.graphs`` retains checked specializations. They are copies, so
    inspection cannot change the compiler's stored check result. Execution and
    gradients remain in PyTorch or JAX. The checks do not prove frontend
    equivalence, numerical accuracy, or correctness for all inputs. Unsupported
    graphs raise rather than silently running an unchecked fallback.

    The checker is located on the first call; decorating never starts a build.
    ``checker`` has the same command convention as ``verify``.
    """
    if function is None:
        return functools.partial(compile, checker=checker)
    if not callable(function):
        raise TypeError("torchlean.compile expects a callable or model class")
    if inspect.isclass(function):
        # Capture the complete module call, including hooks and an overridden
        # __call__, rather than checking only the forward portion of the result.
        name = "__call__"
        original = next((base.__dict__[name] for base in function.__mro__
                         if name in base.__dict__), None)
        if getattr(original, "__torchlean_original__", None) is not None:
            return function
        if hasattr(function, "torchlean"):
            raise ValueError("the model class already has a torchlean attribute")
        if original is None or not callable(original):
            raise TypeError("a compiled model class must define a callable __call__")

        @functools.wraps(original)
        def method(self, *args, **kwargs):
            if _tracing_torch():
                return original(self, *args, **kwargs)
            return _state(self, checker)(original.__get__(self, type(self)), args, kwargs)

        method.__torchlean_original__ = original
        setattr(function, name, method)
        setattr(function, "torchlean", property(lambda self: _state(self, checker)))
        return function

    if hasattr(function, "torchlean"):
        if isinstance(function.torchlean, _Compilation):
            return function
        raise ValueError("the callable already has a torchlean attribute")

    is_torch = any(base.__module__.split(".")[0] == "torch"
                   for base in type(function).__mro__) and hasattr(function, "forward")
    state = _Compilation(checker)
    if is_torch:
        import torch

        if type(function).__call__ is not torch.nn.Module.__call__:
            raise NotImplementedError(
                "compile a module with an overridden __call__ by decorating its class")
        # This is the same entry used by nn.Module.compile. Binding the entry
        # to the module lets deepcopy rebind it to the copied parameter state.
        original = function._call_impl
        unbound = getattr(original, "__func__", None)
        fallback = original if unbound is None else None

        @functools.wraps(unbound if unbound is not None else original)
        def call(self, *args, **kwargs):
            target = unbound.__get__(self, type(self)) if unbound is not None else fallback
            if _tracing_torch():
                return target(*args, **kwargs)
            return self.torchlean(target, args, kwargs)

        call.__torchlean_original__ = unbound if unbound is not None else original
        state._bind_module(function, call)
        function.torchlean = state
        return function

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if _tracing_torch():
            return function(*args, **kwargs)
        return state(function, args, kwargs)

    wrapped.__torchlean_original__ = function
    wrapped.torchlean = state
    return wrapped
