"""One binder for the keyword arguments of a PyTorch graph's calls.

PyTorch's argument parser accepts NumPy's names for four parameters: `axis` for `dim`, `keepdims`
for `keepdim`, `x`, `a` and `x1` for `input`, and `x2` for `other`. It accepts an alias only for an
overload that has the canonical parameter and no parameter of the alias's name. So
`torch.sum(x, axis=1, keepdims=True)` is `torch.sum(x, dim=1, keepdim=True)`, while
`torch.flip(x, axis=0)` is an error, since flip's parameter is `dims`.

`canonical` renames the keywords of one call by that rule. It binds the call to each overload's
signature, with an integer-list vararg such as `x.permute(1, 0)` packed into one list as the parser
packs it, and keeps the renaming of the overloads that bind. It raises `NotImplementedError`, naming
the call and its arguments, when the call gives both spellings of one parameter or binds to no
overload. `bind_arguments(graph)` applies it to every call of an FX graph, once, before any lowering
reads an argument, so the lowerings read canonical keywords only. Positional arguments stay where
they are, and a lowering reads each one at its position in the signature.

The signatures are the parser's: the TorchScript schemas of a torch.* builtin, and the ATen schemas
of a Tensor method. A Python function, such as `torch.nn.functional.softmax`, takes its own
parameter names only, as in eager PyTorch. A call to an ATen operator (as torch.export writes them),
an `operator` function or a module has no aliases, and keeps its arguments.
"""

from __future__ import annotations

import inspect

import torch
from torch.fx.operator_schemas import _torchscript_schema_to_signature, get_signature_for_torch_op

# The NumPy spellings PyTorch's argument parser accepts, with the parameter each one names.
ALIASES = {'axis': 'dim', 'keepdims': 'keepdim', 'x': 'input', 'a': 'input', 'x1': 'input',
           'x2': 'other'}


def _method(op, target):
    # The Tensor method name of a call, or None: a `call_method`, or a call of an unbound method.
    if op == 'call_method':
        return target
    name = getattr(target, '__name__', None)
    if op == 'call_function' and name and getattr(torch.Tensor, name, None) is target:
        return name
    return None


def _signatures(op, target):
    # The overload signatures of a call, and whether the parser's aliases apply to them.
    if op not in ('call_function', 'call_method'):
        return [], False
    if type(target).__module__ == 'torch._ops':
        # An ATen operator or a higher-order operator: bound to its schema already.
        return [], False
    name = _method(op, target)
    if name is not None:
        method = getattr(torch.Tensor, name, None)
        if inspect.isfunction(method):
            return [inspect.signature(method)], False
        try:
            schemas = torch._C._jit_get_schemas_for_operator(f'aten::{name}')
        except RuntimeError:
            schemas = []
        return [_torchscript_schema_to_signature(schema) for schema in schemas], True
    if inspect.isfunction(target):
        return [inspect.signature(target)], False
    if not (getattr(target, '__module__', None) or '').startswith('torch'):
        return [], False
    return list(get_signature_for_torch_op(target) or []), True


def _packed(method, signature, args):
    # The positional arguments as the parser reads them: when the overload's one positional
    # parameter (after the receiver of a method) is an integer list, several integers are one list.
    first = 1 if method else 0
    positional = [parameter for parameter in signature.parameters.values()
                  if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    if (len(positional) == first + 1 and 'List[int]' in str(positional[first].annotation)
            and len(args) > first + 1):
        return (*args[:first], tuple(args[first:]))
    return tuple(args)


def _call(op, target):
    name = _method(op, target)
    if name is not None:
        return f'Tensor.{name}'
    module = getattr(target, '__module__', None)
    return f'{module}.{target.__name__}' if module and hasattr(target, '__name__') else str(target)


def canonical(op, target, args, kwargs):
    """The keyword arguments of the call `op` of `target` at `args` and `kwargs`, each under the
    name of the parameter it binds, in their order.

    An alias is renamed for an overload that has its canonical parameter and no parameter of its
    own name. A call without a known signature keeps its keywords. `NotImplementedError` names the
    call when it gives one parameter twice, binds to no overload, or binds to overloads that name
    its keywords differently.
    """
    kwargs = dict(kwargs)
    signatures, aliases = _signatures(op, target)
    if not signatures:
        return kwargs
    method = _method(op, target) is not None
    names = []
    for signature in signatures:
        parameters = signature.parameters
        renamed = {}
        for key, value in kwargs.items():
            name = ALIASES.get(key)
            if not (aliases and name in parameters and key not in parameters):
                name = key
            if name in renamed:
                spelled = [k for k in kwargs if k == name or ALIASES.get(k) == name]
                raise NotImplementedError(
                    f'cannot bind the arguments of {_call(op, target)}: the parameter {name!r} '
                    f'is given as {" and ".join(map(repr, spelled))}')
            renamed[name] = value
        try:
            signature.bind(*(_packed(method, signature, args) if aliases else args), **renamed)
        except TypeError:
            continue
        if list(renamed) not in names:
            names.append(list(renamed))
    if len(names) != 1:
        reason = 'no overload binds them' if not names else 'its overloads name them differently'
        raise NotImplementedError(
            f'cannot bind the arguments of {_call(op, target)}: {len(args)} positional arguments '
            f'and the keywords {list(kwargs)}, and {reason}')
    return dict(zip(names[0], kwargs.values()))


def bind_arguments(graph):
    """Rename the keywords of every call of the FX graph `graph` in place, as `canonical` does."""
    for node in graph.nodes:
        if node.op in ('call_function', 'call_method') and node.kwargs:
            kwargs = canonical(node.op, node.target, node.args, node.kwargs)
            if list(kwargs) != list(node.kwargs):
                node.kwargs = kwargs
