"""The pointwise operators of a PyTorch graph as capture nodes.

`_pointwise_plan` names a call's operator, attributes and operands, Python scalars included.
`_emit_pointwise` appends the nodes PyTorch's CPU kernels compute: conversions to the common dtype
and, for binary16 and bfloat16, to binary32 (the opmath type), broadcasts and scalar constants.
The modes of gelu and div are the keyword values of the op table's entries (`_keyword`).
"""

from __future__ import annotations

import fractions
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._fx import (_FLOATING, _INDEX_DTYPES, _REDUCED_FLOATS, _broadcast_kind, _constant, _dtype_of,
                  _is_op, _node_dtype, _op_table, _torch_dtype)


_ARITHMETIC_KINDS = {'add': "add", 'sub': "sub", 'mul': "mul_elem"}

# Unary operators PyTorch's CPU kernels compute in the opmath type: a binary16 or bfloat16 operand
# in binary32, rounded once to the result dtype. An integer operand is converted to the result's
# floating dtype first, as PyTorch promotes it.
_OPMATH_UNARY = {
    'expm1': (('aten.expm1.default',), (torch.expm1, torch.special.expm1), ('expm1',)),
    'log1p': (('aten.log1p.default',), (torch.log1p, torch.special.log1p), ('log1p',)),
    'exp2': (('aten.exp2.default',), (torch.exp2, torch.special.exp2), ('exp2',)),
    'log2': (('aten.log2.default',), (torch.log2,), ('log2',)),
    'log10': (('aten.log10.default',), (torch.log10,), ('log10',)),
    'tan': (('aten.tan.default',), (torch.tan,), ('tan',)),
    'sinh': (('aten.sinh.default',), (torch.sinh,), ('sinh',)),
    'cosh': (('aten.cosh.default',), (torch.cosh,), ('cosh',)),
    'asinh': (('aten.asinh.default',), (torch.asinh, torch.arcsinh), ('asinh', 'arcsinh')),
    'acosh': (('aten.acosh.default',), (torch.acosh, torch.arccosh), ('acosh', 'arccosh')),
    'atanh': (('aten.atanh.default',), (torch.atanh, torch.arctanh), ('atanh', 'arctanh')),
    'asin': (('aten.asin.default',), (torch.asin, torch.arcsin), ('asin', 'arcsin')),
    'acos': (('aten.acos.default',), (torch.acos, torch.arccos), ('acos', 'arccos')),
    'atan': (('aten.atan.default',), (torch.atan, torch.arctan), ('atan', 'arctan')),
}

# Unary operators whose result is exact in the operand's floating format, computed there.
_EXACT_UNARY = {
    'floor': (('aten.floor.default',), (torch.floor,), ('floor',)),
    'ceil': (('aten.ceil.default',), (torch.ceil,), ('ceil',)),
    'round': (('aten.round.default', 'aten.round.decimals'), (torch.round,), ('round',)),
    'trunc': (('aten.trunc.default', 'aten.fix.default'), (torch.trunc, torch.fix),
              ('trunc', 'fix')),
    'frac': (('aten.frac.default',), (torch.frac,), ('frac',)),
    'sign': (('aten.sign.default',), (torch.sign,), ('sign',)),
}

# Binary operators computed in the opmath type and rounded once, after both operands are converted
# to the common dtype.
_OPMATH_BINARY = {
    'hypot': (('aten.hypot.default',), (torch.hypot,), ('hypot',)),
    'atan2': (('aten.atan2.default',), (torch.atan2, torch.arctan2), ('atan2', 'arctan2')),
    'remainder': (('aten.remainder.Tensor', 'aten.remainder.Scalar',
                   'aten.remainder.Scalar_Tensor'), (torch.remainder, operator.mod),
                  ('remainder',)),
}

# Comparisons and class predicates, computed exactly on floating operands of one dtype, and the
# logical operators, also on Boolean operands. Each result is Boolean.
_COMPARISONS = ('eq', 'ne', 'lt', 'le', 'gt', 'ge')
_PREDICATES = ('isnan', 'isinf', 'isfinite')
_LOGICAL = {'and': operator.and_, 'or': operator.or_, 'xor': operator.xor}

# Binary operators whose result is exact in the common floating format, computed there.
_EXACT_BINARY = {
    'copysign': (('aten.copysign.Tensor', 'aten.copysign.Scalar'), (torch.copysign,),
                 ('copysign',)),
    'fmod': (('aten.fmod.Tensor', 'aten.fmod.Scalar'), (torch.fmod,), ('fmod',)),
    'fmax': (('aten.fmax.default',), (torch.fmax,), ('fmax',)),
    'fmin': (('aten.fmin.default',), (torch.fmin,), ('fmin',)),
    'nextafter': (('aten.nextafter.default',), (torch.nextafter,), ('nextafter',)),
}

# The shifts of integer operands. On integers, and, or, xor and not are bitwise operators too.
_SHIFTS = {
    'bitwise_left_shift': (('aten.bitwise_left_shift.Tensor',
                            'aten.bitwise_left_shift.Tensor_Scalar',
                            'aten.bitwise_left_shift.Scalar_Tensor', 'aten.__lshift__.Tensor',
                            'aten.__lshift__.Scalar'),
                           (torch.bitwise_left_shift, operator.lshift),
                           ('bitwise_left_shift', '__lshift__')),
    'bitwise_right_shift': (('aten.bitwise_right_shift.Tensor',
                             'aten.bitwise_right_shift.Tensor_Scalar',
                             'aten.bitwise_right_shift.Scalar_Tensor', 'aten.__rshift__.Tensor',
                             'aten.__rshift__.Scalar'),
                            (torch.bitwise_right_shift, operator.rshift),
                            ('bitwise_right_shift', '__rshift__')),
}
_BITWISE = ('bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not', *_SHIFTS)


def _argument(node, position, *names):
    if len(node.args) > position:
        return node.args[position]
    for name in names:
        if name in node.kwargs:
            return node.kwargs[name]
    return None


def _reflected_division(node):
    # torch.export traces Python's s / x through Tensor.__rtruediv__, which PyTorch writes as
    # reciprocal(x) * s, and tags both nodes with that one call in meta['torch_fn']. The pair is
    # the true division s / x: return (s, x) at its product, None elsewhere. A user's own
    # x.reciprocal() * s carries its own tags and stays a reciprocal and a product, and a graph
    # without the tags is lowered as written.
    if node.op != 'call_function' or not _is_op(node, ('aten.mul.Tensor',)):
        return None
    call = node.meta.get('torch_fn')
    if not call or str(call[1]).rsplit('.', 1)[-1] not in ('__rdiv__', '__rtruediv__'):
        return None
    reciprocal, scalar = (list(node.args) + [None, None])[:2]
    if not isinstance(reciprocal, torch.fx.Node) or not isinstance(scalar, (bool, int, float)):
        return None
    if (not _is_op(reciprocal, ('aten.reciprocal.default',))
            or reciprocal.meta.get('torch_fn') != call or len(reciprocal.users) != 1):
        return None
    return scalar, reciprocal.args[0]


def _pointwise_plan(node, model=None):
    # The pointwise operator a call computes, with its attributes and operands, or None.
    reflected = _reflected_division(node)
    if reflected is not None:
        return {'op': 'div', 'rounding_mode': None, 'operands': list(reflected)}
    if node.op == 'call_module':
        mod = model.get_submodule(str(node.target)) if model is not None else None
        if isinstance(mod, nn.GELU):
            return {'op': 'gelu', 'approximate': mod.approximate, 'operands': [node.args[0]]}
        if isinstance(mod, (nn.ELU, nn.Hardtanh)) and mod.inplace:
            raise NotImplementedError(f'in-place {type(mod).__name__} mutation is outside the '
                                      'pure TorchLean graph contract')
        if isinstance(mod, nn.ELU):
            return {'op': 'elu', 'alpha': float(mod.alpha), 'scale': 1.0, 'input_scale': 1.0,
                    'operands': [node.args[0]]}
        if isinstance(mod, nn.Softplus):
            return {'op': 'softplus', 'beta': float(mod.beta), 'threshold': float(mod.threshold),
                    'operands': [node.args[0]]}
        if isinstance(mod, nn.Hardtanh):
            return {'op': 'clamp', 'operands': [node.args[0], mod.min_val, mod.max_val]}
        return None
    first = _argument(node, 0, 'input', 'self')
    second = _argument(node, 1, 'other')
    if _is_op(node, ('aten.neg.default',), (torch.neg, torch.negative, operator.neg),
              ('neg', 'negative')):
        return {'op': 'neg', 'operands': [first]}
    if _is_op(node, ('aten.rsqrt.default',), (torch.rsqrt,), ('rsqrt',)):
        return {'op': 'rsqrt', 'operands': [first]}
    if _is_op(node, ('aten.erf.default',), (torch.erf, torch.special.erf), ('erf',)):
        return {'op': 'erf', 'operands': [first]}
    if _is_op(node, ('aten.gelu.default',), (F.gelu,), ()):
        return {'op': 'gelu', 'approximate': node.kwargs.get('approximate', 'none'),
                'operands': [first]}
    if _is_op(node, ('aten.div.Tensor', 'aten.div.Scalar', 'aten.div.Tensor_mode',
                     'aten.div.Scalar_mode'),
              (torch.div, torch.divide, torch.true_divide, operator.truediv),
              ('div', 'divide', 'true_divide')):
        # FX records Python's s / x as operator.truediv(s, x): true division, though PyTorch's
        # Tensor.__rtruediv__ computes reciprocal(x) * s.
        return {'op': 'div', 'rounding_mode': node.kwargs.get('rounding_mode'),
                'operands': [first, second]}
    if _is_op(node, ('aten.floor_divide.default', 'aten.floor_divide.Scalar'),
              (torch.floor_divide, operator.floordiv),
              ('floor_divide',)):
        return {'op': 'div', 'rounding_mode': 'floor', 'operands': [first, second]}
    if _is_op(node, ('aten.pow.Tensor_Tensor', 'aten.pow.Tensor_Scalar', 'aten.pow.Scalar'),
              (torch.pow, operator.pow), ('pow',)):
        return {'op': 'pow', 'operands': [first, _argument(node, 1, 'exponent')]}
    if _is_op(node, ('aten.clamp_min.default', 'aten.clamp_min.Tensor'), (torch.clamp_min,),
              ('clamp_min',)):
        return {'op': 'clamp_min', 'operands': [first, _argument(node, 1, 'min')]}
    if _is_op(node, ('aten.clamp_max.default', 'aten.clamp_max.Tensor'), (torch.clamp_max,),
              ('clamp_max',)):
        return {'op': 'clamp_max', 'operands': [first, _argument(node, 1, 'max')]}
    for op, (aten, functions, methods) in (*_OPMATH_UNARY.items(), *_EXACT_UNARY.items()):
        if _is_op(node, aten, functions, methods):
            if op == 'round' and (_argument(node, 1, 'decimals') or 0) != 0:
                raise NotImplementedError('round with decimals scales by a power of ten, which '
                                          'is not one pointwise operator')
            return {'op': op, 'operands': [first]}
    for op, (aten, functions, methods) in (*_OPMATH_BINARY.items(), *_EXACT_BINARY.items()):
        if _is_op(node, aten, functions, methods):
            return {'op': op, 'operands': [first, second]}
    for op in _COMPARISONS:
        if _is_op(node, (f'aten.{op}.Tensor', f'aten.{op}.Scalar'),
                  (getattr(torch, op), getattr(operator, op)), (op,)):
            return {'op': 'boolean', 'kind': op, 'operands': [first, second]}
    for op in _PREDICATES:
        if _is_op(node, (f'aten.{op}.default',), (getattr(torch, op),), (op,)):
            return {'op': 'boolean', 'kind': op, 'operands': [first]}
    for op, python in _LOGICAL.items():
        if _is_op(node, (f'aten.logical_{op}.default',), (getattr(torch, f'logical_{op}'),),
                  (f'logical_{op}',)):
            return {'op': 'boolean', 'kind': f'logical_{op}', 'operands': [first, second]}
        if _is_op(node, (f'aten.bitwise_{op}.Tensor', f'aten.bitwise_{op}.Scalar',
                         f'aten.__{op}__.Tensor', f'aten.__{op}__.Scalar'),
                  (getattr(torch, f'bitwise_{op}'), python), (f'bitwise_{op}', f'__{op}__')):
            if isinstance(_node_dtype(node), dict):
                return {'op': f'bitwise_{op}', 'operands': [first, second]}
            return {'op': 'boolean', 'kind': f'logical_{op}', 'bitwise': True,
                    'operands': [first, second]}
    if _is_op(node, ('aten.logical_not.default',), (torch.logical_not,), ('logical_not',)):
        return {'op': 'boolean', 'kind': 'logical_not', 'operands': [first]}
    if _is_op(node, ('aten.bitwise_not.default',), (torch.bitwise_not, operator.invert),
              ('bitwise_not',)):
        if isinstance(_node_dtype(node), dict):
            return {'op': 'bitwise_not', 'operands': [first]}
        return {'op': 'boolean', 'kind': 'logical_not', 'bitwise': True, 'operands': [first]}
    for op, (aten, functions, methods) in _SHIFTS.items():
        if _is_op(node, aten, functions, methods):
            return {'op': op, 'operands': [first, second]}
    if _is_op(node, ('aten.where.self', 'aten.where.ScalarSelf', 'aten.where.ScalarOther',
                     'aten.where.Scalar'), (torch.where,), ()):
        return {'op': 'where', 'operands': [_argument(node, 0, 'condition'),
                                            _argument(node, 1, 'input', 'self'),
                                            _argument(node, 2, 'other')]}
    if _is_op(node, (), (), ('where',)):
        # Tensor.where(condition, other) selects self where the condition holds.
        return {'op': 'where', 'operands': [_argument(node, 1, 'condition'), first,
                                            _argument(node, 2, 'other')]}
    if _is_op(node, ('aten.masked_fill.Scalar', 'aten.masked_fill.Tensor'), (),
              ('masked_fill',)):
        # The value where the mask holds, converted to the result dtype as PyTorch does.
        return {'op': 'where', 'operands': [_argument(node, 1, 'mask'), _argument(node, 2, 'value'),
                                            first]}
    if _is_op(node, ('aten.clamp.default', 'aten.clamp.Tensor', 'aten.clip.default',
                     'aten.clip.Tensor', 'aten.hardtanh.default'),
              (torch.clamp, torch.clip, F.hardtanh), ('clamp', 'clip')):
        hardtanh = _is_op(node, ('aten.hardtanh.default',), (F.hardtanh,), ())
        if node.target is F.hardtanh and _argument(node, 3, 'inplace'):
            raise NotImplementedError('in-place hardtanh mutation is outside the pure TorchLean '
                                      'graph contract')
        names = ('min_val', 'max_val') if hardtanh else ('min', 'max')
        low, high = _argument(node, 1, names[0]), _argument(node, 2, names[1])
        if hardtanh:
            low = -1.0 if low is None else low
            high = 1.0 if high is None else high
        if low is None and high is None:
            raise NotImplementedError('clamp needs a bound')
        if high is None:
            # clamp with only a lower bound is clamp_min, with only an upper one clamp_max.
            return {'op': 'clamp_min', 'operands': [first, low]}
        if low is None:
            return {'op': 'clamp_max', 'operands': [first, high]}
        return {'op': 'clamp', 'operands': [first, low, high]}
    if _is_op(node, ('aten.elu.default',), (F.elu,), ()):
        # F.elu(input, alpha, inplace) and aten.elu(self, alpha, scale, input_scale).
        if node.target is F.elu:
            if _argument(node, 2, 'inplace'):
                raise NotImplementedError('in-place elu mutation is outside the pure TorchLean '
                                          'graph contract')
            names = ((1, 'alpha'),)
        else:
            names = ((1, 'alpha'), (2, 'scale'), (3, 'input_scale'))
        coefficients = [_argument(node, position, name) for position, name in names]
        alpha, scale, input_scale = [1.0 if value is None else float(value) for value in
                                     coefficients + [None] * (3 - len(coefficients))]
        return {'op': 'elu', 'alpha': alpha, 'scale': scale, 'input_scale': input_scale,
                'operands': [first]}
    if _is_op(node, ('aten.softplus.default',), (F.softplus,), ()):
        beta, threshold = _argument(node, 1, 'beta'), _argument(node, 2, 'threshold')
        return {'op': 'softplus', 'beta': 1.0 if beta is None else float(beta),
                'threshold': 20.0 if threshold is None else float(threshold),
                'operands': [first]}
    return None


def _boolean_operands(plan):
    # The dtype a comparison, class predicate or logical operator reads its operands in: the
    # tensors' one floating or integer dtype promoted with the Python scalars as PyTorch's
    # result_type does, or bool for the logical operators of Boolean tensors. An integer tensor
    # and a float scalar compare in the default float dtype. The CPU reference compares
    # integers by their values. It has no Boolean comparisons and no bitwise operators.
    kind, operands = plan['kind'], plan['operands']
    dtypes = [_node_dtype(value) for value in operands if isinstance(value, torch.fx.Node)]
    if any(dtype != dtypes[0] for dtype in dtypes):
        raise NotImplementedError(f'{kind} of operands of two dtypes is not imported')
    dtype = dtypes[0]
    if dtype == 'bool' and kind.startswith('logical') and len(dtypes) == len(operands):
        return dtype
    name = dtype if isinstance(dtype, str) else str(_torch_dtype(dtype)).removeprefix('torch.')
    if plan.get('bitwise'):
        raise NotImplementedError(f'a bitwise operator of {name} operands has no IR element rule')
    if dtype in _FLOATING:
        return dtype
    if not isinstance(dtype, dict) or kind in _PREDICATES:
        raise NotImplementedError(f'{kind} of {name} operands has no element rule in the CPU '
                                  'reference')
    common = _torch_dtype(dtype)
    for value in operands:
        if not isinstance(value, torch.fx.Node):
            common = torch.result_type(torch.empty(1, dtype=common), value)
    return _dtype_of(torch.empty(0, dtype=common))


def _keyword(name):
    # The keyword argument that selects among the op table's entries for the pointwise operator
    # `name`, and the values it has entries for, None for the entry without the keyword.
    entries = _op_table()['pointwise'][name]
    key = next(entry['keyword'][0] for entry in entries if entry['keyword'] is not None)
    return key, [None if entry['keyword'] is None else entry['keyword'][1] for entry in entries]


def _pointwise_kind(node, model=None):
    plan = _pointwise_plan(node, model)
    if plan is None:
        return None
    operands = plan['operands']
    if not any(isinstance(value, torch.fx.Node) for value in operands):
        raise NotImplementedError('a pointwise operation needs a tensor operand')
    for value in operands:
        if not isinstance(value, (torch.fx.Node, bool, int, float)):
            raise NotImplementedError(f'unsupported pointwise operand {value!r}')
    if plan['op'] == 'gelu' and plan['approximate'] not in _keyword('gelu')[1]:
        raise NotImplementedError(f"gelu has no approximate={plan['approximate']!r} mode")
    if plan['op'] == 'div' and plan['rounding_mode'] not in _keyword('div')[1]:
        raise NotImplementedError(f"div has no rounding_mode={plan['rounding_mode']!r}")
    dtype = _node_dtype(node)
    if ((plan['op'] in _EXACT_UNARY or plan['op'] in _EXACT_BINARY
         or plan['op'] in ('clamp', 'elu', 'softplus'))
            and dtype not in ('float64', 'float32', *_REDUCED_FLOATS)):
        raise NotImplementedError(f"{plan['op']} of a {dtype} tensor has no IR element rule")
    if plan['op'] == 'remainder' and dtype not in _FLOATING and dtype not in _INDEX_DTYPES:
        raise NotImplementedError(f"{plan['op']} of a {dtype} tensor has no IR element rule")
    if plan['op'] in _BITWISE and not isinstance(dtype, dict):
        raise NotImplementedError(f"{plan['op']} of a {dtype} tensor has no IR element rule")
    if plan['op'] == 'boolean':
        plan['common'] = _boolean_operands(plan)
    if plan['op'] == 'where':
        condition = operands[0]
        if not isinstance(condition, torch.fx.Node) or _node_dtype(condition) != 'bool':
            raise NotImplementedError('where and masked_fill need a Boolean condition')
        if dtype not in _FLOATING and dtype != 'bool' and not isinstance(dtype, dict):
            raise NotImplementedError(f'where of {dtype} values is not imported')
    if plan['op'] == 'pow' and not isinstance(operands[1], torch.fx.Node):
        exponent = operands[1]
        if isinstance(dtype, dict) and exponent < 0:
            raise NotImplementedError('PyTorch rejects negative integer powers of integers')
        if not isinstance(dtype, dict):
            # The exponent is kept at its value in the computation type (_emit_pointwise). An
            # exponent whose rounding there changes whether it is an integer, or which integer,
            # would change the power's sign or domain, so it is refused.
            opmath = 'float32' if dtype in _REDUCED_FLOATS else dtype
            rounded = _scalar_tensor(exponent, opmath).item()
            if rounded != exponent and (float(rounded).is_integer()
                                        or float(exponent).is_integer()):
                raise NotImplementedError(f'pow exponent {exponent!r} is {rounded!r} in '
                                          f'{opmath}, another integer or an integer from a '
                                          'fraction')
    return {'kind': plan['op'], '_plan': plan}


def _append_node(nodes, entry):
    entry['id'] = len(nodes)
    nodes.append(entry)
    return entry['id']


def _cast_to(nodes, parent, dtype):
    # A conversion is one IR cast. To a floating type it rounds the exact value once, nearest even
    # (IEEE 754-2019 5.4.1, 5.4.2), also from an integer, Boolean or binary64 source to binary16 or
    # bfloat16, which c10::convert rounds twice, through binary32. To an integer type a floating
    # value truncates toward zero, and an integer the type cannot hold is an execution error. An
    # integer value wraps modulo 2 ** width.
    if nodes[parent]['dtype'] == dtype:
        return parent
    rounding = 'towardZero' if isinstance(dtype, dict) else 'nearestEven'
    return _append_node(nodes, {'kind': "custom", 'parents': [parent],
                                'shape': nodes[parent]['shape'], 'dtype': dtype,
                                'primitive': 'cast', 'rounding': rounding})


def _broadcast_to(nodes, parent, shape):
    source = nodes[parent]['shape']
    if source == shape:
        return parent
    return _append_node(nodes, {**_broadcast_kind(source, shape), 'parents': [parent],
                                'shape': shape, 'dtype': nodes[parent]['dtype']})


def _round_once(value, dtype):
    # The binary16 or bfloat16 value nearest the exact Python int or float `value`, ties to even,
    # an infinity from the overflow threshold on, a zero keeping its sign. This is IEEE 754-2019
    # rounding to the format, returned as a Python float, which holds the result exactly.
    if isinstance(value, float) and (value != value or value in (float('inf'), float('-inf'))
                                     or value == 0):
        return value
    precision, emin, emax = (11, -14, 15) if dtype == 'float16' else (8, -126, 127)
    exact = abs(fractions.Fraction(value))
    if exact == 0:
        return 0.0
    exponent = exact.numerator.bit_length() - exact.denominator.bit_length()
    if fractions.Fraction(2) ** exponent > exact:
        exponent -= 1
    quantum = fractions.Fraction(2) ** (max(exponent, emin) - precision + 1)
    rounded = round(exact / quantum) * quantum
    magnitude = float('inf') if rounded >= 2 ** (emax + 1) else float(rounded)
    return -magnitude if value < 0 else magnitude


def _scalar_tensor(value, dtype):
    # A Python number converted once to `dtype`. PyTorch wraps a bool as a Boolean, an int as
    # int64 (uint64 from 2 ** 63) and a float as binary64, and that value is rounded once. An
    # int beyond 2 ** 53 rounds once to binary32, not through binary64. Binary16 and bfloat16
    # receive the nearest value of their format (_round_once), which c10::convert misses by
    # rounding through binary32.
    if isinstance(value, bool):
        wrapped = torch.tensor(value, dtype=torch.bool)
    elif isinstance(value, int):
        if not -2 ** 63 <= value < 2 ** 64:
            raise NotImplementedError(f'PyTorch cannot wrap the Python int {value} as a scalar')
        wrapped = torch.tensor(value, dtype=torch.int64 if value < 2 ** 63 else torch.uint64)
    else:
        wrapped = torch.tensor(value, dtype=torch.float64)
    if dtype in _REDUCED_FLOATS:
        wrapped = torch.tensor(_round_once(value, dtype), dtype=torch.float64)
    return wrapped.to(_torch_dtype(dtype))


def _scalar_constant(nodes, value, dtype):
    return _append_node(nodes, {'parents': [], **_constant(_scalar_tensor(value, dtype))})


def _one_element(value, dtype):
    # Whether TensorIterator's is_scalar(2) holds for this right operand when the common dtype
    # is `dtype`: a Python scalar, a one-element tensor, or a view with stride 0 along every
    # dimension longer than one. Conversion to the common dtype copies a longer operand to
    # contiguous storage, so such a view qualifies only when it already has that dtype.
    if not isinstance(value, torch.fx.Node):
        return True
    meta = value.meta.get('val', value.meta.get('tensor_meta'))
    if not hasattr(meta, 'shape'):
        raise NotImplementedError('a binary operation needs the shapes of its operands')
    sizes = [int(size) for size in meta.shape]
    if all(size == 1 for size in sizes):
        return True
    strides = meta.stride() if callable(getattr(meta, 'stride', None)) else meta.stride
    return _dtype_of(meta) == dtype and all(
        size == 1 or int(step) == 0 for size, step in zip(sizes, strides))


def _arithmetic_kind(node, op):
    # `add`, `sub` or `mul`. With two tensor operands this is the ordinary IR node. A Python
    # scalar operand becomes a constant: add and sub convert it to the result dtype. A binary16
    # or bfloat16 product with an operand that supplies one element and is not of the result
    # dtype (a Python scalar, or a one-element tensor of another dtype) multiplies by that
    # operand's own value in binary32 and rounds once. It does so on either side, so
    # torch.mul(s, x) is x * s. ATen's mul_kernel takes that path for the right operand only and
    # converts a left one to the tensor's dtype first (#199562).
    left, right = _argument(node, 0, 'input', 'self'), _argument(node, 1, 'other')
    scalars = [not isinstance(value, torch.fx.Node) for value in (left, right)]
    for value, scalar in zip((left, right), scalars):
        if scalar and not isinstance(value, (bool, int, float)):
            raise NotImplementedError(f'unsupported {op} operand {value!r}')
    if all(scalars):
        raise NotImplementedError(f'{op} needs a tensor operand')
    if _node_dtype(node) == 'bool' and op in ('add', 'mul') and not any(scalars):
        # PyTorch's add and mul of Boolean tensors are logical or and and.
        kind = 'logical_or' if op == 'add' else 'logical_and'
        return {'kind': kind, '_plan': {'op': 'boolean', 'kind': kind, 'common': 'bool',
                                        'operands': [left, right]}}
    dtype = _node_dtype(node)
    def single(value):
        return _one_element(value, dtype) and (
            not isinstance(value, torch.fx.Node) or _node_dtype(value) != dtype)
    reduced = op == 'mul' and dtype in _REDUCED_FLOATS
    if op == 'mul' and (scalars[0] and not scalars[1]
                        or reduced and single(left) and not single(right)):
        left, right = right, left
    wide = reduced and single(right)
    if not any(scalars) and not wide:
        return {'kind': _ARITHMETIC_KINDS[op]}
    return {'kind': _ARITHMETIC_KINDS[op],
            '_plan': {'op': op, 'wide': wide, 'operands': [left, right]}}


def _pointwise_node(nodes, kind, parents):
    first = nodes[parents[0]]
    return _append_node(nodes, {**kind, 'parents': parents, 'shape': first['shape'],
                                'dtype': first['dtype']})


def _emit_pointwise(plan, nodes, node_to_id, shape, dtype):
    # Append the IR nodes of one pointwise call, conversion, or arithmetic with a Python scalar,
    # and return the last. Binary16 and bfloat16 compute rsqrt, erf, gelu, pow, _OPMATH_UNARY and
    # _OPMATH_BINARY in binary32 (PyTorch's opmath type) and round once to the dtype, and division
    # and multiplication too when an operand supplies one element.
    op, operands = plan['op'], plan['operands']
    reduced = dtype in _REDUCED_FLOATS
    opmath = 'float32' if reduced else dtype
    def operand(value, *dtypes):
        # A tensor converted to each dtype in turn, or a Python scalar made at the first one.
        if isinstance(value, torch.fx.Node):
            parent = node_to_id.get(value)
            if parent is None:
                raise NotImplementedError('pointwise operand has no tensor lowering')
        else:
            parent = _scalar_constant(nodes, value, dtypes[0])
        for target in dtypes:
            parent = _cast_to(nodes, parent, target)
        return _broadcast_to(nodes, parent, shape)
    def apply(name, parents, **attributes):
        kind = {'kind': name}
        kind.update({key: value for key, value in attributes.items() if value is not None})
        return _pointwise_node(nodes, kind, parents)
    def other(kind, parents):
        return _pointwise_node(nodes, {'kind': kind}, parents)
    def where(condition, chosen, alternative):
        return _append_node(nodes, {'kind': 'where', 'shape': shape,
                                    'parents': [condition, chosen, alternative],
                                    'dtype': nodes[chosen]['dtype']})
    if op == 'convert':
        return operand(operands[0], dtype)
    if op == 'boolean':
        # Exact in the operands' dtype, to which a Python scalar converts as PyTorch converts it.
        parents = [operand(value, plan['common']) for value in operands]
        return _append_node(nodes, {'kind': plan['kind'], 'parents': parents,
                                    'shape': shape, 'dtype': 'bool'})
    if op == 'where':
        condition, chosen, alternative = operands
        return where(operand(condition, 'bool'), operand(chosen, dtype),
                     operand(alternative, dtype))
    if op == 'clamp':
        # clamp_scalar_kernel and clamp_kernel: minimum (maximum x lo) hi in the result dtype.
        return apply('clamp', [operand(value, dtype) for value in operands])
    if op == 'elu':
        # elu_kernel: x * scale where x > 0, else expm1 (x * input_scale) * (alpha * scale),
        # binary16 and bfloat16 in binary32. A coefficient of one is not multiplied.
        x = operand(operands[0], dtype, opmath)
        def times(index, coefficient):
            if coefficient == 1: return index
            return other("mul_elem", [index, operand(coefficient, opmath)])
        negative = _scalar_tensor(plan['alpha'], opmath) * _scalar_tensor(plan['scale'], opmath)
        positive = times(x, plan['scale'])
        below = times(apply('expm1', [times(x, plan['input_scale'])]), negative.item())
        condition = _append_node(nodes, {'kind': 'gt', 'shape': shape,
                                         'parents': [x, operand(0, opmath)], 'dtype': 'bool'})
        return _cast_to(nodes, where(condition, positive, below), dtype)
    if op == 'softplus':
        # softplus_kernel: x where x * beta > threshold, else log1p (exp (x * beta)) / beta,
        # binary16 and bfloat16 in binary32. A beta of one is not multiplied or divided by.
        x = operand(operands[0], dtype, opmath)
        beta = plan['beta']
        scaled = x if beta == 1 else other("mul_elem", [x, operand(beta, opmath)])
        smooth = apply('log1p', [other("exp", [scaled])])
        if beta != 1:
            smooth = apply('div', [smooth, operand(beta, opmath)])
        condition = _append_node(nodes, {'kind': 'gt', 'shape': shape,
                                         'parents': [scaled, operand(plan['threshold'], opmath)],
                                         'dtype': 'bool'})
        return _cast_to(nodes, where(condition, x, smooth), dtype)
    if op in _ARITHMETIC_KINDS:
        left, right = operands
        if plan['wide']:
            # The reduced-float product: the tensor operand in the common dtype and the one
            # element's own value, both in binary32, the product rounded once.
            parents = [operand(left, dtype, opmath), operand(right, opmath)]
            return _cast_to(nodes, other(_ARITHMETIC_KINDS[op], parents), dtype)
        return other(_ARITHMETIC_KINDS[op], [operand(left, dtype), operand(right, dtype)])
    if op == 'neg':
        return apply('neg', [operand(operands[0], dtype)])
    if op in _BITWISE:
        # The operands converted to the result's integer dtype, as PyTorch promotes them, then
        # the IR's operator on their words. A right shift fills with the sign bit of a signed type
        # and with zeros of an unsigned one, as ATen's kernels do.
        if op == 'bitwise_right_shift':
            op = 'shift_right_arithmetic' if dtype['signed'] else 'shift_right_logical'
        return apply(op, [operand(value, dtype) for value in operands])
    if op in ('clamp_min', 'clamp_max'):
        # max(x, min) and min(x, max), the IEEE 754-2019 maximum and minimum (max_elem and
        # min_elem: -0 below +0, a NaN propagated), of the tensor and the scalar or tensor bound.
        # Both are converted to the result dtype, which clamp_min and clamp_max promote (an int32
        # tensor with bound 0.5 is clamped in binary32), and broadcast. So clamp_min(-0.0, 0) is
        # +0.0, where PyTorch's scalar-bound kernel keeps the tensor's -0.0.
        kind = "max_elem" if op == 'clamp_min' else "min_elem"
        return other(kind, [operand(operands[0], dtype), operand(operands[1], dtype)])
    if op in ('rsqrt', 'erf', 'gelu'):
        attributes = {_keyword('gelu')[0]: plan['approximate']} if op == 'gelu' else {}
        result = apply(op, [operand(operands[0], dtype, opmath)], **attributes)
        return _cast_to(nodes, result, dtype)
    if op in _OPMATH_UNARY:
        return _cast_to(nodes, apply(op, [operand(operands[0], dtype, opmath)]), dtype)
    if op == 'sign':
        # sign_kernel: 1 where x > 0, -1 where x < 0 and +0 elsewhere, at a NaN and at -0 too. The
        # IR's sign gives the NaN and keeps -0, so PyTorch's sign is this composition.
        x, zero = operand(operands[0], dtype), operand(0, dtype)
        def above(left, right):
            return _append_node(nodes, {'kind': 'gt', 'shape': shape,
                                        'parents': [left, right], 'dtype': 'bool'})
        return where(above(x, zero), operand(1, dtype),
                     where(above(zero, x), operand(-1, dtype), zero))
    if op in _EXACT_UNARY:
        return apply(op, [operand(operands[0], dtype)])
    if op in _OPMATH_BINARY:
        parents = [operand(value, dtype, opmath) for value in operands]
        return _cast_to(nodes, apply(op, parents), dtype)
    if op in _EXACT_BINARY:
        return apply(op, [operand(value, dtype) for value in operands])
    if op == 'div':
        # Division of the operands converted to the result dtype, rounded once (s / x included).
        # For binary16 and bfloat16 with an operand that supplies one element, on either side,
        # the operands are that element's own value and the other operand in the common dtype,
        # both in binary32. The quotient (and its rounding mode) is computed there and rounded to
        # the dtype. ATen's div kernels take that path for a one-element divisor only.
        dividend, divisor = operands
        mode = {_keyword('div')[0]: plan['rounding_mode']}
        if reduced and (_one_element(divisor, dtype) or _one_element(dividend, dtype)):
            parents = [operand(value, opmath) if _one_element(value, dtype)
                       else operand(value, dtype, opmath) for value in operands]
            return _cast_to(nodes, apply('div', parents, **mode), dtype)
        return apply('div', [operand(dividend, dtype), operand(divisor, dtype)], **mode)
    base, exponent = operands
    if isinstance(exponent, torch.fx.Node):
        # pow.Tensor_Tensor: both operands in the common dtype, then in the opmath type. A
        # Python scalar base (pow.Scalar) keeps its own value in the opmath type, as a scalar
        # exponent does below. PyTorch rounds it to the tensor's dtype first and fills ones for
        # the base 1.
        first = (operand(base, dtype, opmath) if isinstance(base, torch.fx.Node)
                 else operand(base, opmath))
        result = apply('pow', [first, operand(exponent, dtype, opmath)])
        return _cast_to(nodes, result, dtype)
    # pow.Tensor_Scalar: x ** s with the exponent's own value, one pow for every exponent, 0 and
    # 1 included. A floating result is FloatLib's Model.pow in the opmath type (binary32 for
    # binary16, bfloat16 and binary32, binary64 for binary64) of x and the exponent, each rounded
    # once to that type, then rounded to the dtype. An integer result is the IR's integer power.
    # PyTorch fills ones for 0, copies for 1 and rounds a binary16 or bfloat16 exponent to the
    # dtype. In binary32, binary64 and bfloat16 it multiplies for 2, 3 and -2 and takes sqrt,
    # rsqrt and reciprocal for 0.5, -0.5 and -1, which differ from pow at -0 and -inf (binary16
    # calls its own pow).
    x = _cast_to(nodes, operand(base, dtype), opmath)
    return _cast_to(nodes, apply('pow', [x, operand(exponent, opmath)]), dtype)
