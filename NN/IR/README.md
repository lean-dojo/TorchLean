# `NN/IR`

`NN.IR` is TorchLean's shape-annotated, op-tagged topological DAG interchange representation. It is
the small shared graph language used by model lowering, verification, export, widgets, and forward
execution by its semantics. It is distinct from the dependent `GraphData` representation that the
`autograd` transforms record.

For public use, prefer the broad library import or the IR entrypoint. Internal code that only needs
one IR component should import the focused leaf directly.

```lean
import NN
-- or, if you only want this subsystem:
import NN.IR
-- or a focused internal leaf, for example:
import NN.IR.Graph
import NN.IR.Semantics
```

`NN/IR.lean` is the subsystem umbrella imported by `NN.IR`. Internal modules should import the
smallest `NN.IR.*` dependency they need.

## Directory Layout

- `Operator.lean`: `OpKind`, its static attributes, and `OpTag` constructor identities with parent
  counts and diagnostic names. This module has no graph, runtime, or artifact-format dependency.
  `OpKind.pointwise` carries an operator of `Spec.PointwiseOp`; see
  [How Core ATen Operators Enter The IR](#how-core-aten-operators-enter-the-ir).
- `Graph.lean`: nodes, dependency edges, declared shapes, and topological well formedness.
- `DType.lean` and `Program.lean`: per-node element types, explicit casts, typed extension
  signatures, and designated inputs/results on that same graph.
- `Interpreter.lean`: heterogeneous tensor evaluation, reusing `Graph.evalNodeRaw` for ordinary
  arithmetic, `Graph.evalTransform` for storage-only operations and `Graph.evalPointwise` for
  pointwise operators. Its scalar family and implementations are explicit; unsupported operations
  fail.
- `Binary.lean` and `BinaryProofs.lean`: FloatLib-backed binary execution at each node's precision
  and the theorem relating interpreter casts to FloatLib's directed conversions.
- `Properties.lean`: shape/type preservation for successful typed execution, plus induction and
  relational rules including accumulated conversion flags and designated outputs.
- `Reference.lean`: the same binary arithmetic plus Boolean and arbitrary-width integer storage
  for layout operations, with checked complete-word encoding and a lossless decoding theorem.
  Integers also implement PyTorch's integer pointwise operators, and integers and Booleans convert
  to binary formats.
- `OpContracts.lean`: shared shape arithmetic for ops such as concat, matmul, pooling, conv, and
  axis-moving utilities. Concat and matmul use list-indexed arbitrary-rank contracts.
- `Infer.lean`: canonical declared-output-shape validation. `Graph.inferShapesFrom` walks the node
  array once, structurally recursive on the remaining node count, and `Graph.checkShapes` wraps it.
- `Check.lean`: public validation wrappers and proposition-level `WellFormed` / `WellShaped` names.
- `Semantics.lean`: denotational evaluator into spec-layer tensor operations with explicit payloads,
  including arbitrary-leading linear and matrix multiplication, checked arbitrary-axis concat, and
  the scoped `IR` notation for graph denotation. `Graph.evalNodeRaw` is the operator dispatch and
  `Graph.evalNode` adds the declared-shape normalization.
- `Payload.lean`: external constants, weights, normalization parameters, and checked custom
  computation bodies keyed by node id.
- `HardMask.lean`: conversions between typed Boolean tensors and the row-major masks stored in
  `OpKind`, shared by graph builders, evaluators, and verifier passes.
- `ShapeSoundness.lean`: the theorems relating `Infer` and `Semantics` (see below).
- `Pretty.lean`: readable text and GraphViz renderers for debugging.

## One Shape Rule Per Operation

Custom operations declare their input shapes in operand order and their output shape. Their
`NN.Kernel.Program` bodies live in the payload, not in a display name or an opaque source string.
Shape inference checks the declared signature; evaluation checks the actual parent tensors and
runs the body with bounded reads. The body's source-correspondence proof also covers failed reads.
This does not imply a derivative rule, a range rule, or an export translation. Those consumers must
report unsupported custom operations until the corresponding implementation is supplied.

Typed `Program` extensions use the same custom-node shape contract, with their element signature
and implementation supplied separately. A typed cast must run through a typed interpreter;
extracting `program.graph` and interpreting it homogeneously is not mixed-precision execution.

Reshape, flatten, broadcast, permutation, transpose, concatenation and slicing need tensor storage, not
scalar arithmetic. Their shared evaluator works unchanged for Boolean, integer, rational, symbolic
and user-defined elements when the interpreter supplies their representation. `Interpreter.reference`
supplies Boolean and fixed-width integer storage alongside FloatLib binary arithmetic. The
binary-only interpreter remains available separately. `detach` remains on the scalar-context path
because dual-valued tensors
must also clear differentiation metadata, rather than merely copy their stored values.

`Infer.nodeOutShape` and `Graph.evalNodeRaw` are both matches over `OpKind`. Apart from `slice`,
they do not contain two copies of the shape arithmetic: every operation whose output shape is not
simply a parent shape or a shape written in the operation goes through a single rule in
`OpContracts`, and both passes call it on the same inputs:

| Operation | Shared rule |
| --- | --- |
| `matmul` | `OpContracts.matmulDims` (operand decomposition); inference reads `MatmulDims.outShape`, the evaluator recovers the typed operands from `leftShape` / `rightShape` |
| `maxPool`, `avgPool` | `OpContracts.planPool` via `inferWindowOutShape`; the plan's `outShape` is the type of the evaluator's result |
| `reduceSum`, `reduceMean` | `OpContracts.checkReductionAxis` (a nonempty axis, as the typed reductions require) and `Tensor.shapeAfterSum` |
| `flatten` | `ShapeUtil.flattenOutShape` |
| `concat` | `OpContracts.inferConcatOutShape`, recomputed by the evaluator on the parent values |
| `slice` | `Spec.Shape.replaceAxis` after an axis and range check that each pass makes itself; the evaluator applies `Tensor.sliceAxisRangeSpec` and checks the declared shape |
| `conv`, `batchNormEval`, `broadcastTo`, `layernorm`, `transpose` | `inferConvConfigOutShape`, `inferBatchNormEvalOutShape`, the decidable `Spec.Shape.CanBroadcastTo`, `layerNormMatrixDims`, `transposePerm` |
| `pointwise op` | `Infer.expectSameShapeParents`: `op.arity` parents of one shape, which is the result shape |

The permute rule is `Spec.Shape.permute?`; the evaluator realizes the permutation by adjacent
swaps (`Graph.swapDepthsForPerm`) and checks the realized shape against the declared one.

## How Core ATen Operators Enter The IR

A core ATen operator enters the IR as an `OpKind` constructor, never as a name registered behind
`Primitive.call`. Operators that share a shape rule share one constructor, which carries the
operator as data; an operator with a shape rule of its own gets its own constructor. Pointwise
operators form such a family. `OpKind.pointwise op` takes an operator of
`Spec.PointwiseOp`: `neg`, `rsqrt`, `erf`, `gelu` with PyTorch's `approximate` mode, `div` with
its `rounding_mode`, `pow`, and `clamp_min` with a scalar bound.

The family has the one shape rule in the table above, and broadcasting is an explicit
`broadcastTo`, as for `add`. One evaluator, `Graph.evalPointwise`, applies the operator's scalar
meaning entry by entry, and one lemma, `Graph.evalNodeRaw_shape_pointwise`, proves that inference
and evaluation agree for every operator of the family. Consumers without rules for the family, such
as the CROWN passes and IR lowering, reject it with one case each; backend planning maps negation,
true division and the tanh GELU to their kernels and rejects the other operators.

Adding an operator to the family adds these pieces and nothing else:

| Piece | Place |
| --- | --- |
| Constructor, attributes, arity and diagnostic name | `Spec.PointwiseOp` in `NN/Spec/Core/Pointwise.lean` |
| Exact real meaning | `Context.realPointwise` in `NN/Spec/Core/Context/Real.lean` |
| FloatLib binary meaning, or no entry when FloatLib lacks the function | `TorchLean.ConfiguredBinary.pointwise` in `NN/Spec/Core/FloatInstances.lean` |
| Host `Float` meaning, when the native runtime provides it | `Context.floatPointwise` in `NN/Spec/Core/Context.lean` |
| Integer meaning, where PyTorch keeps integer results | `Reference.integerPointwise` |
| Artifact spelling and attribute field, with the round trip theorem | `Interop.PyTorch.Wire.pointwiseTag` |
| PyTorch capture and JAX import rules | `NN.Runtime.PyTorch.Export.TorchExport`, `NN/Runtime/JAX/jaxpr.py` |
| Python-scalar spelling for the IR-to-PyTorch exporter | `NN.Runtime.PyTorch.Export.IRPyTorch` |

Negation and `clamp_min` need no table entry, and neither does the tanh GELU, which is
`Activation.Math.geluSpec` for a dictionary without an entry of its own; `Spec.PointwiseOp.meaning`
supplies all three for every scalar type. FloatLib binary has a tanh GELU entry of its own,
described below.

A registered-operator table reached through `Primitive.call` would not satisfy one shape rule per
operation. A custom node's shape contract is its declared signature, so `Infer` would accept
whatever output shape a producer wrote. Its meaning is supplied per interpreter by name, so
`Graph.evalNodeRaw` would have no real semantics for it, and `Infer`, `Semantics` and
`ShapeSoundness` would not cover the operator. One constructor per operator would keep those
theorems, but every exhaustive match over `OpKind` (the CROWN engine and its proofs, IR lowering,
backend planning, the exporters, the wire codec and the shape-soundness dispatch) would grow by one
case per operator. A family grows only in `Spec.PointwiseOp` and the tables above.

A core ATen operator belongs to the family when its operands and result share one shape and one
element type. Comparison operators do not: their result is Boolean, and `Program.checkNodeTypes`
requires every operand of an ordinary node, pointwise ones included, to have the result's type.

### Meanings by scalar type

A scalar dictionary lists the operators whose meaning depends on the type in `Context.pointwise`.
An operator it omits is unsupported for that type, and evaluation reports the operator and the
type rather than substituting an approximation. Exact reals reject poles and inputs outside the
real domain, where PyTorch's IEEE result is an infinity or a NaN in every format, since no real
number is faithful to it. A real result that a binary format rounds to an infinity, such as
`1e30 / 3e-10` in binary32, keeps its real value.

| Operator | Exact reals | FloatLib binary | Host `Float`, `Float32` | Reference integers |
| --- | --- | --- | --- | --- |
| `neg` | `-x` | sign flip, including NaN | sign flip | two's complement |
| `div` | `x / y`, `y ≠ 0` | correctly rounded division | IEEE division | unsupported; PyTorch promotes |
| `div` trunc | `x / y` rounded toward zero, `y ≠ 0` | IEEE quotient rounded toward zero | same | truncating division; a zero divisor is an error |
| `div` floor | `⌊x / y⌋`, `y ≠ 0` | PyTorch's `div_floor_floating`, step by step | unsupported | flooring division; a zero divisor is an error |
| `pow` | `Real.rpow` where the result is real | FloatLib `Model.pow` | C `pow` | PyTorch's `powi`, wrapping |
| `rsqrt` | `1 / √x`, `x > 0` | `1 / sqrt x`, two correct roundings | same | unsupported |
| `erf` | `Spec.erf` | unsupported | unsupported | unsupported |
| `gelu` none | `x Φ(x)` | unsupported | unsupported | unsupported |
| `gelu` tanh | `Activation.Math.geluSpec` | PyTorch's kernel step by step, FloatLib `tanh` | `geluSpec` | unsupported |

Mathlib has no error function, so `Spec.erf` is defined in `NN/Spec/Core/Erf.lean` with its
derivative, oddness, strict monotonicity and limits `±1`. `Spec.PointwiseOp.meaning_gelu_none_real`
gives the exact GELU as `x (1 + erf (x / √2)) / 2`, and `Spec.cdf_gaussianReal` in
`NN/Proofs/Probability/NormalCdf.lean` proves that `(1 + erf (x / √2)) / 2` is Mathlib's
distribution function of the standard normal law, `ProbabilityTheory.cdf (gaussianReal 0 1)`.
`Spec.PointwiseOp.meaning_gelu_none_eq_mul_cdf` therefore states the exact GELU as `x Φ(x)`.
Binary flooring division runs `Spec.divFloorSchedule`, and `Spec.divFloorSchedule_real` proves
that the same schedule computes `⌊x / y⌋` in exact arithmetic.

FloatLib's `Model.pow` is correctly rounded for integral exponents; PyTorch's is not.
`torch.pow(x, torch.tensor(3.0))` and `torch.pow(x, torch.tensor(4.0))` call SLEEF's `powf` and
differ from the correctly rounded power on 3 and 7 of 256 binary32 values, where the IR is correctly
rounded on all 256. For other exponents `Model.pow` computes `exp (y * log x)` with every step
rounded to the node's format, so its error grows with `|y log x|`. On 256 non-integral binary32
powers the result is within 1 ulp of the correctly rounded power where `|y log x| < 1`; where
`|y log x| ≥ 1` it is up to 35 ulps away (`0.001 ^ 10.5`) and at most `1.39 |y log x|` ulps. In
binary64 it is up to 27 ulps away, at most `1.70 |y log x|` ulps where `|y log x| ≥ 1`. PyTorch's
`pow` is within 1 ulp on the same inputs, so a non-integral binary `pow` in the IR is FloatLib's
approximation, not PyTorch's result. In bfloat16 and binary16 the difference mostly disappears. On
4224 samples per case, bfloat16 powers equal PyTorch's vectorized loop apart from the sign and
payload of NaN results, and binary16 powers differ in at most 2 places per case, by 1 ulp.

FloatLib's tanh GELU rounds PyTorch's binary64 constants once to the format and the inner sum
`x + κ x³` once, as a fused multiply-add. One PyTorch binary contains each CPU kernel at three
capability levels, DEFAULT, AVX2 and AVX-512, and selects at run time the highest the processor
supports, or the one that `ATEN_CPU_CAPABILITY` names. On x86-64 (a Xeon Platinum 8275CL), both
loops fuse that sum at the AVX2 and AVX-512 levels and round `κ x³` and the sum separately at
DEFAULT. On 8192 normal samples with standard deviation 4, each loop at the AVX2 and AVX-512 levels
equals the formula with the fused sum on all 8192 in binary32 and binary64; with the unfused sum
the vectorized loop equals it on 8153 and 8161 and the scalar loop on 8156 and 8157. At DEFAULT both
loops equal the unfused formula on all 8192. PyTorch on aarch64, on Apple processors and on CUDA
was not measured. At the AVX2 and AVX-512 levels the loops differ only in `tanh`: the vectorized
loop calls SLEEF's, the scalar loop the C library's and the IR FloatLib's; at DEFAULT the
vectorized loop applies the C library's `tanh` to each element. With the loop's `tanh` in place of
FloatLib's, the IR's formula returns each loop's words on all 8192 samples. FloatLib's `tanh` is
then the only source of difference: the IR differs from the vectorized loop on 30 binary32 and 10
binary64 samples and from the scalar loop on 439 and 452. An unfused sum would raise these counts
to 69, 41, 466 and 482.

The figures below compare the IR with both loops at the AVX-512 level and with the exact value of
the formula, computed with mpmath and rounded once, on 2^18 normal samples with standard deviation
4 and 2^18 evenly spaced inputs in `[-12, -2]` in binary32 and binary64, and on every finite
bfloat16 and binary16 input. For `x ≥ -1` the IR is within 3 ulps of the exact result in binary32
and binary64, as is PyTorch's vectorized loop, while the scalar loop is within 5 and 4 ulps.

For negative `x`, `1 + tanh(inner)` subtracts two numbers close to 1, so an error of a fraction of
an ulp in `tanh(inner)` becomes a large error relative to the small result. This cancellation, not
the IR, makes ulp counts grow. Between `-3` and `-1` the IR and PyTorch's vectorized loop are up to
198 binary32 ulps and 201 binary64 ulps from the exact result, and the scalar loop up to 198 and
196. Below `-3` the largest distances, millions of binary32 ulps and more in binary64, occur next to
the `-0` cutoffs described below. Excluding inputs where one result is zero and the other is not,
the IR differs from PyTorch's vectorized loop by at most 38 binary32 ulps (at `x = -2.36786`) and
15 binary64 ulps (at `x = -1.88297`), and from its scalar loop by up to 295444 binary32 ulps (at
`x = -4.50812`) and 15922 binary64 ulps (at `x = -3.88725`), as much as the two loops differ from
each other. There the absolute error is the meaningful measure. Over every measured `x < -1`
the IR's error is at most `0.708 |x| u` in binary32 and `0.718 |x| u` in binary64, where `u` is
`2^-24` and `2^-53`, as is the vectorized loop's; the scalar loop's reaches `1.105 |x| u` and
`1.210 |x| u`.

Once `tanh(inner)` rounds to `-1`, the result is `-0`. For finite `x`, the IR and PyTorch's scalar
loop return `-0` exactly for `x ≤ -5.157855` in binary32 and `x ≤ -7.200040557303352` in binary64,
the vectorized loop for `x ≤ -5.061381` and `x ≤ -7.145065874418114`, while the exact result rounds
to `-0` only for `x ≤ -10.770868` and `x ≤ -21.547075410436022`. At `x = -∞` the IR and both loops
return a NaN, since `x (1 + tanh(inner))` is `-∞ · 0`. Between the cutoffs of the IR and the
vectorized loop, the vectorized loop returns `-0` and the IR a nonzero result, nearer the exact
value at every one of 13171 binary32 and 12867 binary64 inputs in that band. This accounts for
every zero-versus-nonzero difference between them, 1130 of the binary32 samples and 279 of the
binary64 samples. In bfloat16 and binary16, the IR returns the scalar loop's words for every finite
input except binary16 `-0.249756`, where it differs by 1 ulp. It returns the vectorized loop's
words except for 4 bfloat16 and 25 binary16 inputs in `[-5.15625, -5.0625]`, where the vectorized
loop returns `-0`.

### Frontend lowering

The PyTorch capture writes the arithmetic of the vectorized loops of PyTorch's CPU kernels, not only
the ATen name. Binary16 and bfloat16 inputs of `rsqrt`, `erf`, `gelu` and `pow` are widened to
binary32, where PyTorch computes them, and the result is rounded once, except for the powers PyTorch
computes by multiplication, described below.

Both frontends convert values as their framework does. PyTorch's `c10::convert` and XLA's
`convert_element_type` make binary16 and bfloat16 values from binary32, so converting an integer,
Boolean or binary64 value to either format takes two IR casts, to binary32 and then to the reduced
format, and rounds twice. The IR's single cast rounds the exact value once and stays available to
frontends that convert that way. The two disagree on int32 `16842753`, which becomes bfloat16
`16777216` through binary32 and `16908288` in one rounding, and on 478 of 6008 binary64 test values
converted to bfloat16 and 528 converted to binary16. Integers convert to binary16 alike both ways,
since binary32 holds every integer below binary16's overflow threshold exactly. The PyTorch capture
applies the rule to `Tensor.to` and to the promotions of `add`, `sub`, `mul`, `maximum` and
`minimum`, the JAX importer to `convert_element_type`.

A Python scalar becomes a constant as PyTorch wraps it, a bool as a Boolean, an int as int64 (uint64
from `2^63`) and a float as binary64, then converted to the operation's dtype by the same rule; so
does the bound of `clamp_min`. An int beyond `2^53` therefore rounds once to binary32:
`2^60 + 2^36 + 1` becomes `2^60 + 2^37`, where rounding through binary64 would give `2^60`. A
scalar operand of `add` or `sub` is converted to the tensor's dtype and the operation runs there.

`clamp_min` with a scalar bound returns the bound where the tensor is below it and the tensor's
value everywhere else, as PyTorch's kernel compares, and a NaN where the bound is one. Eager
`clamp_min(-0.0, 0)` is therefore `-0.0`, where IEEE maximum, the IR's `max_elem`, gives `+0.0`;
binary32, binary64, bfloat16 and binary16 agree, in the scalar and the vectorized loop. The capture
lowers it to its own pointwise operator. `clamp_min` with a tensor bound computes IEEE maximum and
is rejected. The IR's `relu` is TorchLean's ReLU, which returns `+0.0` at `-0.0`, where eager
PyTorch's `relu` returns `-0.0`, so a sample comparison of a ReLU at a negative zero fails.

Reduced-precision division and multiplication compute in binary32 only when the right operand
supplies one element to the whole output: a Python scalar, a one-element tensor, or a view with
stride 0 along every dimension longer than one, as TensorIterator's `is_scalar(2)` test decides.
That operand keeps its own value in binary32, so a binary32 rank-zero tensor is not rounded to
bfloat16 first. Any other division or product rounds both operands to the reduced dtype and computes
there, so `torch.div(0.1, x)` divides bfloat16 `0.1`, and `1000 // x` runs the flooring schedule in
bfloat16. The same holds for a scalar on the left of an explicit `torch.mul(s, x)`, which
`torch.export` records as `mul.Tensor(s, x)`: bfloat16 `torch.mul(0.1, x)` is `bfloat16(0.1) * x`,
and it differs from `0.1 * x` on 868 of 4096 values. Python's `s * x` calls `x.__rmul__(s)`, which
is `x * s`; `torch.export` records it as `mul.Tensor(x, s)` and FX as `operator.mul(s, x)`. Python's
`0.1 / x` is `Tensor.__rtruediv__`, which computes `reciprocal(x) * 0.1`. `torch.export` records
those two operators and FX records `operator.truediv(0.1, x)`. The capture lowers both to the
reciprocal and the binary32 product, which return eager PyTorch's words on 4096 values in
bfloat16, binary16 and binary32.

`pow` with a scalar exponent follows `pow_Tensor_Scalar_out` and PowKernel. Exponent `0` fills ones
and `1` copies. In binary32, binary64 and bfloat16, exponents `-1`, `0.5` and `-0.5` use the
reciprocal, square-root and reciprocal-square-root kernels. These tests use the Python value. For
bfloat16 and binary16 the exponent is then rounded to the tensor's dtype. Binary32, binary64 and
bfloat16 compute a remaining exponent of `2`, `3` or `-2` by multiplication, and every other
exponent calls `pow` with the rounded exponent. So `torch.pow(x, 3)` becomes `x * x * x`, two
roundings, not a rounded cube, and bfloat16 `x ** 2.001` is `x * x`. A scalar base of `pow` is
rounded to the tensor's dtype too, and the base `1` fills ones.

PyTorch does not compute every element with the same code. Its vectorized loop takes two vectors at
a time, and a scalar loop covers the remaining elements of each contiguous run and every element of
a non-contiguous input, so a contiguous tensor shorter than two vectors runs entirely in the scalar
loop. The vector width is that of the capability level the kernel runs at. On an AVX-512 CPU (a
Xeon Platinum 8275CL with PyTorch 2.13), `pow` and `gelu` run at the AVX-512 level: the scalar loop
computes all of a contiguous tensor with fewer than 32 binary32 or 16 binary64 elements, or fewer
than 64 bfloat16 or binary16 elements for `gelu` and bfloat16 elements for `x ** -0.5`.
`torch.rsqrt` is registered with `REGISTER_DISPATCH`, which compiles no AVX-512 kernel for it, so
it runs at the AVX2 level and its threshold is 32 bfloat16 or binary16 elements. `x ** -0.5`
reaches the same reciprocal square root kernel from the AVX-512 kernel of `pow`. The scalar loop
evaluates binary32 `pow` with a scalar exponent in binary64 with the unrounded exponent, rounds the
square root to the reduced format before dividing in bfloat16 and binary16 `rsqrt`, and takes
`pow` from the C library instead of SLEEF. For these operators the eager result, and a sample
comparison with it, can depend on tensor length and layout.

The JAX importer records supported operations, shapes, dtypes and casts. Reference execution
uses TorchLean and FloatLib arithmetic: division is IEEE division, `rsqrt` is `1 / sqrt x`
rounded twice, and results underflow gradually. `div`, `pow`, `neg`, `rsqrt` and `erf` map to
IR operators; integer `div`
rounds toward zero. Binary16 and bfloat16 `pow`, `rsqrt` and `erf` are computed in binary32 and
rounded once, as XLA computes them. On 8192 samples per operator and format, the IR returns XLA's
words for every binary16 `pow` and `rsqrt` sample and every bfloat16 `rsqrt` sample, and for 8191
bfloat16 `pow` samples: XLA's result is its own binary32 power rounded once, and FloatLib's binary32
power differs from it. The IR has no binary `erf`, so a graph using it does not execute.
`integer_pow` becomes the products XLA computes, square-and-multiply in JAX's order with a final
division for a negative exponent; on 4096 samples XLA's `integer_pow` for the exponents 3, 5, 7 and
-3 equals that schedule rounded at each step in bfloat16, binary16 and binary32.
Nondefault dot precision and algorithms are rejected: a declared FP32 input and result do not
specify whether the multiplications use BF16, TF32 or FP32, and the current matmul node has no
separate operand/accumulator arithmetic rule. Default matmul capture still does not establish
XLA's reduction order or backend equivalence.

XLA's compiled CPU code departs from these meanings in the following ways, and the importer does
not follow it. Eager JAX compiles each primitive, and each nested `jax.jit` call, as a computation
of its own; `jax.jit` compiles the whole function as one. XLA's results are those of eager JAX
unless `jax.jit` is stated. Counts are of binary32 samples unless another format is named.

- A division by a broadcast divisor is multiplication by the divisor's reciprocal rounded to the
  format, `multiply(x, 0.333333343)` for `x / 3.0`, whose scalar divisor JAX broadcasts within the
  division. The divisor need not be constant: a 0-d argument, or a `[64, 1]` divisor of a
  `[64, 768]` array, is treated the same way. Under `jax.jit` so is a divisor computed from
  constants alone, such as a closed-over array; eager JAX divides by that array as written, and by
  a divisor that `jnp.broadcast_to` broadcast in a primitive of its own. XLA's `x / 3.0` and
  `x / 10.0` differ from the IEEE quotient on 1382 and 839 of 4096 samples. Binary16 multiplies by
  the binary16 reciprocal and differs on 5487 of 16384 samples for `x / 3.0`; bfloat16 multiplies
  in binary32, and the rounded product equals the quotient on all 16384. `jnp.mean` divides its
  sums by the broadcast element count this way: over 64 rows of 768, XLA's mean is the sum times
  binary32 `1/768` in all 64 rows and the sum divided by 768 in 40.
- Under `jax.jit`, XLA's simplifier rewrites divisions: `x / sqrt(y)` becomes `x * rsqrt(y)`,
  `x / rsqrt(y)` becomes `x * sqrt(y)`, `x / y ** c` becomes `x * y ** -c`, `(x / y) / z` becomes
  `x / (y * z)` and `x / (y / z)` becomes `(x * z) / y`. The first, fourth and fifth differ from
  the IR on 1622, 1443 and 1425 of 4096 samples. A broadcast divisor is replaced first, so
  `(x / y) / 3.0` is `(x / y) * 0.333333343` in eager JAX too, and a rewrite's own division is
  rewritten again: `x / (y / (z / w))` is `(x * z) / (y * w)`. Binary16 rewrites with every step
  rounded to binary16. In bfloat16, which XLA computes in binary32, `(x / y) / z` and
  `x / (y / z)` stay as written.
- `pow` is XLA's power function, which is not correctly rounded for integral exponents: eager
  `x ** 3.0` differs from the correctly rounded cube, which the IR computes, on 2 of 4096 samples.
  Under `jax.jit`, XLA computes the constant exponents 2 and 3 as products and -1 as a reciprocal
  in binary32 and binary64, an exponent that is a closed-over array included, and its
  `x * (x * x)` differs from the cube on 1062. In binary16 it rounds each product to binary16, and
  in bfloat16 it computes the products in binary32 and rounds once. On 8192 samples per class,
  binary32 and binary64, XLA's power is within 1 ulp of the correctly rounded power. It differs
  from the IR's by at most 1 ulp for integral exponents, where the IR is correctly rounded, and
  where `|y log x| < 1`, where FloatLib's power is within 1 ulp of the correctly rounded one; where
  `|y log x| ≥ 1`, FloatLib's power is up to 16 ulps from the correctly rounded power, and from
  XLA's.
- `rsqrt` is XLA's own approximation, within 1 ulp of the correctly rounded result and within 2
  ulps of the IR's `1 / sqrt x` rounded twice. It differs from the IR's result by 1 ulp on 5473906
  and by 2 on 50830 of the 16777216 binary32 values in `[1, 4)`, and by 2 ulps on 3106 binary32
  and 1291 binary64 values of 2^20 random normal values in each format. On 65536 samples it
  differs on 22080 binary32 and 18763 binary64 values. In bfloat16 and binary16, which XLA computes
  in binary32 and rounds once, as the IR does, XLA's words equal the IR's for every positive
  normal input.
- Subnormal operands and results are flushed to zero in binary32, binary64 and bfloat16. On 4096
  subnormal operands per format, `x * 1.5`, `x / 1.5`, `x + x` and `x * 1` return zero and
  `rsqrt(|x|)` returns infinity for all 4096, and squares whose IEEE value is subnormal return zero
  for all 4096 operands. Subtraction, `max`, `min`, `sqrt`, `log`, `exp`, the logistic function,
  sums, matrix products and binary64 `tanh` flush too, and so do conversions of binary32 and
  bfloat16 to binary64. Negation, `abs`, `sin`, binary32 and bfloat16 `tanh`, layout
  operations and conversions between binary32 and bfloat16 keep subnormal values, and so does
  binary16 arithmetic, which XLA computes in binary32, where binary16 subnormals are normal. XLA
  also converts binary64 values below binary32's normal range, such as `1e-40`, to zero in binary32
  and bfloat16. Values that `jax.jit` folds at compile time are not flushed, until an operation at
  run time uses them.

`verify`'s `compare` fails whenever the IR's words differ from the reference's, under its NaN
policy. For a Jaxpr artifact it then tries these rules on the mismatch, as advice for the reader.
The JAX importer records `reference_execution`, `jit` for a function that is one `jax.jit` call
and `eager` otherwise, the computation of each node in an eager function, and the primitive that
produced each node. `torchlean.verify` records with the reference the platform that computed it
and jaxlib's version, and, when `jax.jit` compiles any part of the function, the outputs of
evaluating it one primitive at a time, inside nested `jax.jit` calls too, which XLA computes
without `jax.jit`'s rewrites. Only a reference from XLA's CPU backend is emulated. The checker runs
the program a second time as XLA's CPU code computes it, where that code is exactly emulable.
Divisions by broadcast divisors, and under `jax.jit` by constant ones, become products with the
rounded reciprocal. When `reference_execution` is `jit`, `jax.jit`'s rewrites apply: those of
quotients of quotients and by quotients in binary16, binary32 and binary64, those of divisions by
square roots, reciprocal square roots and powers in binary32 and binary64, and the products for
the exponents 2, 3 and -1, in binary32 and binary64 and in binary16 with each product rounded to
binary16. The flushing above applies to the operations listed.

XLA's `rsqrt` and power function are not emulable. A value that one of them computes in binary32
or binary64 counts as reproduced within 2 ulps of that format for `rsqrt` and 1 ulp for the power,
the largest distances measured between XLA's functions and the IR's. The window's ends are carried
to the result through layout operations, negation and conversions, so a bfloat16 or binary16
result, which XLA and the IR compute in binary32 and round once, is reproduced only where the
binary32 window holds a rounding boundary of the reduced format. A window never crosses zero or
reaches an infinity. A power gets a window only when both its operands are arguments of the
program: a power with a small exponent hides a large error in its base, and a changed exponent
moves the result by about one ulp. Eager `x ** 3.0`, whose exponent is a constant, therefore has
no window, and a `jax.jit` cube is listed as consistent with the products only when XLA's power,
primitive by primitive, returns the IR's words on every argument.

If the second run reproduces every differing element, the failed finding lists the behaviours in
`consistent_with`, and its message says that the mismatch is consistent with them and that an
importer error with the same effect cannot be ruled out. Under `jax.jit` the list also requires
the primitive-by-primitive outputs, when the artifact has them, to equal the IR's results or to be
reproduced in the same way without `jax.jit`'s rewrites. A mismatch that the second run does not
reproduce fails with nothing listed, and the message names elements with their IR, emulated and
reference words, and the behaviours emulated. A FloatLib transcendental, or an importer bug, next
to one of these operations therefore lists nothing. Neither does a `jax.jit` `x / sqrt(y)` where
XLA's `rsqrt` differs from the IR's, since its result is a product, not XLA's `rsqrt` itself, nor a
`jax.jit` `x / rsqrt(y)`, whose primitive-by-primitive result is a quotient of XLA's `rsqrt`.

`consistent_with` never excuses a mismatch, because some importer errors are indistinguishable
from XLA's behaviour on the supplied arguments. A divisor one ulp off whose rounded reciprocal is
unchanged gives XLA's words exactly: eager `x / 3.62` written as `x / 3.6200001` is listed as
consistent with the reciprocal. Under `jax.jit` the rewrites map wrong operand orders onto the
reference, such as `(x / z) / y` for `(x / y) / z` and `z / (y / x)` for `x / (y / z)`; without a
primitive-by-primitive reference these too are listed. The checker trusts the artifact's record of
how the reference was computed, so a forged record can at most add this advice to a failure.

## Shape Soundness

`NN/IR/ShapeSoundness.lean` proves that the two passes agree, organized as one lemma per operator
family (`Graph.evalNodeRaw_shape_*`) plus a lockstep induction over the node array:

- `Graph.evalNodeRaw_shape_of_infer`: if inference assigns the declared shape to a node and the
  node's parents carry the shapes inference saw (`Graph.ParentShapesOf`), then the raw evaluator
  value already has the declared shape.
- `Graph.denoteAllRaw_eq_denoteAll`: on a graph accepted by `checkShapes`, evaluating without the
  per-node declared-shape normalization (`Graph.denoteAllRaw`) gives the same result as
  `denoteAll`. The normalization is defense in depth, not a second shape rule.
- `Graph.checkShapes_sound` / `Graph.denoteAll_shape`: every value produced by `denoteAll` has the
  declared shape of its node. The `checkShapes` hypothesis is not needed for this conclusion
  because `evalNode` normalizes; the previous theorem is where it does work.

For `permute`, `transpose`, `conv`, and `slice` the evaluator compares the shape it realizes with
the declared shape, so soundness holds through that comparison; proving that the adjacent-swap
lowering realizes `Shape.permute?`, and that the convolution contract's list arithmetic equals
`Spec.convOutSpatialDilated`, are separate lemmas not attempted here.

Note on partiality: `denoteAll` is not total on well-shaped graphs. `.log` rejects nonpositive
input data, payload-backed nodes reject inconsistent payloads, and a pointwise operator rejects
inputs outside its meaning at the scalar type, or the type itself when it has no meaning for the
operator. See the header of `Semantics.lean`.

## Relationship To `NN.GraphSpec`

`NN.GraphSpec` is a typed authoring DSL for model architectures, with a pure semantics and lowering
to TorchLean runtime programs. `NN.IR` is the lower-level op tagged graph target that
verification/export/runtime tooling can consume after a model has been lowered or traced.

In PyTorch terms, `NN.GraphSpec` is closer to a typed model construction DSL; `NN.IR` is closer to
an FX/TorchScript-style graph with explicit shapes and external parameter payloads.

## How A Tensor Program Reaches IR

The same IR represents general tensor computations. A program need not contain a layer, a loss,
parameters, or an optimizer: its interface is an ordered collection of tensor inputs and results.
Supported array operations and numerical algorithms use the same shape and element-type checks
as model computations. This is not a compiler for arbitrary Lean or Python programs; control flow
and operations without an explicit lowering rule are rejected.

From Python, decorate a function or PyTorch module class with `@torchlean.compile`, or wrap an
existing module with `model = torchlean.compile(model)`, then call it normally. Capture is lazy:
PyTorch supplies a guarded FX graph, or JAX supplies a traced Jaxpr; the model and tensor
arguments select the frontend automatically. A call captures the whole tensor or batch, without
a separate check loop over its entries. TorchLean lowers that graph to the shared IR and requests
structure and type checks.
The framework executes the captured computation, retaining its device and automatic
differentiation. The callable's `.torchlean.artifact` and `.torchlean.report` expose the most
recently compiled graph and its check result; `.torchlean.graphs` retains all successfully checked
specializations. A cached call can reuse an earlier specialization without changing the latest
capture. Unsupported capture or failed checks raise. These checks do not prove frontend equivalence
or correctness on every input.

Use `torchlean.verify(model, *inputs)` for explicit checks, including sample execution and
comparison. It also selects the frontend automatically. Other producers can pass a typed artifact
dictionary directly. Optional execution and comparison concern only supplied arguments.

Add `export="Model.lean"` to `verify` to retain that same capture as a self-contained Lean module.
It embeds the artifact and defines `Model.program` through the shared importer. Lean checks the included
import and shape/type/interface proofs by kernel reduction when the module is compiled; export
does not launch a compiler. Mathematical properties can refer
to that program and the embedded constants; they still require their own proofs. A `.json` export
keeps the interchange artifact instead. Neither export replaces an existing file. See the
[captured-computation example](../../home_page/examples/verify/index.md).

There are several routes into the same graph language:

- TorchLean model code can lower supported fragments into IR for execution, inspection, or
  verification.
- GraphSpec models can be lowered when the architecture needs explicit sharing and
  named parameter layouts.
- PyTorch `torch.export` records positional tensor arguments, including unused arguments, from
  its graph signature and lists them in `input_ids`, as the ONNX adapter does. Lean parses and
  validates the resulting tensor graphs.
- The Jaxpr adapter in `NN/Runtime/JAX/jaxpr.py` lowers static tensor computations to the same
  format. Its tensor arguments are listed in order in `input_ids`; each input has its own shape
  and dtype. `Import.JAX.parseProgram` checks per-node types and explicit casts.
  `Interpreter.binary` uses FloatLib for mixed binary precision; the homogeneous graph path remains
  scalar-polymorphic. Structured control flow is not lowered; see the
  [JAX example](../../home_page/examples/jaxpr/index.md).
- Verification examples can build small graphs directly when the graph itself is the artifact under
  study.

Mixture-of-experts routing is one operation, `top_k_mask k`, whose meaning is
`Spec.MoE.topKMaskSpec`: one for the `k` largest entries of every row, ties to the lower index.
A row of `E` totally ordered scores (no NaN) gets `min k E` ones. With NaN the pairwise comparison
can cycle, so `[0, NaN, 1]` keeps no entry for `k = 1` and all three for `k = 2`. The
PyTorch adapter captures the usual idiom `torch.zeros_like(p).scatter(-1, torch.topk(p, k).indices,
1)` as that node. For `0 < k < E`, agreement requires finite scores and a strict gap between the
`k`-th and `k + 1`-th largest entries; `torch.topk` leaves ties unspecified. Capture records these
assumptions without checking them. Empty and full masks need no cutoff rule.
`NN/Tests/IR/MoEImport.lean` exports a
Mixtral-style block from PyTorch and checks that the imported graph computes `Spec.MoE.forwardSpec`
bit for bit.

Frontend coverage is narrower than the graph's operator vocabulary. JAX division, mean and the
LayerNorm expansion through `rsqrt` import directly. Square root needs a structurally nonnegative
argument to agree with the IR's negative-input clamp. The JAX importer has no `reduce_max`
lowering, so JAX's softmax expansion, and attention code that uses it, does not import. JAX's
exact GELU uses `erfc`, which has no IR operator.

## Role And Scope

Write ordinary models through `TorchLean.nn`, `Trainer`, or `GraphSpec`, then lower them.
Direct `Node` construction is useful for developing IR consumers and certificate examples.

## Current Consumers

| Consumer | How it uses IR |
| --- | --- |
| Execution | `Graph.denote` and `Graph.denoteAll` run a graph by its semantics, and the typed interpreter in `Interpreter.lean` runs it at runtime precision. |
| Verification | Runs IBP/CROWN-style passes, margin checks, and certificate replay over node ids and payloads. |
| TorchLean lowering fragments | Prove that supported source fragments lower to IR with the same denotation. |
| PyTorch/ONNX/export paths | Use a small graph format to make parameter order and tensor shapes explicit at the boundary. |
| Widgets and graph pages | Render graphs, inferred shapes, execution traces, and dependency structure for debugging. |

## Proof And Runtime Status

The IR is the object shared by proofs and runtime code, but proof coverage is named fragment by
fragment. Current theorem work covers shape soundness, typed interpreter casts, relational
execution rules, supported source-to-IR lowering fragments, and verification-oriented bound
propagation. Separately, `GraphData` lowers to the autograd tape; that typed lowering is not an
assertion that every `NN.IR` node is differentiable. A new operator should therefore add three
things in the right places:

- its shape contract in `OpContracts`/`Infer`;
- its semantics in `Semantics` or a documented payload-backed evaluator;
- the runtime/proof/checker coverage that makes the operator usable in the intended workflow.

Adding a tag without these follow-up pieces creates a graph that can be printed but not responsibly
used for verified claims.

## Proving Properties Across A Program

An imported `Program` already has Lean semantics. A proof can work with those operations directly,
without first identifying the graph as an MLP, transformer, or another named architecture.
`NN.IR.Properties` provides the common traversal argument:

- `Interpreter.run_types` proves that every computed tensor has its declared element type and
  shape, with exactly one value per node in graph order. `Interpreter.eval_types` gives the same
  guarantee for designated results, preserving their order and repeated result ids.
- `Interpreter.run_induction` carries a predicate on intermediate values and conversion flags.
  To establish a bound, prove it for the initial state and show that each successful operation
  preserves the predicate.
- `Interpreter.run_rel` relates two executions of the same typed program. The interpreters,
  arguments, and parameter payloads can differ. A relation can compare exact and rounded values,
  or inputs related by a permutation, provided its local rules account for every operation.
- `Interpreter.eval_rel` carries a pointwise relation through to the designated outputs, including
  repeated outputs and arbitrary return order.

The proof rules use the existing `Tensor` values, `Program`, and interpreter. Numerical relations
must use that interpreter's actual arithmetic, casts and parameters. The traversal theorem does not
supply error bounds or invariance rules for an operation whose corresponding proof is missing.

These theorems assume the executions being compared succeed. They do not establish success for
every input, prove a frontend translation faithful, or verify external kernels. Comparing two
different graph structures additionally needs a correspondence between those computations; the
same-program relational rule does not establish arbitrary optimization equivalence.

## Element Descriptors And Extensions

`DType.toJson` and `DType.fromJson` share one frontend-independent representation. Binary
descriptors retain exponent width, fraction width, bias, and exceptional-value encoding. Decimal
descriptors retain FloatLib's declet count, exponent width, and signed bias. Integer widths,
posit widths, complex component types, and named extensions are represented explicitly too.
Named-type component arrays preserve qualified names without losing numeric or escaped components.
Element descriptors and custom calls share the same total decoder for component arrays and Lean
qualified names, including escaped components.
`DType.fromJson_valid` proves that a successfully decoded descriptor passes its parameter checks.

`Reference.read_write` proves lossless element-word decoding for every supported format and
integer width. Boolean words are zero or one. Integers use Lean's `BitVec n`, retaining signedness
in the descriptor; no conversion to a host float is involved. This reference implements their
layout operations, PyTorch's integer pointwise operators, and conversions to binary formats that
round the exact integer once. It implements no other integer arithmetic and no casts between
distinct integer types. Decimal, posit, complex and named element types have no implementation in
this reference.

`Import.IR.parsePayload` takes an interpreter and a word decoder explicitly. The same loader
checks constants and linear parameters at their declared types and shapes, including decoder
results. A custom decoder cannot silently return a tensor at a different type or shape.

A producer can declare a custom tensor node with `primitive = "call"`, a qualified `name`,
`input_shapes`, `input_dtypes`, and `output_dtype`. The shared checker compares this signature
with the actual edges and result. Execution additionally requires an implementation in
`Interpreter.Payload.call`; merely declaring the signature does not register executable code or
prove its mathematical properties. The default binary payload rejects unimplemented calls.
Likewise, a valid decimal, posit, or named descriptor does not imply that the binary interpreter
can execute it. Other interpreters must provide the corresponding scalar operations.

## Payload Discipline

`NN.IR.Payload` stores constants, weights, normalization parameters, and custom bodies by node id.
Topology checking needs no parameter values; evaluation rejects missing or inconsistent payloads.
Verifier input boxes use a separate payload. Imported weights remain artifacts to validate.

## Release Invariants

- Node ids are array indices: `g.nodes[i].id = i`.
- Parents always point backward: every parent id is smaller than the child id.
- Parameter tensors are not embedded in `Graph`; `const`, `linear`, and `conv` use external
  payload stores keyed by node id.
- Shape checking is centralized through `Infer.nodeOutShape`; `Graph.checkShapes` delegates to
  that implementation to avoid duplicate op-contract logic.
