# `TorchLean/NN/IR`

`IR` is TorchLean's graph interchange representation: the one graph language that model lowering,
verification, export, the widgets and forward execution share. A graph (`IR.Graph`) is a list
of nodes in evaluation order, where a node's id is its position, and a list of outputs. A node is a
graph input, a constant of the payload, or a registered operator of `TorchLean.Op` with one
reference (`Ref`) per input port. A graph runs by the rules of its operators, and a custom operation
runs by its checked program (`Graph.eval`). The structural check `Graph.check` accepts exactly the
well-formed graphs (`Graph.check_ok_iff`). `IR` is separate from `GraphData`, the dependent
representation that the `autograd` transforms record.

A file imports the IR modules that declare the names it uses. `Graph.eval`, for example, is in
`TorchLean/NN/IR/Eval.lean`, which also provides the graph type:

```lean
import TorchLean.NN.IR.Eval
```

## Directory Layout

- `DType.lean`: the element types (`DType`) of typed programs, one per step with value ports,
  among them the targets of conversions and the declared types of calls, and their JSON
  (`DType.toJson`, `DType.fromJson`).
- `Scalar.lean`: the element types of the reference runtime
  (`TorchLean.IR.Program.Machine.reference`). They are FloatLib's configured binary formats
  (`Binary`) and, beside them, Booleans and integer words of any positive width (`Reference`), with
  a checked encoding of complete words, a lossless decoding theorem (`Reference.read_write`) and the
  conversions between them. Binary casts are FloatLib's directed conversions (`Binary.cast_value`).
  Integers and Booleans convert to binary formats, and binary values to integer types in the cast's
  rounding direction, with NaN, infinities and integers the type cannot hold as errors
  (`toInteger_cast_binary`). Integer types convert among themselves modulo `2 ^ width`, and integers
  and binary values convert to Booleans as `x != 0`.
- `OpContracts.lean`: the shape helpers used by the model recorder and the bound passes: slices,
  matmul decomposition, pooling and convolution. Other operation shapes are checked when
  `Op.decode` constructs the registered operator from its wire record.
- `MatmulLayout.lean`: the flat layout of a matrix product (`IR.MatmulDims`) and the row-major
  contraction over it (`MatmulDims.matmulFlat`). On operands whose batch prefixes broadcast to the
  product's leading shape, the contraction is the registered batched product `Tensor.batchMatmul`
  (`MatmulDims.matmulFlat_read`, `MatmulDims.matmulFlat_eq_batchMatmul`). The bound passes over
  registered operators use it.
- `OpGraph.lean`: typed straight-line graphs of registered operators (`OpGraph K σ ps`, whose steps
  `Step.op o args h` apply admissible operators to earlier values, with constants at every port).
  It also holds `OpGraph.toOp`, which builds their graph of registered operators node for node, and
  `OpGraph.eval_toOp`: that graph returns the typed values on every input, with no premise.
- `CustomRule.lean`: `Op.customRule` and its checked entry `Op.customCheck`, the rule of a custom
  node, built from the checked program its graph stores because the operator value carries no
  scalar-indexed program.
- `Graph.lean`: the graph of registered operators. A node is a graph input, a payload constant at
  a port (value, index or Boolean), or an erased registered operator with one `(node, output)`
  reference per input port. The payload (`Payload`) holds the constant of each constant node and the
  checked program of each custom node by node id (`Payload.const?`, `Payload.custom?`). `Node.outs`
  gives the ports a node produces.
- `Eval.lean`: `Graph.run` and `Graph.eval`, which invoke each operator (`Op.invoke`) and run a
  custom operation by the checked program of its payload entry (`runCustom`), keeping every output
  (`Node.eval_ports`, `Graph.run_size`, `Graph.run_ports`, `evalOp_eq`). It also holds the
  structural check `Graph.check`, which succeeds exactly on the graphs `Graph.WellFormed` describes
  (`Graph.check_ok_iff`). Those are the graphs in which every operator passes `Op.check` and has
  one reference per input port, each naming an earlier node's output at a port its input accepts
  (`Port.accepts`), and every graph output names an output that exists. Compiled evaluation runs the
  `@[csimp]` fast paths of the rules' forwards, so an `Op.permute` node is one strided pass over its
  output (`Tensor.permuteFast_eq`).
- `Builder.lean`: `Graph.Builder` appends inputs, constants and operators and returns the
  reference of every output (`constant` adds a constant with its payload entry). `build` returns a
  graph only after `Graph.check` accepts it (`Graph.Builder.check_build`).
- `Pretty.lean`: `Node.prettyLine` and `Graph.toDot`, the text and GraphViz renderers of operator
  graphs.
- `Tangent.lean`: `Graph.tangent` appends the forward-mode tangent of every node along given
  seeds, by its own rule per operator written with registered operators (no theorem relates these
  rules to `jvp?`), and outputs the graph's outputs followed by their tangents.
- `Program.lean`: typed programs (`IR.Program`). Their steps are graph nodes, conversions
  and registered calls, with one scalar per step for its value, frozen and detached ports, and
  native Boolean, integer and index ports. `Program.Typed.ofDType?` gives the type of a tensor from
  its element type and shape, and the module holds the reference runtime `Program.Machine.reference`
  over the element types of `Scalar.lean` and the structural checks `Program.check` and
  `Payload.check`. The JSON format of the IR, the typed artifact (a typed program with the words of
  its constants), and its loader are in `TorchLean.NN.IR.Artifact` ([below](#the-artifact)).
- `Loaded.lean` and `Table.lean`: `TorchLean.IR.load`, which loads the document of a PyTorch or
  JAX capture, and `TorchLean.IR.table`, the op table that the Python captures read
  ([below](#the-artifact)).
- `Imported.lean`: `Imported`, an operator graph with its payload and the reference of each
  value of the source computation. The typed graphs of `OpGraph.lean` build one
  (`TorchLean.IR.OpGraph.toOp`), and so does the lowering of a recorded model.
- `Record.lean`: `record op args` appends the steps of one registered operator to the typed graph
  of operators (`IR.OpGraph`) under construction. The `Ops` instance of the recording interpreter
  `RecordM` applies it, and `recordForward` records a forward model.
- `Lowering.lean`: `lowerForward` runs a `TorchLean.Program` with `RecordM`, builds the graph of
  registered operators of the recorded typed graph and checks it with `Graph.check`. That graph
  returns the recorded graph's typed value at the output on every input (`lowerForward_eq_ok`),
  and no theorem relates it to the source program.
- `Pointwise.lean`: `Pointwise.fragment? op s`, the fragment of registered operators for the
  pointwise operator `op` at the shape `s` on operands at the scalar. A `Fragment` lists operators
  in evaluation order, each with one argument `Arg` per input port, and its value is the first
  output of the last one. `Fragment.nodes` gives the operator nodes of a fragment on given operand
  references, and `Fragment.usesZero` says whether it reads the zero tensor. The word operators,
  `cbrt` and `nextafter` have no fragment at the scalar. `Pointwise.form? op operand d s` is the
  fragment of a node with operands of element type `operand` and a result of element type `d`, its
  steps at the element of their type (`Pointwise.element?`, `Pointwise.fragmentAt`).
  `Pointwise.typed?` gives the word operators at an integer type of known width
  (`Pointwise.typed?_isSome_of_integerOnly`) and `cbrt` and `nextafter` at a binary format, and
  `Pointwise.logical?` gives the logical operators of Booleans. The op table lists every form
  ([below](#the-artifact)). No theorem mentions `fragment?`.
- `Attr.lean`: the model recorder's pooling-window record (`WindowConfig`), convolution geometry
  (`ConvConfig`) and `Values.get`, output `k` of a list of values. Index tensors flow through the
  operator's typed index ports.
- Execution and planning of operator graphs: `IR.Graph.runBuffers`
  (`TorchLean.NN.Kernel.Graph`) runs an operator graph on resident binary32 buffers, each custom
  operation by its checked program and every other operator by its entry in `LibTorch.Kernel.ofOp`.
  `TorchLean.IR.Graph.plan` and `TorchLean.IR.Graph.accept` (`TorchLean.NN.Backend.Graph`) select a
  backend capsule for every operator node except `detach`, which needs none
  (`TorchLean.IR.Graph.plan_sound`).

## The Artifact

The IR has one JSON format, `torchlean.ir` (`TorchLean.IR.formatName`): the typed artifact
(`TorchLean.IR.Artifact`), a typed program with the words of its constants. Its steps are inputs at
their argument positions, constants, registered operators, each by its codec record (`Op.encode`)
with one `[node, output]` reference per input port, conversions and calls. Beside them it lists
one element type or none per step and the words of each constant step. Decoding is structural
(`Artifact.ofJson`), and `Artifact.ofJson_toJson` proves that decoding the JSON of any artifact
returns it, a statement about JSON values, not about text. `Artifact.load` checks the program
(`Program.check`), decodes each constant from its words at the type its step declares
(`Program.Typed.decode`), takes the custom operations and calls from a registry (`Program.Env`)
and checks the payload (`Payload.check`). `Artifact.load_ok` proves that an artifact it accepts
gives back its own program, which passes `Program.check`, with a payload that passes
`Payload.check`.

The PyTorch and JAX captures write one capture document per computation, which
`TorchLean.NN.IR.Loaded` reads: the artifact, the arguments and reference outputs by element type,
shape and complete interchange words, and the name of the capture that wrote it. Its
`"declared_types"` give, for each value the capture lowered, the step output that holds it and the
element type and shape that the framework's own metadata records for it. `TorchLean.IR.load`
loads the artifact under the reference runtime with no registered custom operation or call, and
`TorchLean.IR.load_ok` proves that the program it returns passes `Program.check .reference` and its
payload `Payload.check`. `load` then requires the program to give every declared value its declared
element type and shape (`Program.checkDeclared`), and `TorchLean.IR.load_declared` proves that a
loaded program does. So a lowering that changes a value's type fails the load, with a message
`IR types: ...` that names the value, the type the IR computes and the declared one. A document
that names a frontend must declare these types, and a raw IR document declares none.
`TorchLean.JAX.load` first checks that the frontend is `jaxpr` and has the same guarantee
(`TorchLean.JAX.load_ok`). `TorchLean.IR.loadArguments` reads the arguments under a runtime, and
`TorchLean.IR.words` gives the element type, shape and words of a value back.

The PyTorch capture in `python/torchlean/` and the Jaxpr capture (`python/torchlean/jaxpr.py`) lower
their graphs to these steps (`_artifact.lower` in `python/torchlean/_artifact.py`), and the ONNX
adapter (`python/torchlean/onnx.py`) writes typed artifacts too. All three read the facts they need
from Lean's op table, `TorchLean.IR.table` in `TorchLean.NN.IR.Table`, which
`verify ir --op-table` and the Python binding `op_table` return. The table holds the format name,
the wire version, the JSON of each element type, the codec's codes and the typed forms of the
pointwise operators (`Pointwise.form?`). Lean checks every step when it loads the artifact.
Dimensions are natural numbers in a record, and `Op.decode` checks each against its operand's
rank. The PyTorch capture adds the rank to a negative dimension once and refuses one outside
`[-rank, rank)`.

For two operators we state the forward value against the tensor operation on a row-major view of
the operand. `Tensor.mapLeading_eq_reshape` states a map over leading dimensions as the same map
over their row-major fold into one batch axis, and over a batch axis `TorchLean.Conv.grouped`, with
the weights in PyTorch's packed layout (`TorchLean.Conv.packedKernel`), is `Op.conv`'s forward value
(`TorchLean.Conv.mapLeading_grouped_eq_conv`). `Tensor.layerNorm_last_eq_reshape` states the layer
norm over the last axis as the layer norm of the rows of its matrix view, the rule of
`Op.layerNorm`.

## One Shape Rule Per Operation

A custom operation declares its input shapes in operand order and its output shape
(`Op.custom name inputs output`). Its `TorchLean.Kernel.Program` body lives in the payload
(`Payload.custom?`). `Graph.check` checks the declared input shapes against the referenced outputs'
ports (`Port.accepts`), and evaluation (`runCustom`) checks the actual input tensors against the
declared shapes (`TensorPack.ofErased`) and runs the body with bounded reads
(`TorchLean.Kernel.Program.eval`). The body's source-correspondence proof
(`TorchLean.Kernel.Program.eval_eq_reference`) also covers failed reads. `TorchLean.IR.load`
registers no custom program, so it refuses an artifact with a custom step (`Artifact.load`). The
checked program gives no derivative rule, range rule or export translation, so a consumer that needs
one must report the custom operation as unsupported.

A call of a typed program declares the types of its arguments and its result (`Program.Node.call`),
which `Program.check` compares with the referenced outputs, and its implementation is registered
separately (`Program.Env`). A conversion is a step of its own (`Program.Node.cast`), run entry by
entry with the runtime's conversion (`Program.Machine.cast`).

The data-movement operators need tensor storage, not scalar arithmetic. In a typed program,
Boolean and fixed-width integer ports carry their elements natively beside FloatLib's binary
formats. An operator whose ports are all Boolean, integer or index ports has a forward at every
scalar type with `Storage` (`Op.storageForward?`): its rule at `Float` on the same tensors, so a
step without a scalar runs it without a `Context`. An operator with a value, frozen or detached
port has none, so at the scalar a data-movement operator runs by the runtime's arithmetic at that
scalar and fails where the runtime has none.

A registered operator's ports are its shape rule: `Op.decode` checks the wire attributes and
constructs an operator with the resulting ports. `Program.check` accepts a step only when each
reference names an earlier output at a port its input accepts. `Op.check` covers attributes that
are not enforced by the constructor's type, such as a pooling window or distinct reduction axes.

| Operation | Shape check |
| --- | --- |
| `broadcast` | `Op.decode` constructs `Shape.BroadcastDims` |
| `permute` | `TorchLean.Shape.perm?` |
| `slice` | `Op.decode` constructs `Shape.SliceValid`; the model recorder uses `IR.sliceDims` |
| `pad` | One low, high and interior padding value per axis; `TorchLean.Shape.pad` gives the output shape |
| `cat` | Input extents off the concatenation axis agree |
| `gather`, `scatter` | `Shape.GatherValid` for gather and distinct indexed axes for scatter; typed index ports carry the indices |
| `reduce`, `mean` | Checked reduction axes and `TorchLean.Shape.reduce`; extrema require nonempty fibers |
| `contract` | `Shape.ContractValid` and `TorchLean.Shape.contract` |
| `matmul`, `scaledMatmul` | Checked vector promotion and batch broadcasting; bound passes use `IR.inferMatmulDims` |
| `maxPool`, `avgPool` | Checked window geometry; the model recorder uses `IR.planPool` |
| `conv` | Checked layouts and geometry; the model recorder uses `IR.inferConvConfigOutShape` |
| `topk` | A valid axis and `k` no larger than its extent |
| dimensions of `softmax`, `logSoftmax`, `layerNorm`, `batchNorm`, `topKMask`, `cumsum`, `cumprod`, `cummax`, `cummin`, `sort`, `flip`, `indices` | Checked `Fin rank` axes |

## How Core ATen Operators Enter The IR

A core ATen operator enters a typed artifact as steps of registered operators of `TorchLean.Op`,
each by its codec record (`Op.encode`), and never as a name registered behind a call. The
pointwise operators form one family with one shape rule, `TorchLean.PointwiseOp`, which carries the
operator as data. It holds `neg`, `rsqrt`, `erf`, `gelu` with PyTorch's `approximate` mode, `div`
with its `rounding_mode`, `pow` and `clamp_min`, the bound where the operand is below it (the
PyTorch capture writes `Op.max` for `torch.clamp_min`). Further elementwise operators of core ATen
and `jax.lax` are `sign`, `square`, the integral roundings `floor`, `ceil`, `round` and `trunc`,
`frac`, `fmod`, `remainder`, `exp2`, `expm1`, `log2`, `log10`, `log1p`, `tan`, the inverse
trigonometric and hyperbolic functions, `sinh`, `cosh`, `isnan`, `isinf` and `isfinite`. The family
also has the comparisons, the logical operators, `maximum`, `minimum`, `fmax`, `fmin`, `copysign`,
`hypot`, `atan2`, `where`, `clamp` with tensor bounds, `lerp`, `addcmul`, `addcdiv` and
`nan_to_num`. IEEE's square root is `Op.sqrt .raw`, and the captures write `torch.sqrt`, `lax.sqrt`
and ONNX `Sqrt` as `Op.sqrt` at `.raw`.

The operands and the result of a pointwise operator share one shape, and broadcasting is a step of
its own, `Op.broadcast`, as for `add`. A pointwise node is the steps of its typed form
(`Pointwise.form?`), which the op table lists by the operator's name and keyword. At the scalar
the form is the fragment of `Pointwise.fragment?`: negation, true division and the tanh GELU are
the single steps `Op.neg`, `Op.div .none` and `Op.gelu .tanh`. Their rules are the scalar's `-`
and `/` and `TorchLean.Activation.geluTanh`, entry by entry. The word operators, `cbrt` and
`nextafter` have forms only at their element types (`Pointwise.typed?`). The bound passes give no
box to an operator they have no rule for (`TorchLean.CROWN.Op.runIBP`), and backend planning
fails at an operator without an admissible capsule (`TorchLean.IR.Graph.plan`).

Adding an operator to the family adds these pieces and nothing else:

- Constructor, attributes, arity and diagnostic name: `TorchLean.PointwiseOp` in
  `TorchLean/Tensor/Op/Pointwise.lean`.
- Meaning, the rules of the registered operators it lowers to, at each scalar type: `TorchLean.Op`
  in `TorchLean/Tensor/Op/Table.lean` and its rules (`Op.rule?`) in
  `TorchLean/Tensor/Op/Table/Rule.lean`, from the fields of `Context` or a capability of the
  scalar (`Op.Capability`: `Rounding`, `InverseTrig`, `Erf`, `Cbrt`, `NextAfter`).
- Lowering, the registered operators of its fragment and its typed forms: `Pointwise.fragment?` and
  `Pointwise.form?` in `TorchLean/NN/IR/Pointwise.lean`.
- Integer meaning, where PyTorch keeps integer results: `TorchLean.Word` in
  `TorchLean/Tensor/Word.lean`, the rule of the registered operator at an integer element.
- Name and attribute field in the op table: `TorchLean.PointwiseOp.name`,
  `TorchLean.PointwiseOp.keyword`.
- PyTorch and JAX capture rules: the PyTorch capture in `python/torchlean/` and
  `python/torchlean/jaxpr.py`.
- PyTorch function for the exporter, by operator (`torch.<name>` or a formula of its own):
  `TorchLean.NN.PyTorch.Export.IR`.

The rules of negation, true division, `clamp_min`, the tanh GELU (`TorchLean.Activation.geluTanh`)
and the operators defined from the fields of `Context` in `TorchLean/Tensor/Elementwise.lean` exist
at every scalar type. Those of the integral roundings, `frac`, the rounded divisions and the
remainders read `Rounding`, those of the inverse trigonometric functions `InverseTrig`, and those of
`erf` and the exact GELU `Erf`.

A registered-operator table reached through a call would not satisfy one shape rule per operation:
its meaning is supplied per runtime by name (`Program.Env`). A pointwise operator therefore grows
only `TorchLean.PointwiseOp` and the tables above. Operators whose operands or results are indices,
integers or Booleans next to a scalar, such as `argmax`, the index outputs of `sort`, `cummax` and
`cummin`, and `embedding`, are registered operators of `TorchLean.Op` with index, integer or Boolean
ports. `Program.check` also requires such a step to declare a scalar exactly when it has value ports
(`Program.checkScalar`). The lowering (`python/torchlean/_artifact.py`) writes `Op.decodeIndex`
before an index operand and `Op.encodeIndex` after an index result, and Boolean and integer ports
carry their elements natively (`BitVec width` at an integer type). The step succeeds exactly when
invoking the operator on its inputs does (`TorchLean.IR.Program.step_op_arithmetic`,
`step_op_native`), and `Op.invoke_eq_ok` states that a successful invocation returns the rule's
forward values. A `decodeIndex` step outputs the index tuples its words hold and fails on a
component outside its bound, and an `encodeIndex` step outputs the words of its tuples and fails on
a component the integer type cannot hold (`TorchLean.IR.Program.step_decodeIndex`,
`step_encodeIndex`). These are statements about one step. A program written directly in the IR can
also return index tuples. The `verify` report writes such a result with `"element_type": "index"`,
its port and the components of its tuples (`TorchLean.Capture.resultJson`), and a comparison with a
reference tensor fails, since a tensor holds no index tuples.

A core ATen operator belongs to the family when its operands and result share one shape. In a
graph over one scalar type, a comparison, a predicate or a logical operator returns `1` or `0` in
that type, and `where` and the logical operators read a nonzero value, a NaN included, as true.
A typed form may return Booleans instead (`Pointwise.returnsBool`, `Pointwise.form?`): a
comparison, a predicate or a logical operator then takes operands of one type and returns `bool`,
and `where` may take a `bool` condition with branches of the result type. Every other form has
operands of the result's type. Boolean ports carry `Bool` itself.

### Meanings by scalar type

The meaning of an operator at a scalar type is the rule of its registered operator there
(`TorchLean.Op`). A rule that reads a capability the type lacks is absent (`Op.Capability`), and
invoking the operator fails with no rule (`Op.invoke`) rather than substituting an approximation.
At the reals the rules are Mathlib's total functions: at a pole or outside the real domain, where
PyTorch's IEEE result is an infinity or a NaN in every format, they give Mathlib's value there,
such as `x / 0 = 0`. A real result that a binary format rounds to an infinity, such as
`1e30 / 3e-10` in binary32, keeps its real value.

| Operator | Exact reals | FloatLib binary | Host `Float`, `Float32` | Reference integers |
| --- | --- | --- | --- | --- |
| `neg` | `-x` | sign flip, including NaN | sign flip | two's complement |
| `div` | `x / y`, the scalar's `/` (`x / 0 = 0`) | correctly rounded division | IEEE division | unsupported: PyTorch promotes |
| `div` trunc | `x / y` rounded toward zero (`0` at `y = 0`) | the `div` floor schedule on the magnitudes, with the sign of `x / y` | same | truncating division, an error at a zero divisor |
| `div` floor | `⌊x / y⌋` (`0` at `y = 0`) | `TorchLean.divFloorSchedule` over the format's rounded operations | same | flooring division, an error at a zero divisor |
| `pow` | `Real.rpow`, with Mathlib's value at a negative base | FloatLib `Model.pow` | C `pow` | PyTorch's `powi`, wrapping |
| `rsqrt` | `1 / √x` (`0` at `x ≤ 0`) | `1 / sqrt x`, two correct roundings, so binary64's `rsqrt 3` is one ulp above the correctly rounded `rSqrt 3` | same | unsupported |
| `erf` | `TorchLean.erf` | unsupported | unsupported | unsupported |
| `gelu` none | `x Φ(x)` | unsupported | unsupported | unsupported |
| `gelu` tanh | `TorchLean.Activation.geluTanh` | `Activation.geluTanh`, FloatLib `tanh` | `Activation.geluTanh` | unsupported |
| `atan`, `asin`, `acos` | `Real.arctan`, `arcsin`, `arccos`, the last two clamped outside `[-1, 1]` | correctly rounded | C `atan`, `asin`, `acos` | unsupported |
| `atan2` | `InverseTrig.atan2 y x` | correctly rounded, with C99's zeros and infinities | C `atan2` | unsupported |
| `fmod` | `x - y * trunc (x / y)` (`x` at `y = 0`) | exact, from FloatLib's IEEE remainder | C `fmod` | unsupported |
| `remainder` | `x - y * ⌊x / y⌋` (`x` at `y = 0`) | `fmod`, then `+ y` where `fmod` is nonzero with the other sign. A zero has the sign of `y` | same, from C `fmod` | the divisor's sign (`Int.fmod`), an error at a zero divisor |

Mathlib has no error function, so `TorchLean.erf` is defined in `TorchLean/Tensor/Erf.lean`
with its derivative, oddness, strict monotonicity and limits `±1`. `Context.geluErf_real` gives the
exact GELU, the function of the rule of `Op.gelu .none` at the reals, as `x (1 + erf (x / √2)) / 2`,
and `TorchLean.cdf_gaussianReal` in `TorchLean/NN/NormalCDF.lean` proves that
`(1 + erf (x / √2)) / 2` is Mathlib's distribution function of the standard normal law,
`ProbabilityTheory.cdf (gaussianReal 0 1)`. `Context.geluErf_eq_mul_cdf` therefore states the exact
GELU as `x Φ(x)`, and `TorchLean.Tensor.gelu_eq_map_mul_cdf` states it for `Tensor.gelu`.
`TorchLean.Activation.geluTanh_real` gives the tanh GELU as
`x (1 + tanh (√(2 / π) (x + 0.044715 x³))) / 2`, the approximation with its own constants.

Binary flooring division runs `divFloorSchedule`, and `divFloorSchedule_real` proves that
the same schedule computes `⌊x / y⌋` in exact arithmetic. `remainderSchedule_real` proves that
the remainder's schedule computes `x - y ⌊x / y⌋`. A zero remainder has the divisor's sign, as
Python's `%` and `numpy.remainder` give it and as `torch.remainder` documents ("the same sign as the
divisor"), while eager PyTorch 2.13 and `jnp.remainder` return the dividend's sign there.

FloatLib binary's inverse trigonometric functions are those of
`TorchLean.Tensor.ConfiguredBinary.InverseTrigonometric`: FloatLib's exact comparisons of
the real function with rational boundaries decide its nearest-even rounding.
`TorchLean.Verify.RuntimeApprox.InverseTrigonometric` proves the results finite and correctly
rounded at finite inputs in the domain, for every IEEE descriptor, with their special values. On
4000 random binary32 and binary64 inputs each, eager PyTorch 2.13 on x86-64 with AVX-512 is within 1
ulp of a 200-bit mpmath reference rounded to the format. PyTorch's asin is off by 1 ulp on 226
and 339 inputs, acos on 340 and 371, atan on 0 and 2, and atan2 on 94 and 99. On the regression's
inputs, XLA's CPU code (jax 0.12) is within 2, 2, 3 and 1 binary32 ulps of the IR. Binary16 and
bfloat16 compute in binary32 and round once, in the IR as in both frameworks: the IR's result is
`R16(R32(f(x)))`, and a framework's is `R16` of its own binary32 value. In every configured format
the same functions, rounded once in that format, are its `InverseTrig` instance, which `Op.atan`,
`Op.asin`, `Op.acos` and `Op.atan2` compute, though a captured binary16 or bfloat16 graph computes
in binary32 first.

The IR's binary `pow` is FloatLib's `Model.pow`. For an IEEE format, a finite base and a finite
exponent with an integer value, every finite result is the nearest-even rounding of the real power
(`toReal_pow_of_eq_intCast`, which assumes the result finite). For a non-integral exponent and a
finite positive base, `Model.pow` runs a certified enclosure search, at most eight attempts after
the overflow and underflow screens. Every result the search accepts is finite and the nearest-even
rounding of the real power (`pow_spec_of_certified`). A range failure gives the overflow,
and any other failure falls back to `exp (y * log x)`, with no error bound stated. An exact
power that lies on a rounding boundary generally does not settle. A finite negative base with
a non-integral exponent gives NaN, and the branches for zeros, infinities, NaNs and the exponents
`±0` and `1` follow IEEE 754-2019 9.2.1 (`pow (x, ±0) = 1` for every `x` but a signaling NaN,
`pow (-0, 0.5) = +0`, `pow (-∞, 0.5) = +∞`, `pow (-0, -0.5) = +∞`).

On 2000 random non-integral binary32 and 2000 binary64 powers and on 14 chosen pairs near overflow,
underflow and exact results (`0.001 ^ 10.5`, `0.5 ^ 149.5`, `2.25 ^ 0.5`), every IR result equals
the power mpmath computes at 300 or 400 bits, rounded to the format. That agreement is numerical
evidence rather than a certificate. On 6000 integral powers the IR equals the exact power rounded.
PyTorch's `torch.pow` differs from that reference by 1 ulp on 25 and 26 of the random ones, and
`torch.pow(x, torch.tensor(3.0))` and `torch.pow(x, torch.tensor(4.0))`, SLEEF's `powf`, on 3 and 7
of 256 binary32 values.

`TorchLean.Activation.geluTanh` computes `β = √(2 / π)` in the format, rounds `κ = 0.044715` once,
and then rounds each operation of PyTorch's kernel in its order, `κ x³` and `x + κ x³` separately.
PyTorch rounds `√(2 / π)` once to binary64, its `kBeta`, and that value to the computation type. In
binary64 the two `β` are the same word, and the IR's words are those of the same order with
PyTorch's constants. In binary32 the IR's `β` is `0x3f4c4229`, one ulp below PyTorch's `0x3f4c422a`,
so binary32, bfloat16 and binary16 results can differ from PyTorch's in the last bits.

One PyTorch binary contains each CPU kernel at three capability levels, DEFAULT, AVX2 and AVX-512,
and at run time selects the highest level the processor supports, or the one that
`ATEN_CPU_CAPABILITY` names. On x86-64 (a Xeon Platinum 8275CL), both loops fuse the sum `x + κ x³`
at the AVX2 and AVX-512 levels and round `κ x³` and the sum separately at DEFAULT, the order the IR
follows. On 8192 normal samples with standard deviation 4, each loop at the AVX2 and AVX-512 levels
equals the formula with the fused sum on all 8192 in binary32 and binary64. Against the formula with
the unfused sum, the vectorized loop agrees on 8153 and 8161 samples and the scalar loop on 8156
and 8157. At DEFAULT both loops equal the unfused formula on all 8192. These comparisons use
PyTorch's constants. PyTorch on aarch64, on Apple processors and on CUDA was not measured.

At the AVX2 and AVX-512 levels the loops differ only in `tanh`: the vectorized loop calls SLEEF's
and the scalar loop the C library's. At DEFAULT the vectorized loop applies the C library's `tanh`
to each element. The IR calls FloatLib's `tanh`. On another 8192 normal samples with standard
deviation 4 it returns the words of both DEFAULT loops on all but 1014 binary32 and 531 binary64
samples. It differs from the AVX-512 vectorized loop on 1030 and 53 and from the AVX-512 scalar
loop on 1017 and 564.

The figures below compare the IR with the loops at the DEFAULT and AVX-512 levels and with the exact
value of the tanh approximation, `x (1 + tanh (√(2 / π) (x + 0.044715 x³))) / 2`, computed with
mpmath and rounded once. The comparisons use 2^18 normal samples with standard deviation 4 and 2^18
evenly spaced inputs in `[-12, -2]` in binary32 and binary64. The bfloat16 and binary16 comparisons
at the end of this section are of words, the IR's against PyTorch's, on every finite input. At
DEFAULT the two loops return the same words on every sample and every finite bfloat16 and binary16
input. For `x ≥ -1` the IR and PyTorch's AVX-512 vectorized loop are within 3 ulps of the exact
result in binary32 and binary64, while the DEFAULT loops and the AVX-512 scalar loop are within
4 ulps in both.

For negative `x`, `1 + tanh(inner)` subtracts two numbers close to 1, so an error of a fraction of
an ulp in `tanh(inner)` becomes a large error relative to the small result. The cancellation, not
the IR, makes the ulp counts grow. Between `-3` and `-1` the IR is up to 201 binary32 ulps and 200
binary64 ulps from the exact result. The DEFAULT loops are up to 200 and 200, the AVX-512 vectorized
loop up to 198 and 201 and the AVX-512 scalar loop up to 198 and 200. Below `-3` the largest
distances, millions of binary32 ulps and more in binary64, occur next to the `-0` cutoffs described
below. Excluding inputs where one result is zero and the other is not, the IR differs from the
DEFAULT loops by at most 297777 binary32 ulps (at `x = -4.54372`) and 1797 binary64 ulps (at
`x = -3.50971`). With the same exclusion, it differs from each AVX-512 loop by up to 68856 binary32
ulps (at `x = -4.20269`) and 16200 binary64 ulps (at `x = -3.95496`). There the absolute error is
the meaningful measure. Over every measured `x < -1` the IR's distance from the once-rounded exact
value is at most `0.847 |x| u` in binary32 and `0.782 |x| u` in binary64, where `u` is `2^-24` and
`2^-53`. The AVX-512 vectorized loop's distance is at most `0.772 |x| u` and `0.782 |x| u`, while
the DEFAULT loops' and the AVX-512 scalar loop's reach `1.083 |x| u` and `1.211 |x| u`. These are
measured maxima, not proved bounds.

Once `tanh(inner)` rounds to `-1`, the result is `-0`. In the negative tail, PyTorch's DEFAULT loops
return `-0` from `x = -5.1578555` in binary32 and `x = -7.200040557303352` in binary64 on, its
AVX-512 scalar loop from `-5.157855` and `-7.200040557303352` and its AVX-512 vectorized loop from
`-5.061381` and `-7.145065874418114`. The exact result rounds to `-0` only from `-10.770868`
and `-21.547075410436022` on. PyTorch's binary32 cutoffs hold for every input in `[-12, -4]`, and
its binary64 cutoffs, found by bisection, hold on the `2^20` inputs past each. The exact cutoffs are
bisections of the mpmath reference at 50 digits. The IR's cutoffs are the DEFAULT loops': it returns
`-0` exactly where they do on the `10^5` consecutive inputs on each side of each cutoff and on every
sample. At `x = -∞` the IR returns `-0`, the limit, and the loops a NaN, since they compute
`x (1 + tanh(inner))` as `-∞ · 0`.
Between the cutoffs of the IR and the AVX-512 vectorized loop, that loop returns `-0` and the IR a
nonzero result, nearer the exact value at every one of the 3718 binary32 and 1712 binary64 samples
in that band. Those are all the zero-versus-nonzero differences between the two.

In bfloat16 and binary16 the IR computes in binary32 and rounds the result once, as the capture
writes it and PyTorch computes it. So computed, in bfloat16 it returns the words of both DEFAULT
loops and of the AVX-512 scalar loop for every finite input, and in binary16 for every finite input
except `-0.249756`, `-1.99902` and `-2.01367`, where it differs by 1 ulp. It returns the AVX-512
vectorized loop's words except for 4 bfloat16 and 25 binary16 inputs in `[-5.15625, -5.0625]`, where
that loop returns `-0`, and the binary16 inputs `-1.99902` and `-2.01367`, 1 ulp away.

### Frontend lowering

Each captured operator means its mathematical definition, IEEE 754-2019's operation or the
documented API, and the capture writes that meaning. Where PyTorch's CPU kernels compute something
else, the capture does not copy them. The differences are listed here. A tanh `gelu` node computes
in the DEFAULT level's order. Binary16 and bfloat16 inputs of `rsqrt`, `erf`, `gelu`, `pow`, the
exponentials, logarithms, `tan`, the hyperbolic functions and their inverses, `asin`, `acos`,
`atan`, `atan2`, `hypot` and `remainder` are widened to binary32, where PyTorch computes them too.
The binary32 result is rounded once, `R16(f32(x))` for the IR's binary32 operation `f32`, and
`rsqrt`, for one, gives `R16(R32(1 / R32(√x)))`.

A conversion is one IR cast, which rounds the exact value once (IEEE 754-2019 5.4.1 and 5.4.2), in
both frontends. To an integer type a floating value truncates toward zero, as PyTorch's conversion
and `convert_element_type`'s documented `static_cast` do. A truncated value the type cannot hold
is an execution error (`c10::convert` takes a floating value to uint8 through int64 and wraps it).
An integer value converted to an integer type wraps modulo `2 ^ width`. PyTorch's `c10::convert` and
XLA's `convert_element_type` on the CPU make binary16 and bfloat16 values from binary32, so they
round an integer or binary64 value twice. For example, int32 `16842753` becomes bfloat16 `16777216`
there and `16908288` in the IR. Binary64 `1 + 2^-8 + 2^-40` (bits `0x3FF0100000001000`) becomes
binary32 `0x3F808000` there. That is the bfloat16 midpoint `1 + 2^-8`, and ties to even give
`0x3F80` (`1.0`). The IR rounds the exact value once, to `0x3F81` (`1.0078125`), the correctly
rounded result. PyTorch 2.13, ml_dtypes 0.5.4 and XLA's CPU code all give `0x3F80`, so a comparison
rejects such a conversion, by design. The two disagree on 478 of 6008 binary64 test values
converted to bfloat16 and 528 converted to binary16. Integers convert to binary16 alike both ways,
since binary32 holds every integer below binary16's overflow threshold exactly, and Boolean values
are exact either way. The PyTorch capture applies the rule to `Tensor.to` and to the promotions of
`add`, `sub`, `mul`, `maximum` and `minimum`, the JAX importer to `convert_element_type`.

A Python scalar becomes a constant from the value PyTorch wraps it in, a bool as a Boolean, an int
as int64 (uint64 from `2^63`) and a float as binary64, rounded once to the constant's dtype. The
fill values of `full`, `full_like`, `new_full`, `fill` and `scalar_tensor` that the captured graph
records are rounded once in the same way. FX evaluates an input-independent creation while tracing,
so under FX such a constant is the tensor PyTorch made, with its rounding through binary32. An
int beyond `2^53` rounds once to binary32: `2^60 + 2^36 + 1` becomes `2^60 + 2^37`, where rounding
through binary64 would give `2^60`. A scalar operand of `add` or `sub` is converted to the result
dtype and the operation runs there.

`clamp_min(x, min)` is `max (x, min)`, the IR's `Op.max`, IEEE 754-2019 maximum, and
`clamp_max(x, max)` is `min (x, max)`, `Op.min`, IEEE minimum. The bound is converted to the
result dtype (an int32 tensor clamped at `0.5` gives binary32), for a scalar and for a tensor bound
alike. `clamp` and `clip` with one bound are the same, and so is the clamp in `cdist`'s
matrix-product expansion. A NaN operand or bound gives a NaN, and `clamp_min(-0.0, 0)` is `+0.0`.
PyTorch's kernel for a scalar bound returns the bound only where the tensor is below it and keeps
`-0.0` there, in binary32, binary64, bfloat16 and binary16 and in both of its loops. With a tensor
bound, its vectorized loop returns the second operand at a tie of `-0` and `+0`. The IR's `relu` is
TorchLean's ReLU, which returns `+0.0` at `-0.0`, where eager PyTorch's `relu` returns `-0.0`, so a
sample comparison of a ReLU at a negative zero fails.

Binary16 and bfloat16 products and quotients compute in binary32 and round once to the result dtype
when an operand supplies one element to the whole output. Such an operand is a Python scalar, a
one-element tensor, or a view of the result dtype with stride 0 along every dimension longer than
one. A product takes this path for such an operand, on either side, that is not of the result dtype,
and a quotient for such a dividend or divisor. The operand keeps its own value in binary32, so a
binary32 rank-zero tensor or `0.1` is not rounded to bfloat16 first, and an operand that does not
supply one element is converted to the result dtype. PyTorch takes this path only for the right
operand (TensorIterator's `is_scalar(2)`) and converts a left scalar to the tensor's dtype. Its
bfloat16 `torch.mul(0.1, x)` is therefore `bfloat16(0.1) * x`, which differs from its `0.1 * x` on
845 of 4096 values (1323 in binary16), and its `torch.div(0.1, x)` divides bfloat16 `0.1`. In the
IR, `torch.mul(s, x)` and `x * s` are the same graph. The binary32 value is itself a rounding where
the operand is wider: the int `16842753` is binary32 `16842752`, so `torch.mul(16842753, x)` with
bfloat16 ones gives `16777216`, where converting the int once gives `16908288`.

Other divisions and products convert both operands to the result dtype and round once there. `floor`
is the flooring schedule above, in which the exact remainder corrects the rounded quotient, as
Python's `//` does (`1 // 0.1` is `9`). The IR's `trunc` mode runs that schedule on the magnitudes
with the sign of the quotient (`9` at `(1, 0.1)`). PyTorch truncates the rounded quotient (`10`), so
the capture lowers a floating `trunc` division as `div` then `trunc`; an integer one is the IR's
`trunc` mode, which is exact there. Python's
`s / x` is true division, the IR's `div` of the converted operands, which `torch.div(s, x)` computes
too. PyTorch's `Tensor.__rtruediv__` computes `reciprocal(x) * s`, which differs from the correctly
rounded quotient on 26531 of 100000 binary32 values `0.1 / x` (26167 in binary64). FX records
`s / x` as `operator.truediv(s, x)`. `torch.export` records the reciprocal and the product, both
tagged in `meta["torch_fn"]` with the one `__rdiv__` call, and the capture lowers that pair as the
division when the reciprocal has no other user. An explicit `x.reciprocal() * s` has its own tags
and stays a reciprocal and a product, and so does a pair whose reciprocal is used elsewhere or a
graph without the tags.

`pow` of a floating result with a Python scalar exponent is one IR `pow`, FloatLib's power above, of
the tensor and the exponent's own value. Both are in the computation type (binary32 for binary16,
bfloat16 and binary32, binary64 for binary64), and the power is rounded to the result dtype. It
is therefore correctly rounded in binary32 and binary64 where FloatLib's power is, and in binary16
and bfloat16 it is that binary32 power rounded again. Every exponent, `0` and `1` included, is
that one node, so a NaN base gives a NaN for the exponent `1`, and a signaling NaN gives a quiet
NaN at `0`, as `pow` does. Binary16 and bfloat16 operands are widened first, which quiets a
signaling NaN. A binary32 exponent is the Python float rounded once to binary32 (`x ** 2.007` is
`pow (x, 2.00699996948...)`), and an exponent whose rounding there would change whether it is an
integer, or which integer, is refused (Python's `1 + 2^-25` would become `1`). A Python scalar
base of `pow` keeps its value in the computation type in the same way. An integer result is one IR
integer power, which gives ones for the exponent `0` and the tensor for `1`.

PyTorch fills ones for the exponent `0` and copies for `1`. In binary32, binary64 and bfloat16 its
PowKernel takes the square root, reciprocal square root and reciprocal for `0.5`, `-0.5` and `-1`,
which differ from `pow` at `-0` and `-∞`, and multiplies for the exponents `2`, `3` and `-2`. It
rounds a binary16 or bfloat16 exponent to the tensor's dtype before computing: bfloat16
`1000 ** 2.007` is `1000 ** 2 = 999424` there and `1048576`, the correctly rounded power, in the IR.
Binary16 calls its own `pow` with the rounded exponent. PyTorch's binary32 `x ** 3` differs from the
correctly rounded cube on 1058 of 4096 values, `x ** -2` on 1123 and `x ** -0.5` on 1112. PyTorch
rounds a scalar base to the tensor's dtype and fills ones for the base `1`.

PyTorch does not compute every element with the same code. Its vectorized loop takes two vectors at
a time, and a scalar loop covers the remaining elements of each contiguous run and every element of
a non-contiguous input. A contiguous tensor shorter than two vectors therefore runs entirely in
the scalar loop. The vector width is that of the capability level the kernel runs at. On an AVX-512
CPU (a Xeon Platinum 8275CL with PyTorch 2.13), `pow` and `gelu` run at the AVX-512 level. At
that level the scalar loop computes all of a contiguous tensor with fewer than 32 binary32 or 16
binary64 elements, or fewer than 64 bfloat16 or binary16 elements for `gelu` and bfloat16 elements
for `x ** -0.5`. `torch.rsqrt` is registered with `REGISTER_DISPATCH`, which compiles no AVX-512
kernel for it, so it runs at the AVX2 level and its threshold is 32 bfloat16 or binary16 elements.
`x ** -0.5` reaches the same reciprocal square root kernel from the AVX-512 kernel of `pow`. The
scalar loop evaluates binary32 `pow` with a scalar exponent in binary64 with the unrounded exponent.
It rounds the square root to the reduced format before dividing in bfloat16 and binary16 `rsqrt`,
and takes `pow` from the C library instead of SLEEF. For these operators the eager result, and a
sample comparison with it, can depend on tensor length and layout.

PyTorch's CPU `cumsum` and `cumprod` accumulate in `at::acc_type` (binary64 for binary32 and
binary64, binary32 for binary16 and bfloat16) and round each prefix to the operand's format. The
capture writes a cast to that type, the IR scan and a cast back, which return eager PyTorch's words
on binary32, binary16 and bfloat16 samples with signed zeros and NaN. `cummax` and `cummin` take a
later tie and a later NaN, as the IR's do. `max.dim` and `min.dim` keep the first extremum and the
first NaN, and the capture writes them as the IR's stable `topk` with `k = 1`, between two exact
negations for `min.dim`. `torch.sort` is imported with `stable=True` only, because ATen leaves the
order of equal entries, `-0` and `+0` among them, unspecified. `torch.topk` is the stable `topk`
where ATen's result is uniquely determined, and for `k > 0` execution fails where two adjacent
entries of a fiber's first `min(k + 1, n)` in the requested order tie.

`leaky_relu` computes `x * slope` in binary32 for the reduced formats, with the slope rounded there.
Rounding is monotone, so the result is `maximum(x, x * slope)` for `0 < slope <= 1` and
`minimum(x, x * slope)` above one, at every signed zero, infinity and NaN. A slope that is not
positive is refused: at `+inf` the product with `0` is NaN. Constant-index `gather` and `scatter`
nodes hold the complete coordinates of every entry. A repeated scatter target is refused, since
PyTorch leaves the stored value, and for `scatter_add` the order of the sum, unspecified.

The weight, bias, eps and running statistics of a `layer_norm` or `batch_norm` node are constant
steps, interchange words of the node's dtype that `Artifact.load` decodes. An absent weight or bias
is one or zero, and eps is rounded to the computing dtype. Binary16 and bfloat16 normalizations
are computed in binary32, PyTorch's opmath type, between two casts. PyTorch's CPU kernels compute
`(x * rstd - mean * rstd) * gamma + beta` from Welford moments and
`x * alpha + (beta - mean * alpha)`, while the IR computes
`(x - mean) / sqrt (var + eps) * gamma + beta` with two-pass moments, so the words agree where every
step is exact and otherwise differ in the last bits. A negative eps or running variance is refused,
since PyTorch takes the square root of `var + eps` where the IR clamps the variance at zero. A mean
over an empty fiber with a nonempty result is refused (PyTorch returns NaN, the IR mean divides by
one). The variance there is `0 / 0`, NaN.

Comparisons, `isnan`, `isinf`, `isfinite` and the logical operators are Boolean nodes. Comparisons
take floating or integer operands of one dtype, with a Python scalar promoted as PyTorch's
`result_type` promotes it at capture (an integer tensor and a float compare in the default floating
dtype, binary32 unless `torch.set_default_dtype` changes it). Predicates take floating operands, and
the logical operators take such operands or Boolean ones. `where` and `masked_fill` select floating,
integer or Boolean values with a Boolean condition. Bitwise operators of integers are the IR's
`bitwise_and`, `bitwise_or`, `bitwise_xor` and `bitwise_not` of the words. `<<` is
`bitwise_left_shift`, and `>>` is `shift_right_arithmetic` of a signed type and
`shift_right_logical` of an unsigned one. A shift amount is its word read unsigned, so an amount of
the width or more, a negative one among them, shifts every bit out. PyTorch documents such amounts
as undefined, and its CPU kernels compute these values. Integer `add`, `sub` and `mul` wrap modulo
`2 ^ width`, as ATen's integer kernels do.

`scaled_dot_product_attention` of the default graph is its docstring's equivalent code,
`softmax(q @ k^T * scale + bias, dim=-1) @ v`. The bias is zeros with `-inf` where a Boolean mask or
the causal mask `tril(ones(L, S))` is false, or a floating mask plus zeros, so a row with every key
blocked is NaN (PyTorch's math path gives it zero weights). `enable_gqa` repeats each key and value
head, binary16 and bfloat16 compute in binary32 and round once, and `dropout_p > 0` is refused.

`clamp`, `clip` and `hardtanh` with two bounds are the IR's `clamp`, `minimum (maximum x lo) hi`,
the steps `Op.max` and `Op.min`. With only a lower bound they are `Op.max` and with only an upper
bound `Op.min`. `elu` is `where (x > 0, x * scale, expm1 (x * input_scale) * (alpha * scale))` and
`softplus` is `where (x * beta > threshold, x, log1p (exp (x * beta)) / beta)`, with binary16 and
bfloat16 computed in binary32 and coefficients of one not multiplied, as in their kernels.

The JAX importer lowers the comparisons of floating or integer operands, `is_finite` of floating
operands and the logical operators of Boolean operands to Boolean nodes. It lowers
`select_n(p, a, b)` with a Boolean predicate to `where(p, b, a)`, and `lax.clamp(lo, x, hi)` to
`clamp(x, lo, hi)`, or to `Op.min (Op.max x lo) hi` for integers. Integer `add`, `sub` and `mul`
wrap modulo `2 ^ width`. `and`, `or`, `xor` and `not` of integers, `shift_left`,
`shift_right_arithmetic`, `shift_right_logical`, `population_count`, `clz` and `mulhi` are the IR's
integer operators on the words (`mulhi` is the floor of the exact product over `2 ^ width`).
`bitcast_convert_type` between types of one width is an `Op.bitcast` step, every bit kept
(`TorchLean.IR.Program.step_bitcast_integer`, `step_bitcast_scalar_integer`,
`step_bitcast_integer_scalar`).

`reduce_window_sum`, `reduce_window_max`, `reduce_window_min` and `reduce_window` with one add, mul,
max or min as its body pad the operand with the init value (base dilation too). They then fold one
strided slice per window cell into the init value in row-major cell order. `Op.max` and `Op.min` are
IEEE maximum and minimum, so `+0` is above `-0` whatever their order, as XLA's CPU code computes.
`lax.reduce` with another init or body is refused (its documentation calls the result undefined).
`cbrt` is IEEE `rootn(x, 3)`, FloatLib's correctly rounded `rootN x 3` in the operand's format, and
`nextafter` is C's, FloatLib's `nextUp` and `nextDown`: these are the rules of `Op.cbrt` and
`Op.nextafter` at the configured binary formats. `reduce_precision(x, e, m)` is two casts through
the FloatLib format with the operand's exponent field and `m` fraction bits, then a signed infinity
and a signed zero outside the normal range of `e` exponent bits, with a NaN word unchanged.

`triangular_solve` is substitution (the dot products are left folds from `+0`), `cholesky` the
left-looking column algorithm, NaN everywhere unless every pivot is positive, `cholesky_update` the
factor of `R^T R + w w^T`, `symmetric_product` `alpha (a a^T) + beta c` and `custom_linear_solve`
its solve jaxpr. Special functions, complex values and the other factorizations are refused.

`gather`, `dynamic_slice`, `scatter`, `scatter-add`, `scatter-sub` (the sum of the negated updates),
`scatter-mul`, `scatter-max`, `scatter-min` and `dynamic_update_slice` of floating operands at
runtime indices become one registered `gather` or `scatter` at complete int64 coordinates. In
binary16 and bfloat16 a `scatter-mul` product rounds at every step, as XLA's CPU code does
(measured). A dropped `scatter-mul` update is replaced by one first, which keeps a NaN out of the
pullback. `scatter-max` and `scatter-min` are the rules `max` and `min`, IEEE maximum and minimum
as in `Op.max` and `Op.min`, with `-0` below `+0`. Starts are clamped so that the window lies
inside, as XLA clamps them (for `scatter` in the `CLIP` mode). A `FILL_OR_DROP` gather gives its
`fill_value` for a start outside, after converting the indices to int32 as JAX does when the
operand's axes fit, and `scatter` in the other modes drops that window. Updates apply in row-major
order of the index vectors, each window whole, so repeated `scatter-add` targets sum in that order
and the last repeated `scatter` update stays. JAX leaves the order of colliding updates open, and
the tests compare this order with CPU execution on their inputs. A scatter whose update computation
is not its name's (as from `.at[...].apply`) and an operand without entries are refused.

The JAX importer records supported operations, shapes, dtypes and casts. Reference execution uses
TorchLean and FloatLib arithmetic: division is IEEE division, `rsqrt` is `1 / sqrt x` rounded twice
(in binary64 `rsqrt 3` is one ulp above the correctly rounded `rSqrt 3`), and results underflow
gradually. `div`, `pow`, `neg`, `rsqrt` and `erf` map to IR operators, and integer `div` rounds
toward zero. Binary16 and bfloat16 `pow`, `rsqrt` and `erf` are computed in binary32 and rounded
once, as XLA computes them. On 8192 samples per operator and format, the IR returns XLA's words for
every binary16 `pow` and `rsqrt` sample and every bfloat16 `rsqrt` sample. It returns them for
8191 bfloat16 `pow` samples: XLA's result is its own binary32 power rounded once, and FloatLib's
binary32 power differs from it. The IR has no binary `erf`, so a graph using it does not execute.
`integer_pow` becomes the products XLA computes, square-and-multiply in JAX's order with a final
division for a negative exponent. On 4096 samples, XLA's `integer_pow` for the exponents 3, 5, 7
and -3 equals that schedule rounded at each step in bfloat16, binary16 and binary32.

`expm1`, `log1p`, `exp2`, `log2`, `tan`, the hyperbolic functions and their inverses, `asin`,
`acos`, `atan` and `atan2` of binary16 and bfloat16 are computed in binary32 and rounded once, and
an `accuracy` request is refused. The binary `log2` is `RN(RN(ln x) / RN(ln 2))`, not a correctly
rounded `log2`. XLA's bfloat16 `log2` is within one ulp of the IR's on the tested samples, but it is
not XLA's binary32 `log2` rounded once. `floor`, `ceil`, `round` with `TO_NEAREST_EVEN` and `rem`,
the IR's `fmod`, are exact in the format. `lax.sort` with one operand, one key and `is_stable` keeps
`-0` and `+0` in input order and puts NaN last, which is the IR's stable ascending sort. `lax.sign`
keeps `-0` and NaN, as it documents, so it is a `where` around `copysign`, not the IR's `sign`,
which maps both to `0`. `lax.sign` of integers is `-1`, `0` or `1`.

`lax.top_k` is the IR's stable `topk`, its indices the second output of `Op.topk`: equal entries
keep their input order, the order `lax.top_k` documents for `is_stable`. XLA's CPU code ranks in
IEEE totalOrder, `+0` above `-0` and a NaN by its sign and payload, and `lax.top_k` documents
nothing for NaN. For `k > 0` execution therefore fails where a fiber holds a NaN, or both a `-0` and
a `+0` (a sufficient condition, stricter than needed), and elsewhere the two orders agree. `cumsum`
and `cumprod` are the IR's left folds from `+0` and `1`, the order TorchLean chooses. XLA's CPU code
combines the prefixes associatively, so the words agree where each partial result of either grouping
is exact and zero signs and exceptional values are handled alike. `cummax` and `cummin` give the
value and zero sign of the running IEEE maximum and minimum, a NaN propagating as the latest NaN's
word. They are computed by the IR's scan, which takes a later equal entry as ATen does, and a
second scan of a `+0` flag (a `-0` flag for `cummin`) that turns a zero result into `+0` (`-0`).
`cumlogsumexp` is refused: no TorchLean scan computes a running logsumexp.

Nondefault dot precision and algorithms are rejected: a declared FP32 input and result do not say
whether the multiplications use BF16, TF32 or FP32, and the matmul node has no separate arithmetic
rule for operands and accumulator. A default matmul capture does not establish XLA's reduction order
or backend equivalence.

XLA's compiled CPU code departs from these meanings in the following ways, and the importer follows
none of them. Eager JAX compiles each primitive, and each nested `jax.jit` call, as a computation of
its own, while `jax.jit` compiles the whole function as one. XLA's results below are those of eager
JAX unless `jax.jit` is stated, and counts are of binary32 samples unless another format is named.

- A division by a broadcast divisor is multiplication by the divisor's reciprocal rounded to the
  format, `multiply(x, 0.333333343)` for `x / 3.0`, whose scalar divisor JAX broadcasts within the
  division. The divisor need not be constant: a 0-d argument, or a `[64, 1]` divisor of a
  `[64, 768]` array, is treated the same way. Under `jax.jit` so is a divisor computed from
  constants alone, such as a closed-over array. Eager JAX divides by that array as written, and by a
  divisor that `jnp.broadcast_to` broadcast in a primitive of its own. XLA's `x / 3.0` and
  `x / 10.0` differ from the IEEE quotient on 1382 and 839 of 4096 samples. Binary16 multiplies by
  the binary16 reciprocal and differs on 5487 of 16384 samples for `x / 3.0`. Bfloat16 multiplies in
  binary32, and the rounded product equals the quotient on all 16384. `jnp.mean` divides its sums by
  the broadcast element count this way: over 64 rows of 768, XLA's mean is the sum times binary32
  `1/768` in all 64 rows and the sum divided by 768 in 40.
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
  Under `jax.jit`, XLA computes the constant exponents 2 and 3 as products and -1 as a reciprocal in
  binary32 and binary64, an exponent that is a closed-over array included, and its `x * (x * x)`
  differs from the cube on 1062. In binary16 it rounds each product to binary16, and in bfloat16 it
  computes the products in binary32 and rounds once. On 8192 samples per class, binary32 and
  binary64, XLA's power is within 1 ulp of the correctly rounded power. It differs from the IR's by
  at most 1 ulp for integral exponents, where the IR is correctly rounded, and where
  `|y log x| < 1`, where FloatLib's power is within 1 ulp of the correctly rounded one. Where
  `|y log x| ≥ 1`, the samples do not compare XLA's power with FloatLib's.
- `rsqrt` is XLA's own approximation, within 1 ulp of the correctly rounded result and within 2
  ulps of the IR's `1 / sqrt x` rounded twice. It differs from the IR's result by 1 ulp on 5473906
  and by 2 on 50830 of the 16777216 binary32 values in `[1, 4)`. On 2^20 random normal values in
  each format, it differs by 2 ulps on 3106 binary32 and 1291 binary64 values. On 65536 samples it
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
policy. For a Jaxpr document it then tries these rules on the mismatch, as advice for the reader.
The JAX importer records `reference_execution`, `jit` for a function that is one `jax.jit` call
and `eager` otherwise, the computation of each node in an eager function, and the primitive that
produced each node. `torchlean.verify` records with the reference the platform that computed it
and jaxlib's version. When `jax.jit` compiles any part of the function, it also records the outputs
of evaluating the function one primitive at a time, inside nested `jax.jit` calls too, which XLA
computes without `jax.jit`'s rewrites. Only a reference from XLA's CPU backend is emulated. The
checker runs the program a second time as XLA's CPU code computes it, where that code is exactly
emulable. Divisions by broadcast divisors, and under `jax.jit` by constant ones, become products
with the rounded reciprocal. When `reference_execution` is `jit`, `jax.jit`'s rewrites apply:
those of quotients of quotients and by quotients in binary16, binary32 and binary64, and those of
divisions by square roots, reciprocal square roots and powers in binary32 and binary64. So do the
products for the exponents 2, 3 and -1, in binary32 and binary64 and in binary16 with each product
rounded to binary16. The flushing above applies to the operations listed.

XLA's `rsqrt` and power function are not emulable. A value that one of them computes in binary32
or binary64 counts as reproduced within 2 ulps of that format for `rsqrt` and 1 ulp for the power,
the largest distances measured between XLA's functions and the IR's. The window's ends are carried
to the result through layout operations, negation and conversions. A bfloat16 or binary16 result,
which XLA and the IR compute in binary32 and round once, is therefore reproduced only where the
binary32 window holds a rounding boundary of the reduced format. A window never crosses zero or
reaches an infinity. A power gets a window only when both its operands are arguments of the
program: a power with a small exponent hides a large error in its base, and a changed exponent
moves the result by about one ulp. Eager `x ** 3.0`, whose exponent is a constant, therefore has
no window, and a `jax.jit` cube is listed as consistent with the products only when XLA's power,
primitive by primitive, returns the IR's words on every argument.

If the second run reproduces every differing element, the failed finding lists the behaviours in
`consistent_with`, and its message says that the mismatch is consistent with them and that an
importer error with the same effect cannot be ruled out. Under `jax.jit` the list also requires
the primitive-by-primitive outputs, when the document has them, to equal the IR's results or to be
reproduced in the same way without `jax.jit`'s rewrites. A mismatch that the second run does not
reproduce fails with nothing listed, and the message names elements with their IR, emulated and
reference words, and the behaviours emulated. A FloatLib transcendental, or an importer bug, next
to one of these operations therefore lists nothing. Neither does a `jax.jit` `x / sqrt(y)` where
XLA's `rsqrt` differs from the IR's, since its result is a product, not XLA's `rsqrt` itself, nor a
`jax.jit` `x / rsqrt(y)`, whose primitive-by-primitive result is a quotient of XLA's `rsqrt`.

`consistent_with` never excuses a mismatch, because some importer errors are indistinguishable from
XLA's behaviour on the supplied arguments. A divisor one ulp off whose rounded reciprocal is
unchanged gives XLA's words exactly: eager `x / 3.62` written as `x / 3.6200001` is listed as
consistent with the reciprocal. Under `jax.jit` the rewrites map wrong operand orders onto the
reference, such as `(x / z) / y` for `(x / y) / z` and `z / (y / x)` for `x / (y / z)`, and without
a primitive-by-primitive reference these are listed too. The checker trusts the document's record of
how the reference was computed, so a forged record can at most add this advice to a failure.

## Relationship To Sequential Models

`nn.Sequential` is the typed model language: each layer has one forward program, which the pure
interpretation `nn.Sequential.eval`, the eager runtime and the typed graph all run.
`IR` is the lower-level graph of registered operators that verification, export and runtime
tooling consume after a model has been lowered or traced.

In PyTorch terms, `nn.Sequential` is close to a typed `torch.nn.Sequential`, and `IR` to an FX or
TorchScript graph with explicit ports and external constant payloads.

## How A Tensor Program Reaches IR

The IR represents general tensor computations too. A program need not contain a layer, a loss,
parameters or an optimizer: its interface is an ordered collection of tensor inputs and results.
Supported array operations and numerical algorithms get the same shape and element-type checks as
model computations. TorchLean does not compile arbitrary Lean or Python programs: control flow and
operations without a lowering rule are rejected.

From Python, decorate a function or a PyTorch module class with `@torchlean.compile`, or wrap an
existing module with `model = torchlean.compile(model)`, then call it as usual. Capture is lazy:
PyTorch supplies a guarded FX graph or JAX a traced Jaxpr, and the model and tensor arguments select
the frontend. A call captures the whole tensor or batch, with no separate check loop over its
entries. TorchLean lowers the graph to the shared IR and requests the structure and type checks. The
framework executes the captured computation, keeping its device and automatic differentiation.

The callable's `.torchlean.artifact` and `.torchlean.report` hold the latest capture and its check
result, and `.torchlean.graphs` holds every specialization that passed its checks. A cached call can
reuse an earlier specialization without changing the latest capture. An unsupported capture or a
failed check raises. The checks do not prove frontend equivalence or correctness on every input.

Use `torchlean.verify(model, *inputs)` for further checks, including sample execution and
comparison. It selects the frontend in the same way, and other producers can pass a typed artifact
dictionary. Optional execution and comparison concern only the supplied arguments.

Add `export="Model.lean"` to `verify` to keep the same capture as a self-contained Lean module. It
embeds the capture and defines `Model.program`, the program of its typed artifact. Lean checks the
included proofs of artifact decoding and `Program.check .reference` by kernel reduction when the
module is compiled, and export does not launch a compiler. `IR.load` also validates the constant
payload and `declared_types`; the export includes no proof of those load checks. Mathematical
properties can refer to that program and the embedded constants, and they need their own proofs.
A `.json` export keeps the capture document instead. Neither export replaces an existing file. The
[captured-computation example](../../../home_page/examples/verify/index.md) exports a captured
`square` function to `Proofs/Square.lean`.

There are several routes into the same graph language:

- TorchLean model code can lower supported fragments into IR for execution, inspection, or
  verification.
- The PyTorch `torch.export` capture records positional tensor arguments, including unused
  arguments, from its graph signature and writes each as an input step of a typed artifact with its
  argument position, and its other nodes become typed steps (`python/torchlean/_artifact.py`).
  Lean decodes and checks the artifact. The ONNX adapter (`python/torchlean/onnx.py`) writes each
  graph input that is not an initializer as an input step, in the graph's order.
- The Jaxpr capture in `python/torchlean/jaxpr.py` lowers static tensor computations to a typed
  artifact in a capture document. Its tensor arguments are input steps in order, each with its own
  shape and dtype. `TorchLean.JAX.load` checks the frontend and loads the program under the
  reference runtime (`TorchLean.JAX.load_ok`), whose run uses FloatLib at each declared binary
  precision and runs Booleans and integer words natively. Structured control flow is not lowered
  (see the [JAX example](../../../home_page/examples/jaxpr/index.md)).
- Verification examples can build small graphs with `Graph.Builder` when the graph itself is the
  artifact under study.

Mixture-of-experts routing is one registered operator, `Op.topKMask` with `dim` and `k`, whose
meaning is `TorchLean.MoE.topKMask`: one for the `k` largest entries of every fiber along `dim`,
ties to the lower index. A row of `E` totally ordered scores (no NaN) gets `min k E` ones. With NaN
the pairwise comparison can cycle, so `[0, NaN, 1]` keeps no entry for `k = 1` and all three for
`k = 2`. The PyTorch adapter captures the usual idiom
`torch.zeros_like(p).scatter(-1, torch.topk(p, k).indices, 1)` as that step. For `0 < k < E`,
agreement requires finite scores and a strict gap between the `k`-th and `k + 1`-th largest entries,
since `torch.topk` leaves ties unspecified. Capture records these assumptions without checking them.
Empty and full masks need no cutoff rule. `TorchLeanTest/IR/MoEImport.lean` exports a Mixtral-style
block from PyTorch and checks that the artifact's operator graph computes `TorchLean.MoE.forward`
bit for bit.

Frontend coverage is narrower than the graph's operator vocabulary. JAX division, mean and the
LayerNorm expansion through `rsqrt` import directly, and square root as IEEE's root at every
argument. `reduce_max` imports as one `reduce` node at `max` (refused on an empty axis), so JAX's
softmax expansion and attention code that uses it import. JAX's exact GELU uses `erfc`, which has
no IR operator.

## Role And Scope

Write ordinary models through `TorchLean.nn` or `Trainer`, then lower them.
Direct `Node` construction is useful for developing IR consumers and certificate examples.

## Consumers

| Consumer | How it uses IR |
| --- | --- |
| Execution | `TorchLean.IR.Graph.run` and `TorchLean.IR.Graph.eval` run an operator graph on its payload by the rules of its registered operators, each custom operation by its checked program, and a typed artifact runs as its typed program (`TorchLean.IR.Artifact.load`, `TorchLean.IR.Program.run`) at runtime precision. |
| Verification | Runs IBP/CROWN-style passes, margin checks, and certificate replay over node ids and payloads. |
| TorchLean lowering fragments | `IR.lowerForward` lowers a forward model to the operator graph of the typed graph that `recordForward` records, and `IR.lowerForward_eq_ok` proves that a successful lowering returns that typed graph's value at the output for every input. No theorem relates it to the source program. |
| PyTorch/ONNX/export paths | Use a small graph format to make parameter order and tensor shapes explicit at the boundary. |
| Widgets and graph pages | Render operator graphs with their structural checks (`#ir_view`), the ports of every node (`#shape_infer_view`), execution traces (`#ir_exec_trace_view`), and dependency structure for debugging. |

## Proof And Runtime Status

The IR is the object shared by proofs and runtime code, but proof coverage is named fragment by
fragment. Theorems cover the structural check (`Graph.check_ok_iff`), the output ports
(`Graph.run_ports`), the typed graph (`OpGraph.eval_toOp`), the lowering of a recorded forward
(`IR.lowerForward_eq_ok`), typed casts, the step rules of typed programs, and verification-oriented
bound propagation. Separately, `GraphData` lowers to the autograd tape, and that typed lowering does
not mean every `IR` node is differentiable. Adding an operator therefore takes three things in the
right places:

- its shape contract in `OpContracts.lean`, where its shape rule is nontrivial, and its lowering in
  the Python captures (`python/torchlean/_artifact.py`)
- its registered operator of `TorchLean.Op` with its rule and its codec record (`Op.encode`,
  `Op.decode`)
- the runtime, proof and checker coverage that makes the operator usable in the intended workflow

A tag without these pieces gives a graph that prints but cannot support verified claims. The
[contributing guide](../../../docs/CONTRIBUTING.md#adding-an-operator) lists the steps of adding a
registered operator of `TorchLean.Op`, with its rule, codec and runtime entries.

## Proving Properties Across A Program

A loaded `Program` has Lean semantics, so a proof can work with its operations without first
recognizing the graph as an MLP, a transformer or another named architecture. One traversal
argument covers every typed program:

- `TorchLean.IR.Artifact.load_ok` and `TorchLean.IR.load_ok` state that a loaded program passes
  `Program.check` and its payload `Payload.check`, the premises of the theorems below.
- `TorchLean.IR.Program.run_eq_foldlM` states that the run of a checked program with a checked
  payload is its fold over the steps. So a predicate on intermediate values and conversion flags
  follows from the initial state and one step at a time (`check_step`, `step_op_arithmetic`,
  `step_op_native`).
- `TorchLean.IR.Program.run_graph?_ok` and `run_graph?_none` state that a program whose steps are
  all operator-graph nodes runs as its operator graph.

The proof rules use the existing `Tensor` values, the typed program and its runtime. Numerical
relations must use that runtime's actual arithmetic, casts and parameters. The traversal theorem
does not supply error bounds or invariance rules for an operation whose corresponding proof is
missing.

These theorems assume that the executions being compared succeed. They do not establish success on
every input, prove a frontend translation faithful or verify external kernels. Comparing two
different graph structures also needs a correspondence between their computations, because the
same-program relational rule does not establish the equivalence of arbitrary optimizations.

## Element Descriptors And Extensions

`DType.toJson` and `DType.fromJson` share one frontend-independent representation. Binary
descriptors retain exponent width, fraction width, bias, and exceptional-value encoding. Decimal
descriptors retain FloatLib's declet count, exponent width, and signed bias. Integer widths,
posit widths, complex component types, and named extensions are represented explicitly too.
Named-type component arrays preserve qualified names without losing numeric or escaped components.
Element descriptors and call steps share the same total decoder for component arrays and Lean
qualified names, including escaped components.
`DType.fromJson_valid` proves that a successfully decoded descriptor passes its parameter checks.

`Reference.read_write` proves lossless element-word decoding for every supported format and integer
width. Boolean words are zero or one. Integers use Lean's `BitVec n`, with the signedness in the
descriptor and no conversion to a host float. The reference runtime runs their layout operations,
PyTorch's integer pointwise operators, `Op.add`, `Op.sub` and `Op.mul` modulo `2 ^ width`, and
`Op.max`, `Op.min` and the comparisons in the type's order. It also runs conversions to binary
formats that round the exact integer once, conversions between integer types modulo `2 ^ width`,
and conversions from binary formats that round in the cast's direction and fail outside the type.
It runs no other integer arithmetic. Decimal, posit, complex and named element types have no
implementation in this runtime.

When `Artifact.load` decodes a constant (`Program.Typed.decode`), it refuses a word count other
than the shape's size, a Boolean word other than zero or one and an integer word beyond its width.

A producer can declare a call step with a qualified `name`, its argument references, the declared
type of each argument (`signature`) and its result port. Execution also requires an implementation
registered for the payload (`TorchLean.IR.Program.Env`, `Artifact.load`). Declaring the signature
registers no executable code and proves nothing about its mathematics. `TorchLean.IR.load` registers
none, so it refuses an artifact with a call. Likewise, a valid decimal, posit or named descriptor
does not mean that the reference runtime can execute it. Other runtimes must supply the
corresponding scalar operations.

## Payload Discipline

`TorchLean.IR.Payload` stores constants (`const?`) and custom programs (`custom?`) by node id. The
structural check (`Graph.check`) needs no values. Evaluation rejects a missing constant or program
(`Node.eval`, `runCustom`) and inputs that fail a rule's value-domain check (`Op.invoke`). Verifier
input boxes are the seeds of `runIBP`. Imported weights are artifacts to validate.

## Release Invariants

- A node's id is its position in `g.nodes` (`Node` has no id field).
- On a checked graph every reference names an earlier node's output at a port its input accepts
  (`Graph.check`, `Graph.check_ok_iff`).
- Constants, parameters among them, and custom programs live outside `Graph`: `Payload` holds them
  by node id (`Payload.const?`, `Payload.custom?`).
- A loaded artifact's program passes `Program.check` and its payload `Payload.check`
  (`Artifact.load_ok`). A step's shapes are the ports of its registered operator, which `Op.decode`
  recomputes from the record's attributes.
