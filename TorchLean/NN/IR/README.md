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
- `ReducePrecision.lean`: the JAX import's `reduce_precision` rounds once. In every IEEE format,
  converting a finite value to that format with fewer fraction bits and back gives, where the first
  conversion does not overflow, a finite result equal to the narrower format's rounding of the
  value (`toReal_reduce_ieee`). For any formats the premises admit, converting a finite value to a
  wider IEEE format, to a narrower format and back to each gives a finite result equal to the
  narrower format's rounding of the value (`toReal_reduce`). At the six float8 settings it rounds
  (`settings`) the premises hold at every word (`toReal_reduce_float8`).
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
  seeds at value ports with matching shapes, using registered operators for its own per-operator
  rules (no theorem relates them to `jvp?`), and outputs the graph's outputs followed by their
  tangents.
- `Flow.lean`: `IR.Flow`, branches and loops over blocks such as graphs, on a state of values.
  `cond` runs only the branch its index selects (the last one for an index out of range), `scan`
  runs its body a fixed number of times on the carried state, and `whileLoop` runs its body while
  its test holds. `Flow.eval` takes the fuel as a parameter: a `whileLoop` runs its body at most
  `fuel` times each time it is entered and fails if its test still holds then. `Flow.Lang m` gives
  the meaning of blocks and values in a monad `m`: `Flow.graphs` runs graphs by `Graph.eval` in
  `Except String`, and `Flow.programs R` typed programs by `Program.eval` in `StateT
  ConversionStatus (Except String)`, merging the conversion flags of every block it runs. For every
  language, flow, fuel and state, `scan n` equals its body unrolled `n` times (`Flow.eval_scan`).
  For every language in `Except String`, such as `Flow.graphs` and `Flow.Reals.lang`, and every
  flow, fuel and state, `cond` equals evaluating every branch and selecting one wherever every
  branch is defined (`Flow.eval_cond`), and a `whileLoop` whose test holds before exactly
  `k ≤ fuel` iterations equals its body unrolled `k` times (`Flow.eval_whileLoop`). For
  `Flow.programs`, the language of every imported flow, only `Flow.eval_scan` is proved, and the
  `cond` equation is not proved and fails where an unselected branch raises a conversion flag that
  the selected one does not: evaluating every branch merges the flags of all of them, and `cond`
  those of the selected one. The JAX importer writes flows for a `cond` or `while` that the capture
  does not decide, and the PyTorch importer for `torch.cond`, `while_loop` and a `map` whose body
  does not draw from TorchLean's generator ([below](#frontend-lowering)).
- `Flow/Follows.lean`: a flow can run in a second language on states that stand for states of a
  first, such as a dual language whose values carry tangents or error bounds (`Flow.Lang.dual`).
  `Flow.Follows.eval` proves, for every pair of languages whose blocks, indices and tests are
  related along a filter (`Flow.Follows`) and for every flow, fuel and state, that wherever the
  second run succeeds, the first succeeds eventually along the filter with a result related to the
  second's. `Follows.cond` and `Follows.whileLoop` state that the second run of a `cond` is that of
  the branch its index selects, and of a `whileLoop` that stops that of its body unrolled some
  `k ≤ fuel` times, and that eventually along the filter the first run takes the same branch, and
  its `whileLoop` and its body unrolled the same `k` times return one result. They do not state that
  the first run's test fails after `k` iterations. `Follows.scan` states that eventually along the
  filter the first run of a `scan` of `n` and of its body unrolled `n` times return one result; that
  the second run is also that of the body unrolled is `Flow.eval_scan`, which holds in every
  language.
- `Flow/Reals.lean`: `Flow.Reals.lang`, whose blocks compute sums, products, negations and constants
  of reals and comparisons `a < b`, with two duals. `dual E` applies the sum and product rules and
  refuses a comparison of equal reals, and for all of its flows, wherever it succeeds, the flow near
  the point takes its branches, each of its loops returns what its body unrolled as often as the
  dual's loop runs returns, and it has the dual's values and Fréchet derivatives
  (`Flow.Reals.dual_follows`). `rounded rnd err` rounds each constant, sum and product by `rnd`,
  carries error bounds that add `err` of each exact result to the operands' propagated errors, and
  refuses a comparison whose operands are within their bounds of each other. For every `rnd` and
  `err` with `|rnd z - z| ≤ err z` and all of its flows, wherever it succeeds, the exact flow takes
  its branches, each of its loops returns what its body unrolled as often as the rounded loop runs
  returns, and its reals are within its bounds of the rounded ones (`Flow.Reals.rounded_follows`).
  FloatLib's `roundAt fmt` is such an `rnd`, rounding to nearest-even on the format's grid without
  overflow. `square_hasDerivAt` and `square_rounded` in
  `TorchLeanTest/IR/Flow.lean` apply them to a loop. Flows of IR graphs have neither, because no
  theorem relates `Graph.tangent` or `Op.Rule.jvp?` to a derivative and none bounds the rounding
  error of `Graph.eval`.
- `Program.lean`: typed programs (`IR.Program`). Their steps are graph nodes, conversions
  and registered calls, with one scalar per step for its value, frozen and detached ports, and
  native Boolean, integer and index ports. `Program.Typed.ofDType?` gives the type of a tensor from
  its element type and shape, and the module holds the reference runtime `Program.Machine.reference`
  over the element types of `Scalar.lean` and the structural checks `Program.check` and
  `Payload.check`. `Program/Graph.lean` runs a program whose steps are all graph nodes as its
  graph (`Program.run_graph?_ok`, `Program.run_graph?_none`), and `Program/Step.lean` states single
  steps (`Program.step_op_arithmetic`, `Program.step_op_native` and those of `bitcast`,
  `decodeIndex` and `encodeIndex`). The JSON format of the IR, the typed artifact (a typed program
  with the words of its constants), and its loader are in `TorchLean.NN.IR.Artifact`
  ([below](#the-artifact)).
- `Loaded.lean` and `Table.lean`: `TorchLean.IR.load`, which loads the document of a PyTorch or JAX
  capture without control flow, `TorchLean.IR.loadFlow`, which loads one with or without it, and
  `TorchLean.IR.table`, the op table that the Python captures read ([below](#the-artifact)).
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

An artifact with control flow (`TorchLean.IR.FlowArtifact`) is a flow of typed artifacts, each
block run on the state as its arguments, with the fuel of its while loops: `{"format": "torchlean.ir",
"version": 1, "fuel": n, "flow": F}`, where `version` is the flow version
`TorchLean.IR.Flow.formatVersion` and `F` is `["block", A]` with an artifact `A`, `["skip"]`,
`["seq", F, G]`, `["cond", F, ...]`, `["scan", n, F]` or `["while", T, B]`. The states between
blocks hold no constant: each block that reads one holds it, and no other block does, so a weight
that both branches of a `cond` read is written twice, and one read in each iteration of a `scan`
that the capture unrolls with a flow per iteration is written in every iteration's blocks. An
artifact without control flow keeps the format above, word for word. `FlowArtifact.ofJson_toJson`
proves the round trip of every flow artifact, and `FlowArtifact.load` loads each block with
`Artifact.load`, which `FlowArtifact.load_ok` states for every block.

The PyTorch and JAX captures write one capture document per computation, which
`TorchLean.NN.IR.Loaded` reads: the artifact, the arguments and reference outputs by element type,
shape and complete interchange words, and the name of the capture that wrote it. Its
`"declared_types"` give, for each value the capture lowered, the step output that holds it and the
element type and shape that the framework's own metadata records for it. `TorchLean.IR.load`
loads the artifact under the reference runtime with no registered custom operation or call, and
`TorchLean.IR.load_ok` proves that the program it returns passes `Program.check .reference` and its
payload `Payload.check`. `load` then requires the program to give every declared value its declared
element type and shape (`TorchLean.IR.checkDeclared`), and `TorchLean.IR.load_declared` proves
that a loaded program does. So a lowering that changes a declared value's type fails the load, with
a message `IR types: ...` that names the value, the type the IR computes and the declared one. Only
the listed values are checked, so an empty list passes. A document that names a frontend must have
`"declared_types"`, and a raw IR document without it declares no types. `TorchLean.IR.loadFlow`
loads an artifact with control flow as well: an entry of `"declared_types"` names its block
(`"block"`, by the order of `Flow.blocks`, 0 where absent), and `TorchLean.IR.loadFlow_ok` proves
for every block what `load_ok` and `load_declared` prove for the program. An artifact without
control flow is the flow of its one block. `TorchLean.JAX.load` checks that the frontend is `jaxpr`
and then runs `loadFlow`, and `TorchLean.JAX.load_ok` proves for it what `loadFlow_ok` proves for
every block's program and payload. The checker (`TorchLean.Capture.check`) and the error budgets of
`TorchLean.Verify.Capture.Error` load documents with `loadFlow` and run a flow by `Flow.eval` in
`Flow.programs .reference`. The checker's `structure` and `types` findings also check that a flow's
blocks compose (`TorchLean.Capture.flowTypes`): each block is given values its arguments read, in
every typing that some path through the flow's branches and loop iterations gives them, whether or
not a run takes that path. Values that come straight from the flow's arguments are left to
execution.
`TorchLean.IR.loadArguments` reads the arguments under a runtime, and `TorchLean.IR.words` gives
the element type, shape and words of a value back.

The PyTorch capture in `python/torchlean/` and the Jaxpr capture (`python/torchlean/jaxpr.py`) lower
their graphs to these steps (`_artifact.lower` in `python/torchlean/_artifact.py`), and the ONNX
adapter (`python/torchlean/onnx.py`) writes typed artifacts too. All three read the facts they need
from Lean's op table, `TorchLean.IR.table` in `TorchLean.NN.IR.Table`, which
`verify ir --op-table` and the Python binding `op_table` return. The table holds the format name,
the wire version, the flow version, the JSON of each element type, the codec's codes and the
typed forms of the pointwise operators (`Pointwise.form?`). Lean checks every step when it loads
the artifact.
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

`reduce`, `mean`, `contract`, `matmul`, `scaledMatmul`, `pool`, `cumsum`, `cumprod`, `cummax` and
`cummin` take a reduction order last (`Tensor.ReduceOrder`, in
`TorchLean/Tensor/Reduce/Order.lean`): the left fold `leftFold`, the default, the balanced tree
`pairwise`, the blocks `blocked w`, or a given tree `ofTree t`. In the left fold an operator means
its tensor definition, which folds from the left in index order, and its rule is the rule it had
before orders existed. In another order the forward folds in that order (`Tensor.reduceIn`,
`meanIn`, `contractIn`, `batchMatmulIn`, `Attention.scaledMatmulIn`, `poolIn`, `scanIn`, `cummaxIn`,
`cumminIn`), and so does the tangent of a sum, a mean, a contraction, a matrix product and a
cumulative sum. The pullbacks are the left fold's. `Op.check` rejects an order that does not cover
the count it reduces (`Tensor.ReduceOrder.Covers`), and over the reals every order that covers gives
the left fold's value (`Tensor.reduceIn_sum_real` and the other `_real` theorems), and running
maxima and minima the left fold's indices too (`Tensor.cummaxIn_real`, `Tensor.cumminIn_real`). At a
configured binary format running maxima and minima in every covering order are the left fold's, bit
for bit (`ConfiguredBinary.cummaxIn_eq`, `ConfiguredBinary.cumminIn_eq`). The wire record ends with
the order (wire version 5), in JSON `{"order": "leftFold"}`, `{"order": "pairwise"}`,
`{"order": {"blocked": 4}}` or `{"order": {"tree": t}}`, with `t` `null` (the start value), an
entry's index or a pair `[left, right]`. PyTorch, JAX and ONNX fix no order for these operations, so
their captures write the left fold. The deterministic CUDA sum is the order `LibTorch.sumOrder`
(`LibTorch.reduceSumTree_eq_fold`). Because every lane starts from `+0`, it does not satisfy
`ReduceOrder.Covers` for a positive input count.
`TorchLean/NN/IR/Order.lean` sets every order of a graph to the left fold (`IR.Graph.leftFold`).
Over the reals, a graph that evaluates produces the same outputs in the left fold
(`IR.Graph.eval_leftFold_real`), since a node that passes `Op.check` has an order that covers its
count.

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
`cummin`, the `int32` exponent of `frexp`, and `embedding`, are registered operators of
`TorchLean.Op` with index, integer or Boolean ports. `Program.check` also requires such a step to
declare a scalar exactly when it has value ports (`Program.checkScalar`). The lowering
(`python/torchlean/_artifact.py`) writes `Op.decodeIndex` before an index operand and
`Op.encodeIndex` after an index result, and Boolean and integer ports carry their elements natively
(`BitVec width` at an integer type). The step succeeds exactly when invoking the operator on its
inputs does (`TorchLean.IR.Program.step_op_arithmetic`, `step_op_native`), and `Op.invoke_eq_ok`
states that a successful invocation returns the rule's forward values. A `decodeIndex` step outputs
the index tuples its words hold and fails on a component outside its bound, and an `encodeIndex`
step outputs the words of its tuples and fails on a component the integer type cannot hold
(`TorchLean.IR.Program.step_decodeIndex`, `step_encodeIndex`). These are statements about one step.
A program written directly in the IR can also return index tuples. The `verify` report writes such a
result with `"element_type": "index"`, its port and the components of its tuples
(`TorchLean.Capture.resultJson`), and a comparison with a reference tensor fails, since a tensor
holds no index tuples.

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

The unary `torch.special` functions that `TorchLean.SpecialFn` names (`erfc`, `erfcx`, `erfinv`,
`ndtr`, `ndtri`, `lgamma` and `gammaln`, `digamma` and `psi`, `bessel_j0`, `bessel_j1`, `i0`,
`i1`, `i0e`, `i1e`, `spherical_bessel_j0`, and the `torch.special` spellings of `erfc`, `erfinv`,
`digamma` and `i0`) are captured from PyTorch as the registered operator `Op.special [0] f`, whose
rule at the reals is `SpecialFn.real f`, and `polygamma n` (`torch.polygamma`,
`torch.special.polygamma`, the Tensor method) as `Op.special [n] .digamma`, the derivative of order
`n` of `digamma`. `Op.special orders f` is the partial derivative of `f` of order `orders[i]` in
argument `i`, for a special function `f` of `orders.length` arguments. Like `erf`, it has no rule
at any binary format, so a binary32 or binary64 run checks the graph and refuses the step at
execution; binary16 and bfloat16 operands are refused at capture. The functions of two arguments
(`gammainc` and `igamma`, `gammaincc` and `igammac`, and `zeta`) are captured the same way as
`Op.special [0, 0] f`, from two float32 or two float64 tensors broadcast to the result's shape;
operands of two dtypes, which PyTorch would promote, and a Python number are refused at capture.
The JAX importer imports `erfc`, `erf_inv`, `lgamma`, `digamma`, `bessel_i0e` and `bessel_i1e` the
same way, `polygamma` whose order is a constant natural number as `Op.special [n] .digamma`,
`igamma`, `igammac` and `zeta` as `Op.special [0, 0] f`, and `igamma_grad_a`, the derivative of
`igamma` in its first argument, as `Op.special [1, 0] .igamma`, and `regularized_incomplete_beta`
(`lax.betainc(a, b, x)`) as `Op.special [0, 0, 0] .betainc` of its three operands. As from PyTorch,
these imports check at binary32 and binary64 and refuse the step at execution, and binary16 and
bfloat16 operands are refused at import. PyTorch has no incomplete beta function, so the exporter
refuses `betainc`.

Binary flooring division runs `divFloorSchedule`, and `divFloorSchedule_real` proves that
the same schedule computes `⌊x / y⌋` in exact arithmetic. `remainderSchedule_real` proves that
the remainder's schedule computes `x - y ⌊x / y⌋`. A zero remainder has the divisor's sign, as
Python's `%` and `numpy.remainder` give it and as `torch.remainder` documents ("the same sign as the
divisor"), while eager PyTorch 2.13 and `jnp.remainder` return the dividend's sign there.

FloatLib binary's inverse trigonometric functions are those of
`TorchLean.Tensor.ConfiguredBinary.InverseTrigonometric`: FloatLib's exact comparisons of
the real function with rational boundaries decide its nearest-even rounding.
`TorchLean.Verify.RuntimeApprox.InverseTrigonometric` proves the results finite and correctly
rounded at finite inputs in the domain, for every IEEE descriptor, with their special values. Binary16 and bfloat16 captures compute in binary32 and round once to the result format. A
framework uses its own binary32 approximation, so its final word need not match the IR's. The
[inverse-trigonometric regression](../../../scripts/checks/ir_capture_pytorch_operators.py) compares
selected PyTorch inputs and special values; those comparisons give no bound for other inputs.

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

`TorchLean.Activation.geluTanh` computes `β = √(2 / π)` in the format, rounds `κ = 0.044715` once,
and then rounds each operation of PyTorch's kernel in its order, `κ x³` and `x + κ x³` separately.
PyTorch rounds `√(2 / π)` once to binary64, its `kBeta`, and that value to the computation type. In
binary64 the two `β` are the same word, and the IR's words are those of the same order with
PyTorch's constants. In binary32 the IR's `β` is `0x3f4c4229`, one ulp below PyTorch's `0x3f4c422a`,
so binary32, bfloat16 and binary16 results can differ from PyTorch's in the last bits.

PyTorch's CPU kernels also select different arithmetic schedules by CPU capability. Vectorized
kernels can fuse `x + κ x³` and use SLEEF's `tanh`, while other paths round the product and sum
separately or call the C library. The IR follows its explicit, unfused schedule and uses
FloatLib's `tanh`. A source-level formula alone therefore does not establish word-for-word
agreement with a selected PyTorch kernel.

For negative `x`, `1 + tanh(inner)` subtracts nearly equal values. Errors in `tanh` can become large
relative errors in the small GELU result, and the result becomes `-0` once `tanh(inner)` rounds to
`-1`. This can happen well before the exact approximation would round to zero. At `x = -∞`, the IR
returns the limiting value `-0`; a direct evaluation of `x (1 + tanh(inner))` gives NaN through
`-∞ · 0`. Binary16 and bfloat16 captures compute the formula in binary32 and round once to the
result format. The [capture tests](../../../scripts/checks/test_ir_capture.py) exercise these
rounded operations without claiming general bitwise agreement with the framework.

### Frontend lowering

Each captured operator means its mathematical definition, IEEE 754-2019's operation or the
documented API, and the capture writes that meaning. Where PyTorch's CPU kernels compute something
else, the capture does not copy them. The differences are listed here. A tanh `gelu` node computes
in the DEFAULT level's order. Binary16 and bfloat16 inputs of `rsqrt`, `erf`, `gelu`, `pow`, the
exponentials, logarithms, `tan`, the hyperbolic functions and their inverses, `asin`, `acos`,
`atan`, `atan2`, `hypot` and `remainder` are widened to binary32, where PyTorch computes them too.
The binary32 result is rounded once, `R16(f32(x))` for the IR's binary32 operation `f32`, and
`rsqrt`, for one, gives `R16(R32(1 / R32(√x)))`.

#### Scalars and conversions

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

A Python float or complex number that a Jaxpr keeps as a literal, such as `jnp.clip`'s bounds, the
values of `jnp.where`, `jnp.pad` and `jnp.nan_to_num`, and the NaN that `jnp.var` and `jnp.median`
can select, is NumPy's conversion `np.array(value, dtype)` to the literal's declared type, as JAX
0.11.1 lowers it. Without x64, `jnp.clip`'s bounds of a binary16, bfloat16 or float8 array are
declared binary32, and the jitted function that reads them converts them to the array's type. So
`jnp.clip` of a bfloat16 array at `1 + 2^-8 + 2^-25` clips at that bound rounded to binary32,
`1 + 2^-8`, and then to bfloat16, `1`, a tie rounded to even, where the bfloat16 value nearest the
bound is `1 + 2^-7`. A Python complex number that `jnp.where` or `jnp.pad` puts in a complex64
array is declared complex64, each part rounded once to binary32. A literal that overflows its type,
or an infinity that becomes a NaN in a float8 format without infinities, is refused. A NumPy scalar
literal must equal its declared type's value exactly.
The complete computation still needs supported primitives: a median trace that requests unstable
sorting is rejected.

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
the capture lowers a floating `trunc` division as `div` then `trunc`. An integer one is the IR's
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

#### PyTorch reductions and normalization

PyTorch's CPU `cumsum` and `cumprod` accumulate in `at::acc_type` (binary64 for binary32 and
binary64, binary32 for binary16 and bfloat16) and round each prefix to the operand's format. The
capture writes a cast to that type, the IR scan and a cast back, which return eager PyTorch's words
on binary32, binary16 and bfloat16 samples with signed zeros and NaN. `cummax` and `cummin` take a
later tie and a later NaN, as the IR's do. `max.dim` and `min.dim` keep the first extremum and the
first NaN, and the capture writes them as the IR's stable `topk` with `k = 1`, between two exact
negations for `min.dim`. `torch.sort` is imported with `stable=True` only, because ATen leaves the
order of equal entries, `-0` and `+0` among them, unspecified. Its integer and Boolean operands
sort at their element, by value and with `false` first, as ATen's do. `torch.topk` is the stable
`topk` where ATen's result is uniquely determined, and for `k > 0` execution fails where two
adjacent entries of a fiber's first `min(k + 1, n)` in the requested order tie. `torch.frexp` is
`Op.frexp`, C's `frexp`: the mantissa and the `int32` exponent `e` of each entry, the normalized
mantissa of a subnormal, `(x, 0)` at a zero or an infinity, and the quiet NaN with the exponent `0`
at a NaN, which quiets a signaling NaN as PyTorch does. The capture reads `aten.frexp.Tensor` and,
under FX, `torch.frexp` and the `frexp` method, whose result the program may return whole or
project by position or by its fields `mantissa` and `exponent`. A configured format has the
operator only in an IEEE format of exponent width at least three, where every normalized mantissa
is a value. Its pullback is `ldexp g (-e)`, rounded once. PyTorch 2.13's autograd divides by
`exp2(e)` computed in binary32, which gives `0` for `e >= 128` and an infinity for `e <= -150`, in
binary64 too.

`leaky_relu` is `Op.leakyReLU` at the slope, in binary32 for the reduced formats with the slope
rounded there, and the result rounded once to the format. Constant-index `gather` and `scatter`
nodes hold the complete coordinates of every entry. A repeated scatter target is refused, since
PyTorch leaves the stored value, and for `scatter_add` the order of the sum, unspecified.

The weight, bias, eps and running statistics of a `layer_norm` or `batch_norm` node are constant
steps, interchange words of the node's dtype that `Artifact.load` decodes. A weight, bias or running
statistic that is not a parameter, buffer or constant, an input or a value computed from one, is a
parent after the input instead, in the order weight, bias, mean, variance
(`test_normalization_takes_a_weight_not_known_at_capture`). The statistics feed the frozen ports
of `batchNormRunning`, which accept a parent's values. An absent weight or bias is one or zero, and
eps is rounded to the computing dtype. Binary16 and bfloat16 normalizations are computed in
binary32, PyTorch's opmath type, between two casts. PyTorch's CPU kernels compute
`(x * rstd - mean * rstd) * gamma + beta` from Welford moments and
`x * alpha + (beta - mean * alpha)`, while the IR computes
`(x - mean) / sqrt (var + eps) * gamma + beta` with two-pass moments, so the words agree where every
step is exact and otherwise differ in the last bits. Layer norm, group norm and batch norm, with
running or batch statistics, take the IEEE root of `var + eps` as it stands, as PyTorch does, so a
negative or NaN sum gives NaN in its row, group or channel. Where the sum is 0 in inference batch
norm, PyTorch's `alpha` is infinite and its output is NaN unless `x * alpha` and `-mean * alpha` are
infinities of one sign, while the IR's quotient is the infinity of the sign of `x - mean` (NaN at
`x = mean`). A mean over an empty fiber with a nonempty result is refused (PyTorch returns NaN, the
IR mean divides by one). The variance there is `0 / 0`, NaN.

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
bound `Op.min`. `elu`, `selu` and `celu` are `Op.elu` at frozen `alpha`, `scale` and `input_scale`,
`x * scale` where `x > 0` and else `expm1 (x * input_scale) * (alpha * scale)`, the kernel's
selection. `selu` is at ATen's `SELU_ALPHA` and `SELU_SCALE`, and `celu(x, alpha)` at `alpha`, `1`
and `1 / alpha`, the quotient in binary64. `softplus` is
`where (x * beta > threshold, x, Op.softplus (x * beta) / beta)`, the documented
`log (1 + exp (beta * x)) / beta` with its reversion to the linear function above the threshold,
and a `beta` of one is not multiplied or divided by. Binary16 and bfloat16 are computed in
binary32, as in their kernels.

#### JAX reductions and linear algebra

The JAX importer lowers the comparisons of floating or integer operands, `is_finite` of floating
operands and the logical operators of Boolean operands to Boolean nodes. Boolean operands, false
below true as in NumPy, are compared at 0 and 1 in binary64. It lowers `select_n(p, a, b)` with a
Boolean predicate to `where(p, b, a)`, and `lax.clamp(lo, x, hi)` to `clamp(x, lo, hi)`, or to
`Op.min (Op.max x lo) hi` for integers. That is the formula of StableHLO's `clamp` and of XLA, a
refinement of the cases in `lax.clamp`'s docstring, which differ at a NaN bound, at `x = -0` and
where `lo > hi`: `lax.clamp(nan, 1.0, 2.0)` is NaN here and `1.0` by the docstring, and
`lax.clamp(2.0, 1.5, 1.0)` is `1.0` here and `2.0` by the docstring's first case. The import refuses
neither case. Integer `add`, `sub` and `mul` wrap modulo `2 ^ width`. `and`, `or`, `xor` and `not`
of integers, `shift_left`, `shift_right_arithmetic`, `shift_right_logical`, `population_count`,
`clz` and `mulhi` are the IR's integer operators on the words (`mulhi` is the floor of the exact
product over `2 ^ width`). `bitcast_convert_type` between types of one width is an `Op.bitcast`
step, every bit kept (`TorchLean.IR.Program.step_bitcast_integer`, `step_bitcast_scalar_integer`,
`step_bitcast_integer_scalar`). Integer `reduce_sum`, `reduce_prod`, `reduce_max`, `reduce_min`,
`cumsum`, `cumprod`, `cummax` and `cummin` are trees of the IR's `add`, `mul`, `max` and `min` in
the operand's type: a reduction halves the axis in each round, and a scan combines each entry with
the one `2^k` before it in round `k`. These steps wrap modulo `2 ^ width` or follow the type's
order, and each is associative and commutative, so every grouping gives the left fold's value.
StableHLO leaves integer overflow implementation-defined (the "Errors" section of its
specification), so wrapping, there and in `add`, `sub` and `mul`, is the importer's choice. XLA's
CPU code at JAX 0.11.1 wraps in every case the tests check. Integer `argmax` and `argmin` are the
least position whose entry equals the maximum or the minimum, both such trees, so the first extremum
at every width; Boolean ones compare the values in binary64. Boolean `reduce_max` and `reduce_min`
are `reduce_or` and `reduce_and`.

`reduce_window_sum`, `reduce_window_max`, `reduce_window_min` and `reduce_window` with one add, mul,
max or min as its body pad the operand with the init value (base dilation too). They then fold one
strided slice per window cell into the init value in row-major cell order. `Op.max` and `Op.min` are
IEEE maximum and minimum, so `+0` is above `-0` whatever their order, as XLA's CPU code computes.
`lax.reduce` with another init or body is refused (its documentation calls the result undefined).
`cbrt` is IEEE `rootn(x, 3)`, FloatLib's correctly rounded `rootn x 3` in the operand's format, and
`nextafter` is C's, FloatLib's `nextUp` and `nextDown`: these are the rules of `Op.cbrt` and
`Op.nextafter` at the configured binary formats. `reduce_precision(x, e, m)` is two casts through
the FloatLib format with the operand's exponent field and `m` fraction bits, then a signed infinity
and a signed zero outside the normal range of `e` exponent bits, with a NaN word unchanged but for a
reduced float8 one. Where the first cast does not overflow, the two give the operand rounded once
to `m` fraction bits, in every IEEE format (`toReal_reduce_ieee` in `ReducePrecision.lean`). A
float8 operand is reduced in binary32, which holds its values, at the float8 format's exponents:
through the format of one more exponent bit, the float8 bias and `m` fraction bits in IEEE's
encoding, which has the float8 format's subnormal spacing at `m` bits and which no rounding
overflows. The range step follows where `e` is narrower than the float8 exponent field, and the
result is converted back once, so that a value past the format's largest finite one gives its
infinity or, in a format without one, its NaN. Without the range step, a finite result is the
rounding of the operand to that format of `m` fraction bits, one rounding, at every word
(`toReal_reduce_float8`). Unless `e` and `m` are at least the format's,
which gives the operand, a NaN keeps only its sign and payload: the IR has no float8 selection,
FloatLib's conversions give a quiet NaN (IEEE 754-2019 6.2.3), and e5m2's signaling `0x7d` and
`0xfd` give `0x7f` and `0xff`. XLA's CPU code reduces the operand converted to binary16 instead and
converts the result back, and its words differ at 20 of the 96 settings of `e` from 1 to 6 and `m`
from 1 to 4 in the four formats: at `e = 4`, where binary16's range step flushes the e4m3 formats'
values below `2^-6` and takes e4m3fn's from 256 to its NaN, and at `m` below the fraction width,
where it rounds some subnormals at binary16's spacing (`scripts/checks/ir_capture_jax.py`,
`test_float8_reduce_precision_rounds_at_the_format_exponents`).

`select_and_scatter_add`, the transpose of a windowed maximum or minimum, selects in each window,
among its operand cells only as StableHLO's `select_and_scatter` does, the cell that a row-major
left fold keeps, the first of equal extremes. One `scatter` from `+0` then adds each source value to
its window's cell in row-major window order, and a window of padding alone adds nothing. JAX's
default lowering is that operation, and the tests compare with XLA's CPU run of it, which for some
windows padded on their last axis takes the cells in column-major order, so that a tie can select
another cell than StableHLO's order does. JAX's CPU and GPU lowerings, a workaround for XLA crashes,
instead pad the operand with the infinity that no cell exceeds, so a padding cell can tie with an
infinite extreme or follow a NaN and take the source value, and JAX's words differ there on both.
The transpose of a windowed maximum of `[-inf]` with window 2, padding `((1, 0),)` and source
`[1.0]` is `[1.0]` here, from StableHLO's operation, and `[0.0]` in JAX on either target.
`select_and_gather_add`, the tangent of a windowed maximum or minimum, folds the pairs of operand
and tangent entries as `reduce_window` folds entries, from the init pair `(-inf, 0)` (`(+inf, 0)`
for a minimum), the padding being init pairs. Its result is the tangent of the first of equal
extremes. StableHLO leaves the tree of a window's fold open but not the order of its leaves, the
window's entries in row-major order with init values interspersed. With `ge` or `le`, which keep
their left argument at a tie, every such tree selects the first of equal extremes, so the tree
decides only the cell of a window holding a NaN or whose extreme is the init value, and the rounding
of a sum. XLA's CPU loop for `reduce_window` and `select_and_gather_add` skips the padding and, for
some padded windows, takes the cells in another order, neither of which StableHLO allows, so its
words can differ there. `cumlogsumexp` is the `reduce_window` of lax's `logaddexp` over each prefix
(each suffix for `reverse`) that JAX lowers it to, folded the same way. The body is
`max(a, b) + log1p(exp(-|a - b|))`, or `a + b` where `a - b` is NaN, so the result is the left fold
of each prefix, or of each suffix from its first entry. XLA's CPU code folds in that order along up
to 17 entries; along a longer axis it combines blocks of 16 entries, and its words can differ.

`triangular_solve` is substitution (the dot products are left folds from `+0`), `cholesky` the
left-looking column algorithm, NaN everywhere unless every pivot is positive, `cholesky_update` the
factor of `R^T R + w w^T`, `symmetric_product` `alpha (a a^T) + beta c` and `custom_linear_solve`
its solve jaxpr. `householder_product` and `ormqr` of binary32 and binary64 operands apply the
reflectors `H_i = 1 - tau_i v_i v_i^T` that the columns of `a` store one at a time, in the order of
`Q = H_0 ... H_{k-1}`: `c - tau_i (v_i (v_i^T c))` on the left and `c - tau_i ((c v_i) v_i^T)` on
the right, the products left folds from `+0` (`householder_product` applies `Q` to the identity's
first `n` columns). `lu_pivots_to_permutation` swaps the entries `i` and `pivots[i]` of `iota` in
turn, its execution failing at a pivot outside `[0, permutation_size)`, for which JAX documents no
result. `tridiagonal_solve` is LAPACK `gtsv`'s elimination with partial pivoting between
neighbouring rows, then back substitution, each operation rounded once. JAX documents the solution
and names an algorithm only under `perturb_singular`, so this is one admissible refinement, the one
JAX's CPU lowering calls; its GPU lowering calls cuSPARSE's `gtsv2` from three rows, and its default
lowering and the GPU's below three rows eliminate without pivoting, so their roundings differ. Both
multipliers of a step are computed and one is selected; the other divides by 1, so that it is finite
at finite operands and the zero cotangent the selection gives it stays zero in reverse mode (a zero
subdiagonal entry would give `1 / 0`, and `0 * inf` is NaN). `cholesky_update` rotates `(1, 0)` in
the branch it does not take, for the same reason. `tridiagonal_solve`'s execution fails at a zero
pivot (in exact arithmetic, at a singular matrix, for which JAX documents no result), and
`perturb_singular=True` is refused. Complex operands of the four are refused. The PyTorch capture
imports `torch.linalg.householder_product`, its alias `torch.orgqr` and `torch.ormqr`
(`aten.linalg_householder_product`, `aten.orgqr` and `aten.ormqr`, through torch.export and FX)
by the same arithmetic, `LinalgImport` of `python/torchlean/_linalg.py`. The special functions are
imported as `Op.special` (above). The other factorizations are
refused.

#### Complex values

A complex value is held in paired-real form, its real and imaginary parts on an appended last axis
of size two. A complex input or output of shape `s` is the IR's input or output of the parts' format
and shape `s + [2]`, and is declared so, as each complex value inside is; `encode` writes complex
arguments and reference outputs in this form, each element's real part before its imaginary part.
`fft` of a binary32 or binary64 operand transforms the trailing axes
that its lengths name. Along the last one, type RFFT is `Op.rfft` and type IRFFT `Op.irfft`, the
transforms that PyTorch's `rfft` and `irfft` import as. Types FFT and IFFT, and RFFT and IRFFT along
the other axes, transform one axis at a time from `Op.rfft` of each part: with `p + iq` the operand
and `a + ib` and `c + id` the transforms of `p` and `q`, each `Op.rfft`'s frequencies followed by
the conjugates of the ones a real signal repeats, the transform is `(a - d) + i(b + c)` and the
inverse `((a + d) + i(c - b)) / n`. An IRFFT transforms its other axes before its last, as
`numpy.fft.irfftn` does. These are TorchLean's sums, not XLA's FFT schedule, so their rounding can
differ. The imaginary parts of frequency 0 and, for an even length `n`, `n / 2` of the last axis are
set to `+0` before `Op.irfft`, as NumPy's `irfft` ignores them; `Op.irfft` multiplies them by a zero
coefficient, which would turn an infinity or a NaN there into a NaN. `real` and `imag` read a part,
`complex` pairs two, `conj` negates the imaginary part, and `add`, `sub` and `neg` act on each part,
which is complex addition, subtraction and negation exactly; so are `reduce_sum` and `cumsum`, the
IR's sum of each part, as XLA's CPU code sums each part. `mul` and `div` are
written by hand from the definitions of the complex product and quotient, as Mathlib states them
(`Complex.mul_re`, `Complex.mul_im`, `Complex.div_re`, `Complex.div_im`): of `a + ib` and `c + id`,
the product's parts are `a c - b d` and `a d + b c`, and the quotient's `(a c + b d) / (c^2 + d^2)`
and `(b c - a d) / (c^2 + d^2)`. The parts are converted exactly to a binary format chosen for their
format (`_wide` in `python/torchlean/_jax_complex.py`): 581 significand bits and 11 exponent bits
for binary32 parts, 4252 and 14 for binary64 parts. There the products of two parts and their sums
are exact, and a quotient is rounded once at a precision, `2 (emax - emin) + 3 p + 3` for parts of
precision `p` and exponents `emin` to `emax`, at which it never reaches or crosses a midpoint of the
parts' format or its overflow threshold unless it is one (`_wide` gives the argument). Each result
is then rounded once to the parts' format (`product` and `quotient`; the wide quotient is the
registered `Op.div`, as the IR's `div` kind has forms only at the named dtypes). For finite operands
and a nonzero divisor, a part is therefore the exact part rounded to nearest with ties to even, an
infinity only where that overflows. Rounding to binary64 first would not do: the real part of
`(31 2^51 + i)(1082401 2^52 + i)` is `2^128 - 2^103 - 1`, which rounds to the greatest finite
binary32 value but to `2^128 - 2^103` in binary64, and that to `+inf`. `(2^127 + 2^127 i) /
(2^127 + 2^127 i)` is `1 + 0i` in binary32. `lax.mul` with an `out_dtype` converts its operands to
that type first, as JAX 0.11.1 lowers it: `_nary_lower_hlo` (`jax/_src/lax/lax.py`) applies
StableHLO's `convert` to each operand, and the documentation of `lax.mul` says that this is
typically how it is done and that some backends may use a custom kernel. A real operand is the
real part, with imaginary part `+0`. A zero or infinite divisor, or an infinite or NaN part, gives
what IEEE 754-2019's operations give the formula without rounding, where a product of finite parts
is finite, so that `(inf + 2^600 i)(1 + 2^600 i)` is `inf + inf i` in binary64, and without the
recovery of an infinite result that C99's Annex G (G.5.1, WG14 N1256) adds:
`(inf + inf i)(inf + 0i)`, `(1 - i) / 0` and `(1 - i) / (inf + inf i)` are `NaN + NaN i`, where C's
`_Cmultiply` gives `inf + inf i` for the first. XLA's CPU code computes both otherwise (below).
`dot_general` is the product extended by linearity: four real contractions of the parts, each the
IR's `contract` in its order, the real part `a c - b d` and the imaginary part `b c + a d`, so a
contraction of one term rounds `a c` and `b d` apart where `mul` rounds once. A real operand is
converted to the complex type first, with
imaginary part `+0`, so that an infinite `d` makes the real part NaN through `0 inf`. XLA's CPU
code (Eigen's complex GEBP) keeps the four contractions apart too. It gave these words for two
matrices from 2 to 7 terms, in binary32 and binary64, eager and under `jax.jit`, and for a vector
operand (`jnp.vdot`, a matrix times a vector) in binary64; in binary32 those differ in the last
place at 5 terms. It contracts one term as its `mul`, from 8 terms it sums some places of the
output in another order, even and odd terms apart in binary32, and the imaginary part it gives a
real operand at an infinite `d` depends on the shape, so the words can differ there.
`conv_general_dilated` is four real convolutions of the parts in the same way, each the IR's
(below), with the kernel's parts as payloads where it is known at capture. XLA's CPU code sums in
another order: its words are these where every sum is exact and differ in most last places
elsewhere (`test_complex_convolution_is_four_real_convolutions`). `eq` holds where both parts are
equal and `ne` where either differs.
`abs` is the IR's `hypot` of the parts: JAX documents the modulus `sqrt(a^2 + b^2)`, and `hypot` is
that exact value rounded once, `+inf` when a part is infinite even if the other is a quiet NaN, as
IEEE 754-2019 9.2.1 specifies it, and a quiet NaN when a part is a signaling NaN. XLA's CPU code
(jax 0.11.1) returned NaN there, was up to two units in the last place away at the random pairs
sampled and gave 0 for the least subnormal pair. `sign` is the phase `z / |z|` that JAX documents,
written by hand from that definition: `a / sqrt(a^2 + b^2)` and `b / sqrt(a^2 + b^2)` of `z = a +
ib`, and `z` itself where both parts are zero, as `sign` keeps a real zero. They are computed in a
binary format of 734 significand bits for binary32 parts and 5330 for binary64 parts (`phase` in
`python/torchlean/_jax_complex.py`, which derives the precision), where `a^2 + b^2` is exact and the
square root and the quotient are each rounded once, and rounding again to the parts' format gives
each part of finite operands the exact part rounded once; an infinite part beside a finite one gives
a NaN part and a zero, as the formula does (`inf + 1i` gives `NaN + 0i`). XLA's CPU code returned
`z` where both parts are subnormal and `1 + 0i` at `2^-126 + 2^-127 i`, the results of zero parts
(we infer that it loses subnormal parts, but have not read where), `0` at the greatest finite pair,
and parts up to two units in the last place away at 1000 random pairs.
`integer_pow` multiplies in the order of JAX's `lax._integer_pow`,
square-and-multiply and then `1 / product` for a negative exponent, as a real `integer_pow` does,
each product the complex `mul` above and the reciprocal its `div`; so each part of `z^2` and `z^-1`
is the exact part rounded once, and the other powers round at each step. `z^0` is `1 + 0i`, at a NaN
too, and `square` is `z^2`. XLA's CPU code differs at 115 of 500 random binary32 pairs in `square`,
by one unit in the last place, and at 142 to 314 in `integer_pow` of the exponents 2, 3, 5, -1 and
-2, by more where a part cancels. `max` and `min` compare the pairs lexicographically, which is all
that JAX documents for complex operands: of pairs without a NaN part, `x` where the real parts
compare so, or are equal and the imaginary parts do, and `y` elsewhere. Where a NaN part decides the
comparison JAX documents nothing and IEEE 754-2019 has no complex maximum, so there the import
follows, as our convention, JAX 0.11.1's lowering (`_minmax_hlo` in
`jax/_src/interpreters/mlir.py`), whose IEEE comparisons are false at a NaN: `y` where the real
parts are unordered, or equal beside unordered imaginary parts. So `max(NaN + 0i, 1 + 0i)` is `1 +
0i`, `max(1 + 0i, NaN + 0i)` is `NaN + 0i` and `max(2 + NaN i, 1 + 0i)` is `2 + NaN i`. XLA gives
the same words. A stable `sort`, as `jnp.sort`, `jnp.argsort` and `lax.sort_key_val` call it, orders
complex keys as JAX's comparator does (`_sort_lt_comparator` in `jax/_src/lax/lax.py`):
lexicographically, the real part first, as the docstring of `lax.sort` states, the parts with `-0`
taken as `+0` and every NaN as one that follows `+inf`, equal entries in their order and every word
kept. That is not NumPy's order, which puts `1 + 0i` before `0 + NaN i`. The import is the sort of
several operands below, the real and the imaginary part of a complex key two keys, and gathers each
part of a complex operand at the order. XLA gives the same words
(scripts/checks/ir_capture_jax_complex.py, `test_complex_sort_orders_pairs_as_jax_does`), and an
unstable sort is refused. `reduce_prod`, `reduce_max` and `reduce_min`, and `cumprod`, `cummax`
and `cummin` over each prefix or suffix, are
left folds of that complex product or of `max` or `min` from the first entry, the accumulator
first, over the reduced coordinates in row-major order, last to first for a reverse scan, as the
real folds are; over no entry they are `1 + 0i`, `-inf + 0i` and `inf + 0i`, the initial values of
the `reduce` and `reduce_window` that JAX lowers them to. Those values are not identities here:
`(1 + 0i)(inf + 0i)` is `inf + NaN i` and `max(-inf + 0i, -inf - i)` is `-inf + 0i`, and `max` is
not associative at a NaN, where the docstring of `lax.reduce` leaves XLA's result undefined. So
the folds do not start from them: the product of the one entry `inf + 0i` is `inf + 0i` and the
maximum of `-inf - i` and `-inf - i` is `-inf - i`. At parts drawn from small integers, signed
zeros, infinities and NaN, XLA's CPU code gave the words of the fold from the initial value along
2 to 32 entries, over two axes too, and for a cumulative fold along 1 to 8, folding each suffix of
a reverse one first to last (`test_complex_folds_start_from_the_first_entry`); it returns a lone
entry of a reduction, and past 32 entries its extrema follow no fold. The complex
`reduce_window_sum`, `reduce_window_max` and `reduce_window_min`, and `reduce_window` with one add,
mul, max or min as its body, fold each window of the operand, padded and dilated with the init
value's parts, from the init value in row-major order of its cells, by that sum, product, `max` or
`min`, as the real window reductions do; the first three start from `0`, `-inf + 0i` and `inf + 0i`,
the init values JAX 0.11.1 lowers them with. XLA's CPU words are the same where skipping an init
cell or reordering the cells changes nothing, and differ elsewhere, as for real operands
(`test_complex_window_reductions_fold_from_the_init_value`). `exp`, `expm1`, `log`, `log1p`,
`sqrt`, `rsqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `asin`, `acos`, `atan`, `asinh`,
`acosh` and `atanh` are `Op.complex` of that function: FloatLib's `ExecComplex` function of the
pair of parts, each part of a finite argument the part of the exact result rounded once, with the
special values and branch cuts of C23 7.3 and Annex G, where the sign of a zero part selects the
side of a cut (`TorchLean/Tensor/ConfiguredBinary/Complex.lean` says where FloatLib proves the exact
result to be the function's value). `sqrt` is the principal square root: of `z = a + ib`,
`T + i copysign(U, b)` where `a` is not negative, `-0` included, and `U + i copysign(T, b)` where it
is, for `T = sqrt((|z| + |a|) / 2)` and `U = |b| / (2 T)`, and `+0 + ib` at zero. An infinite `b`
gives `+inf + ib`, beside a NaN too, `a = +inf` the real part `+inf` and `a = -inf` the imaginary
part `copysign(inf, b)`. XLA's CPU code gave `0 + 2i` for `sqrt(-4 - 0i)` and the root of the
conjugate at a negative subnormal `b`, the results of a `+0` part (again an inference from its
words), `0` for a subnormal part of the root, `NaN + NaN i` where C's results are infinite beside a
NaN, and a part one or two units in the last place away at about a quarter of the parts of 500
random pairs. `rsqrt` is `1 / √z`, as JAX documents `lax.rsqrt`, each part the exact part rounded
once, not `1` over the rounded root. Its values at `0` and `inf`, which JAX does not name, are
C23's `(1 + 0i) / (0 + 0i)`, `+inf + NaN i`, and the limit `+0 - i copysign(0, b)`. XLA's CPU code
gave `inf + NaN i` and `0 - 0i` there, a nonzero real part at `-4 ∓ 0i`, where `1 / √z` is
`±0.5i`, and a part a unit in the last place from the correctly rounded one at `3 + 4i`
(`test_complex_sqrt_rounds_its_definition_once`, `test_complex_rsqrt_rounds_each_part_once`,
`test_complex_functions_round_each_part_once`). For the other sixteen functions, at 500 random
pairs XLA's CPU parts differed from the exact parts rounded once at 2 to 449 of them for each
function and part, by at most 7 units in the last place except in `expm1` (37 and 159 in
binary32's real and imaginary parts, 58 and 26236 in binary64's), binary32's `tan` (17 and 15) and
the real part of `sinh` and imaginary part of `cosh` (484276 and 605736 in binary32, 62174 and
69129 in binary64); on the cuts its `asin`, `acos`, `acosh` and `asinh` gave the value of the
`+0` side at `-0` too, `asin(2 - 0i) = π/2 + 1.317i` where C23 gives `π/2 - 1.317i`. `pow` of a
complex base is `Op.complexPow`, FloatLib's `ExecComplex.pow`, C23's `cpow` `exp (w log z)`: at a
finite base other than zero and a finite exponent, each part of a result FloatLib certifies is the
exact part rounded once, the sign of the base's zero imaginary part selecting the side of the cut of
`log`. `z^0` is `1 ± 0i`, at a NaN too, with the sign of the base's imaginary part; a zero base
gives `+0 + 0i`, `inf + NaN i` or `NaN + NaN i` as the real part of the exponent is positive,
negative or zero; any other infinite or NaN part gives `NaN + NaN i`. An integer exponent is
converted to the complex type first, as JAX documents for `lax.pow`. `exp2` of a complex `z` is
`2^z`, as JAX documents `lax.exp2`, the power of the base `2 + 0i`. XLA's CPU code gave
`NaN + NaN i` at a zero base, `inf + 0i` for `inf^1`, a nonzero real part for `(-1 ± 0i)^0.5` and
`(1 + i)^2`, and parts 4 and 2 units in the last place from the exact parts rounded once for
`(1 + 2i)^(3 - i)` in binary32 (`test_complex_pow_rounds_each_part_once`). Over the reals the
pullback of `Op.complexPow` is the adjoint of its derivative in both operands where the base is off
the cut, and in the exponent wherever the base or the exponent is nonzero
(`Op.complexPow_backward_eq_adjoint`, `Op.complexPow_backward_eq_adjoint_right`); JAX's own VJP of
`pow` is holomorphic, without the conjugate. In every IEEE configured format, at each entry that
meets FloatLib's premises, each part of both operators is the exact part rounded once
(`Op.complex_forward_round`, `Op.complexPow_forward_round` and their `_value` forms in
TorchLean/NN/Op/ComplexRound.lean). `logistic`, `cumlogsumexp`, `clamp` and
`atan2` of complex operands, which JAX 0.11.1 traces and runs, are refused, and so are `argmax` and
`argmin`, which it traces but does not lower.
A complex constant, a literal or a captured array, is the pair of its parts.
A conversion to a complex type converts the parts, a real value getting imaginary part `+0`.
`reshape`, `squeeze`, `transpose`, `broadcast_in_dim`, `slice`, `rev`, `concatenate`, `stack`,
`split`, `unstack`, `tile`, `copy` and `pad` move the pairs, a `reshape` with `dimensions` permuting
the operand's axes first, and the pieces of a `split` or `unstack` are complex.
`select_n`, `gather`, `dynamic_slice`, `dynamic_update_slice` and `scatter` move each part at the
same real predicate or indices, `scatter-add` and `scatter-sub` add or subtract each part of the
updates, and a `FILL_OR_DROP` gather fills each part with that part of its fill value, by default
NaN converted to the complex type, `NaN + 0i`. `scatter-mul` multiplies each slice by its windows
of updates in turn, in the row-major order of the index vectors in which a real scatter applies
them, each step a gather, the complex `mul` and a replacement in the scatter's mode; StableHLO
leaves the order to the implementation (`schedule`), and XLA's CPU words are these where every
product is exact (`test_complex_scatter_products_multiply_in_index_order`). A scatter with batching
dimensions (`jax.vmap`) is refused there. Every other primitive with a complex operand or result is
refused.
`stack` is one `cat` of its operands, each reshaped to an axis of size one at the new axis, and
`unstack` is one slice of width one per output, reshaped to drop that axis.

#### Runtime indexing

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

The same primitives at integer and Boolean operands are the registered `gather` and `scatter` at
that element (`Op.Element`): an integer scatter sums, subtracts and multiplies modulo `2^width` and
takes the type's maximum and minimum (`Op.rule?_scatter_integer`). JAX has no Boolean scatter but
`scatter`, `scatter-max` and `scatter-min`, and the IR's Boolean scatter only replaces, so a Boolean
maximum or minimum runs on the uint8 words 0 and 1 and comes back as the words that are not 0. A
`FILL_OR_DROP` gather fills with the `fill_value` that `lax.gather` binds, by default the least
signed value, the greatest unsigned one or `True`; a fill value that the type does not hold, such
as `1.5` for int32, is refused. XLA's CPU code agrees at every width of both signs, eager and under
`jax.jit`, in the test of these primitives (`test_runtime_indices_of_integers_and_booleans`).

The JAX importer records supported operations, shapes, dtypes and casts. Reference execution uses
TorchLean and FloatLib arithmetic: division is IEEE division, `rsqrt` is `1 / sqrt x` rounded twice
(in binary64 `rsqrt 3` is one ulp above the correctly rounded `rSqrt 3`), and results underflow
gradually. `div`, `pow`, `neg`, `rsqrt` and `erf` map to IR operators, and integer `div` rounds
toward zero. Integer `rem` is C's `%`, `x - trunc(x / y) * y` modulo 2^width (StableHLO's
`remainder`), and integer `abs` is `-x` below 0 modulo 2^width, so the least signed value is its own
magnitude. StableHLO leaves `rem` implementation-defined at the least signed value and -1 and at a
zero divisor; XLA's CPU code gives 0 at the first and `x` at the second, and the IR gives 0 at the
first and, as its integer division does, fails to execute at the second
(`test_integer_remainders_and_magnitudes`). Binary16 and bfloat16 `pow`, `rsqrt`, `erf`,
contractions and convolutions are computed in binary32 and rounded once, as XLA computes them. On
8192 samples per operator and format, the IR returns XLA's words for every binary16 `pow` and
`rsqrt` sample and every bfloat16 `rsqrt` sample. It returns them for 8191 bfloat16 `pow` samples:
XLA's result is its own binary32 power rounded once, and FloatLib's binary32 power differs from it.
The IR has no binary `erf`, so a graph using it does not execute. `integer_pow` becomes the products
XLA computes, square-and-multiply in JAX's order with a final division for a negative exponent. On
4096 samples, XLA's `integer_pow` for the exponents 3, 5, 7 and -3 equals that schedule rounded at
each step in bfloat16, binary16 and binary32.

`expm1`, `log1p`, `exp2`, `log2`, `tan`, the hyperbolic functions and their inverses, `asin`,
`acos`, `atan` and `atan2` of binary16 and bfloat16 are computed in binary32 and rounded once, and
an `accuracy` request is refused. The binary `log2` is `RN(RN(ln x) / RN(ln 2))`, not a correctly
rounded `log2`. XLA's bfloat16 `log2` is within one ulp of the IR's on the tested samples, but it is
not XLA's binary32 `log2` rounded once. `floor`, `ceil`, `round` with `TO_NEAREST_EVEN` and `rem`,
the IR's `fmod`, are exact in the format, and so is `round` with `AWAY_FROM_ZERO`, `lax.round`'s
default: `t + copysign(1, x)` where `|x - t| >= 1/2` and `t` elsewhere, for `t` the IR's `trunc` of
`x`. `lax.sort` with `is_stable` and floating keys keeps `-0` and `+0` in input order and puts NaN
last, which is the IR's stable ascending sort. With several operands, as from `jnp.argsort`,
`lax.sort_key_val`, `jnp.lexsort` and the JVP of `jnp.sort`, it orders each fiber lexicographically
by the first `num_keys` operands, the first primary, and gives every operand that permutation, as
`lax.sort` documents: the importer sorts stably by each key in turn, from the last to the first,
composes the orders from `Op.sort`'s index output and gathers every operand at the result, its words
unchanged. The comparator compares an integer or Boolean key as it is, and so does `Op.sort` at the
key's element: integer words by their values, signed or unsigned, and `false` below `true`. An
unstable sort is refused. `lax.sign` keeps `-0` and NaN, as it documents, so it is a `where` around
`copysign`, not the IR's `sign`, which maps both to `0`. `lax.sign` of integers is `-1`, `0` or
`1`.

#### Ordering and random draws

`eq_to`, `lt_to` and `le_to`, which JAX's sort and search comparators use (so `jnp.searchsorted`,
`jnp.digitize` and `jnp.histogram` reach them), are StableHLO's `compare` in TOTALORDER for floating
operands and in the plain order for integer and Boolean ones. StableHLO's spec names IEEE 754-2019's
totalOrder (section 5.10) without saying more, and the importer takes that relation: `-0` below
`+0`, a NaN below every number if its sign is negative and above if positive, and NaNs of one sign
ordered by quiet bit and payload. It reads each word as a signed integer of its width, flips the
bits other than the sign where the sign is set, and compares those integers, in binary16 to
binary64, bfloat16, float8_e5m2 and float8_e4m3fn. XLA's CPU code agrees at every pair of float8
words and at special and random words of the other formats, eager and under `jax.jit`. The one NaN
of a fnuz format, the word that would be `-0`, which IEEE 754 does not order, is placed above every
number, as the IR's `sort` places a NaN. XLA's CPU code places it below
(`test_total_order_comparisons`).

`lax.top_k` is the IR's stable `topk`, its indices the second output of `Op.topk`: equal entries
keep their input order, the order `lax.top_k` documents for `is_stable`. XLA's CPU code ranks in
IEEE totalOrder, `+0` above `-0` and a NaN by its sign and payload, and `lax.top_k` documents
nothing for NaN. For `k > 0` execution therefore fails where a fiber holds a NaN, or both a `-0` and
a `+0` (a sufficient condition, stricter than needed), and elsewhere the two orders agree. Integer
and Boolean entries, `false` below `true`, are the first `k` of the stable `sort` of each fiber's
entries reversed by bitwise or logical not, which reverses the type's order, so equal entries keep
their input order at every width; XLA's CPU code gives the same values, and with `is_stable` the
same indices, on every tested fiber (`test_integer_and_boolean_top_k`). `cumsum` and `cumprod` are
the IR's left folds from `+0` and `1`, the order TorchLean chooses and the capture writes
(`{"order": "leftFold"}`). XLA's CPU code combines the prefixes associatively, so the words agree
where each partial result of either grouping is exact and zero signs and exceptional values are
handled alike. `cummax` and `cummin` give the value and zero sign of the running IEEE maximum and
minimum, a NaN propagating as the latest NaN's word. They are computed by the IR's scan, which takes
a later equal entry as ATen does, and a second scan of a `+0` flag (a `-0` flag for `cummin`) that
turns a zero result into `+0` (`-0`).

`approx_top_k` (`lax.approx_max_k` and `lax.approx_min_k`) documents an approximate result. Off
TPU, JAX lowers it to XLA's `ApproxTopK` fallback, a stable sort of each fiber with its indices
under `>` (`<` for the minimum), cut to the result's extent. That is the exact top or bottom of the
fiber, equal entries, `-0` and `+0` among them, in input order, and the importer gives the IR's
stable `topk` for it. Under `>` a NaN is equivalent to every entry, so execution fails where a fiber
holds a NaN. XLA's CPU code rewrites the sort into a reduction that keeps a later one of equal
extremes where the result keeps one entry per fiber, and for the largest binary32 entries cut
short into its totalOrder `TopK`, so its indices of ties and its order of `-0` and `+0` can differ
there.

`rng_uniform` draws from TorchLean's generator, not XLA's stateful one, whose stream the API leaves
to the implementation. It is `Op.randUniform` under the key of its node's position, as a PyTorch
capture keys `rand`: each draw `u` in `[0, 1)` rounded once to the dtype, then
`low + (high - low) * u` in the dtype's arithmetic, or `2 * (low / 2 + (high / 2 - low / 2) * u)`
where `high - low` overflows. Rounding can reach `high`, and such an entry is `nextafter(high,
low)`, the greatest value below `high`, so each entry lies in `[low, high)`, as JAX documents.
An unrolled iteration of a `scan` has nodes of its own, so its draws have keys of their own, and a
scan whose body draws is unrolled, not one `Flow.scan`. A region of a flow is one set of nodes, so
the body of a `while` that the capture does not decide draws the same entries in every iteration.
Execution fails unless both bounds are finite and `low < high`, since JAX leaves `high <= low`
undefined. Integer bounds of at most 32 bits give
`low + mulhi(d, high - low)`, `d` the 32-bit word of the binary64 draw `u = d / 2^32`: the high half
of the product, `floor(d (high - low) / 2^32)`, so each entry lies in `[low, high)`, and each value
takes `floor(2^32 / n)` or `ceil(2^32 / n)` of the words for `n = high - low`. Execution fails
unless `low < high`. Wider integer bounds are refused, and so is `rng_bit_generator`, whose bits are
a function of its key operand, while TorchLean's random operators take their key at capture.

#### Branches and loops

JAX's `cond`, `scan` and `while` are lowered by partial evaluation first: what is decided at capture
is imported as straight-line steps, as the rest of the function is, and only what the capture does
not decide becomes a flow. A `scan` is its body once per iteration, or one `Flow.scan` where its
iterations would all be imported alike with a flow in each (below), a `cond` whose index is known
at capture is the branch that the index selects, and a `while` whose test reads only values known at
capture is its body once per iteration that it runs. Any other `cond` or `while` makes the artifact
a flow artifact ([above](#the-artifact)), whose blocks are the runs of steps between them, and the
state before each block, flow and region end holds the values that the rest of the computation
reads, constants aside, which each block that reads one holds itself. A computation without them
stays in the straight-line artifact format. A value is known at capture when it is a literal, a closed-over
array, or an integer or Boolean value that shape operations, comparisons, selections and integer
`add`, `sub`, `mul`, `neg`, `max` and `min` compute from known values alone (`known_value` in
`python/torchlean/_jax_reference.py`), which JAX folds at capture: `bound[1] - bound[0]` for a
closed-over `bound`, the counter of an iteration of a `scan` or of a `while` that the capture
decides, or `platform_index` (below). A value computed from an input, or converted from a floating
value, is not known, and neither is a floating value that a loop computes: a `while` whose test
reads one, as the halving loop below does, remains a flow even if the initial values are
constant. The scan tests compare the fallback artifact with one explicitly unrolled at capture and
run the resulting program against JAX.

The lowering's contract is two rules. Partial evaluation comes first: what the capture decides is
lowered as straight-line steps, as the importer lowered it before it wrote flows. Dead-code
elimination comes second, in a computation with a flow: every node and every `cond` or `while` that
gives no value the computation returns, directly or through what reads it, is dropped before the
flow is written (`prune` in `python/torchlean/_flow.py`), as JAX's `pe.dce_jaxpr` drops an equation
without effects none of whose outputs is used. A flow that stays keeps all of its results, so every
block has an output, and a computation none of whose flows gives a returned value, such as a `cond`
whose branches return nothing, has the artifact of its other nodes that give one, without a flow
(`test_what_gives_no_returned_value_is_dropped_before_the_flow`). A dropped `while` does not run, so
one that would not stop does not fail, and a dropped node does not fail at execution. Where the
capture decides all control flow, the artifact keeps every node, used or not, as before. The
result is a straight-line artifact when capture resolves the control flow or no flow contributes
to the returned values. Otherwise use `IR.loadFlow` or `TorchLean.JAX.load` to load the flow
artifact. The control-flow tests check the resulting branch and loop structure and compare it
with reference execution on concrete inputs.

`cond`, which `lax.cond` and `lax.switch` bind, is `Flow.cond` when its index is not known at
capture: each branch is a region on its operands, and only the branch that the index selects runs,
as StableHLO's `case` selects it: branch `index` for `0 <= index < n` and branch `n - 1` otherwise
(`lax.switch` clamps its index first). So an unselected branch never fails, as JAX never runs it:
neither its integer division by zero, nor its conversion of an infinity or a NaN to an integer type,
nor its failing read of `rng_uniform`'s bounds or `top_k`'s NaN, nor its loop that never stops. A
selected branch fails where it would outside a `cond`. An index known at capture selects its branch
there, and only that branch is imported, without a flow, so a branch that is never selected, as one
that a scan's counter rules out in every iteration, may hold what the importer refuses.
`lax.platform_dependent`, which `jnp.diagonal` and `jnp.diag` use for a square matrix, binds such an
index, `platform_index`: the position of the branch for the platform that JAX lowers for, or else of
the default branch. The IR's executions stand for JAX's on the CPU and on CUDA, so `platform_index`
is the branch that both of them select, and it is refused where they select different branches or
one of them none.

`scan`, which `lax.scan` and `lax.fori_loop` with static bounds bind, imports its body once per
iteration, as `lax.scan`'s reference implementation loops over it: iteration `i` reads the
constants, the carry and entry `i` of each scanned operand along its leading axis (`length - 1 - i`
in reverse), and its other outputs are entry `i` of the stacked outputs. So its steps are the
unrolled computation, `length` times the body's size, and each iteration's counter is a constant,
which a `cond` or `while` of the body reads at capture: `lax.cond(c[0] < 2, ...)` on a counter
`c[0]` selects its branch in each iteration there. A `cond` or `while` of the body that the capture
does not decide is a flow in each iteration. Where no iteration would decide at capture how a value
it reads is lowered, and the body holds a flow, every iteration would be imported alike, and the
scan is one `Flow.scan` of `length` iterations of one region of its body (`looped` and `trial` in
`python/torchlean/jaxpr.py`): it carries a counter `i` from 0 where there are scanned operands or
stacked outputs, reads entry `i` of each scanned operand by `Op.gather` and writes its outputs
there, into stacked outputs that start at zeros, by `Op.scatter`, as PyTorch's `map` does (below,
`_flow.entry`), so that its document is the size of one iteration, as a `lax.scan` of
`lax.cond(c > 0, ...)` on its carry is at 100,000 iterations
(`test_a_scan_of_conds_on_its_data_is_one_flow`). The carry that an unrolled iteration may know at
capture is the carry known before the first one, as `lax.fori_loop`'s counter is, and the carry
whose new value the body computes from such carry and constants alone, by the integer and Boolean
equations that `known_value` folds, to a fixed point. The region reads that carry unknown, and
wherever its import finds a value unknown at capture (a `cond`'s index, the operands that decide a
`while`, a convolution's kernel, a nested scan's carry), the trial notes the nodes at which the fold
stopped. Where they are all such carry, an unrolled iteration may decide what the region does not,
and the scan is unrolled, as it is where the program reads a last carry that folds so after the
scan, or where the region draws from `rng_uniform`, whose unrolled iterations draw under keys of
their own. A nested scan's trial may decide otherwise where the enclosing scan's carry is known, so
the enclosing scan is unrolled where any of the nested trial's folds stopped at such carry. So
`lax.fori_loop(0, 1000, lambda i, c: flip(c), c)`, whose counter decides no `cond`, is one
`Flow.scan`, and a `lax.cond(c[0] < 2, ...)` on the counter, a `cond` on carry that each iteration
doubles from the counter, or one after the scan on its last counter keeps the scan unrolled
(`test_a_scan_whose_known_carry_decides_a_flow_is_unrolled`). A scan whose scanned operands are
known at capture in part is unrolled even where they decide none of its flows, and so is one with a
complex or float8 scanned operand or stacked output, or a zero-sized carry, scanned operand or
stacked output. The `Flow.scan` is a trial: where anything in its region is refused or fails while
the region is built, its nodes are dropped and the scan is unrolled, so building it adds no refusal
(`test_a_scan_that_its_flow_refuses_is_unrolled`); a region that is built and that the flow's
writing (`_flow.lower`) then refuses is not retried unrolled, and no such region is known. The
flow of a region runs its parts in a `seq` of the flows of its first and second halves
(`_sequence` in `python/torchlean/_flow.py`), which `Flow.eval` runs in order in any bracketing, so
the 1,000 iterations of an unrolled scan with a flow in each nest a few levels, not 1,000, and
Python's `json` and `copy.deepcopy`, which recurse, take the document. A scan of length zero runs no
iteration: its outputs are the carry and stacked outputs of zeros, without entries.

`while`, which `lax.while_loop` binds, as does `lax.fori_loop` when its bounds are traced, is
decided at capture when its test reads only values known there. The test reads some constants and
carry entries, the body computes those entries from more of them, and every one must be known at
capture, and every primitive of the test and of those parts of the body one of the integer and
Boolean ones that `known_value` folds, whose results are the IR's, or a `jax.jit` call of them
(`exact`). The importer counts the iterations by running the test and those parts of the body with
JAX, one primitive at a time (`trips`), and imports the body once per iteration that it runs, as the
Python loop of `lax.while_loop`'s docstring runs it. The test leaves no node, and a loop that runs
no iteration, as that of `lax.fori_loop(0, 0, ...)`, has neither its test nor its body imported. The
while loops of one import unroll at most 65,536 iterations into its document in all, nested ones
included: a loop that would run more is a flow, and so is every later loop that runs an iteration,
while a later loop of no iteration is still decided. A scan's trial that is dropped (above) gives
back the iterations it counted, as it drops its nodes, so the importer may count more than 65,536
iterations to write one document.

Any other `while` is `Flow.whileLoop`: its test and body are regions on their constants and the
carried values, and it runs the body while the test holds, as the Python loop of `lax.while_loop`'s
docstring runs it, whatever the test reads, such as an input, the remainders of `jnp.gcd` and
`jnp.lcm`, a floating or complex value or a nested `cond`, `scan` or `while`. Each loop runs at most
the artifact's `fuel` of iterations each time it is entered, 65,536 unless `export`'s `fuel` names
another, a Python or numpy integer that is not negative (`_flow.natural` refuses a Boolean, a tensor
or an array before the capture): one whose test still holds then fails at execution
(`while: out of fuel`), so a loop that never stops fails instead of running on. The IR's arithmetic
in a loop is IEEE 754's, as everywhere: a binary32 value halved from `2^-126` while it is positive
stops after 24 iterations, and after 1 in XLA's CPU code, which flushes subnormals to zero, so a
test that reads a floating value is not counted at capture. `jnp.gcd` of vectors whose lanes stop at
different iterations, and `jnp.lcm(0, 0)`, compute `lax.rem` by zero in lanes that a `select_n` then
discards: JAX documents that remainder as implementation-defined, and the IR's integer division
fails at a zero divisor, so their execution fails.

A complex carry, scanned operand, stacked output or constant of a loop is carried, cut and stacked
in its paired-real form, and a test may read a complex value
(`test_complex_loops_carry_and_stack_pairs`). XLA compiles an eager `cond`, `scan` or `while` whole,
as it compiles a `jax.jit` call, and `torchlean.jaxpr.evaluate` runs the selected branch, each
iteration of a scan, and a while loop's test and body in turn, one primitive at a time. A flow's
document has no `computation`, `jit_computations` or `primitive_index`, since the checker emulates
XLA's CPU code (below) only for an artifact without control flow, and `TorchLean.IR.load` refuses
it, while `TorchLean.IR.loadFlow` loads it.

torch.export records `torch.cond`, `while_loop` and `map` (`torch._higher_order_ops`) as
higher-order operators whose functions are graphs of their own, and the PyTorch capture lowers
them to the same flows (`python/torchlean/_hop.py`). Each graph's placeholders read its node's
operands, lifted parameters, buffers and constants among them, and a graph is lowered afresh each
time a node reads it. `torch.cond` is `Flow.cond` on its predicate's truth, as Python's `if pred`
reads a one-element tensor: a Boolean as it is and another value compared with zero, at shape `[]`
or `[1]`. Branch 0 is the false graph and branch 1 the true graph, and only the selected one runs,
so an unselected branch's integer division by zero or loop that never stops never fails. A
predicate known at capture selects its graph there, and only that graph is checked and imported,
without a flow, so the other graph may hold what the capture refuses, as an unselected JAX branch
may. `while_loop` is `Flow.whileLoop`, which runs the body while the test's Boolean scalar holds,
as the Python loop in `while_loop`'s docstring runs it, and at most the artifact's `fuel` of
iterations each time it is entered, 65,536 unless `export_torchlean_graph_json`'s `fuel` names
another, a Python or numpy integer that is not negative (`_flow.natural` refuses a Boolean, a tensor
or an array before the capture): its trip count may depend on the inputs. Its additional inputs, the
tensors its functions close over, are read by the test and the body and not carried. `map` is
`Flow.scan` over the leading axis of its mapped operands: the state carries an `int64` counter `i`
from 0 and the stacked outputs, which start at zeros, and iteration `i` reads entry `i` of each
mapped operand by `Op.gather` and writes its outputs to entry `i` of the stacked outputs by
`Op.scatter`; PyTorch refuses a map of length zero. Its body may hold a `torch.cond` or a
`while_loop` of its own, a flow in the body of the `Flow.scan`. A flow's region is one set of nodes,
so a `map` whose body draws from TorchLean's generator (`rand`, `randn`, `randperm`, dropout in
training) is unrolled instead, iteration `i` reading and writing entry `i` at the constant `i`, and
each iteration draws under keys of its own, as in a JAX scan whose body draws; the body of a
`while_loop` draws the same entries in every iteration, as that of a JAX `while` does. PyTorch
2.11's export records a call of a function marked by `mark_compile_region` as `invoke_subgraph`,
whose graph is inlined at each call, and 2.13's export inlines the calls itself. Each lowering of a
graph draws under keys of its own, its dropout masks too. The tests are `PyTorchFlowTests`
(`scripts/checks/ir_capture_pytorch_flow.py`), on the CPU.

#### Contractions

`conv_general_dilated` is the IR's convolution. It holds a kernel known at capture as its payload:
a constant, or a value that JAX computes from constants by `squeeze`, `reshape`,
`broadcast_in_dim`, `transpose`, `rev`, `slice` and `concatenate`, which move its words, and `neg`
and `conj`, which flip sign bits (IEEE 754-2019 5.5.1). The importer computes such a kernel at
capture with JAX's own primitives, as `jnp.convolve` and `jax.scipy.signal.convolve2d` flip and
broadcast their kernel and `jnp.correlate` broadcasts it and conjugates a complex one
(`test_convolutions_execute_with_their_payload`). Any other kernel, an input or a value computed
from one or by arithmetic, is the convolution's second operand, in JAX's layout of output channels
and then each group's input channels (`test_convolutions_take_a_kernel_not_known_at_capture`). A
padding cell is a zero times its kernel entry, as StableHLO's specification of `convolution` pads
the input with zeros before its dot products, so an infinite or NaN kernel entry over padding gives
NaN. XLA's CPU code does not always multiply the padding. Of 15 shapes sampled, with one to three
spatial axes, strides, dilation and groups, 8 of them in float8 too, it gave that NaN in binary32,
binary16, bfloat16 and float8 at all but a 5 × 5 kernel over a 3 × 3 input, and in binary64 only
for a 5 × 5 kernel from 16 to 32 channels over a 20 × 20 input. Elsewhere it skips the padding, so
its words can differ at an infinite or NaN kernel entry over padding.

XLA's CPU code converts binary16 and bfloat16 operands of a contraction or a convolution to
binary32, computes there and converts the result back once. On standard normal operands with 2, 4,
32, 256 and 1024 products per entry, XLA's `x @ y` is the exact sum of products rounded once in
every entry, and the IR's words equal XLA's in every entry but one of 640, where the IR's binary32
fold and XLA's binary32 summation order round differently. The IR's convolutions with 4, 32 and 256
products per entry equal XLA's in 432 of 432 entries.

`ragged_dot_general` is imported in its three modes. With a ragged free axis of the left operand,
the form `lax.ragged_dot` binds, the result at a position is the contraction with that position's
group of the right operand: `+0` in no group, and the first group's where a negative size makes
groups overlap. JAX's reference implementation, which XLA's CPU backend runs, contracts with every
group, the left operand masked to zero outside it, so an infinity in one group of the right operand
makes its column NaN in every group, and overlapping groups add. JAX documents only the shapes of
the three modes, so the selection is TorchLean's reading, and the import refuses neither case. With
a ragged contracting axis, entry `j` of the new leading axis is the contraction with both operands
set to zero outside group `j`. With a ragged batch axis the result is `dot_general`'s and the sizes
are not read. Both follow JAX's reference implementation. Group sizes of shape `p + [g]` give each
position of their leading axes its own groups, as in JAX's reference implementation: the batch axes,
then the left operand's free axes before the ragged one, or its contracting axes before it. A group
offset is refused: JAX 0.11.1 cannot bind an array as that parameter, and its reference
implementation raises `NotImplementedError` for one. `scaled_dot` is the decomposition of its
`xla.scaled_dot` composite: along each contracting axis, entry `k` of a scale covers the `m`
positions from `k * m` of its operand, each operand and its scale are rounded to bfloat16 and
multiplied there, and the products are contracted as by `dot_general`, converted first to a wider
result type. A result type that does not hold every bfloat16 value is refused.

#### Float8 arithmetic

The float8 formats of FloatLib's catalog, `float8_e5m2` (IEEE specials), `float8_e4m3fn` (one NaN of
each sign, no infinity), `float8_e4m3fnuz` and `float8_e5m2fnuz` (one NaN, no infinity, no `-0`),
import as the IR's binary descriptors, through `convert_element_type`, as `scaled_dot` operands and
scales, whose rounding to bfloat16 is exact, as operands of a `dot_general`, `ragged_dot_general` or
`conv_general_dilated` with a `preferred_element_type` that holds them or with a float8 result
(below), and in movement (`reshape`, `squeeze`, `split`, `unstack`, `stack`,
`transpose`, `broadcast_in_dim`, `concatenate`, `slice`, `pad`, `rev`, `copy`), `abs`, `max`, `min`,
`add`, `sub`, `mul`, `square`, `reduce_sum`, `reduce_prod`, `cumsum`, `cumprod`, `integer_pow`,
`scatter-add`, `scatter-sub` and `scatter-mul`, the IR's operations in the format, each sum,
difference and product rounded once to nearest, ties to even, a reduction the IR's left fold from
`0` or `1`, a scan its prefixes (reversed, the suffixes folded last to first, between two flips),
`integer_pow` JAX's square-and-multiply and a scatter its updates in turn, in the order of the other
formats (above). XLA's CPU code computes that arithmetic in binary16 and converts each result back,
also inside a `jax.jit` fusion, and its words are the IR's at every word and pair of words of each
format, at seven powers of every word, in sums and products of rows of 32 words and the forward
scans of rows of 16, and in 512 updates scattered into 16 entries and 64 rows into 4, a NaN matching
any NaN. Its sums of more than 32 entries are in another order (below), and its reverse scans fold
each suffix first to last. The IR has no float8 `neg`, comparison, `where` or `div`, so `neg`, the
comparisons, `select_n` and `div` convert their float8 operands to binary32, which holds every
float8 value, and a float8 result back, and so do `sign`, `sqrt`, `floor`, `ceil` and `round`, each
imported as for a binary32 operand, and the reciprocal of a negative `integer_pow`. The conversions
are exact but for a quotient's and a square root's, which binary32 rounds to a value that rounds to
the format as the exact one does (`_Builder.binary32`), and a `-0` of a format without one, which
converts to `0`. `exp`, `exp2`, `expm1`, `log`, `log1p`, `sin`, `cos`, `tan`, `tanh`, `logistic`,
`sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `asin`, `acos`, `atan`, `rsqrt` and `cbrt` are binary32's
function of the converted operand, rounded once to the format. At each of the 256 words of each
format the exact value is a value of the format, more than a relative 2^-17 from every midpoint, or
a midpoint that binary32 holds (`exp2(-10)` in e4m3fn, `exp2(-11)` in e4m3fnuz), so a binary32
result within a relative 2^-17 rounds as the exact value does, and the IR's words are the exact
value rounded once at every word (`_Builder.binary32`,
`test_float8_elementary_functions_round_once_through_binary32`); FloatLib's binary32 `sin`, `cos`,
`sinh`, `cosh`, `tanh` and `logistic` carry no accuracy theorem, and the words are what is checked.
XLA's CPU code computes these in binary16, and its words are the exact value rounded to binary16 and
then to the format but for `asinh`, `acosh`, `atanh`, `asin` and `acos`, which differ from the IR's
by one place at 18 to 172 of the 1024 words, and `exp2` and `logistic`, at 112 and 262, `exp2` by up
to seven places (`exp2(3)` is 7 in e5m2). `atan2` and `pow` are binary64's function of the
converted operands, rounded once to the format (`_Builder.binary64`). At each of the 65536 pairs of
words of each format the exact value is a value of the format, more than a relative 2^-38 from every
midpoint, or a midpoint, which only powers reach. Binary32 would not do: it holds the e5m2 midpoint
`3 * 2^-17`, within a relative 2^-32 of `atan2(3 * 2^-16, 2)`. The IR's binary64 `atan2` rounds the
polar angle once, and its `pow`, FloatLib's `Model.pow`, is correctly rounded at an integral
exponent with a finite result and where its certified search settles, and otherwise is binary64's
`exp(y * log x)`, which carries no accuracy theorem. The IR's words are the exact value rounded once
at every pair (`test_float8_atan2_and_pow_round_once_through_binary64`). XLA's CPU code computes
both in binary16, and its words differ from the IR's by one place at 110 to 436 of the pairs for
`atan2` and 12 to 64 for `pow`. `gather`, `dynamic_slice`, `dynamic_update_slice` and the
scatters that replace or take a maximum or minimum read and write float8 values in binary32 too,
exactly (`_Builder.indexed`), with a fill value only where the format holds it, and `reduce_max`,
`reduce_min`, `argmax`, `argmin`, `cummax` and `cummin` reduce or scan the binary32 values, whose
order, zero signs and NaNs are the format's: XLA's float8 maximum, minimum, `argmax` and `argmin`
are the IR's at every pair of words of each format and at every word but the NaNs in one row, and
its `cummax` and `cummin` at every pair of words and in rows of 32 words, NaNs among them, forward
and reversed. `rem`, C's `fmod`, exact in every format, and `sort`, a NaN last and `-0` equal to
`+0` as `lax.sort` documents, are binary32's too, and `top_k` and `approx_top_k` rank the binary32
values. `clamp` is the IR's maximum then minimum in the format, StableHLO's `minimum(maximum(x, lo),
hi)`, which is not what the cases of `lax.clamp`'s docstring give at a NaN bound, at `-0` or where
`lo > hi`. The window reductions are the IR's left folds in the format from JAX's identities, which
are the largest finite magnitudes in a format without infinities. `select_and_scatter_add` selects
among the binary32 values and adds the source values in the format, each sum rounded once, and JAX's
CPU lowering pads the operand with those identities, which a cell can equal. XLA's words are the
IR's at every word between bounds drawn from NaN, the zeros, `±0.5`, `±1`, `-2`, the largest
magnitudes and the infinities, at every pair of words in `rem`, and in sorts, rankings and 2 x 3
windows of rows of 32 words, but for the fnuz formats' `sort`, which in XLA's CPU code puts the NaN
first, and the `sort` and `top_k` of float8_e5m2fnuz, which there take the five values `2^-15` to
`2^-14` of each sign as equal. With that lowering XLA's `select_and_scatter_add` words are the
row-major fold's over the padded operand in 2-D windows that overlap, tile and lie partly in
padding. `nextafter` is the IR's C `nextafter` in the format, `bitcast_convert_type` reads each word
at an 8-bit integer type or another float8 format, and a float8 `iota` is each index rounded once,
StableHLO's `iota`. XLA's words are the IR's at every pair of words in `nextafter`, at every word
and 8-bit integer in the bitcasts, and in iotas of 600 entries, but for `nextafter` in the fnuz
formats from the least negative value toward a larger one, which in XLA gives the word `0x80`, their
NaN, and not `+0`. `reduce_precision` rounds the binary32 values at the format's exponents (above),
where XLA's CPU code rounds binary16's. Any other primitive on a float8 value, among them `erf`, is
refused. A conversion to one rounds once to nearest, ties to even, as XLA's CPU
conversions do word for word: above 464 e4m3fn gives its NaN, and e5m2 from 61440 up an infinity. A
NaN's sign and payload are left open, and XLA and the IR choose differently. From binary64,
ml_dtypes, NumPy's float8, rounds through binary32 and so differs near a midpoint. `float8_e8m0fnu`,
`scaled_dot`'s default scale type, has no sign and no fraction bit, which no binary descriptor
holds, so it is refused, and with it `scaled_dot` without scales.

#### Accumulation precision

A `preferred_element_type` of `dot_general`, `conv_general_dilated`, `ragged_dot_general` or
`scaled_dot` is, in JAX's documentation, the type the results are accumulated in and returned as.
When that type holds every value of the operands' type (a binary format with at least their
precision and greatest exponent and a smallest subnormal no larger, as each of binary16, bfloat16,
binary32 and binary64 has for each float8 format, or one whose precision covers an integer type's
magnitude bits), the operands are converted to it exactly, a constant kernel word by word, and the
contraction runs there, as XLA's CPU code converts them. XLA contracts bfloat16 operands with a
binary32 result unconverted, in another binary32 order, so those results agree within a tolerance,
not word for word, and so do float8 operands with a binary32 result, which XLA converts and
contracts in its binary32 order. A type that does not hold the operands, such as bfloat16 for
binary32 operands, binary16 for bfloat16 or binary32 for int32, is refused.

Integer and Boolean `dot_general` follows StableHLO's definition (its spec, "dot_general"): the
operands are moved to their batch, free and contracted axes and broadcast to that joint shape,
multiplied entrywise, and the contracted axes summed, or for Booleans with logical `and` and `or`
(StableHLO's `multiply` and `add` of Booleans). Integer sums and products wrap modulo 2^width, the
importer's choice for the overflow that StableHLO leaves implementation-defined, as for `add`. The
joint product holds every term, so it takes the product of all those extents in memory. The maximum
and minimum of Booleans, false below true, are logical `or` and `and` (StableHLO's `maximum` and
`minimum`), from either frontend and in windows, where JAX's identities are false and true. XLA's
CPU code agrees at full-range operands of every width, eager and under `jax.jit`
(`test_integer_and_boolean_contractions`).

JAX leaves the accumulation type of a `dot_general` or `conv_general_dilated` with a float8 result
to the backend, and the importer contracts or convolves the float8 operands in binary32 and rounds
the result once to the format (`_Builder.widened`), as it does for binary16 and bfloat16. With
operands of magnitude 1/8 to 4, whose binary32 sums are exact in any order, the IR's words are the
exact sums rounded once in 256 products of 16 terms and 304 convolution entries of 9 products in
each format, and in the 30 products of 16 random finite words tried in each format they are the
binary32 left fold rounded once (scripts/checks/ir_capture_jax_contraction.py,
test_float8_contractions_convert_their_operands). A `ragged_dot_general` with a float8 result is
contracted in binary32 too, its groups selected there and the result rounded once
(`ContractionImport.ragged_dot`), and at operands of magnitude 1/8 to 4 the IR's words are the
exact results rounded once in 120 entries of its three modes in each format. XLA's CPU code rounds
twice (below).

Nondefault dot precision and algorithms are rejected: a declared FP32 input and result do not say
whether the multiplications use BF16, TF32 or FP32, and the matmul node has no separate arithmetic
rule for operands and accumulator. A default matmul capture does not establish XLA's reduction order
or backend equivalence.

### Differences from XLA CPU execution

The imported artifact states an arithmetic schedule; XLA can choose another when compiling it.
Eager JAX compiles each primitive and nested control-flow or `jax.jit` call as a computation, while
`jax.jit` can optimize the whole function. The
[JAX capture regressions](../../../scripts/checks/ir_capture_jax.py) exercise these differences:

- Division by a broadcast divisor can become multiplication by a rounded reciprocal. Under
  `jax.jit`, a constant divisor can receive the same rewrite. A mean can therefore multiply by a
  rounded reciprocal of the element count instead of dividing by that count.
- Under `jax.jit`, quotient simplification can rewrite `x / sqrt(y)` to `x * rsqrt(y)`,
  `(x / y) / z` to `x / (y * z)`, or `x / (y / z)` to `(x * z) / y`. The resulting roundings differ.
- A multiplication can fuse into a following addition or subtraction. A complex multiplication
  can fuse its component products even in eager JAX, because it is one primitive. The IR keeps the
  schedule of its paired-real expansion.
- Complex division can use Smith's algorithm with fused products and special-value recovery.
  Its intermediate denominator can overflow even when the exact complex quotient is finite.
- Powers can use an approximation or, for constant exponents under `jax.jit`, repeated products
  or a reciprocal. Neither generally matches a single correctly rounded power.
- XLA's reciprocal square root is an approximation; the IR rounds `sqrt(x)` and then `1 / sqrt(x)`.
- Some CPU operations flush subnormal operands or results to zero. This can change values and
  the control flow of a loop. Layout operations and binary16 arithmetic need separate treatment;
  flushing is not applied uniformly to every operation or format.
- Reductions can regroup their entries rather than use the IR's row-major left fold. Binary16,
  bfloat16 and float8 sums can round partial sums in their own format.
- Float8 contractions can round through binary16 before their final format. In e4m3fn,
  `[1, 1/16, 1/64] · [1, 1, 1/64] = 4353/4096` rounds once to `1.125`, but rounds through binary16
  to the midpoint `1.0625` and then to `1.0`. The IR contracts these operands in binary32 and
  rounds once to float8.

These are backend-dependent observations, not general error bounds. A default precision setting
does not determine a backend's reduction tree, fused operations or transcendental implementation.

`verify`'s `compare` fails whenever the IR's words differ from the reference's, under its NaN
policy. For a Jaxpr document it then tries these rules on the mismatch, as advice for the reader.
The JAX importer records `reference_execution`, `jit` for a function that is one `jax.jit` call and
`eager` otherwise, the computation of each node in an eager function, and the primitive that
produced each node. `torchlean.verify` records with the reference the platform that computed it and
jaxlib's version. When `jax.jit` compiles any part of the function, or the function holds a `cond`,
a `scan` or a `while`, it also records the outputs of evaluating the function one primitive at a
time, inside nested `jax.jit` calls, the branch each `cond` selects and each iteration of a `scan`
or a `while` loop too, which XLA computes without `jax.jit`'s rewrites. Only a reference from XLA's
CPU backend, of an artifact without control flow, is emulated. The checker runs the program a second time as XLA's CPU code computes it,
where that code is exactly emulable. Divisions by broadcast divisors, and under `jax.jit` by
constant ones, become products with the rounded reciprocal. When `reference_execution` is `jit`,
`jax.jit`'s rewrites apply: those of quotients of quotients and by quotients in binary16, binary32
and binary64, and those of divisions by square roots, reciprocal square roots and powers in binary32
and binary64. So do the products for the exponents 2, 3 and -1, in binary32 and binary64 and in
binary16 with each product rounded to binary16. The flushing above applies to the operations listed.

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
model computations. TorchLean does not compile arbitrary Lean or Python programs: control flow
other than JAX's `cond`, `scan` and `while` and torch.export's `torch.cond`, `while_loop`, `map` and
`invoke_subgraph`, and operations without a lowering rule, are rejected.

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
embeds the capture and defines `Model.artifact`, its decoded artifact, and `Model.program`, the
program of a typed artifact (`Model.program_checked`); for an artifact with control flow it defines
no `Model.program`, and `Model.programs_checked` covers the program of each block of
`Model.artifact.flow`. Lean checks the included proofs of artifact decoding and
`Program.check .reference` by kernel reduction when the module is compiled (`PyTorchFlowTests`
compiles a module of each kind), and export does not launch a compiler. `IR.loadFlow` (`IR.load`
for an artifact without control flow) also validates the constant payload and `declared_types`.
The export includes no proof of those load checks. Mathematical properties can refer to that
program or flow and the embedded constants, and they need their own proofs.
A `.json` export keeps the capture document instead. Neither export overwrites a file. The
[captured-computation example](../../../home_page/examples/verify/index.md) exports a captured
`square` function to `Proofs/Square.lean`.

There are several routes into the same graph language:

- TorchLean model code can lower supported fragments into IR for execution, inspection, or
  verification.
- The PyTorch `torch.export` capture records positional tensor arguments, including unused
  arguments, from its graph signature and writes each as an input step of a typed artifact with its
  argument position, and its other nodes become typed steps (`python/torchlean/_artifact.py`).
  `torch.cond`, `while_loop` and `map` are lowered to the IR's flows, after what gives no value
  the program returns is dropped (`_flow.prune`), a `map` whose body draws is unrolled, and
  `invoke_subgraph` is inlined.
  A convolution's or linear map's weight and bias are payload words where they are parameters,
  buffers or constants. Another weight or bias, an input or a value computed from one, is an
  operand after the node's input, the weight in PyTorch's layout: a convolution's of output
  channels and then each group's input channels, and a linear map's of output and then input
  features (`test_convolutions_take_a_weight_not_known_at_capture` and
  `test_linear_takes_a_weight_not_known_at_capture`). Lean decodes and checks the artifact. The
  ONNX adapter (`python/torchlean/onnx.py`) writes each graph input that is not an initializer as
  an input step, in the graph's order.
- The Jaxpr capture in `python/torchlean/jaxpr.py` lowers static tensor computations to a typed
  artifact in a capture document. Its tensor arguments are input steps in order, each with its own
  shape and dtype. `cond`, `scan` and `while` are lowered by partial evaluation, and a `cond` or
  `while` that the capture does not decide to the IR's flows, after what gives no value the
  computation returns is dropped (`_flow.prune`). `TorchLean.JAX.load` checks the
  frontend and loads the flow of programs under the reference runtime (`TorchLean.JAX.load_ok`),
  whose run uses FloatLib at each declared binary precision and runs Booleans and integer words
  natively (see the [JAX example](../../../home_page/examples/jaxpr/index.md)).
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
softmax expansion and attention code that uses it import. Its `max` is IEEE's maximum, a NaN where
an entry is one, as `lax.max`'s `stablehlo.maximum` is. XLA's CPU code (JAX 0.11.1) can drop a NaN
from a reduction of 4096 or more binary32 or binary64 entries: of 4095 ones and a NaN it gives 1
with the NaN first and `-inf` with the NaN last, so COMPARE reports a difference there. JAX's exact
GELU uses `erfc`, which imports as `Op.special`, the real function, which no binary format executes.

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
