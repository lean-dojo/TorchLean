import VersoManual
import TorchLean.NN.Widget.Docs
import TorchLean.NN.Widget.IR.ExecutionTrace
import TorchLean.NN.Widget.IR.Rewrite
import TorchLean.NN.Widget.IR.ShapeInference
import TorchLean.NN.Widget.PyTorchTranslator
import TorchLean.NN.Widget.Float32
import TorchLeanExamples.Models.RL.GridWorld.Widgets
import TorchLean.NN.Widget.RL.Boundary
import TorchLean.NN.Widget.RL.PPO
import TorchLean.NN.Widget.Autograd
import TorchLean.Verify.CROWN.Widget
import TorchLean.Tensor.Einops.Elab.Einsum
import TorchLean.Tensor.Einops.Elab.Pack
import TorchLean.Tensor.Einops.Elab.Transform
import TorchLean.Tensor.Tactic.Einops.Report
import TorchLean.Tensor.Literal
import TorchLean.Tensor.Datatype
import TorchLean.NN.IR.Eval
import TorchLean.NN.IR.Pretty
import FloatLib
import TorchLean.Verify.CROWN.Op.ExecCROWN
import TorchLean.NN.Tape.Apply
import TorchLean.NN.Tape.Backward
import TorchLean.NN.Training
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- Every example in this chapter is elaborated when the page is built, so the namespaces come out
-- here rather than being repeated inside each block. `TorchLean` carries the tensor type and its
-- literal notation, `IR` the graph and its evaluator, FloatLib the binary scalars, `CROWN` the
-- interval boxes and engines, and `Tape` the tape.
open TorchLean
open IR
open CROWN
open Tape

-- A few printed reports are wider than the code column, so their `leanOutput` blocks ask for
-- `whitespace := lax` and are wrapped in the source. The rendered page shows what Lean printed.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Widgets" =>
%%%
tag := "widgets"
%%%

```lean -show
namespace Tutorial.Widgets
```

A wide verifier bound can originate far from the output node. Lean may already hold every
intermediate interval in a graph state, but a long printed record makes it hard to locate the
first loss of precision. A widget displays those node bounds together with the graph that
produced them.

Widgets render existing Lean objects in the Infoview: tensor entries, graph parents, inferred
shapes, Float32 fields, affine bounds, tape gradients, and metric curves. A view helps you locate a
problem, but evaluating or proving a claim about the underlying object remains a separate
operation.

The [widget modules](https://github.com/lean-dojo/TorchLean/tree/main/TorchLean/NN/Widget/) group
the views by the objects they display, and a scratch file imports the modules of the views it uses.
`#crown_view` and `#bounds_tightness_view` come from `TorchLean.Verify.CROWN.Widget`.

# Inspection Workflow

For an unexpected output, the graph and its evaluation trace expose intermediate values. For an
inconclusive verification result, node-by-node interval widths help locate precision loss. A
stalled training run may instead call for comparing the tape, accumulated gradients, and metric
log.

You can evaluate, view, and check the same named object:

1. Keep the value that matters as a named Lean definition or as an artifact at an explicit path.
2. Run the evaluator or checker whose semantics the eventual claim uses.
3. Put the corresponding widget beside that value and find the first surprising node, field, or
   step.
4. Change the definition or producer, elaborate again, and let the checker close the argument
   instead of trusting the picture.

Read-only views such as `#tensor_view` format a value without changing it. Trace views execute a
specific Lean computation: `#ir_exec_trace_view` steps through `Node.eval`, while
`#tape_trace_view` runs reverse mode and recomputes local VJP contributions from the resulting
cotangents, without recording intermediate gradient-map updates. A theorem about either computation
is still a separate proof.

The available views expose different parts of these objects:

:::table +header
*
  * When the question is about…
  * Start with
  * What it exposes
*
  * tensor values or representation
  * `#tensor_view`, `#tensor_stats_view`, `#float32_view`
  * entries, shape, summaries, or binary32 fields
*
  * graph structure or execution
  * `#ir_view`, `#shape_infer_view`, `#ir_exec_trace_view`
  * references, ports, and intermediate values
*
  * a rewrite
  * `#graph_rewrite_view`
  * source and result graphs side by side
*
  * verifier precision
  * `#crown_view`, `#bounds_tightness_view`
  * node bounds, affine state, and interval widths
*
  * gradients
  * `#tape_grads_view`, `#tape_trace_view`
  * tape structure, local VJP contributions, and accumulated gradients
*
  * a completed run
  * `#train_log_view` or a file-backed log view
  * metrics, notes, prompts, samples, policies, or transitions
*
  * a PyTorch sketch
  * `#pytorch_translate_file`
  * an editor-assistant translation, not an importer proof
:::

Choose the view by the question you're asking. If a tensor contains the wrong entries, changing a
plot's scale can't explain where those entries came from. Follow the tensor's parents in the graph
or tape. If the entries are plausible but a bound is wide, inspect the verifier state instead of
rerunning ordinary evaluation at one input. Those two computations answer different questions: one
produces a value at a chosen point, while the other represents a range of possible values. Keeping
the named input object beside its view makes that distinction concrete.

# Application Workflows

Some values live only while a file elaborates. Others are written by a command and inspected later.
A graph or tape definition can sit immediately above its view. A GPT or PPO run instead writes a
named artifact, which a later Lean file reads:

```
-- Inspect the saved artifacts at the same paths used by
-- their producer commands.
#train_log_file_view "data/examples/cnn_trainlog.json"
#train_log_file_view "data/examples/gpt2_trainlog.json"
#rl_boundary_rollout_file_view "data/rl/gym_cartpole_rollout.json", contract, 12
#pytorch_translate_file "TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py"
```

Keep the producer command beside a file-backed view in a comment or experiment note. The renderer
can tell you what the file says, but it can't recover where the file came from. When the artifact
has a checker, check it first and then inspect the accepted object.

# Tensor Viewer

I encode the three indices of `grid` in the hundreds, tens, and units places so we can
recognize an axis change in the entries themselves. The floating-point vectors address a
separate inspection problem: decimal printing can hide differences in scalar representation.

```lean
-- Encode tensor coordinates in decimal digits and keep
-- scalar precision examples separate.
def decimalTenth : Float :=
  Float.ofBits 0x3fb999999999999a

def third : Float :=
  Float.ofBits 0x3fd5555555555555

def native : Tensor Float [4] :=
  [ Float.ofNat 1
  , Float.ofNat 2
  , decimalTenth
  , third ]

def ieee : Tensor (ExecFloat.Binary 8 23) [4] :=
  [ (1 : (ExecFloat.Binary 8 23))
  , (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
      (Float.ofNat 2)
  , (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
      decimalTenth
  , (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
      third ]

def indices : Tensor Nat [5] := [0, 1, 2, 3, 4]

def grid : Tensor Nat [2, 3, 4] :=
  Tensor.generate [2, 3, 4] fun c =>
    c.getD 0 0 * 100 + c.getD 1 0 * 10 + c.getD 2 0

def matrix : Tensor Int [2, 4] :=
  Tensor.generate [2, 4] fun c =>
    Int.ofNat (c.getD 0 0 * 10 + c.getD 1 0)
```

`Tensor.generate` passes the coordinate list to its callback. For shape `[2, 3, 4]`, the three
lookups give the indices that `grid` encodes, and `matrix` uses the first two coordinates of its
rank-two shape.

Those definitions are the widget inputs. In an editor, each `#tensor_view` line renders the value
beside the cursor:

```lean
-- Compare nesting, entries, and summaries without changing
-- the underlying tensors.
#tensor_view indices
#tensor_view grid
#tensor_view native
#tensor_view ieee
#tensor_view matrix

-- Numeric summaries for the small tensors above:
#tensor_stats_view native
```

When this page is built, Lean elaborates the commands above. Interactive panels require the
Infoview, but you can also inspect the same tensor through its printed entries:

```lean (name := wgGrid)
-- Print the same coordinate-coded tensor used by the
-- interactive view.
#eval IO.println s!"{grid}"
```

```leanOutput wgGrid (whitespace := lax)
[[[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]],
 [[100, 101, 102, 103], [110, 111, 112, 113],
  [120, 121, 122, 123]]]
```

Entry `112` sits at $`(1,1,2)`. Moving the first axis would move the hundreds digit's variation to
a different nesting level, making the layout change visible across the whole grid.

`native` and `ieee` use different scalar types. Comparing their decimal renderings doesn't reveal
every difference introduced by binary32 conversion:

```lean (name := wgPrecision)
-- Put the binary32 bits beside rounded decimal text to
-- expose hidden differences.
def compareLine (x : Float) : String :=
  let r : FloatLib.Floats.Binary32 :=
    ExecFloat.Binary.ofFloat32 x.toFloat32
  let bits := String.ofList
    (Nat.toDigits 16 (ExecFloat.Binary.toBits32 r).toNat)
  let decimal := (ExecFloat.Binary.toFloat32 r).toFloat
  s!"binary64 {x}  binary32 {decimal}  bits 0x{bits}"

#eval do
  IO.println (compareLine (Float.ofNat 1))
  IO.println (compareLine decimalTenth)
  IO.println (compareLine third)
```

```leanOutput wgPrecision (whitespace := lax)
binary64 1.000000  binary32 1.000000  bits 0x3f800000
binary64 0.100000  binary32 0.100000  bits 0x3dcccccd
binary64 0.333333  binary32 0.333333  bits 0x3eaaaaab
```

Each row uses the same decimal rendering for its binary64 and binary32 values, and the final column
shows the binary32 bits. One is exact in both formats. One tenth isn't a binary fraction, so
neither format holds it exactly, and the two formats don't hold the same approximation of it
{Informal.citep goldberg1991}[]. In this comparison we convert each binary32 result to the native
decimal display. The configured tensor's own printer can show its exact dyadic value, and
`#float32_view` further down exposes the encoding and exceptional-value fields.

A coordinate-coded grid is useful because a transposition can preserve both the number of entries
and their minimum and maximum, so a summary might look unchanged while the model reads the wrong
axis as channels or time. Here the hundreds digit identifies the first coordinate, the tens digit
the second, and the units digit the third. Reading the nesting and the digits together checks their
correspondence. And the precision example tests another kind of lost information: decimal
formatting can preserve the visual pattern while hiding different stored numbers.

# IR Graph Viewer

For the next graph, we can inspect three things separately:

1. *Structure*: which nodes, references, and ports are present.
2. *Invariants*: whether every reference names an output at a port its operator's input accepts.
3. *Semantics*: when the graph is evaluated, which node fails first and what the intermediate
   values are.

Since the trace command calls the graph evaluator directly, its inputs are values at ports
(`TorchLean.Op.Value`), each carrying its port and shape. Model code keeps accepting and returning
`Tensor α shape`.

```lean
-- Keep the graph structure separate from the input and
-- constant payload it evaluates.
def graph (op : TorchLean.Op [.value [2], .value [2]]
    [.value [2]]) : IR.Graph :=
  { nodes := #[.input 0 (.value [2]), .const (.value [2]),
      .op ⟨_, _, op⟩ #[⟨0, 0⟩, ⟨1, 0⟩]]
    outputs := #[⟨2, 0⟩] }

-- The `.const` node reads an external payload rather than
-- storing its value in the graph.
def input : List (TorchLean.Op.Value Float) :=
  [⟨.value [2], ([0.60, -0.20] : Tensor Float [2])⟩]

def payload : Payload Float :=
  { const? := fun id =>
      if id = 1 then
        some ⟨.value [2], ([0.25, 0.25] : Tensor Float [2])⟩
      else
        none }
```

`#[⟨0, 0⟩, ⟨1, 0⟩]` references output `0` of the input and of the constant node (a `Ref` is a node
id and an output position). For the constant, the graph records its port and the operator that
reads it, and the value itself lives in `payload`.

Four views then attach to those definitions:

```lean
-- Inspect structure first, then compare shapes, the edited
-- graph, and execution.
#ir_view (graph (op := .add))

-- 1) Invariant check:
-- expected input ports vs the ports read.
#shape_infer_view (graph (op := .add))

-- 2) Before/after view:
-- handy for compiler/optimizer passes.
#graph_rewrite_view (graph (op := .add)),
  (graph (op := .sub))

-- 3) Evaluation trace: step through the IR semantics.
#ir_exec_trace_view (graph (op := .add)), payload, input
```

In the graph view, node `2` depends on the input and constant. The shape view lists the length-two
ports the addition expects beside the ports its references name. Calling `Node.eval` node by node,
the trace view retains the outputs computed before a failure. Next, a `Graph.run` call evaluates
the complete graph through the same node semantics and prints every node's output in node order:

```lean (name := wgTrace)
-- Evaluate all nodes so each displayed value can be matched
-- to its parent values.
#eval do
  let g := graph (op := .add)
  match g.run payload input with
  | .ok outs =>
    for (vs, i) in outs.zipIdx do
      match (vs : List (TorchLean.Op.Value Float)) with
      | [⟨⟨.value, _⟩, v⟩] => IO.println s!"node {i}: {v}"
      | _ => pure ()
  | .error e => IO.println s!"evaluation failed: {e}"
```

```leanOutput wgTrace (whitespace := lax)
node 0: [0.600000, -0.200000]
node 1: [0.250000, 0.250000]
node 2: [0.850000, 0.050000]
```

Node `2` adds `[0.25, 0.25]` to `[0.60, -0.20]`, producing `[0.85, 0.05]` to the precision shown.
If that result were unexpected, the two parent rows would distinguish incorrect inputs from an
incorrect addition. Shape checking addresses failures that prevent this evaluation from succeeding.

We compare addition with subtraction as a display example, not as a justified graph rewrite.
`graph (op := .sub)` changes the mathematical operation. Seeing both graphs side by side helps us
locate the change but can't establish that they compute the same result. For an optimizer rewrite,
the correctness statement must relate the source and destination semantics under its hypotheses. A
structural diff remains useful even then: it lets you check which nodes and payloads that statement
is supposed to cover.

## Shape Inference Errors

An operator graph declares no output shapes: `Node.outs` reads every node's ports off its operator,
so a node can't claim a shape its operator doesn't produce. What remains is a reference whose port
the operator's input doesn't accept. In the next graph we add a constant of three entries to an
input of two:

```lean
-- Every node is well formed on its own, but the addition
-- at shape `[2]` reads a constant of shape `[3]`.
def crossedGraph : IR.Graph :=
  { nodes := #[.input 0 (.value [2]), .const (.value [3]),
      .op ⟨_, _, TorchLean.Op.add (s := [2])⟩
        #[⟨0, 0⟩, ⟨1, 0⟩]]
    outputs := #[⟨2, 0⟩] }
```

`Graph.check` decides what `#shape_infer_view` shows node by node. It accepts the addition graph
and stops at the first reference whose port its operator's input doesn't accept:

```lean (name := wgInfer)
-- Derived output ports, then the structural check of both
-- graphs.
#eval do
  let g := graph (op := .add)
  let ports := g.nodes.toList.map Node.outsPretty
  IO.println s!"addition graph produces {ports}"
  match g.check with
  | .ok () => IO.println "addition graph accepted"
  | .error e => IO.println s!"addition graph rejected: {e}"
  match crossedGraph.check with
  | .ok () => IO.println "crossedGraph accepted"
  | .error e =>
    IO.println s!"crossedGraph rejected: {e}"
```

```leanOutput wgInfer (whitespace := lax)
addition graph produces [[2], [2], [2]]
addition graph accepted
crossedGraph rejected: IR check: node 2: output 0 of node 1
  has port { kind := TorchLean.Port.Kind.value, shape := [3] },
  which input port { kind := TorchLean.Port.Kind.value, shape := [2] }
  does not accept
```

For `crossedGraph`, the addition at shape `[2]` receives a constant of shape `[3]`, so changing the
constant's payload can't repair the graph: the reference or the constant's port has to change. For
every node, the shape viewer shows the ports its operator expects, the ports its references name,
and the ports it produces. `Node.eval_ports` proves that every output of a successful node is at
the port the node declares. Acceptance by `Graph.check` doesn't provide missing parameter payloads
or ensure that every runtime operation succeeds.

A shape error and a missing constant payload require different repairs: the former concerns the
operation's tensor interface and can be detected from graph metadata, while the latter concerns the
values supplied when the graph runs. In this example we store the constant outside the graph on
purpose, so that both layers remain visible. When a trace stops, the last successful node narrows
the search. Nodes marked unexecuted are consequences of that stop, not additional independent
failures. Inspect the first error before interpreting later empty rows.

# Float32 Bit Layout Viewer

```lean
-- Use explicit bit patterns to contrast a finite value with
-- a quiet NaN.
def one32 : (ExecFloat.Binary 8 23) :=
  ExecFloat.Binary.ofBits32 (0x3f800000 : UInt32)

def qnan32 : (ExecFloat.Binary 8 23) :=
  ExecFloat.Binary.ofBits32 (0x7fc00000 : UInt32)

#float32_view one32
#float32_view (1 : (ExecFloat.Binary 8 23))
#float32_view qnan32
#float32_compare_view one32, qnan32

-- Compare a Float64 input to its Float32 rounding:
#float32_round_view decimalTenth
#float32_round_view third
```

We can print the same sign, exponent, and fraction fields that the viewer displays:

```lean (name := wgBits)
-- Print the sign, exponent, and fraction fields that drive
-- the bit viewer.
def bitLine (tag : String)
    (x : (ExecFloat.Binary 8 23)) : String :=
  let s := if ExecFloat.Binary.signBit x then 1 else 0
  s!"{tag}: sign={s} exp={Model.expField
    (ExecFloat.Binary.toModel x)}" ++
    s!" frac={Model.fracField
      (ExecFloat.Binary.toModel x)} nan={
        ExecFloat.Binary.isNaN x}"

#eval do
  IO.println (bitLine "one32 " one32)
  IO.println (bitLine "qnan32" qnan32)
```

```leanOutput wgBits (whitespace := lax)
one32 : sign=0 exp=127 frac=0 nan=false
qnan32: sign=0 exp=255 frac=4194304 nan=true
```

`one32` is sign `0`, biased exponent `127`, zero fraction, which is the encoding of $`1.0`. The
quiet NaN has the all-ones exponent field `255` and a nonzero fraction, so its classification stays
visible even though it has no real value to print. That distinction matters in practice: a NaN that
reaches a gradient can contaminate later arithmetic. A loss curve alone may hide that value
(especially when later branches mask the NaN), but the bit view exposes it.

The round views answer a different question: which binary32 pattern is chosen when a binary64
`Float` crosses the scalar boundary.

```lean (name := wgRound)
-- Recover the exact dyadic value behind the six-decimal
-- display.
#eval do
  let r : FloatLib.Floats.Binary32 :=
    ExecFloat.Binary.ofFloat32 decimalTenth.toFloat32
  IO.println s!"rounded value = {
    (ExecFloat.Binary.toFloat32 r).toFloat}"
  let d := repr
    ((Model.toDyadic? ∘ ExecFloat.Binary.toModel) r)
  IO.println s!"exact value = {d}"
```

```leanOutput wgRound (whitespace := lax)
rounded value = 0.100000
exact value = some { negative := false, significand := 13421773, exponent := -27 }
```

That rounded float is $`13421773 \cdot 2^{-27}`. With the significand and exponent returned by
`toDyadic?`, we can state its exact value in a proof even though the display rounds it to
`0.100000`. Flocq's treatment of IEEE-754 in Rocq also reasons from exact representations of
floating-point values {Informal.citep flocq2011}[] {Informal.citep boldo2015}[].

That exact dyadic output also gives us a way to reason about conversion error: once the finite
value is written as an integer times a power of two, you can compute its distance from the intended
rational without relying on the displayed decimal digits. That's why the bit and round views are
useful beside numerical proofs: they expose the data on which an exact statement can be based. For
a NaN or infinity, that finite-value interpretation is unavailable, so classification must come
before treating a bit pattern as a real number.

# Verification (IBP/CROWN State)

TorchLean's verification code includes executable bound propagation engines: IBP boxes and
CROWN affine forms. When debugging a verifier, inspect which nodes
have bounds and whether shapes and flattened dimensions match the intended layout.

```lean
-- The input box seeds both engines. The other boxes and the
-- affine bounds are computed from the graph.
def inputBox : FlatBox Float :=
  { dim := 2
    lo := [-1.0, -1.0]
    hi := [1.0, 1.0] }

def seeds (id : Nat) : Option (FlatBox Float) :=
  if id = 0 then some inputBox else none

def boxes : Array (Option (FlatBox Float)) :=
  Op.runIBP (graph (op := .add)) payload seeds

def affine :
    Array (Option (Graph.FlatAffineBounds Float)) :=
  Op.runCROWN (graph (op := .add)) payload
    { inputId := 0, inputDim := 2 } seeds boxes inputBox

#crown_view (graph (op := .add)), boxes, affine

-- Interval widths (`hi - lo`) are a
-- "where did bounds blow up?" diagnostic.
#bounds_tightness_view (graph (op := .add)), boxes
```

We use the same `graph (op := .add)` as above, so a trace value and a bound refer to the same node
id. Here `boxes` and `affine` are what the operator-graph engines `Op.runIBP` and `Op.runCROWN`
compute from the input box. Those values are diagnostic state, not a checked certificate. Adding
the exact constant $`0.25` shifts $`[-1,1]` to $`[-0.75,1.25]` without changing the width:

```lean (name := wgWidths)
-- Compare componentwise widths before and after adding the
-- point interval.
def boxLine (tag : String) (b : FlatBox Float) : String :=
  s!"{tag}: [{b.lo}, {b.hi}] width {b.hi - b.lo}"

#eval do
  let rows : List (String × Nat) :=
    [("input ", 0), ("const ", 1), ("output", 2)]
  for (tag, i) in rows do
    match boxes[i]? with
    | some (some b) => IO.println (boxLine tag b)
    | _ => IO.println s!"{tag}: no box"
```

```leanOutput wgWidths (whitespace := lax)
input : [[-1.000000, -1.000000], [1.000000, 1.000000]]
  width [2.000000, 2.000000]
const : [[0.250000, 0.250000], [0.250000, 0.250000]]
  width [0.000000, 0.000000]
output: [[-0.750000, -0.750000], [1.250000, 1.250000]]
  width [2.000000, 2.000000]
```

Both input and output have width two, and the constant has width zero. These representable
endpoints make the translation exact. Over real arithmetic, adding a point preserves interval
width, while directed floating-point arithmetic may widen endpoints through rounding. In a larger
graph, inspecting the first unexpected increase helps distinguish rounding from precision lost by a
bound-propagation rule {Informal.citep crown2018}[].

Both engines use flattened boxes of dimension two even though the graph keeps tensor shapes in its
ports. Matching the node id, port, and flattened dimension prevents a convincing plot from being
attached to the wrong operation. Missing bounds also matter: a blank interval entry means no box
was computed for that node, whereas a zero-width box represents an exact point. So the constant's
zero width here is useful information: it explains why translating the input box introduces no
uncertainty in this graph.

## Interval Dependency

Subtracting a value from itself exposes information that separate interval boxes can't retain:

```lean (name := wgDependency)
-- Reusing one box twice shows the dependency information
-- interval subtraction loses.
#eval do
  IO.println (boxLine "x      " inputBox)
  IO.println (boxLine "x - x  "
    (Graph.boxSub inputBox inputBox))
  IO.println (boxLine "relu x "
    (Graph.boxReLU inputBox))
```

```leanOutput wgDependency (whitespace := lax)
x      : [[-1.000000, -1.000000], [1.000000, 1.000000]]
  width [2.000000, 2.000000]
x - x  : [[-2.000000, -2.000000], [2.000000, 2.000000]]
  width [4.000000, 4.000000]
relu x : [[0.000000, 0.000000], [1.000000, 1.000000]]
  width [1.000000, 1.000000]
```

$`x - x` is zero for every $`x`, but interval subtraction returns a width-four box. That box still
contains zero. It also contains values obtained by choosing different points from the two input
intervals, because `boxSub` has no record that both occurrences denote the same variable. This
dependency loss can make interval bound propagation {Informal.citep gowal2018}[] too imprecise for
a network's verification property. CROWN can retain correlations through affine forms where its
transfer rules support them
{Informal.citep crown2018}[]: an affine expression for $`x
- x` cancels its
coefficients before any endpoint is ever computed.

Read `#bounds_tightness_view` with that distinction in hand. Elementwise ReLU clamps interval
endpoints and can't increase a scalar interval's width (the example above shrinks it), but a
separate affine relaxation can lose correlations even when the interval width shrinks. Unexpected
growth at another operation can come from dependency loss, arithmetic rounding, or a transfer-rule
error. Inspect that operation before choosing a tighter relaxation, input partitioning, or a code
fix.

# Autograd (Tape + Gradients)

TorchLean's eager autograd engine records a computation graph into a `Tape` and can run
reverse-mode to accumulate gradients. The widget below shows the recorded tape and the gradients
produced by scalar backprop, corresponding to a call to `loss.backward()` in PyTorch
{Informal.citep pytorch2019}[].

Each step records one registered operator of `TorchLean.Op` (here `Op.mul` and `Op.add` at the
scalar shape) with `Tape.apply`, which runs the operator's rule from `Op.rule?`. A step's parents
are node ids in nested pairs ending in `()`, and the result holds the extended tape and the id of
the operator's value output.

```lean
-- Record ab + b so the shared leaf b must receive two
-- reverse contributions.
def record : Result (Tape (SomeTensor Float)) := do
  let (t0, aId) :=
    Tape.leaf (t := (Tape.empty : Tape (SomeTensor Float)))
      (value := .ofTensor (Tensor.full [] 2.0))
      (name := some "a")
  let (t1, bId) :=
    Tape.leaf (t := t0)
      (value := .ofTensor (Tensor.full [] 3.0))
      (name := some "b")
  let (t2, abId, _) ←
    Tape.apply t1 "mul" (Op.mul (s := [])) (aId, bId, ())
  let (t3, _) ←
    Tape.apply t2 "add" (Op.add (s := [])) (abId, bId, ())
  pure t3

def tape : Tape (SomeTensor Float) :=
  have succeeds : record.toOption.isSome = true := by decide
  record.toOption.get succeeds

#tape_grads_view tape, 3

-- For a step by step account of why a gradient exists
-- (or is missing), use the reverse pass trace:
#tape_trace_view tape, 3
```

The proof in `tape` checks that recording succeeded: it can't silently replace a failed operation
with an earlier tape. We record the scalar $`ab+b` at $`a=2` and $`b=3`, so
$`\partial/\partial a = b = 3` and $`\partial/\partial b = a+1 = 3`. Both derivatives are three by
coincidence at this point, which is convenient for reading the output and worth remembering when
you change the numbers. We run the same reverse pass the widget runs:

```lean (name := wgGrads)
-- Report stored values and accumulated gradients from the
-- reverse pass rooted at node 3.
def tapeLine
    (grads : Std.HashMap Nat (SomeTensor Float))
    (id : Nat) (n : Node (SomeTensor Float)) :
    String :=
  let grad :=
    match grads[id]? with
    | some v => s!"{v.tensor}"
    | none => "none"
  let name := n.name.getD s!"n{id}"
  s!"{name}: value {n.value.tensor} grad {grad}"

#eval do
  match Tape.backwardScalar (t := tape) 3 with
  | .ok grads =>
    for id in [0:tape.nodes.size] do
      match tape.nodes[id]? with
      | some n => IO.println (tapeLine grads id n)
      | none => pure ()
  | .error e => IO.println s!"backward failed: {e}"
```

```leanOutput wgGrads (whitespace := lax)
a: value 2.000000 grad 3.000000
b: value 3.000000 grad 3.000000
mul: value 6.000000 grad 1.000000
add: value 9.000000 grad 1.000000
```

We get an output of nine, and both leaf gradients are three, matching the derivatives above. Node
`b` feeds both multiplication and addition, so its gradient accumulates two contributions: $`a=2`
from multiplication and $`1` from addition. Dropping the addition's contribution would give the
plausible but incorrect gradient `2.0`. TorchLean's reverse traversal combines these contributions
with `SomeTensor.add`, while PyTorch mutates a leaf's `.grad` field {Informal.citep pytorch2019}[].

`Tape.backwardScalar` returns a map from node ids to gradients and an `Except` error if a local VJP
fails, for example because of a shape mismatch. For inspecting this reverse trace, the derivatives
of $`ab+b` give a reference: a missing or incorrect gradient can be traced to the step where a
contribution was omitted or computed incorrectly.

You can read the reverse traversal off the two internal gradients as well. As the selected scalar
output, the final addition is seeded with one. Its multiplication parent receives one, and that
parent sends `b` to `a` and `a` to `b`. And the direct addition edge supplies the extra one to `b`.
Looking only at the final leaf numbers would miss this route through the graph. In reverse node
order, the trace panel shows each local VJP applied to its final accumulated cotangent: it can
explain the contributions from several consumers, but it doesn't show the accumulator after each
individual addition. Nodes with `requiresGrad=false` keep their VJP disabled in the view as in the
engine.

## Unused Leaves And Missing Gradients

The other half of gradient debugging is a missing entry, not a wrong number. Record a leaf that
nothing consumes and the reverse pass never visits it:

```lean (name := wgStray)
-- Add an unused leaf while keeping the reverse pass rooted
-- at the original output.
def strayLeafTape : Tape (SomeTensor Float) :=
  let (t0, _) :=
    Tape.leaf (t := tape)
      (value := .ofTensor (Tensor.full [] 5.0))
      (name := some "c")
  t0

#eval do
  match Tape.backwardScalar (t := strayLeafTape) 3 with
  | .ok grads =>
    for id in [0:strayLeafTape.nodes.size] do
      match strayLeafTape.nodes[id]? with
      | some n => IO.println (tapeLine grads id n)
      | none => pure ()
  | .error e => IO.println s!"backward failed: {e}"
```

```leanOutput wgStray (whitespace := lax)
a: value 2.000000 grad 3.000000
b: value 3.000000 grad 3.000000
mul: value 6.000000 grad 1.000000
add: value 9.000000 grad 1.000000
c: value 5.000000 grad none
```

`c` has a stored value but no entry in the returned gradient map because it has no path to the
chosen output. An unused parameter can produce the same symptom in a model. In the gradient view
the entry is absent, and the trace view shows why reverse traversal never reaches `c`. That case
differs from reaching a node and computing a zero contribution, even if a display that fills
missing entries with zero would make the two cases look alike.

# Training Dashboards

Training logs and confusion matrices are also Lean values. `Training.TrainLog` records
steps, metric series, and notes, while `Training.ConfusionMatrix` stores class counts.
Whether these records came from a Lean training loop or an external experiment, the widget layer
renders them.

These literals are display fixtures. We wrote their values and notes here, and no training run
produced them:

```lean (name := wgLog)
-- These independent records exercise the dashboard without
-- claiming a training run.
def log : Training.TrainLog :=
  { title := "Classifier training run"
    steps := #[0, 1, 2, 3, 4]
    series := #[
      { name := "loss", color := "#c44"
        values := #[1.20, 0.84, 0.59, 0.41, 0.33] }
    , { name := "val_acc", color := "#0a7"
        values := #[0.30, 0.48, 0.61, 0.73, 0.79] }
    , { name := "lr", color := "#06c"
        values := #[0.05, 0.05, 0.01, 0.01, 0.01] }
    ]
    notes := #[
      "optimizer: SGD"
    , "scheduler: StepLR(step_size=2, gamma=0.2)"
    , "dataset: synthetic 3-class classifier"
    ] }

def labels : Array String := #["cat", "dog", "owl"]

def confusion : Training.ConfusionMatrix :=
  { counts := #[
      #[8, 1, 0]
    , #[2, 6, 1]
    , #[0, 1, 7]
    ] }
```

Then the two viewer commands take those names. They render in the Infoview, so this page can only
show the invocation:

```
-- Render the named metric log and the separately supplied
-- confusion matrix.
#train_log_view log
#confusion_view labels, confusion
```

Ordinary array operations can summarize these records without rendering a widget:

```lean (name := wgLogSummary)
-- Summarize endpoints and diagonal counts to check what the
-- two panels would display.
#eval do
  let log := log
  IO.println s!"steps: {log.steps.size}"
  for s in log.series do
    let a := s.values[0]!
    let b := s.values.back!
    IO.println s!"{s.name}: {a} -> {b}"
  let rows := confusion.counts
  let mut total := 0
  let mut hits := 0
  for i in [0:rows.size] do
    let row := rows[i]!
    total := total + row.foldl (· + ·) 0
    hits := hits + row[i]!
  IO.println s!"correct {hits} of {total}"
```

```leanOutput wgLogSummary (whitespace := lax)
steps: 5
loss: 1.200000 -> 0.330000
val_acc: 0.300000 -> 0.790000
lr: 0.050000 -> 0.010000
correct 21 of 26
```

The confusion matrix contains 21 correct predictions among 26, giving $`21/26 \approx 0.808`, which
differs from the fixture log's final `val_acc` of `0.79`: the two literals were constructed
independently. When both artifacts describe the same evaluation in a real run, this comparison can
detect inconsistent metrics or mismatched files.

For logs produced by a training command, see:

- the CSV loader training example,
- the NPY loader training example,
- the CNN and ViT model commands,
- and the callback/reporting helpers exposed through `Trainer` reports.

The metric summary preserves units that a dashboard can make easy to overlook. Loss, accuracy,
and learning rate occupy separate series even though they share step coordinates. A downward
learning-rate curve is a configuration change, while a downward loss curve is an observation about
the supplied metric values. Neither explains the other by itself. The confusion matrix adds a
different view of the same classification task: off-diagonal entries identify which labels are
confused, information that one accuracy number discards. Its class labels must agree with the
matrix dimensions for that interpretation to make sense.

# GPT And Text-Model Logs

Text models write ordinary `TrainLog` artifacts, and the shared viewer preserves their multiline
notes, including prompts and generated samples:

```
-- Read the text model log written by its training or
-- generation command.
#train_log_file_view "data/examples/gpt2_trainlog.json"
```

Run training or generation through the model's CLI, then inspect the saved log. All the widget does
is read and render the artifact.

# RL Boundary And Policy Views

A reward curve leaves out the transitions and actions that produced it. TorchLean's RL views
expose:

- the checked transition boundary for Gymnasium rollouts,
- GridWorld policies and paths,
- PPO rollout curves derived from reward/value/advantage data.

Their main entry files are:

- {src "TorchLeanExamples/Models/RL/GridWorld/Widgets.lean"}[GridWorld widget source]
- [PPO widget
  source](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/NN/Widget/RL/PPO.lean)
- {src "TorchLean/NN/Widget/RL/Boundary.lean"}[RL boundary widget source]

These views expose the transitions received by the learner and the policy/path artifacts written
by a command. MDP validity and boundary soundness are separate statements in the proof layer.

A rollout boundary report counts checked transitions and records failures with their indices. That
index lets you inspect the offending observation, action, or reward in the original artifact. The
report doesn't supply the old policy probabilities or value estimates needed to reconstruct a PPO
objective. Those belong to a different record. An empty rollout deserves special attention: there
are no rejected transitions, but there are also no accepted observations of the boundary. Instead
of presenting emptiness as evidence about an episode, the widget reports that no checks ran.

# PyTorch Translator Widget

In the editor, the PyTorch translator widget proposes a constructor sketch:

```
-- Ask the editor assistant for a constructor sketch of this
-- Python source.
#pytorch_translate_file "TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py"
```

With the translator widget you can see how a simple `torch.nn` snippet maps onto TorchLean
constructors. Checked interop claims should cite the PyTorch roundtrip/export examples and the
artifact bridge, not the heuristic widget alone. In recognized constructor arguments the widget
accepts scalar natural dimensions, and it reports symbolic sizes, tuples, duplicate keywords, and
extra options as unsupported. Convolution rows provide metadata and shape-boundary notes, not an
executable CNN.

# Widget Examples

Open {src "TorchLeanExamples/DeepDives/WidgetGallery.lean"}[
  TorchLeanExamples/DeepDives/WidgetGallery.lean] in
VS Code with the Lean extension enabled. Put the cursor on each widget command and open the
Infoview. We kept the file small enough to elaborate interactively.

Work through four changes:

1. Change one entry of a tensor and confirm that `#tensor_view` and `#tensor_stats_view` update
   from the same Lean value.
2. Give the constant node another port shape. `#shape_infer_view` should then mark the addition,
   whose input doesn't accept the changed port. Restore the shape before continuing.
3. Change `op := .add` to `op := .sub` in the rewrite view and inspect the changed operation tag
   instead of comparing raw record syntax.
4. Compare `(1 : ExecFloat.Binary 8 23)` with a quiet NaN. The bit viewer should expose the
   exponent, fraction, and classification that ordinary decimal printing hides.

You can also elaborate the file from a terminal:

```terminal
# Elaborate the widget examples; interactive panels are
# displayed by the editor.
scripts/lake.sh env lean TorchLeanExamples/DeepDives/WidgetGallery.lean
```

Terminal elaboration checks the commands and definitions, while the VS Code Infoview provides the
interactive rendered panels. If a widget fails to elaborate, first distinguish a malformed Lean
object from a rendering problem: replace the widget command by `#check` on the same object, then
reintroduce the view after its type is correct.

## Other Widget Sources

These focused examples are useful when one artifact is the whole subject:

- {src "TorchLeanExamples/DeepDives/Floats/Float32Semantics.lean"}[Float32 semantics]
  pairs `#float32_view`, `#float32_compare_view`, and `#float32_round_view` with executable binary32
  values.
- {src "TorchLeanExamples/Verification/CROWNOpsWorkflow.lean"}[CROWN workflow]
  and
  {src "TorchLeanExamples/Verification/IBPWorkflow.lean"}[IBP workflow] provide real verifier states
  for `#crown_view` and `#bounds_tightness_view`.
- {src "TorchLeanExamples/DeepDives/WidgetGallery.lean"}[Widget gallery] records a tape for
  `#tape_view`, `#tape_grads_view`, and `#tape_trace_view`.

## File-Backed Views

Training and application commands can write artifacts that outlive the Lean process. Use
`#train_log_file_view` for metric and text-generation logs, and `#rl_boundary_rollout_file_view`
for checked transition records. For the corresponding producers, see the
{src "TorchLeanExamples/Data/Loaders/CSV.lean"}[CSV loader example],
{src "TorchLeanExamples/Models/Sequence/GPT2.lean"}[GPT training command], and
{src "TorchLeanExamples/Models/RL/GymnasiumRollout.lean"}[RL rollout view].

GridWorld file views use the same validated policy/path readers as the runtime artifact API.
Missing fields and invalid actions or positions produce error panels. When an empty file contained
no transitions to check, the rollout boundary view reports that.

File-backed views parse and render the artifact at the named path. A successful visualization
doesn't authenticate its producer or strengthen the artifact's checker claim. When validity
matters, run the corresponding checker first and use the widget to inspect the same accepted file.

A file-backed view can also fail without invalidating the Lean definition that invokes it. For
example, the training-log viewer turns a missing or malformed file into an error panel. That
behavior makes the viewer usable during an experiment, but it means successful elaboration alone
isn't evidence that the intended artifact was displayed. Inspect the panel's path and status as
well as its curves. If a file is replaced between checking and viewing, the two operations don't
necessarily refer to the same data. When you document a result, retain the accepted artifact.

# View Data Sources

Keep the definition or producer path near its view, so that a graph panel can be traced to the
graph record, a bound to its verifier state, and a training curve to the parsed file. This makes a
surprising display traceable to the data that produced it.

```lean -show
end Tutorial.Widgets
```
