import VersoManual
import NN.Widgets
import NN.Tensor
import NN.Spec.Core.Tensor.Core
import NN.IR.Graph
import NN.IR.Semantics
import FloatLib
import NN.MLTheory.CROWN.Graph
import NN.Runtime.Autograd.Engine.Core
import NN.Runtime.Training.Log
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- Every example in this chapter is elaborated when the page is built, so the namespaces come out
-- here rather than being repeated inside each block. `Spec` and `TorchLean` carry the tensor type
-- and its literal notation, `NN.IR` the graph and its evaluator, FloatLib the binary scalars,
-- `CROWN` the interval boxes, and `Runtime.Autograd` the tape.
open Spec TorchLean
open NN.IR
open NN.MLTheory.CROWN
open Runtime.Autograd

-- A few printed reports are wider than the code column, so their `leanOutput` blocks ask for
-- `whitespace := lax` and are wrapped in the source. The rendered page shows what Lean printed.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Widgets" =>
%%%
tag := "widgets"
%%%

A wide verifier bound can originate far from the output node. Lean may already hold every
intermediate interval in a graph state, but a long printed record makes it hard to locate the
first loss of precision. A widget displays those node bounds together with the graph that
produced them.

Widgets render existing Lean objects in the Infoview: tensor entries, graph parents, inferred
shapes, Float32 fields, affine bounds, tape gradients, and metric curves. The view helps locate a
problem; evaluating or proving a claim about the underlying object remains a separate operation.

The [widget modules](https://github.com/lean-dojo/TorchLean/tree/main/NN/Widgets/) are collected by
`NN.Widgets`. Import that module when a scratch file needs the inspection tools together; the
broader `import NN` also brings them into a larger development.

# Inspection Workflow

For an unexpected output, the graph and its evaluation trace expose intermediate values. For an
inconclusive verification result, node-by-node interval widths help locate precision loss. A
stalled training run may instead call for comparing the tape, accumulated gradients, and metric
log.

The same named object can be evaluated, viewed, and checked:

1. Keep the value that matters as a named Lean definition or as an artifact at an explicit path.
2. Run the evaluator or checker whose semantics the eventual claim uses.
3. Put the corresponding widget beside that value and find the first surprising node, field, or
   step.
4. Change the definition or producer, elaborate again, and let the checker close the argument
   instead of trusting the picture.

Read-only views such as `#tensor_view` format a value without changing it. Trace views execute a
specific Lean computation: `#ir_exec_trace_view` steps through `NN.IR.Semantics`, while
`#tape_trace_view` follows the autograd engine's reverse pass. The interpretation is therefore
explicit. A theorem about that same interpretation is still a separate proof.

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
  * parents, declared/inferred shapes, and intermediate values
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
  * tape structure, reverse steps, and accumulated gradients
*
  * a completed run
  * `#train_log_view` or a file-backed log view
  * metrics, notes, prompts, samples, policies, or transitions
*
  * a PyTorch sketch
  * `#pytorch_translate_file`
  * an editor-assistant translation, not an importer proof
:::

Choose the view by the question being asked. If a tensor contains the wrong entries, changing a
plot's scale cannot explain where those entries came from; follow its parents in the graph or tape.
If the entries are plausible but a bound is wide, inspect the verifier state instead of rerunning
ordinary evaluation at one input. The two computations answer different questions: one produces a
value at a chosen point, while the other represents a range of possible values. Keeping the named
input object beside its view makes that distinction concrete.

# Application Workflows

Some values live only while a file elaborates; others are written by a command and inspected later.
A graph or tape definition can sit immediately above its view. A GPT or
PPO run instead writes a named artifact, which a later Lean file reads:

```
-- Inspect the saved artifacts at the same paths used by
-- their producer commands.
#train_log_file_view "data/examples/cnn_trainlog.json"
#train_log_file_view "data/examples/gpt2_trainlog.json"
#rl_boundary_rollout_file_view "data/rl/cartpole_rollout.json", contract, 12
#pytorch_translate_file "NN/Examples/Interop/PyTorch/MLP/train_mlp.py"
```

Keep the producer command beside a file-backed view in a comment or experiment note. The renderer
can tell you what the file says, but it cannot recover where the file came from. When the artifact
has a checker, check it first and then inspect the accepted object.

# Tensor Viewer

I encode the three indices of `rankThreeGrid` in the hundreds, tens, and units places so we can
recognize an axis change in the entries themselves. The floating-point vectors address a separate inspection problem:
decimal printing can hide differences in scalar representation.

```lean
-- Encode tensor coordinates in decimal digits and keep
-- scalar precision examples separate.
def decimalTenth : Float :=
  Float.ofBits 0x3fb999999999999a

def oneThirdFloat : Float :=
  Float.ofBits 0x3fd5555555555555

def floatTensor : Tensor Float [4] :=
  [ Float.ofNat 1
  , Float.ofNat 2
  , decimalTenth
  , oneThirdFloat ]

def ieeeTensor : Tensor (ExecFloat.Binary 8 23) [4] :=
  [ (1 : (ExecFloat.Binary 8 23))
  , (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
      (Float.ofNat 2)
  , (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
      decimalTenth
  , (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
      oneThirdFloat ]

def indexTensor : Tensor Nat [5] := [0, 1, 2, 3, 4]

def rankThreeGrid : Tensor Nat [2, 3, 4] :=
  Tensor.generate [2, 3, 4] fun c =>
    c.getD 0 0 * 100 + c.getD 1 0 * 10 + c.getD 2 0

def sampleMatrix : Tensor Int [2, 4] :=
  Tensor.generate [2, 4] fun c =>
    Int.ofNat (c.getD 0 0 * 10 + c.getD 1 0)
```

`Tensor.generate` passes the coordinate list to its callback. For shape `[2, 3, 4]`, the three
lookups give the indices that `rankThreeGrid` encodes; `sampleMatrix` uses the first two coordinates
of its rank-two shape.

Those definitions are the widget inputs. In an editor, each `#tensor_view` line renders the value
beside the cursor:

```lean
-- Compare nesting, entries, and summaries without changing
-- the underlying tensors.
#tensor_view indexTensor
#tensor_view rankThreeGrid
#tensor_view floatTensor
#tensor_view ieeeTensor
#tensor_view sampleMatrix

-- Numeric summaries for the small tensors above:
#tensor_stats_view floatTensor
```

The commands above are elaborated when this page is built. Interactive panels require the
Infoview; the same tensor can also be inspected through its printed entries:

```lean (name := wgGrid)
-- Print the same coordinate-coded tensor used by the
-- interactive view.
#eval IO.println s!"{rankThreeGrid}"
```

```leanOutput wgGrid (whitespace := lax)
[[[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23]],
 [[100, 101, 102, 103], [110, 111, 112, 113],
  [120, 121, 122, 123]]]
```

Entry `112` sits at $`(1,1,2)`. Moving the first axis would move the hundreds digit's variation to
a different nesting level, making the layout change visible across the whole grid.

`floatTensor` and `ieeeTensor` use different scalar types. Comparing their decimal renderings does
not reveal every difference introduced by binary32 conversion:

```lean (name := wgPrecision)
-- Put the binary32 bits beside rounded decimal text to
-- expose hidden differences.
def compareLine (x : Float) : String :=
  let r := (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32) x
  let bits := String.ofList
    (Nat.toDigits 16 (ExecFloat.Binary.toBits32 r).toNat)
  let decimal := (ExecFloat.Binary.toFloat32 r).toFloat
  s!"binary64 {x}  binary32 {decimal}  bits 0x{bits}"

#eval do
  IO.println (compareLine (Float.ofNat 1))
  IO.println (compareLine decimalTenth)
  IO.println (compareLine oneThirdFloat)
```

```leanOutput wgPrecision (whitespace := lax)
binary64 1.000000  binary32 1.000000  bits 0x3f800000
binary64 0.100000  binary32 0.100000  bits 0x3dcccccd
binary64 0.333333  binary32 0.333333  bits 0x3eaaaaab
```

Each row uses the same decimal rendering for its binary64 and binary32 values; the final column
shows the binary32 bits. One is exact in both formats. One tenth is not a binary
fraction, so neither format holds it exactly, and the two formats do not hold the same
approximation of it {Informal.citep goldberg1991}[]. This comparison explicitly converts each
binary32 result to the native decimal display. The configured tensor's own printer can show its
exact dyadic value; `#float32_view` further down exposes the encoding and exceptional-value fields.

The coordinate-coded grid is useful because a transposition can preserve both the number of
entries and their minimum and maximum. A summary might therefore look unchanged while the model
reads the wrong axis as channels or time. Here the hundreds digit identifies the first coordinate,
the tens digit the second, and the units digit the third. Reading the nesting and the digits
together checks their correspondence. The precision example tests another kind of lost information:
decimal formatting can preserve the visual pattern while hiding different stored numbers.

# IR Graph Viewer

For the next graph, we can inspect three things separately:

1. *Structure*: which nodes, parents, and shapes are present?
2. *Invariants*: do declared node shapes match what the ops infer from parent shapes?
3. *Semantics*: when the graph is evaluated, which node fails first and what are the intermediate
   values?

The trace command calls the internal IR evaluator directly, so its input uses `Spec.SomeTensor` to
carry a runtime shape. Model code continues to accept and return `Tensor α shape`.

```lean
-- Keep the graph structure separate from the input and
-- constant payload it evaluates.
def pairTensor (x y : Float) : Tensor Float [2] :=
  [x, y]

def sampleGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input
        outShape := [2] },
      { id := 1, parents := #[]
        kind := .const [2]
        outShape := [2] },
      { id := 2, parents := #[0, 1], kind := .add
        outShape := [2] }
    ] }

-- Same as `sampleGraph`, but `sub` at the output node.
-- Useful for rewrite and diff examples.
def sampleGraphSub : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input
        outShape := [2] },
      { id := 1, parents := #[]
        kind := .const [2]
        outShape := [2] },
      { id := 2, parents := #[0, 1], kind := .sub
        outShape := [2] }
    ] }

-- The `.const` node reads an external payload rather than
-- storing its value in the graph.
def sampleInput : Spec.SomeTensor Float :=
  { shape := [2], tensor := pairTensor 0.60 (-0.20) }

def samplePayload : Payload Float :=
  { const? := fun id =>
      if id = 1 then
        some { n := 2, v := pairTensor 0.25 0.25 }
      else
        none }
```

`parents := #[0, 1]` stores the input and constant node ids in an `Array Nat`. The constant's
value lives in `samplePayload`; the graph records its shape and the operation that reads it.

The four views then attach to those definitions:

```lean
-- Inspect structure first, then compare shapes, the edited
-- graph, and execution.
#ir_view sampleGraph

-- 1) Invariant check:
-- declared shape tags vs inferred shapes.
#shape_infer_view sampleGraph

-- 2) Before/after view:
-- handy for compiler/optimizer passes.
#graph_rewrite_view sampleGraph, sampleGraphSub

-- 3) Evaluation trace: step through the IR semantics.
#ir_exec_trace_view sampleGraph, samplePayload, sampleInput
```

The graph view shows node `2` depending on the input and constant. The shape view compares all
three declared length-two shapes with inference. The trace view repeatedly calls `Graph.evalAt`,
retaining the values computed before a failure.
The following `Graph.denoteAll` call evaluates the complete graph through the same node semantics
and prints the successful result in node order:

```lean (name := wgTrace)
-- Evaluate all nodes so each displayed value can be matched
-- to its parent values.
#eval do
  let g := sampleGraph
  let r := Graph.denoteAll g samplePayload sampleInput
  match r with
  | .ok vals =>
    for i in [0:vals.size] do
      match vals[i]? with
      | some v => IO.println s!"node {i}: {v.tensor}"
      | none => pure ()
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

The addition/subtraction comparison is a display example, not a justified graph rewrite.
`sampleGraphSub` changes the mathematical operation, so seeing both graphs side by side helps locate
the change but cannot establish that they compute the same result. For an optimizer rewrite, the
corresponding correctness statement must relate the source and destination semantics under its
hypotheses. A structural diff remains useful even then: it lets the reader check which nodes and
payloads that statement is supposed to cover.

## Shape Inference Errors

A wrong declared output shape and incompatible operand shapes fail at different stages. The next
two graphs isolate those cases:

```lean
-- The declared shape at the output node is wrong.
def miscountedGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input
        outShape := [2] },
      { id := 1, parents := #[]
        kind := .const [2]
        outShape := [2] },
      { id := 2, parents := #[0, 1], kind := .add
        outShape := [3] }
    ] }

-- Every declaration agrees with its own node here, but
-- the addition itself cannot happen.
def crossedGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input
        outShape := [2] },
      { id := 1, parents := #[]
        kind := .const [3]
        outShape := [3] },
      { id := 2, parents := #[0, 1], kind := .add
        outShape := [2] }
    ] }
```

`Graph.inferShapes` is the function behind `#shape_infer_view`. It returns one shape per node in id
order, or stops at the first disagreement:

```lean (name := wgInfer)
-- Separate incorrect output declarations from incompatible
-- operand shapes.
#eval do
  match Graph.inferShapes sampleGraph with
  | .ok shapes =>
    IO.println s!"sampleGraph inferred {shapes}"
  | .error e =>
    IO.println s!"sampleGraph rejected: {e}"
  match Graph.checkShapes miscountedGraph with
  | .ok _ => IO.println "miscountedGraph accepted"
  | .error e =>
    IO.println s!"miscountedGraph rejected: {e}"
  match Graph.checkShapes crossedGraph with
  | .ok _ => IO.println "crossedGraph accepted"
  | .error e =>
    IO.println s!"crossedGraph rejected: {e}"
```

```leanOutput wgInfer (whitespace := lax)
sampleGraph inferred #[[2], [2], [2]]
miscountedGraph rejected: IR graph: node 2: outShape
  mismatch: inferred=[2], declared=[3]
  (Node(id=2, kind=add, parents=#[0, 1], outShape=[3]))
crossedGraph rejected: add: shape mismatch: [2] vs [3]
```

For `miscountedGraph`, the operands support a length-two result but the declaration says length
three. For `crossedGraph`, the addition itself receives incompatible lengths, so changing only its
output tag cannot repair the graph. The shape viewer reports both inferred shapes and declaration
checks. `NN.IR.ShapeSoundness` proves that successful evaluation of an accepted graph produces the
declared shapes. Acceptance supplies that structural hypothesis; it does not provide missing
parameter payloads or ensure that every runtime operation succeeds.

A shape error and a missing constant payload require different repairs. The former concerns the
operation's tensor interface and can be detected from graph metadata. The latter concerns the
values supplied when the graph runs. This example deliberately stores the constant outside the
graph so both layers remain visible. When a trace stops, the last successful node narrows the
search; nodes marked unexecuted are consequences of that stop, not additional independent failures.
Inspect the first error before interpreting later empty rows.

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
#float32_round_view oneThirdFloat
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
reaches a gradient can contaminate later arithmetic. The bit view exposes that value directly;
a loss curve alone may hide it, especially when later branches mask the NaN.

The round views answer a different question. When a binary64 `Float` crosses the scalar boundary,
which binary32 pattern is chosen?

```lean (name := wgRound)
-- Recover the exact dyadic value behind the six-decimal
-- display.
#eval do
  let r := (ExecFloat.Binary.ofFloat32 ∘ Float.toFloat32)
    decimalTenth
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

The rounded float is $`13421773 \cdot 2^{-27}`. With the significand and exponent returned by
`toDyadic?`, we can state its exact value in a proof even though the display rounds it to
`0.100000`. Flocq's treatment of IEEE-754 in Rocq also reasons from exact representations of
floating-point values {Informal.citep flocq2011}[] {Informal.citep boldo2015}[].

The exact dyadic output also gives a way to reason about conversion error. Once the finite value
is written as an integer times a power of two, its distance from the intended rational can be
computed without relying on the displayed decimal digits. That is why the bit and round views are
useful beside numerical proofs: they expose the data on which an exact statement can be based.
For a NaN or infinity, that finite-value interpretation is unavailable, so classification must come
before treating a bit pattern as a real number.

# Verification (IBP/CROWN State)

TorchLean's verification code includes executable bound propagation engines: IBP boxes and
CROWN affine forms. When debugging a verifier, inspect which nodes
have bounds and whether shapes and flattened dimensions match the intended layout.

```lean
-- The three boxes are named so the widths below can be
-- printed one at a time.
def inputBox : FlatBox Float :=
  { dim := 2
    lo := pairTensor (-1.0) (-1.0)
    hi := pairTensor 1.0 1.0 }

def constBox : FlatBox Float :=
  { dim := 2
    lo := pairTensor 0.25 0.25
    hi := pairTensor 0.25 0.25 }

def outputBox : FlatBox Float :=
  { dim := 2
    lo := pairTensor (-0.75) (-0.75)
    hi := pairTensor 1.25 1.25 }

def samplePropState : Graph.PropState Float :=
  { inputId := 0
    inputDim := 2
    states := #[
      { shape := [2], ibp? := some inputBox, aff? := none }
    , { shape := [2], ibp? := some constBox, aff? := none }
    , { shape := [2], ibp? := some outputBox, aff? := none }
    ] }

#crown_view sampleGraph, samplePropState

-- Interval widths (`hi - lo`) are a fast
-- "where did bounds blow up?" diagnostic.
#bounds_tightness_view sampleGraph, samplePropState
```

The graph is the same `sampleGraph` from above, so a trace value and a bound refer to the same node
id. Here `samplePropState` is a manually constructed display fixture, not the output of a verifier
or a checked certificate. Adding
the exact constant $`0.25` shifts $`[-1,1]` to $`[-0.75,1.25]` without changing the width:

```lean (name := wgWidths)
-- Compare componentwise widths before and after adding the
-- point interval.
def boxLine (tag : String) (b : FlatBox Float) : String :=
  s!"{tag}: [{b.lo}, {b.hi}] width {b.hi - b.lo}"

#eval do
  IO.println (boxLine "input " inputBox)
  IO.println (boxLine "const " constBox)
  IO.println (boxLine "output" outputBox)
```

```leanOutput wgWidths (whitespace := lax)
input : [[-1.000000, -1.000000], [1.000000, 1.000000]]
  width [2.000000, 2.000000]
const : [[0.250000, 0.250000], [0.250000, 0.250000]]
  width [0.000000, 0.000000]
output: [[-0.750000, -0.750000], [1.250000, 1.250000]]
  width [2.000000, 2.000000]
```

The input and output both have width two; the constant has width zero. These representable
endpoints make the translation exact. Over real arithmetic, adding a point preserves interval
width, while directed floating-point arithmetic may widen endpoints through rounding. In a larger
graph, inspecting the first unexpected increase helps distinguish rounding from precision lost by
a bound-propagation rule {Informal.citep crown2018}[].

The state records use flattened boxes of dimension two even though the graph keeps tensor
shapes separately. Matching the node id, shape, and flattened dimension prevents a convincing plot
from being attached to the wrong operation. Missing bounds also matter: a blank interval entry
means no box was supplied for that node, whereas a zero-width box represents an exact point.
The constant's zero width here is therefore useful information. It explains why translating the
input box introduces no uncertainty in this fixture.

## Interval Dependency

Subtracting a value from itself exposes information that separate interval boxes cannot retain:

```lean (name := wgDependency)
-- Reusing one box twice shows the dependency information
-- interval subtraction loses.
#eval do
  IO.println (boxLine "x      " inputBox)
  IO.println (boxLine "x - x  "
    (Graph.boxSub inputBox inputBox))
  IO.println (boxLine "relu x "
    (Graph.boxRelu inputBox))
```

```leanOutput wgDependency (whitespace := lax)
x      : [[-1.000000, -1.000000], [1.000000, 1.000000]]
  width [2.000000, 2.000000]
x - x  : [[-2.000000, -2.000000], [2.000000, 2.000000]]
  width [4.000000, 4.000000]
relu x : [[0.000000, 0.000000], [1.000000, 1.000000]]
  width [1.000000, 1.000000]
```

$`x - x` is zero for every $`x`, but interval subtraction returns a width-four box. The box still
contains zero. It also contains values obtained by choosing different points from the two input
intervals, because `boxSub` has no record that both occurrences denote the same variable.
This dependency loss can make interval bound propagation
{Informal.citep gowal2018}[] too imprecise for a network's verification property. CROWN can retain
correlations through affine forms where its transfer rules support them
{Informal.citep crown2018}[]: an affine expression for $`x
- x` cancels its
coefficients before any endpoint is ever computed.

Read `#bounds_tightness_view` with that distinction in hand. Elementwise ReLU clamps interval
endpoints and cannot increase a scalar interval's width; the example above shrinks it. A separate
affine relaxation can lose correlations even when the interval width shrinks. Unexpected growth
at another operation can come from dependency loss, arithmetic rounding, or a transfer-rule error.
Inspect that operation before choosing a tighter relaxation, input partitioning, or a code fix.

# Autograd (Tape + Gradients)

TorchLean's eager autograd engine records a computation graph into a `Tape` and can run
reverse-mode to accumulate gradients. The widget below shows the recorded tape and the gradients
produced by scalar backprop, corresponding to a call to `loss.backward()` in PyTorch
{Informal.citep pytorch2019}[].

```lean
-- Record ab + b so the shared leaf b must receive two
-- reverse contributions.
def sampleTape : Tape Float :=
  let (t0, aId) :=
    Tape.leaf (α := Float) (t := Tape.empty)
      (value := Tensor.full [] 2.0) (name := some "a")
  let (t1, bId) :=
    Tape.leaf (α := Float) (t := t0)
      (value := Tensor.full [] 3.0) (name := some "b")
  let (t2, abId) :=
    match Tape.mul (α := Float) (t := t1)
        (s := []) aId bId with
    | .ok r => r
    | .error _ => (t1, 0)
  let (t3, _) :=
    match Tape.add (α := Float) (t := t2)
        (s := []) abId bId with
    | .ok r => r
    | .error _ => (t2, 0)
  t3

#tape_grads_view sampleTape, 3

-- For a step by step account of why a gradient exists
-- (or is missing), use the reverse pass trace:
#tape_trace_view sampleTape, 3
```

The recorded scalar is $`ab+b` at $`a=2` and $`b=3`, so $`\partial/\partial a = b = 3` and
$`\partial/\partial b = a+1 = 3`. Both derivatives are three by coincidence at this point, which is
convenient for reading the output and worth remembering when changing the numbers. Running the same
reverse pass the widget runs:

```lean (name := wgGrads)
-- Report stored values and accumulated gradients from the
-- reverse pass rooted at node 3.
def tapeLine
    (grads : Std.HashMap Nat (Spec.SomeTensor Float))
    (id : Nat) (n : Node Float) : String :=
  let grad :=
    match grads[id]? with
    | some v => s!"{v.tensor}"
    | none => "none"
  let name := n.name.getD s!"n{id}"
  s!"{name}: value {n.value.tensor} grad {grad}"

#eval do
  match Tape.backwardScalar (t := sampleTape) 3 with
  | .ok grads =>
    for id in [0:sampleTape.nodes.size] do
      match sampleTape.nodes[id]? with
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

The output is nine and both leaf gradients are three, matching the derivatives above. Node `b`
feeds both multiplication and addition, so its gradient accumulates two contributions: $`a=2`
from multiplication and $`1` from addition. Dropping the addition's contribution would give the
plausible but incorrect gradient `2.0`. The reverse traversal combines these contributions using
`SomeTensor.add`, rather than mutating a leaf's `.grad` field as PyTorch does
{Informal.citep pytorch2019}[].

`Tape.backwardScalar` returns a map from node ids to gradients and an `Except` error if a local
VJP fails, for example because of a shape mismatch. The derivatives of $`ab+b` give a reference
for inspecting this reverse trace: a missing or incorrect gradient can be traced to the step where
a contribution was omitted or computed incorrectly.

The two internal gradients explain the reverse traversal as well. The final addition is seeded
with one because it is the selected scalar output. Its multiplication parent receives one, and
that parent sends `b` to `a` and `a` to `b`. The direct addition edge supplies the extra one to `b`.
Looking only at the final leaf numbers would miss this route through the graph. The trace is most
useful when a shared parameter has several consumers, because each consumer's contribution must
arrive before the accumulated gradient is interpreted.

## Unused Leaves And Missing Gradients

The other half of gradient debugging is a missing entry rather than a wrong number. Record a leaf
that nothing consumes and the reverse pass never visits it:

```lean (name := wgStray)
-- Add an unused leaf while keeping the reverse pass rooted
-- at the original output.
def strayLeafTape : Tape Float :=
  let (t0, _) :=
    Tape.leaf (α := Float) (t := sampleTape)
      (value := Tensor.full [] 5.0) (name := some "c")
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
chosen output. An unused parameter can produce the same symptom in a model. The gradient view
shows the absent entry; the trace view shows why reverse traversal never reaches `c`. This differs
from reaching a node and computing a zero contribution, even if a display that fills missing
entries with zero would make the two cases look alike.

# Training Dashboards

Training logs and confusion matrices are also Lean values. `Runtime.Training.TrainLog` records
steps, metric series, and notes; `Runtime.Training.ConfusionMatrix` stores class counts. The widget
layer renders these records whether they came from a Lean training loop or an external experiment.

The following literals are display fixtures. Their values and notes are supplied directly here;
no training run produced them:

```lean (name := wgLog)
-- These independent records exercise the dashboard without
-- claiming a training run.
def sampleTrainLog : Runtime.Training.TrainLog :=
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

def sampleLabels : Array String := #["cat", "dog", "owl"]

def sampleCM : Runtime.Training.ConfusionMatrix :=
  { counts := #[
      #[8, 1, 0]
    , #[2, 6, 1]
    , #[0, 1, 7]
    ] }
```

The two viewer commands then take those names. They render in the Infoview, so this page can only
show the invocation:

```
-- Render the named metric log and the separately supplied
-- confusion matrix.
#train_log_view sampleTrainLog
#confusion_view sampleLabels, sampleCM
```

Ordinary array operations can summarize these records without rendering a widget:

```lean (name := wgLogSummary)
-- Summarize endpoints and diagonal counts to check what the
-- two panels would display.
#eval do
  let log := sampleTrainLog
  IO.println s!"steps: {log.steps.size}"
  for s in log.series do
    let a := s.values[0]!
    let b := s.values.back!
    IO.println s!"{s.name}: {a} -> {b}"
  let rows := sampleCM.counts
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

The confusion matrix contains 21 correct predictions among 26, giving $`21/26 \approx 0.808`.
That differs from the fixture log's final `val_acc` of `0.79`: the two literals were constructed
independently. When both artifacts describe the same evaluation in a real run, this comparison
can detect inconsistent metrics or mismatched files.

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

Text models write ordinary `TrainLog` artifacts. The shared viewer preserves multiline notes,
including prompts and generated samples:

```
-- Read the text model log written by its training or
-- generation command.
#train_log_file_view "data/examples/gpt2_trainlog.json"
```

Run training or generation through the model's CLI, then inspect the saved log. The widget only
reads and renders the artifact.

# RL Boundary And Policy Views

A reward curve leaves out the transitions and actions that produced it. The RL views expose:

- the checked transition boundary for Gymnasium rollouts,
- GridWorld policies and paths,
- PPO rollout curves derived from reward/value/advantage data.

The main entry files are:

- {src "NN/Widgets/RL/GridWorld.lean"}[GridWorld widget source]
- [PPO widget source](https://github.com/lean-dojo/TorchLean/blob/main/NN/Widgets/RL/PPO.lean)
- {src "NN/Widgets/RL/Boundary.lean"}[RL boundary widget source]

These views expose the transitions received by the learner and the policy/path artifacts written
by a command. MDP validity and boundary soundness are separate statements in the proof layer.

A rollout boundary report counts checked transitions and records failures with their indices.
That index lets a reader inspect the offending observation, action, or reward in the original
artifact. It does not supply the old policy probabilities or value estimates needed to reconstruct
a PPO objective; those belong to a different record. An empty rollout deserves special attention:
there are no rejected transitions, but there are also no accepted observations of the boundary.
The widget reports that no checks ran rather than presenting emptiness as evidence about an episode.

# PyTorch Translator Widget

The PyTorch translator widget proposes a constructor sketch in the editor:

```
-- Ask the editor assistant for a constructor sketch of this
-- Python source.
#pytorch_translate_file "NN/Examples/Interop/PyTorch/MLP/train_mlp.py"
```

It helps readers see how a simple `torch.nn` snippet maps onto TorchLean constructors. Checked
interop claims should cite the explicit PyTorch roundtrip/export examples and the artifact bridge,
not the heuristic widget alone. The widget accepts scalar natural dimensions in recognized
constructor arguments; symbolic sizes, tuples, duplicate keywords, and extra options are reported
as unsupported. Convolution rows provide metadata and shape-boundary notes, not an executable CNN.

# Widget Examples

Open
{src "NN/Examples/DeepDives/Widgets.lean"}[NN/Examples/DeepDives/Widgets.lean]
in VS Code with the Lean extension enabled. Put the cursor on each widget command and open the
Infoview. The file is deliberately small enough to elaborate interactively.

Work through four changes:

1. Change one entry of a tensor and confirm that `#tensor_view` and `#tensor_stats_view` update from
   the same Lean value.
2. Give an IR node the wrong declared output shape. `#shape_infer_view` should identify the first
   disagreement; restore the shape before continuing.
3. Replace `sampleGraph` by `sampleGraphSub` in the rewrite view and inspect the changed operation
   tag rather than comparing raw record syntax.
4. Compare `(1 : ExecFloat.Binary 8 23)` with a quiet NaN. The bit viewer should expose the
   exponent, fraction, and classification that ordinary decimal printing hides.

The file can also be elaborated from a terminal:

```terminal
# Elaborate the widget examples; interactive panels are
# displayed by the editor.
lake env lean NN/Examples/DeepDives/Widgets.lean
```

Terminal elaboration checks the commands and definitions, while the VS Code Infoview provides the
interactive rendered panels. If a widget fails to elaborate, first distinguish a malformed Lean
object from a rendering problem: replace the widget command by `#check` on the same object, then
reintroduce the view after its type is correct.

## Other Widget Sources

The focused examples are useful when one artifact is the whole subject:

- {src "NN/Examples/DeepDives/Floats/Float32Semantics.lean"}[Float32 semantics]
  pairs `#float32_view`, `#float32_compare_view`, and `#float32_round_view` with executable binary32
  values.
- {src "NN/Verification/Builtin/CrownOpsWorkflow.lean"}[CROWN workflow]
  and
  {src "NN/Verification/Builtin/IBPWorkflow.lean"}[IBP workflow] provide real verifier states for
  `#crown_view` and `#bounds_tightness_view`.
- {src "NN/Examples/Quickstart/AutogradBasics.lean"}[Autograd basics]
  supplies a tape for `#tape_view`, `#tape_grads_view`, and `#tape_trace_view`.

## File-Backed Views

Training and application commands can write artifacts that outlive the Lean process. Use
`#train_log_file_view` for metric and text-generation logs, and
`#rl_boundary_rollout_file_view` for checked transition records. The
{src "NN/Examples/Data/Loaders/Csv.lean"}[CSV loader example],
{src "NN/Examples/Models/Sequence/Gpt2.lean"}[GPT training command],
and
{src "NN/Examples/Models/RL/Views/GymnasiumRollout.lean"}[RL rollout view] show the corresponding
producers.

GridWorld file views use the same validated policy/path readers as the runtime artifact API;
missing fields and invalid actions or positions produce error panels. The rollout boundary view
explicitly reports when an empty file contained no transitions to check.

File-backed views parse and render the artifact at the named path. A successful visualization does
not authenticate its producer or strengthen the artifact's checker claim. When validity matters,
run the corresponding checker first and use the widget to inspect the same accepted file.

A file-backed view can also fail without invalidating the Lean definition that invokes it. For
example, the training-log viewer turns a missing or malformed file into an error panel. This makes
it usable during an experiment, but it means successful elaboration alone is not evidence that the
intended artifact was displayed. Inspect the panel's path and status as well as its curves. If a
file is replaced between checking and viewing, the two operations no longer necessarily refer to
the same data; retain the accepted artifact when documenting a result.

# View Data Sources

Keep the definition or producer path near its view. A graph panel can then be traced to the graph
record, a bound to its verifier state, and a training curve to the parsed file. This makes a
surprising display traceable to the data that produced it.
