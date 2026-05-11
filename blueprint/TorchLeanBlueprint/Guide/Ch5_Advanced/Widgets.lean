import VersoManual
import VersoBlueprint
import NN.Entrypoint.Widgets
import NN.Spec.Core.Tensor.Core
import NN.IR.Graph
import NN.IR.Semantics
import NN.Floats.IEEEExec.Exec32
import NN.MLTheory.CROWN.Graph
import NN.Runtime.Autograd.Engine.Core
import NN.Runtime.Training.Log
import NN.Runtime.Context

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

#doc (Manual) "Widgets" =>
%%%
tag := "widgets"
%%%

Widgets are the human inspection layer of TorchLean. They do not define semantics and they do not
prove theorems. They let a reader see the Lean objects that the semantics and theorems are about:
tensors, IR graphs, shape inference results, Float32 bit patterns, CROWN bounds, autograd tapes,
gradients, and training logs.

The [widgets API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Widgets/) is collected by `NN.Entrypoint.Widgets`. Import that entrypoint
when you want only the widget layer; `import NN` includes it as part of the broad main library
umbrella.

This is the only widget chapter in the guide. It starts with the semantic boundary for widgets,
then gives small examples, then maps the widget families to the objects they inspect.

Formal artifacts are hard to debug because they are often large, nested, and invisible: graph
nodes, tensor shapes, interval boxes, affine forms, tape cotangents, Float32 bit patterns. Widgets
make those artifacts visible inside the same editor where the theorem or checker lives.

# What Widgets Are For

Widgets are a debugging UI, not a proof system. TorchLean keeps the meaning of "execute this"
explicit:

- Widgets that inspect tensors are read-only formatted views of concrete values (they do not change
  the underlying data).
- Widgets that *run* something (IR evaluation, backprop traces) are running a specific computation defined in Lean
  semantics (for example `NN.IR.Semantics` for the IR with operation tags, or `Runtime.Autograd.Engine` for the
  tape engine).

That matters because it keeps assumptions visible instead of conflating visualization with proof:

- An execution trace widget is an evaluator defined in Lean stepping through the chosen semantics.
- A theorem still requires a proof about the same semantics the widget executes.

The intended workflow is simple: a widget renders a Lean object under a chosen Lean interpretation.
It should not invent a new meaning that the proof and checker layers cannot see.

So a widget can make a proof obligation visible, but it does not replace the proof. If a graph view
shows a bad shape, we fix the graph or the compiler. If a CROWN view shows loose bounds, we improve
the bound pass or certificate. The visualization is a microscope, not an oracle.

| Widget | Object inspected | Main question |
|---|---|---|
| `#tensor_view` | typed tensor | what values and shape? |
| `#ir_view` | `NN.IR.Graph` | what nodes and parents? |
| `#shape_infer_view` | graph shape inference | where is the shape mismatch? |
| `#ir_exec_trace_view` | IR execution | what did each node compute? |
| `#float32_view` | `IEEE32Exec` | what are the bits? |
| `#crown_view` | verifier state | where are the bounds? |
| `#tape_trace_view` | autograd tape | how did gradients flow? |
| `#train_log_view` | training log | did metrics move? |

# Tensor Viewer

```
open Spec
open TorchLean.Floats.IEEE754

def f01 : Float :=
  Float.ofBits 0x3fb999999999999a

def fthird : Float :=
  Float.ofBits 0x3fd5555555555555

def vecF64 : Tensor Float (shape![4]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar (Float.ofNat 1)
    | ⟨1, _⟩ => Tensor.scalar (Float.ofNat 2)
    | ⟨2, _⟩ => Tensor.scalar f01
    | ⟨_, _⟩ => Tensor.scalar fthird)

def vecF32 : Tensor IEEE32Exec (shape![4]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar IEEE32Exec.posOne
    | ⟨1, _⟩ =>
        Tensor.scalar (IEEE32Exec.ofFloat (Float.ofNat 2))
    | ⟨2, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat f01)
    | ⟨_, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat fthird))

def vec5 : Tensor Nat (shape![5]) :=
  Tensor.dim (fun i => Tensor.scalar i.1)

def cube234 :
    Tensor Nat (shape![2, 3, 4]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.dim (fun k =>
        Tensor.scalar (i.1 * 100 + j.1 * 10 + k.1))))

def mat2x4 : Tensor Int (shape![2, 4]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (Int.ofNat (i.1 * 10 + j.1))))

#tensor_view vec5
#tensor_view cube234
#tensor_view vecF64
#tensor_view vecF32
#tensor_view mat2x4

-- Quick numeric checks (small tensors):
#tensor_stats_view vecF64
```

# IR Graph Viewer

The IR widget family is meant to answer the three debugging questions that show up in practice:

1. *Structure*: which nodes, parents, and shapes are present?
2. *Invariants*: do declared node shapes match what the ops infer from parent shapes?
3. *Semantics*: when the graph is evaluated, which node fails first and what are the intermediate values?

```
open NN.IR
open Spec

def f2 (x y : Float) : Tensor Float (shape![2]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar x
    | ⟨_, _⟩ => Tensor.scalar y)

def toyGraph : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .add
        outShape := (shape![2]) }
    ] }

def toyGraphSub : Graph :=
  -- Same as `toyGraph`, but uses `sub` instead of `add`.
  -- (Useful for rewrite/diff demos.)
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .sub
        outShape := (shape![2]) }
    ] }

#ir_view toyGraph

-- 1) Invariant check:
-- declared shape tags vs inferred shapes.
#shape_infer_view toyGraph

-- 2) Before/after view:
-- handy for compiler/optimizer passes.
#graph_rewrite_view toyGraph, toyGraphSub

-- 3) Evaluation trace: run the IR semantics step by step.
-- For `.const` nodes, a small external payload is supplied.
def toyInput : Runtime.AnyTensor Float :=
  { s := (shape![2]), t := f2 0.60 (-0.20) }

def toyPayload : NN.IR.Payload Float :=
  { const? := fun id =>
      if id = 1 then
        some { n := 2, v := f2 0.25 0.25 }
      else
        none }

#ir_exec_trace_view toyGraph, toyPayload, toyInput
```

# Float32 Bit Layout Viewer

```
namespace Float32Demo

def one32 : IEEE32Exec :=
  IEEE32Exec.ofBits (0x3f800000 : UInt32)

def qnan32 : IEEE32Exec :=
  IEEE32Exec.ofBits (0x7fc00000 : UInt32)

#float32_view one32
#float32_view (1 : IEEE32Exec)
#float32_view qnan32
#float32_compare_view one32, qnan32

-- Compare a Float64 input to its Float32 rounding:
#float32_round_view f01
#float32_round_view fthird

end Float32Demo
```

# Verification (IBP/CROWN State)

TorchLean's verification stack includes executable bound propagation engines (IBP and CROWN
affine forms). When debugging a verifier, it is often most useful to inspect which nodes have
bounds and whether shapes and flattened dimensions match the intended layout.

```
open NN.IR
open Spec

def toyGraphCROWN : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .add
        outShape := (shape![2]) }
    ] }

def toyPropState :
    NN.MLTheory.CROWN.Graph.PropState Float :=
  let bIn : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := f2 (-1.0) (-1.0)
      hi := f2 (1.0) (1.0) }
  let bConst : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := f2 (0.25) (0.25)
      hi := f2 (0.25) (0.25) }
  let bOut : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := f2 (-0.75) (-0.75)
      hi := f2 (1.25) (1.25) }
  { inputId := 0
    inputDim := 2
    states := #[
      { shape := (shape![2])
        ibp? := some bIn
        aff? := none }
    , { shape := (shape![2])
        ibp? := some bConst
        aff? := none }
    , { shape := (shape![2])
        ibp? := some bOut
        aff? := none }
    ] }

#crown_view toyGraphCROWN, toyPropState

-- Interval widths (`hi - lo`) are a fast
-- "where did bounds blow up?" diagnostic.
#bounds_tightness_view toyGraphCROWN, toyPropState
```

# Autograd (Tape + Gradients)

TorchLean's eager autograd engine records a computation graph into a `Tape` and can run
reverse-mode to accumulate gradients. The widget below shows the recorded tape and the gradients
produced by scalar backprop (like `loss.backward()` in PyTorch).

```
open Runtime.Autograd
open Spec

def toyTape : Tape Float :=
  let (t0, aId) :=
    Tape.leaf (α := Float) (t := Tape.empty)
      (value := Tensor.scalar 2.0) (name := some "a")
  let (t1, bId) :=
    Tape.leaf (α := Float) (t := t0)
      (value := Tensor.scalar 3.0) (name := some "b")
  let (t2, abId) :=
    match Tape.mul (α := Float) (t := t1)
        (s := Shape.scalar) aId bId with
    | .ok r => r
    | .error _ => (t1, 0)
  let (t3, outId) :=
    match Tape.add (α := Float) (t := t2)
        (s := Shape.scalar) abId bId with
    | .ok r => r
    | .error _ => (t2, 0)
  let _ := outId
  t3

#tape_grads_view toyTape, 3

-- For a step by step explanation of why a grad exists (or is missing),
-- use the step by step reverse pass trace:
#tape_trace_view toyTape, 3
```

# Training Dashboards

TorchLean's widget layer is not limited to semantic objects like tensors and tapes. It also
includes a small monitoring surface for training and evaluation artifacts.

These logs are plain data structures, not a hidden runtime UI:

- `Runtime.Training.TrainLog` is just a pure record of steps, metric series, and notes,
- `Runtime.Training.ConfusionMatrix` is just a pure table of class counts,
- the widget layer renders them without changing their meaning.

That makes them suitable for pure Lean toy runs, runtime/autograd training loops, and imported
metrics from external experiments, all rendered through the same viewer.

```
def toyTrainLog : _root_.Runtime.Training.TrainLog :=
  { title := "Toy classifier run"
    steps := #[0, 1, 2, 3, 4]
    series := #[
      { name := "loss", values := #[1.20, 0.84, 0.59, 0.41, 0.33], color := "#c44" }
    , { name := "val_acc", values := #[0.30, 0.48, 0.61, 0.73, 0.79], color := "#0a7" }
    , { name := "lr", values := #[0.05, 0.05, 0.01, 0.01, 0.01], color := "#06c" }
    ]
    notes := #[
      "optimizer: SGD"
    , "scheduler: StepLR(step_size=2, gamma=0.2)"
    , "dataset: toy 3-class classifier"
    ] }

def toyLabels : Array String := #["cat", "dog", "owl"]

def toyCM : _root_.Runtime.Training.ConfusionMatrix :=
  { counts := #[
      #[8, 1, 0]
    , #[2, 6, 1]
    , #[0, 1, 7]
    ] }

#train_log_view toyTrainLog
#confusion_view toyLabels, toyCM
```

This widget family pairs particularly well with:

- the CSV loader training example,
- the NPY loader training example,
- the ResNet basic block training example,
- and the callback/reporting helpers under `API.train`.

# Runtime Context Viewer

When debugging a failed training step, one of the first questions is often not "what is the graph?"
but "which values and gradients are registered now?"

The runtime-context widget answers that question directly.

```
def anyScalar (x : Float) : Runtime.AnyTensor Float :=
  { s := .scalar, t := Tensor.scalar x }

def toyCtx : Runtime.RuntimeContext Float :=
  { var_registry := [
      ("w", anyScalar 3.0)
    , ("x", anyScalar 2.0)
    , ("wx", anyScalar 6.0)
    ]
    gradients := [
      ("w", anyScalar 2.0)
    , ("x", anyScalar 3.0)
    ]
    next_id := 3 }

#runtime_ctx_view toyCtx
```

This view is especially useful for comparing:

- the public training facade,
- the eager autograd tape,
- and the actual runtime registry that stores values and accumulated gradients.

# Widget Families

The following widget families are the usual starting set.

## 1. Tensor viewers

`#tensor_view`, `#anytensor_view`, `#tensor_stats_view`

What they show:

- small tensors as readable tables,
- shape-erased runtime tensors,
- min/max/mean/norm summaries for a quick numerical check.

Open:

- [tensor basics example API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean)
- [widgets example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)

## 2. IR viewer

`#ir_view`

What it shows:

- node ids,
- parent lists,
- op tags,
- output shapes,
- a DOT export for GraphViz.

Open:

- [widgets example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)
- [TorchLean IBP fixture API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Verification/TorchLean/TorchLeanIBP.lean)

## 3. Shape inference view

`#shape_infer_view`

What it shows:

- declared shapes versus inferred shapes,
- the first node where a shape mismatch appears,
- quick confirmation that a graph is well-formed.

Open:

- [widgets example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)
- [TorchLean Transformer IBP fixture API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Verification/TorchLean/TorchLeanTransformerIBP.lean)

## 4. Float32 viewers

`#float32_view`, `#float32_compare_view`, `#float32_round_view`

What they show:

- sign, exponent, and fraction bits,
- `qNaN` versus `sNaN`,
- raw payloads and comparison differences,
- how a `Float` rounds into `IEEE32Exec`.

Open:

- [widgets demo](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)
- [float32 modes demo](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Floats/Float32Modes.lean)

## 5. Verification views

`#crown_view`, `#bounds_tightness_view`

What they show:

- interval boxes,
- affine bounds,
- which nodes widen fastest,
- how a verification pass propagates uncertainty.

Open:

- [CROWN ops fixture](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Verification/TorchLean/TorchLeanCrownOps.lean)
- [IBP fixture](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Verification/TorchLean/TorchLeanIBP.lean)

## 6. Tape and gradient views

`#tape_view`, `#tape_grads_view`, `#tape_trace_view`

What they show:

- the eager autograd tape,
- accumulated gradients,
- the reverse-pass contribution of each node.

Open:

- [autograd basics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean)
- [widgets demo](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)

## 7. Training dashboards

`#train_log_view`, `#confusion_view`

What they show:

- loss, accuracy, and learning-rate curves,
- a compact metric table over the most recent steps,
- classwise confusion counts for quick classifier checks.

Open:

- [widgets demo](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)
- [CSV loader demo](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
- [ResNet basic block training](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/ResnetBasicblockTrain.lean)

## 8. Runtime context views

`#runtime_ctx_view`

What it shows:

- registered runtime variables,
- accumulated gradients,
- the shapes and small concrete values attached to each entry.

Open:

- [widgets demo](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)
- [runtime context API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Context.lean)

# A Good Starting File

For more complete interactive demos (kept fast enough to elaborate in an editor), open:

- [widgets demo](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Advanced/Widgets.lean)

The demo shows the same ideas with short inline explanations around each example.
