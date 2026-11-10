/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import TorchLeanExamples.Models.RL.GridWorld.Env
import TorchLean.Verify.CROWN.Op.ExecCROWN
meta import TorchLean.Verify.CROWN.Op.ExecCROWN
import TorchLean.NN.Tape.Apply
import TorchLean.NN.RL.Transition
import TorchLean.NN.Training
import TorchLean.NN.Widget.IR.ExecutionTrace
import TorchLean.NN.Widget.IR.Rewrite
import TorchLean.NN.Widget.IR.ShapeInference
import TorchLean.NN.Widget.PyTorchTranslator
import TorchLean.NN.Widget.Float32
import TorchLeanExamples.Models.RL.GridWorld.Widgets
import TorchLean.NN.Widget.Autograd
import TorchLean.NN.Widget.Training
import TorchLean.Verify.CROWN.Widget
import TorchLean.Tensor.Literal

/-!
# Widget Gallery

A gallery of the widget commands on small examples, with no command-line entry point. Build it with
`scripts/lake.sh build TorchLeanExamples.DeepDives.WidgetGallery`, then open it in an editor and put
the cursor on a command such as `#tensor_view`, `#ir_view` or `#float32_view`: Lean renders a small
interactive HTML panel in the infoview.

The widgets are inspection tools for teaching, debugging and reviewing artifacts without leaving
Lean. They live in the modules under `TorchLean/NN/Widget/` and `TorchLean/Verify/CROWN/`, which
the runtime and proof modules do not import.
-/


/-!
## RL (GridWorld) widgets

Compact infoview panels for iterating on RL specs and proofs, showing a GridWorld state, a policy
and a rollout path.
-/

open FloatLib.Floats (Binary32)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32)

namespace TorchLean.Examples.Widgets

namespace World

/-- A 4×4 GridWorld with start `(0, 0)` and goal `(3, 3)`. -/
def env : GridWorld 4 4 :=
  { start := (⟨0, by decide⟩, ⟨0, by decide⟩)
    goal := (⟨3, by decide⟩, ⟨3, by decide⟩)
    -- The widgets never read the discount, so any literal serves.
    discount := 0 }

/-- The position `(1, 2)`, drawn by the state widget `#gridworld_view`. -/
def pos : GridWorld.State 4 4 :=
  (⟨1, by decide⟩, ⟨2, by decide⟩)

/-- A constant policy that always moves right, drawn by `#gridworld_policy_view`. -/
def policy : GridWorld.State 4 4 → GridWorld.Action :=
  fun _ => GridWorld.Action.right

/--
A rollout path from the start `(0, 0)` to the goal `(3, 3)`, drawn by `#gridworld_path_view`.
-/
def path : Array (GridWorld.State 4 4) :=
  #[
    (⟨0, by decide⟩, ⟨0, by decide⟩),
    (⟨0, by decide⟩, ⟨1, by decide⟩),
    (⟨0, by decide⟩, ⟨2, by decide⟩),
    (⟨0, by decide⟩, ⟨3, by decide⟩),
    (⟨1, by decide⟩, ⟨3, by decide⟩),
    (⟨2, by decide⟩, ⟨3, by decide⟩),
    (⟨3, by decide⟩, ⟨3, by decide⟩)
  ]

#gridworld_view env, pos
#gridworld_policy_view env, policy
#gridworld_path_view env, path

end World

/--
A synthetic training log: a decaying loss with a small oscillation, and a saturating accuracy.

Hand-written curves have predictable extremes, which suits a check of the renderer's axis scaling
and legend layout.
-/
def log : Training.TrainLog :=
  let n : Nat := 40
  let steps : Array Nat := Array.range n
  let loss : Tensor Float [n] :=
    Tensor.generateFlat [n] (fun i =>
      let t : Float := Float.ofNat i
      -- `exp (-0.08 t)` decays from `1` toward `0`, with a sine wobble of amplitude `0.03`.
      (Float.exp (-0.08 * t)) + 0.03 * Float.sin (0.7 * t))
  let acc : Tensor Float [n] :=
    Tensor.generateFlat [n] (fun i =>
      let t : Float := Float.ofNat i
      -- Rises from `0.4` toward `1`, like a validation accuracy.
      0.4 + 0.6 * (1.0 - Float.exp (-0.12 * t)))
  { title := "Sample training loop"
    steps := steps
    series := #[
      { name := "loss", values := Tensor.to loss (Array Float), color := "#c44" }
    , { name := "acc", values := Tensor.to acc (Array Float), color := "#0a7" }
    ]
    notes := #[
      "sample curve for checking the widget layout"
    , "the same viewer renders JSON logs written by executable training loops"
    ] }

/-- Three class names for the confusion-matrix widget. -/
def labels : Array String := #["cat", "dog", "owl"]

/-- A three-class confusion matrix with a clear diagonal and a few off-diagonal mistakes. -/
def confusion : Training.ConfusionMatrix :=
  { counts := #[
      #[8, 1, 0]
    , #[2, 6, 1]
    , #[0, 1, 7]
    ] }

/-- The simplest possible tensor view: the numbers zero through four. -/
def indices : Tensor Nat [5] :=
  [0, 1, 2, 3, 4]

/--
A rank-three tensor whose entries encode their coordinates as `100 i + 10 j + k`, so the viewer's
axis ordering can be read straight off the values.
-/
def grid : Tensor Nat [2, 3, 4] :=
  Tensor.generate [2, 3, 4] fun coordinates =>
    coordinates.getD 0 0 * 100 + coordinates.getD 1 0 * 10 + coordinates.getD 2 0

/--
`0.1`, given by its bit pattern.

`0.1` has no finite binary expansion, and `0x3fb999999999999a` is the binary64 value nearest to
it, about `5.55e-18` above. The float widgets show this value and how it rounds to binary32.
-/
def decimalTenth : Float :=
  Float.ofBits 0x3fb999999999999a

/-- `1/3`, likewise given by its bit pattern. -/
def third : Float :=
  Float.ofBits 0x3fd5555555555555

/-- Two exact values and two inexact ones, so the float viewer has both cases to render. -/
def native : Tensor Float [4] :=
  [1.0, 2.0, decimalTenth, third]

/--
The same four values in the bit-level binary32 model, where the rounding is visible in the fields.
-/
def ieee : Tensor Binary32 [4] :=
  [ (1 : Binary32)
  , Binary.ofFloat 2.0
  , Binary.ofFloat decimalTenth
  , Binary.ofFloat third ]

/--
A `[2, 2, 3]` tensor with entries `(100 i + 10 j + k + decimalTenth) / 7`, computed in host
`Float` and rounded to binary32. The viewer shows the fields of the rounded values.
-/
def cube : Tensor Binary32 [2, 2, 3] :=
  Tensor.generate [2, 2, 3] fun coordinates =>
    let base : Float := Float.ofNat
      (coordinates.getD 0 0 * 100 + coordinates.getD 1 0 * 10 + coordinates.getD 2 0)
    Binary.ofFloat ((base + decimalTenth) / 7.0)

/-- A small integer matrix, to show the viewer with no rounding to worry about. -/
def matrix : Tensor Int [2, 4] :=
  [[0, 1, 2, 3], [10, 11, 12, 13]]

/--
The same matrix with its shape existentially packed, which is what the shape-agnostic widgets take.
-/
def packedMatrix : SomeTensor Int :=
  SomeTensor.ofTensor matrix

/-- A three-node graph with a selectable final operator.

The gallery compares addition and subtraction with the same inputs and shapes. -/
def graph (op : Op [.value [2], .value [2]] [.value [2]]) : IR.Graph :=
  { nodes := #[.input 0 (.value [2]), .const (.value [2]), .op ⟨_, _, op⟩ #[⟨0, 0⟩, ⟨1, 0⟩]]
    outputs := #[⟨2, 0⟩] }

/-- The concrete input fed to the addition graph. -/
def input : List (Op.Value Float) :=
  [⟨.value [2], ([0.60, -0.20] : Tensor Float [2])⟩]

/-- The constant attached to node 1 of `graph`. -/
def payload : IR.Payload Float :=
  { const? := fun id =>
      if id = 1 then
        -- Node 1 is the `.const` in `graph` (a fixed vector).
        some ⟨.value [2], ([0.25, 0.25] : Tensor Float [2])⟩
      else
        none }

/-- `1.0` in binary32, by bit pattern. -/
def one : Binary32 :=
  ofBits32 (0x3f800000 : UInt32)

/-- A quiet NaN, so the float widget's non-finite rendering path gets exercised. -/
def qnan : Binary32 :=
  ofBits32 (0x7fc00000 : UInt32)

/-- The input box of the addition graph: `[-1, 1]` in both coordinates. -/
def inputBox : CROWN.FlatBox Float :=
  { dim := 2
    lo := [-1.0, -1.0]
    hi := [1.0, 1.0] }

/-- The seed box of each graph input: `inputBox` for input node 0. -/
def seeds (id : Nat) : Option (CROWN.FlatBox Float) :=
  if id = 0 then some inputBox else none

/-- The IBP boxes of the addition graph: the input box, the constant's point box, and their sum. -/
def boxes : Array (Option (CROWN.FlatBox Float)) :=
  CROWN.Op.runIBP (graph (op := .add)) payload seeds

/-- The CROWN affine bounds of every node of the addition graph over input node 0. -/
def affine : Array (Option (CROWN.Graph.FlatAffineBounds Float)) :=
  CROWN.Op.runCROWN (graph (op := .add)) payload { inputId := 0, inputDim := 2 } seeds
    boxes inputBox

private def record : Result (Tape (SomeTensor Float)) := do
  let (t0, aId) := Tape.leaf (Tape.empty : Tape (SomeTensor Float))
    (.ofTensor (Tensor.full [] 2.0)) (name := some "a")
  let (t1, bId) := Tape.leaf t0 (.ofTensor (Tensor.full [] 3.0)) (name := some "b")
  let (t2, abId, _) ← Tape.apply t1 "mul" (Op.mul (s := [])) (aId, bId, ())
  let (t3, _) ← Tape.apply t2 "add" (Op.add (s := [])) (abId, bId, ())
  pure t3

/--
A four-node autograd tape computing `a * b + b` at `a = 2`, `b = 3`.

The `decide` establishes that recording succeeded, so the widget receives the complete tape.
-/
def tape : Tape (SomeTensor Float) :=
  have succeeds : record.toOption.isSome = true := by decide
  record.toOption.get succeeds

-- Put the cursor on any of these commands to see its panel in the infoview.
#tensor_view indices
#tensor_view grid
#tensor_view native
#tensor_view ieee
#tensor_view cube
#tensor_view matrix
#tensor_stats_view native
#tensor_stats_view ([0.60, -0.20] : Tensor Float [2])
#ir_view (graph (op := .add))
#shape_infer_view (graph (op := .add))
#graph_rewrite_view (graph (op := .add)), (graph (op := .sub))
#float32_view one
#float32_view (1 : Binary32)
#float32_view qnan
#float32_compare_view one, qnan
#anytensor_view packedMatrix
#ir_exec_trace_view (graph (op := .add)), payload, input
#train_log_view log
#confusion_view labels, confusion

-- Compare Float64 input to its Float32 rounding.
#float32_round_view decimalTenth
#float32_round_view third

-- Verification: show the IBP boxes and CROWN bounds of `graph`.
#crown_view (graph (op := .add)), boxes, affine
#bounds_tightness_view (graph (op := .add)), boxes

-- Autograd: show a compact tape and its scalar backprop (like `loss.backward()`).
#tape_view tape
#tape_grads_view tape, 3
#tape_trace_view tape, 3

-- Interop: preview how a small PyTorch model maps to TorchLean constructors.
#pytorch_translate_file "TorchLeanExamples/Interop/PyTorch/MLP/train_mlp.py"

end TorchLean.Examples.Widgets
