/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Spec.RL.Envs.GridWorld
import NN.MLTheory.CROWN.Graph.Core
import NN.Runtime.Autograd.Engine.Core.Elementwise
import NN.Runtime.RL.Core
import NN.Runtime.Training.Log
import NN.Widgets.IR.ExecutionTrace
import NN.Widgets.IR.Rewrite
import NN.Widgets.IR.ShapeInference
import NN.Widgets.Interop.PyTorchTranslator
import NN.Widgets.Numerics.Float32
import NN.Widgets.RL.GridWorld
import NN.Widgets.Runtime.Autograd
import NN.Widgets.Runtime.Training
import NN.Widgets.Verification.CROWN

/-!
# Widget Gallery

The gallery is best explored in an editor:
- put the cursor on a `#tensor_view` / `#ir_view` / `#float32_view` line, and
- Lean will render a small interactive HTML panel in the infoview.

These widgets are inspection tools for teaching, debugging, and reviewing artifacts
without leaving Lean. They are available through the dedicated widget entrypoint
(`import NN.Widgets`) so ordinary runtime and proof imports stay focused.
-/

open Spec TorchLean
open NN.IR
open Runtime.Autograd

/-!
## RL (GridWorld) widgets

These compact panels are useful when iterating on RL specs and proofs: they let you inspect
state encodings, policies, and rollout traces in the infoview.
-/

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32 ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model)

namespace Gallery

namespace World

open Spec.RL.Envs

/-- A small 4×4 GridWorld used by the widget gallery. -/
def env : GridWorld 4 4 :=
  { start := (⟨0, by decide⟩, ⟨0, by decide⟩)
    goal := (⟨3, by decide⟩, ⟨3, by decide⟩)
    -- Discount isn't used by the widgets, so we pick a simple literal.
    discount := 0 }

/-- Example GridWorld position rendered by the state widget. -/
def pos : GridWorld.State 4 4 :=
  (⟨1, by decide⟩, ⟨2, by decide⟩)

/-- Constant policy for the policy-visualization demo. -/
def policy : GridWorld.State 4 4 → GridWorld.Action :=
  fun _ => GridWorld.Action.right

/-- Example rollout path rendered by the GridWorld trace widget. -/
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

Synthetic rather than recorded, because the point is to exercise the renderer's axis scaling and
legend layout, and a hand-written curve gives predictable extremes.
-/
def log : Runtime.Training.TrainLog :=
  let n : Nat := 40
  let steps : Array Nat := Array.range n
  let loss : Tensor Float [n] :=
    Tensor.generateFlat [n] (fun i =>
      let t : Float := Float.ofNat i
      -- A compact decreasing curve with a small oscillation, useful for checking rendering.
      (Float.exp (-0.08 * t)) + 0.03 * Float.sin (0.7 * t))
  let acc : Tensor Float [n] :=
    Tensor.generateFlat [n] (fun i =>
      let t : Float := Float.ofNat i
      -- A compact increasing curve that saturates, like a validation metric.
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
def confusion : Runtime.Training.ConfusionMatrix :=
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

Written this way because `0.1` is not representable in binary: the literal is the nearest binary64
value, and the widget's job is to show exactly how far off that is.
-/
def decimalTenth : Float :=
  -- 0.1 as a binary64 literal (exact via bit pattern).
  Float.ofBits 0x3fb999999999999a

/-- `1/3`, likewise given by its bit pattern. -/
def third : Float :=
  -- 1/3 as a binary64 literal.
  Float.ofBits 0x3fd5555555555555

/-- Two exact values and two inexact ones, so the float viewer has both cases to render. -/
def native : Tensor Float [4] :=
  [1.0, 2.0, decimalTenth, third]

/--
The same four values in the bit-level binary32 model, where the rounding is visible in the fields.
-/
def ieee : Tensor (Binary 8 23) [4] :=
  [ (1 : Binary 8 23)
  , (fun x => (ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat x))) : Binary 8 23))
    2.0
  , (ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat decimalTenth))) : Binary 8 23)
  , (ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat third))) : Binary 8
    23) ]

/--
A rank-three tensor built by adding `decimalTenth` to coordinate-derived values and dividing by
seven in host `Float`, then converting to binary32. The viewer exposes the resulting rounded fields.
-/
def cube : Tensor (Binary 8 23) [2, 2, 3] :=
  Tensor.generate [2, 2, 3] fun coordinates =>
    -- Small tensor whose values make the bit patterns interesting.
    let base : Float := Float.ofNat
      (coordinates.getD 0 0 * 100 + coordinates.getD 1 0 * 10 + coordinates.getD 2 0)
    (ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat ((base + decimalTenth) /
      7.0)))) : Binary 8 23)

/-- A small integer matrix, to show the viewer with no rounding to worry about. -/
def matrix : Tensor Int [2, 4] :=
  [[0, 1, 2, 3], [10, 11, 12, 13]]

/--
The same matrix with its shape existentially packed, which is what the shape-agnostic widgets take.
-/
def packedMatrix : Spec.SomeTensor Int :=
  Spec.SomeTensor.ofTensor matrix

/-- A three-node graph with a selectable final operation.

The gallery compares addition and subtraction with the same inputs and shapes. -/
def graph (op : NN.IR.OpKind) : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input
        outShape := [2] },
      { id := 1, parents := #[]
        kind := .const [2]
        outShape := [2] },
      { id := 2, parents := #[0, 1], kind := op
        outShape := [2] }
    ] }

/-- The concrete input fed to the addition graph. -/
def input : Spec.SomeTensor Float :=
  Spec.SomeTensor.ofTensor ([0.60, -0.20] : Tensor Float [2])

/-- The constant attached to node 1 of `graph`. -/
def payload : NN.IR.Payload Float :=
  { const? := fun id =>
      if id = 1 then
        -- Node 1 is the `.const` in `graph` (a fixed vector).
        some { n := 2, v := [0.25, 0.25] }
      else
        none }

/-- `1.0` in binary32, by bit pattern. -/
def one : Binary 8 23 :=
  ofBits32 (0x3f800000 : UInt32)

/-- A quiet NaN, so the float widget's non-finite rendering path gets exercised. -/
def qnan : Binary 8 23 :=
  ofBits32 (0x7fc00000 : UInt32)

/--
A display fixture for the addition graph: the input ranges over `[-1, 1]` in both coordinates,
the constant is a point box, and the output box is what adding them gives.
-/
def bounds : NN.MLTheory.CROWN.Graph.PropState Float :=
  let bIn : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := [-1.0, -1.0]
      hi := [1.0, 1.0] }
  let bConst : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := [0.25, 0.25]
      hi := [0.25, 0.25] }
  let bOut : NN.MLTheory.CROWN.FlatBox Float :=
    { dim := 2
      lo := [-0.75, -0.75]
      hi := [1.25, 1.25] }
  { inputId := 0
    inputDim := 2
    states := #[
      { shape := [2], ibp? := some bIn, aff? := none }
    , { shape := [2], ibp? := some bConst, aff? := none }
    , { shape := [2], ibp? := some bOut, aff? := none }
    ] }

private def record : Result (Tape Float) := do
  let (t0, aId) := Tape.leaf (α := Float) (t := Tape.empty) (value := Tensor.full [] 2.0) (name :=
    some "a")
  let (t1, bId) := Tape.leaf (α := Float) (t := t0) (value := Tensor.full [] 3.0) (name := some "b")
  let (t2, abId) ← Tape.mul (α := Float) (t := t1) (s := []) aId bId
  let (t3, _) ← Tape.add (α := Float) (t := t2) (s := []) abId bId
  pure t3

/--
A four-node autograd tape computing `a * b + b` at `a = 2`, `b = 3`.

The `decide` establishes that recording succeeded, so the widget receives the complete tape.
-/
def tape : Tape Float :=
  have succeeds : record.toOption.isSome = true := by decide
  record.toOption.get succeeds

-- Try hovering/cursoring on these commands in the editor.
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
#float32_view (1 : Binary 8 23)
#float32_view qnan
#float32_compare_view one, qnan
#anytensor_view packedMatrix
#ir_exec_trace_view (graph (op := .add)), payload, input
#train_log_view log
#confusion_view labels, confusion

-- Compare Float64 input to its Float32 rounding.
#float32_round_view decimalTenth
#float32_round_view third

-- Verification: show a small CROWN/IBP state aligned with `graph`.
#crown_view (graph (op := .add)), bounds
#bounds_tightness_view (graph (op := .add)), bounds

-- Autograd: show a compact tape and its scalar backprop (like `loss.backward()`).
#tape_view tape
#tape_grads_view tape, 3
#tape_trace_view tape, 3

-- Interop: preview how a small PyTorch model maps to TorchLean constructors.
#pytorch_translate_file "NN/Examples/Interop/PyTorch/MLP/train_mlp.py"

end Gallery
