/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Floats.IEEEExec.Exec32
import NN.IR.Graph
import NN.IR.Semantics
import NN.MLTheory.CROWN.Graph
import NN.Runtime.Autograd.Engine.Core
import NN.Spec.RL.Envs.GridWorld
import NN.Entrypoint.Widgets
meta import NN.Spec.Core.TensorBridge

/-!
# Widget Gallery

This file is best explored in an editor:
- put the cursor on a `#tensor_view` / `#ir_view` / `#float32_view` line, and
- Lean will render a small interactive HTML panel in the infoview.

These widgets are inspection tools for teaching, debugging, and reviewing artifacts
without leaving Lean. They are intentionally available through the dedicated widget entrypoint
(`import NN.Entrypoint.Widgets`) so ordinary runtime and proof imports stay lightweight.
-/

open Spec
open NN.IR
open TorchLean.Floats.IEEE754
open Runtime.Autograd
open TensorBridge TensorArray

/-!
## RL (GridWorld) widgets

These small panels are useful when iterating on RL specs and proofs: they let you quickly sanity-check
state encodings, policies, and rollout traces in the infoview.
-/

namespace GridWorldWidgets

open Spec.RL.Envs

def gw44 : GridWorld 4 4 :=
  { start := (⟨0, by decide⟩, ⟨0, by decide⟩)
    goal := (⟨3, by decide⟩, ⟨3, by decide⟩)
    -- Discount isn't used by the widgets, so we pick a simple literal.
    discount := 0 }

def pos : GridWorld.State 4 4 :=
  (⟨1, by decide⟩, ⟨2, by decide⟩)

def goRightPolicy : GridWorld.State 4 4 → GridWorld.Action :=
  fun _ => GridAction.right

def samplePath : Array (GridWorld.State 4 4) :=
  #[
    (⟨0, by decide⟩, ⟨0, by decide⟩),
    (⟨0, by decide⟩, ⟨1, by decide⟩),
    (⟨0, by decide⟩, ⟨2, by decide⟩),
    (⟨0, by decide⟩, ⟨3, by decide⟩),
    (⟨1, by decide⟩, ⟨3, by decide⟩),
    (⟨2, by decide⟩, ⟨3, by decide⟩),
    (⟨3, by decide⟩, ⟨3, by decide⟩)
  ]

#gridworld_view gw44, pos
#gridworld_policy_view gw44, goRightPolicy
#gridworld_path_view gw44, samplePath

end GridWorldWidgets

def sampleTrainLog : Runtime.Training.TrainLog :=
  let n : Nat := 40
  let steps : Array Nat := (Array.range n).map (fun i => i)
  let loss : Array Float :=
    (Array.range n).map (fun i =>
      let t : Float := Float.ofNat i
      -- A compact decreasing curve with a small oscillation, useful for checking rendering.
      (Float.exp (-0.08 * t)) + 0.03 * Float.sin (0.7 * t))
  let acc : Array Float :=
    (Array.range n).map (fun i =>
      let t : Float := Float.ofNat i
      -- A compact increasing curve that saturates, like a validation metric.
      0.4 + 0.6 * (1.0 - Float.exp (-0.12 * t)))
  { title := "Sample training loop"
    steps := steps
    series := #[
      { name := "loss", values := loss, color := "#c44" }
    , { name := "acc", values := acc, color := "#0a7" }
    ]
    notes := #[
      "sample curve for checking the widget layout"
    , "the same viewer renders JSON logs written by executable training loops"
    ] }

/-!
Actual training curve (SGD on a small regression dataset).

This is a real “training loop” computed in Lean. It is intentionally compact (2→2→1 MLP) so the
widget evaluator can run it quickly inside the editor.

Note:
We keep this example **self-contained** (no Spec-tensor forward/backward), because the full
spec-side MLP backprop pipeline is large and would make this widget file slow to elaborate.
-/

abbrev Vec2 := Float × Float
abbrev Mat2 := Vec2 × Vec2

private def vsub (a b : Vec2) : Vec2 := (a.1 - b.1, a.2 - b.2)
private def vscale (a : Vec2) (c : Float) : Vec2 := (a.1 * c, a.2 * c)
private def dot (w x : Vec2) : Float := w.1 * x.1 + w.2 * x.2

private def relu (x : Float) : Float := if x <= 0.0 then 0.0 else x
private def drelu (x : Float) : Float := if x <= 0.0 then 0.0 else 1.0
private def relu2 (x : Vec2) : Vec2 := (relu x.1, relu x.2)

private def tinyData : Array (Vec2 × Float) :=
  -- y = x1 + 2*x2, sampled at a few points
  #[
    ((0.0, 0.0), 0.0)
  , ((1.0, -1.0), -1.0)
  , ((1.0, 0.0), 1.0)
  , ((0.0, 1.0), 2.0)
  ]

private structure Params where
  w1 : Mat2
  b1 : Vec2
  w2 : Vec2
  b2 : Float

private def initParams : Params :=
  { w1 := ((0.12, -0.08), (0.05, 0.10))
    b1 := (0.01, -0.02)
    w2 := (0.07, -0.04)
    b2 := 0.0 }

private def lr : Float := 0.03

private def forward (p : Params) (x : Vec2) : (Float × Vec2 × Vec2) :=
  -- Returns (yhat, z1, a1) where z1 is pre-activation and a1 is ReLU(z1).
  let z1 : Vec2 :=
    (dot p.w1.1 x + p.b1.1, dot p.w1.2 x + p.b1.2)
  let a1 : Vec2 := relu2 z1
  let yhat : Float := dot p.w2 a1 + p.b2
  (yhat, z1, a1)

private def sgdStep (p : Params) (x : Vec2) (y : Float) : (Params × Float × Float) :=
  let (yhat, z1, a1) := forward p x
  let err := yhat - y
  let loss := 0.5 * err * err
  -- Gradients.
  let dLdy : Float := err
  let dW2 : Vec2 := vscale a1 dLdy
  let db2 : Float := dLdy
  let da1 : Vec2 := vscale p.w2 dLdy
  let dz1 : Vec2 := (da1.1 * drelu z1.1, da1.2 * drelu z1.2)
  let dW1r1 : Vec2 := vscale x dz1.1
  let dW1r2 : Vec2 := vscale x dz1.2
  let db1 : Vec2 := dz1
  -- SGD update.
  let p' : Params :=
    { w1 := (vsub p.w1.1 (vscale dW1r1 lr), vsub p.w1.2 (vscale dW1r2 lr))
      b1 := vsub p.b1 (vscale db1 lr)
      w2 := vsub p.w2 (vscale dW2 lr)
      b2 := p.b2 - lr * db2 }
  let absErr := Float.abs err
  (p', loss, absErr)

private def trainLoop : Nat → Params → List Float → List Float → (List Float × List Float)
  | 0, _, losses, errs => (losses.reverse, errs.reverse)
  | Nat.succ k, p, losses, errs =>
      let (x, y) := tinyData[(k % tinyData.size)]!
      let (p', loss, absErr) := sgdStep p x y
      trainLoop k p' (loss :: losses) (absErr :: errs)

def mlpTrainLog : Runtime.Training.TrainLog :=
  let stepsN : Nat := 80
  let (losses, errs) := trainLoop stepsN initParams [] []
  { title := "MLP SGD (real run, pure Lean)"
    steps := (Array.range stepsN).map (fun i => i)
    series := #[
      { name := "mse_loss", values := losses.toArray, color := "#c44" }
    , { name := "abs_err", values := errs.toArray, color := "#0a7" }
    ]
    notes := #[
      "model: 2->2->1 ReLU MLP (scalar implementation for speed)"
    , "data: y = x1 + 2*x2 (4 points)"
    , s!"lr={lr}, steps={stepsN}"
    ] }

def sampleLabels : Array String := #["cat", "dog", "owl"]

def sampleConfusionMatrix : Runtime.Training.ConfusionMatrix :=
  { counts := #[
      #[8, 1, 0]
    , #[2, 6, 1]
    , #[0, 1, 7]
    ] }

def vec5 : Tensor Nat (shape![5]) :=
  Tensor.dim (fun i => Tensor.scalar i.1)

def cube234 : Tensor Nat (shape![2, 3, 4]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.dim (fun k =>
        Tensor.scalar (i.1 * 100 + j.1 * 10 + k.1))))

def f01 : Float :=
  -- 0.1 as a binary64 literal (exact via bit pattern).
  Float.ofBits 0x3fb999999999999a

def fthird : Float :=
  -- 1/3 as a binary64 literal.
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
    | ⟨1, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat (Float.ofNat 2))
    | ⟨2, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat f01)
    | ⟨_, _⟩ => Tensor.scalar (IEEE32Exec.ofFloat fthird))

def cubeF32 : Tensor IEEE32Exec (shape![2, 2, 3]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.dim (fun k =>
        -- Small rank-3 tensor with values that make bit-patterns interesting.
        let base : Float := Float.ofNat (i.1 * 100 + j.1 * 10 + k.1)
        let x : Float := (base + f01) / 7.0
        Tensor.scalar (IEEE32Exec.ofFloat x))))

def mat2x4 : Tensor Int (shape![2, 4]) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (Int.ofNat (i.1 * 10 + j.1))))

def anyMat : Runtime.AnyTensor Int :=
  { s := (shape![2, 4]), t := mat2x4 }

def anyF (x : Float) : Runtime.AnyTensor Float :=
  { s := .scalar, t := Tensor.scalar x }

def sampleRuntimeContext : Runtime.RuntimeContext Float :=
  { var_registry := [
      ("w", anyF 3.0)
    , ("x", anyF 2.0)
    ]
    gradients := [
      ("w", anyF 0.1)
    , ("x", anyF 0.0)
    ]
    next_id := 2 }

def sampleGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .add
        outShape := (shape![2]) }
    ] }

def sampleGraphSub : NN.IR.Graph :=
  -- Same as `sampleGraph` but with `sub` instead of `add` at the output.
  { nodes := #[
      { id := 0, parents := [], kind := .input
        outShape := (shape![2]) },
      { id := 1, parents := []
        kind := .const (shape![2])
        outShape := (shape![2]) },
      { id := 2, parents := [0, 1], kind := .sub
        outShape := (shape![2]) }
    ] }

def f2 (x y : Float) : Tensor Float (shape![2]) :=
  Tensor.dim (fun
    | ⟨0, _⟩ => Tensor.scalar x
    | ⟨_, _⟩ => Tensor.scalar y)

def sampleInput : Runtime.AnyTensor Float :=
  { s := (shape![2])
    t := f2 0.60 (-0.20) }

def samplePayload : NN.IR.Payload Float :=
  { const? := fun id =>
      if id = 1 then
        -- Node 1 is the `.const` in `sampleGraph` (a fixed vector).
        some { n := 2, v := f2 0.25 0.25 }
      else
        none }

def one : IEEE32Exec :=
  IEEE32Exec.ofBits (0x3f800000 : UInt32)

def qnan : IEEE32Exec :=
  IEEE32Exec.ofBits (0x7fc00000 : UInt32)

def samplePropState : NN.MLTheory.CROWN.Graph.PropState Float :=
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
      { shape := (shape![2]), ibp? := some bIn, aff? := none }
    , { shape := (shape![2]), ibp? := some bConst, aff? := none }
    , { shape := (shape![2]), ibp? := some bOut, aff? := none }
    ] }

def sampleTape : Tape Float :=
  let (t0, aId) := Tape.leaf (α := Float) (t := Tape.empty) (value := Tensor.scalar 2.0) (name :=
    some "a")
  let (t1, bId) := Tape.leaf (α := Float) (t := t0) (value := Tensor.scalar 3.0) (name := some "b")
  let (t2, abId) :=
    match Tape.mul (α := Float) (t := t1) (s := Shape.scalar) aId bId with
    | .ok r => r
    | .error _ => (t1, 0)
  let (t3, outId) :=
    match Tape.add (α := Float) (t := t2) (s := Shape.scalar) abId bId with
    | .ok r => r
    | .error _ => (t2, 0)
  -- Sanity: in this construction we expect ids [0=a,1=b,2=mul,3=add].
  -- We keep the final tape even if a future implementation changes ids.
  let _ := outId
  t3

/-!
Tensor basics / bridge example:

Widgets are defined for `Spec.Tensor` (shape-indexed, spec-level tensors). If you have an
array-backed `TensorArray.Tensor` (common at IO boundaries), you can convert it with
`TensorBridge.to_tensor` and then use the same `#tensor_view` UI.
-/

meta def taMat23 : TensorArray.Tensor Float [2, 3] :=
  TensorArray.ofArray #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0] [2, 3] (by simp)

meta def taMat23_spec : Tensor Float (listToShape [2, 3]) :=
  toTensor taMat23

-- Try hovering/cursoring on these commands in the editor.
#tensor_view vec5
#tensor_view cube234
#tensor_view vecF64
#tensor_view vecF32
#tensor_view cubeF32
#tensor_view mat2x4
#tensor_view taMat23_spec
#tensor_stats_view vecF64
#tensor_stats_view (f2 0.60 (-0.20))
#ir_view sampleGraph
#shape_infer_view sampleGraph
#graph_rewrite_view sampleGraph, sampleGraphSub
#float32_view one
#float32_view (1 : IEEE32Exec)
#float32_view qnan
#float32_compare_view one, qnan
#anytensor_view anyMat
#runtime_ctx_view sampleRuntimeContext
#ir_exec_trace_view sampleGraph, samplePayload, sampleInput
#train_log_view mlpTrainLog
#train_log_view sampleTrainLog
#confusion_view sampleLabels, sampleConfusionMatrix

-- Compare Float64 input to its Float32 rounding.
#float32_round_view f01
#float32_round_view fthird

-- Verification: show a small CROWN/IBP state aligned with `sampleGraph`.
#crown_view sampleGraph, samplePropState
#bounds_tightness_view sampleGraph, samplePropState

-- Autograd: show a compact tape and its scalar backprop (like `loss.backward()`).
#tape_view sampleTape
#tape_grads_view sampleTape, 3
#tape_trace_view sampleTape, 3
