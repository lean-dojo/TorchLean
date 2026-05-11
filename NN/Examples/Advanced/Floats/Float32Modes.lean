/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Batteries.Lean.Float
public import NN
import NN.Entrypoint.Widgets

/-!
# Float32 modes tutorial

This tutorial runs the *same* compact MLP under two executable scalar backends:

- `Float`: Lean runtime `Float` (`binary64` on ordinary platforms, trusted/external semantics);
- `TorchLean.Floats.IEEE32Exec`: TorchLean's executable bit-level IEEE-754 binary32 model.

We run a single forward pass and a single reverse-mode VJP (seeded with `1.0`) and then report
`max_abs_diff` between the Float result and the IEEE32Exec result (converted back to Float).

PyTorch analogue:

```python
import torch
import torch.nn as nn

model64 = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)).to(torch.float64)
model32 = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)).to(torch.float32)

# Copy the same explicit weights into both models, run forward/backward,
# then compare outputs and gradients after casting to a common dtype.
```

TorchLean's difference is that the scalar backend is part of the typeclass context. The model and
autograd call are generic; only `α` changes.

Run:
  `lake exe torchlean float32_modes`

For editor inspection, put the cursor on the `#float32_*` commands below. Those widgets are for
visualization only; the actual tutorial code uses ordinary `def` and `IO` definitions.
-/

@[expose] public section


namespace NN.Examples.Advanced.Floats.Float32Modes

open _root_.Spec
open _root_.Spec.Tensor
open NN.API
open TorchLean.Floats.IEEE754

/-
This tutorial intentionally uses the *public* TorchLean API surface:

- models: `API.nn.Sequential` built from `API.nn.*`
- execution: `API.nn.compileOut` + `API.nn.predict1`
- autodiff: `API.autograd.model.vjpParams` / `vjpInputs`

No `Runtime.Autograd.*` tape/session machinery appears in this file.
-/

/-! ## Float32 widget probes -/

/--
`0.1` is the canonical "not exactly representable" decimal. The widget shows the binary32 value
that `IEEE32Exec` receives after rounding from the host literal.

PyTorch analogue:

```python
torch.tensor(0.1, dtype=torch.float32)
```
-/
def decimalTenth : Float := 0.1

/-- A simple finite binary32 value for the bit-layout widget. -/
def one32 : IEEE32Exec := IEEE32Exec.ofFloat 1.0

/-- Canonical quiet NaN, useful for showing classification and comparison behavior. -/
def quietNaN32 : IEEE32Exec := IEEE32Exec.canonicalNaN

#float32_round_view decimalTenth
#float32_view one32
#float32_view quietNaN32
#float32_compare_view one32, quietNaN32

def model : nn.Sequential (NN.Tensor.Shape.Vec 2) (NN.Tensor.Shape.Vec 1) :=
  -- A compact 2-layer MLP with ReLU:
  --   Linear(2 -> 3) -> ReLU -> Linear(3 -> 1)
  --
  -- This uses the public `API.nn` facade (PyTorch-like layer stacking and named configs).
  --
  -- Note: this tutorial supplies *explicit* parameter tensors below, so the init seeds are irrelevant
  -- here.
  nn.blocks.mlp 2 1 { hidden := [3], activation := .relu }

-- This tutorial returns a single typed bundle so we can:
-- - print everything in one place, and
-- - compare Float vs IEEE32Exec numerically at the end.
def OutShapes : List Spec.Shape :=
  [NN.Tensor.Shape.Vec 1, NN.Tensor.Shape.Mat 3 2, NN.Tensor.Shape.Vec 3,
   NN.Tensor.Shape.Mat 1 3, NN.Tensor.Shape.Vec 1, NN.Tensor.Shape.Vec 2]
abbrev OutPack (α : Type) :=
  tlist.TList α OutShapes

def runOnce {α : Type}
    [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [Runtime.Scalar α]
    (tag : String) : IO (OutPack α) := do
  -- TorchLean examples typically treat `Float` as the "host literal" type, then inject those
  -- literals into
  -- the chosen executable scalar `α` via `Runtime.ofFloat`.
  let cast : Float → α := Runtime.ofFloat

  /-
  ### 1. Explicit parameters

  PyTorch analogue:

  ```python
  with torch.no_grad():
      model[0].weight.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
      model[0].bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
      model[2].weight.copy_(torch.tensor([[0.7, 0.8, 0.9]]))
      model[2].bias.copy_(torch.tensor([0.4]))
  ```

  `autograd.model.Params model α` is a typed list (`TList`) of tensors. The shapes are determined by
  the model, so the parameter order cannot be silently permuted.
  -/
  let params : autograd.model.Params model α :=
    tlist!
      (NN.Tensor.tensorNDOfLenEq (α := α) [3, 2]
        [cast 0.1, cast 0.2, cast 0.3, cast 0.4, cast 0.5, cast 0.6]
        (by rfl)),
      (NN.Tensor.tensorNDOfLenEq (α := α) [3]
        [cast 0.1, cast 0.2, cast 0.3] (by rfl)),
      (NN.Tensor.tensorNDOfLenEq (α := α) [1, 3]
        [cast 0.7, cast 0.8, cast 0.9] (by rfl)),
      (NN.Tensor.tensorNDOfLenEq (α := α) [1] [cast 0.4] (by rfl))

  -- One input vector x in R^2.
  let x : Spec.Tensor α (NN.Tensor.Shape.Vec 2) :=
    NN.Tensor.tensorNDOfLenEq (α := α) [2] [cast 0.5, cast 0.8] (by rfl)

  /-
  ### 2. Forward pass

  PyTorch analogue:

  ```python
  y = model(x)
  ```

  `compileOut` specializes the model's forward program into a callable object. TorchLean models are
  backend-generic, so this step chooses a concrete executable path for the chosen scalar `α`.
  -/
  let compiled ← nn.compileOut (α := α) model
  -- Run `y = model(params, x)` (single-example predict; no batching here).
  let y := nn.predict1 (α := α) model compiled params x

  /-
  ### 3. Reverse-mode VJP

  PyTorch analogue:

  ```python
  y.sum().backward()
  dparams = [p.grad for p in model.parameters()]
  dX = x.grad
  ```

  A VJP needs an output cotangent seed. Since the output shape is `Vec 1`, seeding with `[1]`
  computes the same gradient as differentiating `sum(y)`.
  -/
  let seedOut : Spec.Tensor α (NN.Tensor.Shape.Vec 1) :=
    Spec.fill (α := α) (cast 1.0) (NN.Tensor.Shape.Vec 1)

  -- Gradients w.r.t. *parameters* (same `TList` structure/order as `params`).
  let dparams ← autograd.model.vjpParams (α := α) model params x seedOut
  -- Gradients w.r.t. *inputs* (here: just the input vector `dX`, no `TList` noise).
  let dX ← autograd.model.vjpInput (α := α) model params x seedOut

  /-
  ### 4. Unpack the typed gradient list

  TorchLean uses a typed list (`TList`) so parameter-gradient shapes stay in the type. The helper
  `tlist.unpack4` is just less noisy than pattern matching on the typed list.
  -/
  let (dW1, db1, dW2, db2) := tlist.unpack4 dparams

  IO.println s!"== {tag} =="
  IO.println s!"y   = {Spec.pretty y}"
  IO.println s!"dW1 = {Spec.pretty dW1}"
  IO.println s!"db1 = {Spec.pretty db1}"
  IO.println s!"dW2 = {Spec.pretty dW2}"
  IO.println s!"db2 = {Spec.pretty db2}"
  IO.println s!"dX  = {Spec.pretty dX}"

  pure (tlist! y, dW1, db1, dW2, db2, dX)

def maxAbsDiffTensor {s : Spec.Shape} (a b : Spec.Tensor Float s) : Float :=
  let diffs :=
    (Spec.toList a).zip (Spec.toList b) |>.map (fun (x, y) => Float.abs (x - y))
  diffs.foldl max 0.0

def unpackOutPack {α : Type} (p : OutPack α) :
    Spec.Tensor α (NN.Tensor.Shape.Vec 1) ×
      Spec.Tensor α (NN.Tensor.Shape.Mat 3 2) ×
      Spec.Tensor α (NN.Tensor.Shape.Vec 3) ×
      Spec.Tensor α (NN.Tensor.Shape.Mat 1 3) ×
      Spec.Tensor α (NN.Tensor.Shape.Vec 1) ×
      Spec.Tensor α (NN.Tensor.Shape.Vec 2) :=
  match p with
  | .cons y (.cons dW1 (.cons db1 (.cons dW2 (.cons db2 (.cons dX .nil))))) =>
      (y, dW1, db1, dW2, db2, dX)

def maxAbsDiffPack (a b : OutPack Float) : Float :=
  let (ay, adW1, adb1, adW2, adb2, adX) := unpackOutPack a
  let (by_, bdW1, bdb1, bdW2, bdb2, bdX) := unpackOutPack b
  max
    (max (maxAbsDiffTensor ay by_) (maxAbsDiffTensor adW1 bdW1))
    (max
      (max (maxAbsDiffTensor adb1 bdb1) (maxAbsDiffTensor adW2 bdW2))
      (max (maxAbsDiffTensor adb2 bdb2) (maxAbsDiffTensor adX bdX)))

def main (_args : List String) : IO Unit := do
  IO.println "== Float32 backend tutorial =="
  IO.println
    "Note: `float32-mode=fp32` is proof-only (noncomputable); use it in theorems, not IO runs."
  TorchLean.Floats.logFloat32Mode .fp32
  TorchLean.Floats.logFloat32Mode .ieee754Exec

  let rFloat ← runOnce (α := Float) "Float (runtime)"
  let r32 ← runOnce (α := TorchLean.Floats.F32 .ieee754Exec) "Float32 (IEEE32Exec)"

  let r32F : OutPack Float :=
    tlist.map (α := TorchLean.Floats.F32 .ieee754Exec) (β := Float)
      (fun {_s} t => Spec.mapTensor TorchLean.Floats.IEEE754.IEEE32Exec.toFloat t)
      r32

  let diff := maxAbsDiffPack rFloat r32F
  IO.println s!"max_abs_diff(Float vs IEEE32Exec) = {diff.toStringFull}"

end NN.Examples.Advanced.Floats.Float32Modes
