/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
public import NN.Tensor.API
import Mathlib.Algebra.Order.Algebra

/-!
# Torch Utils

Small helpers for writing PyTorch-style training loops on top of `Runtime.Autograd.Torch`.

This file focuses on demo/training ergonomics:
- extract scalar values,
- build small `TList`s,
- run simple SGD loops for `Torch.ScalarTrainer`.

Stateful optimizer loops live in `Runtime.Autograd.TorchLean`, because those depend on
`TorchLean.Optim`. Keeping that dependency out of this low-level utility module prevents the
session/ref layer from depending upward on the model/optimizer facade.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/-! ## Initialization helpers (Float constants) -/

namespace Init

/--
Deterministic initialization schemes for small demos (stored as `Float` constants).

PyTorch comparison:
- `.zeros` / `.ones` correspond to `torch.nn.init.zeros_` / `torch.nn.init.ones_`
- `.uniform lo hi` corresponds to `torch.nn.init.uniform_` (with explicit `a=lo`, `b=hi`)
- `.xavierUniform fanIn fanOut` corresponds to `torch.nn.init.xavier_uniform_`
- `.kaimingUniform fanIn` corresponds to `torch.nn.init.kaiming_uniform_` with `nonlinearity="relu"`

References:
- https://pytorch.org/docs/stable/nn.init.html
- https://pytorch.org/docs/stable/generated/torch.nn.init.xavier_uniform_.html
- https://pytorch.org/docs/stable/generated/torch.nn.init.kaiming_uniform_.html
-/
inductive Scheme where
  | zeros
  | ones
  | uniform (lo hi : Float)
  | xavierUniform (fanIn fanOut : Nat)
  | kaimingUniform (fanIn : Nat)
  deriving Repr

namespace Internal

/--
Next state of a simple 32-bit linear congruential generator (LCG).

This exists to make demos deterministic and reproducible; it is not statistically strong and it is
not cryptographically secure.
-/
def lcgNext (s : Nat) : Nat :=
  -- classic LCG parameters (same as many stdlib examples)
  let a := 1664525
  let c := 1013904223
  let m := 4294967296  -- 2^32
  (a * s + c) % m

/-- Interpret the 32-bit LCG state as a `Float` in `[0, 1)`. -/
def lcg01 (s : Nat) : Float :=
  let m : Float := 4294967296.0
  (Float.ofNat s) / m

/--
Deterministic `U[0,1)` sampler derived from a seed and scalar index.

Implementation note: this is the LCG iterated `idx+1` times, then normalized to `[0,1)`.
-/
def rand01 (seed idx : Nat) : Float :=
  -- cheap, deterministic “random”: apply the LCG `idx+1` times
  let rec go : Nat → Nat → Nat
    | 0, s => lcgNext s
    | Nat.succ n, s => go n (lcgNext s)
  lcg01 (go idx seed)

/--
Sample the `idx`-th scalar of a tensor initialized using `Scheme`.

This is the scalar-level primitive used by `Init.tensor`.
-/
def sampleAt (sch : Scheme) (seed idx : Nat) : Float :=
  match sch with
  | .zeros => 0.0
  | .ones => 1.0
  | .uniform lo hi =>
      lo + rand01 seed idx * (hi - lo)
  | .xavierUniform fanIn fanOut =>
      let denom := (Float.ofNat fanIn) + (Float.ofNat fanOut)
      let limit := Float.sqrt (6.0 / denom)
      (-limit) + rand01 seed idx * (2.0 * limit)
  | .kaimingUniform fanIn =>
      let limit := Float.sqrt (6.0 / (Float.ofNat fanIn))
      (-limit) + rand01 seed idx * (2.0 * limit)

end Internal

open Internal

/--
Create a `Tensor Float s` by sampling a `Scheme` deterministically.

This is intended for **small demo models**. It is not optimized.

PyTorch comparison: this mimics using `torch.nn.init.*` routines on freshly allocated parameters,
but here we work with *pure* `Tensor Float s` values (no mutation) and use a deterministic
seeded sampler for reproducibility.
-/
def tensor (sch : Scheme) (seed : Nat := 0) : {s : Shape} → Tensor Float s
  | .scalar =>
      Tensor.scalar (sampleAt sch seed 0)
  | .dim _n s' =>
      let chunk := Shape.size s'
      Tensor.dim (fun i =>
        -- offset-by-block so different blocks get different samples
        let seed' := seed
        let idxBase := i.val * chunk
        -- build a sub-tensor whose scalars use indices `idxBase + k`
        let rec build : {t : Shape} → Nat → Tensor Float t
          | .scalar, k => Tensor.scalar (sampleAt sch seed' (idxBase + k))
          | .dim _m t', k =>
              let chunk' := Shape.size t'
              Tensor.dim (fun j => build (t := t') (k + j.val * chunk'))
        build (t := s') 0)

/--
Xavier/Glorot-uniform initializer for 2D weight matrices.

PyTorch comparison: `torch.nn.init.xavier_uniform_` with `gain=1`.
-/
def xavierW (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor Float (.dim outDim (.dim inDim .scalar)) :=
  tensor (s := .dim outDim (.dim inDim .scalar)) (sch := .xavierUniform inDim outDim) (seed := seed)

/--
Kaiming/He-uniform initializer for 2D weight matrices.

PyTorch comparison: `torch.nn.init.kaiming_uniform_` with `nonlinearity="relu"` and default
parameters (so the bound is `sqrt(6/fan_in)`).
-/
def kaimingW (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor Float (.dim outDim (.dim inDim .scalar)) :=
  tensor (s := .dim outDim (.dim inDim .scalar)) (sch := .kaimingUniform inDim) (seed := seed)

end Init

/-! ## Small Sample Generators (Float Constants) -/

namespace Samples

/-- Turn a point `(x1,x2)` into a `Tensor Float (.dim 2 .scalar)`. -/
def vec2 (x1 x2 : Float) : Tensor Float (.dim 2 .scalar) :=
  Tensor.dim (fun i =>
    Tensor.scalar <|
      match i.val with
      | 0 => x1
      | 1 => x2
      | _ => 0.0)

/-- Turn a scalar `y` into a `Tensor Float (.dim 1 .scalar)`. -/
def vec1 (y : Float) : Tensor Float (.dim 1 .scalar) :=
  Tensor.dim (fun _ => Tensor.scalar y)

/-- Affine map `y = w1*x1 + w2*x2 + b` for building small regression datasets. -/
def affine2 (w1 w2 b : Float) (x1 x2 : Float) : Float :=
  w1 * x1 + w2 * x2 + b

end Samples

/-! ## Small conveniences for scalar training demos -/

/--
Extract the scalar value from a scalar-shaped tensor.

PyTorch comparison: like `t.item()` for a 0-dim tensor.
-/
abbrev scalarOf {α : Type} (t : Tensor α Shape.scalar) : α :=
  t.item

/-- Build a one-element `TList` (useful for curried trainer APIs). -/
def tlist1 {α : Type} {s₁ : Shape} (x₁ : Tensor α s₁) : TList α [s₁] :=
  .cons x₁ .nil

/-! ## `TList` syntax sugar -/

/--
Build a `TList` from a comma-separated list of terms.

This is meant for small demo/training code, where `tlist1`/`tlist2`/… becomes tedious.

Example:

```lean
def xs : TList Float [Shape.Vec 2, Shape.Vec 1] :=
  tlist![x, y]
```
-/
syntax (name := tlistBang) "tlist!" "[" term,* "]" : term

macro_rules
  | `(tlist![ $xs:term,* ]) => do
      let xs := xs.getElems.toList
      let rec go : List (Lean.TSyntax `term) → Lean.MacroM (Lean.TSyntax `term)
        | [] => `(Proofs.Autograd.Algebra.TList.nil)
        | x :: xs => do
            let tail ← go xs
            `(Proofs.Autograd.Algebra.TList.cons $x $tail)
      go xs

/-- Build a two-element `TList` (useful for curried trainer APIs). -/
def tlist2 {α : Type} {s₁ s₂ : Shape} (x₁ : Tensor α s₁) (x₂ : Tensor α s₂) : TList α [s₁, s₂] :=
  .cons x₁ (.cons x₂ .nil)

/-- Build a three-element `TList` (useful for curried trainer APIs). -/
def tlist3 {α : Type} {s₁ s₂ s₃ : Shape}
    (x₁ : Tensor α s₁) (x₂ : Tensor α s₂) (x₃ : Tensor α s₃) : TList α [s₁, s₂, s₃] :=
  .cons x₁ (.cons x₂ (.cons x₃ .nil))

/-- Build a four-element `TList` (useful for curried trainer APIs). -/
def tlist4 {α : Type} {s₁ s₂ s₃ s₄ : Shape}
    (x₁ : Tensor α s₁) (x₂ : Tensor α s₂) (x₃ : Tensor α s₃) (x₄ : Tensor α s₄) : TList α [s₁, s₂,
      s₃, s₄] :=
  .cons x₁ (.cons x₂ (.cons x₃ (.cons x₄ .nil)))

namespace ScalarTrainer

/--
Uncurried forward pass for `ScalarTrainer`.

`ScalarTrainer.forward` is stored as a curried function over the input shapes; this helper lets you
pass a `TList` (like a tuple of tensors).
-/
def forwardT {α : Type} {paramShapes inputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes) (xs : TList α inputShapes) :
    IO (Tensor α Shape.scalar) :=
  Curried.uncurry (α := α) (ss := inputShapes) (β := IO (Tensor α Shape.scalar)) tr.forward xs

/--
Uncurried backward pass for `ScalarTrainer`.

Returns per-parameter gradients (aligned with `paramShapes`).
-/
def backwardT {α : Type} {paramShapes inputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes) (xs : TList α inputShapes) :
    IO (TList α paramShapes) :=
  Curried.uncurry (α := α) (ss := inputShapes) (β := IO (TList α paramShapes)) tr.backward xs

/--
Uncurried SGD step for `ScalarTrainer`.

PyTorch comparison: analogous to `loss.backward(); optimizer.step()` for a fixed SGD optimizer,
except here the trainer bundles the update rule.
-/
def stepT {α : Type} {paramShapes inputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes) (lr : α) (xs : TList α inputShapes) : IO Unit :=
  Curried.uncurry (α := α) (ss := inputShapes) (β := IO Unit) (tr.step lr) xs

end ScalarTrainer

/--
Train `steps` SGD updates, cycling through `samples`.

PyTorch comparison: this matches the common eager training skeleton:

```
for step in range(steps):
  batch = dataset[step % len(dataset)]
  loss = forward(batch)
  step(lr, batch)   # typically: loss.backward(); optimizer.step()
```

Note: `ScalarTrainer.step` is the "bundled SGD optimizer" for the trainer. Stateful optimizers
(Adam, RMSProp, ...) are exposed from `Runtime.Autograd.TorchLean`.
-/
def trainCycleSGD
    {α : Type} [ToString α] {paramShapes inputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes)
    (lr : α) (steps : Nat) (samples : List (TList α inputShapes)) (logEvery : Nat := 1) : IO Unit :=
      do
  match samples with
  | [] =>
      throw <| IO.userError "trainCycleSGD: empty dataset"
  | hd :: _tl =>
      for step in [0:steps] do
        let xs := samples.getD (step % samples.length) hd
        let lossT ← ScalarTrainer.forwardT (α := α) (paramShapes := paramShapes) (inputShapes :=
          inputShapes) tr xs
        if logEvery != 0 && step % logEvery = 0 then
          IO.println s!"step {step}: loss={scalarOf lossT}"
        ScalarTrainer.stepT (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) tr lr
          xs

/--
Evaluate mean loss over a dataset.

PyTorch comparison: like running a model in `torch.no_grad()` over a dataloader and averaging
the scalar loss values, except here we call `ScalarTrainer.forwardT` directly.
-/
def meanLoss
    {α : Type} [ToString α] [Add α] [Div α] [Zero α] [Coe Nat α]
    {paramShapes inputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes)
    (samples : List (TList α inputShapes)) : IO α := do
  match samples with
  | [] =>
      throw <| IO.userError "meanLoss: empty dataset"
  | _ =>
      let mut acc : α := 0
      for xs in samples do
        let lossT ← ScalarTrainer.forwardT (α := α) (paramShapes := paramShapes) (inputShapes :=
          inputShapes) tr xs
        acc := acc + scalarOf lossT
      pure (acc / (samples.length : α))

end Torch
end Autograd
end Runtime
