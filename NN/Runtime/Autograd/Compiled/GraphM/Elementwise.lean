/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.GraphM.Core

/-!
# GraphM Elementwise And Scalar Ops

Arithmetic, activations, scalar reductions, and MSE loss builders for proof-compiled graphs.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Compiled
namespace GraphM

open Spec
open Tensor
open Proofs.Autograd.Algebra
open Runtime.Autograd.TorchLean

/-!
JVP vs VJP in this module

Each compiled node stores both:
- `vjp`: reverse-mode vector-Jacobian product (used by backprop), and
- `jvp`: forward-mode Jacobian-vector product (directional derivative).

The `.compiled` runtime path is primarily exercised via reverse-mode (VJP) and compilation to the
eager tape. Basic elementwise/bilinear ops provide real JVP rules, shape-structural ops (for
example slice/concat) apply the same transformation to the tangent, and heavier ops should expose
named spec-layer JVP helpers before being wired here. Reverse-only ops
it must be listed in `reverseOnlyJvpOps` and call `unsupportedJvp` rather than returning a silent
zero tangent.

Forward-mode coverage is expanded by adding concrete `jvp` rules next to the corresponding
`forward` and `vjp` definitions.
-/

/--
Elementwise addition node (`y = a + b`).

PyTorch comparison: `torch.add(a, b)`.
-/
def add {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s : Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d => addSpec (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
      ctx) ib)
      jvp := fun _ctx dctx _d =>
        addSpec (getIdx (α := α) (xs := dctx) ia) (getIdx (α := α) (xs := dctx) ib)
      vjp := fun _ctx _d δ =>
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia δ)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise subtraction node (`y = a - b`).

PyTorch comparison: `torch.sub(a, b)`.
-/
def sub {α : Type} {Δ : Type} [Sub α] [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s :
  Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d => subSpec (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
      ctx) ib)
      jvp := fun _ctx dctx _d =>
        subSpec (getIdx (α := α) (xs := dctx) ia) (getIdx (α := α) (xs := dctx) ib)
      vjp := fun _ctx _d δ =>
        let negδ : Tensor α s := subSpec (fill (0 : α) s) δ
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia δ)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib negδ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise multiplication node (`y = a ⊙ b`).

PyTorch comparison: `torch.mul(a, b)`.
-/
def mul {α : Type} {Δ : Type} [Mul α] [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s :
  Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d => mulSpec (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
      ctx) ib)
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        addSpec (mulSpec da bv) (mulSpec av db)
      vjp := fun ctx _d δ =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia (mulSpec δ bv))
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib (mulSpec δ av)) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Square `x ↦ x ⊙ x`. -/
def square {α : Type} {Δ : Type} [Mul α] [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s :
  Shape}
    (x : Var s) : MWith α Δ Γ (Var s) :=
  mul (α := α) (Δ := Δ) (Γ := Γ) (s := s) x x

/--
Scale a tensor by a scalar constant `c` (`y = c * x`).

PyTorch comparison: `c * x` / `torch.mul(x, c)`.
-/
def scale {α : Type} {Δ : Type} [Mul α] [Add α] [Zero α] [DecidableEq Shape] {Γ : List Shape} {s :
  Shape}
    (x : Var s) (c : α) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        scaleSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix) c
      jvp := fun _ctx dctx _d =>
        scaleSpec (α := α) (s := s) (getIdx (α := α) (xs := dctx) ix) c
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (scaleSpec (α := α) (s := s) δ c) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise absolute value.

PyTorch comparison: `torch.abs(x)`.
-/
def abs {α : Type} [Context α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        absSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dabs := signSpec (α := α) (s := s) xval
        mulSpec dabs dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dabs := signSpec (α := α) (s := s) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dabs δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise square root.

PyTorch comparison: `torch.sqrt(x)`.
-/
def sqrt {α : Type} [Context α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        sqrtSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dsqrt : Tensor α s :=
          mapSpec (α := α) (s := s) (fun v =>
            if v > 0 then
              (1 : α) / (((2 : Nat) : α) * MathFunctions.sqrt v)
            else
              (0 : α)) xval
        mulSpec dsqrt dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dsqrt : Tensor α s :=
          mapSpec (α := α) (s := s) (fun v =>
            if v > 0 then
              (1 : α) / (((2 : Nat) : α) * MathFunctions.sqrt v)
            else
              (0 : α)) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dsqrt δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise clamp to `[minVal, maxVal]`.

PyTorch comparison: `torch.clamp(x, min=minVal, max=maxVal)`.
-/
def clamp {α : Type} [Context α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) (minVal maxVal : α) : MWith α Δ Γ (Var s) :=
    do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        clampSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix) minVal maxVal
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dclamp : Tensor α s :=
          mapSpec (α := α) (s := s) (fun v =>
            if v > minVal ∧ maxVal > v then (1 : α) else (0 : α)) xval
        mulSpec dclamp dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dclamp : Tensor α s :=
          mapSpec (α := α) (s := s) (fun v =>
            if v > minVal ∧ maxVal > v then (1 : α) else (0 : α)) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dclamp δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise maximum.

At ties we split the gradient equally (`0.5` / `0.5`), matching the tie-handling documented in
the eager tape (`NN.Runtime.Autograd.Engine.Core`).

PyTorch comparison: `torch.maximum(a, b)`.
-/
def max {α : Type} [Context α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        maxSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs := ctx) ib)
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) av bv
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) av bv
        addSpec (mulSpec maskA da) (mulSpec maskB db)
      vjp := fun ctx _d δ =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) av bv
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) av bv
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia (mulSpec maskA δ))
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib (mulSpec maskB δ)) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise minimum.

At ties we split the gradient equally (`0.5` / `0.5`).

PyTorch comparison: `torch.minimum(a, b)`.
-/
def min {α : Type} [Context α] [DecidableEq Shape]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        minSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs := ctx) ib)
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) av bv
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) av bv
        addSpec (mulSpec maskA da) (mulSpec maskB db)
      vjp := fun ctx _d δ =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) av bv
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) av bv
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ia (mulSpec maskA δ))
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) ib (mulSpec maskB δ)) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise ReLU.

PyTorch comparison: `torch.nn.functional.relu(x)`.
-/
def relu {α : Type}
  [Mul α] [Add α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.reluSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let drelu := Activation.reluDerivSpec (α := α) xval
        mulSpec drelu dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let drelu := Activation.reluDerivSpec (α := α) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec drelu δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise sigmoid. PyTorch comparison: `torch.sigmoid(x)`. -/
def sigmoid {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.sigmoidSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dsig := Activation.sigmoidDerivSpec (α := α) xval
        mulSpec dsig dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dsig := Activation.sigmoidDerivSpec (α := α) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dsig δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise tanh. PyTorch comparison: `torch.tanh(x)`. -/
def tanh {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.tanhSpec (α := α) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dtanh := Activation.tanhDerivSpec (α := α) xval
        mulSpec dtanh dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dtanh := Activation.tanhDerivSpec (α := α) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dtanh δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Softmax along the last axis (recursing over outer dimensions).

PyTorch comparison: `torch.softmax(x, dim=-1)`.
-/
def softmax {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.softmaxSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        -- Softmax Jacobian is symmetric, so we can reuse the same JVP/VJP implementation.
        Activation.softmaxBackwardSpec (α := α) (s := s) xval dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := Activation.softmaxBackwardSpec (α := α) (s := s) xval δ
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Stable log-softmax along the last axis.

This is a primitive in the compiled graph, not the composition `log ∘ softmax`, so proof/IR
execution and eager CUDA share the same PyTorch-style numerical contract.
-/
def logSoftmax {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.logSoftmaxSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let yval := Activation.logSoftmaxSpec (α := α) (s := s) xval
        let dx := getIdx (α := α) (xs := dctx) ix
        Activation.logSoftmaxBackwardSpec (α := α) (s := s) yval dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let yval := Activation.logSoftmaxSpec (α := α) (s := s) xval
        let dx := Activation.logSoftmaxBackwardSpec (α := α) (s := s) yval δ
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise softplus. PyTorch comparison: `torch.nn.functional.softplus(x)`. -/
def softplus {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        Activation.softplusSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dsoft := Activation.softplusDerivSpec (α := α) (s := s) xval
        mulSpec dsoft dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dsoft := Activation.softplusDerivSpec (α := α) (s := s) xval
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dsoft δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise exponential. PyTorch comparison: `torch.exp(x)`. -/
def exp {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        expSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        mulSpec (expSpec (α := α) xval) dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec (expSpec (α := α) xval) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise natural logarithm. PyTorch comparison: `torch.log(x)`. -/
def log {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        -- Keep runtime behavior consistent with the eager autograd engine:
        -- `log` rejects non-positive inputs; use `safe_log` for epsilon protection.
        if Tensor.allSpec (α := α) (s := s) (fun v => decide (v > (0 : α))) xval then
          logSpec (α := α) (s := s) xval
        else
          panic! "GraphM: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        mulSpec (invSpec (α := α) xval) dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec (invSpec (α := α) xval) δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise reciprocal `x ↦ 1/x`. PyTorch comparison: `torch.reciprocal(x)`. -/
def inv {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        invSpec (α := α) (s := s) (getIdx (α := α) (xs := ctx) ix)
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx0 := getIdx (α := α) (xs := dctx) ix
        let invx := invSpec (α := α) xval
        let invx2 := mulSpec invx invx
        scaleSpec (α := α) (s := s) (mulSpec dx0 invx2) (-1 : α)
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let invx := invSpec (α := α) xval
        let invx2 := mulSpec invx invx
        let dx := scaleSpec (α := α) (s := s) (mulSpec δ invx2) (-1 : α)
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise numerically-stable log (uses an internal `ε`).

PyTorch comparison: commonly written `torch.log(x + eps)`.
-/
def safeLog {α : Type} [Context α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) (ε : α := Numbers.epsilon) : MWith α Δ Γ (Var
    s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        Activation.safeLogSpec (α := α) (s := s) xval ε
      jvp := fun ctx dctx _d =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dx := getIdx (α := α) (xs := dctx) ix
        let dlog := Activation.safeLogDerivSpec (α := α) (s := s) xval ε
        mulSpec dlog dx
      vjp := fun ctx _d δ =>
        let xval := getIdx (α := α) (xs := ctx) ix
        let dlog := Activation.safeLogDerivSpec (α := α) (s := s) xval ε
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (mulSpec dlog δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Reduce-sum over all entries, producing a scalar.

PyTorch comparison: `torch.sum(x)`.
-/
def sum {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var Shape.scalar) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
    { forward := fun ctx _d => Tensor.scalar (sumSpec (α := α) (s := s) (getIdx (α := α) (xs :=
      ctx) ix))
      jvp := fun _ctx dctx _d =>
        Tensor.scalar (sumSpec (α := α) (s := s) (getIdx (α := α) (xs := dctx) ix))
      vjp := fun _ctx _d dLdy =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := s) ix (replicate (α := α) (s := s) dLdy) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
Mean-squared error loss with `"mean"` reduction, producing a scalar.

PyTorch comparison: `torch.nn.functional.mse_loss(yhat, target, reduction=\"mean\")`.
-/
def mseLoss {α : Type}
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {Δ : Type} {Γ : List Shape} {s : Shape} (yhat target : Var s) : MWith α Δ Γ (Var Shape.scalar) :=
    do
  let ⟨ss, g⟩ ← get
  let iyhat ← liftM (mkIdx (_α := α) (Γ := Γ) ss yhat)
  let itarget ← liftM (mkIdx (_α := α) (Γ := Γ) ss target)
  let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
    { forward := fun ctx _d =>
        let yhatv := getIdx (α := α) (xs := ctx) iyhat
        let targetv := getIdx (α := α) (xs := ctx) itarget
        let diff := subSpec yhatv targetv
        let squared := mulSpec diff diff
        let total := sumSpec (α := α) (s := s) squared
        let denom : Nat := if Shape.size s = 0 then 1 else Shape.size s
        Tensor.scalar (total / (denom : α))
      jvp := fun ctx dctx _d =>
        let yhatv := getIdx (α := α) (xs := ctx) iyhat
        let targetv := getIdx (α := α) (xs := ctx) itarget
        let dyhat := getIdx (α := α) (xs := dctx) iyhat
        let dtarget := getIdx (α := α) (xs := dctx) itarget
        let diff := subSpec yhatv targetv
        let two : α := (1 : α) + 1
        let denom : Nat := if Shape.size s = 0 then 1 else Shape.size s
        let baseGrad : Tensor α s := scaleSpec (α := α) (s := s) diff (two / (denom : α))
        let ddiff := subSpec dyhat dtarget
        Tensor.scalar (sumSpec (α := α) (s := s) (mulSpec baseGrad ddiff))
      vjp := fun ctx _d dLdy =>
        let yhatv := getIdx (α := α) (xs := ctx) iyhat
        let targetv := getIdx (α := α) (xs := ctx) itarget
        let diff := subSpec yhatv targetv
        let two : α := (1 : α) + 1
        let denom : Nat := if Shape.size s = 0 then 1 else Shape.size s
        let baseGrad : Tensor α s := scaleSpec (α := α) (s := s) diff (two / (denom : α))
        let gscalar : α := Tensor.toScalar dLdy
        let dYhat : Tensor α s := scaleSpec (α := α) (s := s) baseGrad gscalar
        let dTarget : Tensor α s := subSpec (fill (0 : α) s) dYhat
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) iyhat dYhat)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := s) itarget dTarget) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
  Affine layer `y = W x + b` in the compiled graph.

  PyTorch comparison: `torch.nn.functional.linear` / `torch.nn.Linear`.

  The JVP is the usual product rule:
  `d(Wx+b) = dW*x + W*dx + db`.
  -/
  def linear {α : Type} {Δ : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {inDim outDim : Nat}
    (w : Var (.dim outDim (.dim inDim .scalar)))
    (b : Var (.dim outDim .scalar))
    (x : Var (.dim inDim .scalar)) : MWith α Δ Γ (Var (.dim outDim .scalar)) := do
  let ⟨ss, g⟩ ← get
  let iW ← liftM (mkIdx (_α := α) (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) (.dim outDim .scalar) :=
    { forward := fun ctx _d =>
        let W := getIdx (α := α) (xs := ctx) iW
        let bv := getIdx (α := α) (xs := ctx) ib
        let xv := getIdx (α := α) (xs := ctx) ix
        let layer : Spec.LinearSpec α inDim outDim := { weights := W, bias := bv }
        Spec.linearSpec (α := α) layer xv
      jvp := fun ctx dctx _d =>
        let W := getIdx (α := α) (xs := ctx) iW
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := getIdx (α := α) (xs := dctx) iW
        let db := getIdx (α := α) (xs := dctx) ib
        let dx := getIdx (α := α) (xs := dctx) ix
        let dLayer : Spec.LinearSpec α inDim outDim := { weights := dW, bias := db }
        let xLayer : Spec.LinearSpec α inDim outDim := { weights := W, bias := fill (0 : α) (.dim outDim .scalar) }
        addSpec (Spec.linearSpec (α := α) dLayer xv) (Spec.linearSpec (α := α) xLayer dx)
      vjp := fun ctx _d dLdy =>
        let W := getIdx (α := α) (xs := ctx) iW
        let xv := getIdx (α := α) (xs := ctx) ix
        let dW := Spec.linearWeightsDerivSpec (α := α) (inDim := inDim) (outDim := outDim) xv
          dLdy
        let db := Spec.linearBiasDerivSpec (α := α) (inDim := inDim) (outDim := outDim) dW dLdy
          xv
        let dx := Spec.linearInputDerivSpec (α := α) (inDim := inDim) (outDim := outDim) W dLdy
        let z0 := TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outDim (.dim inDim .scalar)) iW dW)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim outDim .scalar) ib db)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim inDim .scalar) ix dx) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim outDim .scalar)) g node

/--
  Matrix multiplication (`(m×n) @ (n×p) → (m×p)`).

  PyTorch comparison: `torch.matmul`.

  The JVP is the bilinear product rule `d(A @ B) = dA @ B + A @ dB`.
  -/
  def matmul {α : Type} {Δ : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
    {Γ : List Shape} {m n p : Nat}
    (a : Var (.dim m (.dim n .scalar))) (b : Var (.dim n (.dim p .scalar))) :
    MWith α Δ Γ (Var (.dim m (.dim p .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) (.dim m (.dim p .scalar)) :=
    { forward := fun ctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        Spec.matMulSpec av bv
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        addSpec (Spec.matMulSpec da bv) (Spec.matMulSpec av db)
      vjp := fun ctx _d dLdy =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let (dA, dB) := Spec.Tensor.matMulBackwardSpec av bv dLdy
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim m (.dim n .scalar)) ia dA)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n (.dim p .scalar)) ib dB) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim m (.dim p .scalar))) g node

/--
  Batched matrix multiplication (`batch×m×n` with `batch×n×p`).

  PyTorch comparison: `torch.bmm`.

  The JVP is the batched bilinear product rule `d(A @ B) = dA @ B + A @ dB`.
  -/
  def bmm {α : Type} {Δ : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {batch m n p : Nat}
    (a : Var (.dim batch (.dim m (.dim n .scalar))))
    (b : Var (.dim batch (.dim n (.dim p .scalar)))) :
    MWith α Δ Γ (Var (.dim batch (.dim m (.dim p .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let outS : Shape := .dim batch (.dim m (.dim p .scalar))
  let aS : Shape := .dim batch (.dim m (.dim n .scalar))
  let bS : Shape := .dim batch (.dim n (.dim p .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        Spec.Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) av bv
      jvp := fun ctx dctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        addSpec
          (Spec.Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) da bv)
          (Spec.Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) av db)
      vjp := fun ctx _d dLdy =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        let (dA, dB) :=
          Spec.Tensor.bmmBackwardSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) av bv
            dLdy
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := aS) ia dA)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := bS) ib dB) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Concatenate two vectors (dim-0 concat).

  PyTorch comparison: `torch.cat([a, b], dim=0)` for 1D tensors.
  -/
  def concatVectors {α : Type} {Δ : Type} [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
    {Γ : List Shape} {n m : Nat}
    (a : Var (.dim n .scalar)) (b : Var (.dim m .scalar)) :
    MWith α Δ Γ (Var (.dim (n + m) .scalar)) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) (.dim (n + m) .scalar) :=
      { forward := fun ctx _d =>
          let av := getIdx (α := α) (xs := ctx) ia
          let bv := getIdx (α := α) (xs := ctx) ib
          Spec.Tensor.concatVectorsSpec av bv
        jvp := fun _ctx dctx _d =>
          let da := getIdx (α := α) (xs := dctx) ia
          let db := getIdx (α := α) (xs := dctx) ib
          Spec.Tensor.concatVectorsSpec da db
        vjp := fun _ctx _d dLdy =>
          let dA := Spec.Tensor.sliceVectorSpec dLdy 0 n (by simp)
          let dB := Spec.Tensor.sliceVectorSpec dLdy n m (by exact Nat.le_refl _)
          TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n .scalar) ia dA)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim m .scalar) ib dB) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim (n + m) .scalar)) g node

/--
  Concatenate along the leading dimension (`dim=0`) for tensors of shape `.dim n s`.

  PyTorch comparison: `torch.cat([a, b], dim=0)`.
  -/
  def concatDim0 {α : Type} {Δ : Type} [Add α] [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Var (.dim n s)) (b : Var (.dim m s)) :
    MWith α Δ Γ (Var (.dim (n + m) s)) := do
  let ⟨ss, g⟩ ← get
  let ia ← liftM (mkIdx (_α := α) (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss b)
  let outS : Shape := .dim (n + m) s
  let aS : Shape := .dim n s
  let bS : Shape := .dim m s
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        let av := getIdx (α := α) (xs := ctx) ia
        let bv := getIdx (α := α) (xs := ctx) ib
        Spec.Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := s) av bv
      jvp := fun _ctx dctx _d =>
        let da := getIdx (α := α) (xs := dctx) ia
        let db := getIdx (α := α) (xs := dctx) ib
        Spec.Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := s) da db
      vjp := fun _ctx _d dLdy =>
        let dA := Spec.sliceRangeSpec (α := α) (n := n + m) (s := s) dLdy 0 n
          (by simp)
        let dB := Spec.sliceRangeSpec (α := α) (n := n + m) (s := s) dLdy n m
          (by simp [Nat.add_comm])
        TList.add (α := α) (ss := Γ ++ ss)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := aS) ia dA)
          (TList.single (α := α) (Γ := Γ ++ ss) (s := bS) ib dB) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
  Slice a contiguous range along `dim=0`.

  PyTorch comparison: `x[start : start+len]` for tensors where the leading dimension is indexed.
  -/
  def sliceRange0 {α : Type} {Δ : Type} [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {n : Nat} {s : Shape}
    (x : Var (.dim n s)) (start len : Nat) (h : len + start ≤ n) :
    MWith α Δ Γ (Var (.dim len s)) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let outS : Shape := .dim len s
  let inS : Shape := .dim n s
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.sliceRangeSpec (α := α) (n := n) (s := s) (getIdx (α := α) (xs := ctx) ix) start len
          h
      jvp := fun _ctx dctx _d =>
        let dx := getIdx (α := α) (xs := dctx) ix
        Spec.sliceRangeSpec (α := α) (n := n) (s := s) dx start len h
      vjp := fun _ctx _d δ =>
        TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (Spec.Tensor.sliceRange0BackwardSpec (α := α) (n := n) (s := s) start len h δ) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

end GraphM
end Compiled
end Autograd
end Runtime
