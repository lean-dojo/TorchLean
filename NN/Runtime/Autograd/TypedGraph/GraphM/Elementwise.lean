/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TypedGraph.GraphM.Core
public import NN.Spec.Autograd.Ops
public import NN.Spec.Core.TensorReductionShape.ConcatSlice

/-!
# GraphM Elementwise And Scalar Ops

Arithmetic, activations, scalar reductions, and MSE loss builders for typed graphs.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec
open TorchLean TorchLean.Tensor
open Proofs.Autograd.Algebra
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

/-!
JVP vs VJP in this module

Each typed graph node stores both:
- `vjp`: reverse-mode vector-Jacobian product (used by backprop), and
- `jvp`: forward-mode Jacobian-vector product (directional derivative).

Every operation exposed by the typed graph builder must provide its actual JVP and VJP alongside
its forward map. Shape operations apply the same transformation to the tangent, while nonlinear
operations use named spec-layer derivative formulas. A zero JVP is reserved for constants and
explicit stop-gradient boundaries; it must not stand in for an unimplemented derivative.
-/

/-- Record a unary operation whose JVP uses the same program as its VJP.

The input lookup, saved context, and singleton contribution are shared; this constructor is only
used for operations whose two derivative programs already agree. -/
def Internal.unary {α Δ : Type} [TorchLean.Storage α] [Zero α]
    {Γ : List Shape} {s : Shape} (x : Var s) (op : Spec.OpSpec α s s) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, graph, _⟩ ← get
  let ix ← liftM (mkIdx (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => lookup.read ix)
      (forward := fun input _ => op.forward input)
      (jvp := fun input tangent _ => op.backward input tangent)
      (vjp := fun input _ seed =>
        Contributions.single ix (op.backward input seed))
  push graph node

/--
Elementwise addition node (`y = a + b`).

PyTorch comparison: `torch.add(a, b)`.
-/
def add {α : Type} {Δ : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {s : Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ia ← liftM (mkIdx (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ia, lookup.read ib))
      (forward := fun ctx _d => addSpec (ctx.1) (ctx.2))
      (jvp := fun _ctx dctx _d =>
        addSpec (dctx.1) (dctx.2))
      (vjp := fun _ctx _d δ =>
        Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ia δ)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ib δ))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise subtraction node (`y = a - b`).

PyTorch comparison: `torch.sub(a, b)`.
-/
def sub {α : Type} {Δ : Type} [TorchLean.Storage α] [Sub α] [Add α] [Zero α]
    {Γ : List Shape} {s : Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ia ← liftM (mkIdx (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ia, lookup.read ib))
      (forward := fun ctx _d => subSpec (ctx.1) (ctx.2))
      (jvp := fun _ctx dctx _d =>
        subSpec (dctx.1) (dctx.2))
      (vjp := fun _ctx _d δ =>
        let negδ : Tensor α s := subSpec (Tensor.full s (0 : α)) δ
        Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ia δ)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ib negδ))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise multiplication node (`y = a ⊙ b`).

PyTorch comparison: `torch.mul(a, b)`.
-/
def mul {α : Type} {Δ : Type} [TorchLean.Storage α] [Mul α] [Add α] [Zero α]
    {Γ : List Shape} {s : Shape}
    (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ia ← liftM (mkIdx (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ia, lookup.read ib))
      (forward := fun ctx _d => mulSpec (ctx.1) (ctx.2))
      (jvp := fun ctx dctx _d =>
        let av := ctx.1
        let bv := ctx.2
        let da := dctx.1
        let db := dctx.2
        addSpec (mulSpec da bv) (mulSpec av db))
      (vjp := fun ctx _d δ =>
        let av := ctx.1
        let bv := ctx.2
        Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ia (mulSpec δ bv))
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ib (mulSpec δ av)))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Square `x ↦ x ⊙ x`. -/
def square {α : Type} {Δ : Type} [TorchLean.Storage α] [Mul α] [Add α] [Zero α]
    {Γ : List Shape} {s : Shape}
    (x : Var s) : MWith α Δ Γ (Var s) :=
  mul (α := α) (Δ := Δ) (Γ := Γ) (s := s) x x

/--
Scale a tensor by a scalar constant `c` (`y = c * x`).

PyTorch comparison: `c * x` / `torch.mul(x, c)`.
-/
def scale {α : Type} {Δ : Type} [TorchLean.Storage α] [Mul α] [Zero α]
    {Γ : List Shape} {s : Shape}
    (x : Var s) (c : α) : MWith α Δ Γ (Var s) :=
  Internal.unary x
    { forward := fun input => scaleSpec input c
      backward := fun _ seed => scaleSpec seed c }

/--
Elementwise absolute value.

PyTorch comparison: `torch.abs(x)`.
-/
def abs {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.absOp

/--
Elementwise square root.

PyTorch comparison: `torch.sqrt(x)`.
-/
def sqrt {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.sqrtOp

/--
Elementwise clamp to `[minVal, maxVal]`.

PyTorch comparison: `torch.clamp(x, min=minVal, max=maxVal)`.
-/
def clamp {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) (minVal maxVal : α) : MWith α Δ Γ (Var s) :=
  Internal.unary x (Spec.clampOp minVal maxVal)

/--
Elementwise maximum.

At ties we split the gradient equally (`0.5` / `0.5`), matching the tie-handling documented in
the eager tape (`NN.Runtime.Autograd.Engine.Core`).

PyTorch comparison: `torch.maximum(a, b)`.
-/
def max {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ia ← liftM (mkIdx (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ia, lookup.read ib))
      (forward := fun ctx _d =>
        maxSpec (α := α) (s := s) (ctx.1) (ctx.2))
      (jvp := fun ctx dctx _d =>
        let av := ctx.1
        let bv := ctx.2
        let da := dctx.1
        let db := dctx.2
        addSpec ((Spec.maxOp bv).backward av da) ((Spec.maxOp av).backward bv db))
      (vjp := fun ctx _d δ =>
        let av := ctx.1
        let bv := ctx.2
        Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ia
            ((Spec.maxOp bv).backward av δ))
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ib
            ((Spec.maxOp av).backward bv δ)))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise minimum.

At ties we split the gradient equally (`0.5` / `0.5`).

PyTorch comparison: `torch.minimum(a, b)`.
-/
def min {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (a b : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ia ← liftM (mkIdx (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (Γ := Γ) ss b)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ia, lookup.read ib))
      (forward := fun ctx _d =>
        minSpec (α := α) (s := s) (ctx.1) (ctx.2))
      (jvp := fun ctx dctx _d =>
        let av := ctx.1
        let bv := ctx.2
        let da := dctx.1
        let db := dctx.2
        addSpec ((Spec.minOp bv).backward av da) ((Spec.minOp av).backward bv db))
      (vjp := fun ctx _d δ =>
        let av := ctx.1
        let bv := ctx.2
        Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ia
            ((Spec.minOp bv).backward av δ))
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ib
            ((Spec.minOp av).backward bv δ)))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Elementwise ReLU.

PyTorch comparison: `torch.nn.functional.relu(x)`.
-/
def relu {α : Type} [TorchLean.Storage α]
  [Mul α] [Add α] [Zero α] [Max α] [BEq α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.reluOp

/-- Elementwise sigmoid. PyTorch comparison: `torch.sigmoid(x)`. -/
def sigmoid {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.sigmoidOp

/-- Elementwise tanh. PyTorch comparison: `torch.tanh(x)`. -/
def tanh {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.tanhOp

/--
Elementwise tanh-approximate GELU.

This is one graph node with the specification-level derivative, rather than an expansion into
temporary pointwise nodes. The smaller graph is important for large Transformer activations while
retaining the same JVP and VJP meaning.
-/
def gelu {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.geluOp

/--
Softmax along the last axis (recursing over outer dimensions).

PyTorch comparison: `torch.softmax(x, dim=-1)`.

The symmetric Jacobian uses the same JVP and VJP program.
-/
def softmaxLast {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x
    { forward := Activation.Internal.softmaxInnermostSpec
      backward := Activation.Internal.softmaxInnermostBackwardSpec }

/-- Softmax along an explicitly selected tensor dimension. -/
def softmax {α : Type} [TorchLean.Storage α] [Context α]
    {Δ : Type} {Γ : List Shape} {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x (Spec.softmaxOp axis)

/--
Stable log-softmax along the last axis.

This is a primitive in the typed graph, not the composition `log ∘ softmax`, so proof/IR
execution and eager CUDA share the same PyTorch-style numerical contract.
-/
def logSoftmaxLast {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ix ← liftM (mkIdx (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => lookup.read ix)
      (forward := fun ctx _d =>
        Activation.Internal.logSoftmaxInnermostSpec
          (α := α) (s := s) (ctx))
      (jvp := fun ctx dctx _d =>
        let xval := ctx
        let yval := Activation.Internal.logSoftmaxInnermostSpec (α := α) (s := s) xval
        let dx := dctx
        Activation.Internal.logSoftmaxInnermostJvpSpec (α := α) (s := s) yval dx)
      (vjp := fun ctx _d δ =>
        let xval := ctx
        let yval := Activation.Internal.logSoftmaxInnermostSpec (α := α) (s := s) xval
        let dx := Activation.Internal.logSoftmaxInnermostBackwardSpec
          (α := α) (s := s) yval δ
        Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx)
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Stable log-softmax along an explicitly selected tensor dimension. -/
def logSoftmax {α : Type} [TorchLean.Storage α] [Context α]
    {Δ : Type} {Γ : List Shape} {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ix ← liftM (mkIdx (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => lookup.read ix)
      (forward := fun ctx _d =>
        Activation.logSoftmaxSpec (α := α) (s := s) axis
          (ctx))
      (jvp := fun ctx dctx _d =>
        let xval := ctx
        let yval := Activation.logSoftmaxSpec (α := α) (s := s) axis xval
        let dx := dctx
        Activation.logSoftmaxJvpSpec (α := α) (s := s) axis yval dx)
      (vjp := fun ctx _d δ =>
        let xval := ctx
        let yval := Activation.logSoftmaxSpec (α := α) (s := s) axis xval
        let dx := Activation.logSoftmaxBackwardSpec (α := α) (s := s) axis yval δ
        Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ix dx)
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise softplus. PyTorch comparison: `torch.nn.functional.softplus(x)`. -/
def softplus {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.softplusOp

/-- Elementwise exponential. PyTorch comparison: `torch.exp(x)`. -/
def exp {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.expOp

/--
Elementwise sine, with angles in radians.

Its Jacobian is diagonal: both the forward tangent and the reverse cotangent multiply by
`cos(x)`. Using the shared spec keeps these two derivative paths consistent, including when
dual-number scalars differentiate the VJP again for a Hessian-vector product.
-/
def sin {α : Type} [TorchLean.Storage α] [Context α]
    {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.sinOp

/-- Elementwise cosine; its JVP and VJP multiply by `-sin(x)` at the original input. -/
def cos {α : Type} [TorchLean.Storage α] [Context α]
    {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x Spec.cosOp

/-- Elementwise natural logarithm. PyTorch comparison: `torch.log(x)`. -/
def log {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g, _⟩ ← get
  let ix ← liftM (mkIdx (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    NodeData.ofLocalCompact (fun lookup => lookup.read ix)
      (validate := fun ctx _d =>
        let input := ctx
        if Tensor.allSpec (α := α) (s := s) (fun value => decide (value > (0 : α))) input then
          .ok ()
        else
          .error "autograd: log: input contains values <= 0 (or NaN); \
            `safe_log` computes log(softplus(x) + eps) and accepts every input")
      (forward := fun ctx _d =>
        let xval := ctx
        -- This typed graph closure is pure, so it cannot return the eager engine's `Except`
        -- error. A bad raw-log domain reaches a runtime panic. `safe_log` uses
        -- `log(softplus(x) + eps)` without this domain check; finiteness depends on the backend.
        if Tensor.allSpec (α := α) (s := s) (fun v => decide (v > (0 : α))) xval then
          logSpec (α := α) (s := s) xval
        else
          panic! "GraphM: log: input contains values <= 0 (or NaN); \
            `safe_log` computes log(softplus(x) + eps) and accepts every input")
      (jvp := fun ctx dctx _d =>
        let xval := ctx
        let dx := dctx
        mulSpec (invSpec (α := α) xval) dx)
      (vjp := fun ctx _d δ =>
        let xval := ctx
        Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ix
          (mulSpec (invSpec (α := α) xval) δ))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Elementwise reciprocal `x ↦ 1/x`. PyTorch comparison: `torch.reciprocal(x)`.

The JVP and VJP square the reciprocal first, multiply by the seed, then scale by `-1`.
This fixes the evaluation order for rounded scalars. -/
def inv {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var s) :=
  Internal.unary x
    { forward := invSpec
      backward := fun input seed =>
        let inverse := invSpec input
        let squared := mulSpec inverse inverse
        scaleSpec (mulSpec seed squared) (-1 : α) }

/--
Apply `log(softplus(x) + ε)` elementwise.

Softplus maps each real input to a positive value. A positive `ε` also keeps the logarithm's
argument positive when softplus rounds to zero. The JVP and VJP use
`sigmoid(x) / (softplus(x) + ε)`, with `ε` held fixed.
-/
def safeLog {α : Type} [TorchLean.Storage α] [Context α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) (ε : α := Context.defaultEpsilon) :
    MWith α Δ Γ (Var s) :=
  Internal.unary x (Spec.safeLogOp ε)

/--
Reduce-sum over all entries, producing a scalar.

PyTorch comparison: `torch.sum(x)`.
-/
def sum {α : Type} [TorchLean.Storage α] [Add α] [Zero α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (x : Var s) : MWith α Δ Γ (Var Shape.scalar) := do
  let ⟨ss, g, _⟩ ← get
  let ix ← liftM (mkIdx (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
    NodeData.ofLocalCompact (fun lookup => lookup.read ix)
      (forward := fun ctx _d => Tensor.scalar (sumSpec (α := α) (s := s) (ctx)))
      (jvp := fun _ctx dctx _d =>
        Tensor.scalar (sumSpec (α := α) (s := s) (dctx)))
      (vjp := fun _ctx _d dLdy =>
        Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) ix
          (replicate (α := α) (shape := s) dLdy))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
Mean-squared error loss with `"mean"` reduction, producing a scalar.

PyTorch comparison: `torch.nn.functional.mse_loss(yhat, target, reduction=\"mean\")`.
-/
def mseLoss {α : Type} [TorchLean.Storage α]
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α]
  {Δ : Type} {Γ : List Shape} {s : Shape} (yhat target : Var s) : MWith α Δ Γ (Var Shape.scalar) :=
    do
  let ⟨ss, g, _⟩ ← get
  let iyhat ← liftM (mkIdx (Γ := Γ) ss yhat)
  let itarget ← liftM (mkIdx (Γ := Γ) ss target)
  let node : NodeData α Δ (Γ ++ ss) Shape.scalar :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read iyhat, lookup.read itarget))
      (forward := fun ctx _d =>
        let yhatv := ctx.1
        let targetv := ctx.2
        let diff := subSpec yhatv targetv
        let squared := mulSpec diff diff
        let total := sumSpec (α := α) (s := s) squared
        let denom : Nat := if Spec.Shape.size s = 0 then 1 else Spec.Shape.size s
        Tensor.scalar (total / (denom : α)))
      (jvp := fun ctx dctx _d =>
        let yhatv := ctx.1
        let targetv := ctx.2
        let dyhat := dctx.1
        let dtarget := dctx.2
        let diff := subSpec yhatv targetv
        let two : α := (1 : α) + 1
        let denom : Nat := if Spec.Shape.size s = 0 then 1 else Spec.Shape.size s
        let baseGrad : Tensor α s := scaleSpec (α := α) (s := s) diff (two / (denom : α))
        let ddiff := subSpec dyhat dtarget
        Tensor.scalar (sumSpec (α := α) (s := s) (mulSpec baseGrad ddiff)))
      (vjp := fun ctx _d dLdy =>
        let yhatv := ctx.1
        let targetv := ctx.2
        let diff := subSpec yhatv targetv
        let two : α := (1 : α) + 1
        let denom : Nat := if Spec.Shape.size s = 0 then 1 else Spec.Shape.size s
        let baseGrad : Tensor α s := scaleSpec (α := α) (s := s) diff (two / (denom : α))
        let gscalar : α := Tensor.item dLdy
        let dYhat : Tensor α s := scaleSpec (α := α) (s := s) baseGrad gscalar
        let dTarget : Tensor α s := subSpec (Tensor.full s (0 : α)) dYhat
        Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) iyhat dYhat)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := s) itarget dTarget))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := Shape.scalar) g node

/--
Affine layer `y = W x + b` in the typed graph.

PyTorch comparison: `torch.nn.functional.linear` / `torch.nn.Linear`.

The two terms of the JVP are evaluated as `(dW*x + db) + (W*dx + 0)`.
-/
def linear {α : Type} {Δ : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {Γ : List Shape} {inDim outDim : Nat}
    (w : Var (.dim outDim (.dim inDim .scalar)))
    (b : Var (.dim outDim .scalar))
    (x : Var (.dim inDim .scalar)) : MWith α Δ Γ (Var (.dim outDim .scalar)) := do
  let ⟨ss, g, _⟩ ← get
  let iW ← liftM (mkIdx (Γ := Γ) ss w)
  let ib ← liftM (mkIdx (Γ := Γ) ss b)
  let ix ← liftM (mkIdx (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) (.dim outDim .scalar) :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read iW, lookup.read ib, lookup.read ix))
      (forward := fun ctx _d =>
        let W := ctx.1
        let bv := ctx.2.1
        let xv := ctx.2.2
        let layer : Spec.LinearSpec α inDim outDim := { weights := W, bias := bv }
        Spec.linearSpec (α := α) layer xv)
      (jvp := fun ctx dctx _d =>
        let W := ctx.1
        let xv := ctx.2.2
        let dW := dctx.1
        let db := dctx.2.1
        let dx := dctx.2.2
        let dLayer : Spec.LinearSpec α inDim outDim := { weights := dW, bias := db }
        let xLayer : Spec.LinearSpec α inDim outDim :=
          { weights := W, bias := Tensor.full (.dim outDim .scalar) (0 : α) }
        addSpec (Spec.linearSpec (α := α) dLayer xv) (Spec.linearSpec (α := α) xLayer dx))
      (vjp := fun ctx _d dLdy =>
        let W := ctx.1
        let xv := ctx.2.2
        let dW := Spec.linearWeightsDerivSpec (α := α) (inDim := inDim) (outDim := outDim) xv
          dLdy
        let dx := Spec.linearInputDerivSpec (α := α) (inDim := inDim) (outDim := outDim) W dLdy
        let z0 := Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss)
            (s := .dim outDim (.dim inDim .scalar)) iW dW)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := .dim outDim .scalar) ib dLdy)
        Contributions.add (α := α) (shapes := Γ ++ ss) z0
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := .dim inDim .scalar) ix dx))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim outDim .scalar)) g node

/--
Matrix multiplication with broadcasted batch prefixes.

The operands have shapes `batchA ++ [m, n]` and `batchB ++ [n, p]`. Both batch prefixes
broadcast to `batch`, and the result has shape `batch ++ [m, p]`. The JVP is the bilinear
product rule `d(A @ B) = dA @ B + A @ dB`.

PyTorch comparison: `torch.matmul` for operands of rank at least two.
-/
def matmul {α : Type} {Δ : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {Γ : List Shape} {batchA batchB batch : Shape} {m n p : Nat}
    [broadcastA : Shape.BroadcastTo batchA batch]
    [broadcastB : Shape.BroadcastTo batchB batch]
    (a : Var (batchA.concat [m, n])) (b : Var (batchB.concat [n, p])) :
    MWith α Δ Γ (Var (batch.concat [m, p])) := do
  let ⟨ss, g, _⟩ ← get
  let ia ← liftM (mkIdx (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (Γ := Γ) ss b)
  let aShape := batchA.concat [m, n]
  let bShape := batchB.concat [n, p]
  let outShape := batch.concat [m, p]
  let node : NodeData α Δ (Γ ++ ss) outShape :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ia, lookup.read ib))
      (forward := fun ctx _d =>
        let av := ctx.1
        let bv := ctx.2
        TorchLean.Tensor.matmulSpec broadcastA.proof broadcastB.proof av bv)
      (jvp := fun ctx dctx _d =>
        let av := ctx.1
        let bv := ctx.2
        let da := dctx.1
        let db := dctx.2
        addSpec
          (TorchLean.Tensor.matmulSpec broadcastA.proof broadcastB.proof da bv)
          (TorchLean.Tensor.matmulSpec broadcastA.proof broadcastB.proof av db))
      (vjp := fun ctx _d dLdy =>
        let av := ctx.1
        let bv := ctx.2
        let (dA, dB) :=
          TorchLean.Tensor.matmulBackwardSpec broadcastA.proof broadcastB.proof av bv dLdy
        Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := aShape) ia dA)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := bShape) ib dB))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outShape) g node

/--
Concatenate along the leading dimension (`dim=0`) for tensors of shape `.dim n s`.

PyTorch comparison: `torch.cat([a, b], dim=0)`.
-/
def concat {α : Type} {Δ : Type} [TorchLean.Storage α] [Add α] [Zero α]
    {Γ : List Shape} {n m : Nat} {s : Shape}
    (a : Var (.dim n s)) (b : Var (.dim m s)) :
    MWith α Δ Γ (Var (.dim (n + m) s)) := do
  let ⟨ss, g, _⟩ ← get
  let ia ← liftM (mkIdx (Γ := Γ) ss a)
  let ib ← liftM (mkIdx (Γ := Γ) ss b)
  let outS : Shape := .dim (n + m) s
  let aS : Shape := .dim n s
  let bS : Shape := .dim m s
  let node : NodeData α Δ (Γ ++ ss) outS :=
    NodeData.ofLocalCompact (fun lookup => (lookup.read ia, lookup.read ib))
      (forward := fun ctx _d =>
        let av := ctx.1
        let bv := ctx.2
        TorchLean.Tensor.concatAxisSpec .scalar (α := α) (n := n) (m := m) (suffix := s) av bv)
      (jvp := fun _ctx dctx _d =>
        let da := dctx.1
        let db := dctx.2
        TorchLean.Tensor.concatAxisSpec .scalar (α := α) (n := n) (m := m) (suffix := s) da db)
      (vjp := fun _ctx _d dLdy =>
        let dA := Spec.sliceRangeSpec (α := α) (n := n + m) (shape := s) dLdy 0 n
          (by simp)
        let dB := Spec.sliceRangeSpec (α := α) (n := n + m) (shape := s) dLdy n m
          (by simp)
        Contributions.add (α := α) (shapes := Γ ++ ss)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := aS) ia dA)
          (Contributions.single (α := α) (Γ := Γ ++ ss) (s := bS) ib dB))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Slice a contiguous range along `dim=0`.

PyTorch comparison: `x[start : start+len]` for tensors where the leading dimension is indexed.
-/
def slice {α : Type} {Δ : Type} [TorchLean.Storage α] [Zero α]
    {Γ : List Shape} {n : Nat} {s : Shape}
    (x : Var (.dim n s)) (start len : Nat) (h : start + len ≤ n) :
    MWith α Δ Γ (Var (.dim len s)) := do
  let ⟨ss, g, _⟩ ← get
  let ix ← liftM (mkIdx (Γ := Γ) ss x)
  let outS : Shape := .dim len s
  let inS : Shape := .dim n s
  let node : NodeData α Δ (Γ ++ ss) outS :=
    NodeData.ofLocalCompact (fun lookup => lookup.read ix)
      (forward := fun ctx _d =>
        Spec.sliceRangeSpec (α := α) (n := n) (shape := s)
          (ctx) start len h)
      (jvp := fun _ctx dctx _d =>
        let dx := dctx
        Spec.sliceRangeSpec (α := α) (n := n) (shape := s) dx start len h)
      (vjp := fun _ctx _d δ =>
        Contributions.single (α := α) (Γ := Γ ++ ss) (s := inS) ix
          (TorchLean.Tensor.sliceAxisRangeBackwardSpec (α := α) (s := .dim n s) 0 start len h δ))
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

end GraphM
end TypedGraph
end Autograd
end Runtime
