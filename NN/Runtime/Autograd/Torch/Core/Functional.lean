/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Compiled

/-!
# Backend-Generic Functional API

The `Ops` interface and curried helper syntax used to write one model once and run it on either the
eager runtime or the compiled graph backend.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/-
Backend-generic "one API" layer

The eager backend builds a runtime tape each iteration.
The `GraphM` authoring API provides a proof-compiled model.

The definitions below let you write a single model/loss once (as a polymorphic program over a
small `Ops` interface) and then choose:
- `backend := .eager`    (build a tape each iteration)
- `backend := .compiled` (compile once, run many)
-/

namespace Proofs.Autograd.Algebra.TList

/--
Append two `TList`s.

This is a small utility for bridging between curried APIs and list-of-shapes APIs.
-/
def append {α : Type} : {ss₁ ss₂ : List Shape} → TList α ss₁ → TList α ss₂ → TList α (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/--
Split a `TList α (ss₁ ++ ss₂)` into its left and right parts.

This is the inverse of `TList.append`.
-/
def splitAppend {α : Type} : {ss₁ ss₂ : List Shape} → TList α (ss₁ ++ ss₂) → TList α ss₁ × TList α
  ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (xs₁, xs₂) := splitAppend (α := α) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xs₁, xs₂)

end Proofs.Autograd.Algebra.TList

namespace Curried

/--
Type of a curried function accepting one tensor argument per shape in `ss`.

For example, `Fn α [s₁, s₂] β` is `Tensor α s₁ → Tensor α s₂ → β`.
-/
def Fn (α : Type) : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Tensor α s → Fn α ss β

/-- Convert a function on `TList` inputs into its curried form. -/
def curry {α : Type} {β : Type} : {ss : List Shape} → (TList α ss → β) → Fn α ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/-- Convert a curried function into a function on `TList` inputs. -/
def uncurry {α : Type} {β : Type} : {ss : List Shape} → Fn α ss β → TList α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

end Curried

/-!
`RefList` is the reference-analogue of `TList`: a heterogeneous list of `Ref s` values indexed by
a shape list.

This is used to write backend-generic code over references (e.g. `TensorRef`s in eager mode, or
`GraphM.Var`s in compiled mode).
-/
/-- Reference-analogue of `TList`: a heterogeneous list of `Ref s` values indexed by shapes. -/
inductive RefList (Ref : Shape → Type) : List Shape → Type where
  | nil : RefList Ref []
  | cons {s : Shape} {ss : List Shape} : Ref s → RefList Ref ss → RefList Ref (s :: ss)

namespace RefList

/-- Append two `RefList`s. -/
def append {Ref : Shape → Type} : {ss₁ ss₂ : List Shape} → RefList Ref ss₁ → RefList Ref ss₂ →
  RefList Ref (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a `RefList Ref (ss₁ ++ ss₂)` into its left and right parts. -/
def split {Ref : Shape → Type} : {ss₁ ss₂ : List Shape} →
    RefList Ref (ss₁ ++ ss₂) → RefList Ref ss₁ × RefList Ref ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (l, r) := split (Ref := Ref) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x l, r)

/-- Split a `RefList Ref (ss ++ [τ])` into its prefix and last element. -/
def splitLast {Ref : Shape → Type} : {ss : List Shape} → {τ : Shape} →
    RefList Ref (ss ++ [τ]) → RefList Ref ss × Ref τ
  | [], _τ, .cons x .nil => (.nil, x)
  | _s :: ss, τ, .cons x xs =>
      let (l, last) := splitLast (Ref := Ref) (ss := ss) (τ := τ) xs
      (.cons x l, last)

end RefList

/--
Type of a curried function over references, one `Ref s` argument per shape in `ss`.

This mirrors `Curried.Fn`, but for `Ref`-valued arguments (e.g. `TensorRef`s in eager mode or
`GraphM.Var`s in compiled mode).
-/
def CurriedRef (Ref : Shape → Type) : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Ref s → CurriedRef Ref ss β

namespace CurriedRef

/-- Uncurry a curried reference function to accept a `RefList`. -/
def uncurry {Ref : Shape → Type} {β : Type} : {ss : List Shape} → CurriedRef Ref ss β → RefList Ref
  ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

/-- Curry a reference function that consumes a `RefList`. -/
def curry {Ref : Shape → Type} {β : Type} : {ss : List Shape} → (RefList Ref ss → β) → CurriedRef
  Ref ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/--
Apply a curried reference function to a `GraphM.VarList`.

This is a convenience for the compiled backend, where leaves/inputs are represented as `Var`s.
-/
def applyVarList {Γ : List Shape} {β : Type} :
    CurriedRef (fun s => Runtime.Autograd.Compiled.GraphM.Var s) Γ β →
      Runtime.Autograd.Compiled.GraphM.VarList Γ → β
  | f, .nil => f
  | f, .cons v vs => applyVarList (Γ := _) (β := β) (f v) vs

end CurriedRef

/--
Backend-generic interface for building and executing tensor programs.

This typeclass lets you write a single model/loss once (polymorphic over `Ops m α`) and then choose:
- an eager backend that executes immediately on a runtime tape, or
- a compiled backend that records proved IR (`GraphM`) for later compilation/proofs.

Each method corresponds to a Tensor op; implementations are expected to match the semantics of the
corresponding `Runtime.Autograd.Tape.*` / `Compiled.GraphM.*` operator.
-/
class Ops (m : Type → Type) (α : Type) [Context α] [DecidableEq Shape] where
  Ref : Shape → Type
  const : {s : Shape} → Tensor α s → m (Ref s)
  add : {s : Shape} → Ref s → Ref s → m (Ref s)
  sub : {s : Shape} → Ref s → Ref s → m (Ref s)
  mul : {s : Shape} → Ref s → Ref s → m (Ref s)
  scale : {s : Shape} → Ref s → α → m (Ref s)
  abs : {s : Shape} → Ref s → m (Ref s)
  sqrt : {s : Shape} → Ref s → m (Ref s)
  clamp : {s : Shape} → Ref s → α → α → m (Ref s)
  max : {s : Shape} → Ref s → Ref s → m (Ref s)
  min : {s : Shape} → Ref s → Ref s → m (Ref s)
  broadcastTo : {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Ref s₁ → m (Ref s₂)
  reshape : {s₁ s₂ : Shape} → Ref s₁ → (h : Shape.size s₁ = Shape.size s₂) → m (Ref s₂)
  transpose2d {mDim nDim : Nat} : Ref (.dim mDim (.dim nDim .scalar)) → m (Ref (.dim nDim (.dim mDim
    .scalar)))
  transpose3dFirstToLast {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim b (.dim c (.dim a .scalar))))
  transpose3dLastToFirst {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim c (.dim a (.dim b .scalar))))
  transpose3dLastTwo {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim a (.dim c (.dim b .scalar))))
  swapAdjacentAtDepth {s : Shape} : (depth : Nat) → Ref s → m (Ref (s.swapAdjacentAtDepth depth))
  reduceSum {s : Shape} (axis : Nat) [Shape.valid_axis_inst axis s] [Shape.WellFormed s] :
    Ref s → m (Ref (shapeAfterSum s axis))
  reduceMean {s : Shape} (axis : Nat) [Shape.valid_axis_inst axis s] [Shape.WellFormed s] :
    Ref s → m (Ref (shapeAfterSum s axis))
  gatherScalar {n : Nat} : Ref (.dim n .scalar) → Fin n → m (Ref Shape.scalar)
  gatherRow {rows cols : Nat} : Ref (.dim rows (.dim cols .scalar)) → Fin rows → m (Ref (.dim cols
    .scalar))
  gatherScalarNat {n : Nat} : Ref (.dim n .scalar) → Nat → m (Ref Shape.scalar)
  gatherVecNat {n k : Nat} : Ref (.dim n .scalar) → Tensor Nat (.dim k .scalar) → m (Ref (.dim k
    .scalar))
  gatherRowsNat {rows cols k : Nat} :
    Ref (.dim rows (.dim cols .scalar)) → Tensor Nat (.dim k .scalar) → m (Ref (.dim k (.dim cols
      .scalar)))
  /--
  Read a float vector of integer-valued token ids and return a `Tensor Nat` index vector.

  This is non-differentiable: gradients do not flow back into the float input.  Language-model
  benchmarks pass token ids as float inputs so each step can supply a fresh window without
  re-instantiating the module.
  -/
  tokenIdsFromFloatVec {k : Nat} : Ref (.dim k .scalar) → m (Tensor Nat (.dim k .scalar))
  scatterAddVec {n : Nat} : Ref (.dim n .scalar) → Ref Shape.scalar → Fin n → m (Ref (.dim n
    .scalar))
  scatterAddRow {rows cols : Nat} :
    Ref (.dim rows (.dim cols .scalar)) → Ref (.dim cols .scalar) → Fin rows → m (Ref (.dim rows
      (.dim cols .scalar)))
  matmul {mDim nDim pDim : Nat} :
    Ref (.dim mDim (.dim nDim .scalar)) →
    Ref (.dim nDim (.dim pDim .scalar)) →
    m (Ref (.dim mDim (.dim pDim .scalar)))
  bmm {batch mDim nDim pDim : Nat} :
    Ref (.dim batch (.dim mDim (.dim nDim .scalar))) →
    Ref (.dim batch (.dim nDim (.dim pDim .scalar))) →
    m (Ref (.dim batch (.dim mDim (.dim pDim .scalar))))
  concatVectors {nDim mDim : Nat} :
    Ref (.dim nDim .scalar) →
    Ref (.dim mDim .scalar) →
    m (Ref (.dim (nDim + mDim) .scalar))
  concatLeadingAxis {nDim mDim : Nat} {s : Shape} :
    Ref (.dim nDim s) →
    Ref (.dim mDim s) →
    m (Ref (.dim (nDim + mDim) s))
  sliceLeadingAxisRange {nDim : Nat} {s : Shape} :
    (start len : Nat) → (h : len + start ≤ nDim) →
    Ref (.dim nDim s) → m (Ref (.dim len s))
  maxPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  avgPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  smoothMaxPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    α →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  maxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))))
  maxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
      (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))))
  smoothMaxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    α →
    m (Ref (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))))
  avgPool2d {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar))))
  avgPool2dPad {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
      (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))))
  relu : {s : Shape} → Ref s → m (Ref s)
  sigmoid : {s : Shape} → Ref s → m (Ref s)
  tanh : {s : Shape} → Ref s → m (Ref s)
  softmax : {s : Shape} → Ref s → m (Ref s)
  logSoftmax : {s : Shape} → Ref s → m (Ref s)
  softplus : {s : Shape} → Ref s → m (Ref s)
  exp : {s : Shape} → Ref s → m (Ref s)
  log : {s : Shape} → Ref s → m (Ref s)
  inv : {s : Shape} → Ref s → m (Ref s)
  detach : {s : Shape} → Ref s → m (Ref s)
  safeLog : {s : Shape} → Ref s → α → m (Ref s)
  sum : {s : Shape} → Ref s → m (Ref Shape.scalar)
  flatten : {s : Shape} → Ref s → m (Ref (.dim (Shape.size s) .scalar))
  linear {inDim outDim : Nat} :
    Ref (.dim outDim (.dim inDim .scalar)) →
    Ref (.dim outDim .scalar) →
    Ref (.dim inDim .scalar) →
    m (Ref (.dim outDim .scalar))
  mseLoss : {s : Shape} → Ref s → Ref s → m (Ref Shape.scalar)
  layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0) :
    Ref (.dim seqLen (.dim embedDim .scalar)) →
    Ref (.dim embedDim .scalar) →
    Ref (.dim embedDim .scalar) →
    m (Ref (.dim seqLen (.dim embedDim .scalar)))
  batchnormChannelFirst {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w
    : width > 0) :
    Ref (.dim channels (.dim height (.dim width .scalar))) →
    Ref (.dim channels .scalar) →
    Ref (.dim channels .scalar) →
    m (Ref (.dim channels (.dim height (.dim width .scalar))))
  multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) :
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim (numHeads * headDim) (.dim dModel .scalar)) →
    Ref (.dim n (.dim dModel .scalar)) →
    Option (Tensor Bool (.dim n (.dim n .scalar))) →
    m (Ref (.dim n (.dim dModel .scalar)))
  conv {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (outC :: inC :: kernel.toList)) →
    Ref (.dim outC .scalar) →
    Ref (Shape.ofList (inC :: inSpatial.toList)) →
    m (Ref (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)))
  convTranspose {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (inC :: outC :: kernel.toList)) →
    Ref (.dim outC .scalar) →
    Ref (Shape.ofList (inC :: inSpatial.toList)) →
    m (Ref (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
  conv2d {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} :
    Ref (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) →
    Ref (.dim outC .scalar) →
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1)
      (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar))))

  convTranspose2d {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} :
    Ref (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))) →
    Ref (.dim outC .scalar) →
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
      (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar))))

  /-
  Seeded RNG primitives (first-class in TorchLean graphs).

  These are deterministic functions of:
  - the provided `seed` (user-controlled), and
  - backend-specific internal counters (e.g. node id / call index).

  They do not rely on `IO` randomness, so compiled graphs remain replayable.
  -/
  randUniform : {s : Shape} → (seed : Nat) → m (Ref s)
  bernoulliMask : {s : Shape} → Ref Shape.scalar → (seed : Nat) → m (Ref s)

section

variable {m : Type → Type} {α : Type} [Context α] [DecidableEq Shape] [Monad m] [Ops (m := m) (α :=
  α)]

/--
Reference type for the current `Ops` instance.

In eager mode this will typically be `TensorRef`; in compiled mode it will typically be
  `GraphM.Var`.
-/
abbrev Ref (s : Shape) : Type := Ops.Ref (m := m) (α := α) s

/-- Re-export of `Ops.const`. PyTorch: `torch.tensor(...)` / literal constants. -/
def const {s : Shape} (t : Tensor α s) : m (Ref (m := m) (α := α) s) := Ops.const (m := m) (α := α)
  t
/-- Re-export of `Ops.add`. PyTorch: `torch.add` / `+`. -/
def add {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.add (m :=
  m) (α := α) a b
/-- Re-export of `Ops.sub`. PyTorch: `torch.sub` / `-`. -/
def sub {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sub (m :=
  m) (α := α) a b
/-- Re-export of `Ops.mul`. PyTorch: `torch.mul` / `*`. -/
def mul {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.mul (m :=
  m) (α := α) a b
/-- Re-export of `Ops.scale`. PyTorch: `x * c` for a scalar `c`. -/
def scale {s : Shape} (x : Ref (m := m) (α := α) s) (c : α) : m (Ref (m := m) (α := α) s) :=
  Ops.scale (m := m) (α := α) x c
/-- Re-export of `Ops.abs`. PyTorch: `torch.abs`. -/
def abs {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.abs (m := m)
  (α := α) x
/-- Re-export of `Ops.sqrt`. PyTorch: `torch.sqrt`. -/
def sqrt {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sqrt (m :=
  m) (α := α) x
/-- Re-export of `Ops.clamp`. PyTorch: `torch.clamp`. -/
def clamp {s : Shape} (x : Ref (m := m) (α := α) s) (minVal maxVal : α) :
    m (Ref (m := m) (α := α) s) :=
  Ops.clamp (m := m) (α := α) (s := s) x minVal maxVal
/-- Re-export of `Ops.max`. PyTorch: `torch.maximum`. -/
def max {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.max (m :=
  m) (α := α) a b
/-- Re-export of `Ops.min`. PyTorch: `torch.minimum`. -/
def min {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.min (m :=
  m) (α := α) a b
/-- Re-export of `Ops.broadcastTo`. PyTorch: broadcasting / `expand`. -/
def broadcastTo {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂)
  (x : Ref (m := m) (α := α) s₁) : m (Ref (m := m) (α := α) s₂) :=
  Ops.broadcastTo (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) cb x
/-- Re-export of `Ops.reshape`. PyTorch: `reshape` / `view`. -/
def reshape {s₁ s₂ : Shape} (x : Ref (m := m) (α := α) s₁) (h : Shape.size s₁ = Shape.size s₂) :
  m (Ref (m := m) (α := α) s₂) :=
  Ops.reshape (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) x h
/-- Re-export of `Ops.transpose2d`. PyTorch: `x.t()` / `transpose`. -/
def transpose2d {mDim nDim : Nat}
  (x : Ref (m := m) (α := α) (.dim mDim (.dim nDim .scalar))) :
  m (Ref (m := m) (α := α) (.dim nDim (.dim mDim .scalar))) :=
  Ops.transpose2d (m := m) (α := α) (mDim := mDim) (nDim := nDim) x
/-- Re-export of `Ops.transpose3d_first_to_last`. PyTorch: `permute(1,2,0)`. -/
def transpose3dFirstToLast {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim b (.dim c (.dim a .scalar)))) :=
  Ops.transpose3dFirstToLast (m := m) (α := α) (a := a) (b := b) (c := c) x
/-- Re-export of `Ops.transpose3d_last_to_first`. PyTorch: `permute(2,0,1)`. -/
def transpose3dLastToFirst {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim c (.dim a (.dim b .scalar)))) :=
  Ops.transpose3dLastToFirst (m := m) (α := α) (a := a) (b := b) (c := c) x
/-- Re-export of `Ops.transpose3d_last_two`. PyTorch: `transpose(1,2)`. -/
def transpose3dLastTwo {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim a (.dim c (.dim b .scalar)))) :=
  Ops.transpose3dLastTwo (m := m) (α := α) (a := a) (b := b) (c := c) x
/-- Re-export of `Ops.swapAdjacentAtDepth` (general adjacent-axis swap). -/
def swapAdjacentAtDepth {s : Shape} (depth : Nat)
  (x : Ref (m := m) (α := α) s) :
  m (Ref (m := m) (α := α) (s.swapAdjacentAtDepth depth)) :=
  Ops.swapAdjacentAtDepth (m := m) (α := α) (s := s) depth x
/-- Re-export of `Ops.reduce_sum`. PyTorch: `torch.sum(..., dim=axis)`. -/
def reduceSum {s : Shape} (axis : Nat) [Shape.valid_axis_inst axis s] [Shape.WellFormed s]
  (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceSum (m := m) (α := α) (s := s) axis x
/-- Re-export of `Ops.reduce_mean`. PyTorch: `torch.mean(..., dim=axis)`. -/
def reduceMean {s : Shape} (axis : Nat) [Shape.valid_axis_inst axis s] [Shape.WellFormed s]
  (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceMean (m := m) (α := α) (s := s) axis x
/-- Re-export of `Ops.gather_scalar`. PyTorch: `x[i]` (1D). -/
def gatherScalar {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (i : Fin n) : m (Ref (m := m) (α := α) Shape.scalar)
    :=
  Ops.gatherScalar (m := m) (α := α) (n := n) x i
/-- Re-export of `Ops.gather_row`. PyTorch: `x[i]` (2D row). -/
def gatherRow {rows cols : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  m (Ref (m := m) (α := α) (.dim cols .scalar)) :=
  Ops.gatherRow (m := m) (α := α) (rows := rows) (cols := cols) x i
/-- Re-export of `Ops.gather_scalar_nat` (index is a raw `Nat`). -/
def gatherScalarNat {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (i : Nat) : m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.gatherScalarNat (m := m) (α := α) (n := n) x i
/-- Re-export of `Ops.gather_vec_nat` (index tensor). -/
def gatherVecNat {n k : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (idx : Tensor Nat (.dim k .scalar)) :
  m (Ref (m := m) (α := α) (.dim k .scalar)) :=
  Ops.gatherVecNat (m := m) (α := α) (n := n) (k := k) x idx
/-- Re-export of `Ops.gather_rows_nat` (index tensor). -/
def gatherRowsNat {rows cols k : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar))) (idx : Tensor Nat (.dim k .scalar)) :
  m (Ref (m := m) (α := α) (.dim k (.dim cols .scalar))) :=
  Ops.gatherRowsNat (m := m) (α := α) (rows := rows) (cols := cols) (k := k) x idx
/-- Convert a float vector of integer token ids to `Tensor Nat` (non-differentiable). -/
def tokenIdsFromFloatVec {k : Nat}
  (x : Ref (m := m) (α := α) (.dim k .scalar)) :
  m (Tensor Nat (.dim k .scalar)) :=
  Ops.tokenIdsFromFloatVec (m := m) (α := α) (k := k) x
/-- Re-export of `Ops.scatter_add_vec`. -/
def scatterAddVec {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (v : Ref (m := m) (α := α) Shape.scalar) (i : Fin n)
    :
  m (Ref (m := m) (α := α) (.dim n .scalar)) :=
  Ops.scatterAddVec (m := m) (α := α) (n := n) x v i
/-- Re-export of `Ops.scatter_add_row`. -/
def scatterAddRow {rows cols : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar)))
  (v : Ref (m := m) (α := α) (.dim cols .scalar))
  (i : Fin rows) :
  m (Ref (m := m) (α := α) (.dim rows (.dim cols .scalar))) :=
  Ops.scatterAddRow (m := m) (α := α) (rows := rows) (cols := cols) x v i
/-- Re-export of `Ops.matmul`. PyTorch: `torch.matmul` for 2D tensors. -/
def matmul {mDim nDim pDim : Nat}
  (a : Ref (m := m) (α := α) (.dim mDim (.dim nDim .scalar)))
  (b : Ref (m := m) (α := α) (.dim nDim (.dim pDim .scalar))) :
  m (Ref (m := m) (α := α) (.dim mDim (.dim pDim .scalar))) :=
  Ops.matmul (m := m) (α := α) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b
/-- Re-export of `Ops.bmm`. PyTorch: `torch.bmm`. -/
def bmm {batch mDim nDim pDim : Nat}
  (a : Ref (m := m) (α := α) (.dim batch (.dim mDim (.dim nDim .scalar))))
  (b : Ref (m := m) (α := α) (.dim batch (.dim nDim (.dim pDim .scalar)))) :
  m (Ref (m := m) (α := α) (.dim batch (.dim mDim (.dim pDim .scalar)))) :=
  Ops.bmm (m := m) (α := α) (batch := batch) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b
/-- Re-export of `Ops.concat_vectors`. PyTorch: `torch.cat([a,b], dim=0)` for vectors. -/
def concatVectors {nDim mDim : Nat}
  (a : Ref (m := m) (α := α) (.dim nDim .scalar))
  (b : Ref (m := m) (α := α) (.dim mDim .scalar)) :
  m (Ref (m := m) (α := α) (.dim (nDim + mDim) .scalar)) :=
  Ops.concatVectors (m := m) (α := α) (nDim := nDim) (mDim := mDim) a b
/-- Re-export of `Ops.concat_leading_axis`. PyTorch: `torch.cat(..., dim=0)`. -/
def concatLeadingAxis {nDim mDim : Nat} {s : Shape}
  (a : Ref (m := m) (α := α) (.dim nDim s))
  (b : Ref (m := m) (α := α) (.dim mDim s)) :
  m (Ref (m := m) (α := α) (.dim (nDim + mDim) s)) :=
  Ops.concatLeadingAxis (m := m) (α := α) (nDim := nDim) (mDim := mDim) (s := s) a b
/-- Re-export of `Ops.slice_leading_axis_range`. PyTorch: `x[start:start+len]` on the leading dimension. -/
def sliceLeadingAxisRange {nDim : Nat} {s : Shape} (start len : Nat) (h : len + start ≤ nDim)
  (x : Ref (m := m) (α := α) (.dim nDim s)) :
  m (Ref (m := m) (α := α) (.dim len s)) :=
  Ops.sliceLeadingAxisRange (m := m) (α := α) (nDim := nDim) (s := s) start len h x
/--
Re-export of `Ops.max_pool` (generic N-D max pooling, channels-first; no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.maxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel)
    x
/--
Re-export of `Ops.avg_pool` (generic N-D average pooling, channels-first; no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.avgPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    hKernel
    x
/--
Re-export of `Ops.smooth_max_pool` (generic N-D smooth max pooling, channels-first; no batch axis).

This is a differentiable approximation to max pooling; PyTorch does not expose it as a single
primitive.
-/
def smoothMaxPool {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList)))
  (beta : α) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.smoothMaxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel)
    x beta
/-- Re-export of `Ops.max_pool2d`. PyTorch: `torch.nn.functional.max_pool2d`. -/
def maxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  Ops.maxPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (h1 := h1) (h2 := h2) x
/-- Re-export of `Ops.max_pool2d_pad`. PyTorch: `max_pool2d(..., padding=...)`. -/
def maxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) :=
  Ops.maxPool2dPad (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (padding := padding) (h1 := h1) (h2 := h2) x

/-- Alias for `max_pool2d_pad` (PyTorch-style shorthand). -/
abbrev maxPoolPad := @maxPool2dPad

/-- Re-export of `Ops.smooth_max_pool2d` (softmax pooling). -/
def smoothMaxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  Ops.smoothMaxPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC :=
    inC)
    (stride := stride) (h1 := h1) (h2 := h2) x beta
/-- Re-export of `Ops.avg_pool2d`. PyTorch: `torch.nn.functional.avg_pool2d`. -/
def avgPool2d {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1)
    .scalar)))) :=
  Ops.avgPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) h1 h2 x
/-- Re-export of `Ops.avg_pool2d_pad`. PyTorch: `avg_pool2d(..., padding=...)`. -/
def avgPool2dPad {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) :=
  Ops.avgPool2dPad (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (padding := padding) h1 h2 x

/-- Alias for `avg_pool2d_pad` (PyTorch-style shorthand). -/
abbrev avgPoolPad := @avgPool2dPad
/-- Re-export of `Ops.relu`. -/
def relu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.relu (m :=
  m) (α := α) x
/-- Re-export of `Ops.sigmoid`. -/
def sigmoid {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sigmoid
  (m := m) (α := α) x
/-- Re-export of `Ops.tanh`. -/
def tanh {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.tanh (m :=
  m) (α := α) x
/-- Re-export of `Ops.softmax`. -/
def softmax {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.softmax
  (m := m) (α := α) x
/-- Re-export of `Ops.softplus`. -/
def softplus {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.softplus
  (m := m) (α := α) x
/-- Re-export of `Ops.exp`. -/
def exp {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.exp (m := m)
  (α := α) x
/-- Re-export of `Ops.log`. -/
def log {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.log (m := m)
  (α := α) x
/-- Re-export of `Ops.inv` (reciprocal). -/
def inv {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.inv (m := m)
  (α := α) x
/-- Re-export of `Ops.detach`. PyTorch: `x.detach()`. -/
def detach {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.detach (m := m) (α := α) x
/-- Re-export of `Ops.safe_log`. -/
def safeLog {s : Shape} (x : Ref (m := m) (α := α) s) (ε : α := Numbers.epsilon) :
  m (Ref (m := m) (α := α) s) :=
  Ops.safeLog (m := m) (α := α) (s := s) x ε
/-- Re-export of `Ops.rand_uniform` (deterministic seeded RNG). -/
def randUniform {s : Shape} (seed : Nat) : m (Ref (m := m) (α := α) s) :=
  Ops.randUniform (m := m) (α := α) (s := s) seed
/-- Re-export of `Ops.bernoulli_mask` (deterministic dropout-style mask). -/
def bernoulliMask {s : Shape} (keepProb : Ref (m := m) (α := α) Shape.scalar) (seed : Nat) :
    m (Ref (m := m) (α := α) s) :=
  Ops.bernoulliMask (m := m) (α := α) (s := s) keepProb seed

/--
Stable `log_softmax(x)` along the last axis.

This is a backend primitive with the standard max-shifted formulation
`x - max(x) - log(sum(exp(x - max(x))))`, matching PyTorch's numerical intent.  The optional
`ε` parameter is accepted to keep existing call sites stable and is ignored by this primitive;
callers that need an epsilon-smoothed logarithm should use `safeLog` explicitly.
-/
def logSoftmax {s : Shape} (x : Ref (m := m) (α := α) s) (ε : α := Numbers.epsilon) :
    m (Ref (m := m) (α := α) s) :=
  let _epsilonAcceptedForCallSites := ε
  Ops.logSoftmax (m := m) (α := α) (s := s) x

/-- SiLU / swish: `x * sigmoid(x)`. -/
def silu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := do
  let sx ← sigmoid (m := m) (α := α) (s := s) x
  mul (m := m) (α := α) (s := s) x sx

/--
GELU (approximation used by many ML frameworks):

`0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))`.

This is defined using existing primitives (`tanh`, `mul`, `add`, `scale`), so it works in eager,
compiled, and verifier-IR backends without introducing a new opcode.
-/
def gelu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := do
  let c0 : α := ((44715 : Nat) : α) / ((1000000 : Nat) : α)
  let c1 : α := MathFunctions.sqrt (Numbers.two / MathFunctions.pi)
  let x2 ← mul (m := m) (α := α) (s := s) x x
  let x3 ← mul (m := m) (α := α) (s := s) x2 x
  let inner ← add (m := m) (α := α) (s := s) x (← scale (m := m) (α := α) (s := s) x3 c0)
  let t ← tanh (m := m) (α := α) (s := s) (← scale (m := m) (α := α) (s := s) inner c1)
  let oneScalar ← const (m := m) (α := α) (scalarTensor (Numbers.one : α))
  let oneB ← broadcastTo (m := m) (α := α) (s₁ := Shape.scalar) (s₂ := s)
    (Shape.CanBroadcastTo.scalar_to_any s) oneScalar
  let onePlus ← add (m := m) (α := α) (s := s) t oneB
  let mid ← mul (m := m) (α := α) (s := s) x onePlus
  scale (m := m) (α := α) (s := s) mid (Numbers.pointfive : α)

/--
Global average pooling over the last two axes of a `C×H×W` tensor (channel-first).

Returns a vector `C`, averaging each channel over `H×W`.
-/
def globalAvgPool2dChw {c h w : Nat}
    (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : Ref (m := m) (α := α) (.dim c (.dim h (.dim w .scalar)))) :
    m (Ref (m := m) (α := α) (.dim c .scalar)) := do
  let sCHW : Shape := .dim c (.dim h (.dim w .scalar))
  let _ : Shape.WellFormed sCHW := ⟨⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩
  let axisW : Nat := Shape.rank sCHW - 1
  have hrank : Shape.rank sCHW > 0 := by simp [sCHW, Shape.rank]
  let _ : Shape.valid_axis_inst axisW sCHW := Shape.validAxisLastAuto hrank
  let yCH ← reduceMean (m := m) (α := α) (s := sCHW) axisW x
  let sCH : Shape := shapeAfterSum sCHW axisW
  have hsCH : sCH = .dim c (.dim h .scalar) := by
    simp [sCH, sCHW, axisW, Shape.rank, shapeAfterSum]
  let _ : Shape.WellFormed sCH := by
    simpa [hsCH] using (show Shape.WellFormed (.dim c (.dim h .scalar)) from ⟨⟨h_c_pos, ⟨h_h_pos,
      trivial⟩⟩⟩)
  let axisH : Nat := Shape.rank sCH - 1
  have hrank2 : Shape.rank sCH > 0 := by simp [hsCH, Shape.rank]
  let _ : Shape.valid_axis_inst axisH sCH := Shape.validAxisLastAuto hrank2
  let yCH' : Ref (m := m) (α := α) sCH := by
    change Ref (m := m) (α := α) (shapeAfterSum sCHW axisW)
    exact yCH
  let yC ← reduceMean (m := m) (α := α) (s := sCH) axisH yCH'
  have hsC : shapeAfterSum sCH axisH = .dim c .scalar := by
    simp [hsCH, axisH, Shape.rank]
  return (by simpa [hsC] using yC)

/--
Global average pooling over the last two axes of an `N×C×H×W` tensor (PyTorch default layout).

Returns `N×C`, averaging each channel over `H×W` for each batch element.
-/
def globalAvgPool2dNchw {n c h w : Nat}
    (h_n_pos : n > 0) (h_c_pos : c > 0) (h_h_pos : h > 0) (h_w_pos : w > 0)
    (x : Ref (m := m) (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))) :
    m (Ref (m := m) (α := α) (.dim n (.dim c .scalar))) := do
  let sNCHW : Shape := .dim n (.dim c (.dim h (.dim w .scalar)))
  let _ : Shape.WellFormed sNCHW := ⟨⟨h_n_pos, ⟨h_c_pos, ⟨h_h_pos, ⟨h_w_pos, trivial⟩⟩⟩⟩⟩
  let axisW : Nat := Shape.rank sNCHW - 1
  have hrank : Shape.rank sNCHW > 0 := by simp [sNCHW, Shape.rank]
  let _ : Shape.valid_axis_inst axisW sNCHW := Shape.validAxisLastAuto hrank
  let yNCH ← reduceMean (m := m) (α := α) (s := sNCHW) axisW x
  let sNCH : Shape := shapeAfterSum sNCHW axisW
  have hsNCH : sNCH = .dim n (.dim c (.dim h .scalar)) := by
    simp [sNCH, sNCHW, axisW, Shape.rank]
  let _ : Shape.WellFormed sNCH := by
    simpa [hsNCH] using
      (show Shape.WellFormed (.dim n (.dim c (.dim h .scalar))) from ⟨⟨h_n_pos, ⟨h_c_pos, ⟨h_h_pos,
        trivial⟩⟩⟩⟩)
  let axisH : Nat := Shape.rank sNCH - 1
  have hrank2 : Shape.rank sNCH > 0 := by simp [hsNCH, Shape.rank]
  let _ : Shape.valid_axis_inst axisH sNCH := Shape.validAxisLastAuto hrank2
  let yNCH' : Ref (m := m) (α := α) sNCH := by
    change Ref (m := m) (α := α) (shapeAfterSum sNCHW axisW)
    exact yNCH
  let yNC ← reduceMean (m := m) (α := α) (s := sNCH) axisH yNCH'
  have hsNC : shapeAfterSum sNCH axisH = .dim n (.dim c .scalar) := by
    simp [hsNCH, axisH, Shape.rank, shapeAfterSum]
  return (by simpa [hsNC] using yNC)
/-- Re-export of `Ops.sum`. PyTorch: `x.sum()`. -/
def sum {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.sum (m := m) (α := α) (s := s) x
/-- Re-export of `Ops.flatten`. PyTorch: `torch.flatten`. -/
def flatten {s : Shape} (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) (.dim (Shape.size s) .scalar)) :=
  Ops.flatten (m := m) (α := α) (s := s) x

/-- Re-export of `Ops.linear`. PyTorch: `torch.nn.functional.linear`. -/
def linear {inDim outDim : Nat}
  (w : Ref (m := m) (α := α) (.dim outDim (.dim inDim .scalar)))
  (b : Ref (m := m) (α := α) (.dim outDim .scalar))
  (x : Ref (m := m) (α := α) (.dim inDim .scalar)) :
  m (Ref (m := m) (α := α) (.dim outDim .scalar)) :=
  Ops.linear (m := m) (α := α) (inDim := inDim) (outDim := outDim) w b x

/-- Re-export of `Ops.mse_loss`. PyTorch: `torch.nn.functional.mse_loss`. -/
def mseLoss {s : Shape} (yhat target : Ref (m := m) (α := α) s) :
  m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.mseLoss (m := m) (α := α) (s := s) yhat target

/-- Re-export of `Ops.layer_norm`. PyTorch: `nn.LayerNorm` / `functional.layer_norm`. -/
def layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Ref (m := m) (α := α) (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Ref (m := m) (α := α) (.dim embedDim .scalar))
  (beta : Ref (m := m) (α := α) (.dim embedDim .scalar)) :
  m (Ref (m := m) (α := α) (.dim seqLen (.dim embedDim .scalar))) :=
  Ops.layerNorm (m := m) (α := α) (seqLen := seqLen) (embedDim := embedDim)
    h_seq_pos h_embed_pos x gamma beta

/-- Re-export of `Ops.batchnorm_channel_first`. PyTorch: `nn.BatchNorm2d` (conceptually). -/
def batchnormChannelFirst {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0)
  (h_w : width > 0)
  (x : Ref (m := m) (α := α) (.dim channels (.dim height (.dim width .scalar))))
  (gamma : Ref (m := m) (α := α) (.dim channels .scalar))
  (beta : Ref (m := m) (α := α) (.dim channels .scalar)) :
  m (Ref (m := m) (α := α) (.dim channels (.dim height (.dim width .scalar)))) :=
  Ops.batchnormChannelFirst (m := m) (α := α) (channels := channels) (height := height) (width :=
    width)
    h_c h_h h_w x gamma beta

/-- Re-export of `Ops.multi_head_attention`. -/
def multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Ref (m := m) (α := α) (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Ref (m := m) (α := α) (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  m (Ref (m := m) (α := α) (.dim n (.dim dModel .scalar))) :=
  Ops.multiHeadAttention (m := m) (α := α) (n := n) (numHeads := numHeads) (dModel := dModel)
    (headDim := headDim) h1 wq wk wv wo x mask

/--
Re-export of `Ops.conv` (generic N-D convolution, channels-first).

PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample (no batch axis).
-/
def conv {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (weight : Ref (m := m) (α := α) (Shape.ofList (outC :: inC :: kernel.toList)))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (Shape.ofList (inC :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  Ops.conv (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    weight bias input

/--
Re-export of `Ops.conv_transpose` (generic N-D transpose convolution, channels-first).

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (weight : Ref (m := m) (α := α) (Shape.ofList (inC :: outC :: kernel.toList)))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (Shape.ofList (inC :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))) :=
  Ops.convTranspose (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    weight bias input

/-- Re-export of `Ops.conv2d`. PyTorch: `torch.nn.functional.conv2d` (conceptually, no batch axis).
  -/
def conv2d {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Ref (m := m) (α := α) (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1)
    (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) :=
  Ops.conv2d (m := m) (α := α) (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
      h3)
    kernel bias input

/-- Re-export of `Ops.conv_transpose2d`. PyTorch: `torch.nn.functional.conv_transpose2d`. -/
def convTranspose2d {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Ref (m := m) (α := α) (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
    (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) :=
  Ops.convTranspose2d (m := m) (α := α)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW)
    (h1 := h1) (h2 := h2) (h3 := h3)
    kernel bias input

end
end Torch
end Autograd
end Runtime
