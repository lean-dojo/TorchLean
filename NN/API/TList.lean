/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Runtime

import Mathlib.Algebra.Order.Algebra

/-!
# TList

`TList` / supervised-sample ergonomics.

TorchLean uses dependently-typed heterogeneous lists (`TList α ss`) to keep tensor shapes aligned
with the type-level list `ss`.

This is great for safety, but raw `.cons ...` pattern matching is noisy in tutorials. This module
provides small tuple-like accessors and constructors so end-user code can stay readable.

### PyTorch Mapping

PyTorch typically represents multi-tensor samples as plain tuples `(x, y, ...)`.
`TList` plays a similar role, but with the extra benefit that each component's shape is tracked in
the type, so "wrong order" bugs become type errors.
-/

@[expose] public section


namespace NN
namespace API

namespace tlist

/-!
Small ergonomics for TorchLean's typed tensor lists (`TList`).

`TList α ss` is a heterogeneous list of tensors whose shapes are tracked by the type-level list
  `ss`.
It is great for safety, but raw destructuring via `.cons ...` is noisy in demos.

This namespace provides the small "get/unpack" helpers you would expect from tuple-like samples.
-/

/-- Typed tensor lists, used throughout TorchLean as shape-tracked tuples of tensors. -/
abbrev TList (α : Type) (ss : List Spec.Shape) :=
  TorchLean.TList α ss

/-- Construct a 1-element `TList` (like a 1-tuple). -/
def mk1 {α : Type} {s : Spec.Shape} (x : Spec.Tensor α s) : TList α [s] :=
  TorchLean.tlist1 x

/-- Construct a 2-element `TList` (like a pair). -/
def mk2 {α : Type} {s₁ s₂ : Spec.Shape} (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) :
    TList α [s₁, s₂] :=
  TorchLean.tlist2 x₁ x₂

/-- Construct a 3-element `TList` (like a 3-tuple). -/
def mk3 {α : Type} {s₁ s₂ s₃ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) (x₃ : Spec.Tensor α s₃) :
    TList α [s₁, s₂, s₃] :=
  TorchLean.tlist3 x₁ x₂ x₃

/-- Construct a 4-element `TList` (like a 4-tuple). -/
def mk4 {α : Type} {s₁ s₂ s₃ s₄ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) (x₃ : Spec.Tensor α s₃) (x₄ : Spec.Tensor α s₄)
      :
    TList α [s₁, s₂, s₃, s₄] :=
  TorchLean.tlist4 x₁ x₂ x₃ x₄

/-- Map each tensor entry (shape-preserving). -/
def map {α β : Type} (f : ∀ {s : Spec.Shape}, Spec.Tensor α s → Spec.Tensor β s) :
    {ss : List Spec.Shape} → TList α ss → TList β ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs => .cons (f x) (map (f := f) (ss := ss) xs)

/-- Zip two `TList`s pointwise (shape-preserving). -/
def zipWith {α β γ : Type}
    (f : ∀ {s : Spec.Shape}, Spec.Tensor α s → Spec.Tensor β s → Spec.Tensor γ s) :
    {ss : List Spec.Shape} → TList α ss → TList β ss → TList γ ss
  | [], .nil, .nil => .nil
  | _s :: ss, .cons x xs, .cons y ys => .cons (f x y) (zipWith (f := f) (ss := ss) xs ys)

/-- Append two `TList`s. -/
def append {α : Type} :
    {ss₁ ss₂ : List Spec.Shape} → TList α ss₁ → TList α ss₂ → TList α (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a `TList α (ss₁ ++ ss₂)` into its prefix and suffix. -/
def split {α : Type} :
    {ss₁ ss₂ : List Spec.Shape} → TList α (ss₁ ++ ss₂) → TList α ss₁ × TList α ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (xs₁, xs₂) := split (α := α) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xs₁, xs₂)

/-- First element of a non-empty `TList` (0-indexed). -/
def get0 {α : Type} {s : Spec.Shape} {ss : List Spec.Shape} :
    TList α (s :: ss) → Spec.Tensor α s
  | .cons x _xs => x

/-- Second element of a `TList` with at least two entries (0-indexed). -/
def get1 {α : Type} {s₀ s₁ : Spec.Shape} {ss : List Spec.Shape} :
    TList α (s₀ :: s₁ :: ss) → Spec.Tensor α s₁
  | .cons _x₀ (.cons x₁ _xs) => x₁

/-- Third element of a `TList` with at least three entries (0-indexed). -/
def get2 {α : Type} {s₀ s₁ s₂ : Spec.Shape} {ss : List Spec.Shape} :
    TList α (s₀ :: s₁ :: s₂ :: ss) → Spec.Tensor α s₂
  | .cons _x₀ (.cons _x₁ (.cons x₂ _xs)) => x₂

/-- Fourth element of a `TList` with at least four entries (0-indexed). -/
def get3 {α : Type} {s₀ s₁ s₂ s₃ : Spec.Shape} {ss : List Spec.Shape} :
    TList α (s₀ :: s₁ :: s₂ :: s₃ :: ss) → Spec.Tensor α s₃
  | .cons _x₀ (.cons _x₁ (.cons _x₂ (.cons x₃ _xs))) => x₃

/-- Tail of a non-empty `TList` (drop the first element). -/
def tail {α : Type} {s : Spec.Shape} {ss : List Spec.Shape} :
    TList α (s :: ss) → TList α ss
  | .cons _x xs => xs

/-- Unpack a length-1 `TList` into its element. -/
def unpack1 {α : Type} {s : Spec.Shape} :
    TList α [s] → Spec.Tensor α s
  | .cons x .nil => x

/-- Unpacking `mk1` yields the original element. -/
@[simp] theorem unpack1_mk1 {α : Type} {s : Spec.Shape} (x : Spec.Tensor α s) :
    unpack1 (mk1 (α := α) (s := s) x) = x := by
  simp [unpack1, mk1, TorchLean.tlist1]

/-- Unpack a length-2 `TList` into a Lean pair. -/
def unpack2 {α : Type} {s₁ s₂ : Spec.Shape} :
    TList α [s₁, s₂] → (Spec.Tensor α s₁ × Spec.Tensor α s₂)
  | .cons x₁ (.cons x₂ .nil) => (x₁, x₂)

/-- Unpacking `mk2` yields the original pair. -/
@[simp] theorem unpack2_mk2 {α : Type} {s₁ s₂ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) :
    unpack2 (mk2 (α := α) (s₁ := s₁) (s₂ := s₂) x₁ x₂) = (x₁, x₂) := by
  simp [unpack2, mk2, TorchLean.tlist2]

/-- Unpack a length-3 `TList` into a Lean triple. -/
def unpack3 {α : Type} {s₁ s₂ s₃ : Spec.Shape} :
    TList α [s₁, s₂, s₃] → (Spec.Tensor α s₁ × Spec.Tensor α s₂ × Spec.Tensor α s₃)
  | .cons x₁ (.cons x₂ (.cons x₃ .nil)) => (x₁, x₂, x₃)

/-- Unpacking `mk3` yields the original triple. -/
@[simp] theorem unpack3_mk3 {α : Type} {s₁ s₂ s₃ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) (x₃ : Spec.Tensor α s₃) :
    unpack3 (mk3 (α := α) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃) x₁ x₂ x₃) = (x₁, x₂, x₃) := by
  simp [unpack3, mk3, TorchLean.tlist3]

/-- Unpack a length-4 `TList` into a Lean 4-tuple. -/
def unpack4 {α : Type} {s₁ s₂ s₃ s₄ : Spec.Shape} :
    TList α [s₁, s₂, s₃, s₄] →
      (Spec.Tensor α s₁ × Spec.Tensor α s₂ × Spec.Tensor α s₃ × Spec.Tensor α s₄)
  | .cons x₁ (.cons x₂ (.cons x₃ (.cons x₄ .nil))) => (x₁, x₂, x₃, x₄)

/-- Unpacking `mk4` yields the original 4-tuple. -/
@[simp] theorem unpack4_mk4 {α : Type} {s₁ s₂ s₃ s₄ : Spec.Shape}
    (x₁ : Spec.Tensor α s₁) (x₂ : Spec.Tensor α s₂) (x₃ : Spec.Tensor α s₃) (x₄ : Spec.Tensor α s₄)
      :
    unpack4 (mk4 (α := α) (s₁ := s₁) (s₂ := s₂) (s₃ := s₃) (s₄ := s₄) x₁ x₂ x₃ x₄) =
      (x₁, x₂, x₃, x₄) := by
  simp [unpack4, mk4, TorchLean.tlist4]

end tlist

namespace sample

/-!
Ergonomics for the common supervised-learning sample shape `TList α [xShape, yShape]`.

This keeps tutorial code closer to the PyTorch convention of `(x, y)` pairs without losing
TorchLean's static shape safety.
-/

/-- A supervised sample `(x, y)` with input shape `σ` and target shape `τ`. -/
abbrev Supervised (α : Type) (σ τ : Spec.Shape) :=
  TorchLean.TList α [σ, τ]

/-- A fixed-size minibatch of supervised samples. -/
abbrev Batch (α : Type) (n : Nat) (σ τ : Spec.Shape) :=
  Supervised α (.dim n σ) (.dim n τ)

/-- Build a supervised sample `(x, y)` represented as `TList α [σ, τ]`. -/
def mk {α : Type} {σ τ : Spec.Shape} (x : Spec.Tensor α σ) (y : Spec.Tensor α τ) :
    Supervised α σ τ :=
  TorchLean.tlist2 x y

/-- Build a *batched* supervised sample `(xBatch, yBatch)` for a minibatch of size `n`. -/
def batch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (x : Spec.Tensor α (.dim n σ)) (y : Spec.Tensor α (.dim n τ)) :
    Batch α n σ τ :=
  mk x y

/-- Extract the input tensor `x` from a supervised sample. -/
def x {α : Type} {σ τ : Spec.Shape} (s : Supervised α σ τ) : Spec.Tensor α σ :=
  tlist.get0 s

/-- Extract the target tensor `y` from a supervised sample. -/
def y {α : Type} {σ τ : Spec.Shape} (s : Supervised α σ τ) : Spec.Tensor α τ :=
  tlist.get1 s

/-- `x` of a constructed supervised sample `mk x y` is `x`. -/
@[simp] theorem x_mk {α : Type} {σ τ : Spec.Shape}
    (xT : Spec.Tensor α σ) (yT : Spec.Tensor α τ) :
    x (mk (α := α) (σ := σ) (τ := τ) xT yT) = xT := by
  simp [x, mk, tlist.get0, TorchLean.tlist2]

/-- `y` of a constructed supervised sample `mk x y` is `y`. -/
@[simp] theorem y_mk {α : Type} {σ τ : Spec.Shape}
    (xT : Spec.Tensor α σ) (yT : Spec.Tensor α τ) :
    y (mk (α := α) (σ := σ) (τ := τ) xT yT) = yT := by
  simp [y, mk, tlist.get1, TorchLean.tlist2]

/-- Map a function over the input tensor `x`, leaving the target `y` unchanged. -/
def mapX {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α σ → Spec.Tensor α σ) (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (f (x s)) (y s)

/-- Map a function over the target tensor `y`, leaving the input `x` unchanged. -/
def mapY {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α τ → Spec.Tensor α τ) (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (x s) (f (y s))

/-- Map functions over both `x` and `y` in a supervised sample. -/
def mapXY {α : Type} {σ τ : Spec.Shape}
    (fx : Spec.Tensor α σ → Spec.Tensor α σ)
    (fy : Spec.Tensor α τ → Spec.Tensor α τ)
    (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (fx (x s)) (fy (y s))

end sample

end API
end NN
