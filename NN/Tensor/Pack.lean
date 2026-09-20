/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

/-!
# Heterogeneous Tensor Packs

`TorchLean.TensorPack α shapes` stores one tensor for every shape in `shapes`. Unlike a tensor,
whose entries all have one element type and one rectangular shape, a tensor pack may contain tensors
of different ranks and extents. The shape of every entry is nevertheless known statically.

Tensor packs are the common representation for model parameters, gradients, and typed graph
contexts. Ordinary supervised data uses the named `Sample.Supervised` record; conversion to a pack
happens only at the graph-runtime boundary. This module owns the datatype and its
representation-independent operations so those layers do not depend on autograd proofs or a
particular runtime.
-/

@[expose] public section

namespace TorchLean

open Spec TorchLean

/-- A heterogeneous sequence containing one tensor for every shape in `shapes`. -/
inductive TensorPack (α : Type) [TorchLean.Storage α] : List Shape → Type where
  /-- The empty tensor pack. -/
  | nil : TensorPack α []
  /-- Add a tensor whose shape becomes the head of the pack's shape list. -/
  | cons {s : Shape} {ss : List Shape} :
      TorchLean.Tensor α s → TensorPack α ss → TensorPack α (s :: ss)

namespace TensorPack

variable {α β γ : Type}
  [TorchLean.Storage α] [TorchLean.Storage β]
  [TorchLean.Storage γ]

/-- Construct an empty tensor pack without exposing its recursive representation. -/
def empty : TensorPack α [] :=
  .nil

/-- Construct a pack containing one tensor. -/
def singleton {shape : Shape} (x : TorchLean.Tensor α shape) :
    TensorPack α [shape] :=
  .cons x .nil

/-- Construct a two-tensor pack without exposing its recursive representation. -/
def pair {firstShape secondShape : Shape}
    (first : TorchLean.Tensor α firstShape)
    (second : TorchLean.Tensor α secondShape) :
    TensorPack α [firstShape, secondShape] :=
  .cons first (.cons second .nil)

/-- Return the first tensor in a nonempty pack. -/
def head {shape : Shape} {shapes : List Shape}
    (xs : TensorPack α (shape :: shapes)) :
    TorchLean.Tensor α shape :=
  match xs with
  | .cons x _ => x

/-- Return every tensor after the first in a nonempty pack. -/
def tail {shape : Shape} {shapes : List Shape}
    (xs : TensorPack α (shape :: shapes)) :
    TensorPack α shapes :=
  match xs with
  | .cons _ rest => rest

/-- Return the tensor at position `i`; its shape is determined by the pack's shape list. -/
def get : {ss : List Shape} →
    TensorPack α ss → (i : Fin ss.length) → TorchLean.Tensor α (ss.get i)
  | [], .nil, i => nomatch i
  | _ :: _, .cons x _, ⟨0, _⟩ => x
  | _ :: _ss, .cons _ xs, ⟨Nat.succ i, hi⟩ =>
      get xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩

/-- Reading position zero of a `cons` gives the head tensor. -/
@[simp] theorem get_cons_zero {s : Shape} {ss : List Shape}
    (x : TorchLean.Tensor α s) (xs : TensorPack α ss) (h : 0 < (s :: ss).length) :
    get (.cons x xs) ⟨0, h⟩ = x := by
  rfl

/-- Reading a later position skips the head and recurses into the tail.

With `get_cons_zero`, this pair lets `simp` evaluate any concrete lookup all the way down, which is
what keeps the pack indexing invisible in downstream proofs. -/
@[simp] theorem get_cons_succ {s : Shape} {ss : List Shape}
    (x : TorchLean.Tensor α s) (xs : TensorPack α ss)
    (i : Nat) (h : Nat.succ i < (s :: ss).length) :
    get (.cons x xs) ⟨Nat.succ i, h⟩ =
      get xs ⟨i, Nat.lt_of_succ_lt_succ h⟩ := by
  rfl

/-- Stack a family of packs, giving every parameter tensor the same leading axes. -/
def stackLeading (leading : Shape) :
    {shapes : List Shape} → (Fin leading.size → TensorPack α shapes) →
      TensorPack α (shapes.map leading.concat)
  | [], _ => .nil
  | shape :: shapes, rows =>
      let first : Tensor α (shape.prependDim leading.size) :=
        Tensor.dim fun index => (rows index).head
      .cons (Tensor.Internal.Rep.reshape
        (t := leading.concat shape)
        (by simpa only [Spec.Shape.internalSize_eq] using
          (show (shape.prependDim leading.size).size = (leading.concat shape).size by
            simp [Spec.Shape.size_concat, Spec.Shape.size_prependDim])) first)
        (stackLeading leading (shapes := shapes) fun index => (rows index).tail)

/-- Evaluate a family of packs in row-major order and stack every parameter tensor. -/
def stackLeadingM {m : Type → Type} [Monad m] (leading : Shape) {shapes : List Shape}
    (f : Fin leading.size → m (TensorPack α shapes)) :
    m (TensorPack α (shapes.map leading.concat)) := do
  let rows ← Tensor.Internal.sequenceFinM f
  pure (stackLeading leading rows)

/-- Apply a shape-preserving function to every tensor in a pack. -/
def map (f : ∀ {shape : Shape}, TorchLean.Tensor α shape → TorchLean.Tensor β shape) :
    {ss : List Shape} → TensorPack α ss → TensorPack β ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs => .cons (f x) (map (f := f) (ss := ss) xs)

/-- Combine two packs pointwise with a shape-preserving binary function. -/
def zipWith
    (f : ∀ {shape : Shape},
      TorchLean.Tensor α shape → TorchLean.Tensor β shape → TorchLean.Tensor γ shape) :
    {ss : List Shape} →
      TensorPack α ss → TensorPack β ss → TensorPack γ ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys =>
      .cons (f x y) (zipWith (f := f) (ss := ss) xs ys)

/-- Concatenate two tensor packs. -/
def append : {ss₁ ss₂ : List Shape} →
    TensorPack α ss₁ → TensorPack α ss₂ → TensorPack α (ss₁ ++ ss₂)
  | [], _, .nil, ys => ys
  | _ :: ss₁, ss₂, .cons x xs, ys =>
      .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a tensor pack at a statically known shape-list boundary. -/
def split : {ss₁ ss₂ : List Shape} →
    TensorPack α (ss₁ ++ ss₂) → TensorPack α ss₁ × TensorPack α ss₂
  | [], _, xs => (.nil, xs)
  | _ :: ss₁, ss₂, .cons x xs =>
      let (xsLeft, xsRight) := split (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xsLeft, xsRight)

/-- Splitting a concatenated pair of packs recovers both inputs. -/
@[simp] theorem split_append {ss₁ ss₂ : List Shape}
    (xs : TensorPack α ss₁) (ys : TensorPack α ss₂) :
    split (append xs ys) = (xs, ys) := by
  induction ss₁ with
  | nil => cases xs; rfl
  | cons _ ss₁ ih =>
      cases xs with
      | cons x xs =>
          simp only [append, split]
          rw [ih xs]

/-- Construct the all-zero tensor pack. -/
def zero [Zero α] : {ss : List Shape} → TensorPack α ss
  | [] => .nil
  | shape :: ss => .cons (TorchLean.Tensor.zeros shape) (zero (ss := ss))

/-- Construct a tensor pack whose every entry contains `value`. -/
def fill (value : α) : {ss : List Shape} → TensorPack α ss
  | [] => .nil
  | shape :: ss => .cons (TorchLean.Tensor.full shape value) (fill value (ss := ss))

/-- Print every tensor in a pack in state order, with its statically known shape. -/
instance [Repr α] {shapes : List Shape} : Repr (TensorPack α shapes) where
  reprPrec tensors _ :=
    let rec entries : {ss : List Shape} → TensorPack α ss → List Std.Format
      | [], .nil => []
      | shape :: _, .cons tensor rest =>
          (Std.Format.text shape.pretty ++ ":" ++ Std.Format.line ++ repr tensor) :: entries rest
    Std.Format.bracket "["
      (Std.Format.joinSep (entries tensors) ("," ++ Std.Format.line))
      "]"

/-- Add two tensor packs pointwise. -/
def add [Add α] : {ss : List Shape} →
    TensorPack α ss → TensorPack α ss → TensorPack α ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys =>
      .cons (TorchLean.Tensor.addSpec x y) (add (ss := ss) xs ys)

/-- Multiply every tensor entry by the same scalar. -/
def scale [Mul α] (c : α) : {ss : List Shape} →
    TensorPack α ss → TensorPack α ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs =>
      .cons (TorchLean.Tensor.scaleSpec x c) (scale c (ss := ss) xs)

/-- Subtract two tensor packs pointwise. -/
def sub [Sub α] : {ss : List Shape} →
    TensorPack α ss → TensorPack α ss → TensorPack α ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys =>
      .cons (TorchLean.Tensor.subSpec x y) (sub (ss := ss) xs ys)

/-- Append one tensor to the end of a pack. -/
def snoc {τ : Shape} : {ss : List Shape} →
    TensorPack α ss → TorchLean.Tensor α τ → TensorPack α (ss ++ [τ])
  | [], .nil, x => .cons x .nil
  | _ :: ss, .cons x xs, last => .cons x (snoc (ss := ss) xs last)

/-- Separate a nonempty pack into its prefix and final tensor. -/
def unsnoc {τ : Shape} : {ss : List Shape} →
    TensorPack α (ss ++ [τ]) → TensorPack α ss × TorchLean.Tensor α τ
  | [], .cons x .nil => (.nil, x)
  | _ :: ss, .cons x xs =>
      let (init, last) := unsnoc (ss := ss) xs
      (.cons x init, last)

/-- Transport a tensor pack along an equality between its shape lists. -/
def cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    (xs : TensorPack α ss₁) : TensorPack α ss₂ :=
  Eq.mp (congrArg (TensorPack α) h) xs

/-- Casting along `rfl` does nothing. -/
@[simp] theorem cast_rfl {ss : List Shape} (xs : TensorPack α ss) :
    cast rfl xs = xs := by
  rfl

/-- Two successive casts collapse into one along the composed equality. -/
@[simp] theorem cast_cast {ss₁ ss₂ ss₃ : List Shape}
    (h₁ : ss₁ = ss₂) (h₂ : ss₂ = ss₃) (xs : TensorPack α ss₁) :
    cast h₂ (cast h₁ xs) = cast (h₁.trans h₂) xs := by
  cases h₁
  cases h₂
  rfl

/-- A cast followed by its inverse is the identity.

These three lemmas are the whole reason `cast` is tolerable: casts pile up whenever two packs with
propositionally equal shape lists meet, and as `simp` lemmas they cancel out on their own instead of
being carried through every proof by hand. -/
@[simp] theorem cast_symm {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    (xs : TensorPack α ss₁) :
    cast h.symm (cast h xs) = xs := by
  cases h
  rfl

/-- `unsnoc` recovers the two arguments supplied to `snoc`. -/
@[simp] theorem unsnoc_snoc {ss : List Shape} {τ : Shape}
    (xs : TensorPack α ss) (x : TorchLean.Tensor α τ) :
    unsnoc (snoc xs x) = (xs, x) := by
  induction ss with
  | nil =>
      cases xs
      simp [snoc, unsnoc]
  | cons _ ss ih =>
      cases xs with
      | cons head tail => simp [snoc, unsnoc, ih]

/-- Re-appending the final tensor obtained by `unsnoc` reconstructs the original pack. -/
@[simp] theorem snoc_unsnoc {ss : List Shape} {τ : Shape}
    (xs : TensorPack α (ss ++ [τ])) :
    snoc (unsnoc xs).1 (unsnoc xs).2 = xs := by
  induction ss with
  | nil =>
      cases xs with
      | cons x xs => cases xs; simp [snoc, unsnoc]
  | cons _ ss ih =>
      cases xs with
      | cons head tail => simp [snoc, unsnoc, ih]

/--
Build the extended shape list together with the pack. Each constructor shares the shape-list
suffix returned by the recursive call instead of appending to the original suffix again.
-/
private def snocWithShapes {τ : Shape} : {ss : List Shape} →
    TensorPack α ss → TorchLean.Tensor α τ →
      (shapes : List Shape) × TensorPack α shapes
  | [], .nil, last => ⟨[τ], .cons last .nil⟩
  | shape :: ss, .cons first rest, last =>
      let result := snocWithShapes (ss := ss) rest last
      ⟨shape :: result.1, .cons (ss := result.1) first result.2⟩

private theorem snocWithShapes_eq {ss : List Shape} {τ : Shape}
    (xs : TensorPack α ss) (last : TorchLean.Tensor α τ) :
    snocWithShapes xs last = ⟨ss ++ [τ], snoc xs last⟩ := by
  induction ss with
  | nil =>
      cases xs
      rfl
  | cons shape ss ih =>
      cases xs with
      | cons first rest =>
          exact congrArg
            (fun result : (shapes : List Shape) × TensorPack α shapes =>
              (⟨shape :: result.1, .cons (ss := result.1) first result.2⟩ :
                (shapes : List Shape) × TensorPack α shapes)) (ih rest)

private theorem transport_snd_of_eq {ss : List Shape}
    (result : (shapes : List Shape) × TensorPack α shapes)
    (xs : TensorPack α ss) (h : result = ⟨ss, xs⟩) :
    Eq.mp (congrArg (TensorPack α) (congrArg Sigma.fst h)) result.2 = xs := by
  subst result
  rfl

/-- A shape-sharing implementation of `snoc`, with the same indexed result type. -/
@[no_expose] def Internal.snocLinear {τ : Shape} {ss : List Shape}
    (xs : TensorPack α ss) (last : TorchLean.Tensor α τ) :
    TensorPack α (ss ++ [τ]) :=
  let result := snocWithShapes xs last
  have shapes_eq : result.1 = ss ++ [τ] :=
    congrArg Sigma.fst (snocWithShapes_eq xs last)
  Eq.mp (congrArg (TensorPack α) shapes_eq) result.2

private theorem snocLinear_eq_snoc {ss : List Shape} {τ : Shape}
    (xs : TensorPack α ss) (last : TorchLean.Tensor α τ) :
    Internal.snocLinear xs last = snoc xs last := by
  unfold Internal.snocLinear
  exact transport_snd_of_eq _ _ (snocWithShapes_eq xs last)

/-- Compile `snoc` using shared shape-list suffixes while retaining its logical definition. -/
@[csimp] theorem snoc_eq_snocLinear : @snoc = @Internal.snocLinear := by
  funext α inst τ ss xs last
  exact (snocLinear_eq_snoc xs last).symm

end TensorPack
end TorchLean
