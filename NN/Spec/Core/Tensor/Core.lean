/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Group.Defs
public import Mathlib.Algebra.Group.Pi.Basic
public import Mathlib.Algebra.Group.TransferInstance
public import Mathlib.Logic.Equiv.Defs
public import NN.Spec.Core.Shape

/-!
# Core tensor datatype (`Spec.Tensor`)

This file defines the foundational, shape-indexed tensor type used throughout TorchLean's **spec**
layer:

`Tensor α s`

## Why an inductive / functional representation?

Instead of storing a flat array plus a shape, the spec tensor is a *function from indices*:

- a scalar is `α`
- an `n`-dimensional tensor is `Fin n → ...`

This has three practical benefits:

1) **Shape safety is enforced by the type.**
2) **Proofs are natural:** you reason by recursion on the `Shape` / `Tensor` constructors.
3) **No layout commitment:** the spec layer doesn’t bake in row-major vs column-major storage.

For long executable runs, repeated functional updates can create “closure chains”. Use
`Tensor.materialize` (documented below) to rebuild a tensor into an array-backed normal form.
-/

@[expose] public section


namespace Spec

-- Tensor definition
/--
Shape-indexed tensor datatype for the spec layer.

This is a *functional* representation:
- a scalar tensor is just an `α`,
- an `n`-dimensional tensor is a function `Fin n → Tensor α s`.

This keeps proofs and shape-safe programming simple, and avoids committing to a concrete memory
layout in the spec layer.
-/
inductive Tensor (α : Type) : Shape → Type where
  | scalar : α → Tensor α .scalar
  | dim : ∀ {n s}, (Fin n → Tensor α s) → Tensor α (.dim n s)

/-!
## Runtime note: materialization

`Tensor α s` is a *functional* representation (`Fin n → ...`). This is excellent for proofs, but
repeated updates (for example, many SGD steps) can build deep chains of closures
(`fun i => ... (old (old (old i))) ...`). Evaluating those chains later becomes progressively more
expensive.

`Tensor.materialize` rebuilds a tensor into an array-backed normal form (at every dimension), which
keeps long-running training loops from degrading.

It is extentionally the identity (same mathematical tensor), but it is much friendlier to the
runtime evaluator.
-/

/-- Rebuild a tensor into an array-backed normal form (performance helper). -/
def Tensor.materialize {α : Type} : ∀ {s : Shape}, Tensor α s → Tensor α s
  | .scalar, t => t
  | .dim n s', Tensor.dim f =>
      let arr : Array (Tensor α s') := Array.ofFn (fun i : Fin n => Tensor.materialize (f i))
      Tensor.dim (fun i =>
        -- `arr` has size `n`, so this index is always in-bounds.
        let hn : arr.size = n := by
          simp [arr]
        let hi : i.1 < arr.size :=
          -- Avoid `simp` on `Fin.isLt` goals; transport along `hn.symm` directly.
          Eq.ndrec (motive := fun m => i.1 < m) i.2 hn.symm
        arr[i.1]'hi)

/-- `Tensor.materialize` preserves tensor values (it is extensionally the identity). -/
@[simp]
theorem Tensor.materialize_eq {α : Type} : ∀ {s : Shape} (t : Tensor α s), Tensor.materialize t = t
  := by
  intro s t
  induction t with
  | scalar x =>
      rfl
  | @dim n s f ih =>
      -- Unfold `materialize` and show the inner index function agrees with `f`.
      simp [Tensor.materialize]
      funext i
      simpa using ih i

/-- Default tensor value for any shape (uses `Inhabited.default` at scalars). -/
def Tensor.default {α : Type} [Inhabited α] : ∀ {s : Shape}, Tensor α s
  | .scalar => scalar (@Inhabited.default α _)
  | .dim _ _ => dim (fun _ => default)

/-- Make `Tensor α s` inhabited for any shape `s`. -/
@[reducible, instance]
def Tensor.inhabited {α : Type} [Inhabited α] : ∀ {s : Shape}, Inhabited (Tensor α s)
  | _ => ⟨Tensor.default⟩

/-- Recover the (data) shape from a tensor value. -/
def shapeOf {α : Type} : ∀ {s : Shape}, Tensor α s → Shape
  | .scalar, _ => .scalar
  | .dim 0 s', _ => .dim 0 s'            -- empty dimension
  | .dim (n'+1) _, .dim f => .dim (n'+1) (shapeOf (f ⟨0, Nat.zero_lt_succ n'⟩))

/-- Extract the scalar value from a scalar tensor. -/
def Tensor.toScalar {α : Type} : Tensor α .scalar → α
  | .scalar x => x

/-- Inject a scalar into a scalar tensor. -/
def Tensor.ofScalar {α : Type} (x : α) : Tensor α .scalar := .scalar x

/-- `toScalar (ofScalar x) = x`. -/
@[simp]
lemma Tensor.toScalar_ofScalar {α : Type} (x : α) :
    Tensor.toScalar (Tensor.ofScalar x) = x := rfl

/-- `ofScalar (toScalar t) = t` for scalar tensors. -/
@[simp]
lemma Tensor.ofScalar_toScalar {α : Type} (x : Tensor α .scalar) :
    Tensor.ofScalar (Tensor.toScalar x) = x :=
  by cases x; rfl

/-- Equivalence between `Tensor α .scalar` and `α` (useful to reuse algebra instances). -/
def Tensor.scalarEquiv (α : Type) : Tensor α .scalar ≃ α where
  toFun := Tensor.toScalar
  invFun := Tensor.ofScalar
  left_inv := Tensor.ofScalar_toScalar
  right_inv := Tensor.toScalar_ofScalar

/-- `AddCommMonoid` on scalar tensors, transported from `α` via `Tensor.scalarEquiv`. -/
instance {α : Type} [AddCommMonoid α] : AddCommMonoid (Tensor α .scalar) :=
  Equiv.addCommMonoid (Tensor.scalarEquiv α)

/-- Equivalence between vectors-as-tensors and functions `Fin n → α`. -/
def Tensor.dimScalarEquiv {α : Type} (n : Nat) : Tensor α (.dim n .scalar) ≃ (Fin n → α) :=
Equiv.mk
  (fun t i => match t with
              | Tensor.dim f => Tensor.toScalar (f i))
  (fun f => Tensor.dim (fun i => Tensor.ofScalar (f i)))
  (by
    intro t
    cases t
    simp)
  (by
    intro f
    funext i
    simp)

/-- `AddCommMonoid` on vector tensors, transported from `Fin n → α` via `Tensor.dimScalarEquiv`. -/
instance {α : Type} [AddCommMonoid α] {n : Nat} : AddCommMonoid (Tensor α (.dim n .scalar)) :=
  let _ : AddCommMonoid (Fin n → α) := Pi.addCommMonoid
  Equiv.addCommMonoid (Tensor.dimScalarEquiv n)

namespace Tensor

/-- Cast a tensor along an equality of shapes. -/
def castShape {α : Type} {s₁ s₂ : Shape} (t : Tensor α s₁) (h : s₁ = s₂) : Tensor α s₂ :=
  Eq.mp (congrArg (Tensor α) h) t

/-- Cast a vector tensor along an equality of dimensions. -/
def castVecDim {α : Type} {n m : Nat} (h : n = m) (t : Tensor α (.dim n .scalar)) :
    Tensor α (.dim m .scalar) := by
  cases h
  simpa using t

/-!
### Cast lemmas

In dependently-typed proofs (especially graph/tape correctness proofs), the same cast may arise
with different proof terms. Since equality proofs are proof-irrelevant, we provide a few small
normalization lemmas for `Tensor.cast_shape`.
-/

/-- Casting a tensor along `rfl` is the identity. -/
@[simp] theorem cast_shape_rfl {α : Type} {s : Shape} (t : Tensor α s) :
    castShape (t := t) (h := rfl) = t := by
  rfl

/-- Casting a tensor along a reflexive equality proof is the identity. -/
@[simp] theorem cast_shape_self {α : Type} {s : Shape} (t : Tensor α s) (h : s = s) :
    castShape (t := t) h = t := by
  cases h
  rfl

/-- `Tensor.cast_shape` composes associatively (cast-by-eq is just `Eq.rec`). -/
@[simp] theorem cast_shape_trans {α : Type} {s₁ s₂ s₃ : Shape}
    (t : Tensor α s₁) (h₁₂ : s₁ = s₂) (h₂₃ : s₂ = s₃) :
    castShape (t := castShape (t := t) h₁₂) h₂₃ =
      castShape (t := t) (h₁₂.trans h₂₃) := by
  cases h₁₂
  cases h₂₃
  rfl

/-- Proof-irrelevance for `Tensor.cast_shape`. -/
theorem cast_shape_proof_irrel {α : Type} {s₁ s₂ : Shape}
    (t : Tensor α s₁) {p q : s₁ = s₂} :
    castShape (t := t) p = castShape (t := t) q := by
  have : p = q := Subsingleton.elim _ _
  cases this
  rfl

/-- Rewrite `Eq.rec` (`h ▸ t`) as `Tensor.cast_shape` for uniformity in larger proofs. -/
theorem eqRec_eq_cast_shape {α : Type} {s₁ s₂ : Shape}
    (t : Tensor α s₁) (h : s₁ = s₂) :
    (h ▸ t) = castShape (t := t) h := by
  cases h
  rfl

/-- Proof-irrelevance for `Eq.rec` casts of tensors. -/
theorem eqRec_proof_irrel {α : Type} {s₁ s₂ : Shape}
    (t : Tensor α s₁) {p q : s₁ = s₂} :
    (p ▸ t) = (q ▸ t) := by
  have : p = q := Subsingleton.elim _ _
  cases this
  rfl

-- Tell `grind` about the standard cast normalization lemmas.
attribute [grind =] cast_shape_rfl cast_shape_self cast_shape_trans eqRec_eq_cast_shape

end Tensor

-- Core tensor access operations

/-! ### Indexing helpers -/

/-!
Indexing design notes:

- `get_spec` takes a runtime multi-index (`List Nat`) and returns `Option α`.
  This is intentionally permissive and "frontend-friendly": you can use it for debugging, JSON
  import/export checks, and any place you want to *try* an index without committing to proofs.
- For proof-driven code, you usually want `Fin n` indices and the `get`/`get2` helpers.

PyTorch analogy:
- `get_spec t [i,j,k]` is like `t[i,j,k]` but returns `none` instead of throwing.
- `get t i` is like slicing the first dimension: `t[i]`.
  (TorchLean also supports Lean’s indexing syntax: `t[i]` elaborates to `Spec.get t i`.)
- `get2 A i j` is like `A[i,j]`.
  (And `A[(i, j)]` elaborates to `Spec.get2 A i j` for matrix-shaped scalar tensors.)
-/

/-- Get a scalar by a multi‑index (list of Nats). -/
def getSpec {α : Type} {s : Shape} (t : Tensor α s) : List Nat → Option α :=
  match t with
  | .scalar value =>
      fun
      | [] => some value
      | _ :: _ => none
  | .dim (n := n) values =>
      fun
      | i :: is =>
          if h : i < n then
            getSpec (values ⟨i, h⟩) is
          else
            none
      | [] => none

@[simp] lemma get_spec_scalar_nil {α : Type} (value : α) :
    getSpec (Tensor.scalar value) [] = some value := rfl

@[simp] lemma get_spec_scalar_cons {α : Type} (value : α) (i : Nat) (is : List Nat) :
    getSpec (Tensor.scalar value) (i :: is) = none := rfl

@[simp] lemma get_spec_dim_nil {α : Type} {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) :
    getSpec (Tensor.dim values) [] = none := by
  simp [getSpec]

@[simp] lemma get_spec_dim_cons {α : Type} {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) (i : Nat) (is : List Nat) :
    getSpec (Tensor.dim values) (i :: is) =
      if h : i < n then getSpec (values ⟨i, h⟩) is else none := by
  simp [getSpec]

attribute [grind =] get_spec_scalar_nil get_spec_scalar_cons get_spec_dim_nil get_spec_dim_cons

-- Get tensor at specific index
/-- Extract the subtensor at index `i` along the outermost dimension. -/
def getAtSpec {α : Type} {n s} (t : Tensor α (.dim n s)) (i : Fin n) : Tensor α s :=
  match t with
  | Tensor.dim f => f i

-- Access subtensor
/-- Alias for `get_at_spec` (the standard spec-level indexing helper). -/
def get {α : Type} {n s} (t : Tensor α (.dim n s)) (i : Fin n) : Tensor α s :=
  getAtSpec t i

/--
Enable Lean’s indexing syntax for spec tensors.

After this instance, you can write `t[i]` as notation for `Spec.get t i`.

We use the domain condition `True` because the index is already a `Fin n`, so it is always
in-bounds by construction.
-/
instance {α : Type} {n : Nat} {s : Shape} :
    GetElem (Tensor α (.dim n s)) (Fin n) (Tensor α s) (fun _ _ => True) where
  getElem t i _ := _root_.Spec.get t i

namespace Tensor

/-- Extract the `i`-th entry from a vector tensor. -/
def vecGet {α : Type} {n : Nat} (x : Tensor α (.dim n .scalar)) (i : Fin n) : α :=
  Tensor.toScalar (_root_.Spec.get x i)

end Tensor

-- Matrix element access
/-- Matrix element access: `get2 A i j` returns `A[i, j]` as a scalar. -/
def get2 {α : Type} {m n : ℕ}
    (A : Tensor α (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) : α :=
  match get A i with
  | Tensor.dim row => match row j with
    | Tensor.scalar v => v

/--
Enable Lean’s indexing syntax for matrix-shaped scalar tensors.

After this instance, you can write `A[(i, j)]` as notation for `Spec.get2 A i j`.
-/
instance {α : Type} {m n : Nat} :
    GetElem (Tensor α (.dim m (.dim n .scalar))) (Fin m × Fin n) α (fun _ _ => True) where
  getElem A ij _ := get2 A ij.1 ij.2

-- Helper to safely get a scalar from a multi-dimensional tensor
/-!
`get_at_or_zero` is a total variant of `get_spec` used in places where a default value is more
convenient than `Option`.

We keep both `get_spec` and `get_at_or_zero` because they serve different roles:
- `get_spec` is better when you want to distinguish "out of bounds" explicitly.
- `get_at_or_zero` is better for formulas that naturally treat out-of-bounds as padding.
-/
/-- Total tensor indexing: returns `0` when the index list is out of bounds. -/
def getAtOrZero {α : Type} [Zero α] {s : Shape} (t : Tensor α s) : List Nat → α :=
  match t with
  | .scalar v =>
      fun
      | [] => v
      | _ :: _ => 0
  | .dim (n := n) f =>
      fun
      | i :: is =>
          if h : i < n then
            getAtOrZero (f ⟨i, h⟩) is
          else
            0
      | [] => 0

@[simp] lemma get_at_or_zero_scalar_nil {α : Type} [Zero α] (v : α) :
    getAtOrZero (Tensor.scalar v) [] = v := rfl

@[simp] lemma get_at_or_zero_scalar_cons {α : Type} [Zero α] (v : α) (i : Nat) (is : List Nat) :
    getAtOrZero (Tensor.scalar v) (i :: is) = 0 := rfl

@[simp] lemma get_at_or_zero_dim_nil {α : Type} [Zero α] {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) :
    getAtOrZero (Tensor.dim values) [] = 0 := by
  simp [getAtOrZero]

@[simp] lemma get_at_or_zero_dim_cons {α : Type} [Zero α] {n : Nat} {s : Shape}
    (values : Fin n → Tensor α s) (i : Nat) (is : List Nat) :
    getAtOrZero (Tensor.dim values) (i :: is) =
      if h : i < n then getAtOrZero (values ⟨i, h⟩) is else 0 := by
  simp [getAtOrZero]

attribute [grind =] get_at_or_zero_scalar_nil get_at_or_zero_scalar_cons get_at_or_zero_dim_nil
  get_at_or_zero_dim_cons

-- Create Fin zero with proof
/-- Construct `Fin n` value `0` given a proof that `0 < n`. -/
def finZero {n : Nat} (h : 0 < n) : Fin n :=
  ⟨0, h⟩

/-- Get the first element of a 1st‑dimension tensor (if nonempty). -/
def getHead {α : Type} {n : Nat} {s : Shape} (t : Tensor α (.dim n s)) : Option (Tensor α s) :=
  if h : 0 < n then
    match t with
    | Tensor.dim f => some (f (finZero h))
  else
    none

/-- Drop the first element of a 1st‑dimension tensor (if nonempty). -/
def getTail {α : Type} {n : Nat} {s : Shape} (t : Tensor α (.dim n s)) : Option (Tensor α (.dim (n
  - 1) s)) :=
  if h : 0 < n then
    match t with
    | Tensor.dim f =>
      some (Tensor.dim (fun i : Fin (n - 1) =>
        let i' : Fin n := cast (by rw [Nat.sub_add_cancel h]) (Fin.succ i)
        f i'))
  else
    none

/-- Cast a tensor along an equality of shapes. -/
def tensorCast {α : Type} {s : Shape} (t : Shape) (h : s = t) : Tensor α s → Tensor α t :=
  fun x => Eq.mp (congrArg (Tensor α) h) x

/-- `tensor_cast` is definitionally `Tensor.cast_shape` (a uniform cast normal form). -/
@[simp] theorem tensor_cast_eq_cast_shape {α : Type} {s t : Shape} (h : s = t) (x : Tensor α s) :
    tensorCast (α := α) (s := s) t h x = Tensor.castShape (t := x) h := by
  rfl

attribute [grind =] tensor_cast_eq_cast_shape

/-- Replicate a scalar tensor to any shape. -/
def replicate {α : Type} : ∀ {s : Shape}, Tensor α .scalar → Tensor α s
  | .scalar, t => t
  | .dim n _, t =>
    Tensor.dim (fun _ : Fin n => replicate t)

/-!
Slicing helpers.

We keep `slice_spec` as a focused "first-axis select" operation since it shows up all over the
place in spec definitions.

PyTorch analogy: `slice_spec t i` is `t[i]`.
-/
/-- Slice a tensor along its first axis: `slice_spec t i = t[i]`. -/
def sliceSpec {α : Type} : ∀ {n s}, Tensor α (.dim n s) → Fin n → Tensor α s
  | _, _, Tensor.dim values, idx => values idx

/--
Slice a contiguous range along the first axis.

This is the spec-level analogue of `t[start : start+len]` in array/tensor libraries.
-/
def sliceRangeSpec {α : Type} {n : Nat} {s : Shape}
  (t : Tensor α (.dim n s)) (start : Nat) (len : Nat) (h : len + start ≤ n) :
  Tensor α (.dim len s) :=
  Tensor.dim (fun i : Fin len =>
    get t (Fin.mk (i.val + start)
                  (Nat.lt_of_lt_of_le (Nat.add_lt_add_right i.isLt start) h)))

-- Helper: collect elements at index `j` from all batch elements.
/-!
`collect_at_index_spec` is a "transpose-like" helper that pulls a fixed position out of every
batch entry.

This is a small but surprisingly useful building block in attention-like code and dataset
manipulations, where you frequently want to reorganize `(batch, n, ...)` into `(n, batch, ...)`
without committing to a concrete memory layout.
-/
/--
Collect the `j`-th element from each batch entry, producing a tensor with batch dimension moved to
  the end.

This is a small "transpose-like" helper used in attention-like code and dataset reshaping.
-/
def collectAtIndexSpec {β : Type} {b n : Nat} {shape : Shape}
      (f : Fin b → Tensor β (.dim n shape)) (j : Fin n) :
      Tensor β (shape.appendDim b) :=
      match shape with
      | .scalar =>
        Tensor.dim fun i =>
          match f i with
          | Tensor.dim g => g j
      | .dim _ _ =>
        Tensor.dim fun k =>
          collectAtIndexSpec
            (fun i =>
              match f i with
              | Tensor.dim g => g j
            ) k

end Spec
