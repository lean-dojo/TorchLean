/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module


/-!
# `TensorArray`: a simple array-backed tensor representation

`Spec.Tensor` is the canonical, shape-indexed tensor type for the spec layer. It is great for
proofs and pure definitions, but it is not always the most convenient representation for:

- IO (CSV/JSON/NPY),
- interop with external tooling,
- or performance-sensitive evaluation where you want explicit arrays.

`TensorArray.Tensor` is a lightweight alternative:

- the shape is a runtime `List Nat`,
- the data is a flat `Array α`,
- and a proof field (`shape_valid`) records that the array length matches the shape product.

The bridge back to `Spec.Tensor` lives in `NN/Spec/Core/TensorBridge.lean`.

Why this representation exists:

- Most "paper definitions" are easiest with `Spec.Tensor` because shape correctness is enforced by
  types and proofs follow the inductive structure.
- Most integration tasks (file IO, external tools, or simple numerics) are easier with a flat
  array plus a runtime shape.
- Having both makes the trust boundaries explicit: array-backed tensors are convenient, while the
  shape-indexed tensors are the place we do most proofs.
-/

@[expose] public section


namespace TensorArray

/--
  A tensor is represented as a flat array of elements and a shape (as a list of dimensions).
  The shape_valid proof ensures the array size matches the product of the shape dimensions.
-/
structure Tensor (α : Type) (shape : List Nat) where
  /-- Flat row-major data buffer. -/
  data : Array α
  /-- Proof that the buffer length matches the product of the runtime dimensions. -/
  shape_valid : data.size = shape.foldl (· * ·) 1

/--
Product of dimensions for a runtime shape list.

This is the runtime analogue of `Spec.Shape.size` for `Spec.Shape`.
We keep it as a `def` (not just a local `let`) because it appears everywhere
`shape_valid` is constructed or rewritten.
-/
def shapeProd (shape : List Nat) : Nat := shape.foldl (· * ·) 1

/--
Build a tensor from an array when you already have a size proof.

Design choice:
- `Tensor` stores the shape at the type level (`Tensor α shape`), so callers must provide
  `h : arr.size = shapeProd shape`.
- This makes "I reshaped / reinterpreted the data" a conscious action with an explicit proof.
-/
def ofArray {α : Type} (arr : Array α) (shape : List Nat) (h : arr.size = shapeProd shape) : Tensor
  α shape :=
  { data := arr, shape_valid := h }

/-- Base case for `shapeProd`: the empty shape has product `1`. -/
@[simp]
theorem shapeProd_nil : shapeProd [] = 1 := rfl

/--
Helper lemma: factoring a left-multiplication out of the `foldl` product.

This is used to prove `shapeProd_cons` and similar "shape product algebra" facts.
-/
theorem foldl_mul_factor (n : Nat) (ns : List Nat) :
  List.foldl (fun x1 x2 ↦ x1 * x2) n ns = n * List.foldl (fun x1 x2 ↦ x1 * x2) 1 ns := by
  induction ns generalizing n with
  | nil =>
    simp [List.foldl]
  | cons head tail ih =>
    simp [List.foldl]
    rw [ih]
    grind

/-- Step case for `shapeProd`: product of `(n :: ns)` is `n * shapeProd ns`. -/
@[simp]
theorem shapeProd_cons (n : Nat) (ns : List Nat) :
  shapeProd (n :: ns) = n * shapeProd ns := by
  unfold shapeProd
  rw [List.foldl_cons]
  simpa using (foldl_mul_factor n ns)

-- Tell `grind` about the standard runtime-shape "numel algebra" rules.
attribute [grind =] shapeProd_nil shapeProd_cons

-- Helper lemmas for list/array length calculations used in proofs below.
/--
Length of `List.zipWith` is the minimum of the input lengths.

We keep these small list lemmas local to this file because they are only needed to justify
`shape_valid` proofs for array operations like `zipWith`.
-/
theorem List.length_zipWith {α β γ : Type} (f : α → β → γ) (l1 : List α) (l2 : List β) :
  (List.zipWith f l1 l2).length = min l1.length l2.length := by
  simp

/-- Length of a `map` is the same as the input length. -/
theorem List.length_map {α β : Type} (f : α → β) (l : List α) :
  (l.map f).length = l.length := by
  simp

/--
Length of `flatten` is the sum of lengths of each inner list.

This kind of fact comes up any time we build an `Array`/`List` by flattening a list-of-lists.
-/
theorem List.length_flatten {α : Type} (ll : List (List α)) :
  ll.flatten.length = (ll.map List.length).sum := by
  induction ll with
  | nil => simp [List.flatten]
  | cons head tail ih => simp [List.flatten, List.sum_cons, ih]

/-- `Array.toList` preserves size as `List.length`. -/
theorem Array.size_toList {α : Type} (a : Array α) : a.toList.length = a.size := by
  cases a
  simp [Array.size]

-- These are small list/array bookkeeping lemmas that `grind` can use as rewrite rules.
attribute [grind =] List.length_zipWith List.length_map List.length_flatten Array.size_toList

/-- Compute the flat index for a given multi-index.

Returns `none` if the indices are out of bounds or the rank mismatches.
-/
def flatIndexAux : List Nat → List Nat → Nat → Option Nat
  | [], [], acc => some acc
  | d :: ds, i :: is, acc =>
    if i < d then
      flatIndexAux ds is (acc * d + i)
    else
      none
  | _, _, _ => none

def flatIndex (shape : List Nat) (indices : List Nat) : Option Nat :=
  flatIndexAux shape indices 0

/--
`flatIndexAux` returns an index that is bounded by the "mixed-radix" size implied by the
remaining `shape`.

Intuition: starting with accumulator `acc`, the recursion computes something of the form
`acc * shapeProd shape + tail`, where `tail < shapeProd shape`.
-/
theorem flatIndexAux_lt (shape indices : List Nat) (acc idx : Nat) :
  flatIndexAux shape indices acc = some idx → idx < (acc + 1) * shapeProd shape := by
  induction shape generalizing indices acc idx with
  | nil =>
    cases indices with
    | nil =>
      intro h
      -- idx = acc and (acc + 1) * shapeProd [] = acc + 1
      cases h
      simp
    | cons _ _ =>
      intro h
      simp [flatIndexAux] at h
  | cons d ds ih =>
    cases indices with
    | nil =>
      intro h
      simp [flatIndexAux] at h
    | cons i is =>
      intro h
      by_cases hi : i < d
      · have hrec : flatIndexAux ds is (acc * d + i) = some idx := by
          simpa [flatIndexAux, hi] using h
        have hlt : idx < ((acc * d + i) + 1) * shapeProd ds :=
          ih (indices := is) (acc := acc * d + i) (idx := idx) hrec
        have hle : acc * d + i + 1 ≤ (acc + 1) * d := by
          calc
            acc * d + i + 1 = acc * d + (i + 1) := by simp [Nat.add_assoc]
            _ ≤ acc * d + d := Nat.add_le_add_left (Nat.succ_le_of_lt hi) (acc * d)
            _ = (acc + 1) * d := by
              -- (acc + 1) * d = acc * d + d
              simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
                (Eq.symm (Nat.succ_mul acc d))
        have hle' : (acc * d + i + 1) * shapeProd ds ≤ ((acc + 1) * d) * shapeProd ds :=
          Nat.mul_le_mul_right _ hle
        have : idx < ((acc + 1) * d) * shapeProd ds :=
          Nat.lt_of_lt_of_le hlt hle'
        simpa [shapeProd_cons, Nat.mul_assoc] using this
      · simp [flatIndexAux, hi] at h

/-- If `flatIndex` succeeds, the resulting index is in-bounds for the flattened tensor. -/
theorem flatIndex_lt_shapeProd (shape indices : List Nat) (idx : Nat) :
  flatIndex shape indices = some idx → idx < shapeProd shape := by
  intro h
  have : idx < (0 + 1) * shapeProd shape :=
    flatIndexAux_lt shape indices 0 idx (by simpa [flatIndex] using h)
  simpa using this

/--
Get an element at the given multi-index.

Returns `none` if:
- the rank mismatches, or
- any index is out of bounds.

This is the array-backed analogue of `Spec.get_spec`.
-/
def get? {α : Type} {shape : List Nat} [Inhabited α] (t : Tensor α shape) (indices : List Nat) :
  Option α :=
  match h : flatIndex shape indices with
  | some idx =>
    have hlt : idx < shapeProd shape := flatIndex_lt_shapeProd shape indices idx h
    have hsize : t.data.size = shapeProd shape := by
      -- `shapeProd` is definitional equal to `List.foldl (· * ·) 1`.
      simpa [shapeProd] using t.shape_valid
    have hlt' : idx < t.data.size := Nat.lt_of_lt_of_eq hlt hsize.symm
    some (t.data[idx]'hlt')
  | none => none

/--
Map a function over all elements (shape preserved).

The only subtlety is the `shape_valid` proof: mapping doesn't change array length.
-/
def map {α β : Type} {shape : List Nat} (f : α → β) (t : Tensor α shape) : Tensor β shape :=
  { data := t.data.map f
    shape_valid := by
      simp [Array.size_map]   -- (Array.map f t.data).size = t.data.size
      rw [t.shape_valid] }

/--
Elementwise binary operation (shape preserved).

We require both tensors to have the same `shape` at the type level, so shape mismatches are
unrepresentable here.
-/
def map2 {α β γ : Type} {shape : List Nat}
  (f : α → β → γ) (t₁ : Tensor α shape) (t₂ : Tensor β shape) : Tensor γ shape :=
  { data := Array.zipWith f t₁.data t₂.data
    shape_valid := by
      simp [Array.size_zipWith]
      rw [t₁.shape_valid, t₂.shape_valid]
      exact Nat.min_self _ }

/--
Reduce by summing all elements (flattened).

This ignores the tensor's rank and sums over `data` directly.
PyTorch analogy: `t.sum()` (over all axes).
-/
def sum {α : Type} [Add α] [Zero α] {shape : List Nat} (t : Tensor α shape) : α :=
  t.data.toList.foldl (· + ·) 0

/--
Reshape a tensor to a new shape with the same number of elements.

This is "view"-style: it reuses the same underlying `data` array.
The proof `h` is the only thing that changes.
-/
def reshape {α : Type} {shape1 shape2 : List Nat}
  (t : Tensor α shape1) (h : shapeProd shape1 = shapeProd shape2) : Tensor α shape2 :=
  { data := t.data
    shape_valid := by
      exact Eq.trans t.shape_valid h }

/--
Create a tensor filled with a constant value.

PyTorch analogy: `torch.full(shape, val)`.
-/
def full {α : Type} (shape : List Nat) (val : α) : Tensor α shape :=
  { data := Array.replicate (shape.foldl (· * ·) 1) val
    shape_valid := by simp [Array.size_replicate] }

/-- Elementwise addition. -/
def add {α : Type} [Add α] {shape : List Nat} (t₁ t₂ : Tensor α shape) : Tensor α shape :=
  map2 (· + ·) t₁ t₂

/-- Elementwise multiplication. -/
def mul {α : Type} [Mul α] {shape : List Nat} (t₁ t₂ : Tensor α shape) : Tensor α shape :=
  map2 (· * ·) t₁ t₂

/--
ReLU activation (elementwise max with zero).

This is written in the simplest "executable" style: a branch on `x > 0`.
For interval/real semantics, use `Spec` layer ops; this module is for array-backed computations.
-/
def relu {α : Type} [LT α] [Zero α] [DecidableLT α] {shape : List Nat} (t : Tensor α shape) : Tensor
  α shape :=
  map (fun x => if x > 0 then x else 0) t

/--
Matrix-vector multiplication: (m x n) matrix times (n) vector gives (m) vector.

This is a direct reference implementation intended for small sizes and clarity.
If you need performance, you generally want the runtime/TorchLean path instead.
-/
def matvec {α : Type} [Add α] [Mul α] [Zero α] [Inhabited α]
  {m n : Nat} (mat : Tensor α [m, n]) (vec : Tensor α [n]) : Tensor α [m] :=
  let rows : List (Fin m) := List.finRange m
  let matSize : Nat := m * n
  have hMatSize : mat.data.size = matSize := by
    -- Normalize `shape_valid` to `shapeProd` and then compute `shapeProd [m, n] = m * n`.
    have : mat.data.size = shapeProd ([m, n] : List Nat) := by
      simpa [shapeProd] using mat.shape_valid
    simpa [matSize] using this
  let getRow (i : Fin m) : List α :=
    (List.finRange n).map (fun j : Fin n =>
      let idx : Nat := i.1 * n + j.1
      have hIdx : idx < mat.data.size := by
        have h1 : idx < i.1 * n + n := Nat.add_lt_add_left j.2 (i.1 * n)
        have h2 : i.1 * n + n = (i.1 + 1) * n := by
          simpa [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using
            (Eq.symm (Nat.succ_mul i.1 n))
        have h3 : (i.1 + 1) * n ≤ m * n := Nat.mul_le_mul_right n (Nat.succ_le_of_lt i.2)
        have h4 : idx < (i.1 + 1) * n := by simpa [h2] using h1
        have h5 : idx < m * n := Nat.lt_of_lt_of_le h4 h3
        exact Nat.lt_of_lt_of_eq h5 hMatSize.symm
      mat.data[idx]'hIdx)
  let result : List α :=
    rows.map (fun i =>
      (getRow i).zip vec.data.toList |>.foldl (fun acc (a, b) => acc + a * b) 0)
  { data := ⟨result⟩
    shape_valid := by
      -- (Array.mk result).size = result.length; length of a map over `rows`
      -- is `rows.length = m`; and shapeProd [m] = m.
      simp [rows, result] }

/--
Linear layer: `y = W x + b`.

PyTorch analogy: `torch.nn.Linear(n, m)` forward pass with weight `W` and bias `b`.
-/
def linear {α : Type} [Add α] [Mul α] [Zero α] [Inhabited α]
  {m n : Nat} (W : Tensor α [m, n]) (b : Tensor α [m]) (x : Tensor α [n]) : Tensor α [m] :=
  add (matvec W x) b

end TensorArray
