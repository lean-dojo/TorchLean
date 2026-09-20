/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core
import Mathlib.Tactic.Bound.Init

/-!
# Tensor constructors (spec layer)

These are small, **total** constructors for building `TorchLean.Tensor` values directly.

They are used heavily inside the spec layer (models/layers) and in proofs, where we want:

- straightforward definitional unfolding, and
- no dependence on `IO` or dynamic shape checks.

For ordinary literals, in-memory conversion, reshape, and scalar casts, import `NN.Tensor`.

Design choice (why these are "total"):

- In the spec layer we would rather make edge cases explicit than throw runtime exceptions.
- If something is shape-invalid, we want Lean to reject it at elaboration time.
- Existing in-memory collections enter through the total `Tensor.from` conversion boundary.
- External parsers validate untrusted dimensions before constructing a tensor.
-/

@[expose] public section


open TorchLean

open Spec TorchLean

namespace TorchLean.Tensor

/-! ## Constant tensors -/

/-- Fill a tensor of arbitrary shape with one value.

PyTorch analogy: `torch.full(shape, value)`.
-/
def full {α : Type} [TorchLean.Storage α]
    (shape : Shape) (value : α) : Tensor α shape :=
  TorchLean.Tensor.Internal.Rep.const value

/-- Construct an all-zero tensor of arbitrary shape. Reducible, so lemmas about `full` apply. -/
abbrev zeros {α : Type} [TorchLean.Storage α] [Zero α]
    (shape : Shape) : Tensor α shape :=
  full shape 0

/-- Construct an all-one tensor of arbitrary shape. Reducible, so lemmas about `full` apply. -/
abbrev ones {α : Type} [TorchLean.Storage α] [One α]
    (shape : Shape) : Tensor α shape :=
  full shape 1

/-- Every coordinate of a filled tensor contains its fill value. -/
@[simp] theorem full_apply {α : Type} [TorchLean.Storage α]
    (shape : Shape) (value : α) (coordinate : shape.Coord) :
    full shape value coordinate = value := by
  exact TorchLean.Tensor.Internal.Rep.const_apply value coordinate

/-- Reading a scalar filled tensor returns its fill value. -/
@[simp] theorem item_full_scalar {α : Type} [TorchLean.Storage α] (value : α) :
    (full .scalar value).item = value := by
  simp [Tensor.item]

/-- The item of an internally constant scalar tensor is the constant. -/
@[simp] theorem item_rep_const {α : Type} [TorchLean.Storage α] (value : α) :
    Tensor.item (TorchLean.Tensor.Internal.Rep.const value : Tensor α .scalar) = value :=
  TorchLean.Tensor.Internal.Rep.const_apply value PUnit.unit

/-- Every entry of an internally constant vector is the constant. -/
@[simp] theorem getScalar_rep_const {α : Type} [TorchLean.Storage α] {n : Nat}
    (value : α) (i : Fin n) :
    Tensor.getScalar (TorchLean.Tensor.Internal.Rep.const value : Tensor α [n]) i = value := by
  rw [Tensor.getScalar_eq_apply]
  exact TorchLean.Tensor.Internal.Rep.const_apply (s := [n]) value (i, PUnit.unit)

/-- Replicating a scalar tensor observes that scalar at every target coordinate. -/
@[simp] theorem replicate_scalar_apply {α : Type} [TorchLean.Storage α]
    (value : α) (shape : Shape) (coordinate : shape.Coord) :
    replicate (shape := shape) (Tensor.scalar value) coordinate = value := by
  cases shape <;> simp [replicate]

end TorchLean.Tensor

namespace Spec

/-- Every outer coordinate of a filled tensor is the corresponding filled subtensor. -/
@[simp] theorem get_full {α : Type} [TorchLean.Storage α]
    (n : Nat) (s : Shape) (value : α) (i : Fin n) :
    get (TorchLean.Tensor.full (.dim n s) value) i = TorchLean.Tensor.full s value := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [get, Tensor.unstack, TorchLean.Tensor.full]

/-- Every coordinate of a filled matrix contains its fill value. -/
@[simp] theorem get2_full {α : Type} [TorchLean.Storage α]
    (m n : Nat) (value : α) (i : Fin m) (j : Fin n) :
    get2 (TorchLean.Tensor.full (.dim m (.dim n .scalar)) value) i j = value := by
  simp only [get2, get_full, Tensor.getScalar, Tensor.item]
  exact Tensor.full_apply .scalar value PUnit.unit

end Spec

open Spec TorchLean

namespace TorchLean.Tensor

/--
Construct an arbitrary-rank tensor from a coordinate function.

At each coordinate, `f` receives one natural-number index per dimension,
outermost first. The indices are in bounds by construction.

For example, `Tensor.generate [2, 3] f` has shape `[2, 3]`, and its entry at row `i` and column `j`
is `f [i, j]`.
-/
def generate {α : Type} [TorchLean.Storage α]
    (shape : Shape) (f : List Nat → α) : Tensor α shape :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate =>
    f (Shape.Coord.toList shape coordinate)

/--
The buffer is filled directly from the index function, without building one scalar tensor per
entry.

PyTorch analogy: `torch.tensor([...])` with shape `(n,)`, but our input is a function, not a list.

Example:
```lean
-- `torch.tensor([0.0, 1.0, 2.0, 3.0])`, except the entries arrive from a function on `Fin n` and
-- the length is part of the type.
def ramp : Tensor Float [4] := Tensor.ofFn fun index => index.val.toFloat
```
-/
def ofFn {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) : Tensor α [n] :=
  TorchLean.Tensor.Internal.Rep.ofFn fun coordinate => values coordinate.1

/-- Evaluating `ofFn` at a coordinate returns the value supplied at its index. -/
@[simp] theorem ofFn_apply {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) (coordinate : Shape.Coord [n]) :
    ofFn values coordinate = values coordinate.1 :=
  TorchLean.Tensor.Internal.Rep.get_ofFn _ coordinate

/-- Every entry of `ofFn` is the scalar tensor of the supplied value. -/
@[simp] theorem unstack_ofFn {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) (i : Fin n) :
    Tensor.unstack (ofFn values) i = Tensor.scalar (values i) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  cases coordinate
  simp [Tensor.unstack]

/-- Reading a scalar entry of an internally generated vector evaluates the generator. -/
@[simp] theorem getScalar_rep_ofFn {α : Type} [TorchLean.Storage α] {n : Nat}
    (values : Shape.Coord [n] → α) (i : Fin n) :
    Tensor.getScalar (TorchLean.Tensor.Internal.Rep.ofFn values : Tensor α [n]) i =
      values (i, PUnit.unit) := by
  rw [Tensor.getScalar_eq_apply]
  exact TorchLean.Tensor.Internal.Rep.get_ofFn values (i, PUnit.unit)

/-- The item of an internally generated scalar tensor is the generator's value. -/
@[simp] theorem item_rep_ofFn {α : Type} [TorchLean.Storage α]
    (values : Shape.Coord .scalar → α) :
    Tensor.item (TorchLean.Tensor.Internal.Rep.ofFn values : Tensor α .scalar) =
      values PUnit.unit :=
  TorchLean.Tensor.Internal.Rep.get_ofFn values PUnit.unit

/-- Slicing an internally generated tensor fixes the leading coordinate of the generator. -/
@[simp] theorem unstack_rep_ofFn {α : Type} [TorchLean.Storage α] {n : Nat} {shape : Shape}
    (values : Shape.Coord (.dim n shape) → α) (i : Fin n) :
    Tensor.unstack (TorchLean.Tensor.Internal.Rep.ofFn values : Tensor α (.dim n shape)) i =
      TorchLean.Tensor.Internal.Rep.ofFn fun coordinate => values (i, coordinate) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp [Tensor.unstack]

/-- `ofFn` is extensionally the stack of its scalar entries. -/
theorem ofFn_eq_dim_scalar {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) :
    ofFn values = Tensor.dim (fun i => Tensor.scalar (values i)) := by
  rw [← Tensor.dim_unstack (ofFn values)]
  congr 1
  funext i
  exact unstack_ofFn values i

/-- Reading a coordinate from `ofFn` returns the value supplied at that coordinate. -/
@[simp] theorem getScalar_ofFn {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) (i : Fin n) :
    (ofFn values).getScalar i = values i := by
  simp [Tensor.getScalar, Spec.get]

/-- Rebuilding a vector from all of its coordinates returns the original vector. -/
@[simp] theorem ofFn_getScalar {α : Type} [TorchLean.Storage α]
    {n : Nat} (t : Tensor α [n]) :
    ofFn (fun i => t.getScalar i) = t := by
  apply Tensor.ext_vector
  intro i
  simp

/-- Construct a matrix from its row and column coordinate function.

PyTorch analogy: `torch.tensor([...]).reshape(m, n)` (again, function input rather than a list).
-/
def matrix {α : Type} [TorchLean.Storage α]
    {m n : Nat} (values : Fin m → Fin n → α) :
    Tensor α [m, n] :=
  Tensor.dim (fun i => ofFn (fun j => values i j))

end TorchLean.Tensor

open Spec TorchLean

namespace TorchLean.Tensor

/-- Every coordinate of a filled vector contains the fill value. -/
@[simp] theorem getScalar_full {α : Type} [TorchLean.Storage α]
    (n : Nat) (value : α) (i : Fin n) :
    (full (.dim n .scalar) value).getScalar i = value := by
  simp only [Tensor.getScalar, get_full, Tensor.item]
  exact Tensor.full_apply .scalar value PUnit.unit

end TorchLean.Tensor

namespace Spec

/-- A singleton vector.

PyTorch analogy: `x.unsqueeze(0)` for a scalar `x`.
-/
def singleton {α : Type} [TorchLean.Storage α] (x : α) : Tensor α [1] :=
  Tensor.dim (fun _ => Tensor.scalar x)

/--
Pad a tensor with `n` leading dimensions of size 1.

This is the tensor-level companion of `Shape.padLeft`. The row-major buffer is unchanged, so the
padding is a zero-copy reinterpretation of the static shape. Broadcasting uses it to align ranks
before expanding singleton axes.

PyTorch analogy: repeated `unsqueeze(0)`.
-/
def padLeft {α : Type} [TorchLean.Storage α]
    {n : Nat} {s : Shape} (x : Tensor α s) : Tensor α (Shape.padLeft n s) :=
  TorchLean.Tensor.Internal.Rep.reshape (by simp only [Shape.internalSize_eq, Shape.size_padLeft]) x

/-- Padding with zero axes is the identity. -/
@[simp] theorem padLeft_zero {α : Type} [TorchLean.Storage α]
    {s : Shape} (x : Tensor α s) : padLeft (n := 0) x = x :=
  TorchLean.Tensor.Internal.Rep.reshape_rfl x

/-- Padding one more axis stacks the padded tensor along a new singleton axis. -/
theorem padLeft_succ {α : Type} [TorchLean.Storage α]
    {n : Nat} {s : Shape} (x : Tensor α s) :
    padLeft (n := n + 1) x = Tensor.dim fun _ => padLeft (n := n) x := by
  have h₁ : TorchLean.Tensor.Internal.Shape.size s.toList =
      TorchLean.Tensor.Internal.Shape.size (Shape.padLeft n s).toList := by
    simp only [Shape.internalSize_eq, Shape.size_padLeft]
  have h₂ : TorchLean.Tensor.Internal.Shape.size (Shape.padLeft n s).toList =
      TorchLean.Tensor.Internal.Shape.size (1 :: (Shape.padLeft n s).toList) := by
    simp
  exact TorchLean.Tensor.Internal.Rep.reshape_one_cons h₂
    (TorchLean.Tensor.Internal.Rep.reshape h₁ x)

end Spec

namespace TorchLean.Tensor

/-- Stack an array of equal-shaped tensors along a new leading dimension.

The explicit size proof prevents silent truncation or padding. Taking tensors as array elements
makes this constructor independent of rank: use scalar tensors for a vector, vectors for a matrix,
or arbitrary inner tensors for higher-rank values.
-/
def stackArray {α : Type} [TorchLean.Storage α] {n : Nat} {s : Shape}
    (xs : Array (Tensor α s)) (_h : n = xs.size) : Tensor α (.dim n s) :=
  Tensor.dim (fun i : Fin n =>
    xs[i.val]'(by simpa [_h] using i.2))

end TorchLean.Tensor

open Spec TorchLean

namespace TorchLean.Tensor

/-- A filled tensor satisfies every pointwise property satisfied by its value. -/
theorem forall_full {α : Type} [TorchLean.Storage α]
    {p : α → Prop} {s : Shape} {x : α}
    (hx : p x) : Tensor.Forall p (full s x) := by
  induction s with
  | scalar =>
      change p ((full .scalar x).item)
      change p (full .scalar x PUnit.unit)
      rw [Tensor.full_apply]
      exact hx
  | dim _ _ ih =>
      intro index
      change Tensor.Forall p (get (full _ x) index)
      rw [get_full]
      exact ih

end TorchLean.Tensor

namespace Spec

/-- Build a matrix when every row has the same length; reject ragged input. -/
def matrixFromRows? {α : Type} [TorchLean.Storage α]
    (rows : List (List α)) :
    Option (Tensor α [rows.length, Option.getD (rows.head?.map List.length) 0]) :=
  match rows with
  | [] => some (Tensor.dim fun i => nomatch i)
  | first :: rest =>
      let allRows := first :: rest
      let columnCount := first.length
      if hRectangular : ∀ row ∈ allRows, row.length = columnCount then
        some <| Tensor.dim fun i =>
          let row := allRows.get i
          have hLength : row.length = columnCount :=
            hRectangular row (List.get_mem allRows i)
          Tensor.ofFn fun j =>
            have hIndex : j.val < row.length := by simpa [hLength] using j.isLt
            row.get ⟨j.val, hIndex⟩
      else
        none

end Spec
