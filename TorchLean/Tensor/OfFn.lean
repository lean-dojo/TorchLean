/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Datatype
import Mathlib.Tactic.Bound.Init

/-!
# Tensor constructors (spec layer)

Small, **total** constructors for `TorchLean.Tensor` values: the constant tensors `Tensor.zeros` and
`Tensor.ones`, tensors from index functions (`Tensor.generate`, `Tensor.ofFn`, `Tensor.matrix`) and
`singleton`, with the lemmas that evaluate them. The constants unfold to `Tensor.full`, defined in
`TorchLean.Tensor.Rep.Pointwise`. The spec layer (models and layers) and the proofs use these
constructors heavily, since they unfold definitionally and need neither `IO` nor dynamic shape
checks. Ordinary literals come from `TorchLean.Tensor.Constructors`, in-memory conversion and
reshape from `TorchLean.Tensor.Conversion`, and scalar casts from `TorchLean.Tensor.Promotion`.

We made them total on purpose. The spec layer handles edge cases in its definitions and never throws
a runtime exception, and Lean rejects a shape-invalid tensor at elaboration time. Existing in-memory
collections enter through the total `Tensor.from` conversion boundary, and external parsers validate
untrusted dimensions before they construct a tensor.
-/

@[expose] public section


namespace TorchLean.Tensor

/-! ## Constant tensors -/

/-- Construct an all-zero tensor of arbitrary shape. Reducible, so lemmas about `full` apply. -/
abbrev zeros {α : Type} [TorchLean.Storage α] [Zero α]
    (shape : Shape) : Tensor α shape :=
  full shape 0

/-- Construct an all-one tensor of arbitrary shape. Reducible, so lemmas about `full` apply. -/
abbrev ones {α : Type} [TorchLean.Storage α] [One α]
    (shape : Shape) : Tensor α shape :=
  full shape 1

/-- Reading a scalar filled tensor returns its fill value. -/
@[simp] theorem item_full_scalar {α : Type} [TorchLean.Storage α] (value : α) :
    (full .scalar value).item = value := by
  simp [Tensor.item]

/-- Filling the scalar shape with the item of a scalar tensor returns that tensor. -/
@[simp] theorem full_item {α : Type} [TorchLean.Storage α] (tensor : Tensor α .scalar) :
    full .scalar tensor.item = tensor := by
  apply Tensor.Rep.ext
  intro coordinate
  cases coordinate
  rw [full_apply]
  rfl

end TorchLean.Tensor

namespace TorchLean.Tensor

/-- Every coordinate of a filled matrix contains its fill value. -/
@[simp] theorem get2_full {α : Type} [Storage α]
    (m n : Nat) (value : α) (i : Fin m) (j : Fin n) :
    get2 (Tensor.full (.dim m (.dim n .scalar)) value) i j = value := by
  simp only [get2, Tensor.unstack_full, Tensor.getScalar, Tensor.item]
  exact Tensor.full_apply .scalar value PUnit.unit

end TorchLean.Tensor

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
  Tensor.Rep.ofFn fun coordinate =>
    f (Shape.Coord.toList shape coordinate)

/--
The vector whose entry `i` is `values i`. The buffer is filled from the index function, without
building one scalar tensor per entry.

PyTorch analogy: `torch.tensor([...])` with shape `(n,)`, with a function in place of the list.

Example:
```lean
-- `torch.tensor([0.0, 1.0, 2.0, 3.0])`, with the entries taken from a function on `Fin n` and the
-- length in the type.
def ramp : Tensor Float [4] := Tensor.ofFn fun index => index.val.toFloat
```
-/
def ofFn {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) : Tensor α [n] :=
  Tensor.Rep.ofFn fun coordinate => values coordinate.1

/-- Evaluating `ofFn` at a coordinate returns the value supplied at its index. -/
@[simp] theorem ofFn_apply {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) (coordinate : Shape.Coord [n]) :
    ofFn values coordinate = values coordinate.1 :=
  Tensor.Rep.get_ofFn _ coordinate

/-- Every entry of `ofFn` is the scalar tensor of the supplied value. -/
@[simp] theorem unstack_ofFn {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) (i : Fin n) :
    Tensor.unstack (ofFn values) i = Tensor.scalar (values i) := by
  apply Tensor.Rep.ext
  intro coordinate
  cases coordinate
  simp

/-- Reading a scalar entry of an internally generated vector evaluates the generator. -/
@[simp] theorem getScalar_rep_ofFn {α : Type} [TorchLean.Storage α] {n : Nat}
    (values : Shape.Coord [n] → α) (i : Fin n) :
    Tensor.getScalar (Tensor.Rep.ofFn values : Tensor α [n]) i =
      values (i, PUnit.unit) := by
  rw [Tensor.getScalar_eq_apply]
  exact Tensor.Rep.get_ofFn values (i, PUnit.unit)

/-- The item of an internally generated scalar tensor is the generator's value. -/
@[simp] theorem item_rep_ofFn {α : Type} [TorchLean.Storage α]
    (values : Shape.Coord .scalar → α) :
    Tensor.item (Tensor.Rep.ofFn values : Tensor α .scalar) =
      values PUnit.unit :=
  Tensor.Rep.get_ofFn values PUnit.unit

/-- Slicing an internally generated tensor fixes the leading coordinate of the generator. -/
@[simp] theorem unstack_rep_ofFn {α : Type} [TorchLean.Storage α] {n : Nat} {shape : Shape}
    (values : Shape.Coord (.dim n shape) → α) (i : Fin n) :
    Tensor.unstack (Tensor.Rep.ofFn values : Tensor α (.dim n shape)) i =
      Tensor.Rep.ofFn fun coordinate => values (i, coordinate) := by
  apply Tensor.Rep.ext
  intro coordinate
  simp

/-- `ofFn values` is the stack of the scalar tensors `Tensor.scalar (values i)`. -/
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
  simp [Tensor.getScalar]

/-- Rebuilding a vector from all of its coordinates returns the original vector. -/
@[simp] theorem ofFn_getScalar {α : Type} [TorchLean.Storage α]
    {n : Nat} (t : Tensor α [n]) :
    ofFn (fun i => t.getScalar i) = t := by
  apply Tensor.ext_vector
  intro i
  simp

/-- The `m × n` matrix whose entry at row `i` and column `j` is `values i j`.

PyTorch analogy: `torch.tensor([...]).reshape(m, n)`, with a function in place of the list.
-/
def matrix {α : Type} [TorchLean.Storage α]
    {m n : Nat} (values : Fin m → Fin n → α) :
    Tensor α [m, n] :=
  Tensor.dim (fun i => ofFn (fun j => values i j))

end TorchLean.Tensor

namespace TorchLean.Tensor

/-- Every coordinate of a filled vector contains the fill value. -/
@[simp] theorem getScalar_full {α : Type} [TorchLean.Storage α]
    (n : Nat) (value : α) (i : Fin n) :
    (full (.dim n .scalar) value).getScalar i = value := by
  simp only [Tensor.getScalar, Tensor.unstack_full, Tensor.item]
  exact Tensor.full_apply .scalar value PUnit.unit

end TorchLean.Tensor

namespace TorchLean.Tensor

/-- The vector of length one holding `x`.

PyTorch analogy: `x.unsqueeze(0)` for a scalar `x`.
-/
def singleton {α : Type} [Storage α] (x : α) : Tensor α [1] :=
  Tensor.dim (fun _ => Tensor.scalar x)

end TorchLean.Tensor

namespace TorchLean.Tensor

/-- A filled tensor satisfies every pointwise property satisfied by its value. -/
theorem forall_full {α : Type} [TorchLean.Storage α]
    {p : α → Prop} {s : Shape} {x : α}
    (hx : p x) : Tensor.Forall p (full s x) := by
  rw [Tensor.forall_iff]
  intro coordinate
  rw [full_apply]
  exact hx

end TorchLean.Tensor
