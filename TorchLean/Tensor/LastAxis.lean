/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.OfFn

@[expose] public section


open TorchLean.Shape

/-!
# Last-axis rows

Each coordinate of a tensor lies on one last-axis row, along which only its last index varies, and
a scalar is a row with one entry. A shape-polymorphic map that acts on the slices of a nonempty
outer axis separately is read at a coordinate of a nonscalar shape as its value on the row through
that coordinate (`Tensor.rowwise_apply_of_ne`). `Tensor.rowwise_apply` covers scalars too, for a
map that is a constant `value` on them such as the last-axis softmax, and `Tensor.lastAxisMap`
applies a vector map to every row.
-/

namespace TorchLean.Shape

/-- The extent of the last axis, `1` for a scalar. -/
abbrev rowWidth : Shape → Nat
  | .scalar => 1
  | .dim n .scalar => n
  | .dim _ (.dim n s) => rowWidth (.dim n s)

/-- The last index of a coordinate, `0` for a scalar. -/
def Coord.getLast : {s : Shape} → Shape.Coord s → Fin (rowWidth s)
  | .scalar, _ => (0 : Fin 1)
  | .dim _ .scalar, p => p.1
  | .dim _ (.dim n s), p => getLast (s := .dim n s) p.2

/-- Replace the last index of a coordinate, keeping its leading indices. -/
def Coord.updateLast : {s : Shape} → Shape.Coord s → Fin (rowWidth s) → Shape.Coord s
  | .scalar, _, _ => PUnit.unit
  | .dim _ .scalar, _, j => (j, PUnit.unit)
  | .dim _ (.dim _ _), p, j => (p.1, updateLast p.2 j)

/-- Moving along a row to index `j` gives a coordinate with last index `j`. -/
@[simp] theorem Coord.getLast_updateLast :
    {s : Shape} → (p : Shape.Coord s) → (j : Fin (rowWidth s)) →
      getLast (updateLast p j) = j
  | .scalar, _, j => (Fin.fin_one_eq_zero j).symm
  | .dim _ .scalar, _, _ => rfl
  | .dim _ (.dim n s), p, j => getLast_updateLast (s := .dim n s) p.2 j

/-- Moving along a row to its own last index stays put. -/
@[simp] theorem Coord.updateLast_getLast :
    {s : Shape} → (p : Shape.Coord s) → updateLast p (getLast p) = p
  | .scalar, _ => rfl
  | .dim _ .scalar, _ => rfl
  | .dim _ (.dim n s), p => congrArg (p.1, ·) (updateLast_getLast (s := .dim n s) p.2)

/-- The last axis of `s ++ [n]` has extent `n`. -/
theorem rowWidth_append_singleton (s : Shape) (n : Nat) : rowWidth (s ++ [n]) = n := by
  induction s with
  | scalar => rfl
  | dim m s ih => cases s <;> simp_all [rowWidth]

/-- Moving along a row twice is moving once. -/
@[simp] theorem Coord.updateLast_updateLast :
    {s : Shape} → (p : Shape.Coord s) → (j k : Fin (rowWidth s)) →
      updateLast (updateLast p j) k = updateLast p k
  | .scalar, _, _, _ => rfl
  | .dim _ .scalar, _, _, _ => rfl
  | .dim _ (.dim n s), p, j, k => by
      simp only [updateLast, updateLast_updateLast (s := .dim n s) p.2 j k]

end TorchLean.Shape

namespace TorchLean.Tensor

variable {α : Type} [Storage α]

/-- The last-axis row of `tensor` through the coordinate `p`. -/
def row {s : Shape} (tensor : Tensor α s) (p : s.Coord) : Tensor α [s.rowWidth] :=
  ofFn fun j => tensor (Shape.Coord.updateLast p j)

/-- Reading a row reads the tensor at the coordinate moved along the row. -/
theorem row_apply {s : Shape} (tensor : Tensor α s) (p : s.Coord) (c : Shape.Coord [s.rowWidth]) :
    tensor.row p c = tensor (Shape.Coord.updateLast p c.1) :=
  ofFn_apply _ c

/-- The row through any coordinate of a row is that row. -/
@[simp] theorem row_updateLast {s : Shape} (tensor : Tensor α s) (p : s.Coord)
    (j : Fin s.rowWidth) : tensor.row (Shape.Coord.updateLast p j) = tensor.row p := by
  simp only [row, Shape.Coord.updateLast_updateLast]

/-- The row of a vector is the vector. -/
theorem row_vector {n : Nat} (tensor : Tensor α [n]) (p : Shape.Coord [n]) :
    tensor.row p = tensor :=
  Rep.ext fun _ => ofFn_apply _ _

/-- The row of a tensor through `(i, p)` is the row of its slice `i` through `p`. -/
theorem row_mk {m n : Nat} {s : Shape} (tensor : Tensor α (.dim m (.dim n s))) (i : Fin m)
    (p : Shape.Coord (.dim n s)) :
    tensor.row (i, p) = (tensor.unstack i).row p :=
  Rep.ext fun j => by
    simp only [row, ofFn_apply, Shape.Coord.updateLast, Tensor.unstack_apply]

/-- A shape-polymorphic map of two tensors that acts on the slices of a nonempty outer axis
separately is read at a coordinate of a nonscalar shape as its value on the last-axis rows through
it. -/
theorem rowwise_apply_of_ne {β : Type} [Storage β]
    (map : ∀ {s : Shape}, Tensor α s → Tensor β s → Tensor α s)
    (outer : ∀ {m n : Nat} {s : Shape} (first : Tensor α (.dim (m + 1) (.dim n s)))
      (second : Tensor β (.dim (m + 1) (.dim n s))),
      map first second = Tensor.dim fun i => map (first.unstack i) (second.unstack i))
    {s : Shape} (hs : s ≠ .scalar) (first : Tensor α s) (second : Tensor β s) (p : s.Coord) :
    map first second p = (map (first.row p) (second.row p)).getScalar (Shape.Coord.getLast p) := by
  induction s with
  | scalar => exact absurd rfl hs
  | dim m s ih =>
      cases s with
      | scalar =>
          rw [row_vector, row_vector, getScalar_eq_apply]
          rfl
      | dim n s =>
          cases m with
          | zero => exact p.1.elim0
          | succ m =>
              rw [outer, Tensor.dim_apply, ih (List.cons_ne_nil _ _), row_mk, row_mk]
              rfl

/-- A shape-polymorphic map of two tensors that is `value` on scalars and acts on the slices of a
nonempty outer axis separately is read at a coordinate as its value on the last-axis rows through
it. A map of one tensor is the case that ignores its second argument. -/
theorem rowwise_apply (map : ∀ {s : Shape}, Tensor α s → Tensor α s → Tensor α s) (value : α)
    (scalar : ∀ first second : Tensor α .scalar, map first second = Tensor.scalar value)
    (outer : ∀ {m n : Nat} {s : Shape} (first second : Tensor α (.dim (m + 1) (.dim n s))),
      map first second = Tensor.dim fun i => map (first.unstack i) (second.unstack i))
    {s : Shape} (first second : Tensor α s) (p : s.Coord) :
    map first second p =
      if s = .scalar then value
      else (map (first.row p) (second.row p)).getScalar (Shape.Coord.getLast p) := by
  by_cases hs : s = .scalar
  · subst hs
    rw [scalar, scalar_apply, ite_eq_left rfl]
  · rw [ite_eq_right hs]
    exact rowwise_apply_of_ne map outer hs first second p

/-- `f` on every last-axis row of a tensor, and `Tensor.scalar value` on a scalar tensor. -/
def lastAxisMap (value : α) (f : (n : Nat) → Tensor α [n] → Tensor α [n]) :
    {s : Shape} → Tensor α s → Tensor α s
  | .scalar, _ => Tensor.scalar value
  | .dim n .scalar, tensor => f n tensor
  | .dim _ inner, tensor =>
      Tensor.dim fun i => lastAxisMap value f (s := inner) (Tensor.unstack tensor i)

/-- An entry of `lastAxisMap value f x` is `value` on a scalar and otherwise the entry of `f` on
the last-axis row through it. -/
theorem lastAxisMap_apply (value : α) (f : (n : Nat) → Tensor α [n] → Tensor α [n]) {s : Shape}
    (x : Tensor α s) (p : s.Coord) :
    lastAxisMap value f x p =
      if s = .scalar then value else (f s.rowWidth (x.row p)).getScalar (Shape.Coord.getLast p) :=
  rowwise_apply (fun x _ => lastAxisMap value f x) value (fun _ _ => rfl) (fun _ _ => rfl) x x p

end TorchLean.Tensor
