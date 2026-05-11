/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Vector tensors (`Spec.Vec`)

TorchLean’s spec tensor type is shape-indexed:

`Spec.Tensor α s`

For many developments we need the special case “a length-`n` vector of scalars”, i.e.

`Spec.Tensor α (.dim n .scalar)`.

This file packages that case up as `Spec.Vec n α` and provides basic, representation-agnostic
helpers:

- convert to/from functions `Fin n → α`,
- coordinate access, and
- total “update one coordinate” / “remove one coordinate” operations.

These utilities are used by learning-theory code (datasets as vectors of examples) and are also
generally handy for connecting tensor-shaped objects to `Fin`-indexed mathlib APIs.

References / analogies:
- This is the shape-indexed tensor analogue of the function type `Fin n -> α` (a dependent `Pi`),
  packaged so we can reuse `Tensor` infrastructure while still interoperating with mathlib's
  `Fin`-indexed APIs.
- Lean core / mathlib building blocks used here: `Fin`, `Function.update`, and `Fin.succAbove`.
-/

@[expose] public section


namespace Spec

open scoped BigOperators

/-- A length-`n` vector of scalar entries `α` as a spec tensor. -/
abbrev Vec (n : Nat) (α : Type) : Type :=
  Tensor α (.dim n .scalar)

namespace Vec

variable {n : Nat} {α : Type}

/-- View a vector tensor as a function `Fin n → α`. -/
abbrev toFn (v : Vec n α) : Fin n → α :=
  (Tensor.dimScalarEquiv (α := α) n).toFun v

/-- Build a vector tensor from a function `Fin n → α`. -/
abbrev ofFn (f : Fin n → α) : Vec n α :=
  (Tensor.dimScalarEquiv (α := α) n).invFun f

@[simp] theorem toFn_ofFn (f : Fin n → α) : toFn (n := n) (α := α) (ofFn (n := n) (α := α) f) = f :=
  by
  simp [toFn, ofFn]

@[simp] theorem ofFn_toFn (v : Vec n α) : ofFn (n := n) (α := α) (toFn (n := n) (α := α) v) = v :=
  by
  simp [toFn, ofFn]

/-- Coordinate access for vector tensors. -/
abbrev get (v : Vec n α) (i : Fin n) : α :=
  toFn (n := n) (α := α) v i

@[simp] theorem get_ofFn (f : Fin n → α) (i : Fin n) :
    get (n := n) (α := α) (ofFn (n := n) (α := α) f) i = f i := by
  simp [get, toFn, ofFn]

/-- Update one coordinate (total, using `Fin` indices). -/
def update [DecidableEq (Fin n)] (v : Vec n α) (i : Fin n) (a' : α) : Vec n α :=
  ofFn (n := n) (α := α) (Function.update (toFn (n := n) (α := α) v) i a')

/--
Remove one coordinate from a vector of length `n+1`.

This uses `Fin.succAbove` to reindex the remaining entries into `Fin n`.
-/
def removeAt (v : Vec (n + 1) α) (i : Fin (n + 1)) : Vec n α :=
  ofFn (n := n) (α := α) (fun j => get (n := n + 1) (α := α) v (i.succAbove j))

end Vec

end Spec
