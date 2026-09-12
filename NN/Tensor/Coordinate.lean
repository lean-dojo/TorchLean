/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Shape

/-!
# Tensor Coordinates

This module connects the public `Spec.Shape` API to row-major coordinate geometry.
Both the specification and buffer layers use the same list of dimensions; their
size functions are connected explicitly so that each layer retains its own simplification API.
`Shape.Coord` supplies the finite coordinate space used by verified tensor lowering.
-/

@[expose] public section

open TorchLean

namespace Spec
namespace Shape

/-- The finite multidimensional coordinate type of a TorchLean shape. -/
abbrev Coord (shape : Shape) : Type :=
  TorchLean.Tensor.Internal.Coord shape

/--
The buffer-level element count agrees with the spec-level one.

Not a `simp` lemma. `Spec.Shape.size` and `Tensor.Internal.Shape.size` are two definitions of the
same function on the same type, one for each layer, and rewriting every buffer-level count into a
spec-level one would strand the buffer lemmas that `Rep` proofs rely on. Bridge explicitly where a
spec-level fact has to meet a buffer-level obligation.
-/
theorem internalSize_eq (shape : Shape) :
    TorchLean.Tensor.Internal.Shape.size shape = shape.size := by
  induction shape with
  | scalar => rfl
  | dim length rest inductionHypothesis =>
      simp [Shape.size, inductionHypothesis]

/--
Transport a spec-level size equality down to the buffer level.

The companion of `internalSize_eq` for the common case where a definition demands the buffer-level
equality and the caller has the spec-level one.
-/
theorem internalSize_congr {source target : Shape} (hSize : source.size = target.size) :
    TorchLean.Tensor.Internal.Shape.size source =
      TorchLean.Tensor.Internal.Shape.size target := by
  rw [internalSize_eq, internalSize_eq]
  exact hSize

/-- Convert a multidimensional coordinate to its row-major flat index. -/
def Coord.linearize {shape : Shape} (coordinate : shape.Coord) :
    Fin shape.size :=
  ⟨(TorchLean.Tensor.Internal.Coord.linearize coordinate).val, by
    rw [← internalSize_eq]
    exact (TorchLean.Tensor.Internal.Coord.linearize coordinate).isLt⟩

/-- Recover a multidimensional coordinate from a row-major flat index. -/
def Coord.unlinearize {shape : Shape} (index : Fin shape.size) :
    shape.Coord :=
  TorchLean.Tensor.Internal.Coord.unlinearize
    ⟨index.val, by
      rw [internalSize_eq]
      exact index.isLt⟩

/-- Expose a statically valid coordinate as outermost-first natural indices. -/
def Coord.toList : (shape : Shape) → shape.Coord → List Nat
  | .scalar, _ => []
  | .dim _ rest, coordinate =>
      coordinate.1.val :: Coord.toList rest coordinate.2

/--
Validate a runtime coordinate list against a static tensor shape.

The list is outermost-first and must contain exactly one in-bounds index per
axis. Successful validation returns the ordinary proof-carrying coordinate.
-/
def Coord.ofList? : (shape : Shape) → List Nat → Option shape.Coord
  | .scalar, [] => some PUnit.unit
  | .scalar, _ :: _ => none
  | .dim _ _, [] => none
  | .dim length rest, index :: indices =>
      if hIndex : index < length then
        (Coord.ofList? rest indices).map fun coordinate =>
          (⟨index, hIndex⟩, coordinate)
      else
        none

/-- Validate runtime array coordinates against a static tensor shape. -/
def Coord.ofArray? (shape : Shape) (coordinates : Array Nat) :
    Option shape.Coord :=
  Coord.ofList? shape coordinates.toList

/-- A statically valid coordinate validates when converted to runtime indices. -/
@[simp] theorem Coord.ofList?_toList (shape : Shape) (coordinate : shape.Coord) :
    Coord.ofList? shape (Coord.toList shape coordinate) = some coordinate := by
  induction shape with
  | scalar =>
      cases coordinate
      rfl
  | dim length rest inductionHypothesis =>
      rcases coordinate with ⟨index, inner⟩
      simp [Coord.ofList?, Coord.toList, index.isLt, inductionHypothesis]

/-- Unlinearizing and then linearizing recovers the flat index. -/
@[simp] theorem Coord.linearize_unlinearize {shape : Shape}
    (index : Fin shape.size) :
    Coord.linearize (Coord.unlinearize index) = index := by
  apply Fin.ext
  simp [Coord.linearize, Coord.unlinearize]

/-- Linearizing and then unlinearizing recovers the multidimensional coordinate. -/
@[simp] theorem Coord.unlinearize_linearize {shape : Shape}
    (coordinate : shape.Coord) :
    Coord.unlinearize (Coord.linearize coordinate) = coordinate := by
  exact TorchLean.Tensor.Internal.Coord.unlinearize_linearize coordinate

end Shape
end Spec
