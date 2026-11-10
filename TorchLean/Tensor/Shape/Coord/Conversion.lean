/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Shape

/-!
# Tensor Coordinates

`Shape.Coord` is the finite coordinate space used by verified tensor lowering, and
`Shape.Coord.linearize` and `Shape.Coord.unlinearize` are its row-major index maps. This module
adds the runtime view of a coordinate: `Coord.toList` lists its indices outermost first, and
`Coord.ofList?` and `Coord.ofArray?` validate runtime indices against a static shape.
-/

@[expose] public section

namespace TorchLean
namespace Shape

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

/-- A coordinate is determined by its indices. -/
theorem Coord.toList_injective (s : Shape) : Function.Injective (Coord.toList s) := fun c d h =>
  Option.some.inj ((Coord.ofList?_toList s c).symm.trans (h ▸ Coord.ofList?_toList s d))

/-- A coordinate has one index per axis. -/
theorem Coord.length_toList (s : Shape) (c : s.Coord) :
    (Coord.toList s c).length = s.length := by
  induction s with
  | scalar => rfl
  | dim n s ih => exact congrArg Nat.succ (ih c.2)

/-- Casting a coordinate along a shape equality keeps its indices. -/
theorem Coord.toList_cast {s t : Shape} (h : s = t) (c : s.Coord) :
    Coord.toList t (_root_.cast (congrArg Coord h) c) = Coord.toList s c := by
  subst h
  rfl

end Shape
end TorchLean
