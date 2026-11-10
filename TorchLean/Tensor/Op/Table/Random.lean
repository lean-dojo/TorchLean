/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the random operators

Each lemma here states one arm of `Op.rule?` for a keyed draw (`randUniform`, `randn`,
`bernoulliMask`) or for `dropout`, which reads its uniform draws from an input.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.randUniform`. -/
theorem rule?_randUniform [Op.Capabilities α] {s : Shape} (key : UInt64) :
    (Op.randUniform (s := s) key).rule? (α := α) = some
      { forward := fun _ => (Random.uniform key, ())
        backward := fun _ _ => ()
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  rfl

/-- The rule of `Op.randn`. -/
theorem rule?_randn [Op.Capabilities α] {s : Shape} (key : UInt64) :
    (Op.randn (s := s) key).rule? (α := α) = some
      { forward := fun _ => (Random.normal key, ())
        backward := fun _ _ => ()
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  rfl

/-- The rule of `Op.dropout`. -/
theorem rule?_dropout [Op.Capabilities α] {s : Shape} :
    (Op.dropout (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          (Dropout.masked xs.2.2.1.item (Tensor.ge xs.2.1 (Tensor.full s xs.2.2.1.item))
            xs.1, ())
        backward := fun xs g =>
          (Dropout.masked xs.2.2.1.item
            (Tensor.ge xs.2.1 (Tensor.full s xs.2.2.1.item)) g.1, (), (), ())
        jvp? := some fun xs dxs =>
          (Dropout.masked xs.2.2.1.item (Tensor.ge xs.2.1 (Tensor.full s xs.2.2.1.item))
            dxs.1, ()) } :=
  rfl

/-- The rule of `Op.bernoulliMask`. -/
theorem rule?_bernoulliMask [Op.Capabilities α] {s : Shape} (key : UInt64) :
    (Op.bernoulliMask (s := s) key).rule? (α := α) = some
      { forward := fun xs => (Random.mask key xs.1.item, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  rfl

end Op

end TorchLean
