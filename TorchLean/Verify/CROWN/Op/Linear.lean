/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op
public import TorchLean.Verify.CROWN.FlatBox.Validity

/-!
# The linear-layer rule on operator graphs

`Op.ibpLinearAt p x Xin` is the interval box of a linear layer with parameters `p` on an input box
`Xin` at port shape `x`. Over `ℝ` it is none or a valid box whenever `Xin` is valid
(`ibpLinearAt_valid_real`). Over any scalar type, an input box of width `p.inDim` gives a box of
width `p.outDim`, or of width `rows * p.outDim` when `p.inDim = 0`, with `rows` the size of the
leading axes of `x` (`ibpLinearAt_dim`).
-/

@[expose] public section

namespace TorchLean.CROWN.Op

/-- On a valid input box `Xin`, the operator-graph linear rule `ibpLinearAt p x Xin` is none or a
valid box, for all parameters `p` and input port shapes `x` (over `ℝ`). -/
theorem ibpLinearAt_valid_real (p : SomeLinear ℝ) (x : Shape) (Xin : FlatBox ℝ)
    (hXin : Xin.Valid) :
    match ibpLinearAt (α := ℝ) p x Xin with
    | none => True
    | some Bout => Bout.Valid := by
  unfold ibpLinearAt
  by_cases hn : p.inDim = 0
  · by_cases hX : Xin.dim = 0
    · simp only [hn, hX, ↓reduceIte]
      exact fun _ => le_rfl
    · simp [hn, hX]
  · simp only [hn, ↓reduceIte]
    have h := Graph.ibpLinearParams_valid_real p Xin hXin
    cases hb : Graph.ibpLinearParams p Xin with
    | none => trivial
    | some Bout => simpa [hb] using h

section

variable {α : Type} [Storage α] [Context α] [BoundOps α]

/-- If `Xin.dim = p.inDim`, then `ibpLinearAt p parentShape Xin` is a box of width `p.outDim` when
`p.inDim ≠ 0`, and of width `rows * p.outDim` when `p.inDim = 0`, with `rows` the size of the
leading axes of `parentShape`. -/
theorem ibpLinearAt_dim
    (p : SomeLinear α) (parentShape : Shape) (Xin : FlatBox α)
    (h : Xin.dim = p.inDim) :
    (ibpLinearAt (α := α) p parentShape Xin).map (·.dim) =
      some (if p.inDim = 0 then Shape.size parentShape.dropLast * p.outDim else p.outDim) := by
  by_cases hn : p.inDim = 0
  · simp [ibpLinearAt, hn, h]
  · simp [ibpLinearAt, Graph.ibpLinearParams, h, hn, Graph.toFlatBox]

end

end TorchLean.CROWN.Op
