/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import Lean.Meta.Tactic.Simp.Attr
import Lean.Meta.Tactic.Simp.RegisterCommand

/-!
# Output shapes of the general tensor operators

`contract`, `cat`, `gather` and `permute` take their output shape as an argument, with a proof
that it equals the shape they compute. `tensor_shape` is the default proof: `rfl` when both sides
compute to the same shape, then `simp only [tensor_shape_simps]` when that closes the goal, then
`decide`. A lemma tagged `@[tensor_shape_simps]` rewrites a computed shape to its normal form.
-/

/-- Equations rewriting the output shape computed by a general tensor operator to its normal
form, used by `tensor_shape`. -/
register_simp_attr tensor_shape_simps

public meta section

/-- Prove the output-shape equation of a general tensor operator. -/
macro "tensor_shape" : tactic =>
  `(tactic| first
    | rfl
    | (simp only [tensor_shape_simps]; done)
    | decide)

end
