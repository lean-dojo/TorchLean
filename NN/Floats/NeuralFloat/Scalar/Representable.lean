/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding.Generic
public import NN.Floats.NeuralFloat.Scalar.NF

/-!
# Grid Invariants for `NF`

`NF` permits raw real-valued construction for approximation proofs, while its smart constructor and
primitive arithmetic round onto the declared format. This file proves the corresponding
`NF.IsRepresentable` closure properties without adding generic-format theory to the core scalar
module's import surface.
-/

@[expose] public section

namespace TorchLean.Floats.NF

variable {β : NeuralRadix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}
variable [NeuralValidExp fexp] [NeuralValidRnd rnd]

/-- The smart constructor rounds its input onto the declared format grid. -/
theorem isRepresentable_ofReal (x : ℝ) :
    IsRepresentable (ofReal (β := β) (fexp := fexp) (rnd := rnd) x) := by
  exact neural_generic_format_round rnd x

/-- The negation of representable values is representable, since the operation rounds back into
the format. -/
@[simp] theorem isRepresentable_neg (a : NF β fexp rnd) : IsRepresentable (-a) := by
  change IsRepresentable (ofReal (β := β) (fexp := fexp) (rnd := rnd) (-a.val))
  exact isRepresentable_ofReal _

/-- The sum of representable values is representable, since the operation rounds back into
the format. -/
@[simp] theorem isRepresentable_add (a b : NF β fexp rnd) : IsRepresentable (a + b) := by
  change IsRepresentable (ofReal (β := β) (fexp := fexp) (rnd := rnd) (a.val + b.val))
  exact isRepresentable_ofReal _

/-- The difference of representable values is representable, since the operation rounds back into
the format. -/
@[simp] theorem isRepresentable_sub (a b : NF β fexp rnd) : IsRepresentable (a - b) := by
  change IsRepresentable (ofReal (β := β) (fexp := fexp) (rnd := rnd) (a.val - b.val))
  exact isRepresentable_ofReal _

/-- The product of representable values is representable, since the operation rounds back into
the format. -/
@[simp] theorem isRepresentable_mul (a b : NF β fexp rnd) : IsRepresentable (a * b) := by
  change IsRepresentable (ofReal (β := β) (fexp := fexp) (rnd := rnd) (a.val * b.val))
  exact isRepresentable_ofReal _

/-- The quotient of representable values is representable, since the operation rounds back into
the format. -/
@[simp] theorem isRepresentable_div (a b : NF β fexp rnd) : IsRepresentable (a / b) := by
  change IsRepresentable (ofReal (β := β) (fexp := fexp) (rnd := rnd) (a.val / b.val))
  exact isRepresentable_ofReal _

end TorchLean.Floats.NF
