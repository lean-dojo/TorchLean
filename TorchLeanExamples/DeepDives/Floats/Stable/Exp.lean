/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Stable.Once
public import TorchLeanExamples.DeepDives.Floats.LibraryBugs
public import TorchLeanExamples.DeepDives.Floats.ProbBugs.SurvivalComplement
public import TorchLean.NN.Gradients.Activation
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Rewrites with one exponential, rounded once

Three functions are ratios `(α * exp r + β) / (γ * exp r + δ)` with coefficients and exponent
that are rational at the rational points (`Target.ofExp`):
* `sigmoid`: TorchLean's sigmoid `σ x = 1 / (1 + exp (-x))` on every real `x`.
* `softMargin`: the gradient `-y σ (-y x)` of the soft-margin loss in `x`
  (`LossBugs.SoftMargin.lossBackward`), on every pair, the ratio `-y / (exp (y x) + 1)`.
* `weibull k`: the Weibull survival function `exp (-(x / λ) ^ k)` of a natural exponent `k`
  (`ProbBugs.Survival.sf` at `(x / λ) ^ k`), on `x ≥ 0` and `λ > 0`. It reads `x` and `λ`, so the
  power is not rounded first.

Each target's function is the library's function over ℝ. The program that rounds it once has the
certificate of `Once.claim` in every IEEE binary format (`cert_sigmoid`, `cert_softMargin`,
`cert_weibull`): where the magnitude of the answer is below the overflow threshold `Θ`, its result
is the nearest-even rounding of the answer, within the mixed budget `(u, max 0 (η - u |F|))`, which
is informative on the band `η < |F| < Θ`.

TorchLean's soft-margin gradient rounds the sigmoid before the product with `y`. In binary32 at
`x = 53/8` and `y = 16` the sigmoid of `-106` rounds to zero, so the computed gradient is zero,
while the gradient is about `-1.48e-45`, in the band. `softMarginRepaired`, TorchLean's gradient
repaired by the gradient rounded once (`Target.repair`), returns `-2⁻¹⁴⁹` there. In every IEEE
binary format it returns TorchLean's result wherever that is finite and not zero
(`Target.repair_eq`), and its result is finite and not zero wherever the magnitude of the gradient
lies in the band (`Target.repair_finite`).
-/

@[expose] public section

namespace TorchLean.Examples.Stable

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open TorchLean.Stable
open Arith (valid)

/-- TorchLean's sigmoid over ℝ, the ratio `1 / (exp (-x) + 1)`. -/
noncomputable def sigmoid : Target 1 :=
  Target.ofExp (fun v => Context.sigmoid (v 0)) Set.univ (fun _ => 0) (fun _ => 1) (fun _ => 1)
    (fun _ => 1) (fun q => -q 0) fun q _ => by
      simp only [Context.sigmoid_eq_inv_exp]
      push_cast
      ring

/-- The sigmoid rounded once has the certificate of `Once.claim` on every input in every IEEE
binary format. -/
theorem cert_sigmoid :
    Cert sigmoid.prog sigmoid.F {d | valid d} (Once.claim sigmoid.F fun _ => sigmoid.D) := by
  refine sigmoid.computes_prog.cert (fun _ => 0) (Set.mem_univ _) ?_ ?_ <;>
    norm_num [sigmoid, Target.ofExp, Context.sigmoid_eq_inv_exp]

/-- The gradient `-y σ (-y x)` of the soft-margin loss in `x`, the ratio `-y / (exp (y x) + 1)`. -/
noncomputable def softMargin : Target 2 :=
  Target.ofExp (fun v => LossBugs.SoftMargin.lossBackward (v 0) (v 1)) Set.univ (fun _ => 0)
    (fun q => -q 1) (fun _ => 1) (fun _ => 1) (fun q => q 1 * q 0) fun q _ => by
      simp only [LossBugs.SoftMargin.lossBackward, Context.sigmoid_eq_inv_exp, neg_neg]
      push_cast
      ring

/-- The soft-margin gradient rounded once has the certificate of `Once.claim` on every pair in
every IEEE binary format. -/
theorem cert_softMargin :
    Cert softMargin.prog softMargin.F {d | valid d}
      (Once.claim softMargin.F fun _ => softMargin.D) := by
  refine softMargin.computes_prog.cert ![0, 1] (Set.mem_univ _) ?_ ?_ <;>
    norm_num [softMargin, Target.ofExp, LossBugs.SoftMargin.lossBackward,
      Context.sigmoid_eq_inv_exp]

/-- TorchLean's soft-margin gradient `LossBugs.SoftMargin.lossBackward` where its result is finite
and not zero, and the gradient rounded once where it is zero or not finite. -/
noncomputable def softMarginRepaired : Prog Binary.Desc 2 :=
  softMargin.repair (Prog.ofContext fun v => LossBugs.SoftMargin.lossBackward (v 0) (v 1))

/-- In binary32 at `x = 53/8` and `y = 16` TorchLean's soft-margin gradient is zero, while the
repaired gradient decodes to `-2⁻¹⁴⁹`, the gradient rounded once. -/
example :
    let x : Fin 2 → Binary32 := ![53 / 8, 16]
    ExecFloat.Binary.toRat? (LossBugs.SoftMargin.lossBackward (x 0) (x 1)) = some 0 ∧
      Model.toRat? (Binary.toModel (softMarginRepaired.float Binary.binary32 x)) =
        some (-1 / 2 ^ 149) := by
  decide +kernel

/-- The Weibull survival function `exp (-(x / λ) ^ k)` of the natural exponent `k`, on `x ≥ 0`
and `λ > 0`. -/
noncomputable def weibull (k : ℕ) : Target 2 :=
  Target.ofExp (fun v => ProbBugs.Survival.sf ((v 0 / v 1) ^ k)) {v | 0 ≤ v 0 ∧ 0 < v 1}
    (fun _ => 1) (fun _ => 0) (fun _ => 0) (fun _ => 1) (fun q => -((q 0 / q 1) ^ k))
    fun q _ => by
      simp only [ProbBugs.Survival.sf, MathFunctions.exp_real]
      push_cast
      ring

/-- The Weibull survival function rounded once has the certificate of `Once.claim` on `x ≥ 0` and
`λ > 0` in every IEEE binary format, for every natural exponent `k`. -/
theorem cert_weibull (k : ℕ) :
    Cert (weibull k).prog (weibull k).F {d | valid d}
      (Once.claim (weibull k).F fun _ => (weibull k).D) := by
  have he : Real.exp 1 < 4 := Real.exp_one_lt_d9.trans (by norm_num)
  have h1 : 1 / 4 < Real.exp (-1) := by
    rw [Real.exp_neg, one_div]
    exact inv_strictAnti₀ (Real.exp_pos 1) he
  refine (weibull k).computes_prog.cert (fun _ => 1) ?_ ?_ ?_ <;>
    norm_num [weibull, Target.ofExp, ProbBugs.Survival.sf, MathFunctions.exp_real]
  exact h1

end TorchLean.Examples.Stable
