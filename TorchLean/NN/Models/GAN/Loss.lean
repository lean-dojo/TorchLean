/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.GAN
public import TorchLean.Tensor.Context.Real

/-!
# GAN theory

TorchLean's baseline GAN spec uses least-squares losses, which capture the generator/discriminator
game without partial logarithms. The discriminator regresses real samples toward score `1` and
generated samples toward `0`. The generator tries to make its own samples score as real.

The theorems below (`generatorLoss_eq_zero`, `discriminatorLoss_eq_zero`) show that each LSGAN
loss vanishes when the discriminator scores equal their regression targets. Both go through
`Tensor.meanSquaredError_of_isFinite`, the finite branch of the MSE, which every real input takes.
The `rfl`-level unfoldings of the spec (fake scores as
discriminator-after-generator, and the two-term discriminator objective) are proved with the spec
in `TorchLean.NN.Models.GAN`.

The public spec uses the LSGAN square-loss objective rather than the minimax
$\log D+\log(1-D)$ objective of Goodfellow et al., which keeps the executable Lean surface total
over every supported scalar backend. The logarithmic objective is finite only where $0<D(x)<1$.

- Ian Goodfellow et al., "Generative Adversarial Nets", NIPS 2014, arXiv:1406.2661, eq. (1).
- Xudong Mao et al., "Least Squares Generative Adversarial Networks", ICCV 2017, arXiv:1611.04076,
  eq. (9).
- Martin Arjovsky, Soumith Chintala, and Léon Bottou, "Wasserstein Generative Adversarial
  Networks", ICML 2017 (arXiv:1701.07875, "Wasserstein GAN"), for the broader
  critic-vs-discriminator viewpoint that motivates keeping the score head abstract.
-/

@[expose] public section

namespace TorchLean.GAN


variable {latent obs : Shape}

/-! ## Target-score checks -/

/-- Score-regression MSE is zero when the scalar prediction equals the scalar target. -/
private theorem mse_scalar_self_zero (x : Tensor ℝ .scalar) :
    Tensor.meanSquaredError (shape := .scalar) x x = 0 := by
  have hdiff : x - x = Tensor.scalar 0 := by
    apply Tensor.ext_scalar
    simp
  rw [Tensor.meanSquaredError_of_isFinite x x
    (by simp [Context.isFinite, Context.isNaN, Context.isInf]), hdiff]
  simp [Tensor.mul_def, Tensor.meanDenominator]

/-- If generated samples receive the real target score, the LSGAN generator loss is zero. -/
theorem generatorLoss_eq_zero
    (model : GAN ℝ latent obs) (z : Tensor ℝ latent)
    (hfake : fakeScore model z = realTarget (α := ℝ)) :
    generatorLoss model z = 0 := by
  rw [generatorLoss, ← hfake]
  exact mse_scalar_self_zero (fakeScore model z)

/--
If the real sample scores `1` and the generated sample scores `0`, the LSGAN discriminator loss
is zero.
-/
theorem discriminatorLoss_eq_zero
    (model : GAN ℝ latent obs) (xReal : Tensor ℝ obs) (z : Tensor ℝ latent)
    (hreal : model.discriminator xReal = realTarget (α := ℝ))
    (hfake : fakeScore model z = fakeTarget (α := ℝ)) :
    discriminatorLoss model xReal z = 0 := by
  rw [discriminatorLoss, hreal, ← hfake]
  simp [mse_scalar_self_zero]

end TorchLean.GAN
