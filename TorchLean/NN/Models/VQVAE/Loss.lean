/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.VQVAE
public import Mathlib.Analysis.Normed.Lp.PiLp
public import TorchLean.Tensor.Context.Real

/-!
# VQ-VAE theory

The one mathematically delicate step in VQ-VAE is nearest-neighbor code assignment. TorchLean's
spec takes the assigned code as an argument of type `Fin numCodes`, so the codebook semantics are
total and easy to audit. Runtime code may compute the index with a CUDA, Python or Lean argmin.
Once the index is supplied, the reconstruction and loss facts here follow from the definitions,
and the two purely definitional ones (codebook lookup and the three-term loss split) are proved
next to the spec in `TorchLean.NN.Models.VQVAE`.

The nearest-code predicate `IsNearestCode` compares Euclidean distances of real coordinate
vectors, measured in `WithLp 2 (Fin d → ℝ)`. A code equal to the encoder output is a nearest
code, and a nonempty codebook has a nearest code for every encoder output. The predicate restates
the `argmin_j ‖z_e(x) − e_j‖₂` of eqs. (1) and (2) in van den Oord et al. (2017).

Reference:
- Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu (2017), "Neural Discrete
  Representation Learning", NIPS 2017, arXiv:1711.00937, eqs. (1) and (2).
-/

@[expose] public section

namespace TorchLean.VQVAE


variable {α : Type} [Storage α] [Context α]
variable {obs latent : Shape} {numCodes : Nat}

omit [Context α] in
/-- VQ-VAE reconstruction decodes the selected codebook vector. -/
@[simp] theorem forward_eq_decoder_codebook
    (model : VQVAE α obs latent numCodes) (x : Tensor α obs) (idx : Fin numCodes) :
    forward model x idx = model.decoder (model.codebook.embedding idx) := by
  rfl

/-! ## Loss identities -/

/-- At commitment weight $\beta=0$, the VQ-VAE loss is reconstruction plus codebook loss. -/
@[simp] theorem loss_zero_beta
    (model : VQVAE ℝ obs latent numCodes) (x : Tensor ℝ obs) (idx : Fin numCodes) :
    loss model 0 x idx = reconstructionLoss model x idx + codebookLoss model x idx := by
  simp [loss]

/--
If both quantization penalties of the selected code vanish, the VQ-VAE objective reduces to
reconstruction loss for any β.
-/
theorem loss_eq_of_zero_quantization
    (model : VQVAE ℝ obs latent numCodes) (beta : ℝ) (x : Tensor ℝ obs) (idx : Fin numCodes)
    (hcode : codebookLoss model x idx = 0) (hcommit : commitmentLoss model x idx = 0) :
    loss model beta x idx = reconstructionLoss model x idx := by
  simp [loss, hcode, hcommit]

/--
Commitment-weight monotonicity for the executable VQ-VAE loss.

Once a verifier or model-specific theorem establishes that the commitment term is nonnegative,
increasing β cannot decrease the objective.
-/
theorem loss_mono_beta_of_nonneg
    (model : VQVAE ℝ obs latent numCodes) (x : Tensor ℝ obs) (idx : Fin numCodes)
    {beta₁ beta₂ : ℝ} (hbeta : beta₁ ≤ beta₂)
    (hcommit : 0 ≤ commitmentLoss model x idx) :
    loss model beta₁ x idx ≤ loss model beta₂ x idx := by
  unfold loss
  exact add_le_add le_rfl (mul_le_mul_of_nonneg_right hbeta hcommit)

/-! ## Nearest codes -/

/--
`idx` is a nearest code for the encoder output `z`: no code is closer to `z` in Euclidean distance.
Ties are allowed.
-/
def IsNearestCode {numCodes d : Nat}
    (embedding : Fin numCodes → Fin d → ℝ) (z : Fin d → ℝ) (idx : Fin numCodes) : Prop :=
  ∀ j, dist (WithLp.toLp 2 z) (WithLp.toLp 2 (embedding idx)) ≤
    dist (WithLp.toLp 2 z) (WithLp.toLp 2 (embedding j))

/-- A code equal to the encoder output is a nearest code. -/
theorem isNearestCode_of_eq
    {numCodes d : Nat} {embedding : Fin numCodes → Fin d → ℝ}
    {z : Fin d → ℝ} {idx : Fin numCodes}
    (hmatch : z = embedding idx) :
    IsNearestCode embedding z idx := by
  intro j
  rw [hmatch, dist_self]
  exact dist_nonneg

/-- A nonempty codebook has a nearest code for every encoder output. -/
theorem exists_isNearestCode {numCodes d : Nat} [NeZero numCodes]
    (embedding : Fin numCodes → Fin d → ℝ) (z : Fin d → ℝ) :
    ∃ idx, IsNearestCode embedding z idx :=
  Finite.exists_min fun i => dist (WithLp.toLp 2 z) (WithLp.toLp 2 (embedding i))

end TorchLean.VQVAE
