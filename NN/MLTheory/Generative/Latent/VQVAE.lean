/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.VqVae
public import NN.MLTheory.Generative.Latent.Objective
public import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# VQ-VAE theory

VQ-VAE has one mathematically delicate implementation choice: nearest-neighbor code assignment.
TorchLean's spec keeps that assignment explicit as a `Fin numCodes`, so the core codebook semantics
are total and easy to audit.  Runtime code may compute the index using a CUDA, Python, or Lean
argmin; once the index is supplied, the following facts are definitional.

We also prove the real-valued nearest-code optimality lemma used by vector quantization: if an
index is selected as an argmin of squared Euclidean distance to the encoder output, then the
corresponding quantization loss is minimal among all codebook choices.

Reference:
- Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu, "Neural Discrete Representation
  Learning", NeurIPS 2017.
-/

@[expose] public section

namespace NN.MLTheory.Generative.Latent.VQVAE

open _root_.Spec
open _root_.Generative.VQVAE
open NN.MLTheory.Generative.Latent.Objective
open BigOperators

variable {α : Type} [Context α]
variable {obs latent : Shape} {numCodes : Nat}

/-- Quantization with an explicit code index is codebook lookup. -/
@[simp] theorem quantized_is_codebook_lookup
    (model : Model α obs latent numCodes) (idx : Fin numCodes) :
    quantized model idx = model.codebook.embedding idx := by
  rfl

/-- VQ-VAE reconstruction decodes the selected codebook vector. -/
@[simp] theorem forward_eq_decoder_codebook
    (model : Model α obs latent numCodes) (x : Tensor α obs) (idx : Fin numCodes) :
    forward model x idx = model.decoder.forward (model.codebook.embedding idx) := by
  rfl

/-- The VQ-VAE loss splits into reconstruction, codebook, and commitment terms. -/
@[simp] theorem vqvae_loss_decomposition
    [DecidableRel ((· > ·) : α → α → Prop)] [LE α]
    (model : Model α obs latent numCodes) (beta : α) (x : Tensor α obs) (idx : Fin numCodes) :
    loss model beta x idx =
      reconstructionLoss model x idx + codebookLoss model x idx +
        beta * commitmentLoss model x idx := by
  rfl

/-! ## Connection to the shared latent-objective algebra -/

/-- Package VQ-VAE reconstruction, codebook, and commitment terms as a weighted three-term objective. -/
noncomputable def vqvaeObjectiveTerms
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent numCodes) (x : Tensor ℝ obs) (idx : Fin numCodes) :
    WeightedThreeTerm :=
  { base := reconstructionLoss model x idx
    middle := codebookLoss model x idx
    regularizer := commitmentLoss model x idx }

/-- VQ-VAE loss is exactly the shared `base + middle + β * regularizer` objective. -/
theorem vqvae_loss_eq_weightedThreeTerm
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent numCodes) (beta : ℝ) (x : Tensor ℝ obs) (idx : Fin numCodes) :
    loss model beta x idx = weightedThreeTerm beta (vqvaeObjectiveTerms model x idx) := by
  rfl

/-- At commitment weight `β = 0`, VQ-VAE keeps reconstruction plus codebook loss. -/
@[simp] theorem vqvae_loss_zero_beta
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent numCodes) (x : Tensor ℝ obs) (idx : Fin numCodes) :
    loss model 0 x idx = reconstructionLoss model x idx + codebookLoss model x idx := by
  simp [loss]

/--
If the selected code matches the encoder in the sense that both quantization penalties vanish, the
VQ-VAE objective reduces to reconstruction loss.
-/
theorem vqvae_loss_eq_reconstruction_of_zero_quantization
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent numCodes) (beta : ℝ) (x : Tensor ℝ obs) (idx : Fin numCodes)
    (hcode : codebookLoss model x idx = 0) (hcommit : commitmentLoss model x idx = 0) :
    loss model beta x idx = reconstructionLoss model x idx := by
  simp [loss, hcode, hcommit]

/--
Commitment-weight monotonicity for the executable VQ-VAE loss.

Once a verifier or model-specific theorem establishes that the commitment term is nonnegative,
increasing β cannot decrease the objective.
-/
theorem vqvae_loss_mono_beta_of_commitment_nonneg
    [DecidableRel ((· > ·) : ℝ → ℝ → Prop)]
    (model : Model ℝ obs latent numCodes) (x : Tensor ℝ obs) (idx : Fin numCodes)
    {beta₁ beta₂ : ℝ} (hbeta : beta₁ ≤ beta₂)
    (hcommit : 0 ≤ commitmentLoss model x idx) :
    loss model beta₁ x idx ≤ loss model beta₂ x idx := by
  simp [loss]
  exact mul_le_mul_of_nonneg_right hbeta hcommit

/-! ## Nearest-code optimality -/

/-- Squared Euclidean distance on finite real coordinate vectors. -/
noncomputable def squaredL2 {d : Nat} (x y : Fin d → ℝ) : ℝ :=
  ∑ k, (x k - y k) ^ 2

/--
The predicate that `idx` is a nearest code for encoder output `z` under squared Euclidean
distance.  Ties are allowed; tie-breaking is an implementation detail outside this theorem.
-/
def IsNearestCode {numCodes d : Nat}
    (embedding : Fin numCodes → Fin d → ℝ) (z : Fin d → ℝ) (idx : Fin numCodes) : Prop :=
  ∀ j, squaredL2 z (embedding idx) ≤ squaredL2 z (embedding j)

/-- Squared Euclidean distance is nonnegative. -/
theorem squaredL2_nonneg {d : Nat} (x y : Fin d → ℝ) :
    0 ≤ squaredL2 x y := by
  unfold squaredL2
  exact Finset.sum_nonneg (fun k _ => sq_nonneg (x k - y k))

/-- Exact code matches have zero quantization distance. -/
@[simp] theorem squaredL2_self {d : Nat} (x : Fin d → ℝ) :
    squaredL2 x x = 0 := by
  unfold squaredL2
  simp

/-- If the encoder output is exactly one code, that code is a nearest code. -/
theorem exactCodeMatch_isNearestCode
    {numCodes d : Nat} {embedding : Fin numCodes → Fin d → ℝ}
    {z : Fin d → ℝ} {idx : Fin numCodes}
    (hmatch : z = embedding idx) :
    IsNearestCode embedding z idx := by
  intro j
  subst hmatch
  simpa using squaredL2_nonneg (embedding idx) (embedding j)

/-- Exact code matches have zero selected quantization distance. -/
theorem exactCodeMatch_selected_distance_zero
    {numCodes d : Nat} {embedding : Fin numCodes → Fin d → ℝ}
    {z : Fin d → ℝ} {idx : Fin numCodes}
    (hmatch : z = embedding idx) :
    squaredL2 z (embedding idx) = 0 := by
  subst hmatch
  simp

/--
Nearest-code optimality for VQ-VAE.

Once the runtime argmin has returned an index satisfying `IsNearestCode`, the selected code's
quantization distance is no larger than the distance to any other code.  This is the formal
contract that lets CUDA/Python/Lean argmin implementations plug into the same spec semantics.
-/
theorem nearestCode_minimizes_quantization_loss
    {numCodes d : Nat} {embedding : Fin numCodes → Fin d → ℝ}
    {z : Fin d → ℝ} {idx j : Fin numCodes}
    (hidx : IsNearestCode embedding z idx) :
    squaredL2 z (embedding idx) ≤ squaredL2 z (embedding j) :=
  hidx j

end NN.MLTheory.Generative.Latent.VQVAE
