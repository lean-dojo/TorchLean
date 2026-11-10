/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Norm.BatchNorm.FDeriv

/-!
# Batch Normalization Backward Correctness

The training-time BatchNorm differential couples all positions in a channel through the channel mean
and variance. The reverse-mode rule of TorchLean is proved here for an arbitrary spatial shape, as
JVP/VJP adjointness: pairing an input and parameter tangent with the forward differential gives the
same scalar as pairing the upstream gradient with `TorchLean.BatchNorm.backward`.

The theorem is independent of a memory layout. The spatial shape is flattened only inside the
BatchNorm definitions, so one statement covers vectors, images, volumes and higher-rank tensors.

The companion module `TorchLean.NN.Norm.BatchNorm.FDeriv` proves that
`TorchLean.BatchNorm.jvp` is the Fréchet derivative of `Tensor.batchNorm x 0 gamma beta .batch ε` in
`(x, gamma, beta)` whenever `0 < ε` (`hasFDerivAt_forwardVec`, `fderiv_forwardVec_eq_jvp`).
With the adjointness proved here, `TorchLean.BatchNorm.backward` is the adjoint of that derivative.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean
namespace BatchNorm

open Tensor
open Rep (dot)

noncomputable section

open scoped BigOperators

private theorem inner_getScalarE_cast {n m : Nat} (h : n = m)
    (a b : Tensor ℝ [n]) :
    inner ℝ (getScalarE (h ▸ a)) (getScalarE (h ▸ b)) =
      inner ℝ (getScalarE a) (getScalarE b) := by
  subst m
  rfl

private theorem dot_reshape {s₁ s₂ : Shape} (a b : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    Rep.dot (Tensor.reshape a _ h) (Tensor.reshape b _ h) = Rep.dot a b := by
  rw [dot_eq_inner_toVec, dot_eq_inner_toVec]
  unfold toVec
  rw [reshape_vector_reshape, reshape_vector_reshape]
  exact inner_getScalarE_cast h _ _

private theorem dot_reshape_left {s₁ s₂ : Shape} (a : Tensor ℝ s₁)
    (b : Tensor ℝ s₂) (h : s₁.size = s₂.size) :
    Rep.dot (Tensor.reshape a _ h) b = Rep.dot a (Tensor.reshape b _ h.symm) := by
  have hDot := dot_reshape a (Tensor.reshape b _ h.symm) h
  rw [reshape_reshape_self] at hDot
  exact hDot

private theorem channel_normalization_adjoint
    (positions : Nat) (hPositions : 0 < positions)
    (tangent gradOutput xHat : Fin positions → ℝ)
    (invStd gamma dgamma dbeta : ℝ) :
    (∑ position,
        (invStd *
              (tangent position - (∑ k, tangent k) / positions -
                xHat position * ((∑ k, tangent k * xHat k) / positions)) * gamma +
            xHat position * dgamma + dbeta) * gradOutput position) =
      (∑ position,
          tangent position *
            (invStd *
              (gradOutput position * gamma - (∑ k, gradOutput k * gamma) / positions -
                xHat position *
                  ((∑ k, gradOutput k * gamma * xHat k) / positions)))) +
        dgamma * (∑ position, gradOutput position * xHat position) +
        dbeta * (∑ position, gradOutput position) := by
  let meanTangent := (∑ k, tangent k) / (positions : ℝ)
  let meanTangentXHat := (∑ k, tangent k * xHat k) / (positions : ℝ)
  let meanGradGamma := (∑ k, gradOutput k * gamma) / (positions : ℝ)
  let meanGradGammaXHat :=
    (∑ k, gradOutput k * gamma * xHat k) / (positions : ℝ)
  let scale := invStd * gamma
  have hForward :
      (∑ position,
          (invStd *
                (tangent position - meanTangent - xHat position * meanTangentXHat) * gamma +
              xHat position * dgamma + dbeta) * gradOutput position) =
        scale * (∑ position, tangent position * gradOutput position) -
          scale * meanTangent * (∑ position, gradOutput position) -
          scale * meanTangentXHat * (∑ position, xHat position * gradOutput position) +
          dgamma * (∑ position, xHat position * gradOutput position) +
          dbeta * (∑ position, gradOutput position) := by
    calc
      _ = ∑ position,
          (scale * (tangent position * gradOutput position) -
            (scale * meanTangent) * gradOutput position -
            (scale * meanTangentXHat) * (xHat position * gradOutput position) +
            dgamma * (xHat position * gradOutput position) +
            dbeta * gradOutput position) := by
              apply Finset.sum_congr rfl
              intro position _
              dsimp [scale]
              ring
      _ = _ := by
        simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  have hBackward :
      (∑ position,
          tangent position *
            (invStd *
              (gradOutput position * gamma - meanGradGamma -
                xHat position * meanGradGammaXHat))) =
        scale * (∑ position, tangent position * gradOutput position) -
          invStd * meanGradGamma * (∑ position, tangent position) -
          invStd * meanGradGammaXHat *
            (∑ position, tangent position * xHat position) := by
    calc
      _ = ∑ position,
          (scale * (tangent position * gradOutput position) -
            (invStd * meanGradGamma) * tangent position -
            (invStd * meanGradGammaXHat) *
              (tangent position * xHat position)) := by
              apply Finset.sum_congr rfl
              intro position _
              dsimp [scale]
              ring
      _ = _ := by
        simp only [Finset.sum_sub_distrib, ← Finset.mul_sum]
  change
    (∑ position,
        (invStd *
              (tangent position - meanTangent - xHat position * meanTangentXHat) * gamma +
            xHat position * dgamma + dbeta) * gradOutput position) = _
  rw [hForward, hBackward]
  dsimp [meanTangent, meanTangentXHat, meanGradGamma, meanGradGammaXHat]
  have hGradGamma :
      (∑ k, gradOutput k * gamma) = (∑ k, gradOutput k) * gamma := by
    simpa using (Finset.sum_mul Finset.univ gradOutput gamma).symm
  have hGradGammaXHat :
      (∑ k, gradOutput k * gamma * xHat k) =
        gamma * (∑ k, gradOutput k * xHat k) := by
    calc
      _ = ∑ k, gamma * (gradOutput k * xHat k) := by
        apply Finset.sum_congr rfl
        intro k _
        ring
      _ = _ := by
        simpa using
          (Finset.mul_sum Finset.univ (fun k => gradOutput k * xHat k) gamma).symm
  have hXHatGrad :
      (∑ k, xHat k * gradOutput k) = ∑ k, gradOutput k * xHat k := by
    apply Finset.sum_congr rfl
    intro k _
    ring
  rw [hGradGamma, hGradGammaXHat, hXHatGrad]
  dsimp [scale]
  have hPositionsNe : (positions : ℝ) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt hPositions)
  field_simp

/-- On a `[channels, positions]` matrix with at least one position, the normalized BatchNorm
reverse rule `normalizedBackward` is adjoint to its forward differential `normalizedJVP`. -/
theorem normalizedJVP_adjoint
    {channels positions : Nat} (hPositions : 0 < positions)
    (tangent gradOutput xHat : Tensor ℝ [channels, positions])
    (invStd gamma dgamma dbeta : Tensor ℝ [channels]) :
    Rep.dot (BatchNorm.normalizedJVP tangent xHat invStd gamma dgamma dbeta)
        gradOutput =
      let backward := BatchNorm.normalizedBackward gradOutput xHat invStd gamma
      Rep.dot tangent backward.1 + Rep.dot dgamma backward.2.1 + Rep.dot dbeta backward.2.2 := by
  rw [dot_mat_eq_sum]
  simp only [BatchNorm.normalizedBackward]
  rw [dot_mat_eq_sum]
  erw [Tensor.dot_vec_eq_sum, Tensor.dot_vec_eq_sum]
  simp only [BatchNorm.normalizedJVP]
  have hb (v : Tensor ℝ ([channels, positions].eraseIdx 1)) (c : Fin channels)
      (p : Fin positions) :
      Tensor.get2 (Tensor.broadcast v [channels, positions] _
          (Shape.eraseIdx_broadcastDims [channels, positions] 1)) c p =
        getScalar (n := channels) v c :=
    Normalization.get2_rowBroadcast v c p
  have hb' (v : Tensor ℝ [channels]) (c : Fin channels) (p : Fin positions) :
      Tensor.get2 (Tensor.broadcast (s := [channels, positions].eraseIdx 1) v
          [channels, positions] _ (Shape.eraseIdx_broadcastDims [channels, positions] 1)) c p =
        getScalar v c :=
    Normalization.get2_rowBroadcast v c p
  have hs (x : Tensor ℝ [channels, positions]) (c : Fin channels) :
      getScalar (n := channels) (castShape (target := [channels, positions].eraseIdx 1)
          (reduce .sum x [⟨1, Nat.one_lt_two⟩] false) (Shape.reduce_singleton _ _)) c =
        ∑ k : Fin positions, Tensor.get2 x c k :=
    Normalization.getScalar_rowSum x c
  have hm (x : Tensor ℝ [channels, positions]) (c : Fin channels) :
      getScalar (n := channels) (Tensor.map (shape := [channels, positions].eraseIdx 1)
          (Context.divNat · positions) (castShape (target := [channels, positions].eraseIdx 1)
            (reduce .sum x [⟨1, Nat.one_lt_two⟩] false) (Shape.reduce_singleton _ _))) c =
        (∑ k : Fin positions, Tensor.get2 x c k) / positions :=
    Normalization.getScalar_rowMean x c
  simp only [get2_add, get2_sub, Tensor.get2_mul, hb, hb', hm, hs]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro channel _
  exact channel_normalization_adjoint positions hPositions
    (fun position => get2 tangent channel position)
    (fun position => get2 gradOutput channel position)
    (fun position => get2 xHat channel position)
    (getScalar invStd channel) (getScalar gamma channel)
    (getScalar dgamma channel) (getScalar dbeta channel)

private theorem normalizedJVP_spatial_adjoint
    {channels positions : Nat} {sSpatial : Shape}
    (hPositions : 0 < positions)
    (hReshape : Shape.size (Shape.dim channels sSpatial) =
      Shape.size (Shape.dim channels (.dim positions .scalar)))
    (tangent gradOutput : Tensor ℝ (.dim channels sSpatial))
    (xHat : Tensor ℝ [channels, positions])
    (invStd gamma dgamma dbeta : Tensor ℝ [channels]) :
    Rep.dot
        (Tensor.reshape (BatchNorm.normalizedJVP (Tensor.reshape tangent _ hReshape)
            xHat invStd gamma dgamma dbeta) _ hReshape.symm)
        gradOutput =
      let backward := BatchNorm.normalizedBackward
        (Tensor.reshape gradOutput _ hReshape) xHat invStd gamma
      Rep.dot tangent (Tensor.reshape backward.1 _ hReshape.symm) +
        Rep.dot dgamma backward.2.1 + Rep.dot dbeta backward.2.2 := by
  rw [dot_reshape_left]
  let backward := BatchNorm.normalizedBackward
    (Tensor.reshape gradOutput _ hReshape) xHat invStd gamma
  change
    Rep.dot
        (BatchNorm.normalizedJVP (Tensor.reshape tangent _ hReshape)
          xHat invStd gamma dgamma dbeta)
        (Tensor.reshape gradOutput _ hReshape) =
      Rep.dot tangent (Tensor.reshape backward.1 _ hReshape.symm) +
        Rep.dot dgamma backward.2.1 + Rep.dot dbeta backward.2.2
  rw [← dot_reshape_left tangent backward.1 hReshape]
  exact normalizedJVP_adjoint hPositions
    (Tensor.reshape tangent _ hReshape) (Tensor.reshape gradOutput _ hReshape)
    xHat invStd gamma dgamma dbeta

/--
The rank-general BatchNorm reverse rule is adjoint to its forward differential, for every
`epsilon`.

It is the correctness theorem for the public channel-first operator, and the conclusion accounts
for the input tangent and both affine parameters. The spatial shape is arbitrary: the definitions
flatten it to a `[channels, positions]` matrix, compute there and reshape back.
-/
theorem jvp_backward_adjoint
    {channels : Nat} {sSpatial : Shape}
    (x tangent gradOutput : Tensor ℝ (.dim channels sSpatial))
    (gamma dgamma beta dbeta : Tensor ℝ [channels])
    (epsilon : ℝ := normalizationEpsilon)
    [Shape.WellFormed (.dim channels sSpatial)] :
    Rep.dot (jvp x tangent gamma dgamma beta dbeta epsilon) gradOutput =
      let backward := BatchNorm.backward x gamma gradOutput epsilon
      Rep.dot tangent backward.inputGradient +
        Rep.dot dgamma backward.scaleGradient +
        Rep.dot dbeta backward.biasGradient := by
  unfold jvp BatchNorm.backward
  exact normalizedJVP_spatial_adjoint
    (Shape.size_pos_of_positive (Shape.WellFormed.proof (s := .dim channels sSpatial)).2)
    _ _ _ _ _ _ _ _

end
end BatchNorm
end TorchLean
