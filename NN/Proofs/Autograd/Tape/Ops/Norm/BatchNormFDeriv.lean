/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.BatchNorm
public import NN.Proofs.Autograd.Tape.Ops.Norm.MatrixEntries
public import NN.Proofs.Autograd.Tape.Ops.Norm.RowNormalization

/-!
# Fréchet derivative of BatchNorm

`Spec.batchNorm` flattens the spatial axes of a channel-first tensor to a `[channels, positions]`
matrix, normalizes every row of that matrix with the row statistics, and applies the per-channel
affine parameters. This file proves that, for `0 < ε`, the map `(x, gamma, beta) ↦ batchNorm` is
Fréchet differentiable on the flattened vectors and that its derivative is exactly
`Spec.batchNormJvp`.

The calculus is delegated to `RowNorm.hasFDerivAt_nrm`, the derivative of one normalized row entry.
The rest of the file is bookkeeping: reshaping commutes with pointwise operations, the clamps in
the spec are inactive because the variance is a mean of squares, and every entry of the JVP is the
closed-form row differential.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace BatchNorm

open Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open _root_.Proofs.Autograd.Norm _root_.Proofs.Autograd.RowNorm TapeNodes.Matmul

open scoped BigOperators

noncomputable section

/-! ## Reshaping -/

/-- Reshaping commutes with pointwise binary operations. -/
theorem reshapeSpec_map2Spec {s₁ s₂ : Shape} (f : ℝ → ℝ → ℝ) (a b : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    reshapeSpec (map2Spec f a b) h = map2Spec f (reshapeSpec a h) (reshapeSpec b h) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro c
  simp [reshapeSpec, map2Spec]

/-- Reshaping commutes with pointwise unary operations. -/
theorem reshapeSpec_mapSpec {s₁ s₂ : Shape} (f : ℝ → ℝ) (a : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    reshapeSpec (mapSpec f a) h = mapSpec f (reshapeSpec a h) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro c
  simp [reshapeSpec, mapSpec, Tensor.map]

/-- Reshaping a filled tensor fills the target shape. -/
theorem reshapeSpec_full {s₁ s₂ : Shape} (v : ℝ) (h : s₁.size = s₂.size) :
    reshapeSpec (Tensor.full s₁ v) h = Tensor.full s₂ v := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro c
  simp [reshapeSpec, Tensor.full]

/-- Reshaping back and forth is the identity, for any proofs of the size equalities. -/
theorem reshapeSpec_reshapeSpec {s₁ s₂ : Shape} (t : Tensor ℝ s₁) (h : s₁.size = s₂.size)
    (h' : s₂.size = s₁.size) :
    reshapeSpec (reshapeSpec t h) h' = t :=
  reshapeSpec_roundtrip t h

/-- Reshaping commutes with tensor addition. -/
theorem reshapeSpec_addSpec {s₁ s₂ : Shape} (a b : Tensor ℝ s₁) (h : s₁.size = s₂.size) :
    reshapeSpec (addSpec a b) h = addSpec (reshapeSpec a h) (reshapeSpec b h) :=
  reshapeSpec_map2Spec _ a b h

/-- Reshaping commutes with tensor subtraction. -/
theorem reshapeSpec_subSpec {s₁ s₂ : Shape} (a b : Tensor ℝ s₁) (h : s₁.size = s₂.size) :
    reshapeSpec (subSpec a b) h = subSpec (reshapeSpec a h) (reshapeSpec b h) :=
  reshapeSpec_map2Spec _ a b h

/-- Reshaping commutes with pointwise multiplication. -/
theorem reshapeSpec_mulSpec {s₁ s₂ : Shape} (a b : Tensor ℝ s₁) (h : s₁.size = s₂.size) :
    reshapeSpec (mulSpec a b) h = mulSpec (reshapeSpec a h) (reshapeSpec b h) :=
  reshapeSpec_map2Spec _ a b h

/-- Reshaping commutes with pointwise division. -/
theorem reshapeSpec_divSpec {s₁ s₂ : Shape} (a b : Tensor ℝ s₁) (h : s₁.size = s₂.size) :
    reshapeSpec (divSpec a b) h = divSpec (reshapeSpec a h) (reshapeSpec b h) :=
  reshapeSpec_map2Spec _ a b h

/-- Reshaping commutes with the clamped square root. -/
theorem reshapeSpec_sqrtSpec {s₁ s₂ : Shape} (a : Tensor ℝ s₁) (h : s₁.size = s₂.size) :
    reshapeSpec (sqrtSpec a) h = sqrtSpec (reshapeSpec a h) :=
  reshapeSpec_mapSpec _ a h

/-- Transporting a vector along a length equality reindexes its Euclidean view. -/
theorem getScalarE_cast {a b : Nat} (h : a = b) (v : Tensor ℝ [a]) (k : Fin b) :
    (getScalarE (h ▸ v)) k = getScalarE v (Fin.cast h.symm k) := by
  subst h
  rfl

/-- `tensorToVec` of a reshaped tensor is a reindexing of `tensorToVec`. -/
theorem tensorToVec_reshapeSpec {s₁ s₂ : Shape} (t : Tensor ℝ s₁) (h : s₁.size = s₂.size)
    (k : Fin (Spec.Shape.size s₂)) :
    tensorToVec (t := reshapeSpec t h) k = tensorToVec (t := t) (Fin.cast h.symm k) := by
  simp only [tensorToVec, flatten_reshapeSpec]
  exact getScalarE_cast h _ k

/-- Entries of the clamped square root of a matrix. -/
theorem get2_sqrtSpec {m n : Nat} (a : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    Spec.get2 (sqrtSpec a) i j = Real.sqrt (max (Spec.get2 a i j) 0) := by
  simp [sqrtSpec]
  rfl

/-- Row means when the axis is written as `rank - 1`, as in the spec. -/
theorem getScalar_reduceMean_rank {m n : Nat} (x : Tensor ℝ [m, n])
    (h : Shape.NonemptyAxis (Spec.Shape.rank (.dim m (.dim n .scalar)) - 1)
      (.dim m (.dim n .scalar))) (i : Fin m) :
    getScalar (reduceMean (Spec.Shape.rank (.dim m (.dim n .scalar)) - 1) x h) i =
      (∑ j : Fin n, Spec.get2 x i j) / n :=
  getScalar_reduceMean_one x h i

/-! ## The flattened matrix form -/

variable {channels : Nat} {sSpatial : Shape}

/-- Number of positions per channel. -/
abbrev positions (sSpatial : Shape) : Nat := Spec.Shape.size sSpatial

/-- Flattening the spatial axes preserves the size. -/
theorem size_flat (channels : Nat) (sSpatial : Shape) :
    Spec.Shape.size (.dim channels sSpatial) =
      Spec.Shape.size (.dim channels (.dim (positions sSpatial) .scalar)) := by
  simp [Spec.Shape.size, positions]

/-- The number of positions is positive for a well-formed input shape. -/
theorem positions_pos (channels : Nat) [Shape.WellFormed (.dim channels sSpatial)] :
    0 < positions sSpatial :=
  Shape.size_pos_of_well_formed (Shape.WellFormed.proof (s := .dim channels sSpatial)).2

/-- The input flattened to a `[channels, positions]` matrix. -/
def flat (x : Tensor ℝ (.dim channels sSpatial)) : Tensor ℝ [channels, positions sSpatial] :=
  reshapeSpec x (size_flat channels sSpatial)

/-- Row broadcast of a channel vector to the flattened matrix shape. -/
abbrev bAS (v : Tensor ℝ [channels]) : Tensor ℝ [channels, positions sSpatial] :=
  broadcastAfterSum (.dim channels (.dim (positions sSpatial) .scalar)) 1 v

/-- `Spec.batchNorm` with its axis and reshape evidence spelled out. -/
def batchNormExplicit (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels])
    (ε : ℝ) (hP : 0 < positions sSpatial) : Tensor ℝ (.dim channels sSpatial) :=
  let mean : Tensor ℝ [channels] := reduceMean 1 (flat x) (nonemptyAxis_one hP)
  let centered := subSpec (flat x) (bAS (sSpatial := sSpatial) mean)
  let variance :=
    maxSpec (reduceMean 1 (mulSpec centered centered) (nonemptyAxis_one hP))
      (Tensor.full (.dim channels .scalar) 0)
  let meanB := reshapeSpec (bAS (sSpatial := sSpatial) mean) (size_flat channels sSpatial).symm
  let varianceB :=
    reshapeSpec (bAS (sSpatial := sSpatial) variance) (size_flat channels sSpatial).symm
  let gammaB := reshapeSpec (bAS (sSpatial := sSpatial) gamma) (size_flat channels sSpatial).symm
  let betaB := reshapeSpec (bAS (sSpatial := sSpatial) beta) (size_flat channels sSpatial).symm
  addSpec
    (mulSpec
      (divSpec (subSpec x meanB)
        (sqrtSpec (addSpec varianceB (Tensor.full (.dim channels sSpatial) ε))))
      gammaB)
    betaB

/-- `Spec.batchNorm` is `batchNormExplicit` by unfolding. -/
theorem batchNorm_eq_explicit [Shape.WellFormed (.dim channels sSpatial)]
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) (ε : ℝ) :
    Spec.batchNorm x gamma beta ε = batchNormExplicit x gamma beta ε (positions_pos channels) :=
  rfl

/-- Entries of `Spec.batchNorm` in the flattened matrix: each row is centered, divided by the
clamped standard deviation, scaled and shifted by the channel parameters. -/
theorem get2_flat_batchNorm [Shape.WellFormed (.dim channels sSpatial)]
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) (ε : ℝ)
    (c : Fin channels) (p : Fin (positions sSpatial)) :
    Spec.get2 (reshapeSpec (Spec.batchNorm x gamma beta ε) (size_flat channels sSpatial)) c p =
      (Spec.get2 (flat x) c p - rowMeanE (flat x) c) /
          Real.sqrt (max (max (rowVarE (flat x) c) 0 + ε) 0) * getScalar gamma c +
        getScalar beta c := by
  rw [batchNorm_eq_explicit]
  simp only [batchNormExplicit]
  rw [reshapeSpec_addSpec, reshapeSpec_mulSpec, reshapeSpec_divSpec, reshapeSpec_subSpec,
    reshapeSpec_sqrtSpec, reshapeSpec_addSpec, reshapeSpec_full, reshapeSpec_reshapeSpec,
    reshapeSpec_reshapeSpec, reshapeSpec_reshapeSpec, reshapeSpec_reshapeSpec]
  simp only [get2_addSpec, get2_mulSpec, get2_divSpec, get2_subSpec, get2_sqrtSpec, get2_full,
    get2_broadcastAfterSum_one, getScalar_maxSpec, Norm.getScalar_full, getScalar_reduceMean_one,
    flat, rowMeanE, rowVarE]

/-- `Spec.batchNormJvp` with its axis and reshape evidence spelled out. -/
def batchNormJvpExplicit (x dx : Tensor ℝ (.dim channels sSpatial))
    (gamma dgamma dbeta : Tensor ℝ [channels]) (ε : ℝ) (hP : 0 < positions sSpatial) :
    Tensor ℝ (.dim channels sSpatial) :=
  let mean : Tensor ℝ [channels] := reduceMean 1 (flat x) (nonemptyAxis_one hP)
  let centered := subSpec (flat x) (bAS (sSpatial := sSpatial) mean)
  let variance :=
    maxSpec (reduceMean 1 (mulSpec centered centered) (nonemptyAxis_one hP))
      (Tensor.full (.dim channels .scalar) 0)
  let invStd :=
    divSpec (Tensor.full (.dim channels .scalar) 1)
      (sqrtSpec (addSpec variance (Tensor.full (.dim channels .scalar) ε)))
  let xHat := mulSpec centered (bAS (sSpatial := sSpatial) invStd)
  reshapeSpec (Spec.BatchNorm.normalizedJvp hP (flat dx) xHat invStd gamma dgamma dbeta)
    (size_flat channels sSpatial).symm

/-- `Spec.batchNormJvp` is `batchNormJvpExplicit` by unfolding. -/
theorem batchNormJvp_eq_explicit [Shape.WellFormed (.dim channels sSpatial)]
    (x dx : Tensor ℝ (.dim channels sSpatial)) (gamma dgamma beta dbeta : Tensor ℝ [channels])
    (ε : ℝ) :
    Spec.batchNormJvp x dx gamma dgamma beta dbeta ε =
      batchNormJvpExplicit x dx gamma dgamma dbeta ε (positions_pos channels) :=
  rfl

/-- Entries of the normalized-data JVP. -/
theorem get2_normalizedJvp {C P : Nat} (hP : 0 < P) (t xHat : Tensor ℝ [C, P])
    (invStd gamma dgamma dbeta : Tensor ℝ [C]) (c : Fin C) (p : Fin P) :
    Spec.get2 (Spec.BatchNorm.normalizedJvp hP t xHat invStd gamma dgamma dbeta) c p =
      getScalar invStd c *
          (Spec.get2 t c p - (∑ k : Fin P, Spec.get2 t c k) / P -
            Spec.get2 xHat c p * ((∑ k : Fin P, Spec.get2 t c k * Spec.get2 xHat c k) / P)) *
          getScalar gamma c +
        Spec.get2 xHat c p * getScalar dgamma c + getScalar dbeta c := by
  simp only [Spec.BatchNorm.normalizedJvp, get2_addSpec, get2_mulSpec, get2_subSpec,
    get2_broadcastAfterSum_one, getScalar_reduceMean_one]

/-! ## Vector form -/

/-- Domain of the vectorized BatchNorm: flattened input and the two channel parameters. -/
abbrev Domain (channels : Nat) (sSpatial : Shape) : Type :=
  Vec (Spec.Shape.size (.dim channels sSpatial)) × Vec (vecSize channels) × Vec (vecSize channels)

/-- `Spec.batchNorm` on flattened vectors. -/
def bnVec [Shape.WellFormed (.dim channels sSpatial)] (ε : ℝ) (q : Domain channels sSpatial) :
    Vec (Spec.Shape.size (.dim channels sSpatial)) :=
  tensorToVec (Spec.batchNorm (vecToTensor (s := .dim channels sSpatial) q.1)
    (vecToTensor (s := .dim channels .scalar) q.2.1)
    (vecToTensor (s := .dim channels .scalar) q.2.2) ε)

/-- Flattened row index of a coordinate of the input. -/
def rowIdx (k : Fin (Spec.Shape.size (.dim channels sSpatial))) : Fin channels :=
  rowOf (Fin.cast (size_flat channels sSpatial) k)

/-- Flattened column index of a coordinate of the input. -/
def colIdx (k : Fin (Spec.Shape.size (.dim channels sSpatial))) : Fin (positions sSpatial) :=
  colOf (Fin.cast (size_flat channels sSpatial) k)

/-- Channel index of a coordinate of the input, as an index into a channel vector. -/
def chanIdx (k : Fin (Spec.Shape.size (.dim channels sSpatial))) : Fin (vecSize channels) :=
  Fin.cast (vecSize_eq channels).symm (rowIdx k)

/-- The input as a flattened `channels × positions` matrix vector. -/
def matVec (v : Vec (Spec.Shape.size (.dim channels sSpatial))) :
    Vec (matSize channels (positions sSpatial)) :=
  castVec (size_flat channels sSpatial) v

/-- Every input coordinate is `idxMN` of its row and column, up to the size cast. -/
theorem cast_eq_idxMN (k : Fin (Spec.Shape.size (.dim channels sSpatial))) :
    Fin.cast (size_flat channels sSpatial) k =
      idxMN (m := channels) (n := positions sSpatial) (rowIdx k) (colIdx k) :=
  (idxMN_rowOf_colOf _).symm

/-- Entries of a flattened tensor are coordinates of `matVec` of its vectorization. -/
theorem get2_flat (t : Tensor ℝ (.dim channels sSpatial)) (c : Fin channels)
    (p : Fin (positions sSpatial)) :
    Spec.get2 (flat t) c p =
      matVec (tensorToVec t) (idxMN (m := channels) (n := positions sSpatial) c p) := by
  rw [← tensorToVec_idxMN, flat, tensorToVec_reshapeSpec, matVec, castVec_apply]

/-- Entries of the flattened input tensor are coordinates of `matVec`. -/
theorem get2_flat_vecToTensor (v : Vec (Spec.Shape.size (.dim channels sSpatial)))
    (c : Fin channels) (p : Fin (positions sSpatial)) :
    Spec.get2 (flat (vecToTensor (s := .dim channels sSpatial) v)) c p =
      matVec v (idxMN (m := channels) (n := positions sSpatial) c p) := by
  rw [get2_flat, tensorToVec_vecToTensor]

/-- Row means of a flattened tensor are `RowNorm.rowMean` of its vectorization. -/
theorem rowMeanE_flat (t : Tensor ℝ (.dim channels sSpatial)) (c : Fin channels) :
    rowMeanE (flat t) c = rowMean (matVec (tensorToVec t)) c := by
  simp only [rowMeanE, rowMean, get2_flat]

/-- Row variances of a flattened tensor are `RowNorm.rowVar` of its vectorization. -/
theorem rowVarE_flat (t : Tensor ℝ (.dim channels sSpatial)) (c : Fin channels) :
    rowVarE (flat t) c = rowVar (matVec (tensorToVec t)) c := by
  simp only [rowVarE, rowVar, centered, get2_flat, rowMeanE_flat]

/-- Entries of a channel vector are coordinates of its vectorization. -/
theorem getScalar_chan (g : Tensor ℝ [channels]) (c : Fin channels) :
    getScalar g c = tensorToVec g (Fin.cast (vecSize_eq channels).symm c) := by
  rw [tensorToVec_vec]
  rfl

/-- Closed form of the vectorized BatchNorm for positive `ε`. -/
def bnClosed (ε : ℝ) (q : Domain channels sSpatial) :
    Vec (Spec.Shape.size (.dim channels sSpatial)) :=
  vecOfFun (n := Spec.Shape.size (.dim channels sSpatial)) fun k =>
    nrm (matVec q.1) ε (rowIdx k) (colIdx k) * q.2.1 (chanIdx k) + q.2.2 (chanIdx k)

/-- For positive `ε` the clamps in `Spec.batchNorm` are inactive and the vectorized map is its
closed form. -/
theorem bnVec_eq_bnClosed [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε) :
    bnVec (channels := channels) (sSpatial := sSpatial) ε = bnClosed ε := by
  funext q
  apply PiLp.ext
  intro k
  have hL : (bnVec ε q).ofLp k =
      Spec.get2
        (reshapeSpec
          (Spec.batchNorm (vecToTensor (s := .dim channels sSpatial) q.1)
            (vecToTensor (s := .dim channels .scalar) q.2.1)
            (vecToTensor (s := .dim channels .scalar) q.2.2) ε)
          (size_flat channels sSpatial))
        (rowIdx k) (colIdx k) := by
    rw [← tensorToVec_idxMN, ← cast_eq_idxMN, tensorToVec_reshapeSpec]
    rfl
  rw [hL, get2_flat_batchNorm, get2_flat_vecToTensor, rowMeanE_flat, rowVarE_flat,
    tensorToVec_vecToTensor, getScalar_chan, getScalar_chan, tensorToVec_vecToTensor,
    tensorToVec_vecToTensor, max_eq_left (rowVar_nonneg (matVec q.1) (rowIdx k)),
    max_eq_left (rowVar_add_pos hε (matVec q.1) (rowIdx k)).le]
  simp only [bnClosed, vecOfFun_ofLp, nrm, centered, invStd, chanIdx, div_eq_mul_inv]

/-- `Graph.castCLM` acts by `castVec`. -/
theorem castCLM_apply {a b : Nat} (h : a = b) (v : Vec a) :
    Graph.castCLM (h := h) v = castVec h v :=
  rfl

/-- Derivative of the closed form at `(X, g, _)`, as a continuous linear map. -/
def bnD (X : Vec (matSize channels (positions sSpatial))) (g : Vec (vecSize channels)) (ε : ℝ) :
    Domain channels sSpatial →L[ℝ] Vec (Spec.Shape.size (.dim channels sSpatial)) := by
  classical
  let fLin : Domain channels sSpatial →ₗ[ℝ] Vec (Spec.Shape.size (.dim channels sSpatial)) :=
    { toFun := fun d =>
        vecOfFun (n := Spec.Shape.size (.dim channels sSpatial)) fun k =>
          nrmJvpCLM X ε (matVec d.1) (idxMN (m := channels) (n := positions sSpatial)
              (rowIdx k) (colIdx k)) * g (chanIdx k) +
            nrm X ε (rowIdx k) (colIdx k) * d.2.1 (chanIdx k) + d.2.2 (chanIdx k)
      map_add' := by
        intro a b
        apply PiLp.ext
        intro k
        simp only [vecOfFun_ofLp, matVec, Prod.fst_add, Prod.snd_add, castVec_add, map_add,
          PiLp.add_apply]
        ring
      map_smul' := by
        intro r a
        apply PiLp.ext
        intro k
        simp only [vecOfFun_ofLp, matVec, Prod.smul_fst, Prod.smul_snd, castVec_smul, map_smul,
          PiLp.smul_apply, smul_eq_mul, RingHom.id_apply]
        ring }
  exact { toLinearMap := fLin, cont := LinearMap.continuous_of_finiteDimensional fLin }

/-- Coordinates of the batch-norm differential, split into its three contributions.

Reading the summands left to right: the normalization's own Jacobian applied to the input
perturbation and then scaled by `γ`, the normalized activation times the perturbation of `γ`, and
the perturbation of `β`. That decomposition is why the three parameter groups can be differentiated
independently downstream. -/
@[simp] theorem bnD_apply (X : Vec (matSize channels (positions sSpatial)))
    (g : Vec (vecSize channels)) (ε : ℝ) (d : Domain channels sSpatial)
    (k : Fin (Spec.Shape.size (.dim channels sSpatial))) :
    bnD X g ε d k =
      nrmJvpCLM X ε (matVec d.1) (idxMN (m := channels) (n := positions sSpatial)
          (rowIdx k) (colIdx k)) * g (chanIdx k) +
        nrm X ε (rowIdx k) (colIdx k) * d.2.1 (chanIdx k) + d.2.2 (chanIdx k) := by
  simp [bnD]

/-- The closed form is differentiable for positive `ε`. -/
theorem hasFDerivAt_bnClosed (hP : 0 < positions sSpatial) {ε : ℝ} (hε : 0 < ε)
    (q : Domain channels sSpatial) :
    HasFDerivAt (bnClosed (channels := channels) (sSpatial := sSpatial) ε)
      (bnD (matVec q.1) q.2.1 ε) q := by
  rw [← hasFDerivWithinAt_univ, hasFDerivWithinAt_euclidean]
  intro k
  rw [hasFDerivWithinAt_univ]
  have hX : HasFDerivAt (fun d : Domain channels sSpatial => matVec d.1)
      ((Graph.castCLM (h := size_flat channels sSpatial)).comp
        (ContinuousLinearMap.fst ℝ (Vec (Spec.Shape.size (.dim channels sSpatial)))
          (Vec (vecSize channels) × Vec (vecSize channels)))) q :=
    (Graph.castCLM (h := size_flat channels sSpatial)).hasFDerivAt.comp q hasFDerivAt_fst
  have hnrm := (hasFDerivAt_nrm hε (matVec q.1) (rowIdx k) (colIdx k)).comp q hX
  have hg : HasFDerivAt (fun d : Domain channels sSpatial => d.2.1 (chanIdx k))
      ((EuclideanSpace.proj (𝕜 := ℝ) (chanIdx k) : Vec (vecSize channels) →L[ℝ] ℝ).comp
        ((ContinuousLinearMap.fst ℝ (Vec (vecSize channels)) (Vec (vecSize channels))).comp
          (ContinuousLinearMap.snd ℝ (Vec (Spec.Shape.size (.dim channels sSpatial)))
            (Vec (vecSize channels) × Vec (vecSize channels))))) q :=
    (EuclideanSpace.proj (𝕜 := ℝ) (chanIdx k)).hasFDerivAt.comp q
      (hasFDerivAt_fst.comp q hasFDerivAt_snd)
  have hb : HasFDerivAt (fun d : Domain channels sSpatial => d.2.2 (chanIdx k))
      ((EuclideanSpace.proj (𝕜 := ℝ) (chanIdx k) : Vec (vecSize channels) →L[ℝ] ℝ).comp
        ((ContinuousLinearMap.snd ℝ (Vec (vecSize channels)) (Vec (vecSize channels))).comp
          (ContinuousLinearMap.snd ℝ (Vec (Spec.Shape.size (.dim channels sSpatial)))
            (Vec (vecSize channels) × Vec (vecSize channels))))) q :=
    (EuclideanSpace.proj (𝕜 := ℝ) (chanIdx k)).hasFDerivAt.comp q
      (hasFDerivAt_snd.comp q hasFDerivAt_snd)
  have h := (hnrm.mul hg).add hb
  have hfun : (fun d : Domain channels sSpatial => bnClosed ε d k) =
      fun d : Domain channels sSpatial =>
        nrm (matVec d.1) ε (rowIdx k) (colIdx k) * d.2.1 (chanIdx k) + d.2.2 (chanIdx k) := by
    funext d
    simp [bnClosed]
  rw [hfun]
  refine h.congr_fderiv (ContinuousLinearMap.ext fun d => ?_)
  simp only [ContinuousLinearMap.comp_apply, _root_.add_apply, smul_apply, smul_eq_mul,
    PiLp.proj_apply,
    ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd', bnD_apply, nrmJvpCLM_apply,
    rowOf_idxMN, colOf_idxMN, nrmD_apply hP hε, castCLM_apply, matVec, Function.comp_def]
  ring

/-! ## Main theorems -/

/-- `Spec.batchNorm` is Fréchet differentiable in `(x, gamma, beta)` for positive `ε`. -/
theorem hasFDerivAt_batchNorm [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε)
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) :
    HasFDerivAt (bnVec (channels := channels) (sSpatial := sSpatial) ε)
      (bnD (matVec (tensorToVec x)) (tensorToVec gamma) ε)
      (tensorToVec x, tensorToVec gamma, tensorToVec beta) := by
  rw [bnVec_eq_bnClosed hε]
  exact hasFDerivAt_bnClosed (positions_pos channels) hε
    (tensorToVec x, tensorToVec gamma, tensorToVec beta)

/-- The vectorized BatchNorm at packed tensors is the vectorized spec output. -/
theorem bnVec_tensors [Shape.WellFormed (.dim channels sSpatial)] (ε : ℝ)
    (x : Tensor ℝ (.dim channels sSpatial)) (gamma beta : Tensor ℝ [channels]) :
    bnVec ε (tensorToVec x, tensorToVec gamma, tensorToVec beta) =
      tensorToVec (Spec.batchNorm x gamma beta ε) := by
  simp [bnVec]

/-- Entries of `Spec.batchNormJvp` in the flattened matrix are the closed-form row
differential, for positive `ε`. -/
theorem get2_flat_batchNormJvp [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε)
    (x dx : Tensor ℝ (.dim channels sSpatial)) (gamma dgamma beta dbeta : Tensor ℝ [channels])
    (c : Fin channels) (p : Fin (positions sSpatial)) :
    Spec.get2
        (reshapeSpec (Spec.batchNormJvp x dx gamma dgamma beta dbeta ε)
          (size_flat channels sSpatial)) c p =
      nrmJvp (matVec (tensorToVec x)) ε (matVec (tensorToVec dx)) c p * getScalar gamma c +
        nrm (matVec (tensorToVec x)) ε c p * getScalar dgamma c + getScalar dbeta c := by
  have hV := max_eq_left (rowVar_nonneg (matVec (tensorToVec x)) c)
  have hVε := max_eq_left (rowVar_add_pos hε (matVec (tensorToVec x)) c).le
  simp only [rowVar, centered, rowMean] at hV hVε
  rw [batchNormJvp_eq_explicit]
  simp only [batchNormJvpExplicit, reshapeSpec_reshapeSpec, get2_normalizedJvp, get2_mulSpec,
    get2_subSpec, get2_broadcastAfterSum_one, getScalar_divSpec, Norm.getScalar_full,
    getScalar_sqrtSpec, getScalar_addSpec', getScalar_maxSpec, getScalar_reduceMean_one, get2_flat]
  rw [hV, hVε]
  simp only [nrmJvp, nrm, centered, invStd, rowMean, rowVar, one_div]

/-- The derivative of BatchNorm applied to packed tangents is `Spec.batchNormJvp`. -/
theorem bnD_tensors [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ} (hε : 0 < ε)
    (x dx : Tensor ℝ (.dim channels sSpatial)) (gamma dgamma beta dbeta : Tensor ℝ [channels]) :
    bnD (matVec (tensorToVec x)) (tensorToVec gamma) ε
        (tensorToVec dx, tensorToVec dgamma, tensorToVec dbeta) =
      tensorToVec (Spec.batchNormJvp x dx gamma dgamma beta dbeta ε) := by
  apply PiLp.ext
  intro k
  have hRHS : (tensorToVec (Spec.batchNormJvp x dx gamma dgamma beta dbeta ε)).ofLp k =
      Spec.get2
        (reshapeSpec (Spec.batchNormJvp x dx gamma dgamma beta dbeta ε)
          (size_flat channels sSpatial)) (rowIdx k) (colIdx k) := by
    rw [← tensorToVec_idxMN, ← cast_eq_idxMN, tensorToVec_reshapeSpec]
    rfl
  rw [hRHS, get2_flat_batchNormJvp hε, bnD_apply, nrmJvpCLM_apply, rowOf_idxMN, colOf_idxMN,
    getScalar_chan, getScalar_chan, getScalar_chan]
  rfl

/-- The Fréchet derivative of `Spec.batchNorm` in `(x, gamma, beta)` applied to a tangent triple
is `Spec.batchNormJvp`, for positive `ε`. -/
theorem fderiv_batchNorm_eq_batchNormJvp [Shape.WellFormed (.dim channels sSpatial)] {ε : ℝ}
    (hε : 0 < ε) (x dx : Tensor ℝ (.dim channels sSpatial))
    (gamma dgamma beta dbeta : Tensor ℝ [channels]) :
    fderiv ℝ (bnVec (channels := channels) (sSpatial := sSpatial) ε)
        (tensorToVec x, tensorToVec gamma, tensorToVec beta)
        (tensorToVec dx, tensorToVec dgamma, tensorToVec dbeta) =
      tensorToVec (Spec.batchNormJvp x dx gamma dgamma beta dbeta ε) := by
  rw [(hasFDerivAt_batchNorm hε x gamma beta).fderiv, bnD_tensors hε]

end

end BatchNorm
end Autograd
end Proofs
