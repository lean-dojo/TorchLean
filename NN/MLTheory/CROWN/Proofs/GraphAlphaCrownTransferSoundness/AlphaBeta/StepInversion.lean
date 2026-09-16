/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Common

/-!
# α/β-CROWN step function: case inversion

Lemmas that read off what `stepAlphaBeta` computed at a node.

Away from ReLU nodes that carry a β vector, `stepAlphaBeta` is `stepAlpha`. On a ReLU node with a
β vector, the step succeeds only when the unary parent has a certificate entry and an IBP box of
the same output dimension, and `phaseRelaxVec?` accepts the phases against that box. The
inversion lemma `stepAlphaBeta_relu_beta_inv` exposes exactly those ingredients, together with the
range of the α vector actually used (explicit or default).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open scoped BigOperators
open Proofs.TensorAlgebra

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness
open CertSoundness

/-! ## Safe lookups -/

/-- A successful `Cert.getAff?` lookup is an in-bounds array read. -/
theorem getElem!_of_getAff?_eq_some {cert : Array (Option (FlatAffineBounds ℝ))} {p : Nat}
    {xin : FlatAffineBounds ℝ} (h : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p = some xin) :
    cert[p]! = some xin := by
  by_cases hlt : p < cert.size
  · simpa [NN.MLTheory.CROWN.Cert.getAff?, hlt] using h
  · simp [NN.MLTheory.CROWN.Cert.getAff?, hlt] at h

/-!
`getElem!_of_getVal?_eq_some` is `CertSoundness`'s lemma, opened at the top of this file. Two files
in this directory carried their own copy of it, proved with `by_cases` where the original uses
`unfold` and `split`; the statements were the same, so the copies are gone.
-/
/-- A successful `getAlpha?` lookup is an in-bounds array read. -/
theorem getElem!_of_getAlpha?_eq_some {alpha : Array (Option (FlatTensor ℝ))} {id : Nat}
    {αv : FlatTensor ℝ} (h : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    id < alpha.size ∧ alpha[id]! = some αv := by
  by_cases hlt : id < alpha.size
  · exact ⟨hlt, by simpa [NN.MLTheory.CROWN.Cert.getAlpha?, hlt] using h⟩
  · simp [NN.MLTheory.CROWN.Cert.getAlpha?, hlt] at h

/-! ## α vector range -/

/-- Under `AlphaOK`, an α vector read by `getAlpha?` has every entry in `[0, 1]`. -/
theorem getAlpha?_unit_range {alpha : Array (Option (FlatTensor ℝ))} {id : Nat}
    {αv : FlatTensor ℝ} (halpha : AlphaOK (alpha := alpha))
    (h : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    ∀ i : Fin αv.n, (0 : ℝ) ≤ getScalar αv.v i ∧ getScalar αv.v i ≤ (1 : ℝ) := by
  obtain ⟨hlt, hentry⟩ := getElem!_of_getAlpha?_eq_some h
  simpa [hentry] using halpha id hlt

/-- Casting the dimension of a tensor preserves the unit-interval range of its entries. -/
theorem castDimScalar_unit_range {n n' : Nat} (h : n = n') (t : Tensor ℝ [n])
    (hr : ∀ i : Fin n, (0 : ℝ) ≤ getScalar t i ∧ getScalar t i ≤ (1 : ℝ)) :
    ∀ i : Fin n', (0 : ℝ) ≤ getScalar (castDimScalar (α := ℝ) h t) i ∧
      getScalar (castDimScalar (α := ℝ) h t) i ≤ (1 : ℝ) := by
  intro i
  rw [getScalar_castDimScalar]
  exact hr _

/-! ## Reduction to α-CROWN -/

/--
`stepAlphaBeta` is `stepAlpha` at every node that is not a ReLU carrying a β vector.

The hypothesis is phrased as an implication so that a single lemma covers both the non-ReLU
kinds and ReLU nodes with no β entry.
-/
theorem stepAlphaBeta_eq_stepAlpha
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (ctx : AffineCtx) (id : Nat)
    (hnb : (g.nodes[id]!).kind = .relu → getBeta? (beta := beta) id = none) :
    stepAlphaBeta g ps ibp alpha beta ctx cert id = stepAlpha g ps ibp alpha ctx cert id := by
  cases hk : (g.nodes[id]!).kind
  case relu => simp [stepAlphaBeta, stepAlpha, alphaBetaCrownStepNode?, hk, hnb hk]
  all_goals simp [stepAlphaBeta, stepAlpha, alphaBetaCrownStepNode?, hk]

/-! ## Inversion of the β-phase ReLU branch -/

/--
What a successful α/β step on a ReLU node with a β vector must have computed.

The parent `p1` has a certificate entry `xin` and an IBP box `preB` of matching output
dimension, `phaseRelaxVec?` accepted `phases` against `preB` for some α vector `αt` with entries
in `[0, 1]` (the explicit one, cast to `preB.dim`, or the default one), and the produced bounds
are the phase relaxations propagated through the parent's affine bounds.
-/
theorem stepAlphaBeta_relu_beta_inv
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (ctx : AffineCtx) (id : Nat)
    (b : FlatAffineBounds ℝ) (phases : Array Int) (p1 : Nat)
    (halpha : AlphaOK (alpha := alpha))
    (hk : (g.nodes[id]!).kind = .relu)
    (hbeta : getBeta? (beta := beta) id = some phases)
    (hps : NN.IR.unaryParent? (g.nodes[id]!).parents = some p1)
    (hs : stepAlphaBeta g ps ibp alpha beta ctx cert id = some b) :
    ∃ (xin : FlatAffineBounds ℝ) (preB : FlatBox ℝ) (hout : xin.outDim = preB.dim)
      (αt : Tensor ℝ [preB.dim])
      (relaxLo relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) [preB.dim]),
      NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = some xin ∧
      ibp[p1]! = some preB ∧
      (∀ i : Fin preB.dim, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ)) ∧
      phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo preB.hi αt phases =
        some (relaxLo, relaxHi) ∧
      b = { inDim := xin.inDim
            outDim := preB.dim
            loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
              (inDim := xin.inDim) (hidDim := preB.dim) relaxLo
              (NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) hout xin.loAff)
            hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := ℝ)
              (inDim := xin.inDim) (hidDim := preB.dim) relaxHi
              (NN.MLTheory.CROWN.Graph.castAffineOut (α := ℝ) hout xin.hiAff) } := by
  have hs' := hs
  simp only [stepAlphaBeta, alphaBetaCrownStepNode?, hk, hbeta, hps] at hs'
  cases hxin : NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 with
  | none => simp [hxin] at hs'
  | some xin =>
  cases hpre : ibp[p1]! with
  | none => simp [hxin, hpre] at hs'
  | some preB =>
  refine ⟨xin, preB, ?_⟩
  cases hαopt : NN.MLTheory.CROWN.Cert.getAlpha? (α := ℝ) alpha id with
  | some αv =>
      simp only [hxin, hpre, hαopt] at hs'
      by_cases hout : xin.outDim = preB.dim
      · by_cases hα : αv.n = preB.dim
        · rw [dite_eq_left hout, dite_eq_left hα] at hs'
          cases hrelax : phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo preB.hi
              (castDimScalar (α := ℝ) hα αv.v) phases with
          | none => simp [hrelax] at hs'
          | some rpair =>
              obtain ⟨relaxLo, relaxHi⟩ := rpair
              simp only [hrelax] at hs'
              exact ⟨hout, castDimScalar (α := ℝ) hα αv.v, relaxLo, relaxHi, rfl, rfl,
                castDimScalar_unit_range hα αv.v (getAlpha?_unit_range halpha hαopt), hrelax,
                (Option.some.inj hs').symm⟩
        · rw [dite_eq_left hout, dite_eq_right hα] at hs'
          cases hs'
      · rw [dite_eq_right hout] at hs'
        cases hs'
  | none =>
      simp only [hxin, hpre, hαopt] at hs'
      by_cases hout : xin.outDim = preB.dim
      · rw [dite_eq_left hout] at hs'
        cases hrelax : phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo preB.hi
            (defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi) phases with
        | none => simp [hrelax] at hs'
        | some rpair =>
            obtain ⟨relaxLo, relaxHi⟩ := rpair
            simp only [hrelax] at hs'
            exact ⟨hout, defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi, relaxLo,
              relaxHi, rfl, rfl, defaultAlphaVec_range preB.lo preB.hi, hrelax,
              (Option.some.inj hs').symm⟩
      · rw [dite_eq_right hout] at hs'
        cases hs'

end

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
