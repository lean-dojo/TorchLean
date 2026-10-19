/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Common

/-!
# Phased α-CROWN step function: case inversion

Lemmas that read off what `stepPhased` computed at a node.

Away from ReLU nodes that carry phases, `stepPhased` is `stepAlpha`. On a ReLU node with
phases, the step succeeds only when the unary parent has a certificate entry and an IBP box of
the same output dimension, and `phaseRelaxVec?` accepts the phases against that box. The
inversion lemma `stepPhased_relu_phase_inv` exposes exactly those ingredients, together with the
range of the α vector actually used (explicit or default).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness
open CertSoundness

/-! ## Reduction to α-CROWN -/

/--
`stepPhased` is `stepAlpha` at every node that is not a ReLU carrying phases.

The hypothesis is phrased as an implication so that a single lemma covers both the non-ReLU
kinds and ReLU nodes with no phases.
-/
theorem stepPhased_eq_stepAlpha
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (phases : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (ctx : AffineCtx) (id : Nat)
    (hnoPhases : (g.nodes[id]!).kind = .relu → getPhases? phases id = none) :
    stepPhased g ps ibp alpha phases ctx cert id = stepAlpha g ps ibp alpha ctx cert id := by
  cases hk : (g.nodes[id]!).kind
  case relu => simp [stepPhased, stepAlpha, phasedCrownStepNode?, hk, hnoPhases hk]
  all_goals simp [stepPhased, stepAlpha, phasedCrownStepNode?, hk]

/-! ## Inversion of the phased ReLU branch -/

/--
What a successful phased step on a ReLU node with phases must have computed.

The parent `p1` has a certificate entry `xin` and an IBP box `preB` of matching output
dimension, `phaseRelaxVec?` accepted `nodePhases` against `preB` for some α vector `αt` with entries
in `[0, 1]` (the explicit one, cast to `preB.dim`, or the default one), and the produced bounds
are the phase relaxations propagated through the parent's affine bounds.
-/
theorem stepPhased_relu_phase_inv
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (phases : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (ctx : AffineCtx) (id : Nat)
    (b : FlatAffineBounds ℝ) (nodePhases : Array Int) (p1 : Nat)
    (halpha : AlphaOK (alpha := alpha))
    (hk : (g.nodes[id]!).kind = .relu)
    (hphases : getPhases? phases id = some nodePhases)
    (hps : NN.IR.unaryParent? (g.nodes[id]!).parents = some p1)
    (hs : stepPhased g ps ibp alpha phases ctx cert id = some b) :
    ∃ (xin : FlatAffineBounds ℝ) (preB : FlatBox ℝ) (hout : xin.outDim = preB.dim)
      (αt : Tensor ℝ [preB.dim])
      (relaxLo relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax ℝ) [preB.dim]),
      NN.MLTheory.CROWN.Cert.getAff? (α := ℝ) cert p1 = some xin ∧
      ibp[p1]! = some preB ∧
      (∀ i : Fin preB.dim, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ (1 : ℝ)) ∧
      phaseRelaxVec? (α := ℝ) (n := preB.dim) preB.lo preB.hi αt nodePhases =
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
  simp only [stepPhased, phasedCrownStepNode?, hk, hphases, hps] at hs'
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
              (castDimScalar (α := ℝ) hα αv.v) nodePhases with
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
            (defaultAlphaVec (α := ℝ) (n := preB.dim) preB.lo preB.hi) nodePhases with
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
