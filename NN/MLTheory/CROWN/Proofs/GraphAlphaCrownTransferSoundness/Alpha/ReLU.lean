/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Basic

/-!
# α-CROWN Transfer: ReLU Nodes

The `.relu` case of `alphaCrown_transfer_sound`. The step propagates the parent's affine bounds
through the α lower relaxation and the CROWN upper relaxation built from the IBP pre-activation
box. The proof is componentwise: scalar inequalities for the two relaxations, a vector-level
enclosure for a fixed α vector in `[0, 1]`, and the range facts for the α vector the step selects.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert
open CrownCertSoundness
open CertSoundness

/-! ## Scalar relaxation inequalities -/

/-- The α lower relaxation on `[l, u]`, evaluated at a lower estimate `lAff ≤ z`, lies below
`relu z`. -/
theorem alpha_relax_lower_le_relu (l u a lAff z : ℝ) (hl : l ≤ z) (hu : z ≤ u)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (hlz : lAff ≤ z) :
    (alphaRelaxLowerScalar (α := ℝ) l u a).slope * lAff +
        (alphaRelaxLowerScalar (α := ℝ) l u a).bias ≤
      Activation.Math.reluSpec (α := ℝ) z := by
  have hslope := alphaRelaxLowerScalar_slope_nonneg l u a ha0
  have hsound := NN.MLTheory.CROWN.Proofs.alphaRelaxLowerScalar_sound l u a z hl hu ha0 ha1
  exact le_trans (add_le_add (mul_le_mul_of_nonneg_left hlz hslope) le_rfl) hsound

/-- `relu z` lies below the CROWN upper relaxation on `[l, u]` evaluated at an upper estimate
`z ≤ uAff`. -/
theorem relu_le_relax_upper (l u uAff z : ℝ) (hl : l ≤ z) (hu : z ≤ u) (hzu : z ≤ uAff) :
    Activation.Math.reluSpec (α := ℝ) z ≤
      (Runtime.Ops.ReLU.relaxScalar (α := ℝ) l u).slope * uAff +
        (Runtime.Ops.ReLU.relaxScalar (α := ℝ) l u).bias := by
  have hslope := relax_scalar_slope_nonneg l u
  have hsound := relu_relax_scalar_upper_real_runtime l u z hl hu
  exact le_trans hsound (add_le_add (mul_le_mul_of_nonneg_left hzu hslope) le_rfl)

/-! ## Vector-level enclosure -/

/-- Componentwise ReLU bound: if `z` lies in both the affine estimate box and the IBP box, and the
α vector lies in `[0, 1]`, then the propagated lower and upper affine forms bracket `relu z` at
every index. -/
theorem relu_propagate_getScalar_bounds {inDim n : Nat} (lo hi αt z : Tensor ℝ [n])
    (xLo xHi : AffineVec ℝ inDim n) (x' : Tensor ℝ [inDim])
    (hαrange : ∀ i : Fin n, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ 1)
    (hzAff : Theorems.Semantics.encloses (α := ℝ)
      { dim := n, lo := affineEvalAt (α := ℝ) xLo x', hi := affineEvalAt (α := ℝ) xHi x' } z)
    (hzIbp : Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := lo, hi := hi } z) (i : Fin n) :
    getScalar (affineEvalAt (α := ℝ)
        (Runtime.Ops.ReLU.propagateAffine (α := ℝ) (alphaRelaxLowerVec (α := ℝ) lo hi αt) xLo)
        x') i ≤ getScalar (Activation.reluSpec (α := ℝ) z) i ∧
      getScalar (Activation.reluSpec (α := ℝ) z) i ≤
        getScalar (affineEvalAt (α := ℝ)
          (Runtime.Ops.ReLU.propagateAffine (α := ℝ) (Runtime.Ops.ReLU.relaxVector (α := ℝ) lo hi)
            xHi) x') i := by
  obtain ⟨hzLo, hzHi⟩ := (encloses_iff_getScalar _ _ _).1 hzAff i
  obtain ⟨hzIlo, hzIhi⟩ := (encloses_iff_getScalar _ _ _).1 hzIbp i
  simp only [getScalar_affineEvalAt_relu_propagate_affine, getScalar_alphaRelaxLowerVec,
    getScalar_runtime_relu_relax_vector, getScalar_relu_spec]
  exact ⟨alpha_relax_lower_le_relu _ _ _ _ _ hzIlo hzIhi (hαrange i).1 (hαrange i).2 hzLo,
    relu_le_relax_upper _ _ _ _ hzIlo hzIhi hzHi⟩

/-- ReLU transfer core: if `xin` encloses the parent value `vp` at `x`, the IBP box `preB`
encloses `vp`, and the α vector lies in `[0, 1]`, then propagating `xin` through the α lower and
CROWN upper relaxations encloses `relu vp`. -/
theorem enclosesAtInput_relu_propagate {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {xin : FlatAffineBounds ℝ} {vp : Val} {preB : FlatBox ℝ} (αt : Tensor ℝ [preB.dim])
    (hαrange : ∀ i : Fin preB.dim, (0 : ℝ) ≤ getScalar αt i ∧ getScalar αt i ≤ 1)
    (hout : xin.outDim = preB.dim)
    (hpar : EnclosesAtInput (α := ℝ) ctx x xin vp)
    (hEncIbp : EnclosesBox preB vp) :
    EnclosesAtInput (α := ℝ) ctx x
      { inDim := xin.inDim
        outDim := preB.dim
        loAff := Runtime.Ops.ReLU.propagateAffine (α := ℝ) (inDim := xin.inDim)
          (hidDim := preB.dim) (alphaRelaxLowerVec (α := ℝ) (n := preB.dim) preB.lo preB.hi αt)
          (Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim) (m' := preB.dim) hout
            xin.loAff)
        hiAff := Runtime.Ops.ReLU.propagateAffine (α := ℝ) (inDim := xin.inDim)
          (hidDim := preB.dim)
          (Runtime.Ops.ReLU.relaxVector (α := ℝ) (n := preB.dim) preB.lo preB.hi)
          (Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim) (m' := preB.dim) hout
            xin.hiAff) }
      { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } := by
  obtain ⟨hinDim, -⟩ := id hpar
  obtain ⟨hdimIbp, hencIbp⟩ := hEncIbp
  have hzAff := encloses_castOut_of_enclosesAtInput hout hdimIbp.symm hinDim hpar
  refine ⟨hinDim, hdimIbp, ?_⟩
  refine sem_encloses_transport rfl
    (heq_of_eq (relu_spec_castDimScalar hdimIbp.symm vp.v)).symm ?_
  refine (encloses_iff_getScalar (n := preB.dim) _ _ _).2 fun i => ?_
  exact relu_propagate_getScalar_bounds preB.lo preB.hi αt _ _ _ _ hαrange hzAff hencIbp i

/-! ## The α vector selected by the step -/

/-- A certificate α entry, cast to the pre-activation dimension, has all components in `[0, 1]`
under `AlphaOK`. -/
theorem alpha_cast_range {alpha : Array (Option (FlatTensor ℝ))} {id : Nat} {αv : FlatTensor ℝ}
    {n : Nat} (halpha : AlphaOK (alpha := alpha))
    (hαopt : Cert.getAlpha? (α := ℝ) alpha id = some αv) (hα : αv.n = n) :
    ∀ i : Fin n, (0 : ℝ) ≤ getScalar (castDimScalar (α := ℝ) hα αv.v) i ∧
      getScalar (castDimScalar (α := ℝ) hα αv.v) i ≤ 1 := by
  obtain ⟨hidA, hentry⟩ := lt_size_and_getElem!_of_getAlpha?_eq_some hαopt
  have hrange : ∀ i : Fin αv.n, (0 : ℝ) ≤ getScalar αv.v i ∧ getScalar αv.v i ≤ 1 := by
    simpa [hentry] using halpha id hidA
  intro i
  simpa [getScalar_castDimScalar] using hrange (Fin.cast hα.symm i)

/-- The α vector the step selects when a certificate entry is present (the entry cast to the
pre-activation dimension if the sizes agree, the default relaxation otherwise) has all components
in `[0, 1]`. -/
theorem selected_alpha_range {alpha : Array (Option (FlatTensor ℝ))} {id : Nat}
    {αv : FlatTensor ℝ} {preB : FlatBox ℝ} (halpha : AlphaOK (alpha := alpha))
    (hαopt : Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    ∀ i : Fin preB.dim,
      (0 : ℝ) ≤ getScalar (if hα : αv.n = preB.dim then castDimScalar (α := ℝ) hα αv.v
        else defaultAlphaVec (α := ℝ) preB.lo preB.hi) i ∧
      getScalar (if hα : αv.n = preB.dim then castDimScalar (α := ℝ) hα αv.v
        else defaultAlphaVec (α := ℝ) preB.lo preB.hi) i ≤ 1 := by
  by_cases hα : αv.n = preB.dim
  · rw [dite_eq_left hα]
    exact alpha_cast_range halpha hαopt hα
  · rw [dite_eq_right hα]
    exact defaultAlphaVec_range preB.lo preB.hi

/-! ## The `.relu` case -/

/-- The `.relu` case: the step propagates the parent's affine bounds through the α lower and CROWN
upper relaxations built from the IBP pre-activation box at the parent, and the evaluator applies
`reluSpec` to the parent's value. The α vector is the certificate entry when present and well-sized,
and the default relaxation otherwise. -/
theorem relu_sound {g : Graph} {ps : ParamStore ℝ} {ibp : Array (Option (FlatBox ℝ))}
    {alpha : Array (Option (FlatTensor ℝ))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {b : FlatAffineBounds ℝ} {v : Val}
    (hk : (g.nodes[id]!).kind = .relu)
    (hs : stepAlpha g ps ibp alpha ctx cert id = some b)
    (hEvalSome : evalNode? g.nodes ps inputs vals id = some v)
    (hpar : ParentsEnclosed g cert vals ctx x id)
    (hparLt : ∀ p : Nat, p ∈ (g.nodes[id]!).parents → p < vals.size)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha)) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  cases hps : NN.IR.unaryParent? (g.nodes[id]!).parents with
  | none => simp only [stepAlpha, alphaCrownStepNode?, hk, hps, reduceCtorEq] at hs
  | some p1 =>
    simp only [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
    simp only [CertSoundness.evalNode?, hk, hps] at hEvalSome
    obtain ⟨xin, hxin⟩ : ∃ xin, Cert.getAff? (α := ℝ) cert p1 = some xin :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [h, reduceCtorEq] at hs
    obtain ⟨preB, hpre⟩ : ∃ preB, ibp[p1]! = some preB :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [hxin, h, reduceCtorEq] at hs
    obtain ⟨vp, hgv⟩ : ∃ vp, CertSoundness.getVal? vals p1 = some vp :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [h, reduceCtorEq] at hEvalSome
    simp only [hgv] at hEvalSome
    cases hEvalSome
    have hpar' := parent_encloses_of_unaryParent? hpar hps hxin hgv
    -- The IBP box at the parent encloses the pre-activation value.
    have hEncIbp : EnclosesBox preB vp := by
      have h := hibp p1 (hparLt p1 (NN.IR.mem_of_unaryParent?_eq_some hps))
      simpa only [hpre, getElem!_of_getVal?_eq_some hgv] using h
    cases hαopt : Cert.getAlpha? (α := ℝ) alpha id with
    | none =>
      simp only [hxin, hpre, hαopt] at hs
      by_cases hout : xin.outDim = preB.dim
      · rw [dite_eq_left hout] at hs
        cases hs
        exact enclosesAtInput_relu_propagate _ (defaultAlphaVec_range preB.lo preB.hi) hout hpar'
          hEncIbp
      · rw [dite_eq_right hout] at hs
        cases hs
    | some αv =>
      simp only [hxin, hpre, hαopt] at hs
      by_cases hout : xin.outDim = preB.dim
      · rw [dite_eq_left hout] at hs
        cases hs
        exact enclosesAtInput_relu_propagate _ (selected_alpha_range halpha hαopt) hout hpar'
          hEncIbp
      · rw [dite_eq_right hout] at hs
        cases hs

end NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

end
