/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Basic

/-!
# α-CROWN Transfer: Linear Nodes

The `.linear`, `.matmul`, and `.sum` cases of `alphaCrown_transfer_sound`. All three are instances
of the sign-splitting rule `linearBoundsFromAffine`, proved once in
`enclosesAtInput_linearBoundsFromAffine` and instantiated with the node's weights (all-ones row
for `.sum`) and bias (zero for `.matmul` and `.sum`).
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert
open CrownCertSoundness
open CertSoundness

/-- Sign-splitting transfer through `y = W x + bv`: if the affine bounds `xin` enclose the parent
value `vp` at `x`, then `linearBoundsFromAffine W bv xin hout` encloses `W · vp + bv`. -/
theorem enclosesAtInput_linearBoundsFromAffine {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {xin : FlatAffineBounds ℝ} {vp : Val} {n m : Nat}
    (W : Tensor ℝ [m, n]) (bv : Tensor ℝ [m])
    (hout : xin.outDim = n) (hvIn : vp.n = n)
    (hpar : EnclosesAtInput (α := ℝ) ctx x xin vp) :
    EnclosesAtInput (α := ℝ) ctx x
      (Cert.linearBoundsFromAffine (α := ℝ) (inDim := xin.inDim) (n := n) (m := m) W bv xin hout)
      { n := m
        v := Spec.linearSpec (α := ℝ) { weights := W, bias := bv }
          (castDimScalar (α := ℝ) hvIn vp.v) } := by
  obtain ⟨hinDim, -⟩ := id hpar
  have hxCast := encloses_castOut_of_enclosesAtInput hout hvIn hinDim hpar
  have hy := encloses_linear_signSplit (m := m) (n := n) W bv _ _ _ hxCast
  refine ⟨hinDim, rfl, ?_⟩
  exact sem_encloses_transport
    (boundsEvalAt_linear_bounds_from_affine W bv xin hout _).symm HEq.rfl hy

/-- The `.linear` case: the step applies the sign-splitting rule to the parent's affine bounds with
the node's weights and bias, and the evaluator applies `linearSpec` to the parent's value. -/
theorem linear_sound {g : Graph} {ps : ParamStore ℝ} {ibp : Array (Option (FlatBox ℝ))}
    {alpha : Array (Option (FlatTensor ℝ))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {b : FlatAffineBounds ℝ} {v : Val}
    (hk : (g.nodes[id]!).kind = .linear)
    (hs : stepAlpha g ps ibp alpha ctx cert id = some b)
    (hEvalSome : evalNode? g.nodes ps inputs vals id = some v)
    (hpar : ParentsEnclosed g cert vals ctx x id) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  cases hps : NN.IR.unaryParent? (g.nodes[id]!).parents with
  | none => simp only [stepAlpha, alphaCrownStepNode?, hk, hps, reduceCtorEq] at hs
  | some p1 =>
    simp only [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
    simp only [CertSoundness.evalNode?, hk, hps] at hEvalSome
    obtain ⟨xin, hxin⟩ : ∃ xin, Cert.getAff? (α := ℝ) cert p1 = some xin :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [h, reduceCtorEq] at hs
    obtain ⟨p, hwb⟩ : ∃ p, ps.linearWB[id]? = some p :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [hxin, h, reduceCtorEq] at hs
    obtain ⟨vp, hgv⟩ : ∃ vp, CertSoundness.getVal? vals p1 = some vp :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [h, hwb, reduceCtorEq] at hEvalSome
    simp only [hxin, hwb] at hs
    simp only [hgv, hwb] at hEvalSome
    by_cases hout : xin.outDim = p.n
    · rw [dif_pos hout] at hs
      by_cases hvIn : vp.n = p.n
      · rw [dif_pos hvIn] at hEvalSome
        cases hs
        cases hEvalSome
        exact enclosesAtInput_linearBoundsFromAffine p.w p.b hout hvIn
          (parent_encloses_of_unaryParent? hpar hps hxin hgv)
      · rw [dif_neg hvIn] at hEvalSome
        cases hEvalSome
    · rw [dif_neg hout] at hs
      cases hs

/-- The `.matmul` case: as `.linear`, with the node's weight matrix and a zero bias. -/
theorem matmul_sound {g : Graph} {ps : ParamStore ℝ} {ibp : Array (Option (FlatBox ℝ))}
    {alpha : Array (Option (FlatTensor ℝ))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {b : FlatAffineBounds ℝ} {v : Val}
    (hk : (g.nodes[id]!).kind = .matmul)
    (hs : stepAlpha g ps ibp alpha ctx cert id = some b)
    (hEvalSome : evalNode? g.nodes ps inputs vals id = some v)
    (hpar : ParentsEnclosed g cert vals ctx x id) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  cases hps : NN.IR.unaryParent? (g.nodes[id]!).parents with
  | none => simp only [stepAlpha, alphaCrownStepNode?, hk, hps, reduceCtorEq] at hs
  | some p1 =>
    simp only [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
    simp only [CertSoundness.evalNode?, hk, hps] at hEvalSome
    obtain ⟨xin, hxin⟩ : ∃ xin, Cert.getAff? (α := ℝ) cert p1 = some xin :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [h, reduceCtorEq] at hs
    obtain ⟨p, hwb⟩ : ∃ p, ps.matmulW[id]? = some p :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [hxin, h, reduceCtorEq] at hs
    obtain ⟨vp, hgv⟩ : ∃ vp, CertSoundness.getVal? vals p1 = some vp :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [h, hwb, reduceCtorEq] at hEvalSome
    simp only [hxin, hwb] at hs
    simp only [hgv, hwb] at hEvalSome
    by_cases hout : xin.outDim = p.n
    · rw [dif_pos hout] at hs
      by_cases hvIn : vp.n = p.n
      · rw [dif_pos hvIn] at hEvalSome
        cases hs
        cases hEvalSome
        exact enclosesAtInput_linearBoundsFromAffine p.w
          (Tensor.full (α := ℝ) (.dim p.m .scalar) 0) hout hvIn
          (parent_encloses_of_unaryParent? hpar hps hxin hgv)
      · rw [dif_neg hvIn] at hEvalSome
        cases hEvalSome
    · rw [dif_neg hout] at hs
      cases hs

/-- The `.sum` case: the step treats the sum as the `1 × n` linear layer with all-ones weights and
zero bias, and the evaluator multiplies the parent's value by the all-ones row. -/
theorem sum_sound {g : Graph} {ps : ParamStore ℝ} {ibp : Array (Option (FlatBox ℝ))}
    {alpha : Array (Option (FlatTensor ℝ))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {b : FlatAffineBounds ℝ} {v : Val}
    (hk : (g.nodes[id]!).kind = .sum)
    (hs : stepAlpha g ps ibp alpha ctx cert id = some b)
    (hEvalSome : evalNode? g.nodes ps inputs vals id = some v)
    (hpar : ParentsEnclosed g cert vals ctx x id) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  cases hps : NN.IR.unaryParent? (g.nodes[id]!).parents with
  | none => simp only [stepAlpha, alphaCrownStepNode?, hk, hps, reduceCtorEq] at hs
  | some p1 =>
    simp only [stepAlpha, alphaCrownStepNode?, hk, hps] at hs
    simp only [CertSoundness.evalNode?, hk, hps] at hEvalSome
    obtain ⟨xin, hxin⟩ : ∃ xin, Cert.getAff? (α := ℝ) cert p1 = some xin :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [h, reduceCtorEq] at hs
    obtain ⟨vp, hgv⟩ : ∃ vp, CertSoundness.getVal? vals p1 = some vp :=
      Option.ne_none_iff_exists'.1 fun h => by simp only [h, reduceCtorEq] at hEvalSome
    simp only [hxin] at hs
    simp only [hgv] at hEvalSome
    cases hs
    cases hEvalSome
    have hpar' := parent_encloses_of_unaryParent? hpar hps hxin hgv
    -- The parent enclosure fixes the parent's dimension; the evaluator does not check it.
    have ⟨_, hdimB, _⟩ := hpar'
    have hvIn : vp.n = xin.outDim := hdimB.symm
    have hval :
        Spec.linearSpec (α := ℝ)
            { weights := Tensor.full (α := ℝ) (.dim 1 (.dim xin.outDim .scalar)) 1
              bias := Tensor.full (α := ℝ) (.dim 1 .scalar) 0 }
            (castDimScalar (α := ℝ) hvIn vp.v) =
          Spec.matVecMulSpec (α := ℝ)
            (Tensor.full (α := ℝ) (.dim 1 (.dim vp.n .scalar)) (1 : ℝ)) vp.v :=
      (linear_spec_bias_zero_eq_matvec _ _).trans
        (mat_vec_mul_full_one_castDimScalar hvIn vp.v).symm
    exact enclosesAtInput_congr_val (congrArg (FlatTensor.mk 1) hval)
      (enclosesAtInput_linearBoundsFromAffine _ _ rfl hvIn hpar')

end NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

end
