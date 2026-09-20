/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Basic

/-!
# α-CROWN Transfer: Value-Preserving Nodes

The `.detach`, `.reshape`, and `.flatten` cases of `alphaCrown_transfer_sound`. These operators
forward the parent's value, at most recasting its dimension, so the parent's affine bounds are
forwarded (and recast) as well.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert
open CrownCertSoundness
open CertSoundness

/-- The `.detach` case: the step forwards the parent's affine bounds and the evaluator forwards the
parent's value, so the parent enclosure is the result. -/
theorem detach_sound {g : Graph} {ps : ParamStore ℝ} {ibp : Array (Option (FlatBox ℝ))}
    {alpha : Array (Option (FlatTensor ℝ))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {b : FlatAffineBounds ℝ} {v : Val}
    (hk : (g.nodes[id]!).kind = .detach)
    (hs : stepAlpha g ps ibp alpha ctx cert id = some b)
    (hEvalSome : evalNode? g.nodes ps inputs vals id = some v)
    (hpar : ParentsEnclosed g cert vals ctx x id) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  cases hps : NN.IR.unaryParent? (g.nodes[id]!).parents with
  | none => simp only [stepAlpha, alphaCrownStepNode?, hk, hps, reduceCtorEq] at hs
  | some p1 =>
    have hxin : Cert.getAff? (α := ℝ) cert p1 = some b := by
      simpa only [stepAlpha, alphaCrownStepNode?, hk, hps] using hs
    have hgv : CertSoundness.getVal? vals p1 = some v := by
      simpa only [CertSoundness.evalNode?, hk, hps] using hEvalSome
    exact parent_encloses_of_unaryParent? hpar hps hxin hgv

/-- The shared `.reshape` and `.flatten` case: the step casts the parent's affine bounds to the
node's output size and the evaluator casts the parent's value, both guarded by the same size check,
so the parent enclosure transports along the cast. -/
theorem reshape_flatten_sound {g : Graph} {ps : ParamStore ℝ} {ibp : Array (Option (FlatBox ℝ))}
    {alpha : Array (Option (FlatTensor ℝ))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {b : FlatAffineBounds ℝ} {v : Val}
    (hk : (∃ s₁ s₂ : Shape, (g.nodes[id]!).kind = .reshape s₁ s₂) ∨
      ∃ s : Shape, (g.nodes[id]!).kind = .flatten s)
    (hs : stepAlpha g ps ibp alpha ctx cert id = some b)
    (hEvalSome : evalNode? g.nodes ps inputs vals id = some v)
    (hpar : ParentsEnclosed g cert vals ctx x id) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  -- Both kinds unfold to the same step and evaluator code.
  rcases hk with ⟨_, _, hk⟩ | ⟨_, hk⟩
  all_goals
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
      by_cases hout : xin.outDim = (g.nodes[id]!).outShape.size
      · rw [dite_eq_left hout] at hs
        by_cases hvout : vp.n = (g.nodes[id]!).outShape.size
        · rw [dite_eq_left hvout] at hEvalSome
          cases hs
          cases hEvalSome
          exact enclosesAtInput_castOut ctx x xin vp hout hvout
            (parent_encloses_of_unaryParent? hpar hps hxin hgv)
        · rw [dite_eq_right hvout] at hEvalSome
          cases hEvalSome
      · rw [dite_eq_right hout] at hs
        cases hs

end NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

end
