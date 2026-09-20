/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Basic

/-!
# α-CROWN Transfer: Leaf Nodes

The `.input` and `.const` cases of `alphaCrown_transfer_sound`. Neither reads a parent
certificate: the input node returns the identity bounds and the constant node returns the point
box of its stored value.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert
open CrownCertSoundness
open CertSoundness

/-- The `.input` case: the step succeeds only at the designated input node and returns the identity
bounds, which enclose the input point `x`, and `InputsMatch` identifies the semantic value with
`x`. -/
theorem input_sound {g : Graph} {ps : ParamStore ℝ} {ibp : Array (Option (FlatBox ℝ))}
    {alpha : Array (Option (FlatTensor ℝ))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {b : FlatAffineBounds ℝ} {v : Val}
    (hk : (g.nodes[id]!).kind = .input)
    (hs : stepAlpha g ps ibp alpha ctx cert id = some b)
    (hEvalSome : evalNode? g.nodes ps inputs vals id = some v)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  have hidCtx : id = ctx.inputId := by
    by_contra hne
    simp [stepAlpha, alphaCrownStepNode?, hk, hne] at hs
  subst hidCtx
  have hb : Cert.boundsIdentity (α := ℝ) ctx.inputDim = b :=
    Option.some.inj (by simpa [stepAlpha, alphaCrownStepNode?, hk] using hs)
  subst hb
  rcases hinputs with ⟨vin, hmap, hdim, hxEq⟩
  have hiv : inputs[ctx.inputId]? = some v := by
    simpa [CertSoundness.evalNode?, hk] using hEvalSome
  obtain rfl : vin = v := Option.some.inj (hmap.symm.trans hiv)
  refine ⟨rfl, hdim.symm, ?_⟩
  -- The identity bounds evaluate to the point box `{x}`, and the cast value is `x`.
  refine sem_encloses_transport (boundsEvalAt_bounds_identity x).symm ?_ (encloses_point_box x)
  exact heq_of_eq (hxEq.symm.trans (castDimScalar_proof_irrel _ _ _))

/-- The `.const` case: the step and the evaluator both read `ps.constVals[id]?`, and the step
returns the point box of the constant, which encloses it. -/
theorem const_sound {g : Graph} {ps : ParamStore ℝ} {ibp : Array (Option (FlatBox ℝ))}
    {alpha : Array (Option (FlatTensor ℝ))} {cert : Array (Option (FlatAffineBounds ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {ctx : AffineCtx}
    {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {b : FlatAffineBounds ℝ} {v : Val} {shape : Shape}
    (hk : (g.nodes[id]!).kind = .const shape)
    (hs : stepAlpha g ps ibp alpha ctx cert id = some b)
    (hEvalSome : evalNode? g.nodes ps inputs vals id = some v) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  cases hcv : ps.constVals[id]? with
  | none => simp [stepAlpha, alphaCrownStepNode?, hk, hcv] at hs
  | some vc =>
    have hb : Cert.boundsConst (α := ℝ) ctx.inputDim vc.n vc.v vc.v = b :=
      Option.some.inj (by simpa [stepAlpha, alphaCrownStepNode?, hk, hcv] using hs)
    have hev : ps.constVals[id]? = some v := by
      simpa [CertSoundness.evalNode?, hk] using hEvalSome
    obtain rfl : vc = v := Option.some.inj (hcv.symm.trans hev)
    subst hb
    exact enclosesAtInput_boundsConst_of_enclosesBox (B0 := { dim := vc.n, lo := vc.v, hi := vc.v })
      ⟨rfl, encloses_point_box vc.v⟩

end NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

end
