/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Common

/-!
# α-CROWN Transfer: Shared Extraction Lemmas

Helper lemmas shared by the per-operator cases of `alphaCrown_transfer_sound`. They turn the safe
lookups `getAff?`, `getVal?`, and `getAlpha?` into plain array lookups, package the parent
hypothesis of `CrownTransferSound`, and transport enclosures across output-dimension casts and
constant boxes.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open CrownCertSoundness
open CertSoundness

/-- Every parent `p` of node `id` whose certificate entry and semantic value are both present is
enclosed at the input point `x`. This is the parent hypothesis of `CrownTransferSound` with its
`match` unfolded. -/
def ParentsEnclosed (g : Graph) (cert : Array (Option (FlatAffineBounds ℝ)))
    (vals : Array (Option Val)) (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim]) (id : Nat) :
    Prop :=
  ∀ p : Nat, p ∈ (g.nodes[id]!).parents →
    ∀ (bp : FlatAffineBounds ℝ) (vp : Val),
      cert[p]! = some bp → vals[p]! = some vp → EnclosesAtInput (α := ℝ) ctx x bp vp

/-- The `match`-shaped parent hypothesis of `CrownTransferSound` implies `ParentsEnclosed`. -/
theorem parentsEnclosed_of_match {g : Graph} {cert : Array (Option (FlatAffineBounds ℝ))}
    {vals : Array (Option Val)} {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]} {id : Nat}
    (hparents : ∀ p : Nat, p ∈ (g.nodes[id]!).parents →
      match cert[p]!, vals[p]! with
      | some bp, some vp => EnclosesAtInput (α := ℝ) ctx x bp vp
      | _, _ => True) :
    ParentsEnclosed g cert vals ctx x id := by
  intro p hp bp vp hbp hvp
  have h := hparents p hp
  simpa [hbp, hvp] using h

/-- A successful safe certificate lookup is a plain array lookup. -/
theorem getElem!_of_getAff?_eq_some {cert : Array (Option (FlatAffineBounds ℝ))} {p : Nat}
    {xin : FlatAffineBounds ℝ} (h : Cert.getAff? (α := ℝ) cert p = some xin) :
    cert[p]! = some xin := by
  by_cases hlt : p < cert.size
  · simpa [Cert.getAff?, hlt] using h
  · simp [Cert.getAff?, hlt] at h

/-!
`getElem!_of_getVal?_eq_some` is `CertSoundness`'s lemma, opened at the top of this file. Two files
in this directory carried their own copy of it, proved with `by_cases` where the original uses
`unfold` and `split`; the statements were the same, so the copies are gone.
-/
/-- A successful safe α lookup is in bounds and is a plain array lookup. -/
theorem lt_size_and_getElem!_of_getAlpha?_eq_some {alpha : Array (Option (FlatTensor ℝ))}
    {id : Nat} {αv : FlatTensor ℝ} (h : Cert.getAlpha? (α := ℝ) alpha id = some αv) :
    id < alpha.size ∧ alpha[id]! = some αv := by
  by_cases hlt : id < alpha.size
  · exact ⟨hlt, by simpa [Cert.getAlpha?, hlt] using h⟩
  · simp [Cert.getAlpha?, hlt] at h

/-- The unique parent of a unary node is enclosed whenever its certificate entry and semantic
value are both found by the safe lookups. -/
theorem parent_encloses_of_unaryParent? {g : Graph} {cert : Array (Option (FlatAffineBounds ℝ))}
    {vals : Array (Option Val)} {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]} {id p1 : Nat}
    {xin : FlatAffineBounds ℝ} {vp : Val}
    (hpar : ParentsEnclosed g cert vals ctx x id)
    (hps : NN.IR.unaryParent? (g.nodes[id]!).parents = some p1)
    (hxin : Cert.getAff? (α := ℝ) cert p1 = some xin)
    (hgv : CertSoundness.getVal? vals p1 = some vp) :
    EnclosesAtInput (α := ℝ) ctx x xin vp :=
  hpar p1 (NN.IR.mem_of_unaryParent?_eq_some hps) xin vp
    (getElem!_of_getAff?_eq_some hxin) (getElem!_of_getVal?_eq_some hgv)

/-- Under `SemLocalOK`, a present semantic value at `id` is the evaluator's output. -/
theorem evalNode?_eq_some_of_semLocalOK {g : Graph} {ps : ParamStore ℝ}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {id : Nat} {v : Val}
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hid : id < g.nodes.size) (hv : vals[id]! = some v) :
    evalNode? g.nodes ps inputs vals id = some v :=
  (hsem.2 id hid).symm.trans hv

/-- Transport an enclosure along an equality of boxes and a heterogeneous equality of values. -/
theorem sem_encloses_transport {B1 B2 : FlatBox ℝ} (h : B1 = B2) {x : Tensor ℝ [B1.dim]}
    {y : Tensor ℝ [B2.dim]} (hxy : HEq x y)
    (hx : Theorems.Semantics.encloses (α := ℝ) B1 x) :
    Theorems.Semantics.encloses (α := ℝ) B2 y := by
  subst h
  cases hxy
  exact hx

/-- A point box encloses its own point. -/
theorem encloses_point_box {n : Nat} (t : Tensor ℝ [n]) :
    Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := t, hi := t } t :=
  (encloses_iff_getScalar t t t).2 fun _ => ⟨le_rfl, le_rfl⟩

/-- `EnclosesAtInput` respects equality of value payloads. -/
theorem enclosesAtInput_congr_val {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {b : FlatAffineBounds ℝ} {v w : Val} (h : v = w)
    (hv : EnclosesAtInput (α := ℝ) ctx x b v) : EnclosesAtInput (α := ℝ) ctx x b w :=
  h ▸ hv

/-- A constant affine enclosure built from a box `B0` is sound at every input point whenever `B0`
encloses the value. -/
theorem enclosesAtInput_boundsConst_of_enclosesBox {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {B0 : FlatBox ℝ} {v : Val} (hEnc : EnclosesBox B0 v) :
    EnclosesAtInput (α := ℝ) ctx x
      (Cert.boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) v := by
  rcases hEnc with ⟨hdim, hbox⟩
  refine ⟨rfl, hdim, ?_⟩
  have hBoxEval :
      boundsEvalAt (α := ℝ) (Cert.boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi) x = B0 := by
    cases B0
    exact boundsEvalAt_bounds_const _ _ x
  exact sem_encloses_transport hBoxEval.symm HEq.rfl hbox

/-- Casting the output dimension of enclosing affine bounds yields a box, evaluated at the input
point, that encloses the correspondingly cast value. This is the componentwise form of
`enclosesAtInput_castOut` used by the linear and ReLU cases. -/
theorem encloses_castOut_of_enclosesAtInput {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]}
    {xin : FlatAffineBounds ℝ} {vp : Val} {n : Nat}
    (hout : xin.outDim = n) (hvn : vp.n = n) (hinDim : xin.inDim = ctx.inputDim)
    (hpar : EnclosesAtInput (α := ℝ) ctx x xin vp) :
    Theorems.Semantics.encloses (α := ℝ)
      { dim := n
        lo := affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := n)
          (Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim) (m' := n) hout xin.loAff)
          (castDimScalar (α := ℝ) hinDim.symm x)
        hi := affineEvalAt (α := ℝ) (inDim := xin.inDim) (outDim := n)
          (Graph.castAffineOut (α := ℝ) (n := xin.inDim) (m := xin.outDim) (m' := n) hout xin.hiAff)
          (castDimScalar (α := ℝ) hinDim.symm x) }
      (castDimScalar (α := ℝ) hvn vp.v) := by
  obtain ⟨hinDim', hvec⟩ := enclosesAtInput_castOut ctx x xin vp hout hvn hpar
  obtain ⟨_, henc⟩ := hvec
  simpa [boundsEvalAt] using henc

end NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

end
