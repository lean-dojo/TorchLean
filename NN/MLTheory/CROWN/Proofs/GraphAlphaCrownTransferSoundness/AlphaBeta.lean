/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.AlphaBeta.StepInversion
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.AlphaBeta.ReLUPhase
public import NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness

/-!
# α/β-CROWN Graph Transfer Soundness

Pointwise soundness theorem for the β-extended graph transfer rule.

The proof is a dispatch over two cases. Every node that is not a ReLU carrying a β vector is
handled by `alphaCrown_transfer_sound`, since `stepAlphaBeta` agrees with `stepAlpha` there
(`stepAlphaBeta_eq_stepAlpha`). A ReLU node with a β vector is handled by inverting the step
(`stepAlphaBeta_relu_beta_inv`), reading off the parent's value from the semantics, and applying
the pointwise β-phase ReLU lemma `enclosesAtInput_relu_beta`.
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

/-! ## Semantic glue -/

/--
The parent hypothesis of `CrownTransferSound`, specialised to a parent whose certificate entry and
semantic value are both present.
-/
theorem enclosesAtInput_of_parents
    {g : Graph} {cert : Array (Option (FlatAffineBounds ℝ))} {vals : Array (Option Val)}
    {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]} {id p : Nat}
    {xin : FlatAffineBounds ℝ} {vp : Val}
    (hparents : ∀ p : Nat, p ∈ (g.nodes[id]!).parents →
      match cert[p]!, vals[p]! with
      | some bp, some vp => EnclosesAtInput (α := ℝ) ctx x bp vp
      | _, _ => True)
    (hp : p ∈ (g.nodes[id]!).parents)
    (hcert : cert[p]! = some xin) (hval : vals[p]! = some vp) :
    EnclosesAtInput (α := ℝ) ctx x xin vp := by
  have h := hparents p hp
  rw [hcert, hval] at h
  exact h

/-- A ReLU node's semantic value is `relu` of its unary parent's value. -/
theorem vals_relu_eq
    {g : Graph} {ps : ParamStore ℝ} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {id p1 : Nat} {v : Val}
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hid : id < g.nodes.size)
    (hk : (g.nodes[id]!).kind = .relu)
    (hps : NN.IR.unaryParent? (g.nodes[id]!).parents = some p1)
    (hv : vals[id]! = some v) :
    ∃ vp : Val,
      vals[p1]! = some vp ∧ v = { n := vp.n, v := Activation.reluSpec (α := ℝ) vp.v } := by
  have hEval : evalNode? g.nodes ps inputs vals id = some v := by
    rw [← hsem.2 id hid]
    exact hv
  simp only [CertSoundness.evalNode?, hk, hps] at hEval
  cases hgv : CertSoundness.getVal? vals p1 with
  | none => simp [hgv] at hEval
  | some vp =>
      simp only [hgv] at hEval
      exact ⟨vp, getElem!_of_getVal?_eq_some hgv, (Option.some.inj hEval).symm⟩

/-- The IBP box of a parent node encloses that parent's semantic value. -/
theorem enclosesBox_parent
    {g : Graph} {ps : ParamStore ℝ} {inputs : Std.HashMap Nat Val}
    {ibp : Array (Option (FlatBox ℝ))} {vals : Array (Option Val)}
    {id p : Nat} {preB : FlatBox ℝ} {vp : Val}
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (hid : id < g.nodes.size) (hp : p ∈ (g.nodes[id]!).parents)
    (hpre : ibp[p]! = some preB) (hvp : vals[p]! = some vp) :
    EnclosesBox preB vp := by
  have hlt : p < vals.size := by
    rw [hsem.1]
    exact lt_trans (htopo id hid p hp) hid
  have h := hibp p hlt
  rw [hpre, hvp] at h
  exact h

/-! ## Per-case soundness -/

/--
Soundness of `stepAlphaBeta` at every node handled by the α-CROWN rule: all non-ReLU kinds, and
ReLU nodes with no β entry. The step is literally `stepAlpha` there, so this is
`alphaCrown_transfer_sound` read at one node.
-/
theorem alphaBetaCrown_alpha_case_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha))
    (id : Nat) (hid : id < g.nodes.size)
    (hparents : ∀ p : Nat, p ∈ (g.nodes[id]!).parents →
      match cert[p]!, vals[p]! with
      | some bp, some vp => EnclosesAtInput (α := ℝ) ctx x bp vp
      | _, _ => True)
    (hnb : (g.nodes[id]!).kind = .relu → getBeta? (beta := beta) id = none)
    {b : FlatAffineBounds ℝ} {v : Val}
    (hs : stepAlphaBeta g ps ibp alpha beta ctx cert id = some b)
    (hv : vals[id]! = some v) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  have hA := alphaCrown_transfer_sound g ps ibp alpha cert inputs vals ctx x
    htopo hsem hinputs hibp halpha id hid hparents
  rw [stepAlphaBeta_eq_stepAlpha g ps ibp alpha beta cert ctx id hnb] at hs
  rw [hs, hv] at hA
  exact hA

/--
Soundness of `stepAlphaBeta` at a ReLU node carrying a β vector.

The step inversion supplies the parent's certificate entry, its IBP box, the α vector used, and
the accepted phase relaxations; the semantics supplies the parent's value;
`enclosesAtInput_relu_beta` does the rest.
-/
theorem alphaBetaCrown_relu_beta_case_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha))
    (id : Nat) (hid : id < g.nodes.size)
    (hparents : ∀ p : Nat, p ∈ (g.nodes[id]!).parents →
      match cert[p]!, vals[p]! with
      | some bp, some vp => EnclosesAtInput (α := ℝ) ctx x bp vp
      | _, _ => True)
    {phases : Array Int}
    (hk : (g.nodes[id]!).kind = .relu)
    (hbeta : getBeta? (beta := beta) id = some phases)
    {b : FlatAffineBounds ℝ} {v : Val}
    (hs : stepAlphaBeta g ps ibp alpha beta ctx cert id = some b)
    (hv : vals[id]! = some v) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  cases hps : NN.IR.unaryParent? (g.nodes[id]!).parents with
  | none => simp [stepAlphaBeta, alphaBetaCrownStepNode?, hk, hbeta, hps] at hs
  | some p1 =>
  have hpMem : p1 ∈ (g.nodes[id]!).parents := NN.IR.mem_of_unaryParent?_eq_some hps
  obtain ⟨xin, preB, hout, αt, relaxLo, relaxHi, hxin, hpre, hαrange, hrelax, hb⟩ :=
    stepAlphaBeta_relu_beta_inv g ps ibp alpha beta cert ctx id b phases p1 halpha hk hbeta hps hs
  obtain ⟨vp, hvp, hvEq⟩ := vals_relu_eq hsem hid hk hps hv
  have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
    enclosesAtInput_of_parents hparents hpMem (getElem!_of_getAff?_eq_some hxin) hvp
  have hbox : EnclosesBox preB vp := enclosesBox_parent htopo hsem hibp hid hpMem hpre hvp
  subst hb hvEq
  exact enclosesAtInput_relu_beta ctx x xin vp preB hout αt phases relaxLo relaxHi hαrange hrelax
    hpar hbox

/-! ## Main transfer theorem -/

/--
Pointwise soundness of the graph-dialect α/β-CROWN transfer rule.

This is the β-extended analog of `alphaCrown_transfer_sound`. The step function additionally
receives a `beta` array of per-ReLU phase constraints (active, inactive, unstable). At a ReLU node
with a β vector, `phaseRelaxVec?` checks each phase against the IBP pre-activation interval via
`phaseConsistentScalar?` (inactive needs `u ≤ 0`, active needs `0 ≤ l`) and, if every phase
passes, uses the phase's exact affine rule for that unit; an inconsistent phase rejects the step
rather than falling back to another relaxation. All other nodes use the α-CROWN rule.

What the β relaxation contributes here, and what it does not. Because a phase is accepted only
when the IBP interval already implies it, a β vector can never certify a sign that the supplied
`ibp` box does not fix on its own; for such stable units the phase rule coincides with the
standard relaxation. The theorem therefore establishes two things about β: the exact rules are
sound whenever the consistency check passes, and inconsistent phase vectors are rejected. It does
not model branch-and-bound split constraints. A split that tightens beyond IBP would have to be
reflected in a tighter `ibp` argument, which this theorem takes as given through
`IBPEnclosesVals`.

The theorem states that this concrete step function satisfies `CrownTransferSound`, and thus can
be used as the trusted checker semantics in `crown_checker_encloses_semantics`.
-/
theorem alphaBetaCrown_transfer_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha)) :
    CrownTransferSound
      (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
      (ctx := ctx) (x := x)
      (step := stepAlphaBeta g ps ibp alpha beta ctx) (cert := cert) := by
  classical
  intro id hid hparents
  cases hs : stepAlphaBeta g ps ibp alpha beta ctx cert id with
  | none => trivial
  | some b =>
  cases hv : vals[id]! with
  | none => trivial
  | some v =>
  cases hbeta : getBeta? (beta := beta) id with
  | none =>
      exact alphaBetaCrown_alpha_case_sound g ps ibp alpha beta cert inputs vals ctx x
        htopo hsem hinputs hibp halpha id hid hparents (fun _ => hbeta) hs hv
  | some phases =>
      by_cases hk : (g.nodes[id]!).kind = .relu
      · exact alphaBetaCrown_relu_beta_case_sound g ps ibp alpha beta cert inputs vals ctx x
          htopo hsem hibp halpha id hid hparents hk hbeta hs hv
      · exact alphaBetaCrown_alpha_case_sound g ps ibp alpha beta cert inputs vals ctx x
          htopo hsem hinputs hibp halpha id hid hparents (fun h => absurd h hk) hs hv

end

open CrownCertSoundness
open CertSoundness

/--
A locally replayed α/β-CROWN certificate encloses every corresponding graph value.

This is the user-facing composition of `alphaBetaCrown_transfer_sound` with the generic graph
certificate checker. The certificate producer remains untrusted: `hcert` requires its entries to
agree node-by-node with TorchLean's α/β transfer function.
-/
theorem alphaBetaCrown_cert_encloses_semantics
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha))
    (hcert : CrownCertLocalOK (g := g) (step := stepAlphaBeta g ps ibp alpha beta ctx) cert) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds ℝ) (v : Val),
        cert[id]! = some b →
        vals[id]! = some v →
        EnclosesAtInput (α := ℝ) ctx x b v := by
  apply crown_checker_encloses_semantics
    (g := g) (ps := ps) (step := stepAlphaBeta g ps ibp alpha beta ctx)
    (cert := cert) (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
    htopo hsem hcert
  exact alphaBetaCrown_transfer_sound
    (g := g) (ps := ps) (ibp := ibp) (alpha := alpha) (beta := beta) (cert := cert)
    (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
    htopo hsem hinputs hibp halpha

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
