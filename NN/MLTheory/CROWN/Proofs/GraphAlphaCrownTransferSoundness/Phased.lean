/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Phased.StepInversion
public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Phased.ReLUPhase

/-!
# Phased α-CROWN Graph Transfer Soundness

Pointwise soundness theorem for the phase-aware graph transfer rule.

The proof is a dispatch over two cases. Every node that is not a ReLU carrying phases is
handled by `alphaCrown_transfer_sound`, since `stepPhased` agrees with `stepAlpha` there
(`stepPhased_eq_stepAlpha`). A ReLU node with phases is handled by inverting the step
(`stepPhased_relu_phase_inv`), reading off the parent's value from the semantics, and applying
the pointwise phased ReLU lemma `enclosesAtInput_relu_phase`.
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

/-! ## Semantic glue -/

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
  have hEval := Alpha.evalNode?_eq_some_of_semLocalOK hsem hid hv
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
  have h := hibp p (parent_lt_array_size vals hsem.1 htopo hid hp)
  rw [hpre, hvp] at h
  exact h

/-! ## Per-case soundness -/

/--
Soundness of `stepPhased` at every node handled by the α-CROWN rule: all non-ReLU kinds, and
ReLU nodes with no phases. The step is literally `stepAlpha` there, so this is
`alphaCrown_transfer_sound` read at one node.
-/
theorem phasedCrown_alpha_case_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (phases : Array (Option (Array Int)))
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
    (hnoPhases : (g.nodes[id]!).kind = .relu → getPhases? phases id = none)
    {b : FlatAffineBounds ℝ} {v : Val}
    (hs : stepPhased g ps ibp alpha phases ctx cert id = some b)
    (hv : vals[id]! = some v) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  have hA := alphaCrown_transfer_sound g ps ibp alpha cert inputs vals ctx x
    htopo hsem hinputs hibp halpha id hid hparents
  rw [stepPhased_eq_stepAlpha g ps ibp alpha phases cert ctx id hnoPhases] at hs
  rw [hs, hv] at hA
  exact hA

/--
Soundness of `stepPhased` at a ReLU node carrying phases.

The step inversion supplies the parent's certificate entry, its IBP box, the α vector used, and
the accepted phase relaxations; the semantics supplies the parent's value;
`enclosesAtInput_relu_phase` does the rest.
-/
theorem phasedCrown_relu_phase_case_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (phases : Array (Option (Array Int)))
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
    {nodePhases : Array Int}
    (hk : (g.nodes[id]!).kind = .relu)
    (hphases : getPhases? phases id = some nodePhases)
    {b : FlatAffineBounds ℝ} {v : Val}
    (hs : stepPhased g ps ibp alpha phases ctx cert id = some b)
    (hv : vals[id]! = some v) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  cases hps : NN.IR.unaryParent? (g.nodes[id]!).parents with
  | none => simp [stepPhased, phasedCrownStepNode?, hk, hphases, hps] at hs
  | some p1 =>
  have hpMem : p1 ∈ (g.nodes[id]!).parents := NN.IR.mem_of_unaryParent?_eq_some hps
  obtain ⟨xin, preB, hout, αt, relaxLo, relaxHi, hxin, hpre, hαrange, hrelax, hb⟩ :=
    stepPhased_relu_phase_inv g ps ibp alpha phases cert ctx id b nodePhases p1 halpha hk hphases
      hps hs
  obtain ⟨vp, hvp, hvEq⟩ := vals_relu_eq hsem hid hk hps hv
  have hpar : EnclosesAtInput (α := ℝ) ctx x xin vp :=
    Alpha.parentsEnclosed_of_match hparents p1 hpMem xin vp (getElem!_of_getAff?_eq_some hxin) hvp
  have hbox : EnclosesBox preB vp := enclosesBox_parent htopo hsem hibp hid hpMem hpre hvp
  subst hb hvEq
  exact enclosesAtInput_relu_phase ctx x xin vp preB hout αt nodePhases relaxLo relaxHi hαrange
    hrelax hpar hbox

/-! ## Main transfer theorem -/

/--
Pointwise soundness of the graph-dialect phased α-CROWN transfer rule.

This is the phase-aware analog of `alphaCrown_transfer_sound`. The step function additionally
receives a `phases` array of per-ReLU phase constraints (active, inactive, unstable). At a ReLU node
with phases, `phaseRelaxVec?` checks each phase against the IBP pre-activation interval via
`phaseConsistentScalar?` (inactive needs `u ≤ 0`, active needs `0 ≤ l`) and, if every phase
passes, uses the phase's exact affine rule for that unit; an inconsistent phase rejects the step
rather than falling back to another relaxation. All other nodes use the α-CROWN rule.

What the phases contribute here, and what they do not. Because a phase is accepted only when
the IBP interval already implies it, the phases can never certify a sign that the supplied
`ibp` box does not fix on its own; for such stable units the phase rule coincides with the
standard relaxation, except for an active phase with a zero lower endpoint, where the phase rule
uses the identity and α-CROWN uses the slope α (or zero when the upper endpoint is zero too).
The theorem therefore establishes two things about phases: the exact rules are sound whenever the
consistency check passes, and inconsistent phases are rejected. It does not model branch-and-bound
split constraints. A split that tightens beyond IBP would have to be
reflected in a tighter `ibp` argument, which this theorem takes as given through
`IBPEnclosesVals`.

The theorem states that this concrete step function satisfies `CrownTransferSound`, and thus can
be used as the trusted checker semantics in `crown_checker_encloses_semantics`.
-/
theorem phasedCrown_transfer_sound
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (phases : Array (Option (Array Int)))
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
      (step := stepPhased g ps ibp alpha phases ctx) (cert := cert) := by
  classical
  intro id hid hparents
  cases hs : stepPhased g ps ibp alpha phases ctx cert id with
  | none => trivial
  | some b =>
  cases hv : vals[id]! with
  | none => trivial
  | some v =>
  cases hphases : getPhases? phases id with
  | none =>
      exact phasedCrown_alpha_case_sound g ps ibp alpha phases cert inputs vals ctx x
        htopo hsem hinputs hibp halpha id hid hparents (fun _ => hphases) hs hv
  | some nodePhases =>
      by_cases hk : (g.nodes[id]!).kind = .relu
      · exact phasedCrown_relu_phase_case_sound g ps ibp alpha phases cert inputs vals ctx x
          htopo hsem hibp halpha id hid hparents hk hphases hs hv
      · exact phasedCrown_alpha_case_sound g ps ibp alpha phases cert inputs vals ctx x
          htopo hsem hinputs hibp halpha id hid hparents (fun h => absurd h hk) hs hv

end

open CrownCertSoundness
open CertSoundness

/--
A locally replayed phased α-CROWN certificate encloses every corresponding graph value, given IBP
boxes that enclose those values (`IBPEnclosesVals`).

This is the composition of `phasedCrown_transfer_sound` with the generic graph certificate
checker; `EndToEnd` discharges `IBPEnclosesVals` to obtain the user-facing
`phasedCrown_cert_encloses_semantics`. The certificate producer remains untrusted: `hcert`
requires its entries to agree node-by-node with TorchLean's phased transfer function.
-/
theorem phasedCrown_cert_encloses_semantics_of_ibpEnclosesVals
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (phases : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals))
    (halpha : AlphaOK (alpha := alpha))
    (hcert : CrownCertLocalOK (g := g) (step := stepPhased g ps ibp alpha phases ctx) cert) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds ℝ) (v : Val),
        cert[id]! = some b →
        vals[id]! = some v →
        EnclosesAtInput (α := ℝ) ctx x b v := by
  apply crown_checker_encloses_semantics
    (g := g) (ps := ps) (step := stepPhased g ps ibp alpha phases ctx)
    (cert := cert) (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
    htopo hsem hcert
  exact phasedCrown_transfer_sound
    (g := g) (ps := ps) (ibp := ibp) (alpha := alpha) (phases := phases) (cert := cert)
    (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
    htopo hsem hinputs hibp halpha

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
