/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.LeafOps
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.ArithOps
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.UnaryOps
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Softplus
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.AffineOps
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Conv

/-!
# Graph IBP Certificate Soundness

The induction theorem: local IBP certificate consistency plus local semantic consistency implies
that every certified node box encloses the corresponding semantic value.

The per-operator cases live under `GraphCertSoundness/Main/`; this file states the graph-level
assumptions, dispatches on the node kind, and runs the strong induction over node ids.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## Main theorem: local IBP certificate implies semantic enclosure (supported subset)

We use strong induction on node id, assuming a topological order:
every parent id is strictly smaller than the node id.
-/

/-- Topological order assumption: all parent ids are strictly smaller than the node id. -/
def TopoSorted (g : Graph) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    ∀ p : Nat, p ∈ (g.nodes[id]!).parents → p < id

/-- A graph is supported by this soundness theorem if every node kind is in our supported subset. -/
def Supported (g : Graph) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    match (g.nodes[id]!).kind with
    | .input | .const _ | .detach
    | .add | .sub | .mulElem | .relu
    | .linear | .matmul | .concat _ | .conv _
    | .tanh | .sigmoid | .softplus | .safeLog | .sin | .cos => True
    | _ => False

/-- Inputs are well-formed if every `.input` node has a value, and that value is enclosed by
its input box from `ParamStore.inputBoxes`. -/
def InputsEnclosed (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    (g.nodes[id]!).kind = .input →
      ∃ B v, ps.inputBoxes[id]? = some B ∧ inputs[id]? = some v ∧ EnclosesBox B v

/-- A topologically earlier parent fits every node-indexed array with graph-sized storage. -/
theorem parent_lt_array_size
    {β : Type} {g : Graph} (entries : Array β)
    (hSize : entries.size = g.nodes.size) (htopo : TopoSorted g)
    {id parent : Nat} (hid : id < g.nodes.size) (hparent : parent ∈ (g.nodes[id]!).parents) :
    parent < entries.size := by
  rw [hSize]
  exact lt_trans (htopo id hid parent hparent) hid

/-- The induction hypothesis for all earlier nodes yields `ParentsEnclosed` at node `k`, because
`TopoSorted` places every parent strictly before `k`. -/
theorem parents_enclosed_of_ih
    {g : Graph} {cert : Array (Option (FlatBox ℝ))} {vals : Array (Option Val)} {k : Nat}
    (htopo : TopoSorted g) (hk : k < g.nodes.size)
    (ih : ∀ p : Nat, p < k → p < g.nodes.size → ∀ (B : FlatBox ℝ) (v : Val),
      cert[p]! = some B → vals[p]! = some v → EnclosesBox B v) :
    ParentsEnclosed g.nodes cert vals k := by
  intro p hp Bp vp hBp hvp
  have hpk : p < k := htopo k hk p hp
  exact ih p hpk (lt_trans hpk hk) Bp vp
    (getElem!_of_getBox?_eq_some hBp) (getElem!_of_getVal?_eq_some hvp)

/-- One step of the induction: a supported node whose parents are enclosed is itself enclosed.
This is the dispatch over operator kinds; each arm is a lemma under `Main/`. -/
theorem node_encloses_of_parents
    {g : Graph} {ps : ParamStore ℝ} {cert : Array (Option (FlatBox ℝ))}
    {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)} {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hsupp : Supported g) (hinputs : InputsEnclosed g ps inputs) (hk : k < g.nodes.size)
    (hcertStep : certStepNode? g.nodes ps cert k = some B)
    (hvalStep : evalNode? g.nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed g.nodes cert vals k) : EnclosesBox B v := by
  have hsupk := hsupp k hk
  cases hkKind : (g.nodes[k]!).kind <;> simp only [hkKind] at hsupk
  case input => exact input_node_encloses hkKind hcertStep hvalStep (hinputs k hk hkKind)
  case const _ => exact const_node_encloses hkKind hcertStep hvalStep
  case detach => exact detach_node_encloses hkKind hcertStep hvalStep hpe
  case add => exact add_node_encloses hkKind hcertStep hvalStep hpe
  case sub => exact sub_node_encloses hkKind hcertStep hvalStep hpe
  case mulElem => exact mulElem_node_encloses hkKind hcertStep hvalStep hpe
  case relu => exact relu_node_encloses hkKind hcertStep hvalStep hpe
  case tanh => exact tanh_node_encloses hkKind hcertStep hvalStep hpe
  case sigmoid => exact sigmoid_node_encloses hkKind hcertStep hvalStep hpe
  case softplus => exact softplus_node_encloses hkKind hcertStep hvalStep hpe
  case safeLog => exact safeLog_node_encloses hkKind hcertStep hvalStep hpe
  case sin => exact sin_node_encloses hkKind hcertStep hvalStep hpe
  case cos => exact cos_node_encloses hkKind hcertStep hvalStep hpe
  case linear => exact linear_node_encloses hkKind hcertStep hvalStep hpe
  case matmul => exact matmul_node_encloses hkKind hcertStep hvalStep hpe
  case concat axis => exact concat_node_encloses hkKind hcertStep hvalStep hpe
  case conv configuration => exact conv_node_encloses hkKind hcertStep hvalStep hpe

/-!
### The enclosure theorem

Assumptions:
* `TopoSorted g`: induction works (parents are earlier).
* `Supported g`: every node kind is handled by the proof.
* `CertLocalOK g ps cert`: the certificate is locally consistent with the IBP step.
* `InputsEnclosed g ps inputs`: semantic inputs are inside the certified input boxes.
* `SemLocalOK g ps inputs vals`: `vals` is a locally-consistent semantic interpretation.

Conclusion:
* For every node `id`, if the semantics produces a value `v` and the certificate has a box `B`,
  then `B` encloses `v`.
-/

/-- Enclosure for every node where both the certificate box and the semantic value are present.

This is the non-vacuous companion of `cert_encloses_semantics`: it quantifies over the box and
value explicitly instead of matching on `cert[id]!` and `vals[id]!`.  The strong induction runs
here, since this shape is the one the operator lemmas need for their parents. -/
theorem cert_encloses_semantics_of_some
    (g : Graph) (ps : ParamStore ℝ)
    (cert : Array (Option (FlatBox ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hcert : CertLocalOK (g := g) (ps := ps) cert)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsEnclosed g ps inputs) :
    ∀ id : Nat, id < g.nodes.size → ∀ (B : FlatBox ℝ) (v : Val),
      cert[id]! = some B → vals[id]! = some v → EnclosesBox B v := by
  obtain ⟨-, hnode⟩ := hcert
  obtain ⟨-, hsemNode⟩ := hsem
  intro id
  induction id using Nat.strong_induction_on with
  | _ k ih =>
    intro hk B v hck hvk
    exact node_encloses_of_parents hsupp hinputs hk
      ((hnode k hk).symm.trans hck) ((hsemNode k hk).symm.trans hvk)
      (parents_enclosed_of_ih htopo hk ih)

/-- Enclosure of every certified node box around the corresponding semantic value.

The conclusion matches on `cert[id]!` and `vals[id]!` and is trivially true when either entry is
missing.  This shape is kept because downstream files instantiate it directly with the runtime
arrays produced by `runIBP?` and the evaluator, where presence of an entry is not known up front.
See `cert_encloses_semantics_of_some` for the explicitly quantified form. -/
theorem cert_encloses_semantics
    (g : Graph) (ps : ParamStore ℝ)
    (cert : Array (Option (FlatBox ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hcert : CertLocalOK (g := g) (ps := ps) cert)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsEnclosed g ps inputs) :
    ∀ id : Nat, id < g.nodes.size →
      match cert[id]!, vals[id]! with
      | some B, some v => EnclosesBox B v
      | _, _ => True := by
  intro id hid
  cases hck : cert[id]! with
  | none => exact trivial
  | some B =>
    cases hvk : vals[id]! with
    | none => exact trivial
    | some v =>
      exact cert_encloses_semantics_of_some g ps cert inputs vals htopo hsupp hcert hsem hinputs
        id hid B v hck hvk

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
