/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Extraction
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas

/-!
# Certificate Soundness: Leaf and Pass-Through Nodes

Operator cases `input`, `const`, and `detach` of the IBP certificate induction.  These nodes do
no arithmetic: `input` is discharged by the input-box assumption, `const` by the point-box
enclosure, and `detach` by forwarding the parent enclosure.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-- Certificate soundness at an `input` node, given that the semantic input lies in the certified
input box. -/
theorem input_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .input)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hin : ∃ Bin vin, ps.inputBoxes[k]? = some Bin ∧ inputs[k]? = some vin ∧ EnclosesBox Bin vin) :
    EnclosesBox B v := by
  obtain ⟨Bin, vin, hB, hv, hEnc⟩ := hin
  simp only [certStepNode?, hkKind, hB] at hcertStep
  simp only [evalNode?, hkKind, hv] at hvalStep
  obtain rfl := Option.some.inj hcertStep
  obtain rfl := Option.some.inj hvalStep
  exact hEnc

/-- Certificate soundness at a `const` node: the point box `[c, c]` encloses `c`. -/
theorem const_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val} {valueShape : Shape}
    (hkKind : (nodes[k]!).kind = .const valueShape)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v) :
    EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hconst : ps.constVals[k]? with _ | c <;>
    simp only [hconst, reduceCtorEq] at hcertStep hvalStep
  obtain rfl := Option.some.inj hcertStep
  obtain rfl := Option.some.inj hvalStep
  exact ⟨rfl, encloses_point_self_real c.v⟩

/-- Certificate soundness at a `detach` node: box and value are forwarded from the parent. -/
theorem detach_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .detach)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hparents : NN.IR.unaryParent? (nodes[k]!).parents with _ | p1 <;>
    simp only [hparents, reduceCtorEq] at hcertStep hvalStep
  exact parents_enclosed_unary hpe hparents hcertStep hvalStep

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
