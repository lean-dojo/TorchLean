/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Extraction
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Softplus

/-!
# Softplus and safeLog certificate steps

Each proof reads the same parent values and boxes as the executable checker. Softplus uses one
parent; safeLog also reads a scalar epsilon parent, whose enclosure is part of the induction
hypothesis. The scalar inequalities and coordinatewise box arguments live in the imported module.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.CertSoundness

open Spec TorchLean
open TorchLean.Tensor

noncomputable section

/-- A successful softplus certificate step encloses the corresponding semantic value. -/
theorem softplus_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .softplus)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hparents : NN.IR.unaryParent? (nodes[k]!).parents with _ | p1 <;>
    simp only [hparents, reduceCtorEq] at hcertStep hvalStep
  rcases hgb : getBox? cert p1 with _ | B1 <;> simp only [hgb, reduceCtorEq] at hcertStep
  rcases hgv : getVal? vals p1 with _ | v1 <;> simp only [hgv, reduceCtorEq] at hvalStep
  have h1 := parents_enclosed_unary hpe hparents hgb hgv
  obtain rfl := Option.some.inj hvalStep
  exact enclosesBox_boxSoftplus h1 hcertStep

/-- SafeLog uses the enclosure of its actual scalar epsilon parent in every coordinate. -/
theorem safeLog_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .safeLog)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hpe : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  rcases hparents : NN.IR.binaryParents? (nodes[k]!).parents with _ | ⟨p1, p2⟩ <;>
    simp only [hparents, reduceCtorEq] at hcertStep hvalStep
  rcases hgb1 : getBox? cert p1 with _ | B1 <;> simp only [hgb1, reduceCtorEq] at hcertStep
  rcases hgb2 : getBox? cert p2 with _ | B2 <;> simp only [hgb2, reduceCtorEq] at hcertStep
  rcases hgv1 : getVal? vals p1 with _ | v1 <;> simp only [hgv1, reduceCtorEq] at hvalStep
  rcases hgv2 : getVal? vals p2 with _ | v2 <;> simp only [hgv2, reduceCtorEq] at hvalStep
  obtain ⟨h1, h2⟩ := parents_enclosed_binary hpe hparents hgb1 hgb2 hgv1 hgv2
  obtain ⟨ne, epsilon⟩ := v2
  obtain ⟨heDim, hvalStep⟩ := dite_eq_some_elim hvalStep
  simp only at heDim
  subst ne
  simp only [castDimScalar_self] at hvalStep
  obtain rfl := Option.some.inj hvalStep
  exact enclosesBox_boxSafeLog h1 h2 hcertStep

end

end NN.MLTheory.CROWN.Graph.CertSoundness
