/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Extraction
public import NN.MLTheory.CROWN.Extras.IntervalLemmas
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.NonlinearOps
public import NN.Proofs.Gradients.Activation

/-!
# Certificate Soundness: Elementwise Activation Nodes

Operator cases `relu`, `tanh`, `sigmoid`, `sin`, and `cos` of the IBP certificate induction.
`tanh` and `sigmoid` share one box-level lemma about `Runtime.Ops.IBP.mapMinmax` for monotone
scalar maps; `sin` and `cos` use the Lipschitz enclosures from `NonlinearOps`.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-! ### Monotonicity of the scalar activations over the reals -/

/-- The scalar `tanh` specification is monotone over `ℝ`. -/
theorem tanh_spec_monotone_real : Monotone (Activation.Math.tanhSpec (α := ℝ)) := by
  intro a b hab
  simpa [Activation.Math.tanhSpec, MathFunctions.tanh] using
    NN.MLTheory.CROWN.IntervalLemmas.monotone_real_tanh hab

/-- The scalar sigmoid specification is monotone over `ℝ`. -/
theorem sigmoid_spec_monotone_real : Monotone (Activation.Math.sigmoidSpec (α := ℝ)) := by
  intro a b hab
  simpa only [Proofs.sigmoid_eq_inv_exp,
    NN.MLTheory.CROWN.IntervalLemmas.realSigmoid, one_div] using
    NN.MLTheory.CROWN.IntervalLemmas.monotone_realSigmoid hab

/-! ### Box-level enclosure -/

/-- `boxRelu` encloses the ReLU of an enclosed value. -/
theorem enclosesBox_boxRelu {B1 : FlatBox ℝ} {v1 : Val} (h1 : EnclosesBox B1 v1) :
    EnclosesBox (boxRelu (α := ℝ) B1) ⟨v1.n, Activation.reluSpec (α := ℝ) v1.v⟩ := by
  obtain ⟨hDim, hx⟩ := h1
  obtain ⟨n1, lo, hi⟩ := B1
  obtain ⟨m, x⟩ := v1
  simp only at hDim
  subst hDim
  simp only [castDimScalar_self] at hx
  exact ⟨rfl, NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_relu_sound (α := ℝ) n1 lo hi
    relu_mono_real x hx⟩

/-- Endpoint min/max propagation encloses any monotone elementwise map of an enclosed value. -/
theorem enclosesBox_map_minmax {B1 : FlatBox ℝ} {v1 : Val} (f : ℝ → ℝ) (hf : Monotone f)
    (h1 : EnclosesBox B1 v1) :
    EnclosesBox
      (toFlatBox (α := ℝ) B1.dim
        (NN.MLTheory.CROWN.Runtime.Ops.IBP.mapMinmax (α := ℝ) f (ofFlatBox (α := ℝ) B1)))
      ⟨v1.n, Tensor.mapSpec (α := ℝ) f v1.v⟩ := by
  obtain ⟨hDim, hx⟩ := h1
  obtain ⟨n1, lo, hi⟩ := B1
  obtain ⟨m, x⟩ := v1
  simp only at hDim
  subst hDim
  simp only [castDimScalar_self] at hx
  exact ⟨rfl, encloses_of_contains _ _ (map_minmax_sound_real f hf _ x
    (contains_of_encloses { dim := n1, lo := lo, hi := hi } x hx))⟩

/-- The `tanh` IBP transfer encloses `tanh` of an enclosed value. -/
theorem enclosesBox_ibp_tanh {B1 : FlatBox ℝ} {v1 : Val} (h1 : EnclosesBox B1 v1) :
    EnclosesBox
      (toFlatBox (α := ℝ) B1.dim
        (NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (ofFlatBox (α := ℝ) B1)))
      ⟨v1.n, Activation.tanhSpec (α := ℝ) v1.v⟩ :=
  enclosesBox_map_minmax _ tanh_spec_monotone_real h1

/-- The sigmoid IBP transfer encloses the sigmoid of an enclosed value. -/
theorem enclosesBox_ibp_sigmoid {B1 : FlatBox ℝ} {v1 : Val} (h1 : EnclosesBox B1 v1) :
    EnclosesBox
      (toFlatBox (α := ℝ) B1.dim
        (NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (ofFlatBox (α := ℝ) B1)))
      ⟨v1.n, Activation.sigmoidSpec (α := ℝ) v1.v⟩ :=
  enclosesBox_map_minmax _ sigmoid_spec_monotone_real h1

/-- The `sin` IBP transfer encloses `sin` of an enclosed value. -/
theorem enclosesBox_ibp_sin {B1 : FlatBox ℝ} {v1 : Val} (h1 : EnclosesBox B1 v1) :
    EnclosesBox
      (toFlatBox (α := ℝ) B1.dim
        (NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (ofFlatBox (α := ℝ) B1)))
      ⟨v1.n, Tensor.mapSpec (α := ℝ) (s := .dim v1.n .scalar) (fun z => Real.sin z) v1.v⟩ := by
  obtain ⟨hDim, hx⟩ := h1
  obtain ⟨n1, lo, hi⟩ := B1
  obtain ⟨m, x⟩ := v1
  simp only at hDim
  subst hDim
  simp only [castDimScalar_self] at hx
  exact ⟨rfl, encloses_of_contains _ _ (ibp_sin_sound_real _ x
    (contains_of_encloses { dim := n1, lo := lo, hi := hi } x hx))⟩

/-- The `cos` IBP transfer encloses `cos` of an enclosed value. -/
theorem enclosesBox_ibp_cos {B1 : FlatBox ℝ} {v1 : Val} (h1 : EnclosesBox B1 v1) :
    EnclosesBox
      (toFlatBox (α := ℝ) B1.dim
        (NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (ofFlatBox (α := ℝ) B1)))
      ⟨v1.n, Tensor.mapSpec (α := ℝ) (s := .dim v1.n .scalar) (fun z => Real.cos z) v1.v⟩ := by
  obtain ⟨hDim, hx⟩ := h1
  obtain ⟨n1, lo, hi⟩ := B1
  obtain ⟨m, x⟩ := v1
  simp only at hDim
  subst hDim
  simp only [castDimScalar_self] at hx
  exact ⟨rfl, encloses_of_contains _ _ (ibp_cos_sound_real _ x
    (contains_of_encloses { dim := n1, lo := lo, hi := hi } x hx))⟩

/-! ### Node-level soundness -/

/-- Certificate soundness at a `relu` node. -/
theorem relu_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .relu)
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
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_boxRelu h1

/-- Certificate soundness at a `tanh` node. -/
theorem tanh_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .tanh)
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
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_ibp_tanh h1

/-- Certificate soundness at a `sigmoid` node. -/
theorem sigmoid_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .sigmoid)
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
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_ibp_sigmoid h1

/-- Certificate soundness at a `sin` node. -/
theorem sin_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .sin)
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
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_ibp_sin h1

/-- Certificate soundness at a `cos` node. -/
theorem cos_node_encloses {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val} {vals : Array (Option Val)}
    {k : Nat} {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .cos)
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
  obtain rfl := Option.some.inj hcertStep
  exact enclosesBox_ibp_cos h1

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
