/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.ConvGraphEnclosure
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Main.Extraction

/-!
# Certificate soundness for convolution

The interval and value evaluators use the same checked convolution payload and leading shape.
The node enclosure theorem therefore applies after reconstructing the semantic input tensor.
Parent lookup and the certificate induction supply its input enclosure.
-/

public section

namespace NN.MLTheory.CROWN.Graph.CertSoundness

open Spec TorchLean TorchLean.Tensor NN.MLTheory.CROWN

/-- The checked interval convolution encloses the checked value convolution on an enclosed
flat input, including all dimension casts performed by both evaluators. -/
theorem ibpConvNode_encloses_evalConvNode
    {configuration : NN.IR.ConvConfig} {parentShape outShape : Shape}
    {id : Nat} {ps : ParamStore ℝ} {Xin Bout : FlatBox ℝ} {input output : Val}
    (hinput : EnclosesBox Xin input)
    (hout : ibpConvNode configuration parentShape outShape id ps Xin = some Bout)
    (hvalue : evalConvNode? configuration parentShape outShape id ps input = some output) :
    EnclosesBox Bout output := by
  cases hparameters : ps.convCfg[id]? with
  | none => simp [evalConvNode?, hparameters] at hvalue
  | some parameters =>
      cases hplan : planConvTransfer? configuration parameters parentShape outShape with
      | none => simp [evalConvNode?, hparameters, hplan] at hvalue
      | some leading =>
          simp only [evalConvNode?, hparameters, Bind.bind, Option.bind, hplan] at hvalue
          split at hvalue
          next hdim =>
            obtain rfl := Option.some.inj hvalue
            have hflat : EnclosesBox Xin
                ⟨(parameters.input leading).size,
                  flattenSpec (ibpUnflatten input.n input.v hdim)⟩ := by
              rcases input with ⟨dim, value⟩
              dsimp only at hdim
              subst dim
              simpa [ibpUnflatten] using hinput
            exact ibpConvNode_encloses_real hparameters hplan _ hflat hout
          next => cases hvalue

/-- Certificate soundness at a convolution node with successfully validated payload geometry. -/
theorem conv_node_encloses
    {nodes : Array Node} {ps : ParamStore ℝ}
    {cert : Array (Option (FlatBox ℝ))} {inputs : Std.HashMap Nat Val}
    {vals : Array (Option Val)} {k : Nat} {configuration : NN.IR.ConvConfig}
    {B : FlatBox ℝ} {v : Val}
    (hkKind : (nodes[k]!).kind = .conv configuration)
    (hcertStep : certStepNode? nodes ps cert k = some B)
    (hvalStep : evalNode? nodes ps inputs vals k = some v)
    (hparents : ParentsEnclosed nodes cert vals k) : EnclosesBox B v := by
  simp only [certStepNode?, hkKind] at hcertStep
  simp only [evalNode?, hkKind] at hvalStep
  cases hparent : NN.IR.unaryParent? (nodes[k]!).parents with
  | none => simp [hparent] at hcertStep
  | some parent =>
      cases hnode : nodes[parent]? with
      | none => simp [hparent, hnode] at hcertStep
      | some parentNode =>
          cases hbox : getBox? cert parent with
          | none => simp [hparent, hnode, hbox] at hcertStep
          | some box =>
              cases hinput : getVal? vals parent with
              | none => simp [hparent, hnode, hinput] at hvalStep
              | some input =>
                  have hb : ibpConvNode configuration parentNode.outShape
                      (nodes[k]!).outShape k ps box = some B := by
                    simpa [hparent, hnode, hbox] using hcertStep
                  have hv : evalConvNode? configuration parentNode.outShape
                      (nodes[k]!).outShape k ps input = some v := by
                    simpa [hparent, hnode, hinput] using hvalStep
                  exact ibpConvNode_encloses_evalConvNode
                    (parents_enclosed_unary hparents hparent hbox hinput) hb hv

end NN.MLTheory.CROWN.Graph.CertSoundness
