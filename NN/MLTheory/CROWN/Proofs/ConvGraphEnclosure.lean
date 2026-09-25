/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.ConvEnclosure
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas

/-!
# Enclosure by the checked convolution node

The graph transfer validates the convolution geometry and reconstructs its shaped input from a
flat interval. The tensor enclosure theorem then applies to the actual directed convolution
folds. Flattening the result gives the same dimension-carrying enclosure used by graph
certificates.
-/

public section

namespace NN.MLTheory.CROWN.Graph.CertSoundness

open Spec TorchLean TorchLean.Tensor NN.MLTheory.CROWN

private theorem contains_ibpUnflatten {s : Shape} {box : FlatBox ℝ}
    (input : Tensor ℝ s) (hdim : box.dim = s.size)
    (h : EnclosesBox box ⟨s.size, flattenSpec input⟩) :
    Box.contains
      { lo := ibpUnflatten box.dim box.lo hdim
        hi := ibpUnflatten box.dim box.hi hdim } input := by
  rcases box with ⟨dim, lo, hi⟩
  dsimp only at hdim
  subst dim
  obtain ⟨_, h⟩ := h
  simp only [castDimScalar_self] at h
  apply (ConvProof.box_contains_flatten_iff _ input).mp
  simpa [flattenBox, ibpUnflatten, ofFlatBox] using
    contains_of_encloses ⟨s.size, lo, hi⟩ (flattenSpec input) h

private theorem enclosesBox_flatten {s : Shape} {box : Box ℝ s} {input : Tensor ℝ s}
    (h : Box.contains box input) :
    EnclosesBox
      ⟨s.size, flattenSpec box.lo, flattenSpec box.hi⟩
      ⟨s.size, flattenSpec input⟩ := by
  refine ⟨rfl, ?_⟩
  exact encloses_of_contains (flattenBox box) (flattenSpec input)
    ((ConvProof.box_contains_flatten_iff box input).mpr h)

/-- A successful checked convolution transfer encloses the actual grouped/batched convolution
of an enclosed shaped input. All geometry is the payload geometry accepted by the planner. -/
theorem ibpConvNode_encloses_real
    {configuration : NN.IR.ConvConfig} {parentShape outShape : Shape}
    {id : Nat} {ps : ParamStore ℝ} {Xin Bout : FlatBox ℝ}
    {parameters : NN.IR.ConvParams ℝ} {leading : Shape}
    (hparameters : ps.convCfg[id]? = some parameters)
    (hplan : planConvTransfer? configuration parameters parentShape outShape = some leading)
    (input : Tensor ℝ (parameters.input leading))
    (hinput : EnclosesBox Xin ⟨(parameters.input leading).size, flattenSpec input⟩)
    (hout : ibpConvNode configuration parentShape outShape id ps Xin = some Bout) :
    EnclosesBox Bout
      ⟨(parameters.output leading).size,
        flattenSpec (Tensor.mapLeading leading
          (groupedConvSpec (inSpatial := parameters.inputSpatial)
            (stride := parameters.stride) (dilation := parameters.dilation)
            (paddingBefore := parameters.padding) (paddingAfter := parameters.paddingAfter)
            parameters.groups parameters.spec.kernel parameters.spec.bias) input)⟩ := by
  simp only [ibpConvNode, hparameters, Bind.bind, Option.bind, hplan] at hout
  split at hout
  next hdim =>
    obtain rfl := Option.some.inj hout
    have hshaped := contains_ibpUnflatten input hdim hinput
    have houtput := ConvProof.ibpConv_contains_groupedConv_real parameters.spec
      parameters.dilation parameters.paddingAfter parameters.groups leading _ input hshaped
    simpa only [NN.IR.ConvParams.output, Tensor.to_list_eq_data] using
      enclosesBox_flatten houtput
  next => cases hout

end NN.MLTheory.CROWN.Graph.CertSoundness
