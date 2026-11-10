/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Conv.Affine
public import TorchLean.NN.FDeriv
public import TorchLean.NN.Tape.RealGraph.FDeriv
public import TorchLean.Tensor.Contract
public import TorchLean.Tensor.Layers.Conv.Window

/-!
# Input derivative of the grouped convolution

`Conv.grouped`, mapped over any leading batch shape, is affine in its input. Its Fréchet derivative
is the CROWN convolution matrix `convLinearMatrix` acting through `Matrix.toEuclideanLin`, and its
adjoint is the vector-matrix product with the same matrix. Dilation, asymmetric padding and channel
groups live in that matrix, and the fixed bias drops out of the derivative.

The statements work over exact reals and pass between tensors and Euclidean vectors with
`Tensor.toVec` and `Tensor.ofVec`. Over one batch axis, `Conv.mapLeading_grouped_eq_conv` carries
them to `Tensor.conv` in PyTorch's layout, for `groups ∣ outC`, the packed kernel `packedKernel` of
the dense weights and the bias broadcast along the output channels: `hasFDerivAt_conv_input`,
`fderiv_conv_input_apply`, `fderiv_conv_input_eq_conv` and `adjoint_fderiv_conv_input`.
-/

public section

namespace TorchLean.CROWN

open Tensor Conv

variable {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
  (layer : Conv d inC outC kernel stride padding ℝ)
  (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)

/-- `convLinearMatrix` times the flattened input is the flattened grouped convolution with zero
bias, over the leading batch shape `leading`. -/
theorem convLinearMatrix_zero_bias
    (input : Tensor ℝ (leading ++ (inC :: inSpatial.data.toList))) :
    Tensor.contract (convLinearMatrix layer dilation paddingAfter groups leading)
        (Tensor.reshape input [_] (Shape.size_vector _).symm) _ _ [_] .matVec =
      Tensor.reshape (Tensor.mapLeading leading
        (grouped (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding)
          (paddingAfter := paddingAfter) groups layer.kernel (Tensor.full [outC] 0)) input) [_]
              (Shape.size_vector _).symm := by
  have hbias :
      convBiasBroadcast (outSpatial :=
        outSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
          (Tensor.full [outC] (0 : ℝ)) leading = Tensor.full _ 0 := by
    apply Tensor.ext_vector
    intro row
    simp [convBiasBroadcast, getScalar_eq_apply,
      getAtOrZero_eq_sum_indicator [outC], Tensor.dim_apply]
  have h := convLinearMatrix_spec (stride := stride) (padding := padding)
    { layer with bias := Tensor.full [outC] 0 } dilation paddingAfter groups leading input
  rw [hbias, Tensor.add_full_zero] at h
  exact h

/-- The grouped convolution over the leading batch shape `leading`, read on Euclidean vectors, has
Fréchet derivative `Matrix.toEuclideanLin` of `convLinearMatrix` at every input, for any bias. -/
theorem hasFDerivAt_grouped_input
    (input : Vec (Shape.size (leading ++ (inC :: inSpatial.data.toList)))) :
    HasFDerivAt
      (fun x => Tensor.toVec (Tensor.mapLeading leading
        (grouped (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding)
          (paddingAfter := paddingAfter) groups layer.kernel layer.bias)
        (Tensor.ofVec x)))
      (Matrix.toEuclideanLin (Rep.matrixEquiv ℝ _ _
        (convLinearMatrix (inSpatial := inSpatial) layer dilation paddingAfter groups
          leading))).toContinuousLinearMap
      input := by
  have h := Vec.hasFDerivAt_affine
    (Rep.matrixEquiv ℝ _ _ (convLinearMatrix (inSpatial := inSpatial)
      layer dilation paddingAfter groups leading))
    (getScalarE (convBiasBroadcast (outSpatial :=
      outSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
        layer.bias leading)) input
  apply h.congr_of_eventuallyEq
  exact Filter.Eventually.of_forall fun x => by
    change getScalarE (Tensor.reshape _ [_] (Shape.size_vector _).symm) = _
    rw [← convLinearMatrix_spec, getScalarE_add]
    erw [getScalarE_contract_matVec]
    change (Matrix.toEuclideanLin (Rep.matrixEquiv ℝ _ _ (convLinearMatrix (inSpatial := inSpatial)
      layer dilation paddingAfter groups leading))).toContinuousLinearMap
        (Tensor.toVec (Tensor.ofVec x)) + _ = _
    rw [toVec_ofVec]
    rfl

/-- The input derivative of the grouped convolution, applied to the flattened variation, is the
matrix-vector product of `convLinearMatrix` with the flattened variation. -/
theorem fderiv_grouped_input_apply
    (input variation : Tensor ℝ (leading ++ (inC :: inSpatial.data.toList))) :
    fderiv ℝ
        (fun x => Tensor.toVec (Tensor.mapLeading leading
          (grouped (inSpatial := inSpatial) (stride := stride)
            (dilation := dilation) (paddingBefore := padding)
            (paddingAfter := paddingAfter) groups layer.kernel layer.bias)
          (Tensor.ofVec x)))
        (Tensor.toVec input) (Tensor.toVec variation) =
      getScalarE (Tensor.contract (convLinearMatrix layer dilation paddingAfter groups leading)
        (Tensor.reshape variation [_] (Shape.size_vector _).symm) _ _ [_] .matVec) := by
  rw [(hasFDerivAt_grouped_input layer dilation paddingAfter groups leading
    (Tensor.toVec input)).fderiv]
  erw [getScalarE_contract_matVec]
  rfl

/-- The input derivative of the grouped convolution, applied to a variation, is the same grouped
convolution with zero bias evaluated at the variation. -/
theorem fderiv_grouped_input_zero_bias
    (input variation : Tensor ℝ (leading ++ (inC :: inSpatial.data.toList))) :
    fderiv ℝ
        (fun x => Tensor.toVec (Tensor.mapLeading leading
          (grouped (inSpatial := inSpatial) (stride := stride)
            (dilation := dilation) (paddingBefore := padding)
            (paddingAfter := paddingAfter) groups layer.kernel layer.bias)
          (Tensor.ofVec x)))
        (Tensor.toVec input) (Tensor.toVec variation) =
      Tensor.toVec (Tensor.mapLeading leading
        (grouped (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding)
          (paddingAfter := paddingAfter) groups layer.kernel (Tensor.full [outC] 0))
        variation) := by
  rw [fderiv_grouped_input_apply, convLinearMatrix_zero_bias]
  rfl

/-- The adjoint of the input derivative of the grouped convolution sends the flattened cotangent
to its vector-matrix product with `convLinearMatrix`. -/
theorem adjoint_fderiv_grouped_input
    (input : Tensor ℝ (leading ++ (inC :: inSpatial.data.toList)))
    (cotangent : Tensor ℝ (leading ++ (outC ::
      (outSpatialDilated inSpatial kernel stride dilation padding paddingAfter).data.toList))) :
    (fderiv ℝ
        (fun x => Tensor.toVec (Tensor.mapLeading leading
          (grouped (inSpatial := inSpatial) (stride := stride)
            (dilation := dilation) (paddingBefore := padding)
            (paddingAfter := paddingAfter) groups layer.kernel layer.bias)
          (Tensor.ofVec x)))
        (Tensor.toVec input)).adjoint (Tensor.toVec cotangent) =
      getScalarE (Tensor.contract (Tensor.reshape cotangent [_] (Shape.size_vector _).symm)
        (convLinearMatrix layer dilation paddingAfter groups leading) _ _ [_] .vecMat) := by
  rw [(hasFDerivAt_grouped_input layer dilation paddingAfter groups leading
    (Tensor.toVec input)).fderiv]
  apply ext_inner_right ℝ
  intro variation
  rw [ContinuousLinearMap.adjoint_inner_left]
  have h := dot_contract_matVec
    (convLinearMatrix layer dilation paddingAfter groups leading)
    (Tensor.reshape cotangent [_] (Shape.size_vector _).symm) (ofFnE variation)
  rw [dot_eq_inner_getScalarE, dot_eq_inner_getScalarE, getScalarE_contract_matVec,
    getScalarE_ofFnE] at h
  exact h

/-- `Tensor.conv` in PyTorch's layout over one batch axis, with `groups ∣ outC`, the packed kernel
`packedKernel groups hdvd layer.kernel` of the dense weights and the bias broadcast along the output
channels, has Fréchet derivative `Matrix.toEuclideanLin` of `convLinearMatrix` in its input. -/
theorem hasFDerivAt_conv_input {N : Nat}
    (h : kernel.data.toList.length = inSpatial.data.toList.length) (hdvd : groups ∣ outC)
    (input : Vec (Shape.size (N :: inC :: inSpatial.data.toList))) :
    HasFDerivAt
      (fun x => Tensor.toVec (Tensor.castShape
        (Tensor.conv (Tensor.ofVec x) (packedKernel groups hdvd layer.kernel)
            (Shape.ConvDims.channelsFirst h)
            (fun j => stride.getScalar (Fin.cast (by simp) j))
            (fun j => padding.getScalar (Fin.cast (by simp) j))
            (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
            (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1 +
          Tensor.broadcast layer.bias
            (Shape.conv (N :: inC :: inSpatial.data.toList)
              (outC :: inC / groups :: kernel.data.toList) (Shape.ConvDims.channelsFirst h)
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => padding.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1)
            (fun _ => (Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC)
              (i := inC / groups) h).output
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => padding.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1 ⟨1, by omega⟩)
            ((Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC) (i := inC / groups)
              h).broadcastDims_conv
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => padding.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1))
        (Shape.conv_channelsFirst_dilated inSpatial kernel stride dilation padding
          paddingAfter)))
      (Matrix.toEuclideanLin (Rep.matrixEquiv ℝ _ _
        (convLinearMatrix (inSpatial := inSpatial) layer dilation paddingAfter groups
          [N]))).toContinuousLinearMap
      input :=
  (hasFDerivAt_grouped_input layer dilation paddingAfter groups [N]
    input).congr_of_eventuallyEq (Filter.Eventually.of_forall fun x => congrArg Tensor.toVec
      (mapLeading_grouped_eq_conv (stride := stride) (dilation := dilation)
        (paddingBefore := padding) (paddingAfter := paddingAfter) h groups hdvd layer.kernel
        layer.bias (Tensor.ofVec x)).symm)

/-- For `Tensor.conv` with `groups ∣ outC`, the packed kernel of the dense weight tensor
`layer.kernel` and the broadcast bias, the input derivative applied to a variation is the
matrix-vector product with `convLinearMatrix`. -/
theorem fderiv_conv_input_apply {N : Nat}
    (h : kernel.data.toList.length = inSpatial.data.toList.length) (hdvd : groups ∣ outC)
    (input variation : Tensor ℝ (N :: inC :: inSpatial.data.toList)) :
    fderiv ℝ
        (fun x => Tensor.toVec (Tensor.castShape
          (Tensor.conv (Tensor.ofVec x) (packedKernel groups hdvd layer.kernel)
              (Shape.ConvDims.channelsFirst h)
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => padding.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1 +
            Tensor.broadcast layer.bias
              (Shape.conv (N :: inC :: inSpatial.data.toList)
                (outC :: inC / groups :: kernel.data.toList) (Shape.ConvDims.channelsFirst h)
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1)
              (fun _ => (Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC)
                (i := inC / groups) h).output
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1 ⟨1, by omega⟩)
              ((Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC) (i := inC / groups)
                h).broadcastDims_conv
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1))
          (Shape.conv_channelsFirst_dilated inSpatial kernel stride dilation padding
            paddingAfter)))
        (Tensor.toVec input) (Tensor.toVec variation) =
      getScalarE (Tensor.contract (convLinearMatrix layer dilation paddingAfter groups [N])
        (Tensor.reshape variation [_] (Shape.size_vector _).symm) _ _ [_] .matVec) := by
  rw [(hasFDerivAt_conv_input layer dilation paddingAfter groups h hdvd
    (Tensor.toVec input)).fderiv.trans (hasFDerivAt_grouped_input layer dilation
      paddingAfter groups [N] (Tensor.toVec input)).fderiv.symm]
  exact fderiv_grouped_input_apply layer dilation paddingAfter groups [N] input variation

/-- For `Tensor.conv` with `groups ∣ outC`, the packed kernel of the dense weight tensor
`layer.kernel` and the broadcast bias, the input derivative applied to a variation is
`Tensor.conv` of the variation with the same packed kernel and no bias. -/
theorem fderiv_conv_input_eq_conv {N : Nat}
    (h : kernel.data.toList.length = inSpatial.data.toList.length) (hdvd : groups ∣ outC)
    (input variation : Tensor ℝ (N :: inC :: inSpatial.data.toList)) :
    fderiv ℝ
        (fun x => Tensor.toVec (Tensor.castShape
          (Tensor.conv (Tensor.ofVec x) (packedKernel groups hdvd layer.kernel)
              (Shape.ConvDims.channelsFirst h)
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => padding.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1 +
            Tensor.broadcast layer.bias
              (Shape.conv (N :: inC :: inSpatial.data.toList)
                (outC :: inC / groups :: kernel.data.toList) (Shape.ConvDims.channelsFirst h)
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1)
              (fun _ => (Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC)
                (i := inC / groups) h).output
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1 ⟨1, by omega⟩)
              ((Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC) (i := inC / groups)
                h).broadcastDims_conv
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1))
          (Shape.conv_channelsFirst_dilated inSpatial kernel stride dilation padding
            paddingAfter)))
        (Tensor.toVec input) (Tensor.toVec variation) =
      Tensor.toVec (Tensor.castShape
        (Tensor.conv variation (packedKernel groups hdvd layer.kernel)
          (Shape.ConvDims.channelsFirst h)
          (fun j => stride.getScalar (Fin.cast (by simp) j))
          (fun j => padding.getScalar (Fin.cast (by simp) j))
          (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
          (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1)
        (Shape.conv_channelsFirst_dilated inSpatial kernel stride dilation padding
          paddingAfter)) := by
  rw [(hasFDerivAt_conv_input layer dilation paddingAfter groups h hdvd
    (Tensor.toVec input)).fderiv.trans (hasFDerivAt_grouped_input layer dilation
      paddingAfter groups [N] (Tensor.toVec input)).fderiv.symm]
  refine (fderiv_grouped_input_zero_bias layer dilation paddingAfter groups [N] input
    variation).trans ?_
  have hb : ∀ (t : Shape) dims hd,
      Tensor.broadcast (Tensor.full [outC] (0 : ℝ)) t dims hd = Tensor.full t 0 :=
    fun t dims hd => Tensor.Rep.ext fun c => by simp
  rw [mapLeading_grouped_eq_conv (stride := stride) (dilation := dilation)
    (paddingBefore := padding) (paddingAfter := paddingAfter) h groups hdvd]
  erw [hb, Tensor.add_full_zero]

/-- For `Tensor.conv` with `groups ∣ outC`, the packed kernel of the dense weight tensor
`layer.kernel` and the broadcast bias, the adjoint of the input derivative is the vector-matrix
product with `convLinearMatrix`. -/
theorem adjoint_fderiv_conv_input {N : Nat}
    (h : kernel.data.toList.length = inSpatial.data.toList.length) (hdvd : groups ∣ outC)
    (input : Tensor ℝ (N :: inC :: inSpatial.data.toList))
    (cotangent : Tensor ℝ (N :: outC ::
      (outSpatialDilated inSpatial kernel stride dilation padding paddingAfter).data.toList)) :
    (fderiv ℝ
        (fun x => Tensor.toVec (Tensor.castShape
          (Tensor.conv (Tensor.ofVec x) (packedKernel groups hdvd layer.kernel)
              (Shape.ConvDims.channelsFirst h)
              (fun j => stride.getScalar (Fin.cast (by simp) j))
              (fun j => padding.getScalar (Fin.cast (by simp) j))
              (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
              (fun j => dilation.getScalar (Fin.cast (by simp) j)) groups 1 +
            Tensor.broadcast layer.bias
              (Shape.conv (N :: inC :: inSpatial.data.toList)
                (outC :: inC / groups :: kernel.data.toList) (Shape.ConvDims.channelsFirst h)
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1)
              (fun _ => (Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC)
                (i := inC / groups) h).output
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1 ⟨1, by omega⟩)
              ((Shape.ConvDims.channelsFirst (n := N) (c := inC) (o := outC) (i := inC / groups)
                h).broadcastDims_conv
                (fun j => stride.getScalar (Fin.cast (by simp) j))
                (fun j => padding.getScalar (Fin.cast (by simp) j))
                (fun j => paddingAfter.getScalar (Fin.cast (by simp) j)) (fun _ => 1)
                (fun j => dilation.getScalar (Fin.cast (by simp) j)) 1))
          (Shape.conv_channelsFirst_dilated inSpatial kernel stride dilation padding
            paddingAfter)))
        (Tensor.toVec input)).adjoint (Tensor.toVec cotangent) =
      getScalarE (Tensor.contract (Tensor.reshape cotangent [_] (Shape.size_vector _).symm)
        (convLinearMatrix layer dilation paddingAfter groups [N]) _ _ [_] .vecMat) := by
  rw [(hasFDerivAt_conv_input layer dilation paddingAfter groups h hdvd
    (Tensor.toVec input)).fderiv.trans (hasFDerivAt_grouped_input layer dilation
      paddingAfter groups [N] (Tensor.toVec input)).fderiv.symm]
  exact adjoint_fderiv_grouped_input layer dilation paddingAfter groups [N] input cotangent

end TorchLean.CROWN
