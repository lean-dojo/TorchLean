/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.Conv
public import NN.Proofs.Autograd.FDeriv.Core
public import NN.Proofs.Autograd.Tape.Core.FDeriv

/-!
# Grouped convolution input derivatives

The actual grouped convolution, mapped over arbitrary leading batch dimensions, is affine in its
input. Its Fréchet derivative multiplies by the CROWN convolution matrix, and its adjoint multiplies
by the transpose. Dilation, asymmetric padding, and grouped channels are already included in that
matrix. The fixed bias disappears from the derivative.

The statements use the existing tensor/vector round trip to express derivatives in Euclidean
space. They concern exact real arithmetic and the total convolution definitions.
-/

public section

namespace NN.MLTheory.CROWN.ConvProof

open Spec TorchLean TorchLean.Tensor
open Proofs.Autograd Spec.Conv.Internal

variable {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
  (layer : ConvSpec d inC outC kernel stride padding ℝ)
  (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)

/-- Removing the bias gives exactly the matrix action, including all leading batch dimensions. -/
theorem conv_linear_matrix_eq_grouped_conv_zero_bias
    (input : Tensor ℝ (leading.concat (Shape.ofList (inC :: inSpatial.data.toList)))) :
    matVecMulSpec (convLinearMatrix layer dilation paddingAfter groups leading)
        (flattenSpec input) =
      flattenSpec (Tensor.mapLeading leading
        (groupedConvSpec (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding)
          (paddingAfter := paddingAfter) groups layer.kernel (Tensor.full [outC] 0)) input) := by
  have hbias :
      convBiasBroadcast (outSpatial :=
        convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
          (Tensor.full [outC] (0 : ℝ)) leading = Tensor.full _ 0 := by
    apply Tensor.ext_vector
    intro row
    simp [convBiasBroadcast, getScalar_eq_apply, Tensor.dim,
      getAtOrZero_eq_sum_indicator [outC], MultiIndex.get, Tensor.full, Tensor.unstack]
  have h := conv_linear_matrix_add_bias_eq_grouped_conv (stride := stride) (padding := padding)
    { layer with bias := Tensor.full [outC] 0 } dilation paddingAfter groups leading input
  rw [hbias, add_spec_zero_right] at h
  exact h

/-- The grouped/batched convolution specification has the convolution matrix as its input
derivative. No restriction on the fixed bias is needed. -/
theorem hasFDerivAt_grouped_conv_input
    (input : Vec (leading.concat (Shape.ofList (inC :: inSpatial.data.toList))).size) :
    HasFDerivAt
      (fun x => tensorToVec (Tensor.mapLeading leading
        (groupedConvSpec (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding)
          (paddingAfter := paddingAfter) groups layer.kernel layer.bias)
        (vecToTensor x)))
      (matCLM (tensorToMatrix
        (convLinearMatrix (inSpatial := inSpatial) layer dilation paddingAfter groups leading)))
      input := by
  have h := hasFDerivAt_affine
    (tensorToMatrix (convLinearMatrix (inSpatial := inSpatial)
      layer dilation paddingAfter groups leading))
    (getScalarE (convBiasBroadcast (outSpatial :=
      convOutSpatialDilated inSpatial kernel stride dilation padding paddingAfter)
        layer.bias leading)) input
  apply h.congr_of_eventuallyEq
  exact Filter.Eventually.of_forall fun x => by
    change getScalarE (flattenSpec _) = _
    rw [← conv_linear_matrix_add_bias_eq_grouped_conv,
      getScalarE_add_spec, getScalarE_mat_vec_mul_spec]
    change matCLM _ (tensorToVec (vecToTensor x)) + _ = _
    rw [tensorToVec_vecToTensor]
    rfl

/-- Applying the actual input derivative to a tensor variation is a matrix-vector product. -/
theorem fderiv_grouped_conv_input_apply
    (input variation :
      Tensor ℝ (leading.concat (Shape.ofList (inC :: inSpatial.data.toList)))) :
    fderiv ℝ
        (fun x => tensorToVec (Tensor.mapLeading leading
          (groupedConvSpec (inSpatial := inSpatial) (stride := stride)
            (dilation := dilation) (paddingBefore := padding)
            (paddingAfter := paddingAfter) groups layer.kernel layer.bias)
          (vecToTensor x)))
        (tensorToVec input) (tensorToVec variation) =
      getScalarE (matVecMulSpec (convLinearMatrix layer dilation paddingAfter groups leading)
        (flattenSpec variation)) := by
  rw [(hasFDerivAt_grouped_conv_input layer dilation paddingAfter groups leading
    (tensorToVec input)).fderiv, getScalarE_mat_vec_mul_spec]
  rfl

/-- The input derivative is the same grouped/batched convolution with zero bias, evaluated at
the input variation. This form can be used directly for interval propagation of derivatives. -/
theorem fderiv_grouped_conv_input_eq_zero_bias
    (input variation :
      Tensor ℝ (leading.concat (Shape.ofList (inC :: inSpatial.data.toList)))) :
    fderiv ℝ
        (fun x => tensorToVec (Tensor.mapLeading leading
          (groupedConvSpec (inSpatial := inSpatial) (stride := stride)
            (dilation := dilation) (paddingBefore := padding)
            (paddingAfter := paddingAfter) groups layer.kernel layer.bias)
          (vecToTensor x)))
        (tensorToVec input) (tensorToVec variation) =
      tensorToVec (Tensor.mapLeading leading
        (groupedConvSpec (inSpatial := inSpatial) (stride := stride)
          (dilation := dilation) (paddingBefore := padding)
          (paddingAfter := paddingAfter) groups layer.kernel (Tensor.full [outC] 0))
        variation) := by
  rw [fderiv_grouped_conv_input_apply,
    conv_linear_matrix_eq_grouped_conv_zero_bias]
  rfl

/-- The adjoint of the actual convolution input derivative is the vector-matrix product with
the same CROWN matrix. Reshaping this vector gives the input cotangent. -/
theorem adjoint_fderiv_grouped_conv_input
    (input : Tensor ℝ (leading.concat (Shape.ofList (inC :: inSpatial.data.toList))))
    (cotangent : Tensor ℝ (leading.concat (Shape.ofList (outC ::
      (convOutSpatialDilated inSpatial kernel stride dilation padding
        paddingAfter).data.toList)))) :
    (fderiv ℝ
        (fun x => tensorToVec (Tensor.mapLeading leading
          (groupedConvSpec (inSpatial := inSpatial) (stride := stride)
            (dilation := dilation) (paddingBefore := padding)
            (paddingAfter := paddingAfter) groups layer.kernel layer.bias)
          (vecToTensor x)))
        (tensorToVec input)).adjoint (tensorToVec cotangent) =
      getScalarE (vecMatMulSpec (flattenSpec cotangent)
        (convLinearMatrix layer dilation paddingAfter groups leading)) := by
  rw [(hasFDerivAt_grouped_conv_input layer dilation paddingAfter groups leading
    (tensorToVec input)).fderiv]
  apply ext_inner_right ℝ
  intro variation
  rw [ContinuousLinearMap.adjoint_inner_left]
  have h := dot_mat_linear_adjoint
    (convLinearMatrix layer dilation paddingAfter groups leading)
    (flattenSpec cotangent) (ofFnE variation)
  rw [dot_eq_inner_vec, dot_eq_inner_vec, getScalarE_mat_vec_mul_spec,
    getScalarE_ofFnE] at h
  exact h

end NN.MLTheory.CROWN.ConvProof
