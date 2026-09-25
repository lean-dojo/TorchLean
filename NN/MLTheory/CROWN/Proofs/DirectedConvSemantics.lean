/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.Conv
public import NN.MLTheory.CROWN.Proofs.DirectedIBPConvolution

/-!
# The scalar semantics of the convolution affine map

Positive dilation makes each matrix coefficient a stored weight or zero. Consequently scalar
interpretation commutes with the matrix converter, even when ordinary scalar addition rounds.
The real spatial convolution then satisfies the affine equation used by backward CROWN.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean NN.IR
open TorchLean.Tensor
open NN.MLTheory.CROWN
open scoped BigOperators

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- Interpret the stored kernel and bias without performing scalar arithmetic. -/
def realConvSpec {d inC outC : Nat} {kernel stride padding : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α) :
    ConvSpec d inC outC kernel stride padding ℝ :=
  ⟨Tensor.map value layer.kernel, Tensor.map value layer.bias⟩

/-- For positive dilation, the scalar convolution matrix denotes its exact real counterpart. -/
theorem convLinearMatrix_map_value
    {d inC outC : Nat} {kernel stride padding inSpatial : Tensor Nat [d]}
    (layer : ConvSpec d inC outC kernel stride padding α)
    (dilation paddingAfter : Tensor Nat [d]) (groups : Nat) (leading : Shape)
    (hdil : ∀ d ∈ Tensor.to dilation (List Nat), 0 < d) :
    convLinearMatrix (inSpatial := inSpatial) (realConvSpec layer)
        dilation paddingAfter groups leading =
      Tensor.map value (convLinearMatrix (inSpatial := inSpatial) layer
        dilation paddingAfter groups leading) := by
  simp only [convLinearMatrix, realConvSpec, Tensor.map_dim, Tensor.map_scalar]
  apply congrArg Tensor.dim
  funext i
  apply congrArg Tensor.dim
  funext j
  apply congrArg Tensor.scalar
  split
  · exact (LawfulBoundOps.toReal_zero (α := α)).symm
  · simp only [getAtOrZero_map_value]
    exact ConvProof.convKernelCoefficient_map value LawfulBoundOps.toReal_zero
      _ _ _ _ _ _ _ hdil

/-- Bias broadcasting only copies stored values, so it commutes with scalar interpretation. -/
theorem convBiasBroadcast_map_value
    {d outC : Nat} {outSpatial : Tensor Nat [d]} (bias : Tensor α [outC]) (leading : Shape) :
    convBiasBroadcast (outSpatial := outSpatial) (Tensor.map value bias) leading =
      Tensor.map value (convBiasBroadcast (outSpatial := outSpatial) bias leading) := by
  apply Tensor.ext_vector
  intro i
  simp only [convBiasBroadcast, Tensor.getScalar_map, Tensor.getScalar_dim_entry,
    Tensor.item_scalar, getAtOrZero_map_value]

/-- The exact spatial convolution is the real affine map of the stored scalar coefficients. -/
theorem convolutionPoint_eq_affine (parameters : ConvParams α) (leading : Shape)
    (f : Nat → ℝ) (hdil : ∀ d ∈ Tensor.to parameters.dilation (List Nat), 0 < d) :
    convolutionPoint parameters leading f =
      addSpec
        (matVecMulSpec (Tensor.map value (affOfConv parameters leading).A)
          (pointVector (parameters.input leading).size f))
        (Tensor.map value (affOfConv parameters leading).c) := by
  have h := ConvProof.conv_linear_matrix_add_bias_eq_grouped_conv
    (realConvSpec parameters.spec) parameters.dilation parameters.paddingAfter parameters.groups
    leading (unflattenSpec (parameters.input leading)
      (pointVector (parameters.input leading).size f))
  rw [convLinearMatrix_map_value _ _ _ _ _ hdil] at h
  simpa only [realConvSpec, convBiasBroadcast_map_value, ConvParams.input, ConvParams.output,
    Tensor.to_list_eq_data, flattenSpec_unflattenSpec, convolutionPoint, affOfConv,
    AffineVec.ofLinear] using h.symm

private theorem inferConvDims_dilation_pos (tag : String) (axisNames : List String)
    (inputs kernels strides dilations paddingBefore paddingAfter output : List Nat)
    (h : OpContracts.inferConvDims tag axisNames inputs kernels strides dilations
      paddingBefore paddingAfter = .ok output) :
    ∀ d ∈ dilations, 0 < d := by
  unfold OpContracts.inferConvDims at h
  fun_induction OpContracts.inferConvDims.go tag axisNames inputs kernels strides dilations
      paddingBefore paddingAfter generalizing output
  case case1 => simp
  case case2 axis axes input inputs kernel kernels stride strides dilation dilations
      low lows high highs ih =>
    simp only [OpContracts.checkPositive, pure, Except.pure, bind, Except.bind, throw] at h
    split at h
    · simp_all
    · split at h
      · simp_all
      · split at h
        · simp_all
        · rename_i hdil
          have hpositive : 0 < dilation := by
            cases dilation with
            | zero => cases hdil
            | succ n => exact Nat.succ_pos n
          split at h
          · contradiction
          · cases hrec : OpContracts.inferConvDims.go tag axes inputs kernels strides
                dilations lows highs with
            | error err => simp [hrec] at h
            | ok rest =>
                have htail := ih rest hrec
                intro d hm
                rcases List.mem_cons.mp hm with rfl | hm
                · exact hpositive
                · exact htail d hm
  case case3 => contradiction

private theorem inferConvConfigOutShape_dilation_pos
    (tag : String) (config : ConvConfig) (parent output : Shape)
    (h : OpContracts.inferConvConfigOutShape tag config parent = .ok output) :
    ∀ d ∈ Tensor.to config.dilation (List Nat), 0 < d := by
  simp only [OpContracts.inferConvConfigOutShape, bind, Except.bind] at h
  repeat' (split at h <;> try contradiction)
  rename_i hdims
  exact inferConvDims_dilation_pos _ _ _ _ _ _ _ _ _ hdims

omit [BoundOps α] [LawfulBoundOps α] in
/-- Every accepted convolution plan has positive stored dilation on each spatial axis. -/
theorem planConvTransfer?_dilation_pos
    {configuration : ConvConfig} {parameters : ConvParams α}
    {parentShape outShape leading : Shape}
    (h : planConvTransfer? configuration parameters parentShape outShape = some leading) :
    ∀ d ∈ Tensor.to parameters.dilation (List Nat), 0 < d := by
  unfold planConvTransfer? at h
  cases hinfer : OpContracts.inferConvConfigOutShape "conv" configuration parentShape with
  | error err => simp [hinfer, Except.toOption] at h
  | ok inferred =>
      simp only [hinfer, Except.toOption, Option.bind_eq_bind, Option.bind_some] at h
      split at h
      · contradiction
      · rename_i hmatch
        have hm : parameters.matchesConfig configuration = true := by
          cases hm : parameters.matchesConfig configuration with
          | false => simp [hm] at hmatch
          | true => rfl
        simp only [ConvParams.matchesConfig, Bool.and_eq_true, beq_iff_eq] at hm
        have hdil : Tensor.to parameters.dilation (List Nat) =
            Tensor.to configuration.dilation (List Nat) := hm.1.2
        rw [hdil]
        exact inferConvConfigOutShape_dilation_pos _ _ _ _ hinfer

/-- The actual real convolution equation implies the affine equation used by backward CROWN.

All geometry assumptions follow from the same successful plan that the graph engine consumes.
-/
theorem ConvolutionNodeEquation.nodeEquation
    {nodes : Array Node} {ps : ParamStore α} {ibp : Array (Option (FlatBox α))}
    {dims : Nat → Nat} {v : Nat → Nat → ℝ} {id : Nat} {configuration : ConvConfig}
    (heq : ConvolutionNodeEquation nodes ps dims v id configuration)
    (hkind : nodes[id]!.kind = .conv configuration) :
    NodeEquation nodes ps ibp dims v id := by
  simp only [NodeEquation, hkind]
  intro p hp parameters hparameters parent hparent leading hplan
  obtain ⟨hpdim, hid, hvalue⟩ := heq p hp parameters hparameters parent hparent leading hplan
  refine ⟨hid, hpdim, ?_⟩
  intro i
  rw [hvalue i, convolutionPoint_eq_affine _ _ _ (planConvTransfer?_dilation_pos hplan)]
  simp only [read_fin, getScalar_add_spec, Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec,
    Tensor.getScalar_map, pointVector, Tensor.getScalar_ofFn, get2, Spec.get, Tensor.unstack_map]

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
