/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.ShapeSoundness
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Semantics

/-!
# Runtime convolution and flattened graph semantics

The checked convolution planner identifies the same leading axes and shaped input used by the
runtime evaluator. Matching payloads therefore give identical values after flattening, including
the dimension casts performed by the checked flat evaluator.
-/

public section

namespace NN.MLTheory.CROWN.Graph.CertSoundness

open Spec TorchLean TorchLean.Tensor NN.MLTheory.CROWN

attribute [local simp] Bind.bind Pure.pure Except.bind Except.pure NN.IR.throw_eq_error
  Option.bind Except.toOption

private theorem tensor_list_eq_ofFn {n : Nat} (tensor : Tensor Nat [n]) :
    Tensor.to tensor (List Nat) = List.ofFn tensor.getScalar := by
  have hdata : tensor.data = Array.ofFn tensor.getScalar := by
    apply Array.ext
    · simp
    · intro index hleft hright
      have h := Internal.Rep.data_getFlat tensor
        ⟨index, by simpa using hleft⟩
      have hindex :
          Internal.Coord.linearize (s := [n])
              ((⟨index, by simpa using hleft⟩ : Fin n), PUnit.unit) =
            ⟨index, by simpa using hleft⟩ := by
        apply Fin.ext
        change (Internal.Coord.linearize (s := [n])
          ((⟨index, by simpa using hleft⟩ : Fin n), PUnit.unit)).val = index
        erw [Internal.Coord.linearize_cons_val]
        have htail : (Internal.Coord.linearize (s := []) PUnit.unit).val < 1 :=
          (Internal.Coord.linearize (s := []) PUnit.unit).isLt
        simp only [Internal.Shape.size_nil, Nat.one_mul]
        omega
      simpa only [Array.getElem_ofFn, Tensor.getScalar_eq_apply, Internal.Rep.get,
        hindex] using h
  simpa only [Tensor.to_list_eq_data, Array.toList_ofFn] using congrArg Array.toList hdata

private theorem inferConvDims_ofFn
    (tag : String) {n : Nat} (axes : List String)
    (inputs kernels strides dilations lows highs : Fin n → Nat) {outputs : List Nat}
    (h : NN.IR.OpContracts.inferConvDims tag axes
      (List.ofFn inputs) (List.ofFn kernels) (List.ofFn strides)
      (List.ofFn dilations) (List.ofFn lows) (List.ofFn highs) = .ok outputs) :
    outputs = List.ofFn (fun axis =>
      NN.IR.OpContracts.slideOutDilated (inputs axis) (kernels axis) (strides axis)
        (dilations axis) (lows axis) (highs axis)) := by
  induction n generalizing axes outputs with
  | zero =>
      cases axes <;> simp_all [NN.IR.OpContracts.inferConvDims,
        NN.IR.OpContracts.inferConvDims.go]
  | succ n ih =>
      cases axes with
      | nil =>
          simp [NN.IR.OpContracts.inferConvDims, NN.IR.OpContracts.inferConvDims.go,
            List.ofFn_succ] at h
      | cons axis axes =>
          simp only [List.ofFn_succ, NN.IR.OpContracts.inferConvDims,
            NN.IR.OpContracts.inferConvDims.go] at h
          obtain ⟨_, _, h⟩ := NN.IR.Graph.bind_ok_iff.mp h
          obtain ⟨_, _, h⟩ := NN.IR.Graph.bind_ok_iff.mp h
          obtain ⟨_, _, h⟩ := NN.IR.Graph.bind_ok_iff.mp h
          split at h
          next => simp at h
          next =>
            obtain ⟨rest, hrest, hresult⟩ := NN.IR.Graph.bind_ok_iff.mp h
            have htail := ih axes
              (fun i => inputs i.succ) (fun i => kernels i.succ)
              (fun i => strides i.succ) (fun i => dilations i.succ)
              (fun i => lows i.succ) (fun i => highs i.succ) hrest
            simpa [htail, List.ofFn_succ] using hresult.symm

private theorem inferConvDims_tensor
    (tag : String) {n : Nat} (axes : List String)
    (inputs kernels strides dilations lows highs : Tensor Nat [n]) {outputs : List Nat}
    (h : NN.IR.OpContracts.inferConvDims tag axes
      (Tensor.to inputs (List Nat)) (Tensor.to kernels (List Nat))
      (Tensor.to strides (List Nat)) (Tensor.to dilations (List Nat))
      (Tensor.to lows (List Nat)) (Tensor.to highs (List Nat)) = .ok outputs) :
    outputs = Tensor.to (convOutSpatialDilated inputs kernels strides dilations lows highs)
      (List Nat) := by
  simp only [tensor_list_eq_ofFn] at h ⊢
  have hget :
      (convOutSpatialDilated inputs kernels strides dilations lows highs).getScalar =
        fun axis => NN.IR.OpContracts.slideOutDilated (inputs.getScalar axis)
          (kernels.getScalar axis) (strides.getScalar axis) (dilations.getScalar axis)
          (lows.getScalar axis) (highs.getScalar axis) := by
    funext axis
    simp only [convOutSpatialDilated, Tensor.getScalar_ofFn,
      NN.IR.OpContracts.slideOutDilated]
  rw [hget]
  exact inferConvDims_ofFn tag axes inputs.getScalar kernels.getScalar strides.getScalar
    dilations.getScalar lows.getScalar highs.getScalar h

private theorem inferConvConfigOutShape_output
    {configuration : NN.IR.ConvConfig} {parameters : NN.IR.ConvParams ℝ}
    {parentShape inferred : Shape}
    (hmatches : parameters.matchesConfig configuration = true)
    (hinput : parentShape = parameters.input
      (Shape.ofList (parentShape.toList.take configuration.channelAxis)))
    (hinfer : NN.IR.OpContracts.inferConvConfigOutShape "conv" configuration parentShape =
      .ok inferred) :
    inferred = parameters.output
      (Shape.ofList (parentShape.toList.take configuration.channelAxis)) := by
  have hgeometry :
      parameters.spatialRank = configuration.spatialRank ∧
      parameters.inChannels = configuration.inChannels ∧
      parameters.outChannels = configuration.outChannels ∧
      Tensor.to parameters.kernel (List Nat) = Tensor.to configuration.kernel (List Nat) ∧
      Tensor.to parameters.stride (List Nat) = Tensor.to configuration.stride (List Nat) ∧
      Tensor.to parameters.padding (List Nat) = Tensor.to configuration.padding (List Nat) ∧
      Tensor.to parameters.paddingAfter (List Nat) =
        Tensor.to configuration.paddingAfter (List Nat) ∧
      Tensor.to parameters.dilation (List Nat) = Tensor.to configuration.dilation (List Nat) ∧
      parameters.groups = configuration.groups := by
    simpa [NN.IR.ConvParams.matchesConfig, and_assoc] using hmatches
  obtain ⟨_, _, hchannels, hkernel, hstride, hlow, hhigh, hdilation, _⟩ := hgeometry
  have hsuffix :
      parentShape.toList.drop configuration.channelAxis =
        parameters.inChannels :: Tensor.to parameters.inputSpatial (List Nat) := by
    apply List.append_cancel_left (as := parentShape.toList.take configuration.channelAxis)
    have hinputList :
        parentShape.toList = parentShape.toList.take configuration.channelAxis ++
          parameters.inChannels :: Tensor.to parameters.inputSpatial (List Nat) := by
      simpa only [NN.IR.ConvParams.input, Shape.concat_eq_append] using hinput
    exact (List.take_append_drop configuration.channelAxis parentShape.toList).trans hinputList
  have hdrop :
      parentShape.toList.drop (configuration.channelAxis + 1) =
        Tensor.to parameters.inputSpatial (List Nat) := by
    simpa only [List.drop_drop, Nat.add_comm, List.drop_succ_cons, List.drop_zero]
      using congrArg (List.drop 1) hsuffix
  unfold NN.IR.OpContracts.inferConvConfigOutShape at hinfer
  obtain ⟨_, _, hinfer⟩ := NN.IR.Graph.bind_ok_iff.mp hinfer
  obtain ⟨_, _, hinfer⟩ := NN.IR.Graph.bind_ok_iff.mp hinfer
  obtain ⟨_, _, hinfer⟩ := NN.IR.Graph.bind_ok_iff.mp hinfer
  dsimp only at hinfer
  split at hinfer
  next => simp at hinfer
  next =>
    split at hinfer
    next => simp at hinfer
    next =>
      obtain ⟨_, _, hinfer⟩ := NN.IR.Graph.bind_ok_iff.mp hinfer
      cases hactual : parentShape.toList[configuration.channelAxis]? with
      | none => simp [hactual] at hinfer
      | some actual =>
          simp only [hactual] at hinfer
          split at hinfer
          next => simp at hinfer
          next =>
            split at hinfer
            next => simp at hinfer
            next =>
              obtain ⟨outputs, hdims, hresult⟩ := NN.IR.Graph.bind_ok_iff.mp hinfer
              rw [hdrop, ← hkernel, ← hstride, ← hdilation, ← hlow, ← hhigh] at hdims
              have houtputs := inferConvDims_tensor "conv" _ parameters.inputSpatial
                parameters.kernel parameters.stride parameters.dilation parameters.padding
                parameters.paddingAfter hdims
              simpa [NN.IR.ConvParams.output, Shape.concat_eq_append, houtputs, hchannels]
                using hresult.symm

private theorem planConvTransfer_input
    {configuration : NN.IR.ConvConfig} {parameters : NN.IR.ConvParams ℝ}
    {parentShape outShape leading : Shape}
    (hplan : planConvTransfer? configuration parameters parentShape outShape = some leading) :
    leading = Shape.ofList (parentShape.toList.take configuration.channelAxis) ∧
      parentShape = parameters.input leading := by
  unfold planConvTransfer? at hplan
  cases hinfer : NN.IR.OpContracts.inferConvConfigOutShape "conv" configuration parentShape with
  | error message => simp [hinfer] at hplan
  | ok inferred =>
      simp only [hinfer, Except.toOption, Bind.bind, Option.bind] at hplan
      split at hplan
      next => cases hplan
      next =>
        split at hplan
        next hshapes =>
          obtain rfl := Option.some.inj hplan
          have hshape :
              parentShape =
                parameters.input
                  (Shape.ofList (parentShape.toList.take configuration.channelAxis)) ∧
              outShape =
                parameters.output
                  (Shape.ofList (parentShape.toList.take configuration.channelAxis)) := by
            simpa using hshapes
          exact ⟨rfl, hshape.1⟩
        next => cases hplan

private theorem evalConvNode_of_evalConv_of_plan
    {payload : NN.IR.Payload ℝ} {ps : ParamStore ℝ} {id : Nat}
    {configuration : NN.IR.ConvConfig} {parentShape outShape : Shape}
    {parameters : NN.IR.ConvParams ℝ} {leading : Shape}
    {input output : SomeTensor ℝ}
    (hstores : ps.convCfg[id]? = payload.conv? id)
    (hparameters : ps.convCfg[id]? = some parameters)
    (hplan : planConvTransfer? configuration parameters parentShape outShape = some leading)
    (hshape : input.shape = parentShape)
    (heval : NN.IR.Graph.evalConv payload id configuration input = .ok output) :
    evalConvNode? configuration parentShape outShape id ps
        ⟨input.shape.size, flattenSpec input.tensor⟩ =
      some ⟨output.shape.size, flattenSpec output.tensor⟩ := by
  have hpayload : payload.conv? id = some parameters := hstores.symm.trans hparameters
  obtain ⟨hleading, hinput⟩ := planConvTransfer_input hplan
  have hinputShape : input.shape = parameters.input leading := hshape.trans hinput
  have hleadingInput :
      Shape.ofList (input.shape.toList.take configuration.channelAxis) = leading := by
    rw [hshape]
    exact hleading.symm
  rcases input with ⟨shape, tensor⟩
  dsimp only at hinputShape hleadingInput heval ⊢
  subst shape
  unfold NN.IR.Graph.evalConv at heval
  dsimp only at heval
  rw [hleadingInput] at heval
  cases hinfer : NN.IR.OpContracts.inferConvConfigOutShape "conv" configuration
      (parameters.input leading) with
  | error message => simp [hinfer] at heval
  | ok inferred =>
      cases hmatches : parameters.matchesConfig configuration with
      | false => simp [hinfer, hpayload, hmatches] at heval
      | true =>
          simp [hinfer, hpayload, hmatches] at heval
          split at heval
          next =>
            obtain rfl := Except.ok.inj heval
            simp [evalConvNode?, hparameters, hplan, ibpUnflatten,
              NN.IR.ConvParams.output]
          next hne _ => exact False.elim (hne rfl)

/-- The checked flat evaluator reproduces an actual successful runtime convolution. Payload
agreement and the input/output shapes suffice: the runtime's validation determines the same
grouped, dilated, asymmetrically padded geometry as the interval planner. -/
theorem evalConvNode?_of_evalConv
    {payload : NN.IR.Payload ℝ} {ps : ParamStore ℝ} {id : Nat}
    {configuration : NN.IR.ConvConfig} {parentShape outShape : Shape}
    {input output : SomeTensor ℝ}
    (hstores : ps.convCfg[id]? = payload.conv? id)
    (hinput : input.shape = parentShape)
    (houtput : output.shape = outShape)
    (heval : NN.IR.Graph.evalConv payload id configuration input = .ok output) :
    evalConvNode? configuration parentShape outShape id ps
        ⟨input.shape.size, flattenSpec input.tensor⟩ =
      some ⟨output.shape.size, flattenSpec output.tensor⟩ := by
  subst parentShape outShape
  have hexec := heval
  unfold NN.IR.Graph.evalConv at hexec
  obtain ⟨inferred, hinfer, hexec⟩ := NN.IR.Graph.bind_ok_iff.mp hexec
  cases hpayload : payload.conv? id with
  | none => simp [hpayload] at hexec
  | some parameters =>
      cases hmatches : parameters.matchesConfig configuration with
      | false => simp [hpayload, hmatches] at hexec
      | true =>
          simp [hpayload, hmatches] at hexec
          split at hexec
          next hshape _ =>
            let leading := Shape.ofList (input.shape.toList.take configuration.channelAxis)
            have hresult : output.shape = parameters.output leading := by
              have h := congrArg SomeTensor.shape (Except.ok.inj hexec)
              simpa [leading, NN.IR.ConvParams.output] using h.symm
            have hinferred := inferConvConfigOutShape_output hmatches hshape hinfer
            have hinferredOutput : inferred = output.shape := hinferred.trans hresult.symm
            have hshapes :
                (input.shape == parameters.input leading &&
                  output.shape == parameters.output leading) = true := by
              simpa using And.intro hshape hresult
            have hplan :
                planConvTransfer? configuration parameters input.shape output.shape =
                  some leading := by
              simp [planConvTransfer?, hinfer, hinferredOutput, hmatches, hshapes, leading]
            exact evalConvNode_of_evalConv_of_plan hstores (hstores.trans hpayload)
              hplan rfl heval
          next => simp at hexec

end NN.MLTheory.CROWN.Graph.CertSoundness
