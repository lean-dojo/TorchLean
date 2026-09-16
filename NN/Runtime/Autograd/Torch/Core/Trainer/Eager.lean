/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Types
public import NN.Runtime.Autograd.Torch.Core.Trainer.Recording
public import NN.Runtime.Autograd.Torch.Core.Trainer.CheckpointSchema

/-!
# Eager Trainer

Record each forward pass on a fresh tape. CUDA updates keep gradients and optimizer
moments on the device; only requested losses and state reads cross back to the host.
-/

public section

namespace Runtime.Autograd.Torch

open Spec TorchLean TorchLean.Tensor

/-- Download dense CUDA gradients in reference order, retaining each buffer's stored shape. -/
private def Internal.gradsOfRefsCuda {α : Type} [TorchLean.Storage α] [TensorTransfer α] :
    {ss : List Shape} → Array Runtime.Autograd.Cuda.AnyBuffer → RefList (TensorRef α) ss →
    IO (TorchLean.TensorPack α ss)
  | .nil, _, .nil => pure .nil
  | s :: ss, gradients, .cons ref refs => do
      let gradient ← match gradients[ref.id]? with
        | some stored => Internal.CudaBridge.ofAnyBuffer (α := α) stored
        | none => throw <| IO.userError "torch: gradient array out of bounds"
      if h : gradient.shape = s then
        let rest ← Internal.gradsOfRefsCuda (α := α) (ss := ss) gradients refs
        pure (.cons (gradient.cast h) rest)
      else
        throw <| IO.userError <|
          s!"torch: grad shape mismatch (expected {Shape.pretty s}, " ++
            s!"got {Shape.pretty gradient.shape})"

/-- Construct the eager CPU or CUDA backend for a scalar trainer. -/
def Internal.eagerScalarTrainer {α δ : Type} [TorchLean.Storage α] [TorchLean.Storage δ]
    [Context α] [TensorTransfer α] {paramShapes inputShapes dataInputShapes : List Shape}
    (options : Config) (parameters : ParamList α paramShapes)
    (validateDataInputsIO : TorchLean.TensorPack δ dataInputShapes → IO Unit)
    (loss : ScalarLoss α δ paramShapes inputShapes dataInputShapes) :
    IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes) := do
  let session ← Internal.EagerSession.new (α := α) options
  let adamStateRef ←
    IO.mkRef (Std.HashMap.emptyWithCapacity : Internal.EagerSession.CudaAdamState)
  let adamConfigRef ← IO.mkRef (none : Option Internal.EagerSession.CudaAdamConfig)
  let optimizerPathRef ← IO.mkRef (none : Option OptimizerUpdatePath)
  let checkOptimizerPath (path : OptimizerUpdatePath) : IO Unit := do
    if options.usesCuda then
      if let some current ← optimizerPathRef.get then
        unless current == path do
          throw <| IO.userError
            "torch: cannot mix generic and native optimizer updates; \
             call initOptimizer to start a fresh optimizer history"
  let useOptimizerPath (path : OptimizerUpdatePath) : IO Unit := do
    if options.usesCuda then
      checkOptimizerPath path
      optimizerPathRef.set (some path)
  let resetOptimizerState : IO Unit := do
    if options.usesCuda then
      let previous ← adamStateRef.get
      adamStateRef.set Std.HashMap.emptyWithCapacity
      adamConfigRef.set none
      optimizerPathRef.set none
      Internal.EagerSession.releaseCudaAdamState previous
  let adamSchema ← Internal.checkpointParameterSchema parameters
  let eagerLoss := loss (m := Internal.EagerM α)
  let recordLoss (inputs : TorchLean.TensorPack α inputShapes)
      (dataInputs : TorchLean.TensorPack δ dataInputShapes) :
      IO (TensorRef α [] × RefList (TensorRef α) paramShapes) := do
    validateDataInputsIO dataInputs
    session.resetTape
    (do
      let parameterRefs ← Internal.useParams (α := α) (ss := paramShapes) parameters
      let inputRefs ← Internal.useInputs (α := α) (ss := inputShapes) inputs
      let arguments :=
        RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) parameterRefs inputRefs
      let lossWithData :=
        CurriedRef.uncurry (ss := paramShapes ++ inputShapes) eagerLoss arguments
      let lossRef ←
        CurriedRef.uncurryPack (α := δ) (ss := dataInputShapes) lossWithData dataInputs
      pure (lossRef, parameterRefs)) |>.run session
  -- Scalar results and gradient packs have been copied to host tensors before this cleanup.
  -- Reset only the completed recording; parameters, Adam moments, and the RNG remain available.
  let finishRecording : IO Unit := do
    session.resetTape
    if options.usesCuda then
      Runtime.Autograd.Cuda.Buffer.collectGarbage
  let backwardParameterGradients (lossRef : TensorRef α [])
      (parameterRefs : RefList (TensorRef α) paramShapes) :
      IO (TorchLean.TensorPack α paramShapes) := do
    if Config.device session.options == .cuda then
      -- Preserve the full dense sweep, including frozen and disconnected entries.
      -- The returned host pack needs only the recorded parameter references.
      let gradients ← Internal.EagerSession.backwardDenseAllCuda (α := α) session
        lossRef (Tensor.scalar (1 : α))
      Internal.gradsOfRefsCuda (α := α) gradients parameterRefs
    else
      let gradients ← Internal.EagerSession.backwardScalarDenseAll (α := α) session lossRef
      Internal.gradsOfRefs (α := α) gradients parameterRefs
  let applyNativeGradient (optimizer : NativeOptimizer α)
      (gradients : Internal.EagerSession.CudaGradMap) : IO Unit := do
    useOptimizerPath .native
    match optimizer with
    | .sgd learningRate =>
      Internal.EagerSession.sgdStepAllCudaMap (α := α) session learningRate gradients
    | .adam learningRate beta1 beta2 epsilon =>
      Internal.EagerSession.adamStepAllCudaMap (α := α) session
        adamConfigRef adamStateRef learningRate beta1 beta2 epsilon gradients
    | .adamW learningRate weightDecay beta1 beta2 epsilon =>
      Internal.EagerSession.adamWStepAllCudaMap (α := α) session
        adamConfigRef adamStateRef learningRate weightDecay beta1 beta2 epsilon gradients
  -- The callback reads the loss before the update only when the caller needs it.
  let nativeStep {result : Type} (optimizer : NativeOptimizer α)
      (readOutput : IO (Tensor α []) → IO result) :
      Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO result)) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes (IO result)) fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes) (β := IO result) fun dataInputs => do
          checkOptimizerPath .native
          try
            let (lossRef, _) ← recordLoss inputs dataInputs
            let output ← readOutput (Internal.EagerSession.getValue (α := α) session lossRef)
            let gradients ←
              Internal.EagerSession.backwardScalarParamGradsCuda (α := α) session lossRef
            Internal.EagerSession.withCudaGradMap gradients (applyNativeGradient optimizer)
            pure output
          finally
            finishRecording
  let nativeBatchStep (optimizer : NativeOptimizer α)
      (batch : Array
        (TorchLean.TensorPack α inputShapes × TorchLean.TensorPack δ dataInputShapes))
      (readLoss : Bool) : IO (Option (Tensor α [])) := do
    if batch.isEmpty then
      throw <| IO.userError "torch: native optimizer batch must be nonempty"
    checkOptimizerPath .native
    let accumulator ← IO.mkRef
      (Std.HashMap.emptyWithCapacity : Internal.EagerSession.CudaGradMap)
    let lossAccumulator ← IO.mkRef (none : Option Runtime.Autograd.Cuda.Buffer)
    let sampleWeight := 1.0 / batch.size.toFloat
    try
      for (inputs, dataInputs) in batch do
        let (lossRef, _) ← recordLoss inputs dataInputs
        if readLoss then
          -- Weight before adding: finite losses can overflow an unnormalized batch sum.
          let tape ← session.cudaTape.get
          let sampleLoss ← okOrThrow <|
            Runtime.Autograd.Cuda.Tape.requireValue tape lossRef.id []
          let previous ← lossAccumulator.get
          let weighted := Runtime.Autograd.Cuda.Buffer.scale sampleLoss sampleWeight
          let summed := match previous with
            | none => weighted
            | some total => Runtime.Autograd.Cuda.Buffer.add total weighted
          lossAccumulator.set (some summed)
          if let some previous := previous then
            Internal.EagerSession.releaseCudaBuffer previous
            Internal.EagerSession.releaseCudaBuffer weighted
        let gradients ←
          Internal.EagerSession.backwardScalarParamGradsCuda
            (α := α) session lossRef (seed := 1 / (batch.size : α))
        Internal.EagerSession.withCudaGradMap gradients fun gradients =>
          Internal.EagerSession.accumulateCudaGradMap accumulator gradients
      let gradients ← accumulator.get
      applyNativeGradient optimizer gradients
      match ← lossAccumulator.get with
      | none => pure none
      | some total =>
          some <$> Internal.CudaBridge.ofBuffer (α := α) (s := []) total
    finally
      -- The batch owns these buffers beyond the lifetime of any individual sample tape.
      -- Release them before allocator maintenance, including when a later sample is rejected.
      try
        Internal.EagerSession.releaseCudaGradMap (← accumulator.get)
        if let some total ← lossAccumulator.get then
          Internal.EagerSession.releaseCudaBuffer total
      finally
        finishRecording
  let lossFn :
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (Tensor α []))) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes (IO (Tensor α []))) (fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes)
          (β := IO (Tensor α [])) (fun dataInputs => do
            try
              let (lossRef, _) ← recordLoss inputs dataInputs
              Internal.EagerSession.getValue (α := α) session (sh := []) lossRef
            finally
              finishRecording))
  let diff :
      Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes
        (IO (Tensor α [] × TorchLean.TensorPack α paramShapes))) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes
        (IO (Tensor α [] × TorchLean.TensorPack α paramShapes))) (fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes)
          (β := IO (Tensor α [] × TorchLean.TensorPack α paramShapes)) (fun dataInputs => do
            try
              let (lossRef, parameterRefs) ← recordLoss inputs dataInputs
              let lossValue ←
                Internal.EagerSession.getValue (α := α) session (sh := []) lossRef
              let parameterGradients ← backwardParameterGradients lossRef parameterRefs
              pure (lossValue, parameterGradients)
            finally
              finishRecording))
  let grad :
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (TorchLean.TensorPack α paramShapes))) :=
    Curried.curry (α := α) (ss := inputShapes)
      (β := Curried.Fn δ dataInputShapes (IO (TorchLean.TensorPack α paramShapes))) (fun inputs =>
        Curried.curry (α := δ) (ss := dataInputShapes)
          (β := IO (TorchLean.TensorPack α paramShapes)) (fun dataInputs => do
            try
              let (lossRef, parameterRefs) ← recordLoss inputs dataInputs
              backwardParameterGradients lossRef parameterRefs
            finally
              finishRecording))
  let stepWithLoss (learningRate : α) :
      Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO (Tensor α []))) :=
    if options.usesCuda then nativeStep (.sgd learningRate) id
    else
      Curried.curry (α := α) (ss := inputShapes)
        (β := Curried.Fn δ dataInputShapes (IO (Tensor α []))) fun inputs =>
          Curried.curry (α := δ) (ss := dataInputShapes) (β := IO (Tensor α []))
            fun dataInputs => do
              let diffForData := Curried.uncurry (α := α) (ss := inputShapes)
                (β := Curried.Fn δ dataInputShapes
                  (IO (Tensor α [] × TorchLean.TensorPack α paramShapes))) diff inputs
              let (lossValue, gradients) ← Curried.uncurry (α := δ) (ss := dataInputShapes)
                (β := IO (Tensor α [] × TorchLean.TensorPack α paramShapes)) diffForData dataInputs
              ParamList.sgdStep (α := α) (ss := paramShapes) parameters learningRate gradients
              pure lossValue
  let step (learningRate : α) :
      Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit)) :=
    if options.usesCuda then nativeStep (.sgd learningRate) (fun _ => pure ())
    else
      Curried.curry (α := α) (ss := inputShapes)
        (β := Curried.Fn δ dataInputShapes (IO Unit)) fun inputs =>
          Curried.curry (α := δ) (ss := dataInputShapes) (β := IO Unit) fun dataInputs => do
            let gradForData := Curried.uncurry (α := α) (ss := inputShapes)
              (β := Curried.Fn δ dataInputShapes (IO (TorchLean.TensorPack α paramShapes)))
              grad inputs
            let gradients ← Curried.uncurry (α := δ) (ss := dataInputShapes)
              (β := IO (TorchLean.TensorPack α paramShapes)) gradForData dataInputs
            ParamList.sgdStep (α := α) (ss := paramShapes) parameters learningRate gradients
  let adamStep? := if options.usesCuda then
      some fun learningRate beta1 beta2 epsilon =>
        nativeStep (.adam learningRate beta1 beta2 epsilon) (fun _ => pure ())
    else none
  let adamStepWithLoss? := if options.usesCuda then
      some fun learningRate beta1 beta2 epsilon =>
        nativeStep (.adam learningRate beta1 beta2 epsilon) id
    else none
  let adamWStep? := if options.usesCuda then
      some fun learningRate weightDecay beta1 beta2 epsilon =>
        nativeStep (.adamW learningRate weightDecay beta1 beta2 epsilon) (fun _ => pure ())
    else none
  let adamWStepWithLoss? := if options.usesCuda then
      some fun learningRate weightDecay beta1 beta2 epsilon =>
        nativeStep (.adamW learningRate weightDecay beta1 beta2 epsilon) id
    else none
  let optimizerStateCheckpoint? : Option OptimizerStateCheckpoint :=
    if options.usesCuda then
      some
        { save := fun path =>
            Internal.EagerSession.writeCudaAdamStateFloat32
              path adamSchema adamConfigRef adamStateRef
          load := fun path => do
            -- Restored moments select the native history too. A failed load leaves the path alone.
            checkOptimizerPath .native
            Internal.EagerSession.readCudaAdamStateFloat32
              path adamSchema adamConfigRef adamStateRef
            useOptimizerPath .native }
    else
      none
  pure
    { state := parameters
      loss := lossFn
      diff := diff
      grad := grad
      stepWithLoss := stepWithLoss
      step := step
      adamStep? := adamStep?
      adamStepWithLoss? := adamStepWithLoss?
      adamWStep? := adamWStep?
      adamWStepWithLoss? := adamWStepWithLoss?
      nativeBatchStep? := if options.usesCuda then some nativeBatchStep else none
      optimizerStateCheckpoint? := optimizerStateCheckpoint?
      resetOptimizerState := resetOptimizerState
      useOptimizerPath := useOptimizerPath
      getState := ParamList.valuesSynced (α := α) (ss := paramShapes) parameters }

end Runtime.Autograd.Torch
