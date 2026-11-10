/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.ScalarTrainer
public import TorchLean.NN.ScalarTrainer.Recording
public import TorchLean.NN.ScalarTrainer.Checkpoint

/-!
# Eager Trainer

The eager trainer records each forward pass on a fresh tape. Native CUDA updates keep gradients and
optimizer moments on the device, and only requested losses and state reads cross back to the host.
-/

public section

namespace TorchLean

open TorchLean.Tensor

/-- Download dense CUDA gradients in reference order, checking each buffer's stored shape. -/
private def RefList.gradsCUDA {α : Type} [Storage α] [TensorTransfer α]
    {ss : List Shape} (gradients : Array LibTorch.SomeBuffer)
    (refs : Pack (TensorRef α) ss) : IO (TensorPack α ss) :=
  RefList.read (fun {s} ref => do
    let gradient ← match gradients[ref.id]? with
      | some stored => TensorTransfer.ofSomeBuffer (α := α) stored
      | none => throw <| IO.userError "torch: gradient array out of bounds"
    if h : gradient.shape = s then
      pure (gradient.cast h)
    else
      throw <| IO.userError <|
        s!"torch: grad shape mismatch (expected {s}, " ++
          s!"got {gradient.shape})") refs

/-- Construct the eager CPU or CUDA backend for a scalar trainer. -/
def ScalarTrainer.eager {α δ : Type} [Storage α] [Storage δ]
    [Context α] [TensorTransfer α] {paramShapes inputShapes dataInputShapes : List Shape}
    (options : Session.Config) (parameters : ParamList α paramShapes)
    (validateDataInputsIO : TensorPack δ dataInputShapes → IO Unit)
    (loss : ScalarLoss α δ paramShapes inputShapes dataInputShapes) :
    IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes) := do
  let session ← Session.create (α := α) options
  let options := session.options
  let adamStateRef ←
    IO.mkRef (Std.HashMap.emptyWithCapacity : Session.CUDAAdamState)
  let adamConfigRef ← IO.mkRef (none : Option Session.CUDAAdamConfig)
  let optimizerPathRef ← IO.mkRef (none : Option OptimizerUpdatePath)
  let checkOptimizerPath (path : OptimizerUpdatePath) : IO Unit := do
    if options.usesCUDA then
      if let some current ← optimizerPathRef.get then
        unless current == path do
          throw <| IO.userError
            "torch: cannot mix generic and native optimizer updates; \
             call initOptimizer to start a fresh optimizer history"
  let useOptimizerPath (path : OptimizerUpdatePath) : IO Unit := do
    if options.usesCUDA then
      checkOptimizerPath path
      optimizerPathRef.set (some path)
  let resetOptimizerState : IO Unit := do
    if options.usesCUDA then
      let previous ← adamStateRef.get
      adamStateRef.set Std.HashMap.emptyWithCapacity
      adamConfigRef.set none
      optimizerPathRef.set none
      Session.releaseAdam previous
  let adamSchema ← ParamList.checkpointSchema parameters
  let eagerLoss := loss (m := SessionM α)
  let recordLoss (inputs : TensorPack α inputShapes)
      (dataInputs : TensorPack δ dataInputShapes) :
      IO (TensorRef α [] × Pack (TensorRef α) paramShapes) := do
    validateDataInputsIO dataInputs
    session.resetTape
    (do
      let parameterRefs ← ParamList.use (α := α) (ss := paramShapes) parameters
      let inputRefs ← Session.useInputs (α := α) (ss := inputShapes) inputs
      let arguments :=
        Pack.append (ss₁ := paramShapes) (ss₂ := inputShapes) parameterRefs inputRefs
      let lossWithData :=
        CurriedRef.uncurry (ss := paramShapes ++ inputShapes) eagerLoss arguments
      let lossRef ←
        CurriedRef.uncurry (Ref := (Tensor δ ·)) (ss := dataInputShapes) lossWithData dataInputs
      pure (lossRef, parameterRefs)) |>.run session
  -- Scalar results and gradient packs are copied to host tensors before this cleanup. Only the
  -- finished recording is reset: parameters, Adam moments and the random counter survive it.
  let finishRecording : IO Unit := do
    session.resetTape
    if options.usesCUDA then
      LibTorch.Buffer.collectGarbage
  let backwardParameterGradients (lossRef : TensorRef α [])
      (parameterRefs : Pack (TensorRef α) paramShapes) :
      IO (TensorPack α paramShapes) := do
    if session.options.usesCUDA then
      -- The dense sweep gives every tape entry a gradient, frozen and disconnected parameters
      -- included, and only the gradients of the recorded parameter references reach the host pack.
      let gradients ← Session.backwardDenseAllCUDA (α := α) session
        lossRef (Tensor.scalar (1 : α))
      RefList.gradsCUDA (α := α) gradients parameterRefs
    else
      let gradients ← Session.backwardDenseAll (α := α) session
        lossRef (Tensor.scalar (1 : α))
      RefList.grads (α := α) gradients parameterRefs
  let applyNativeGradient (optimizer : NativeOptimizer α)
      (gradients : LibTorch.Tape.SparseGradMap) : IO Unit := do
    useOptimizerPath .native
    match optimizer with
    | .sgd lr =>
      Session.sgdStepCUDA (α := α) session lr gradients
    | .adam lr beta1 beta2 epsilon =>
      Session.adamStepCUDA (α := α) session
        adamConfigRef adamStateRef lr beta1 beta2 epsilon gradients
    | .adamW lr weightDecay beta1 beta2 epsilon =>
      Session.adamWStepCUDA (α := α) session
        adamConfigRef adamStateRef lr weightDecay beta1 beta2 epsilon gradients
  let nativeBatchStep (optimizer : NativeOptimizer α)
      (batch : Array
        (TensorPack α inputShapes × TensorPack δ dataInputShapes))
      (readLoss : Bool) : IO (Option (Tensor α [])) := do
    if batch.isEmpty then
      throw <| IO.userError "torch: native optimizer batch must be nonempty"
    checkOptimizerPath .native
    let accumulator ← IO.mkRef
      (Std.HashMap.emptyWithCapacity : LibTorch.Tape.SparseGradMap)
    let lossAccumulator ← IO.mkRef (none : Option LibTorch.Buffer)
    try
      let mut sampleCount := 0
      for (inputs, dataInputs) in batch do
        sampleCount := sampleCount + 1
        let sampleWeight := 1.0 / sampleCount.toFloat
        let (lossRef, _) ← recordLoss inputs dataInputs
        if readLoss then
          -- A running mean also avoids overflow from rounding a sum of preweighted samples.
          let tape ← session.cudaTape
          let sampleLoss ← IO.ofExcept <|
            LibTorch.Tape.requireValue tape lossRef.id []
          let previous ← lossAccumulator.get
          let mean := match previous with
            | none => LibTorch.Buffer.copy sampleLoss
            | some total =>
                LibTorch.Buffer.meanUpdate total sampleLoss sampleWeight
          lossAccumulator.set (some mean)
          if let some previous := previous then
            Session.releaseCUDABuffer previous
        let gradients ←
          Session.paramGradsCUDA
            (α := α) session lossRef
        Session.withSparseGrads gradients fun gradients =>
          Session.accumulateMeanGradients accumulator gradients sampleWeight
      let gradients ← accumulator.get
      applyNativeGradient optimizer gradients
      match ← lossAccumulator.get with
      | none => pure none
      | some total =>
          some <$> TensorTransfer.ofBuffer (α := α) (s := []) total
    finally
      -- The batch owns these buffers beyond the lifetime of any individual sample tape.
      -- Release them before allocator maintenance, including when a later sample is rejected.
      try
        LibTorch.Tape.releaseSparseGrads (← accumulator.get)
        if let some total ← lossAccumulator.get then
          Session.releaseCUDABuffer total
      finally
        finishRecording
  let lossFn :
      Curried.Function α inputShapes
        (Curried.Function δ dataInputShapes (IO (Tensor α []))) :=
    CurriedRef.curry (Ref := (Tensor α ·)) (ss := inputShapes)
      (β := Curried.Function δ dataInputShapes (IO (Tensor α []))) (fun inputs =>
        CurriedRef.curry (Ref := (Tensor δ ·)) (ss := dataInputShapes)
          (β := IO (Tensor α [])) (fun dataInputs => do
            try
              let (lossRef, _) ← recordLoss inputs dataInputs
              Session.getValue (α := α) session (sh := []) lossRef
            finally
              finishRecording))
  let diff :
      Curried.Function α inputShapes (Curried.Function δ dataInputShapes
        (IO (Tensor α [] × TensorPack α paramShapes))) :=
    CurriedRef.curry (Ref := (Tensor α ·)) (ss := inputShapes)
      (β := Curried.Function δ dataInputShapes
        (IO (Tensor α [] × TensorPack α paramShapes))) (fun inputs =>
        CurriedRef.curry (Ref := (Tensor δ ·)) (ss := dataInputShapes)
          (β := IO (Tensor α [] × TensorPack α paramShapes)) (fun dataInputs => do
            try
              let (lossRef, parameterRefs) ← recordLoss inputs dataInputs
              let lossValue ←
                Session.getValue (α := α) session (sh := []) lossRef
              let parameterGradients ← backwardParameterGradients lossRef parameterRefs
              pure (lossValue, parameterGradients)
            finally
              finishRecording))
  let grad :
      Curried.Function α inputShapes
        (Curried.Function δ dataInputShapes (IO (TensorPack α paramShapes))) :=
    CurriedRef.curry (Ref := (Tensor α ·)) (ss := inputShapes)
      (β := Curried.Function δ dataInputShapes (IO (TensorPack α paramShapes)))
      (fun inputs =>
        CurriedRef.curry (Ref := (Tensor δ ·)) (ss := dataInputShapes)
          (β := IO (TensorPack α paramShapes)) (fun dataInputs => do
            try
              let (lossRef, parameterRefs) ← recordLoss inputs dataInputs
              backwardParameterGradients lossRef parameterRefs
            finally
              finishRecording))
  let optimizerStateCheckpoint? : Option OptimizerStateCheckpoint :=
    if options.usesCUDA then
      some
        { save := fun path => do
            Session.saveAdam
              path adamSchema adamConfigRef adamStateRef (some (← session.rngCounter.get))
          load := fun path => do
            -- Restored moments select the native history too. A failed load leaves the path alone.
            checkOptimizerPath .native
            Session.loadAdam
              path adamSchema adamConfigRef adamStateRef (some session.rngCounter)
                (← TensorTransfer.dtype (α := α))
            useOptimizerPath .native }
    else
      none
  pure
    { state := parameters
      lossFn := lossFn
      diffFn := diff
      gradFn := grad
      nativeBatchStep? := if options.usesCUDA then some nativeBatchStep else none
      optimizerStateCheckpoint? := optimizerStateCheckpoint?
      resetOptimizerState := resetOptimizerState
      useOptimizerPath := useOptimizerPath
      getState := ParamList.valuesSynced (α := α) (ss := paramShapes) parameters }

end TorchLean
