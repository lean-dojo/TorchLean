/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer.Constructor
public import TorchLean.NN.Trainer.Session
public import TorchLean.NN.RL.DQN.Autograd

/-!
# LibTorch trainer state and validation checks

Evaluation-mode dropout, current parameter reads, view resets, batch averaging and result snapshots
exercise state transitions through the public trainer and runtime session APIs. Adam checks cover
history resets, rejection before mutation or allocation, and checkpoint decoding. These are native
runtime regressions, not consequences of the real-valued optimizer theorems. DQN Huber loss on the
CUDA eager tape has the bits of the specification for both reductions.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace Trainer

open Backend.Device
open Param
open Session

/-- Dropout with probability `1` on one entry: zero in training, the identity in evaluation. -/
def dropoutModel : nn.Builder (nn.Sequential [1] [1]) :=
  nn.dropout (shape := [1]) 1.0

/-- The sample with input `[2]` and target `[0]`. -/
def sample : Sample.Supervised Float [1] [1] :=
  { input := [2.0], target := [0.0] }

/-- Whether `actual` is within `1e-5` of `expected`. -/
def close (actual expected : Float) : Bool :=
  Float.abs (actual - expected) <= 1e-5

/--
Evaluation must use the evaluation-mode CUDA objective without changing subsequent training.

Full-probability dropout makes the distinction exact: training sees zero, while evaluation is the
identity and therefore has mean-squared error four against the zero target.
-/
def checkEvaluationMode : IO Unit := do
  let trainer := Trainer.create dropoutModel
    { device := cuda }
  let session ← trainer.open
  let prediction ← session.predict sample.input
  let evaluationLoss ← session.loss sample
  unless close prediction[0] 2.0 do
    throw <| IO.userError
      s!"CUDA trainer evaluation prediction: got {prediction[0]}, expected 2"
  unless close evaluationLoss 4.0 do
    throw <| IO.userError
      s!"CUDA trainer evaluation loss: got {evaluationLoss}, expected 4"
  let trainingLoss ← session.step sample (loss := true)
  unless close trainingLoss 0.0 do
    throw <| IO.userError
      s!"CUDA trainer training loss after evaluation: got {trainingLoss}, expected 0"
  let repeatedEvaluationLoss ← session.loss sample
  unless close repeatedEvaluationLoss 4.0 do
    throw <| IO.userError
      s!"CUDA trainer repeated evaluation loss: got {repeatedEvaluationLoss}, expected 4"

/-- Reject a shape whose element count wraps to zero in the CUDA size ABI. -/
def checkBufferSizeBoundary : IO Unit := do
  let buffer ← LibTorch.Buffer.zerosIO 0
  let check (value : LibTorch.SomeBuffer) : IO Unit :=
    discard <| IO.ofExcept (LibTorch.SomeBuffer.validate value)
  check { s := [0], buf := buffer }
  let rejected ← try
    check { s := [UInt32.size], buf := buffer }
    pure false
  catch _ => pure true
  unless rejected do
    throw <| IO.userError "CUDA buffer size validation accepted a wrapped element count"

/-- Parameter values remain available through detached views and tape resets, even when frozen. -/
def checkParameterViews : IO Unit := do
  for requiresGrad in #[true, false] do
    let parameter ← Param.create
      ([1.0, 2.0] : Tensor Float [2]) (requiresGrad := requiresGrad)
    let session ← Session.create (α := Float)
      { device := cuda }
    for _ in [0:3] do
      let reference ← session.use parameter
      let detached ← session.detach reference
      let reshaped ← session.apply "reshape"
        (.reshape (s := [2]) (t := [1, 2]) (by decide)) (detached, ())
      let flattened ← session.apply "flatten"
        (.reshape (s := [1, 2]) (t := [2]) (by decide)) (reshaped, ())
      let value ← session.getValue flattened
      unless close value[0] 1.0 && close value[1] 2.0 do
        throw <| IO.userError "CUDA parameter view changed after a tape reset"
      session.resetTape

/--
Repeating one sample in an accumulated batch leaves its mean gradient unchanged. Adam and AdamW
must therefore follow the same parameter trajectory when batch sizes and loss reporting vary.
-/
def checkMixedBatchOptimizer (optimizer : optim.Optimizer) : IO Unit := do
  let trainer := Trainer.create (nn.linear 1 1)
    { device := cuda, optimizer, seed := 37 }
  let reference ← trainer.open
  let mixed ← trainer.open
  for i in [0:8] do
    let x := (i % 3).toFloat - 1.0
    let y := if i % 2 == 0 then 2.0 else -3.0
    let current : Sample.Supervised Float [1] [1] := { input := [x], target := [y] }
    let expectedLoss ← reference.step current (loss := true)
    let batch := Array.replicate (if i % 3 == 0 then 1 else 3) current
    if i % 2 == 0 then
      let actualLoss ← mixed.step (batch := true) (loss := true) batch
      unless Float.abs (actualLoss - expectedLoss) ≤ 1e-4 do
        throw <| IO.userError s!"CUDA mixed batch loss diverged at update {i}"
    else
      mixed.step (batch := true) batch
    for probe in #[-1.0, 0.0, 2.0] do
      let input : Tensor Float [1] := [probe]
      let expected : Tensor Float [1] ← reference.predict input
      let actual : Tensor Float [1] ← mixed.predict input
      unless Float.abs (actual[0] - expected[0]) ≤ 1e-4 do
        throw <| IO.userError
          s!"CUDA mixed batch optimizer diverged at update {i}: \
            got {actual[0]}, expected {expected[0]}"

/-- Keep both the reported loss and parameter update finite at a native format's range limit. -/
def checkLargeBatchMeanAt {α : Type} [Storage α] [Context α]
    [TensorTransfer α] (large rate : α) : IO Unit := do
  let makeTrainer := ScalarTrainer.create
    (α := α) (δ := α) (paramShapes := [[]]) (inputShapes := [])
    (dataInputShapes := []) (options := { device := cuda })
    (loss := fun {m} _ _ parameter =>
      (show m (Ops.Ref m α []) from do
        let value ← Ops.const
          (m := m) (α := α) (Tensor.scalar large)
        let product ← Ops.apply (m := m) .mul (parameter, value, ())
        Ops.apply (m := m) .add (value, product, ())))
  let trainer ← makeTrainer (Tensor.scalar 0)
  let some step := trainer.nativeBatchStep?
    | throw <| IO.userError "CUDA trainer omitted native batch update"
  let some mean ← step (.sgd rate) (Array.replicate 10 (.nil, .nil)) true
    | throw <| IO.userError "CUDA trainer omitted requested batch loss"
  let magnitude ← TensorTransfer.toFloat large
  let actualMean ← TensorTransfer.toFloat mean.item
  unless actualMean.isFinite && Float.abs (actualMean / magnitude - 1.0) < 1e-5 do
    throw <| IO.userError s!"CUDA batch mean overflowed: {actualMean}"
  let .cons parameter .nil ← trainer.getState
  let actualParameter ← TensorTransfer.toFloat parameter.item
  let expectedStep := magnitude * (← TensorTransfer.toFloat rate)
  unless actualParameter.isFinite && Float.abs (actualParameter + expectedStep) < 1e-5 do
    throw <| IO.userError s!"CUDA batch gradient overflowed: {actualParameter}"

/--
A trainable parameter that the sweep does not reach, unused or read only through `detach`, gets the
zero gradient on the native CUDA step, as on the dense sweeps: SGD and Adam leave it bit for bit
unchanged and AdamW only decays it, for one sample and for a batch.
-/
def checkUnreachedParameters : IO Unit := do
  let optimizers : Array (NativeOptimizer Float) :=
    #[.sgd 0.1, .adam 0.1 0.9 0.999 1e-8, .adamW 0.1 0.5 0.9 0.999 1e-8]
  for optimizer in optimizers do
    for samples in #[1, 3] do
      let makeTrainer := ScalarTrainer.create
        (α := Float) (δ := Float) (paramShapes := [[], [], []]) (inputShapes := [])
        (dataInputShapes := []) (options := { device := cuda })
        (loss := fun {m} _ _ used _ detached =>
          (show m (Ops.Ref m Float []) from do
            let square ← Ops.apply (m := m) .mul (used, used, ())
            let stopped ← Ops.apply (m := m) .detach (detached, ())
            Ops.apply (m := m) .add (square, stopped, ())))
      let trainer ← makeTrainer (Tensor.scalar 2.0) (Tensor.scalar 3.0) (Tensor.scalar 5.0)
      let .cons usedGradient (.cons unusedGradient (.cons detachedGradient .nil)) ←
        (trainer.gradFn : IO (TensorPack Float [[], [], []]))
      unless close usedGradient.item 4.0 && unusedGradient.item.toBits == (0.0 : Float).toBits &&
          detachedGradient.item.toBits == (0.0 : Float).toBits do
        throw <| IO.userError "CUDA dense gradients of unreached parameters are not +0"
      let some step := trainer.nativeBatchStep?
        | throw <| IO.userError "CUDA trainer omitted native batch update"
      let _ ← step optimizer (Array.replicate samples (.nil, .nil)) false
      let .cons used (.cons unused (.cons detached .nil)) ← trainer.getState
      unless used.item.isFinite && !close used.item 2.0 do
        throw <| IO.userError s!"CUDA native step did not move the used parameter: {used.item}"
      match optimizer with
      | .adamW .. =>
          unless close unused.item 2.85 && close detached.item 4.75 do
            throw <| IO.userError
              s!"CUDA AdamW did not only decay unreached parameters: {unused.item}, {detached.item}"
      | _ =>
          unless unused.item.toBits == (3.0 : Float).toBits &&
              detached.item.toBits == (5.0 : Float).toBits do
            throw <| IO.userError
              s!"CUDA native step changed unreached parameters: {unused.item}, {detached.item}"
      if let .sgd _ := optimizer then
        unless close used.item 1.6 do
          throw <| IO.userError s!"CUDA SGD step of the used parameter: got {used.item}"

/-- A finite batch mean must not overflow, including after rounding preweighted samples. -/
def checkLargeBatchMean : IO Unit := do
  checkLargeBatchMeanAt (α := Float32) (Float32.ofBits 0x7f7fffff) 1e-38
  checkLargeBatchMeanAt (α := Float) (Float.ofBits 0x7fefffffffffffff) 1e-308

/-- Running statistics survive single-sample and accumulated native updates on one GPU. -/
def checkBatchNormState : IO Unit := do
  let model := nn.batchNorm (batchShape := [1]) (channels := 1) ([2] : Tensor Nat [1])
    (momentum := 0.5)
  let trainer := Trainer.create model { device := cuda }
  let session ← trainer.open
  let sample : Sample.Supervised Float [1, 1, 2] [1, 1, 2] :=
    { input := [[[1, 3]]], target := Tensor.zeros [1, 1, 2] }
  session.step sample
  session.step (batch := true) #[sample, sample]
  let state ← session.state
  for (index, expected) in #[(2, 1.75), (3, 1.875)] do
    if h : index < (nn.Sequential.stateShapes trainer.model).length then
      let values := (state.get ⟨index, h⟩).to (Array Float)
      unless values.size == 1 && (values[0]?.map (fun value => close value expected)).getD false do
        throw <| IO.userError s!"CUDA BatchNorm lost buffer {index}: {values}"
    else
      throw <| IO.userError "CUDA BatchNorm state is missing its running statistics"

/-- A scalar square gives Adam a changing gradient with just one parameter. -/
def quadraticObjective (initial : Float) :
    IO (Module.Objective Float Float [[]] []) := do
  let objective ← Objective.create
    (α := Float) (β := Float) (stateShapes := [[]]) (inputShapes := [])
    (dataInputShapes := []) (runtime := { device := cuda })
    (loss := fun {m} _ _ parameter =>
      Ops.apply (m := m) .mul (parameter, parameter, ()))
    (initState := .cons (Tensor.scalar initial) .nil)
  pure ⟨objective⟩

/-- Starting a fresh optimizer history resets device moments and permits changing update paths. -/
def checkOptimizerReset : IO Unit := do
  let optimizer := Optimizer.adam
    (α := Float) (paramShapes := [[]]) 0.3 0.9 0.999 1e-8
  let objective ← quadraticObjective 1.0
  let initialState ← objective.runtime.initOptimizer optimizer
  let state ← objective.runtime.step optimizer initialState .nil .nil
  let state ← objective.runtime.step optimizer state .nil .nil
  let before := (← objective.runtime.state).get 0
  let gradients ← objective.runtime.grad .nil .nil
  let rejected ← try
    let _ ← objective.update optimizer state gradients
    pure false
  catch _ => pure true
  unless rejected && close ((← objective.runtime.state).get 0).item before.item do
    throw <| IO.userError "CUDA accepted a generic update into native Adam history"

  let fresh ← quadraticObjective before.item
  let freshState ← fresh.runtime.initOptimizer optimizer
  let resetState ← objective.runtime.initOptimizer optimizer
  let _ ← fresh.runtime.step optimizer freshState .nil .nil
  let _ ← objective.runtime.step optimizer resetState .nil .nil
  unless close ((← fresh.runtime.state).get 0).item ((← objective.runtime.state).get 0).item do
    throw <| IO.userError "CUDA optimizer reinitialization retained old Adam moments"

  let genericState ← objective.runtime.initOptimizer optimizer
  let gradients ← objective.runtime.grad .nil .nil
  let genericState ← objective.update optimizer genericState gradients
  let before := (← objective.runtime.state).get 0
  let rejected ← try
    let _ ← objective.runtime.step optimizer genericState .nil .nil
    pure false
  catch _ => pure true
  unless rejected && close ((← objective.runtime.state).get 0).item before.item do
    throw <| IO.userError "CUDA accepted a native update into generic Adam history"
  let resetState ← objective.runtime.initOptimizer optimizer
  let _ ← objective.runtime.step optimizer resetState .nil .nil
  pure ()

/-- Reject epsilon underflow/overflow before a native update changes parameters or moments. -/
def checkAdamEpsilon : IO Unit := do
  -- The smallest positive float32 subnormal is valid: a positive epsilon is rejected only when it
  -- rounds to zero or overflows in float32.
  for (kind, acceptedEpsilon) in (#[
      (.adam, 1e-8), (.adam, 1e-45), (.adamW, 1e-8), (.adamW, 1e-45)] :
      Array (CUDAAdamKind × Float)) do
    let parameter ← Param.create (Tensor.scalar (2.0 : Float32))
    let session ← Session.create (α := Float32) { device := cuda }
    let config ← IO.mkRef (none : Option CUDAAdamConfig)
    let state ← IO.mkRef (Std.HashMap.emptyWithCapacity : CUDAAdamState)
    discard <| session.use parameter
    let zero ← LibTorch.Buffer.zerosIO 1
    let gradients := (Std.HashMap.emptyWithCapacity : LibTorch.Tape.SparseGradMap)
      |>.insert 0 { s := [], buf := zero }
    let step := fun (epsilon : Float) =>
      match kind with
      | .adam => session.adamStepCUDA config state 0.1 0.9 0.999 epsilon.toFloat32 gradients
      | .adamW =>
          session.adamWStepCUDA config state 0.1 0.0 0.9 0.999 epsilon.toFloat32 gradients
    try
      for epsilon in #[1e-50, 1e50] do
        let before ← LibTorch.Buffer.Memory.read
        let rejected ← try
          step epsilon
          pure false
        catch error => pure (error.toString.contains "float32")
        let after ← LibTorch.Buffer.Memory.read
        unless rejected && (← config.get).isNone && (← state.get).size == 0 &&
            after.allocCount == before.allocCount do
          throw <| IO.userError "CUDA Adam accepted epsilon outside the float32 denominator range"
        syncToHost parameter
        unless (← parameter.value.get).item.toBits == (2.0 : Float32).toBits do
          throw <| IO.userError "rejected CUDA Adam epsilon changed the parameter"
      step acceptedEpsilon
      syncToHost parameter
      unless (← parameter.value.get).item.toBits == (2.0 : Float32).toBits do
        throw <| IO.userError "CUDA Adam zero-gradient update changed the parameter"
      let some entry := (← state.get).get? 0
        | throw <| IO.userError "valid CUDA Adam epsilon did not initialize moments"
      unless entry.t == 1 do
        throw <| IO.userError "valid CUDA Adam epsilon produced an incorrect step count"
      for buffer in #[entry.m, entry.v] do
        let values ← LibTorch.Buffer.toFloatArrayIO buffer
        unless values.size == 1 && values.get! 0 == 0.0 do
          throw <| IO.userError "CUDA Adam zero-gradient update produced invalid moments"
    finally
      session.resetTape
      LibTorch.Tape.releaseSparseGrads gradients
      releaseAdam (← state.get)

/-- Checkpoint decoding applies the same effective-epsilon check before allocating replacements. -/
def checkAdamCheckpointEpsilon : IO Unit := do
  IO.FS.withTempDir fun directory => do
    let schema : Checkpoint.Schema :=
      { shapes := #[[]], requiresGrad := #[true] }
    for kind in #[CUDAAdamKind.adam, CUDAAdamKind.adamW] do
      let config ← IO.mkRef (none : Option CUDAAdamConfig)
      let m ← LibTorch.Buffer.fullIO 1 1.25
      let v ← LibTorch.Buffer.fullIO 1 2.5
      let state ← IO.mkRef <| (Std.HashMap.emptyWithCapacity : CUDAAdamState)
        |>.insert 0 { m, v, t := 7, firstPower := (npowRec 7 (0.9 : Float32)).toFloat,
                      secondPower := (npowRec 7 (0.999 : Float32)).toFloat }
      let mBytes ← LibTorch.Buffer.toBytesIO m
      let vBytes ← LibTorch.Buffer.toBytesIO v
      try
        for epsilon in #[1e-50, 1e50, 1e-8] do
          let candidate : CUDAAdamConfig :=
            { kind, beta1 := 0.9, beta2 := 0.999, epsilon, weightDecay := 0.0 }
          let path := directory / "epsilon.bin"
          -- Write the wire format directly so invalid metadata reaches the decoder.
          IO.FS.withFile path .write fun handle => do
            Checkpoint.writeFormat adamCheckpointFormat handle
            Checkpoint.writeNat64 checkpointName handle (dtypeCode .float32)
            writeConfig handle candidate
            Checkpoint.writeNat64 checkpointName handle 0
            schema.write adamCheckpointFormat handle
            schema.writeRepresentatives adamCheckpointFormat handle
            for value in #[1, 0, 7, 1] do
              Checkpoint.writeNat64 checkpointName handle value
            handle.write mBytes
            handle.write vBytes
          let before ← LibTorch.Buffer.Memory.read
          let rejected ← try
            loadAdam path schema config state
            pure false
          catch error =>
            unless error.toString.contains "float32" do throw error
            pure true
          let after ← LibTorch.Buffer.Memory.read
          if epsilon == 1e-8 then
            let some restored := ← config.get
              | throw <| IO.userError "valid CUDA Adam checkpoint omitted its configuration"
            unless !rejected && restored.sameBits candidate do
              throw <| IO.userError "valid CUDA Adam checkpoint epsilon failed to round-trip"
          else
            unless rejected && (← config.get).isNone && after.allocCount == before.allocCount do
              throw <| IO.userError "invalid CUDA Adam checkpoint allocated replacement moments"
          let some entry := (← state.get).get? 0
            | throw <| IO.userError "CUDA Adam checkpoint discarded live moments"
          unless entry.t == 7 &&
              entry.firstPower.toBits == (npowRec 7 (0.9 : Float32)).toFloat.toBits &&
              entry.secondPower.toBits == (npowRec 7 (0.999 : Float32)).toFloat.toBits &&
              (← LibTorch.Buffer.toBytesIO entry.m) == mBytes &&
              (← LibTorch.Buffer.toBytesIO entry.v) == vBytes do
            throw <| IO.userError "CUDA Adam checkpoint changed retained moment values"
      finally
        releaseAdam (← state.get)

/-- A CUDA result owns its parameter snapshot after the live trainer moves on. -/
def checkSnapshot : IO Unit := do
  let trainer := Trainer.create (nn.linear 1 1)
    { device := cuda
      optimizer := optim.adam { lr := 0.1 }, seed := 37 }
  let session ← trainer.open
  let sample : Sample.Supervised Float [1] [1] := { input := [1.0], target := [4.0] }
  let before ← session.loss sample
  session.step sample
  let after ← session.loss sample
  let result ← session.finish { before, after }
  let savedPrediction : Tensor Float [1] ← result.predict sample.input
  for _ in [0:4] do
    session.step (batch := true) #[sample, sample]
  let livePrediction : Tensor Float [1] ← session.predict sample.input
  let repeatedPrediction : Tensor Float [1] ← result.predict sample.input
  unless close savedPrediction[0] repeatedPrediction[0] do
    throw <| IO.userError "CUDA result prediction changed after further training"
  unless Float.abs (livePrediction[0] - savedPrediction[0]) > 0.01 do
    throw <| IO.userError "CUDA snapshot regression did not move the live parameters"
  unless result.report.steps == 1 do
    throw <| IO.userError "CUDA result step count changed after further training"

/-- `RL.DQN.Autograd.huber`, one `Op.huberLoss`, on the CUDA eager tape: the loss
and the gradient have the bits of `Tensor.huberLoss` and `Tensor.huberLossBackward`, both
reductions. -/
def checkDQNHuberAt {α : Type} [Storage α] [Context α] [TensorTransfer α]
    (predictions targets : Tensor α [6]) (delta : α) : IO Unit := do
  let bits := fun (x : Float) => if x.isNaN then none else some x.toBits
  for device in (#[cuda] : Array Backend.Device) do
    for reduction in (#[.mean, .sum] : Array Loss.Reduction) do
      let mean := reduction == .mean
      let makeTrainer := ScalarTrainer.create
        (α := α) (δ := α) (paramShapes := [[6]]) (inputShapes := [])
        (dataInputShapes := []) (options := { device })
        (loss := fun {m} _ _ prediction =>
          (show m (Ops.Ref m α []) from do
            let target ← Ops.const (m := m) (α := α) targets
            RL.DQN.Autograd.huber (m := m) (α := α) prediction target delta reduction))
      let trainer ← makeTrainer predictions
      let (loss, .cons gradient .nil) ←
        (trainer.diffFn : IO (Tensor α [] × TensorPack α [[6]]))
      let toFloat := TensorTransfer.toFloat (α := α)
      let toFloats := fun (x : Tensor α [6]) =>
        (fun (y : Tensor Float [6]) => y.to (Array Float)) <$>
          TensorTransfer.toFloatTensor x
      let label := s!"DQN Huber on {repr device}, mean {mean}"
      unless bits (← toFloat loss.item) ==
          bits (← toFloat (Tensor.huberLoss predictions targets delta mean)) do
        throw <| IO.userError s!"{label}: loss {← toFloat loss.item}"
      let expected ← toFloats (Tensor.huberLossBackward predictions targets delta mean)
      unless ((← toFloats gradient).map bits) == expected.map bits do
        throw <| IO.userError s!"{label}: gradient {← toFloats gradient}, rule {expected}"

/-- The DQN Huber loss in binary32 and binary64 at both branches, the join `|d| = delta` and a
difference of signed zeros. Its terms are dyadic and their sum is exact in any order. -/
def checkDQNHuber : IO Unit := do
  checkDQNHuberAt (α := Float32) [3, -0.5, 0.25, -4, 1.5, 0] [0.5, 0.5, 0, 1, -1, -0.0] 1
  checkDQNHuberAt (α := Float) [3, -0.5, 0.25, -4, 1.5, 0] [0.5, 0.5, 0, 1, -1, -0.0] 1

/-- The parameter words of `steps` updates of `Adam.update` (`weightDecay = none`) or
`AdamW.update` from `start` at a constant `gradient`, `none` at a NaN. -/
def ruleWords {α : Type} [Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    (word : α → Option UInt64) (lr beta1 beta2 epsilon : α) (weightDecay : Option α)
    (start gradient : α) (steps : Nat) : Array (Option UInt64) :=
  let parameters : Tensor α [] := Tensor.scalar start
  let gradients : Tensor α [] := Tensor.scalar gradient
  match weightDecay with
  | none => Id.run do
      let mut state := Adam.init lr beta1 beta2 epsilon parameters
      let mut current := parameters
      let mut out := #[]
      for _ in [0:steps] do
        let step := Adam.update state current gradients
        state := step.optimizerState
        current := step.parameters
        out := out.push (word current.item)
      return out
  | some decay => Id.run do
      let mut state := AdamW.init lr decay beta1 beta2 epsilon parameters
      let mut current := parameters
      let mut out := #[]
      for _ in [0:steps] do
        let step := AdamW.update state current gradients
        state := step.optimizerState
        current := step.parameters
        out := out.push (word current.item)
      return out

/-- The word of a `Float32`, `none` at a NaN. -/
def word32 (x : Float32) : Option UInt64 := if x.isNaN then none else some x.toBits.toUInt64

/-- The word of a `Float`, `none` at a NaN. -/
def word64 (x : Float) : Option UInt64 := if x.isNaN then none else some x.toBits

/-- CUDA Adam (`weightDecay = none`) or AdamW gives `expected`, the words of `ruleWords`, over
`expected.size` steps from `start` at a constant `gradient`. -/
def checkAdamSpecAt {α : Type} [Storage α] [Context α] [TensorTransfer α]
    [DecidableRel ((· > ·) : α → α → Prop)] (word : α → Option UInt64)
    (lr beta1 beta2 epsilon : α) (weightDecay : Option α) (start gradient : α)
    (expected : Array (Option UInt64)) : IO Unit := do
  let rule := ruleWords word lr beta1 beta2 epsilon weightDecay start gradient expected.size
  let parameter ← Param.create (Tensor.scalar start)
  let session ← Session.create (α := α) { device := cuda }
  let config ← IO.mkRef (none : Option CUDAAdamConfig)
  let state ← IO.mkRef (Std.HashMap.emptyWithCapacity : CUDAAdamState)
  try
    for index in [0:expected.size] do
      session.resetTape
      discard <| session.use parameter
      let buffer ← TensorTransfer.toSomeBuffer (Tensor.scalar gradient : Tensor α [])
      let empty : LibTorch.Tape.SparseGradMap := Std.HashMap.emptyWithCapacity
      let gradients := empty.insert 0 buffer
      withSparseGrads gradients fun gradients =>
        match weightDecay with
        | none => session.adamStepCUDA config state lr beta1 beta2 epsilon gradients
        | some decay =>
            session.adamWStepCUDA config state lr decay beta1 beta2 epsilon gradients
      syncToHost parameter
      let device := word (← parameter.value.get).item
      unless device == rule[index]! && rule[index]! == expected[index]! do
        throw <| IO.userError
          s!"CUDA Adam step {index + 1}: device {device}, rule {rule[index]!}, \
            expected {expected[index]!}"
  finally
    session.resetTape
    releaseAdam (← state.get)

/-- CUDA Adam and AdamW at `weightDecay = 0` give the words of `Adam.update` and
`AdamW.update` in `Float32` and `Float`: four steps from `0` at the gradient `1`, and one
step at the gradient `0` that keeps `+∞`, `-∞`, `-0` and `+0` for Adam and gives two NaNs and two
`+0` for AdamW. -/
def checkAdamSpec : IO Unit := do
  let inf32 : Float32 := 1 / 0
  let inf64 : Float := 1 / 0
  for decay in ([none, some 0] : List (Option Float32)) do
    checkAdamSpecAt word32 1 0.9 0.999 1e-8 decay 0 1
      #[some 0xbf800000, some 0xbfffffe4, some 0xc03fffd5, some 0xc07fffc8]
    checkAdamSpecAt word64 1 0.9 0.999 1e-8 (decay.map (·.toFloat)) 0 1
      #[some 0xbfeffffffaa19c49, some 0xbffffffffaa19c29, some 0xc007fffffbf93526,
        some 0xc00ffffffaa19c2d]
  for (start, adam, adamW) in ([(inf32, some 0x7f800000, none), (-inf32, some 0xff800000, none),
      (-0.0, some 0x80000000, some 0), (0, some 0, some 0)] :
      List (Float32 × Option UInt64 × Option UInt64)) do
    checkAdamSpecAt word32 0.1 0.9 0.999 1e-8 none start 0 #[adam]
    checkAdamSpecAt word32 0.1 0.9 0.999 1e-8 (some 0) start 0 #[adamW]
  for (start, adam, adamW) in ([(inf64, some 0x7ff0000000000000, none),
      (-inf64, some 0xfff0000000000000, none), (-0.0, some 0x8000000000000000, some 0),
      (0, some 0, some 0)] : List (Float × Option UInt64 × Option UInt64)) do
    checkAdamSpecAt word64 0.1 0.9 0.999 1e-8 none start 0 #[adam]
    checkAdamSpecAt word64 0.1 0.9 0.999 1e-8 (some 0) start 0 #[adamW]

/-- Run every check of this file. -/
def run : IO Unit := do
  IO.println "=== CUDA trainer coverage ==="
  checkEvaluationMode
  checkBufferSizeBoundary
  checkParameterViews
  checkUnreachedParameters
  checkMixedBatchOptimizer (optim.adam { lr := 0.03 })
  checkMixedBatchOptimizer (optim.adamW { lr := 0.03, weightDecay := 0.1 })
  checkLargeBatchMean
  checkBatchNormState
  checkOptimizerReset
  checkAdamEpsilon
  checkAdamCheckpointEpsilon
  checkAdamSpec
  checkSnapshot
  checkDQNHuber

end Trainer
end CUDA
end TorchLean.Test
