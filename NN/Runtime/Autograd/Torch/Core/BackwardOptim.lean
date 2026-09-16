/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.OptimizerCheckpoint.CudaAdam
public import NN.Runtime.Autograd.Torch.Core.ParameterGroups
public import NN.Runtime.Autograd.Engine.Core.Backward

/-!
# Backward Passes and Optimizers

Gradient extraction and optimizer updates for eager sessions, including the CUDA paths that keep
parameter mirrors and moment buffers on device.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean TorchLean.Tensor

namespace Internal

namespace EagerSession

/--
Run reverse-mode backprop on the CUDA tape, returning device gradients for all tape entries.

This is the CUDA analogue of `backwardDenseAll`, but it does *not* download gradients back to the
host. This is primarily useful for implementing GPU-native optimizer steps.
-/
def backwardDenseAllCuda {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (s : EagerSession α) [Add α] [Zero α]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array Runtime.Autograd.Cuda.AnyBuffer) := do
  if Config.device s.options != .cuda then
    throw <| IO.userError "torch: backwardDenseAllCuda called on non-CUDA eager session"
  let t ← s.cudaTape.get
  let seedAny ← CudaBridge.toAnyBuffer (α := α) (s := sh) seed
  okOrThrow <|
    Runtime.Autograd.Cuda.Tape.backwardDenseAll (t := t) (outId := out.id) (seed := seedAny)

/--
Run scalar-loss CUDA backprop and return gradients only for trainable parameter leaves.

`seed` scales the loss cotangent. Batch training uses `1 / batch.size` so it can accumulate the
mean directly, without first forming a larger unnormalized sum.

The traversal is the engine's sparse CUDA sweep `Cuda.Tape.backwardSparse`, which walks node
ids in reverse, runs a node's VJP only when that node has received a cotangent, retires every
activation cotangent as soon as it has been propagated, and keeps owned device buffers only for
the ids selected by `retain`. Here `retain` selects exactly the trainable parameter leaves, so the
returned map has one entry per trainable parameter and optimizer updates stay on device without
paying hash-table costs at every intermediate node. A trainable parameter that receives no
gradient is an error, as before.
-/
def backwardScalarParamGradsCuda {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (s : EagerSession α)
    [One α]
    (loss : TensorRef α Shape.scalar) (seed : α := 1) : IO CudaGradMap := do
  if Config.device s.options != .cuda then
    throw <| IO.userError "torch: backwardScalarParamGradsCuda called on non-CUDA eager session"
  let t ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let seedAny ← CudaBridge.toAnyBuffer (α := α) (s := Shape.scalar) (Tensor.scalar seed)
  checkCudaAnyBufferSize "scalar CUDA backward seed" seedAny
  if loss.id ≥ t.nodes.size then
    releaseCudaAnyBuffer seedAny
    throw <| IO.userError "torch: scalar CUDA backward seed id out of bounds"
  let retain (id : Nat) : Bool :=
    match params.get? id with
    | some p => p.requiresGrad
    | none => false
  let grads ← Runtime.Autograd.Cuda.Tape.backwardSparse (t := t) loss.id seedAny retain
  for (id, _p) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    unless grads.contains id do
      Runtime.Autograd.Cuda.Tape.releaseSparseGrads grads
      throw <| IO.userError s!"torch: missing CUDA gradient for parameter leaf {id}"
  pure grads

/--
Add borrowed sample gradients into an owned accumulator without downloading them.

The accumulator owns every inserted buffer. Its caller must release it on success and failure.
Parameter leaf ids are stable across recordings by `scalarTrainer`.
-/
def accumulateCudaGradMap (accumulator : IO.Ref CudaGradMap)
    (sample : CudaGradMap) : IO Unit := do
  for (id, gradient) in sample.toList do
    let current ← accumulator.get
    let summed ← match current.get? id with
      | none =>
          pure { gradient with buf := Runtime.Autograd.Cuda.Buffer.copy gradient.buf }
      | some old =>
          okOrThrow (Runtime.Autograd.Cuda.AnyBuffer.add old gradient)
    accumulator.set (current.insert id summed)
    if let some old := current.get? id then
      releaseCudaAnyBuffer old

/--
Run reverse-mode backprop and return a dense gradient array for all tape entries.

`seed` is the upstream gradient for `out` (like PyTorch's `backward(gradient=...)`).
-/
def backwardDenseAll {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (s : EagerSession α) [Add α] [Zero α]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Spec.SomeTensor α)) := do
  if Config.device s.options == .cuda then
    let gradsDev ← backwardDenseAllCuda (α := α) s (sh := sh) out seed
    gradsDev.mapM (fun g => CudaBridge.ofAnyBuffer (α := α) g)
  else
    let t ← s.tape.get
    okOrThrow (Runtime.Autograd.Tape.backwardDenseAll (t := t) (outId := out.id)
      (seed := Spec.SomeTensor.ofTensor seed))

/--
Run backward from a scalar loss with seed `1`.

PyTorch comparison: `loss.backward()` for a scalar loss.
-/
def backwardScalarDenseAll {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (s : EagerSession α) [Add α]
  [Zero α] [One α]
  (loss : TensorRef α Shape.scalar) : IO (Array (Spec.SomeTensor α)) := do
  backwardDenseAll (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Extract the gradient for a particular `TensorRef` from a dense gradient array.
-/
def grad {α : Type} [TorchLean.Storage α] {sh : Shape}
  (grads : Array (Spec.SomeTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torch: gradient array out of bounds"
  if h : gAny.shape = sh then
    pure (gAny.cast h)
  else
    throw <| IO.userError
      s!"torch: grad shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty gAny.shape})"

/--
Apply an SGD update to all parameters recorded via `use`.

PyTorch comparison: `for p in params: p.data -= lr * p.grad`.
-/
def sgdStepAll {α : Type} [TorchLean.Storage α] [TensorTransfer α] (s : EagerSession α)
  [Sub α] [Mul α] [Add α] [Zero α]
  (lr : α) (grads : Array (Spec.SomeTensor α)) : IO Unit := do
  let groups ← parameterGroups s
  -- Validate and combine all leaf cotangents before changing any parameter.
  let prepared ← groups.mapM fun group => do
    pure (group, ← denseGroupGradient group grads)
  if Config.device s.options == .cuda then
    let lrF ← TensorTransfer.toFloat (α := α) lr
    let t0 ← s.cudaTape.get
    for group in groups do
      for id in group.leaves do
        let value ← okOrThrow <|
          Runtime.Autograd.Cuda.Tape.requireValue (t := t0) (id := id) (s := group.parameter.s)
        checkCudaAnyBufferSize "SGD parameter value" { s := group.parameter.s, buf := value }
    let currentValues ← groups.mapM ParameterGroup.currentCudaValue
    for ((group, gAny), currentValue) in prepared.zip currentValues do
      let p := group.parameter
      if hs : gAny.shape = p.s then
        let gT : Tensor α p.s := gAny.cast hs
        let gDev ← CudaBridge.toAnyBuffer (α := α) (s := p.s) gT
        try
          let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
            { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy currentValue.buf gDev.buf (-lrF) }
          p.setCuda updatedDev
        finally
          releaseCudaAnyBuffer gDev
      else
        throw <| IO.userError "torch: internal grad shape mismatch during SGD"
  else
    for (group, gAny) in prepared do
      let p := group.parameter
      if hs : gAny.shape = p.s then
        let pv ← group.currentHostValue
        if hp : pv.shape = p.s then
          let pvT : Tensor α p.s := pv.cast hp
          let gT : Tensor α p.s := gAny.cast hs
          let updated : Tensor α p.s :=
            subSpec pvT (scaleSpec (α := α) (s := p.s) gT lr)
          p.set (Spec.SomeTensor.ofTensor updated)
        else
          throw <| IO.userError "torch: internal param shape mismatch"
      else
        throw <| IO.userError "torch: internal grad shape mismatch during SGD"

/--
Check every trainable parameter, gradient, tape value, and optional Adam state before an optimizer
changes a parameter. This prevents a malformed gradient map or checkpoint state from producing a
partially applied update.
-/
def checkCudaOptimizerInputs {α : Type} [TorchLean.Storage α] (operation : String)
    (tape : Runtime.Autograd.Cuda.Tape) (params : Std.HashMap Nat (AnyParam α))
    (grads : CudaGradMap) (state : Option CudaAdamState := none) : IO Unit := do
  for (id, param) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    let grad ← match grads.get? id with
      | some grad => pure grad
      | none => throw <| IO.userError s!"torch: gradient map missing parameter during {operation}"
    unless grad.s == param.s do
      throw <| IO.userError s!"torch: gradient shape mismatch during {operation}"
    checkCudaAnyBufferSize s!"{operation} gradient for parameter leaf {id}" grad
    let value ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.requireValue (t := tape) (id := id) (s := param.s)
    checkCudaAnyBufferSize s!"{operation} value for parameter leaf {id}"
      { s := param.s, buf := value }
    match state with
    | none => pure ()
    | some states =>
        match states.get? id with
        | none => pure ()
        | some adamState =>
            let expected := Runtime.Autograd.Cuda.Buffer.size value
            if Runtime.Autograd.Cuda.Buffer.size adamState.m != expected ||
                Runtime.Autograd.Cuda.Buffer.size adamState.v != expected then
              throw <| IO.userError
                s!"torch: CUDA Adam state size mismatch for parameter leaf {id}"

/-- Reject a non-finite or negative learning rate before an optimizer mutates device state. -/
def checkCudaLearningRate (operation : String) (learningRate : Float) : IO Unit := do
  unless learningRate.isFinite && 0.0 ≤ learningRate do
    throw <| IO.userError s!"torch: {operation} learning rate must be finite and nonnegative"

/-- A shared parameter has one optimizer history, keyed by its first recorded leaf.

If an existing state contains separate histories for later aliases, there is no justified way to
merge their moments. Reject that state before updating instead of silently discarding a history.
Ordinary trainer slots and their checkpoint keys are unchanged.
-/
def checkSharedCudaAdamState {α : Type} [Storage α]
    (groups : Array (ParameterGroup α)) (state : CudaAdamState) : IO Unit := do
  for group in groups do
    for id in group.leaves.toList.drop 1 do
      if state.contains id then
        throw <| IO.userError
          s!"torch: shared parameter leaves {group.id} and {id} have separate Adam histories; \
             initialize fresh optimizer state"

/--
Apply SGD from a sparse CUDA gradient map.

This is the path used by the CUDA trainer.  It updates only parameter leaves and avoids allocating
zero gradients for every forward activation in the tape.
-/
def sgdStepAllCudaMap {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (s : EagerSession α)
    (lr : α) (grads : CudaGradMap) : IO Unit := do
  if Config.device s.options != .cuda then
    throw <| IO.userError "torch: sgdStepAllCudaMap called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  checkCudaLearningRate "CUDA SGD" lrF
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  checkCudaOptimizerInputs "CUDA SGD" t0 params grads
  let groups ← parameterGroups s
  let currentValues ← groups.mapM ParameterGroup.currentCudaValue
  for (group, currentValue) in groups.zip currentValues do
    let p := group.parameter
    withCudaGroupGradient group grads fun gAny => do
      if _hs : gAny.s = p.s then
        let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
          { s := p.s, buf := Runtime.Autograd.Cuda.Buffer.axpy currentValue.buf gAny.buf (-lrF) }
        p.setCuda updatedDev
      else
        throw <| IO.userError "torch: internal grad shape mismatch during CUDA SGD"

/-- Apply Adam using an already-computed sparse CUDA gradient map. -/
def adamStepAllCudaMap {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (s : EagerSession α)
    (configRef : IO.Ref (Option CudaAdamConfig)) (stateRef : IO.Ref CudaAdamState)
    (lr beta1 beta2 epsilon : α)
    (grads : CudaGradMap) : IO Unit := do
  if Config.device s.options != .cuda then
    throw <| IO.userError "torch: adamStepAllCudaMap called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  let beta1F ← TensorTransfer.toFloat (α := α) beta1
  let beta2F ← TensorTransfer.toFloat (α := α) beta2
  let epsF ← TensorTransfer.toFloat (α := α) epsilon
  checkCudaLearningRate "CUDA Adam" lrF
  let config : CudaAdamConfig :=
    { kind := .adam, beta1 := beta1F, beta2 := beta2F, epsilon := epsF, weightDecay := 0.0 }
  match config.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  checkCudaOptimizerInputs "CUDA Adam" t0 params grads (some state)
  let groups ← parameterGroups s
  checkSharedCudaAdamState groups state
  let currentValues ← groups.mapM ParameterGroup.currentCudaValue
  ensureCudaAdamConfig configRef config
  for (group, currentValue) in groups.zip currentValues do
    let id := group.id
    let p := group.parameter
    let nextState ← withCudaGroupGradient group grads fun gAny => do
      if _hs : gAny.s = p.s then
        let pBuf := currentValue.buf
        let n := Runtime.Autograd.Cuda.Buffer.size pBuf
        let st :=
          match state.get? id with
          | some st => st
          | none =>
              { m := Runtime.Autograd.Cuda.Buffer.zeros n
                v := Runtime.Autograd.Cuda.Buffer.zeros n
                t := 0 }
        let t' := st.t + 1
        let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
        let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
        let (updated, m', v') := Runtime.Autograd.Cuda.Buffer.adamStep
          pBuf gAny.buf st.m st.v
          beta1F oneMinusBeta1 beta2F oneMinusBeta2
          mHatScale vHatScale epsF 0.0 (-lrF)
        let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
          { s := p.s, buf := updated }
        p.setCuda updatedDev
        releaseCudaBuffer st.m
        releaseCudaBuffer st.v
        pure ({ m := m', v := v', t := t' } : CudaAdamParamState)
      else
        throw <| IO.userError "torch: internal grad shape mismatch during CUDA Adam"
    state := state.insert id nextState
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

/--
Apply AdamW from a sparse CUDA gradient map.

Normal training uses this sparse map so activation gradients can be released as soon as their
contributions have been propagated.
-/
def adamWStepAllCudaMap {α : Type} [TorchLean.Storage α] [TensorTransfer α]
    (s : EagerSession α)
    (configRef : IO.Ref (Option CudaAdamConfig)) (stateRef : IO.Ref CudaAdamState)
    (lr weightDecay beta1 beta2 epsilon : α)
    (grads : CudaGradMap) : IO Unit := do
  if Config.device s.options != .cuda then
    throw <| IO.userError "torch: adamWStepAllCudaMap called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  let wdF ← TensorTransfer.toFloat (α := α) weightDecay
  let beta1F ← TensorTransfer.toFloat (α := α) beta1
  let beta2F ← TensorTransfer.toFloat (α := α) beta2
  let epsF ← TensorTransfer.toFloat (α := α) epsilon
  checkCudaLearningRate "CUDA AdamW" lrF
  let config : CudaAdamConfig :=
    { kind := .adamW, beta1 := beta1F, beta2 := beta2F, epsilon := epsF, weightDecay := wdF }
  match config.validate with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape.get
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  checkCudaOptimizerInputs "CUDA AdamW" t0 params grads (some state)
  let groups ← parameterGroups s
  checkSharedCudaAdamState groups state
  let currentValues ← groups.mapM ParameterGroup.currentCudaValue
  ensureCudaAdamConfig configRef config
  for (group, currentValue) in groups.zip currentValues do
    let id := group.id
    let p := group.parameter
    let nextState ← withCudaGroupGradient group grads fun gAny => do
      if _hs : gAny.s = p.s then
        let pBuf := currentValue.buf
        let n := Runtime.Autograd.Cuda.Buffer.size pBuf
        let st :=
          match state.get? id with
          | some st => st
          | none =>
              { m := Runtime.Autograd.Cuda.Buffer.zeros n
                v := Runtime.Autograd.Cuda.Buffer.zeros n
                t := 0 }
        let t' := st.t + 1
        let mHatScale := 1.0 / (1.0 - Float.pow beta1F (Float.ofNat t'))
        let vHatScale := 1.0 / (1.0 - Float.pow beta2F (Float.ofNat t'))
        let (updated, m', v') := Runtime.Autograd.Cuda.Buffer.adamStep
          pBuf gAny.buf st.m st.v
          beta1F oneMinusBeta1 beta2F oneMinusBeta2
          mHatScale vHatScale epsF (-(lrF * wdF)) (-lrF)
        let updatedDev : Runtime.Autograd.Cuda.AnyBuffer :=
          { s := p.s, buf := updated }
        p.setCuda updatedDev
        releaseCudaBuffer st.m
        releaseCudaBuffer st.v
        pure ({ m := m', v := v', t := t' } : CudaAdamParamState)
      else
        throw <| IO.userError "torch: internal grad shape mismatch during CUDA AdamW"
    state := state.insert id nextState
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCudaBuffer st.m
      releaseCudaBuffer st.v
      state := state.erase id
  stateRef.set state

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
