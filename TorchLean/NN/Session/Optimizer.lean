/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.CUDAAdam
public import TorchLean.NN.Session.ParameterGroups
public import TorchLean.NN.Tape.Backward
public import TorchLean.NN.Session.References

/-!
# Backward Passes and Optimizers

Gradient extraction and optimizer updates for eager sessions, including the CUDA paths that keep
parameter mirrors and moment buffers on device. The CUDA optimizer steps check every gradient, tape
value and Adam moment before they change any parameter. A parameter read at several leaves of one
recording is updated once, from the sum of their cotangents.
-/


@[expose] public section

namespace TorchLean

open TorchLean.Tensor

namespace Session

/--
Run reverse-mode backprop on the CUDA tape, returning device gradients for all tape entries.

The gradients stay on the device. `backwardDenseAll` downloads all of them on a CUDA session, and
the trainer's gradient readback downloads only those of the recorded parameter references.
-/
def backwardDenseAllCUDA {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α)
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array LibTorch.SomeBuffer) := do
  s.validateTensorRef out
  if !s.options.usesCUDA then
    throw <| IO.userError "torch: backwardDenseAllCUDA called on non-CUDA eager session"
  let t ← s.cudaTape
  let seedAny ← TensorTransfer.toSomeBuffer (α := α) (s := sh) seed
  IO.ofExcept <|
    Tape.backwardDenseAll (t := t) (outId := out.id) (seed := seedAny)

/--
Run scalar-loss CUDA backprop and return gradients only for trainable parameter leaves.

`seed` scales the loss cotangent. Native batch training keeps the default `1` and folds each
sample's gradients into a running mean with `accumulateMeanGradients`, so it never forms the larger
unnormalized sum.

The sweep is the engine's `LibTorch.Tape.backwardSparse`. It walks node ids in reverse and runs a
node's VJP only once that node has received a cotangent. Every activation cotangent is retired as
soon as it has been propagated, and owned device buffers are kept only for the ids that `retain`
selects, here the trainable parameter leaves. A trainable leaf the sweep never reaches (unused, or
read only through `detach`) gets the zero of its tape value, `GradZero.zerosLike`, which is the zero
that `Tape.totalize` gives the dense sweeps. The map therefore has one entry per trainable parameter
leaf, and the optimizer update that consumes it stays on device.
-/
def paramGradsCUDA {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α)
    [One α]
    (loss : TensorRef α Shape.scalar) (seed : α := 1) : IO LibTorch.Tape.SparseGradMap := do
  s.validateTensorRef loss
  if !s.options.usesCUDA then
    throw <| IO.userError "torch: paramGradsCUDA called on non-CUDA eager session"
  let t ← s.cudaTape
  let params ← s.paramsByLeaf.get
  let seedAny ← TensorTransfer.toSomeBuffer (α := α) (s := Shape.scalar) (Tensor.scalar seed)
  discard <| IO.ofExcept <| (LibTorch.SomeBuffer.validate seedAny).mapError
    (s!"torch: scalar CUDA backward seed: " ++ ·)
  if loss.id ≥ t.nodes.size then
    releaseCUDASomeBuffer seedAny
    throw <| IO.userError "torch: scalar CUDA backward seed id out of bounds"
  let retain (id : Nat) : Bool :=
    match params.get? id with
    | some p => p.requiresGrad
    | none => false
  let mut grads ← LibTorch.Tape.backwardSparse (t := t) loss.id seedAny retain
  for (id, _p) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    unless grads.contains id do
      match t.getValue? id with
      | some value => grads := grads.insert id (GradZero.zerosLike value)
      | none =>
          LibTorch.Tape.releaseSparseGrads grads
          throw <| IO.userError s!"torch: parameter leaf {id} is not on the CUDA tape"
  pure grads

/--
Fold one sample's borrowed gradients into an owned running mean without downloading them.

`weight` is the reciprocal of the number of samples seen, this one included. The accumulator owns
every buffer inserted into it, and its caller must release it on success and on failure. Entries of
different samples line up because `ScalarTrainer.create` keeps parameter leaf ids stable across
recordings.
-/
def accumulateMeanGradients (accumulator : IO.Ref LibTorch.Tape.SparseGradMap)
    (sample : LibTorch.Tape.SparseGradMap) (weight : Float) : IO Unit := do
  for (id, gradient) in sample.toList do
    let current ← accumulator.get
    let mean ← match current.get? id with
      | none =>
          pure { gradient with buf := LibTorch.Buffer.copy gradient.buf }
      | some old =>
          unless old.s == gradient.s do
            throw <| IO.userError "torch: batch gradient shape mismatch"
          IO.ofExcept <| LibTorch.SomeBuffer.checkDtype old gradient
          discard <| IO.ofExcept (LibTorch.SomeBuffer.validate old)
          discard <| IO.ofExcept (LibTorch.SomeBuffer.validate gradient)
          pure { old with buf := LibTorch.Buffer.meanUpdate old.buf gradient.buf weight }
    accumulator.set (current.insert id mean)
    if let some old := current.get? id then
      releaseCUDASomeBuffer old

/--
Run reverse-mode backprop and return a dense gradient array for all tape entries.

`seed` is the upstream gradient for `out` (like PyTorch's `backward(gradient=...)`).
-/
def backwardDenseAll {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α) [Add α] [Zero α]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (SomeTensor α)) := do
  s.validateTensorRef out
  if s.options.usesCUDA then
    let gradsDev ← backwardDenseAllCUDA (α := α) s (sh := sh) out seed
    gradsDev.mapM (fun g => TensorTransfer.ofSomeBuffer (α := α) g)
  else
    let t ← s.cpuTape
    IO.ofExcept (Tape.backwardDenseAll (t := t) (outId := out.id)
      (seed := SomeTensor.ofTensor seed))

/--
Read the gradient of `x` from a dense gradient array.

`grads` must come from the same recording as `x`. The array holds tensor values without session
identity, so `grad` checks only the index and the shape, and a saved array can be read with its
original references after the session is reset.
-/
def grad {α : Type} [Storage α] {sh : Shape}
  (grads : Array (SomeTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torch: gradient array out of bounds"
  if h : gAny.shape = sh then
    pure (gAny.cast h)
  else
    throw <| IO.userError
      s!"torch: grad shape mismatch (expected {sh}, got {gAny.shape})"

/--
Check the gradient, tape value and optional Adam state of every trainable parameter, so that a
malformed gradient map or checkpoint state is rejected before an optimizer changes any parameter.
-/
def checkOptimizerInputs {α : Type} [Storage α] (operation : String)
    (tape : Tape LibTorch.SomeBuffer) (params : Std.HashMap Nat
        (SomeParam α))
    (grads : LibTorch.Tape.SparseGradMap) (state : Option CUDAAdamState := none) : IO Unit := do
  for (id, param) in params.toList.filter (fun entry => entry.2.requiresGrad) do
    let grad ← match grads.get? id with
      | some grad => pure grad
      | none => throw <| IO.userError s!"torch: gradient map missing parameter during {operation}"
    unless grad.s == param.s do
      throw <| IO.userError s!"torch: gradient shape mismatch during {operation}"
    discard <| IO.ofExcept <| (LibTorch.SomeBuffer.validate grad).mapError
      (s!"torch: {operation} gradient for parameter leaf {id}: " ++ ·)
    let value ← IO.ofExcept <|
      LibTorch.Tape.requireValue (t := tape) (id := id) (s := param.s)
    unless LibTorch.Buffer.dtype grad.buf ==
        LibTorch.Buffer.dtype value do
      throw <| IO.userError s!"torch: gradient dtype mismatch during {operation}"
    discard <| IO.ofExcept <|
      (LibTorch.SomeBuffer.validate { s := param.s, buf := value }).mapError
        (s!"torch: {operation} value for parameter leaf {id}: " ++ ·)
    match state with
    | none => pure ()
    | some states =>
        match states.get? id with
        | none => pure ()
        | some adamState =>
            unless LibTorch.Buffer.dtype adamState.m ==
                LibTorch.Buffer.dtype value &&
                LibTorch.Buffer.dtype adamState.v ==
                LibTorch.Buffer.dtype value do
              throw <| IO.userError s!"torch: CUDA Adam state dtype mismatch during {operation}"
            let expected := LibTorch.Buffer.size value
            if LibTorch.Buffer.size adamState.m != expected ||
                LibTorch.Buffer.size adamState.v != expected then
              throw <| IO.userError
                s!"torch: CUDA Adam state size mismatch for parameter leaf {id}"

/-- Reject a non-finite or negative learning rate before an optimizer mutates device state. -/
def checkCUDALr (operation : String) (lr : Float) : IO Unit := do
  unless lr.isFinite && 0.0 ≤ lr do
    throw <| IO.userError s!"torch: {operation} learning rate must be finite and nonnegative"

/-- Reject Adam state that keeps a separate history for a later leaf of a shared parameter.

A shared parameter has one optimizer history, keyed by its first recorded leaf. Two histories of
one parameter have no justified merge of their moments, so the update fails before it starts and
discards neither. An unshared parameter has a single leaf, so its trainer slot and checkpoint key
are unaffected.
-/
def checkSharedAdamState {α : Type} [Storage α]
    (groups : Array (ParameterGroup α)) (state : CUDAAdamState) : IO Unit := do
  for group in groups do
    for id in group.leaves.toList.drop 1 do
      if state.contains id then
        throw <| IO.userError
          s!"torch: shared parameter leaves {group.id} and {id} have separate Adam histories; \
             initialize fresh optimizer state"

/--
Apply SGD from a sparse CUDA gradient map.

The CUDA trainer's native SGD step calls it. Each parameter is updated once, from the sum of its
leaves' gradients when it is shared, and no zero gradients are allocated for the forward activations
on the tape.
-/
def sgdStepCUDA {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α)
    (lr : α) (grads : LibTorch.Tape.SparseGradMap) : IO Unit := do
  if !s.options.usesCUDA then
    throw <| IO.userError "torch: sgdStepCUDA called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  checkCUDALr "CUDA SGD" lrF
  let t0 ← s.cudaTape
  let params ← s.paramsByLeaf.get
  checkOptimizerInputs "CUDA SGD" t0 params grads
  let groups ← parameterGroups s
  let currentValues ← groups.mapM ParameterGroup.cudaValue
  for (group, currentValue) in groups.zip currentValues do
    let p := group.parameter
    withCUDAGroupGradient group grads fun gAny => do
      if _hs : gAny.s = p.s then
        let updatedDev : LibTorch.SomeBuffer :=
          { s := p.s
            buf := LibTorch.Buffer.axpy currentValue.buf gAny.buf (-lrF) }
        p.setCUDA updatedDev
      else
        throw <| IO.userError "torch: internal grad shape mismatch during CUDA SGD"

/-- Run the Adam update shared by `adamStepCUDA` and `adamWStepCUDA`, once the entry
point has converted and validated its options.

A parameter without state starts from zero moments at step `0`, with bias-correction powers `1`.
Each step multiplies each power by its `beta` once, rounded in the parameter dtype, the recurrence
of `npowRec`. Adam passes `decay = none`, which skips the decay step, and AdamW passes
`some (-(lr * weightDecay))`. State kept for a leaf that is not a session parameter is released
and dropped.
-/
def applyAdamUpdateCUDA {α : Type} [Storage α]
    (s : Session α) (operation : String)
    (configRef : IO.Ref (Option CUDAAdamConfig)) (stateRef : IO.Ref CUDAAdamState)
    (config : CUDAAdamConfig) (lrF : Float) (decay : Option Float)
    (grads : LibTorch.Tape.SparseGradMap) : IO Unit := do
  let beta1F := config.beta1
  let beta2F := config.beta2
  let epsF := config.epsilon
  let oneMinusBeta1 := 1.0 - beta1F
  let oneMinusBeta2 := 1.0 - beta2F
  let t0 ← s.cudaTape
  let params ← s.paramsByLeaf.get
  let mut state ← stateRef.get
  checkOptimizerInputs operation t0 params grads (some state)
  let groups ← parameterGroups s
  checkSharedAdamState groups state
  let currentValues ← groups.mapM ParameterGroup.cudaValue
  bindAdamConfig configRef config
  for (group, currentValue) in groups.zip currentValues do
    let id := group.id
    let p := group.parameter
    let nextState ← withCUDAGroupGradient group grads fun gAny => do
      if _hs : gAny.s = p.s then
        let pBuf := currentValue.buf
        let n := LibTorch.Buffer.size pBuf
        let st ←
          match state.get? id with
          | some st => pure st
          | none =>
              pure { m := ← LibTorch.Buffer.zerosIO n
                       (LibTorch.Buffer.dtype pBuf)
                     v := ← LibTorch.Buffer.zerosIO n
                       (LibTorch.Buffer.dtype pBuf)
                     t := 0, firstPower := 1, secondPower := 1 }
        let t' := st.t + 1
        let firstPower := config.nextPower beta1F st.firstPower
        let secondPower := config.nextPower beta2F st.secondPower
        let firstDenominator := config.correctionDenominator firstPower
        let secondDenominator := config.correctionDenominator secondPower
        let (updated, m', v') := LibTorch.Buffer.adamStep
          pBuf gAny.buf st.m st.v
          beta1F oneMinusBeta1 beta2F oneMinusBeta2
          firstDenominator secondDenominator epsF decay (-lrF)
        let updatedDev : LibTorch.SomeBuffer :=
          { s := p.s, buf := updated }
        p.setCUDA updatedDev
        releaseCUDABuffer st.m
        releaseCUDABuffer st.v
        pure ({ m := m', v := v', t := t', firstPower, secondPower } : CUDAAdamParamState)
      else
        throw <| IO.userError s!"torch: internal grad shape mismatch during {operation}"
    state := state.insert id nextState
    -- Store the state at once: the replaced moments are released, and a later group can throw.
    stateRef.set state
  for (id, st) in state.toList do
    if params.contains id then
      pure ()
    else
      releaseCUDABuffer st.m
      releaseCUDABuffer st.v
      state := state.erase id
  stateRef.set state

/-- Apply Adam using an already-computed sparse CUDA gradient map. -/
def adamStepCUDA {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α)
    (configRef : IO.Ref (Option CUDAAdamConfig)) (stateRef : IO.Ref CUDAAdamState)
    (lr beta1 beta2 epsilon : α)
    (grads : LibTorch.Tape.SparseGradMap) : IO Unit := do
  if !s.options.usesCUDA then
    throw <| IO.userError "torch: adamStepCUDA called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  let beta1F ← TensorTransfer.toFloat (α := α) beta1
  let beta2F ← TensorTransfer.toFloat (α := α) beta2
  let epsF ← TensorTransfer.toFloat (α := α) epsilon
  checkCUDALr "CUDA Adam" lrF
  let config : CUDAAdamConfig :=
    { dtype := ← TensorTransfer.dtype (α := α)
      kind := .adam, beta1 := beta1F, beta2 := beta2F, epsilon := epsF, weightDecay := 0.0 }
  IO.ofExcept config.validate
  applyAdamUpdateCUDA s "CUDA Adam" configRef stateRef config lrF none grads

/--
Apply AdamW from a sparse CUDA gradient map.

Training passes a sparse map, so each activation gradient is released as soon as its contributions
have been propagated.
-/
def adamWStepCUDA {α : Type} [Storage α] [TensorTransfer α]
    (s : Session α)
    (configRef : IO.Ref (Option CUDAAdamConfig)) (stateRef : IO.Ref CUDAAdamState)
    (lr weightDecay beta1 beta2 epsilon : α)
    (grads : LibTorch.Tape.SparseGradMap) : IO Unit := do
  if !s.options.usesCUDA then
    throw <| IO.userError "torch: adamWStepCUDA called on non-CUDA eager session"
  let lrF ← TensorTransfer.toFloat (α := α) lr
  let wdF ← TensorTransfer.toFloat (α := α) weightDecay
  let beta1F ← TensorTransfer.toFloat (α := α) beta1
  let beta2F ← TensorTransfer.toFloat (α := α) beta2
  let epsF ← TensorTransfer.toFloat (α := α) epsilon
  checkCUDALr "CUDA AdamW" lrF
  let config : CUDAAdamConfig :=
    { dtype := ← TensorTransfer.dtype (α := α)
      kind := .adamW, beta1 := beta1F, beta2 := beta2F, epsilon := epsF, weightDecay := wdF }
  IO.ofExcept config.validate
  applyAdamUpdateCUDA s "CUDA AdamW" configRef stateRef config lrF
    (some (-(lrF * wdF))) grads

end Session

end TorchLean
