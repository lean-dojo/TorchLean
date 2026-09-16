/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.ScalarTrainer
public import NN.Runtime.Optim.Optimizers

/-!
# Optim

TorchLean optimizer wrappers.

This connects the pure tensor optimizers in `NN/Runtime/Optim/Optimizers.lean` to the runtime
training structures (`Torch.ParamList` + gradient `TorchLean.TensorPack`).

Design notes:
- Optimizer state is stored in a shape-indexed list aligned with the parameter shapes.
- Trainable storage aliases share one history and one update from their summed gradients.
- Updates run on *plain tensors* (not via the autograd tape), so they work the same for eager and
  typed graph training loops.
- Parameters marked `requiresGrad := false` are left unchanged (state is preserved).

### PyTorch references

- `torch.optim` overview: https://pytorch.org/docs/stable/optim.html
- `torch.optim.SGD`: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
- `torch.optim.Adam`: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
- `torch.optim.AdamW`: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

For the *core math* and algorithm-level citations (Adam, AdamW, RMSProp, etc.), see
`NN/Runtime/Optim/Optimizers.lean`.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Optim

/-! ## Generic optimizer interface -/

/-- Optimizer state paired with the loss produced by the same training step. -/
structure StepWithLoss (α : Type) [TorchLean.Storage α] (OptimizerState : Type) where
  /-- Optimizer state to use for the next training step. -/
  optimizerState : OptimizerState
  /-- Scalar loss whose backward pass produced this update. -/
  loss : Tensor α (Shape.ofList [])

set_option genSizeOf false in
/--
A shape-indexed list of optimizer state values.

This mirrors the parameter-shape list used by `Torch.ParamList`. Trainable storage aliases carry
copies of one immutable optimizer state. Initialize through the optimizer and retain the complete
returned list between steps; alias entries must not be edited independently.
-/
inductive StateList
    (State : (α : Type) → [TorchLean.Storage α] → Shape → Type)
    (α : Type) [TorchLean.Storage α] : List Shape → Type where
  | nil : StateList State α []
  | cons {s : Shape} {ss : List Shape} : State α s → StateList State α ss → StateList State α (s ::
    ss)

/--
Runtime-facing optimizer interface.

This is the analogue of a PyTorch `torch.optim.Optimizer`, but made explicit about:
- which parameter shapes it manages (`paramShapes`), and
- how it stores internal state (`State`) aligned with those shapes.
-/
structure Optimizer (α : Type) [TorchLean.Storage α] [Context α] (paramShapes : List Shape) where
  /-- The optimizer's own state type: momentum buffers, second moments, step counts. Leaving it
  abstract here is what lets SGD and Adam share one interface despite storing different things. -/
  State : Type
  /-- Allocate initial state for the given parameters. -/
  init : Torch.ParamList α paramShapes → IO State
  /-- One update: state, parameters and gradients in, new state out. Parameters are updated in
  place through `ParamList`, which is why only the state is returned. -/
  step : State → Torch.ParamList α paramShapes → TorchLean.TensorPack α paramShapes → IO State
  /--
  Optional trainer-native step.

  Most optimizers are implemented by first materializing a gradient `TorchLean.TensorPack` and then
  updating host parameter tensors.  Some trainers can do better.  In eager CUDA mode, for example,
  the trainer can
  keep gradients and optimizer moments on device for SGD/Adam.  When this hook returns `some st'`,
  callers should treat the step as complete and use `st'` as the next optimizer state.  Returning
  `none` asks the caller to fall back to the generic `backward` + `step` path.
  -/
  trainerStep? : {β : Type} → [TorchLean.Storage β] →
    {inputShapes dataInputShapes : List Shape} →
    Torch.ScalarTrainer α β paramShapes inputShapes dataInputShapes → State →
      TorchLean.TensorPack α inputShapes → TorchLean.TensorPack β dataInputShapes →
      IO (Option State) :=
    fun {_β} _ {_inputShapes _dataInputShapes} _tr _st _xs _dataInputs => pure none
  /--
  Optional trainer-native step that returns the loss used for the update.

  This is the logging/inspection counterpart of `trainerStep?`. The returned scalar was evaluated
  on the same tape that produced the gradients; `none` requests the generic
  `lossAndBackward` + `step` path.
  -/
  trainerStepWithLoss? : {β : Type} → [TorchLean.Storage β] →
    {inputShapes dataInputShapes : List Shape} →
    Torch.ScalarTrainer α β paramShapes inputShapes dataInputShapes → State →
      TorchLean.TensorPack α inputShapes → TorchLean.TensorPack β dataInputShapes →
      IO (Option (StepWithLoss α State)) :=
    fun {_β} _ {_inputShapes _dataInputShapes} _tr _st _xs _dataInputs => pure none
  /--
  Optional native mean-gradient update. The outer option reports whether the backend handled
  the update; the inner loss is present exactly when `readLoss` was requested.
  -/
  trainerBatchStep? : {β : Type} → [TorchLean.Storage β] →
    {inputShapes dataInputShapes : List Shape} →
    Torch.ScalarTrainer α β paramShapes inputShapes dataInputShapes → State →
    Array (TorchLean.TensorPack α inputShapes × TorchLean.TensorPack β dataInputShapes) →
    (readLoss : Bool) → IO (Option (State × Option (Tensor α []))) :=
    fun {_β} _ {_inputShapes _dataInputShapes} _tr _st _batch _readLoss => pure none

namespace Internal

/-- An optimizer state with its shape, used to copy a canonical state into later alias slots. -/
structure SomeState
    (State : (α : Type) → [TorchLean.Storage α] → Shape → Type)
    (α : Type) [TorchLean.Storage α] where
  shape : Shape
  state : State α shape

/-- Retrieve an earlier canonical state without comparing state or tensor contents. -/
def canonicalState
    {State : (α : Type) → [TorchLean.Storage α] → Shape → Type}
    {α : Type} [TorchLean.Storage α] (shape : Shape)
    (states : Array (SomeState State α)) (index : Nat) : IO (State α shape) := do
  let some entry := states[index]?
    | throw <| IO.userError "torch: canonical optimizer state missing"
  if h : entry.shape = shape then
    pure (h ▸ entry.state)
  else
    throw <| IO.userError "torch: canonical optimizer state shape mismatch"

/-- Invoke a backend batch update while retaining the wrapper's scheduled optimizer state. -/
def nativeBatchStep {α β State : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (optimizer : Torch.NativeOptimizer α)
    (trainer : Torch.ScalarTrainer α β paramShapes inputShapes dataInputShapes)
    (state : State)
    (batch : Array (TorchLean.TensorPack α inputShapes ×
      TorchLean.TensorPack β dataInputShapes))
    (readLoss : Bool) : IO (Option (State × Option (Tensor α []))) := do
  match trainer.nativeBatchStep? with
  | none => pure none
  | some step => pure (some (state, ← step optimizer batch readLoss))

/-- Read a shape-independent field from the first optimizer state, or use `fallback` when there
are no parameters. -/
def firstStateValue
    {State : (α : Type) → [TorchLean.Storage α] → Shape → Type}
    {α β : Type} [TorchLean.Storage α] (fallback : β)
    (get : {s : Shape} → State α s → β) :
    {ss : List Shape} → StateList State α ss → β
  | [], .nil => fallback
  | _ :: _, .cons st _ => get st

/--
Initialize once per trainable storage and copy that immutable state into its alias slots.

Frozen slots retain independent placeholder states so the public shape-indexed list is unchanged.
All alias decisions use the same canonical-slot map as SGD and checkpoints.
-/
def initStateList {α : Type} [TorchLean.Storage α] [Context α]
    {State : (α : Type) → [TorchLean.Storage α] → Shape → Type}
    {ss : List Shape}
    (initOne : {s : Shape} → Tensor α s → State α s)
    (parameters : Torch.ParamList α ss) : IO (StateList State α ss) := do
  let slots ← Torch.ParamList.canonicalSlotIndices parameters
  let rec buildStates : {shapes : List Shape} → Torch.ParamList α shapes → Nat →
      Array (SomeState State α) → IO (StateList State α shapes)
    | [], .nil, _, _ => pure .nil
    | shape :: _, .cons parameter rest, index, previous => do
        let some slot := slots[index]?
          | throw <| IO.userError "torch: canonical parameter slot missing"
        let state ← match slot with
          | none => pure (initOne (← parameter.value.get))
          | some canonical =>
              if canonical = index then
                pure (initOne (← parameter.value.get))
              else if canonical < index then
                canonicalState shape previous canonical
              else
                throw <| IO.userError "torch: invalid canonical optimizer slot"
        let remaining ← buildStates rest (index + 1) (previous.push ⟨shape, state⟩)
        pure (.cons state remaining)
  buildStates parameters 0 #[]

/--
Run one optimizer update per trainable storage using its summed occurrence gradients.

`updateOne` is called only at canonical slots. Its one immutable returned state is copied into
all aliases with a shape check; frozen states are preserved.

Alias states must remain coherent, as produced by initialization or previous steps for the same
storage layout. Generic `State` has no equality operation: divergent external histories are outside
this contract and are not detected or merged. Checkpoint restoration must preserve this invariant.
-/
def stepStateList {α : Type} [TorchLean.Storage α] [Context α]
    {State : (α : Type) → [TorchLean.Storage α] → Shape → Type}
    {ss : List Shape}
    (updateOne :
      {s : Shape} → State α s → Tensor α s → Tensor α s →
        Optim.Step α s (State α s))
    (parameters : Torch.ParamList α ss) (states : StateList State α ss)
    (gradients : TorchLean.TensorPack α ss) : IO (StateList State α ss) := do
  let slots ← Torch.ParamList.canonicalSlotIndices parameters
  let sums ← Torch.ParamList.canonicalGradients slots gradients
  let rec update : {shapes : List Shape} → Torch.ParamList α shapes →
      StateList State α shapes → Nat → Array (SomeState State α) →
      IO (StateList State α shapes)
    | [], .nil, .nil, _, _ => pure .nil
    | shape :: _, .cons parameter restParameters, .cons state restStates, index, previous => do
        let some slot := slots[index]?
          | throw <| IO.userError "torch: canonical parameter slot missing"
        let nextState ← match slot with
          | none => pure state
          | some canonical =>
              if canonical = index then do
                let gradient ← Torch.ParamList.canonicalGradient shape sums index
                let current ← parameter.value.get
                let result := updateOne state current gradient
                Torch.Internal.setParamHostValue parameter result.parameters
                pure result.optimizerState
              else if canonical < index then
                canonicalState shape previous canonical
              else
                throw <| IO.userError "torch: invalid canonical optimizer slot"
        let rest ← update restParameters restStates (index + 1)
          (previous.push ⟨shape, nextState⟩)
        pure (.cons nextState rest)
  update parameters states 0 #[]

end Internal

/-! ## Concrete optimizers -/

/--
Stochastic gradient descent.

PyTorch analogy: `torch.optim.SGD(lr=lr)` without momentum.
-/
def sgd {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.SGD.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.SGD.State)
        (initOne := fun {s} t =>
          Optim.SGD.init (α := α) (s := s) learningRate t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.SGD.State)
        (updateOne := fun {s} stOne parameters gradients =>
          Optim.SGD.update (α := α) (s := s)
            stOne parameters gradients)
        ps st grads
    trainerStep? := fun {_β} _ {_inputShapes _dataInputShapes} tr st xs dataInputs => do
      let currentLearningRate :=
        Internal.firstStateValue learningRate (fun state => state.learningRate) st
      Torch.ScalarTrainer.runStep tr currentLearningRate xs dataInputs
      pure (some st)
    trainerStepWithLoss? := fun {_β} _ {_inputShapes _dataInputShapes} tr st xs dataInputs => do
      let currentLearningRate :=
        Internal.firstStateValue learningRate (fun state => state.learningRate) st
      let loss ← Torch.ScalarTrainer.runStepWithLoss tr currentLearningRate xs dataInputs
      pure (some { optimizerState := st, loss := loss })
    trainerBatchStep? := fun tr st batch readLoss =>
      Internal.nativeBatchStep
        (.sgd (Internal.firstStateValue learningRate (fun state => state.learningRate) st))
        tr st batch readLoss
  }

/--
SGD with classical momentum.

PyTorch analogy: `torch.optim.SGD(lr=lr, momentum=momentum)`.
-/
def momentumSGD {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate momentum : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.MomentumSGD.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.MomentumSGD.State)
        (initOne := fun {s} t =>
          Optim.MomentumSGD.init
            (α := α) (s := s) learningRate momentum t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.MomentumSGD.State)
        (updateOne := fun {s} stOne params g => Optim.MomentumSGD.update (α := α) (s := s)
          stOne params g)
        ps st grads
  }

/--
AdaGrad (per-parameter learning rate scaling by accumulated squared gradients).

PyTorch analogy: `torch.optim.Adagrad(lr=lr, eps=epsilon)`.
-/
def adagrad {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.AdaGrad.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.AdaGrad.State)
        (initOne := fun {s} t =>
          Optim.AdaGrad.init
            (α := α) (s := s) learningRate epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.AdaGrad.State)
        (updateOne := fun {s} stOne params g => Optim.AdaGrad.update (α := α) (s := s) stOne
          params g)
        ps st grads
  }

/--
RMSProp (exponentially-decayed second moment / running average of squared gradients).

PyTorch analogy: `torch.optim.RMSprop(lr=lr, alpha=decay, eps=epsilon)` (we use the common naming
`decay` for `alpha`).
-/
def rmsprop {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate decay epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.RMSProp.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.RMSProp.State)
        (initOne := fun {s} t =>
          Optim.RMSProp.init
            (α := α) (s := s) learningRate decay epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.RMSProp.State)
        (updateOne := fun {s} stOne params g => Optim.RMSProp.update (α := α) (s := s) stOne
          params g)
        ps st grads
  }

/--
Adam (first/second moment estimates).

PyTorch analogy: `torch.optim.Adam(lr=lr, betas=(beta1,beta2), eps=epsilon)`.
-/
def adam {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate beta1 beta2 epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.Adam.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.Adam.State)
        (initOne := fun {s} t =>
          Optim.Adam.init
            (α := α) (s := s) learningRate beta1 beta2 epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.Adam.State)
        (updateOne := fun {s} stOne params g => Optim.Adam.update (α := α) (s := s) stOne
          params g)
        ps st grads
    trainerStep? := fun {_β} _ {_inputShapes _dataInputShapes} tr st xs dataInputs =>
      match tr.adamStep? with
      | none => pure none
      | some step => do
          let currentLearningRate :=
            Internal.firstStateValue
              learningRate (fun state => state.learningRate) st
          let stepWithData := Torch.Curried.uncurry (α := α) (ss := _inputShapes)
            (β := Torch.Curried.Fn _ _ (IO Unit))
            (step currentLearningRate beta1 beta2 epsilon) xs
          Torch.Curried.uncurry (β := IO Unit) stepWithData dataInputs
          pure (some st)
    trainerStepWithLoss? := fun {_β} _ {_inputShapes _dataInputShapes} tr st xs dataInputs =>
      match tr.adamStepWithLoss? with
      | none => pure none
      | some step => do
          let currentLearningRate :=
            Internal.firstStateValue
              learningRate (fun state => state.learningRate) st
          let stepWithData := Torch.Curried.uncurry (α := α) (ss := _inputShapes)
            (β := Torch.Curried.Fn _ _ (IO (Tensor α (Shape.ofList []))))
            (step currentLearningRate beta1 beta2 epsilon) xs
          let loss ← Torch.Curried.uncurry
            (β := IO (Tensor α (Shape.ofList []))) stepWithData dataInputs
          pure (some { optimizerState := st, loss := loss })
    trainerBatchStep? := fun tr st batch readLoss =>
      Internal.nativeBatchStep
        (.adam (Internal.firstStateValue learningRate (fun state => state.learningRate) st)
          beta1 beta2 epsilon)
        tr st batch readLoss
  }

/--
AdamW (Adam with decoupled weight decay).

PyTorch analogy: `torch.optim.AdamW(lr=lr, weight_decay=weightDecay, betas=(beta1,beta2),
  eps=epsilon)`.
-/
def adamw {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate weightDecay beta1 beta2 epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.AdamW.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.AdamW.State)
        (initOne := fun {s} t =>
          Optim.AdamW.init
            (α := α) (s := s) learningRate weightDecay beta1 beta2 epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.AdamW.State)
        (updateOne := fun {s} stOne params g => Optim.AdamW.update (α := α) (s := s) stOne
          params g)
        ps st grads
    trainerStep? := fun {_β} _ {_inputShapes _dataInputShapes} tr st xs dataInputs =>
      match tr.adamWStep? with
      | none => pure none
      | some step => do
          let currentLearningRate :=
            Internal.firstStateValue
              learningRate (fun state => state.learningRate) st
          let stepWithData := Torch.Curried.uncurry (α := α) (ss := _inputShapes)
            (β := Torch.Curried.Fn _ _ (IO Unit))
            (step currentLearningRate weightDecay beta1 beta2 epsilon) xs
          Torch.Curried.uncurry (β := IO Unit) stepWithData dataInputs
          pure (some st)
    trainerStepWithLoss? := fun {_β} _ {_inputShapes _dataInputShapes} tr st xs dataInputs =>
      match tr.adamWStepWithLoss? with
      | none => pure none
      | some step => do
          let currentLearningRate :=
            Internal.firstStateValue
              learningRate (fun state => state.learningRate) st
          let stepWithData := Torch.Curried.uncurry (α := α) (ss := _inputShapes)
            (β := Torch.Curried.Fn _ _ (IO (Tensor α (Shape.ofList []))))
            (step currentLearningRate weightDecay beta1 beta2 epsilon) xs
          let loss ← Torch.Curried.uncurry
            (β := IO (Tensor α (Shape.ofList []))) stepWithData dataInputs
          pure (some { optimizerState := st, loss := loss })
    trainerBatchStep? := fun tr st batch readLoss =>
      Internal.nativeBatchStep
        (.adamW (Internal.firstStateValue learningRate (fun state => state.learningRate) st)
          weightDecay beta1 beta2 epsilon)
        tr st batch readLoss
  }

/--
AdaDelta (adaptive learning rate method similar to RMSProp but with a running RMS of updates).

PyTorch analogy: `torch.optim.Adadelta(lr=lr, rho=rho, eps=epsilon)`.
-/
def adadelta {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate rho epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.Adadelta.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.Adadelta.State)
        (initOne := fun {s} t =>
          Optim.Adadelta.init
            (α := α) (s := s) learningRate rho epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.Adadelta.State)
        (updateOne := fun {s} stOne params g => Optim.Adadelta.update (α := α) (s := s) stOne
          params g)
        ps st grads
  }

/-! ## Optimizer extension points -/

/--
Projected SGD.

This is the runtime-safe part of a GaLore-style optimizer: every parameter gets a same-shape
projector/lift pair, and the update applies `p ← p - lr * lift(project(g))`.

Full GaLore also needs a rank-changing projector and a refresh schedule. Those pieces require
matrix-specific state and SVD/randomized-SVD infrastructure, so they are not hidden
inside this generic constructor.
-/
def projectedSGD {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate : α)
    (projector : {s : Shape} → Optim.GaLore.Projector α s s :=
      fun {s} => Optim.GaLore.identityProjector (α := α) (s := s))
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList (fun α _ s => Optim.GaLore.SGDState α s s) α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α)
        (State := fun α _ s => Optim.GaLore.SGDState α s s)
        (initOne := fun {s} _t =>
          { learningRate := learningRate, projector := projector (s := s) }) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α)
        (State := fun α _ s => Optim.GaLore.SGDState α s s)
        (updateOne := fun {s} stOne parameters gradients =>
          Optim.GaLore.update
            (α := α) (full := s) (low := s) stOne parameters gradients)
        ps st grads
  }

/--
Muon-style momentum with a caller-supplied same-shape orthogonalization backend.

Using the identity backend gives ordinary momentum-SGD behavior. A production Muon backend should
provide a matrix-specific Newton-Schulz orthogonalizer and optional CUDA kernels.
-/
def muon {α : Type} [TorchLean.Storage α] [Context α]
    (learningRate momentum : α)
    (orthogonalizer : {s : Shape} → Optim.Muon.Orthogonalizer α s :=
      fun {s} => Optim.Muon.identityOrthogonalizer (α := α) (s := s))
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.Muon.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.Muon.State)
        (initOne := fun {s} t =>
          Optim.Muon.init
            (α := α) (s := s) learningRate momentum
            (orthogonalizer (s := s)) t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.Muon.State)
        (updateOne := fun {s} stOne params g =>
          Optim.Muon.update (α := α) (s := s) stOne params g)
        ps st grads
  }

end Optim

end Model
end Autograd
end Runtime
