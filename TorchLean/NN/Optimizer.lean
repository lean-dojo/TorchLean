/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.ScalarTrainer.Loss
public import TorchLean.NN.Optimizer.Update

/-!
# Optimizer

TorchLean optimizer wrappers.

They connect the pure tensor optimizers of `TorchLean/NN/Optimizer/Update.lean` to the runtime
training structures (`ParamList` with a gradient `TorchLean.TensorPack`). That file also
holds the core math and the algorithm-level citations (Adam, AdamW, RMSProp and others).

Design notes:
- Optimizer state is stored in a shape-indexed list aligned with the parameter shapes.
- Trainable storage aliases share one history and one update from their summed gradients.
- Updates run on *plain tensors* after the backward pass, so they are not recorded on the
  autograd tape.
- Parameters marked `requiresGrad := false` are left unchanged (state is preserved).

### PyTorch references

- `torch.optim` overview: https://pytorch.org/docs/stable/optim.html
- `torch.optim.SGD`: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
- `torch.optim.Adam`: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
- `torch.optim.AdamW`: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
-/

@[expose] public section


namespace TorchLean
namespace Optimizer

open TorchLean.Tensor

/-! ## Generic optimizer interface -/

/--
Runtime-facing optimizer interface.

It is the analogue of a PyTorch `torch.optim.Optimizer` that also declares:
- which parameter shapes it manages (`paramShapes`), and
- how it stores internal state (`State`) aligned with those shapes.
-/
structure _root_.TorchLean.Optimizer (α : Type) [Storage α] [Context α]
    (paramShapes : List Shape) where
  /-- The optimizer's own state type: momentum buffers, second moments, step counts. Keeping it
  abstract lets SGD and Adam share one interface while storing different things. -/
  State : Type
  /-- Allocate initial state for the given parameters. -/
  init : ParamList α paramShapes → IO State
  /-- One update, taking the state, parameters and gradients and returning the next state.
  Parameters are updated in place through `ParamList`, so only the state is returned. -/
  step : State → ParamList α paramShapes → TensorPack α paramShapes → IO State
  /-- The same state with every parameter's learning rate set to the given value. Schedules call
  it between steps. -/
  setLr : α → State → State
  /--
  The update this state names among the trainer's native optimizers, if any. A trainer whose
  backend keeps parameters, gradients and optimizer moments on the device (eager CUDA) runs it in
  place of the generic `backward` + `step` path, and the state is then kept as is, since the moments
  live on the device. `none` always takes the generic path.
  -/
  native? : State → Option (NativeOptimizer α) := fun _ => none

/--
Run `optimizer`'s native update on a nonempty batch when both the optimizer and the trainer's
backend have one, where a single sample is a batch of one. The outer option reports whether the
native path ran, and the inner loss is present exactly when `readLoss` is true.
-/
def nativeBatchStep? {α β : Type} [Storage α] [Storage β] [Context α]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (optimizer : Optimizer α paramShapes)
    (trainer : ScalarTrainer α β paramShapes inputShapes dataInputShapes)
    (state : optimizer.State)
    (batch : Array (TensorPack α inputShapes ×
      TensorPack β dataInputShapes))
    (readLoss : Bool) : IO (Option (Option (Tensor α []))) :=
  match optimizer.native? state, trainer.nativeBatchStep? with
  | some native, some step => some <$> step native batch readLoss
  | _, _ => pure none

/-- An optimizer state with its shape, used to copy a canonical state into later alias slots. -/
structure SomeState
    (State : (α : Type) → [Storage α] → Shape → Type)
    (α : Type) [Storage α] where
  /-- Shape of the parameter that owns the state. -/
  shape : Shape
  /-- The optimizer state. -/
  state : State α shape

/-- Retrieve an earlier canonical state without comparing state or tensor contents. -/
def canonicalState
    {State : (α : Type) → [Storage α] → Shape → Type}
    {α : Type} [Storage α] (shape : Shape)
    (states : Array (SomeState State α)) (index : Nat) : IO (State α shape) := do
  let some entry := states[index]?
    | throw <| IO.userError "torch: canonical optimizer state missing"
  if h : entry.shape = shape then
    pure (h ▸ entry.state)
  else
    throw <| IO.userError "torch: canonical optimizer state shape mismatch"

/-- Read a shape-independent field from the first optimizer state, or use `fallback` when there
are no parameters. -/
def firstStateValue {F : Shape → Type} {β : Type} (fallback : β)
    (get : {s : Shape} → F s → β) : {ss : List Shape} → Pack F ss → β
  | [], .nil => fallback
  | _ :: _, .cons st _ => get st

/--
Initialize once per trainable storage and copy that immutable state into its alias slots.

Frozen slots get independent placeholder states, so the public shape-indexed list keeps one entry
per parameter. All alias decisions use the same canonical-slot map as SGD and checkpoints.
-/
def initStateList {α : Type} [Storage α] [Context α]
    {State : (α : Type) → [Storage α] → Shape → Type}
    {ss : List Shape}
    (initOne : {s : Shape} → Tensor α s → State α s)
    (parameters : ParamList α ss) : IO (Pack (State α) ss) := do
  let slots ← ParamList.canonicalSlotIndices parameters
  let rec
    /-- Build the states of the remaining parameters, given the slot index and the states of the
    earlier slots. -/
    buildStates : {shapes : List Shape} → ParamList α shapes → Nat →
        Array (SomeState State α) → IO (Pack (State α) shapes)
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

`updateOne` is called only at canonical slots. The one immutable state it returns is copied into
every alias with a shape check, and frozen states are preserved.

Alias states must be coherent, as initialization or previous steps for the same storage layout
produce them. Generic `State` has no equality operation, so divergent external histories are
outside this contract and are neither detected nor merged. Checkpoint restoration must preserve
this invariant.
-/
def stepStateList {α : Type} [Storage α] [Context α]
    {State : (α : Type) → [Storage α] → Shape → Type}
    {ss : List Shape}
    (updateOne :
      {s : Shape} → State α s → Tensor α s → Tensor α s →
        OptimizerStep α s (State α s))
    (parameters : ParamList α ss) (states : Pack (State α) ss)
    (gradients : TensorPack α ss) : IO (Pack (State α) ss) := do
  let slots ← ParamList.canonicalSlotIndices parameters
  let sums ← ParamList.canonicalGradients slots gradients
  let rec
    /-- Update the remaining parameters and their states, given the slot index and the next
    states of the earlier slots. -/
    update : {shapes : List Shape} → ParamList α shapes →
        Pack (State α) shapes → Nat → Array (SomeState State α) →
        IO (Pack (State α) shapes)
      | [], .nil, .nil, _, _ => pure .nil
      | shape :: _, .cons parameter restParameters, .cons state restStates, index, previous => do
          let some slot := slots[index]?
            | throw <| IO.userError "torch: canonical parameter slot missing"
          let nextState ← match slot with
            | none => pure state
            | some canonical =>
                if canonical = index then do
                  let gradient ← ParamList.canonicalGradient shape sums index
                  let current ← parameter.value.get
                  let result := updateOne state current gradient
                  Param.setHostValue parameter result.parameters
                  pure result.optimizerState
                else if canonical < index then
                  canonicalState shape previous canonical
                else
                  throw <| IO.userError "torch: invalid canonical optimizer slot"
          let rest ← update restParameters restStates (index + 1)
            (previous.push ⟨shape, nextState⟩)
          pure (.cons nextState rest)
  update parameters states 0 #[]

/-! ## Concrete optimizers -/

/--
Stochastic gradient descent.

PyTorch analogy: `torch.optim.SGD(lr=lr)` without momentum.
-/
def sgd {α : Type} [Storage α] [Context α]
    (lr : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := Pack (SGD.State α) paramShapes
    init := fun ps =>
      initStateList (α := α) (State := SGD.State)
        (initOne := fun {s} t =>
          SGD.init (α := α) (s := s) lr t) ps
    step := fun st ps grads =>
      stepStateList (α := α) (State := SGD.State)
        (updateOne := fun {s} stOne parameters gradients =>
          SGD.update (α := α) (s := s)
            stOne parameters gradients)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
    native? := fun st =>
      some (.sgd (firstStateValue lr (fun state => state.lr) st))
  }

/--
SGD with classical momentum.

PyTorch analogy: `torch.optim.SGD(lr=lr, momentum=momentum)`.
-/
def momentumSGD {α : Type} [Storage α] [Context α]
    (lr momentum : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := Pack (MomentumSGD.State α) paramShapes
    init := fun ps =>
      initStateList (α := α) (State := MomentumSGD.State)
        (initOne := fun {s} t =>
          MomentumSGD.init
            (α := α) (s := s) lr momentum t) ps
    step := fun st ps grads =>
      stepStateList (α := α) (State := MomentumSGD.State)
        (updateOne := fun {s} stOne params g => MomentumSGD.update (α := α) (s := s)
          stOne params g)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
  }

/--
AdaGrad (per-parameter learning rate scaling by accumulated squared gradients).

PyTorch analogy: `torch.optim.Adagrad(lr=lr, eps=epsilon)`.
-/
def adagrad {α : Type} [Storage α] [Context α]
    (lr epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := Pack (AdaGrad.State α) paramShapes
    init := fun ps =>
      initStateList (α := α) (State := AdaGrad.State)
        (initOne := fun {s} t =>
          AdaGrad.init
            (α := α) (s := s) lr epsilon t) ps
    step := fun st ps grads =>
      stepStateList (α := α) (State := AdaGrad.State)
        (updateOne := fun {s} stOne params g => AdaGrad.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
  }

/--
RMSProp (exponentially-decayed second moment / running average of squared gradients).

PyTorch analogy: `torch.optim.RMSprop(lr=lr, alpha=decay, eps=epsilon)` with the common name `decay`
for PyTorch's `alpha`.
-/
def rmsprop {α : Type} [Storage α] [Context α]
    (lr decay epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := Pack (RMSProp.State α) paramShapes
    init := fun ps =>
      initStateList (α := α) (State := RMSProp.State)
        (initOne := fun {s} t =>
          RMSProp.init
            (α := α) (s := s) lr decay epsilon t) ps
    step := fun st ps grads =>
      stepStateList (α := α) (State := RMSProp.State)
        (updateOne := fun {s} stOne params g => RMSProp.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
  }

/--
Adam (first/second moment estimates).

PyTorch analogy: `torch.optim.Adam(lr=lr, betas=(beta1,beta2), eps=epsilon)`.
-/
def adam {α : Type} [Storage α] [Context α]
    (lr beta1 beta2 epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := Pack (Adam.State α) paramShapes
    init := fun ps =>
      initStateList (α := α) (State := fun _ _ => Adam.State α)
        (initOne := fun {s} t =>
          Adam.init
            (α := α) (s := s) lr beta1 beta2 epsilon t) ps
    step := fun st ps grads =>
      stepStateList (α := α) (State := fun _ _ => Adam.State α)
        (updateOne := fun {s} stOne params g => Adam.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
    native? := fun st =>
      some (.adam (firstStateValue lr (fun state => state.lr) st)
        beta1 beta2 epsilon)
  }

/--
AdamW (Adam with decoupled weight decay).

PyTorch analogy: `torch.optim.AdamW(lr=lr, weight_decay=weightDecay, betas=(beta1,beta2),
  eps=epsilon)`.
-/
def adamw {α : Type} [Storage α] [Context α]
    (lr weightDecay beta1 beta2 epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := Pack (AdamW.State α) paramShapes
    init := fun ps =>
      initStateList (α := α) (State := fun _ _ => AdamW.State α)
        (initOne := fun {s} t =>
          AdamW.init
            (α := α) (s := s) lr weightDecay beta1 beta2 epsilon t) ps
    step := fun st ps grads =>
      stepStateList (α := α) (State := fun _ _ => AdamW.State α)
        (updateOne := fun {s} stOne params g => AdamW.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
    native? := fun st =>
      some (.adamW (firstStateValue lr (fun state => state.lr) st)
        weightDecay beta1 beta2 epsilon)
  }

/--
AdaDelta (adaptive learning rate method similar to RMSProp but with a running RMS of updates).

PyTorch analogy: `torch.optim.Adadelta(lr=lr, rho=rho, eps=epsilon)`.
-/
def adadelta {α : Type} [Storage α] [Context α]
    (lr rho epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := Pack (Adadelta.State α) paramShapes
    init := fun ps =>
      initStateList (α := α) (State := Adadelta.State)
        (initOne := fun {s} t =>
          Adadelta.init
            (α := α) (s := s) lr rho epsilon t) ps
    step := fun st ps grads =>
      stepStateList (α := α) (State := Adadelta.State)
        (updateOne := fun {s} stOne params g => Adadelta.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
  }

end Optimizer
end TorchLean
