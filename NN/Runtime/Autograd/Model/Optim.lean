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
- Updates run on *plain tensors* after the backward pass, so they are not recorded on the
  autograd tape.
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
  /-- The same state with every parameter's learning rate set to the given value; schedules use it
  between steps. -/
  setLr : α → State → State
  /--
  The update this state names among the trainer's native optimizers, if any. A trainer whose
  backend keeps parameters, gradients and optimizer moments on the device (eager CUDA) runs it in
  place of the generic `backward` + `step` path, and the state is then kept as is, since the moments
  live on the device. `none` always takes the generic path.
  -/
  native? : State → Option (Torch.NativeOptimizer α) := fun _ => none

/--
Run `optimizer`'s native update on a nonempty batch when both the optimizer and the trainer's
backend have one; one sample is a batch of one. The outer option reports whether the native path
ran, and the inner loss is present exactly when `readLoss` was requested.
-/
def Optimizer.nativeBatchStep? {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β] [Context α]
    {paramShapes inputShapes dataInputShapes : List Shape}
    (optimizer : Optimizer α paramShapes)
    (trainer : Torch.ScalarTrainer α β paramShapes inputShapes dataInputShapes)
    (state : optimizer.State)
    (batch : Array (TorchLean.TensorPack α inputShapes ×
      TorchLean.TensorPack β dataInputShapes))
    (readLoss : Bool) : IO (Option (Option (Tensor α []))) :=
  match optimizer.native? state, trainer.nativeBatchStep? with
  | some native, some step => some <$> step native batch readLoss
  | _, _ => pure none

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

/-- Read a shape-independent field from the first optimizer state, or use `fallback` when there
are no parameters. -/
def firstStateValue
    {State : (α : Type) → [TorchLean.Storage α] → Shape → Type}
    {α β : Type} [TorchLean.Storage α] (fallback : β)
    (get : {s : Shape} → State α s → β) :
    {ss : List Shape} → StateList State α ss → β
  | [], .nil => fallback
  | _ :: _, .cons st _ => get st

/-- Apply `f` to every per-parameter optimizer state. -/
def mapStateList
    {State : (α : Type) → [TorchLean.Storage α] → Shape → Type}
    {α : Type} [TorchLean.Storage α]
    (f : {s : Shape} → State α s → State α s) :
    {ss : List Shape} → StateList State α ss → StateList State α ss
  | [], .nil => .nil
  | _ :: _, .cons st rest => .cons (f st) (mapStateList f rest)

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
    (lr : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.SGD.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.SGD.State)
        (initOne := fun {s} t =>
          Optim.SGD.init (α := α) (s := s) lr t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.SGD.State)
        (updateOne := fun {s} stOne parameters gradients =>
          Optim.SGD.update (α := α) (s := s)
            stOne parameters gradients)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := Optim.SGD.State)
        (fun stOne => { stOne with lr := rate }) st
    native? := fun st =>
      some (.sgd (Internal.firstStateValue lr (fun state => state.lr) st))
  }

/--
SGD with classical momentum.

PyTorch analogy: `torch.optim.SGD(lr=lr, momentum=momentum)`.
-/
def momentumSGD {α : Type} [TorchLean.Storage α] [Context α]
    (lr momentum : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.MomentumSGD.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.MomentumSGD.State)
        (initOne := fun {s} t =>
          Optim.MomentumSGD.init
            (α := α) (s := s) lr momentum t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.MomentumSGD.State)
        (updateOne := fun {s} stOne params g => Optim.MomentumSGD.update (α := α) (s := s)
          stOne params g)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := Optim.MomentumSGD.State)
        (fun stOne => { stOne with lr := rate }) st
  }

/--
AdaGrad (per-parameter learning rate scaling by accumulated squared gradients).

PyTorch analogy: `torch.optim.Adagrad(lr=lr, eps=epsilon)`.
-/
def adagrad {α : Type} [TorchLean.Storage α] [Context α]
    (lr epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.AdaGrad.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.AdaGrad.State)
        (initOne := fun {s} t =>
          Optim.AdaGrad.init
            (α := α) (s := s) lr epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.AdaGrad.State)
        (updateOne := fun {s} stOne params g => Optim.AdaGrad.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := Optim.AdaGrad.State)
        (fun stOne => { stOne with lr := rate }) st
  }

/--
RMSProp (exponentially-decayed second moment / running average of squared gradients).

PyTorch analogy: `torch.optim.RMSprop(lr=lr, alpha=decay, eps=epsilon)` (we use the common naming
`decay` for `alpha`).
-/
def rmsprop {α : Type} [TorchLean.Storage α] [Context α]
    (lr decay epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.RMSProp.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.RMSProp.State)
        (initOne := fun {s} t =>
          Optim.RMSProp.init
            (α := α) (s := s) lr decay epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.RMSProp.State)
        (updateOne := fun {s} stOne params g => Optim.RMSProp.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := Optim.RMSProp.State)
        (fun stOne => { stOne with lr := rate }) st
  }

/--
Adam (first/second moment estimates).

PyTorch analogy: `torch.optim.Adam(lr=lr, betas=(beta1,beta2), eps=epsilon)`.
-/
def adam {α : Type} [TorchLean.Storage α] [Context α]
    (lr beta1 beta2 epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList (fun _ _ => Optim.Adam.State α) α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := fun _ _ => Optim.Adam.State α)
        (initOne := fun {s} t =>
          Optim.Adam.init
            (α := α) (s := s) lr beta1 beta2 epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := fun _ _ => Optim.Adam.State α)
        (updateOne := fun {s} stOne params g => Optim.Adam.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := fun _ _ => Optim.Adam.State α)
        (fun stOne => { stOne with lr := rate }) st
    native? := fun st =>
      some (.adam (Internal.firstStateValue lr (fun state => state.lr) st)
        beta1 beta2 epsilon)
  }

/--
AdamW (Adam with decoupled weight decay).

PyTorch analogy: `torch.optim.AdamW(lr=lr, weight_decay=weightDecay, betas=(beta1,beta2),
  eps=epsilon)`.
-/
def adamw {α : Type} [TorchLean.Storage α] [Context α]
    (lr weightDecay beta1 beta2 epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList (fun _ _ => Optim.AdamW.State α) α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := fun _ _ => Optim.AdamW.State α)
        (initOne := fun {s} t =>
          Optim.AdamW.init
            (α := α) (s := s) lr weightDecay beta1 beta2 epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := fun _ _ => Optim.AdamW.State α)
        (updateOne := fun {s} stOne params g => Optim.AdamW.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := fun _ _ => Optim.AdamW.State α)
        (fun stOne => { stOne with lr := rate }) st
    native? := fun st =>
      some (.adamW (Internal.firstStateValue lr (fun state => state.lr) st)
        weightDecay beta1 beta2 epsilon)
  }

/--
AdaDelta (adaptive learning rate method similar to RMSProp but with a running RMS of updates).

PyTorch analogy: `torch.optim.Adadelta(lr=lr, rho=rho, eps=epsilon)`.
-/
def adadelta {α : Type} [TorchLean.Storage α] [Context α]
    (lr rho epsilon : α)
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.Adadelta.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.Adadelta.State)
        (initOne := fun {s} t =>
          Optim.Adadelta.init
            (α := α) (s := s) lr rho epsilon t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.Adadelta.State)
        (updateOne := fun {s} stOne params g => Optim.Adadelta.update (α := α) (s := s) stOne
          params g)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := Optim.Adadelta.State)
        (fun stOne => { stOne with lr := rate }) st
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
    (lr : α)
    (projector : {s : Shape} → Optim.GaLore.Projector α s s :=
      fun {s} => Optim.GaLore.identityProjector (α := α) (s := s))
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList (fun α _ s => Optim.GaLore.SGDState α s s) α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α)
        (State := fun α _ s => Optim.GaLore.SGDState α s s)
        (initOne := fun {s} _t =>
          { lr := lr, projector := projector (s := s) }) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α)
        (State := fun α _ s => Optim.GaLore.SGDState α s s)
        (updateOne := fun {s} stOne parameters gradients =>
          Optim.GaLore.update
            (α := α) (full := s) (low := s) stOne parameters gradients)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := fun α _ s => Optim.GaLore.SGDState α s s)
        (fun stOne => { stOne with lr := rate }) st
  }

/--
Muon-style momentum with a caller-supplied same-shape orthogonalization backend.

Using the identity backend gives ordinary momentum-SGD behavior. A production Muon backend should
provide a matrix-specific Newton-Schulz orthogonalizer and optional CUDA kernels.
-/
def muon {α : Type} [TorchLean.Storage α] [Context α]
    (lr momentum : α)
    (orthogonalizer : {s : Shape} → Optim.Muon.Orthogonalizer α s :=
      fun {s} => Optim.Muon.identityOrthogonalizer (α := α) (s := s))
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList Optim.Muon.State α paramShapes
    init := fun ps =>
      Internal.initStateList (α := α) (State := Optim.Muon.State)
        (initOne := fun {s} t =>
          Optim.Muon.init
            (α := α) (s := s) lr momentum
            (orthogonalizer (s := s)) t) ps
    step := fun st ps grads =>
      Internal.stepStateList (α := α) (State := Optim.Muon.State)
        (updateOne := fun {s} stOne params g =>
          Optim.Muon.update (α := α) (s := s) stOne params g)
        ps st grads
    setLr := fun rate st =>
      Internal.mapStateList (State := Optim.Muon.State)
        (fun stOne => { stOne with lr := rate }) st
  }

end Optim

end Model
end Autograd
end Runtime
