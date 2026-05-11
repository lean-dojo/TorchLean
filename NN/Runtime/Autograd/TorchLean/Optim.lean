/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
public import NN.Runtime.Optim.Optimizers

import Mathlib.Algebra.Order.Algebra

/-!
# Optim

TorchLean optimizer wrappers.

This connects the pure tensor optimizers in `NN/Runtime/Optim/Optimizers.lean` to the runtime
training structures (`Torch.ParamList` + gradient `TList`).

Design notes:
- Optimizer state is stored in a shape-indexed list aligned with the parameter shapes.
- Updates run on *plain tensors* (not via the autograd tape), so they work the same for eager and
  compiled training loops.
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
namespace TorchLean

open Spec
open Tensor

namespace Optim

/-! ## Generic optimizer interface -/

/--
A shape-indexed list of optimizer state values.

This mirrors the parameter-shape list used by `Torch.ParamList`. Each parameter gets its own
per-parameter optimizer state (e.g. momentum buffers) with the same shape as the parameter.
-/
inductive StateList (State : Type → Shape → Type) (α : Type) : List Shape → Type where
  | nil : StateList State α []
  | cons {s : Shape} {ss : List Shape} : State α s → StateList State α ss → StateList State α (s ::
    ss)

/--
Runtime-facing optimizer interface.

This is the analogue of a PyTorch `torch.optim.Optimizer`, but made explicit about:
- which parameter shapes it manages (`paramShapes`), and
- how it stores internal state (`State`) aligned with those shapes.
-/
structure Optimizer (α : Type) [Context α] (paramShapes : List Shape) where
  /-- State. -/
  State : Type
  /-- init. -/
  init : Torch.ParamList α paramShapes → IO State
  /-- step. -/
  step : State → Torch.ParamList α paramShapes → Torch.TList α paramShapes → IO State
  /--
  Optional trainer-native step.

  Most optimizers are implemented by first materializing a gradient `TList` and then updating host
  parameter tensors.  Some trainers can do better.  In eager CUDA mode, for example, the trainer can
  keep gradients and optimizer moments on device for SGD/Adam.  When this hook returns `some st'`,
  callers should treat the step as complete and use `st'` as the next optimizer state.  Returning
  `none` asks the caller to fall back to the generic `backward` + `step` path.
  -/
  trainerStep? : {inputShapes : List Shape} →
    Torch.ScalarTrainer α paramShapes inputShapes → State → Torch.TList α inputShapes →
      IO (Option State) := fun {_inputShapes} _tr _st _xs => pure none

namespace Private

/--
Initialize an optimizer state list by reading the current parameter tensors.

This is used to build the per-parameter state buffers (for example: momentum vectors) with the
correct shape.
-/
def initStateList {α : Type} [Context α] {State : Type → Shape → Type} :
    {ss : List Shape} →
    (initOne : {s : Shape} → Tensor α s → State α s) →
    Torch.ParamList α ss → IO (StateList State α ss)
  | [], _initOne, .nil => pure .nil
  | _s :: ss, initOne, .cons p ps => do
      let v ← p.value.get
      let st := initOne (s := _s) v
      let rest ← initStateList (α := α) (State := State) (ss := ss) initOne ps
      pure (.cons st rest)

/--
Run one optimizer update step over a parameter list.

`updateOne` receives `(state, param, grad)` and returns `(newState, newParam)`. Parameters are
updated in-place via `IO.Ref` in the `Torch.ParamList`.
-/
def stepStateList {α : Type} [Context α] {State : Type → Shape → Type} :
    {ss : List Shape} →
    (updateOne : {s : Shape} → State α s → Tensor α s → Tensor α s → (State α s × Tensor α s)) →
    Torch.ParamList α ss → StateList State α ss → Torch.TList α ss → IO (StateList State α ss)
  | [], _updateOne, .nil, .nil, .nil => pure .nil
  | _s :: ss, updateOne, .cons p ps, .cons st sts, .cons g gs => do
      let st' ←
        if p.requiresGrad then
          let pv ← p.value.get
          let (stNew, pNew) := updateOne (s := _s) st pv g
          Torch.Internal.setParamHostValue (α := α) (sh := _s) p (Tensor.materialize pNew)
          pure stNew
        else
          pure st
      let rest ← stepStateList (α := α) (State := State) (ss := ss) updateOne ps sts gs
      pure (.cons st' rest)

end Private

/-! ## Concrete optimizers -/

/--
Stochastic gradient descent.

PyTorch analogy: `torch.optim.SGD(lr=lr)` without momentum.
-/
def sgd {α : Type} [Context α] (lr : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList _root_.Optim.SGD.State α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := _root_.Optim.SGD.State)
        (initOne := fun {s} t => _root_.Optim.SGD.init (α := α) (s := s) lr t) ps
    step := fun st ps grads =>
      Private.stepStateList (α := α) (State := _root_.Optim.SGD.State)
        (updateOne := fun {s} stOne params g =>
          (stOne, _root_.Optim.SGD.update (α := α) (s := s) stOne params g))
        ps st grads
    trainerStep? := fun {_inputShapes} tr st xs => do
      Torch.Curried.uncurry (α := α) (ss := _inputShapes) (β := IO Unit)
        (tr.step lr) xs
      pure (some st)
  }

/--
SGD with classical momentum.

PyTorch analogy: `torch.optim.SGD(lr=lr, momentum=momentum)`.
-/
def momentumSGD {α : Type} [Context α] (lr momentum : α) {paramShapes : List Shape} : Optimizer α
  paramShapes :=
  { State := StateList _root_.Optim.MomentumSGD.State α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := _root_.Optim.MomentumSGD.State)
        (initOne := fun {s} t => _root_.Optim.MomentumSGD.init (α := α) (s := s) lr momentum t) ps
    step := fun st ps grads =>
      Private.stepStateList (α := α) (State := _root_.Optim.MomentumSGD.State)
        (updateOne := fun {s} stOne params g => _root_.Optim.MomentumSGD.update (α := α) (s := s)
          stOne params g)
        ps st grads
  }

/--
AdaGrad (per-parameter learning rate scaling by accumulated squared gradients).

PyTorch analogy: `torch.optim.Adagrad(lr=lr, eps=epsilon)`.
-/
def adagrad {α : Type} [Context α] (lr epsilon : α) {paramShapes : List Shape} : Optimizer α
  paramShapes :=
  { State := StateList _root_.Optim.AdaGrad.State α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := _root_.Optim.AdaGrad.State)
        (initOne := fun {s} t => _root_.Optim.AdaGrad.init (α := α) (s := s) lr epsilon t) ps
    step := fun st ps grads =>
      Private.stepStateList (α := α) (State := _root_.Optim.AdaGrad.State)
        (updateOne := fun {s} stOne params g => _root_.Optim.AdaGrad.update (α := α) (s := s) stOne
          params g)
        ps st grads
  }

/--
RMSProp (exponentially-decayed second moment / running average of squared gradients).

PyTorch analogy: `torch.optim.RMSprop(lr=lr, alpha=decay, eps=epsilon)` (we use the common naming
`decay` for `alpha`).
-/
def rmsprop {α : Type} [Context α] (lr decay epsilon : α) {paramShapes : List Shape} : Optimizer α
  paramShapes :=
  { State := StateList _root_.Optim.RMSProp.State α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := _root_.Optim.RMSProp.State)
        (initOne := fun {s} t => _root_.Optim.RMSProp.init (α := α) (s := s) lr decay epsilon t) ps
    step := fun st ps grads =>
      Private.stepStateList (α := α) (State := _root_.Optim.RMSProp.State)
        (updateOne := fun {s} stOne params g => _root_.Optim.RMSProp.update (α := α) (s := s) stOne
          params g)
        ps st grads
  }

/--
Adam (first/second moment estimates).

PyTorch analogy: `torch.optim.Adam(lr=lr, betas=(beta1,beta2), eps=epsilon)`.
-/
def adam {α : Type} [Context α]
    (lr beta1 beta2 epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList _root_.Optim.Adam.State α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := _root_.Optim.Adam.State)
        (initOne := fun {s} t => _root_.Optim.Adam.init (α := α) (s := s) lr beta1 beta2 epsilon t)
          ps
    step := fun st ps grads =>
      Private.stepStateList (α := α) (State := _root_.Optim.Adam.State)
        (updateOne := fun {s} stOne params g => _root_.Optim.Adam.update (α := α) (s := s) stOne
          params g)
        ps st grads
    trainerStep? := fun {_inputShapes} tr st xs =>
      match tr.adamStep? with
      | none => pure none
      | some step => do
          Torch.Curried.uncurry (α := α) (ss := _inputShapes) (β := IO Unit)
            (step lr beta1 beta2 epsilon) xs
          pure (some st)
  }

/--
AdamW (Adam with decoupled weight decay).

PyTorch analogy: `torch.optim.AdamW(lr=lr, weight_decay=weightDecay, betas=(beta1,beta2),
  eps=epsilon)`.
-/
def adamw {α : Type} [Context α]
    (lr weightDecay beta1 beta2 epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList _root_.Optim.AdamW.State α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := _root_.Optim.AdamW.State)
        (initOne := fun {s} t => _root_.Optim.AdamW.init (α := α) (s := s) lr weightDecay beta1
          beta2 epsilon t) ps
    step := fun st ps grads =>
      Private.stepStateList (α := α) (State := _root_.Optim.AdamW.State)
        (updateOne := fun {s} stOne params g => _root_.Optim.AdamW.update (α := α) (s := s) stOne
          params g)
        ps st grads
    trainerStep? := fun {_inputShapes} tr st xs =>
      match tr.adamWStep? with
      | none => pure none
      | some step => do
          Torch.Curried.uncurry (α := α) (ss := _inputShapes) (β := IO Unit)
            (step lr weightDecay beta1 beta2 epsilon) xs
          pure (some st)
  }

/--
AdaDelta (adaptive learning rate method similar to RMSProp but with a running RMS of updates).

PyTorch analogy: `torch.optim.Adadelta(lr=lr, rho=rho, eps=epsilon)`.
-/
def adadelta {α : Type} [Context α]
    (lr rho epsilon : α) {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList _root_.Optim.Adadelta.State α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := _root_.Optim.Adadelta.State)
        (initOne := fun {s} t => _root_.Optim.Adadelta.init (α := α) (s := s) lr rho epsilon t) ps
    step := fun st ps grads =>
      Private.stepStateList (α := α) (State := _root_.Optim.Adadelta.State)
        (updateOne := fun {s} stOne params g => _root_.Optim.Adadelta.update (α := α) (s := s) stOne
          params g)
        ps st grads
  }

/-! ## Optimizer extension points -/

/--
Projected SGD.

This is the runtime-safe part of a GaLore-style optimizer: every parameter gets a same-shape
projector/lift pair, and the update applies `p ← p - lr * lift(project(g))`.

Full GaLore also needs a rank-changing projector and a refresh schedule. Those pieces require
matrix-specific state and SVD/randomized-SVD infrastructure, so they are intentionally not hidden
inside this generic constructor.
-/
def projectedSGD {α : Type} [Context α] (lr : α)
    (projector : {s : Shape} → _root_.Optim.GaLore.Projector α s s :=
      fun {s} => _root_.Optim.GaLore.identityProjector (α := α) (s := s))
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList (fun α s => _root_.Optim.GaLore.SGDState α s s) α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := fun α s => _root_.Optim.GaLore.SGDState α s s)
        (initOne := fun {s} _t => { lr := lr, projector := projector (s := s) }) ps
    step := fun st ps grads =>
      Private.stepStateList (α := α)
        (State := fun α s => _root_.Optim.GaLore.SGDState α s s)
        (updateOne := fun {s} stOne params g =>
          (stOne, _root_.Optim.GaLore.projectedSGDUpdate (α := α) (full := s) (low := s)
            stOne params g))
        ps st grads
  }

/--
Muon-style momentum with a caller-supplied same-shape orthogonalization backend.

Using the identity backend gives ordinary momentum-SGD behavior. A production Muon backend should
provide a matrix-specific Newton-Schulz orthogonalizer and optional CUDA kernels.
-/
def muon {α : Type} [Context α] (lr momentum : α)
    (orthogonalizer : {s : Shape} → _root_.Optim.Muon.Orthogonalizer α s :=
      fun {s} => _root_.Optim.Muon.identityOrthogonalizer (α := α) (s := s))
    {paramShapes : List Shape} : Optimizer α paramShapes :=
  { State := StateList _root_.Optim.Muon.State α paramShapes
    init := fun ps =>
      Private.initStateList (α := α) (State := _root_.Optim.Muon.State)
        (initOne := fun {s} t =>
          _root_.Optim.Muon.init (α := α) (s := s) lr momentum (orthogonalizer (s := s)) t) ps
    step := fun st ps grads =>
      Private.stepStateList (α := α) (State := _root_.Optim.Muon.State)
        (updateOne := fun {s} stOne params g =>
          _root_.Optim.Muon.update (α := α) (s := s) stOne params g)
        ps st grads
  }

end Optim

end TorchLean
end Autograd
end Runtime
