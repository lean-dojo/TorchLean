/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core
public import NN.Runtime.Optim.Optimizers
public import NN.Runtime.Optim.Schedulers
public import Std.Data.HashMap

/-!
# Optimizer integration for Runtime.Autograd

This module is the training-loop side of autograd: it takes a gradient map produced by
`Runtime.Autograd` and applies parameter updates.

PyTorch analogy:
- `ParamTable` is like an ordered list of parameters, but we key everything by a stable `Nat` id
  (closer to `state_dict` keys than pointer identity).
- `ParamGroup` and `OptimizerState` mirror `torch.optim.Optimizer` parameter groups and state.
- `LRScheduler` is a small wrapper around our scheduler implementations, similar to
  `torch.optim.lr_scheduler.*`.

All updates are *shape checked* and implemented using the pure `Spec` tensor operators, so they
can be used in eager execution or lowered into the compiled IR.

Formula ownership:
- this file owns the heterogeneous parameter-table handling, parameter groups, lazy state maps,
  scheduler stepping, and PyTorch-style coupled weight decay at the training-loop boundary;
- `NN.Runtime.Optim.Optimizers` owns the canonical per-tensor optimizer equations.

The important rule is: this file must not define a second public optimizer-formula surface. The
`step` implementation below constructs canonical optimizer states from the dynamic parameter-table
buffers and calls `NN.Runtime.Optim.Optimizers` directly. The only local algebra left here is
training-loop glue that is not represented by the canonical pure states, such as coupled
weight-decay preprocessing and PyTorch-style momentum dampening/Nesterov handling.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

open Spec
open Tensor

/-!
## Parameter table
-/
/-!
This section provides the parameter registry used by the training loop.

Unlike PyTorch (where parameters are objects with identity), we use an explicit `Nat` id so that:
- gradients can be stored in a `HashMap Nat _`,
- optimizer state buffers can be stored in a `HashMap Nat _`,
- serialization can be done by a pure `state_dict` record.
-/
/--
A single trainable parameter entry.

This is the Runtime.Autograd equivalent of a "parameter tensor" in PyTorch, except we make the
identifier explicit (`id : Nat`) so we can key gradients and optimizer state in pure maps.
-/
structure ParamEntry (α : Type) where
  /-- Stable identifier used to key gradients and optimizer state. -/
  id : Nat
  /-- Optional human-readable name (e.g. module path); used only for reporting/debugging. -/
  name : Option String := none
  /-- The parameter value, stored as an `AnyTensor` (shape erased). -/
  value : Runtime.AnyTensor α

/-- A flat list of parameters used by the training loop. -/
abbrev ParamTable (α : Type) := List (ParamEntry α)

namespace ParamEntry

/-!
### Constructors
-/
/--
Create a `ParamEntry` from a typed tensor.

This is mostly a convenience for assembling a `ParamTable` from known-shaped tensors.
-/
def ofTensor {α : Type} {s : Shape} (id : Nat) (t : Tensor α s) (name : Option String := none) :
  ParamEntry α :=
  { id := id, name := name, value := AnyTensor.mk t }

end ParamEntry

namespace ParamTable

variable {α : Type}

/-- List of ids for quick membership checks. -/
def ids (ps : ParamTable α) : List Nat :=
  ps.map (·.id)

/-- Find a parameter entry by id. -/
def find? (ps : ParamTable α) (id : Nat) : Option (ParamEntry α) :=
  List.find? (fun p => p.id == id) ps

/-- Get a typed tensor from the table, with shape checking. -/
def getTensor {α : Type} [DecidableEq Shape] {s : Shape}
  (tag : String) (ps : ParamTable α) (id : Nat) : Result (Tensor α s) := by
  match find? ps id with
  | none =>
      exact .error (tagError tag s!"missing param id {id}")
  | some p =>
      if h : p.value.s = s then
        exact .ok (Tensor.castShape p.value.t h)
      else
        exact .error (tagError tag s!"param shape mismatch for id {id}")

/-- Replace a parameter entry value by id. -/
def set (ps : ParamTable α) (id : Nat) (value : Runtime.AnyTensor α) : ParamTable α :=
  ps.map (fun p => if p.id = id then { p with value := value } else p)

end ParamTable

/-!
## Scheduler wrapper
-/
/--
Learning-rate scheduler wrapper used by the training loop.

PyTorch analogy: this plays the role of `torch.optim.lr_scheduler.*` objects, except we keep the
state as an inductive value and expose a pure `getLR`/`advance` API.
-/
inductive LRScheduler (α : Type) where
  | constant : Optim.ConstantScheduler α -> LRScheduler α
  | exponential : Optim.ExponentialDecayScheduler α -> LRScheduler α
  | step : Optim.StepDecayScheduler α -> LRScheduler α
  | cosine : Optim.CosineAnnealingScheduler α -> LRScheduler α
  | linearWarmup : Optim.LinearWarmupScheduler α -> LRScheduler α
  | warmupCosine : Optim.WarmupCosineScheduler α -> LRScheduler α
  | cyclic : Optim.CyclicScheduler α -> LRScheduler α
  | triangular : Optim.TriangularCycleScheduler α -> LRScheduler α
  | oneCycle : Optim.OneCycleScheduler α -> LRScheduler α
  | lrFinder : Optim.LRFinder α -> LRScheduler α
  /--
  Custom schedule with an explicit step counter.

`custom f k` means "use learning rate `f k` now, and increment to `k+1` on `advance`".
  -/
  | custom : (Nat -> α) -> Nat -> LRScheduler α

namespace LRScheduler

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Read current learning rate from the scheduler state. -/
def getLR : LRScheduler α -> α
  | constant s => Optim.ConstantScheduler.getLr s 0
  | exponential s => Optim.ExponentialDecayScheduler.getLr s
  | step s => Optim.StepDecayScheduler.getLr s
  | cosine s => Optim.CosineAnnealingScheduler.getLr s
  | linearWarmup s => Optim.LinearWarmupScheduler.getLr s
  | warmupCosine s => Optim.WarmupCosineScheduler.getLr s
  | cyclic s => Optim.CyclicScheduler.getLr s
  | triangular s => Optim.TriangularCycleScheduler.getLr s
  | oneCycle s => Optim.OneCycleScheduler.getLr s
  | lrFinder s => Optim.LRFinder.getLr s
  | custom f k => f k

/-- Advance scheduler state by one step. -/
def advance : LRScheduler α -> LRScheduler α
  | constant s => constant (Optim.ConstantScheduler.step s)
  | exponential s => exponential (Optim.ExponentialDecayScheduler.step s)
  | step s => step (Optim.StepDecayScheduler.step s)
  | cosine s => cosine (Optim.CosineAnnealingScheduler.step s)
  | linearWarmup s => linearWarmup (Optim.LinearWarmupScheduler.step s)
  | warmupCosine s => warmupCosine (Optim.WarmupCosineScheduler.step s)
  | cyclic s => cyclic (Optim.CyclicScheduler.step s)
  | triangular s => triangular (Optim.TriangularCycleScheduler.step s)
  | oneCycle s => oneCycle (Optim.OneCycleScheduler.step s)
  | lrFinder s => lrFinder (Optim.LRFinder.step s)
  | custom f k => custom f (k + 1)

end LRScheduler

/-!
## Optimizer configuration
-/
/--
Which optimizer update rule to apply.

PyTorch analogy: these correspond roughly to `torch.optim.SGD`, `Adam`, `AdamW`, etc.
-/
inductive OptimizerKind
  | sgd
  | momentum
  | adagrad
  | rmsprop
  | adam
  | adamw
  | adadelta
  deriving Repr, DecidableEq

/--
Optimizer hyperparameters for a subset of parameters.

PyTorch analogy: this is a single entry in the optimizer's param-group list
(`optimizer.param_groups`).
-/
structure ParamGroup (α : Type) [Context α] where
  /-- Parameter ids that belong to this group. -/
  params : List Nat
  /-- Base learning rate (possibly overridden by `scheduler` on each step). -/
  lr : α
  /-- L2 regularization coefficient (behavior depends on the optimizer kind; see AdamW). -/
  weight_decay : α := 0
  /-- Momentum factor (SGD with momentum). -/
  momentum : α := 0
  /-- Dampening for momentum updates. -/
  dampening : α := 0
  /-- Use Nesterov variant for momentum updates. -/
  nesterov : Bool := false
  /-- Adam beta1 parameter (exponential decay for the first moment). -/
  beta1 : α := Numbers.one - Numbers.pointone
  /-- Adam beta2 parameter (exponential decay for the second moment). -/
  beta2 : α := Numbers.one - (Numbers.one / (-Numbers.neg_thousand))
  /-- Numerical stability term used by adaptive optimizers. -/
  epsilon : α := Numbers.epsilon
  /-- "Rho" decay parameter for RMSProp/AdaDelta style optimizers. -/
  rho : α := Numbers.one - Numbers.pointone
  /-- Optional learning-rate scheduler for this group. -/
  scheduler : Option (LRScheduler α) := none

/--
Full optimizer state used by the training loop.

This mirrors PyTorch's optimizer state:
- a global step counter,
- hyperparameter groups,
- and per-parameter state buffers keyed by parameter id (`Nat`).
-/
structure OptimizerState (α : Type) [Context α] where
  /-- Which update rule to apply on `step`. -/
  kind : OptimizerKind
  /-- Parameter groups (hyperparameters + membership). -/
  groups : List (ParamGroup α)
  /-- Global step counter (increments once per `step`). -/
  step : Nat := 0
  /-- Momentum buffer (SGD with momentum / Nesterov), keyed by parameter id. -/
  momentum_buf : Std.HashMap Nat (Runtime.AnyTensor α) := {}
  /-- Adam first-moment estimate, keyed by parameter id. -/
  m : Std.HashMap Nat (Runtime.AnyTensor α) := {}
  /-- Adam second-moment estimate, keyed by parameter id. -/
  v : Std.HashMap Nat (Runtime.AnyTensor α) := {}
  /-- Accumulator buffer (AdaGrad/RMSProp/AdaDelta), keyed by parameter id. -/
  acc : Std.HashMap Nat (Runtime.AnyTensor α) := {}
  /-- Second accumulator buffer (AdaDelta), keyed by parameter id. -/
  acc2 : Std.HashMap Nat (Runtime.AnyTensor α) := {}

/--
A pure state snapshot for saving/restoring optimizer state.

PyTorch analogy: this is the data carried by `optimizer.state_dict()` (modulo naming/layout).
We use association lists instead of `HashMap` so the result is deterministic and easy to serialize.
-/
structure OptimStateDict (α : Type) [Context α] where
  /-- Optimizer algorithm used to interpret the stored buffers. -/
  kind : OptimizerKind
  /-- Global optimizer step at the time the snapshot was taken. -/
  step : Nat
  /-- Parameter groups, including scheduler state and hyperparameters. -/
  groups : List (ParamGroup α)
  /-- Momentum buffers keyed by parameter id. -/
  momentum_buf : List (Nat × Runtime.AnyTensor α)
  /-- Adam-family first-moment buffers keyed by parameter id. -/
  m : List (Nat × Runtime.AnyTensor α)
  /-- Adam-family second-moment buffers keyed by parameter id. -/
  v : List (Nat × Runtime.AnyTensor α)
  /-- AdaGrad/RMSProp/Adadelta accumulator buffers keyed by parameter id. -/
  acc : List (Nat × Runtime.AnyTensor α)
  /-- Adadelta second accumulator buffers keyed by parameter id. -/
  acc2 : List (Nat × Runtime.AnyTensor α)

namespace OptimizerState

variable {α : Type} [Context α]

/--
Serialize optimizer state to a pure record.

PyTorch analogy: this is the "export" step for `state_dict()`.
-/
def toStateDict (opt : OptimizerState α) : OptimStateDict α :=
  { kind := opt.kind
  , step := opt.step
  , groups := opt.groups
  , momentum_buf := opt.momentum_buf.toList
  , m := opt.m.toList
  , v := opt.v.toList
  , acc := opt.acc.toList
  , acc2 := opt.acc2.toList
  }

/--
Restore optimizer state from a state dict.

PyTorch analogy: this is the "import" step for `load_state_dict(...)`.
-/
def ofStateDict (d : OptimStateDict α) : OptimizerState α :=
  { kind := d.kind
  , step := d.step
  , groups := d.groups
  , momentum_buf := Std.HashMap.ofList d.momentum_buf
  , m := Std.HashMap.ofList d.m
  , v := Std.HashMap.ofList d.v
  , acc := Std.HashMap.ofList d.acc
  , acc2 := Std.HashMap.ofList d.acc2
  }

end OptimizerState

/-!
## Optimizer step
-/
namespace Optim

variable {α : Type} [Context α] [DecidableEq Shape] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Create a zero-filled buffer with the same shape as a parameter value. -/
def zerosLike (p : Runtime.AnyTensor α) : Runtime.AnyTensor α :=
  { s := p.s, t := Spec.fill (0 : α) p.s }

/--
Lookup a per-parameter state buffer, initializing it with zeros if absent.

This is used for momentum/Adam accumulator initialization (PyTorch does this lazily on first step).
-/
def getOrInit
  (m : Std.HashMap Nat (Runtime.AnyTensor α))
  (id : Nat) (p : Runtime.AnyTensor α) : Runtime.AnyTensor α :=
  m.getD id (zerosLike p)

/--
Shape-check and cast an optimizer state buffer to match the current parameter value.

This prevents silent shape mismatches when reloading a checkpoint into a model with different
parameter shapes.
-/
def castState
  (tag : String) (id : Nat) (buf pval : Runtime.AnyTensor α) : Result (Tensor α pval.s) := do
  if h : buf.s = pval.s then
    pure (Tensor.castShape buf.t h)
  else
    throw (tagError tag s!"state shape mismatch for id {id}")

/--
Add an L2 regularization term to the gradient: `g + weight_decay * param`.

Note: this is the *coupled* weight decay used by classic SGD-style updates.
For AdamW the integration step delegates to the canonical optimizer's decoupled update.
-/
def addWeightDecay {s : Shape}
  (param grad : Tensor α s) (weight_decay : α) : Tensor α s :=
  addSpec grad (scaleSpec param weight_decay)

/--
Convert the training-loop Adam step number into the canonical optimizer state's previous step.

The public training helper receives the step being applied (`1` for the first Adam/AdamW update).
`NN.Runtime.Optim.Optimizers` stores the previous step in the state and increments internally.
For direct calls with `t = 0`, we still return `0` so the helper stays total and behaves like a
first update rather than constructing a negative predecessor.
-/
def adamPreviousStep (t : Nat) : Nat :=
  if t = 0 then 0 else t - 1

/--
Update each group's learning rate from its scheduler (if present) and advance the scheduler state.

This matches the common training-loop pattern: "read LR, then call `scheduler.step()`".
-/
def updateGroupSchedulers {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (groups : List (ParamGroup α)) : List (ParamGroup α) :=
  groups.map (fun g =>
    let lr := match g.scheduler with
      | none => g.lr
      | some s => LRScheduler.getLR s
    let sched := g.scheduler.map LRScheduler.advance
    { g with lr := lr, scheduler := sched })

/--
Build a map from parameter id to its `ParamGroup`.

Fails if an id appears in multiple groups (PyTorch also disallows overlapping param groups).
-/
def groupMap {α : Type} [Context α]
  (groups : List (ParamGroup α)) : Result (Std.HashMap Nat (ParamGroup α)) := do
  let mut m : Std.HashMap Nat (ParamGroup α) := {}
  for g in groups do
    for id in g.params do
      if m.contains id then
        throw (tagError "optim" s!"param id {id} appears in multiple groups")
      else
        m := m.insert id g
  pure m

/--
Apply one optimizer step to a parameter table.

Inputs:
- `opt` is the current optimizer state (including per-parameter buffers),
- `params` is the current parameter table,
- `grads` maps parameter ids to gradients (as produced by autograd).

Behavior:
- applies LR schedulers (if configured) per group,
- shape-checks gradients and state buffers against each parameter,
- updates per-parameter state buffers (momentum / Adam m,v / accumulators),
- returns the updated optimizer state and an updated parameter table.
-/
def step
  (opt : OptimizerState α)
  (params : ParamTable α)
  (grads : Std.HashMap Nat (Runtime.AnyTensor α)) : Result (OptimizerState α × ParamTable α) := do
  let groups' := updateGroupSchedulers opt.groups
  let gmap <- groupMap groups'
  let tNext := opt.step + 1
  let mut momentum_buf := opt.momentum_buf
  let mut m := opt.m
  let mut v := opt.v
  let mut acc := opt.acc
  let mut acc2 := opt.acc2

  let mut updated : List (ParamEntry α) := []
  for p in params do
    let g ← match gmap.get? p.id with
      | some g => pure g
      | none =>
          throw (tagError "optim" s!"no param group for id {p.id}")

    let pval := p.value
    let gradOpt := grads.get? p.id
    match gradOpt with
    | none =>
        updated := { p with value := pval } :: updated
    | some gAny =>
        if h : gAny.s = pval.s then
          let grad : Tensor α pval.s := Tensor.castShape gAny.t h
          let param : Tensor α pval.s := pval.t
          match opt.kind with
          | .sgd =>
              let gradWD := addWeightDecay param grad g.weight_decay
              let state : _root_.Optim.SGD.State α pval.s := { lr := g.lr }
              let param' := Tensor.materialize <|
                _root_.Optim.SGD.update (α := α) (s := pval.s) state param gradWD
              updated := { p with value := AnyTensor.mk param' } :: updated
          | .momentum =>
              let v0 := getOrInit momentum_buf p.id pval
              let v0t ← castState "optim" p.id v0 pval
              let gradWD := addWeightDecay param grad g.weight_decay
              let gradDamped := scaleSpec gradWD (1 - g.dampening)
              let momentumState : _root_.Optim.MomentumSGD.State α pval.s :=
                { lr := g.lr, momentum := g.momentum, buf := v0t }
              let (momentumState', paramClassic) :=
                _root_.Optim.MomentumSGD.update (α := α) (s := pval.s)
                  momentumState param gradDamped
              let updateDir :=
                if g.nesterov then
                  addSpec gradDamped (scaleSpec momentumState'.buf g.momentum)
                else
                  momentumState'.buf
              let param' :=
                if g.nesterov then
                  subSpec param (scaleSpec updateDir g.lr)
                else
                  paramClassic
              let v' := momentumState'.buf
              momentum_buf := momentum_buf.insert p.id (AnyTensor.mk (Tensor.materialize v'))
              updated := { p with value := AnyTensor.mk (Tensor.materialize param') } :: updated
          | .adagrad =>
              let acc0 := getOrInit acc p.id pval
              let acc0t ← castState "optim" p.id acc0 pval
              let gradWD := addWeightDecay param grad g.weight_decay
              let state : _root_.Optim.AdaGrad.State α pval.s :=
                { lr := g.lr, epsilon := g.epsilon, accumulator := acc0t }
              let (state', param') :=
                _root_.Optim.AdaGrad.update (α := α) (s := pval.s) state param gradWD
              let acc' := state'.accumulator
              acc := acc.insert p.id (AnyTensor.mk (Tensor.materialize acc'))
              updated := { p with value := AnyTensor.mk (Tensor.materialize param') } :: updated
          | .rmsprop =>
              let acc0 := getOrInit acc p.id pval
              let acc0t ← castState "optim" p.id acc0 pval
              let gradWD := addWeightDecay param grad g.weight_decay
              let state : _root_.Optim.RMSProp.State α pval.s :=
                { lr := g.lr, decay := g.rho, epsilon := g.epsilon, accumulator := acc0t }
              let (state', param') :=
                _root_.Optim.RMSProp.update (α := α) (s := pval.s) state param gradWD
              let acc' := state'.accumulator
              acc := acc.insert p.id (AnyTensor.mk (Tensor.materialize acc'))
              updated := { p with value := AnyTensor.mk (Tensor.materialize param') } :: updated
          | .adam =>
              let m0 := getOrInit m p.id pval
              let v0 := getOrInit v p.id pval
              let m0t ← castState "optim" p.id m0 pval
              let v0t ← castState "optim" p.id v0 pval
              let gradWD := addWeightDecay param grad g.weight_decay
              let state : _root_.Optim.Adam.State α pval.s :=
                { lr := g.lr
                  beta1 := g.beta1
                  beta2 := g.beta2
                  epsilon := g.epsilon
                  m := m0t
                  v := v0t
                  t := adamPreviousStep tNext }
              let (state', param') :=
                _root_.Optim.Adam.update (α := α) (s := pval.s) state param gradWD
              let m' := state'.m
              let v' := state'.v
              m := m.insert p.id (AnyTensor.mk (Tensor.materialize m'))
              v := v.insert p.id (AnyTensor.mk (Tensor.materialize v'))
              updated := { p with value := AnyTensor.mk (Tensor.materialize param') } :: updated
          | .adamw =>
              let m0 := getOrInit m p.id pval
              let v0 := getOrInit v p.id pval
              let m0t ← castState "optim" p.id m0 pval
              let v0t ← castState "optim" p.id v0 pval
              let state : _root_.Optim.AdamW.State α pval.s :=
                { lr := g.lr
                  beta1 := g.beta1
                  beta2 := g.beta2
                  epsilon := g.epsilon
                  weight_decay := g.weight_decay
                  m := m0t
                  v := v0t
                  t := adamPreviousStep tNext }
              let (state', param') :=
                _root_.Optim.AdamW.update (α := α) (s := pval.s) state param grad
              let m' := state'.m
              let v' := state'.v
              m := m.insert p.id (AnyTensor.mk (Tensor.materialize m'))
              v := v.insert p.id (AnyTensor.mk (Tensor.materialize v'))
              updated := { p with value := AnyTensor.mk (Tensor.materialize param') } :: updated
          | .adadelta =>
              let acc0 := getOrInit acc p.id pval
              let acc20 := getOrInit acc2 p.id pval
              let acc0t ← castState "optim" p.id acc0 pval
              let acc20t ← castState "optim" p.id acc20 pval
              let gradWD := addWeightDecay param grad g.weight_decay
              let state : _root_.Optim.Adadelta.State α pval.s :=
                { lr := g.lr, rho := g.rho, epsilon := g.epsilon, v := acc0t, u := acc20t }
              let (state', param') :=
                _root_.Optim.Adadelta.update (α := α) (s := pval.s) state param gradWD
              let acc' := state'.v
              let acc2' := state'.u
              acc := acc.insert p.id (AnyTensor.mk (Tensor.materialize acc'))
              acc2 := acc2.insert p.id (AnyTensor.mk (Tensor.materialize acc2'))
              updated := { p with value := AnyTensor.mk (Tensor.materialize param') } :: updated
        else
          throw (tagError "optim" s!"gradient shape mismatch for id {p.id}")

  let opt' : OptimizerState α :=
    { kind := opt.kind
    , groups := groups'
    , step := tNext
    , momentum_buf := momentum_buf
    , m := m
    , v := v
    , acc := acc
    , acc2 := acc2
    }
  pure (opt', updated.reverse)

end Optim

end Train
end Autograd
end Runtime
