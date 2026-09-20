/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core
public import NN.Runtime.Optim.Optimizers
public import NN.Runtime.Optim.Schedulers.Native
public import NN.Runtime.Optim.Schedulers.PyTorch

/-!
# Optimizer integration for Runtime.Autograd

This module is the training-loop side of autograd: it takes a gradient map produced by
`Runtime.Autograd` and applies parameter updates.

PyTorch analogy:
- `ParameterTable` is like an ordered list of parameters, but we key everything by a stable `Nat` id
  (closer to `state_dict` keys than pointer identity).
- `ParameterGroup` and `OptimizerState` mirror `torch.optim.Optimizer` parameter groups and state.
- `LearningRateScheduler` is a small wrapper around our scheduler implementations, similar to
  `torch.optim.lr_scheduler.*`.

All updates are *shape checked* and implemented using the pure `Spec` tensor operators, so they
can be used in eager execution or lowered into a typed graph.

Formula ownership:
- this file owns the heterogeneous parameter-table handling, parameter groups, lazily created
  per-parameter buffers, scheduler stepping, and PyTorch-style coupled weight decay at the
  training-loop boundary;
- `NN.Runtime.Optim.Optimizers` owns the canonical per-tensor optimizer equations.

The important rule is: this file must not define a second public optimizer-formula surface. Each
`Optimizer.Internal.<algorithm>` function below constructs the canonical optimizer state for one
parameter, typed by that parameter's shape, and calls `NN.Runtime.Optim.Optimizers` directly. The
only local algebra left here is training-loop glue that is not represented by the canonical pure
states, such as coupled weight-decay preprocessing and PyTorch-style momentum dampening/Nesterov
handling.

Per-parameter buffers are stored shape-erased (`ParameterState`) because the parameter table is
heterogeneous; `ParameterState.cast` is the single place where the stored shape is checked against
the live parameter shape.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

open Spec TorchLean
open TorchLean TorchLean.Tensor

/-!
## Parameter table
-/
/-!
The declarations below provide the parameter registry used by the training loop.

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
structure Parameter (α : Type) [TorchLean.Storage α] where
  /-- Stable identifier used to key gradients and optimizer state. -/
  id : Nat
  /-- Optional label, such as a module path; used only for reporting and debugging. -/
  name : Option String := none
  /-- The shape-erased parameter value. -/
  value : Spec.SomeTensor α

/-- A flat runtime array of parameters used by the training loop. -/
abbrev ParameterTable (α : Type) [TorchLean.Storage α] := Array (Parameter α)

namespace Parameter

/-!
### Constructors
-/
/--
Create a `ParamEntry` from a typed tensor.

This is mostly a convenience for assembling a `ParamTable` from known-shaped tensors.
-/
def create {α : Type} [TorchLean.Storage α] {s : Shape}
    (id : Nat) (t : Tensor α s) (name : Option String := none) :
  Parameter α :=
  { id := id, name := name, value := Spec.SomeTensor.ofTensor t }

end Parameter

namespace ParameterTable

variable {α : Type} [TorchLean.Storage α]

/-- Array of ids for membership checks. -/
def ids (parameters : ParameterTable α) : Array Nat :=
  parameters.map (·.id)

/-- Find a parameter entry by id. -/
def find? (parameters : ParameterTable α) (id : Nat) : Option (Parameter α) :=
  Array.find? (fun parameter => parameter.id == id) parameters

/-- Get a typed tensor from the table, with shape checking. -/
def get {α : Type} [TorchLean.Storage α] {s : Shape}
  (tag : String) (parameters : ParameterTable α) (id : Nat) : Result (Tensor α s) := by
  match find? parameters id with
  | none =>
      exact .error (tagError tag s!"missing param id {id}")
  | some parameter =>
      if h : parameter.value.shape = s then
        exact .ok (Tensor.castShape parameter.value.tensor h)
      else
        exact .error (tagError tag s!"param shape mismatch for id {id}")

/-- Replace a parameter entry value by id. -/
def set (parameters : ParameterTable α) (id : Nat)
    (value : Spec.SomeTensor α) : ParameterTable α :=
  parameters.map
    (fun parameter => if parameter.id = id then { parameter with value := value } else parameter)

/--
Build the set of parameter identifiers, rejecting duplicate ids.

Optimizer buffers and gradients are keyed by `id`, so accepting two table entries with the same id
would make both entries consume one gradient and mutate one shared optimizer state.
-/
def Internal.checkedIdSet
    (parameters : ParameterTable α) : Result (Std.HashMap Nat Unit) := do
  let mut ids : Std.HashMap Nat Unit := {}
  for parameter in parameters do
    if ids.contains parameter.id then
      throw (tagError "optim" s!"duplicate parameter id {parameter.id}")
    else
      ids := ids.insert parameter.id ()
  pure ids

end ParameterTable

/-!
## Scheduler wrapper
-/
/--
Learning-rate scheduler wrapper used by the training loop.

PyTorch analogy: this plays the role of `torch.optim.lr_scheduler.*` objects, except we keep the
state as an inductive value and expose a pure `current`/`advance` API.

The `torch*` constructors wrap the schedules from `Optim.Scheduler.PyTorch`, whose phase and
step-count conventions follow PyTorch exactly where the native schedules deliberately differ.
-/
inductive LearningRateScheduler (α : Type) where
  | constant : Optim.Scheduler.Constant α -> LearningRateScheduler α
  | exponential : Optim.Scheduler.ExponentialDecay α -> LearningRateScheduler α
  | step : Optim.Scheduler.StepDecay α -> LearningRateScheduler α
  | cosine : Optim.Scheduler.CosineAnnealing α -> LearningRateScheduler α
  | linearWarmup : Optim.Scheduler.LinearWarmup α -> LearningRateScheduler α
  | warmupCosine : Optim.Scheduler.WarmupCosine α -> LearningRateScheduler α
  | cyclic : Optim.Scheduler.Cyclic α -> LearningRateScheduler α
  | triangular : Optim.Scheduler.TriangularCycle α -> LearningRateScheduler α
  | oneCycle : Optim.Scheduler.OneCycle α -> LearningRateScheduler α
  | rangeTest : Optim.Scheduler.RangeTest α -> LearningRateScheduler α
  /--
  PyTorch `CosineAnnealingLR`: the cosine continues past `T_max` with period `2 * T_max` instead of
  clamping at the minimum learning rate as the native `cosine` schedule does.
  -/
  | torchCosineAnnealing : Optim.Scheduler.PyTorch.CosineAnnealing α -> LearningRateScheduler α
  /--
  PyTorch `OneCycleLR` (learning rate only): fractional phase endpoints `pct_start * total - 1`,
  `min_lr = initial_lr / final_div_factor`, cosine or linear annealing, and the optional
  three-phase variant. The native `oneCycle` uses linear ramps with a `max_lr / final_div_factor`
  floor.
  -/
  | torchOneCycle : Optim.Scheduler.PyTorch.OneCycle α -> LearningRateScheduler α
  /--
  Custom schedule with an explicit step counter.

`custom f k` means "use learning rate `f k` for this step, and increment to `k+1` on `advance`".
  -/
  | custom : (Nat -> α) -> Nat -> LearningRateScheduler α

namespace LearningRateScheduler

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Read current learning rate from the scheduler state. -/
def current : LearningRateScheduler α -> α
  | constant s => s.current
  | exponential s => s.current
  | step s => s.current
  | cosine s => s.current
  | linearWarmup s => s.current
  | warmupCosine s => s.current
  | cyclic s => s.current
  | triangular s => s.current
  | oneCycle s => s.current
  | rangeTest s => s.current
  | torchCosineAnnealing s => s.current
  | torchOneCycle s => s.current
  | custom f k => f k

/-- Advance scheduler state by one step. -/
def advance : LearningRateScheduler α -> LearningRateScheduler α
  | constant s => constant s.advance
  | exponential s => exponential s.advance
  | step s => step s.advance
  | cosine s => cosine s.advance
  | linearWarmup s => linearWarmup s.advance
  | warmupCosine s => warmupCosine s.advance
  | cyclic s => cyclic s.advance
  | triangular s => triangular s.advance
  | oneCycle s => oneCycle s.advance
  | rangeTest s => rangeTest s.advance
  | torchCosineAnnealing s => torchCosineAnnealing s.advance
  | torchOneCycle s => torchOneCycle s.advance
  | custom f k => custom f (k + 1)

end LearningRateScheduler

/-!
## Optimizer configuration
-/
/--
Which optimizer update rule to apply.

PyTorch analogy: these correspond approximately to `torch.optim.SGD`, `Adam`, `AdamW`, etc.
-/
inductive OptimizerAlgorithm
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
structure ParameterGroup (α : Type) [Context α] where
  /-- Parameter ids that belong to this group. -/
  parameterIds : Array Nat
  /-- Base learning rate (possibly overridden by `scheduler` on each step). -/
  learningRate : α
  /-- $\ell_2$ regularization coefficient (behavior depends on the optimizer kind; see AdamW). -/
  weightDecay : α := 0
  /-- Momentum factor (SGD with momentum). -/
  momentum : α := 0
  /-- Dampening for momentum updates. -/
  dampening : α := 0
  /-- Use Nesterov variant for momentum updates. -/
  nesterov : Bool := false
  /-- Adam beta1 parameter (exponential decay for the first moment). -/
  beta1 : α := 1 - (1 / 10)
  /-- Adam beta2 parameter (exponential decay for the second moment). -/
  beta2 : α :=
    1 - (1 / (10 * 10 * 10))
  /-- Numerical stability term used by adaptive optimizers. -/
  epsilon : α := Context.defaultEpsilon
  /-- "Rho" decay parameter for RMSProp/AdaDelta style optimizers. -/
  rho : α := 1 - (1 / 10)
  /-- Optional learning-rate scheduler for this group. -/
  scheduler : Option (LearningRateScheduler α) := none

/-!
## Per-parameter optimizer buffers
-/
/--
Optimizer buffers for one parameter of shape `s`.

Each constructor belongs to one algorithm family, so the buffers are typed by the parameter shape
and there is no separate map per buffer kind. Buffers stored by a different family (for example
after switching `algorithm` on a restored state) are treated as absent and re-initialised lazily,
which is what the previous per-kind maps did implicitly.
-/
inductive ParameterBuffers (α : Type) [TorchLean.Storage α] (s : Shape) where
  /-- SGD with momentum or Nesterov: the velocity buffer. -/
  | momentum (buffer : Tensor α s)
  /-- AdaGrad and RMSProp: the squared-gradient accumulator. -/
  | squaredGradient (accumulator : Tensor α s)
  /-- Adam and AdamW: the parameter-local step count and both moment estimates. -/
  | adam (stepCount : Nat) (firstMoment secondMoment : Tensor α s)
  /-- Adadelta: squared-gradient and squared-update running averages. -/
  | adadelta (squaredGradientAverage squaredUpdateAverage : Tensor α s)

/--
Shape-erased optimizer buffers for one parameter.

The parameter table is heterogeneous, so buffers are stored with their runtime shape and cast back
to the live parameter shape by `ParameterState.cast`.
-/
structure ParameterState (α : Type) [TorchLean.Storage α] where
  /-- Shape of the parameter the buffers belong to. -/
  shape : Shape
  /-- The typed buffers. -/
  buffers : ParameterBuffers α shape

namespace ParameterState

variable {α : Type} [TorchLean.Storage α]

/-- Erase the shape of typed buffers. -/
def ofBuffers {s : Shape} (buffers : ParameterBuffers α s) : ParameterState α :=
  { shape := s, buffers := buffers }

/--
Recover typed buffers for a parameter of shape `s`.

This is the only shape cast on optimizer state. It fails when a checkpoint is reloaded into a model
whose parameter `id` now has a different shape.
-/
def cast (id : Nat) (state : ParameterState α) (s : Shape) : Result (ParameterBuffers α s) :=
  if h : state.shape = s then
    pure (h ▸ state.buffers)
  else
    throw (tagError "optim" s!"state shape mismatch for id {id}")

end ParameterState

/--
Full optimizer state used by the training loop.

This mirrors PyTorch's optimizer state:
- a global optimizer-call counter,
- hyperparameter groups, and
- per-parameter buffers keyed by parameter id (`Nat`), including the parameter-local Adam step.
-/
structure OptimizerState (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Which update rule to apply on `step`. -/
  algorithm : OptimizerAlgorithm
  /-- Parameter groups (hyperparameters + membership). -/
  parameterGroups : Array (ParameterGroup α)
  /-- Global optimizer-call counter (increments once per `step`, including sparse steps). -/
  stepCount : Nat := 0
  /--
  Per-parameter buffers keyed by parameter id.

  Adam bias correction is parameter-local: a parameter whose gradient is absent does not advance
  its moment step. The step lives inside `ParameterBuffers.adam`, so it cannot drift from the
  moments it corrects.
  -/
  parameterStates : Std.HashMap Nat (ParameterState α) := {}

/--
A pure state snapshot for saving/restoring optimizer state.

PyTorch analogy: this is the data carried by `optimizer.state_dict()` (modulo naming/layout).
We use association lists instead of `HashMap` so the result is deterministic and easy to serialize.
-/
structure OptimizerSnapshot (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Optimizer algorithm used to interpret the stored buffers. -/
  algorithm : OptimizerAlgorithm
  /-- Global optimizer step at the time the snapshot was taken. -/
  stepCount : Nat
  /-- Parameter groups, including scheduler state and hyperparameters. -/
  parameterGroups : Array (ParameterGroup α)
  /-- Per-parameter buffers keyed by parameter id. -/
  parameterStates : Array (Nat × ParameterState α)

namespace OptimizerState

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Serialize optimizer state to a pure record.

PyTorch analogy: this is the "export" step for `state_dict()`.
-/
def snapshot (optimizerState : OptimizerState α) : OptimizerSnapshot α :=
  { algorithm := optimizerState.algorithm
  , stepCount := optimizerState.stepCount
  , parameterGroups := optimizerState.parameterGroups
  , parameterStates := optimizerState.parameterStates.toArray
  }

/--
Restore optimizer state from a state dict.

PyTorch analogy: this is the "import" step for `load_state_dict(...)`.
-/
def restore (snapshot : OptimizerSnapshot α) : OptimizerState α :=
  { algorithm := snapshot.algorithm
  , stepCount := snapshot.stepCount
  , parameterGroups := snapshot.parameterGroups
  , parameterStates := Std.HashMap.ofArray snapshot.parameterStates
  }

/-- Number of Adam/AdamW updates applied to parameter `id`, if it has Adam-family buffers. -/
def parameterStepCount? (optimizerState : OptimizerState α) (id : Nat) : Option Nat :=
  match optimizerState.parameterStates.get? id with
  | some ⟨_, .adam stepCount _ _⟩ => some stepCount
  | _ => none

end OptimizerState

/-!
## Optimizer step
-/
namespace Optimizer

variable {α : Type} [TorchLean.Storage α] [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)]

namespace Internal

variable {s : Shape}

/--
Add an $\ell_2$ regularization term to the gradient:
$g+\operatorname{weightDecay}\,\operatorname{param}$.

Note: this is the *coupled* weight decay used by classic SGD-style updates.
For AdamW the integration step delegates to the canonical optimizer's decoupled update.
-/
def addWeightDecay (parameters gradients : Tensor α s) (weightDecay : α) : Tensor α s :=
  addSpec gradients (scaleSpec parameters weightDecay)

/-- Zero buffer used when a parameter has no stored state yet (PyTorch initialises lazily). -/
def zeroBuffer (s : Shape) : Tensor α s :=
  Tensor.full s (0 : α)

/-- Plain SGD: `p - lr * (g + wd * p)`. Stateless. -/
def sgd (group : ParameterGroup α) (parameters gradient : Tensor α s) : Tensor α s :=
  let gradientWithDecay := addWeightDecay parameters gradient group.weightDecay
  let state : Optim.SGD.State α s := { learningRate := group.learningRate }
  (Optim.SGD.update (α := α) (s := s) state parameters gradientWithDecay).parameters

/--
SGD with momentum, PyTorch conventions.

The momentum buffer is initialised from the first raw (decayed) gradient; dampening is applied only
once a buffer already exists. Nesterov uses `g + momentum * buffer` as the update direction.
-/
def momentum (group : ParameterGroup α) (buffers : Option (ParameterBuffers α s))
    (parameters gradient : Tensor α s) : Tensor α s × ParameterBuffers α s :=
  let previousBuffer? : Option (Tensor α s) :=
    match buffers with
    | some (.momentum buffer) => some buffer
    | _ => none
  let gradientWithDecay := addWeightDecay parameters gradient group.weightDecay
  let bufferGradient :=
    match previousBuffer? with
    | none => gradientWithDecay
    | some _ => scaleSpec gradientWithDecay (1 - group.dampening)
  let state : Optim.MomentumSGD.State α s :=
    { learningRate := group.learningRate
      momentum := group.momentum
      momentumBuffer := previousBuffer?.getD (zeroBuffer s) }
  let result :=
    Optim.MomentumSGD.update (α := α) (s := s) state parameters bufferGradient
  let nextParameters :=
    if group.nesterov then
      let updateDirection :=
        addSpec gradientWithDecay (scaleSpec result.optimizerState.momentumBuffer group.momentum)
      subSpec parameters (scaleSpec updateDirection group.learningRate)
    else
      result.parameters
  (nextParameters, .momentum result.optimizerState.momentumBuffer)

/-- Squared-gradient accumulator shared by AdaGrad and RMSProp, or zeros when absent. -/
def squaredGradientBuffer (buffers : Option (ParameterBuffers α s)) : Tensor α s :=
  match buffers with
  | some (.squaredGradient accumulator) => accumulator
  | _ => zeroBuffer s

/-- AdaGrad with coupled weight decay. -/
def adagrad (group : ParameterGroup α) (buffers : Option (ParameterBuffers α s))
    (parameters gradient : Tensor α s) : Tensor α s × ParameterBuffers α s :=
  let gradientWithDecay := addWeightDecay parameters gradient group.weightDecay
  let state : Optim.AdaGrad.State α s :=
    { learningRate := group.learningRate
      epsilon := group.epsilon
      squaredGradientSum := squaredGradientBuffer buffers }
  let result := Optim.AdaGrad.update (α := α) (s := s) state parameters gradientWithDecay
  (result.parameters, .squaredGradient result.optimizerState.squaredGradientSum)

/-- RMSProp with coupled weight decay. -/
def rmsprop (group : ParameterGroup α) (buffers : Option (ParameterBuffers α s))
    (parameters gradient : Tensor α s) : Tensor α s × ParameterBuffers α s :=
  let gradientWithDecay := addWeightDecay parameters gradient group.weightDecay
  let state : Optim.RMSProp.State α s :=
    { learningRate := group.learningRate
      decay := group.rho
      epsilon := group.epsilon
      squaredGradientAverage := squaredGradientBuffer buffers }
  let result := Optim.RMSProp.update (α := α) (s := s) state parameters gradientWithDecay
  (result.parameters, .squaredGradient result.optimizerState.squaredGradientAverage)

/-- Adam-family buffers `(stepCount, firstMoment, secondMoment)`, zero-initialised when absent. -/
def adamBuffers (buffers : Option (ParameterBuffers α s)) : Nat × Tensor α s × Tensor α s :=
  match buffers with
  | some (.adam stepCount firstMoment secondMoment) => (stepCount, firstMoment, secondMoment)
  | _ => (0, zeroBuffer s, zeroBuffer s)

/-- Adam with coupled weight decay and parameter-local bias correction. -/
def adam (group : ParameterGroup α) (buffers : Option (ParameterBuffers α s))
    (parameters gradient : Tensor α s) : Tensor α s × ParameterBuffers α s :=
  let (stepCount, firstMoment, secondMoment) := adamBuffers buffers
  let gradientWithDecay := addWeightDecay parameters gradient group.weightDecay
  let state : Optim.Adam.State α s :=
    { learningRate := group.learningRate
      beta1 := group.beta1
      beta2 := group.beta2
      epsilon := group.epsilon
      firstMoment := firstMoment
      secondMoment := secondMoment
      stepCount := stepCount }
  let result := Optim.Adam.update (α := α) (s := s) state parameters gradientWithDecay
  (result.parameters,
    .adam result.optimizerState.stepCount result.optimizerState.firstMoment
      result.optimizerState.secondMoment)

/-- AdamW: decoupled weight decay handled by the canonical update, no gradient preprocessing. -/
def adamw (group : ParameterGroup α) (buffers : Option (ParameterBuffers α s))
    (parameters gradient : Tensor α s) : Tensor α s × ParameterBuffers α s :=
  let (stepCount, firstMoment, secondMoment) := adamBuffers buffers
  let state : Optim.AdamW.State α s :=
    { learningRate := group.learningRate
      beta1 := group.beta1
      beta2 := group.beta2
      epsilon := group.epsilon
      weightDecay := group.weightDecay
      firstMoment := firstMoment
      secondMoment := secondMoment
      stepCount := stepCount }
  let result := Optim.AdamW.update (α := α) (s := s) state parameters gradient
  (result.parameters,
    .adam result.optimizerState.stepCount result.optimizerState.firstMoment
      result.optimizerState.secondMoment)

/-- Adadelta with coupled weight decay. -/
def adadelta (group : ParameterGroup α) (buffers : Option (ParameterBuffers α s))
    (parameters gradient : Tensor α s) : Tensor α s × ParameterBuffers α s :=
  let (squaredGradientAverage, squaredUpdateAverage) :=
    match buffers with
    | some (.adadelta squaredGradientAverage squaredUpdateAverage) =>
        (squaredGradientAverage, squaredUpdateAverage)
    | _ => (zeroBuffer s, zeroBuffer s)
  let gradientWithDecay := addWeightDecay parameters gradient group.weightDecay
  let state : Optim.Adadelta.State α s :=
    { learningRate := group.learningRate
      rho := group.rho
      epsilon := group.epsilon
      squaredGradientAverage := squaredGradientAverage
      squaredUpdateAverage := squaredUpdateAverage }
  let result := Optim.Adadelta.update (α := α) (s := s) state parameters gradientWithDecay
  (result.parameters,
    .adadelta result.optimizerState.squaredGradientAverage
      result.optimizerState.squaredUpdateAverage)

/--
Apply the configured update rule to one parameter.

Returns the new parameter value and the buffers to store for it (`none` for stateless SGD, which
leaves any stored buffers untouched).
-/
def updateParameter (algorithm : OptimizerAlgorithm) (group : ParameterGroup α)
    (buffers : Option (ParameterBuffers α s)) (parameters gradient : Tensor α s) :
    Tensor α s × Option (ParameterBuffers α s) :=
  match algorithm with
  | .sgd => (sgd group parameters gradient, none)
  | .momentum => let r := momentum group buffers parameters gradient; (r.1, some r.2)
  | .adagrad => let r := adagrad group buffers parameters gradient; (r.1, some r.2)
  | .rmsprop => let r := rmsprop group buffers parameters gradient; (r.1, some r.2)
  | .adam => let r := adam group buffers parameters gradient; (r.1, some r.2)
  | .adamw => let r := adamw group buffers parameters gradient; (r.1, some r.2)
  | .adadelta => let r := adadelta group buffers parameters gradient; (r.1, some r.2)

/-- Shape-check a gradient delivered as a shape-erased tensor against its parameter. -/
def castGradient (id : Nat) (gradient : Spec.SomeTensor α) (s : Shape) : Result (Tensor α s) :=
  if h : gradient.shape = s then
    pure (Tensor.castShape gradient.tensor h)
  else
    throw (tagError "optim" s!"gradient shape mismatch for id {id}")

/--
Update each group's learning rate from its scheduler (if present) and advance the scheduler state.

This matches the common training-loop pattern: "read LR, then call `scheduler.step()`".
-/
def advanceSchedulers (parameterGroups : Array (ParameterGroup α)) : Array (ParameterGroup α) :=
  parameterGroups.map (fun group =>
    let learningRate := match group.scheduler with
      | none => group.learningRate
      | some scheduler => LearningRateScheduler.current scheduler
    let scheduler := group.scheduler.map LearningRateScheduler.advance
    { group with learningRate := learningRate, scheduler := scheduler })

/--
Build a map from parameter id to its `ParameterGroup`.

Fails if an id appears in multiple groups (PyTorch also disallows overlapping param groups).
-/
def parameterGroupMap (parameterGroups : Array (ParameterGroup α)) :
    Result (Std.HashMap Nat (ParameterGroup α)) := do
  let mut groupByParameterId : Std.HashMap Nat (ParameterGroup α) := {}
  for group in parameterGroups do
    for id in group.parameterIds do
      if groupByParameterId.contains id then
        throw (tagError "optim" s!"param id {id} appears in multiple groups")
      else
        groupByParameterId := groupByParameterId.insert id group
  pure groupByParameterId

end Internal

/-- Updated dynamic optimizer state and parameter table produced by one step. -/
structure Step (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Optimizer state after applying the gradients. -/
  optimizerState : OptimizerState α
  /-- Updated parameter tensors. -/
  parameters : ParameterTable α

/--
Apply one optimizer step to a parameter table.

Inputs:
- `optimizerState` is the current optimizer state (including per-parameter buffers),
- `parameters` is the current parameter table,
- `gradients` maps parameter ids to gradients (as produced by autograd).

Behavior:
- rejects duplicate parameter ids, overlapping groups, and group ids missing from the table,
- applies learning-rate schedulers (if configured) per group,
- shape-checks gradients and stored buffers against each parameter,
- updates per-parameter buffers (momentum, Adam moments, and accumulators),
- returns the updated optimizer state and an updated parameter table.

Parameters without a gradient are left untouched, including their buffers.
-/
def step
    (optimizerState : OptimizerState α)
    (parameters : ParameterTable α)
    (gradients : Std.HashMap Nat (Spec.SomeTensor α)) : Result (Step α) := do
  let parameterIds ← ParameterTable.Internal.checkedIdSet parameters
  let parameterGroups := Internal.advanceSchedulers optimizerState.parameterGroups
  let groupByParameterId ← Internal.parameterGroupMap parameterGroups
  for (id, _) in groupByParameterId.toList do
    if !parameterIds.contains id then
      throw (tagError "optim" s!"parameter group references unknown id {id}")
  let mut parameterStates := optimizerState.parameterStates
  let mut updatedParameters : ParameterTable α := #[]
  for parameter in parameters do
    let group ← match groupByParameterId.get? parameter.id with
      | some group => pure group
      | none =>
          throw (tagError "optim" s!"no parameter group for id {parameter.id}")
    match gradients.get? parameter.id with
    | none =>
        updatedParameters := updatedParameters.push parameter
    | some packedGradient =>
        let s := parameter.value.shape
        let gradient ← Internal.castGradient parameter.id packedGradient s
        let buffers ← match parameterStates.get? parameter.id with
          | none => pure none
          | some state => some <$> state.cast parameter.id s
        let (nextValue, nextBuffers) :=
          Internal.updateParameter optimizerState.algorithm group buffers
            parameter.value.tensor gradient
        if let some nextBuffers := nextBuffers then
          parameterStates :=
            parameterStates.insert parameter.id (ParameterState.ofBuffers nextBuffers)
        updatedParameters := updatedParameters.push
          { parameter with value := Spec.SomeTensor.ofTensor nextValue }
  let nextOptimizerState : OptimizerState α :=
    { algorithm := optimizerState.algorithm
    , parameterGroups := parameterGroups
    , stepCount := optimizerState.stepCount + 1
    , parameterStates := parameterStates
    }
  pure { optimizerState := nextOptimizerState, parameters := updatedParameters }

end Optimizer

end Train
end Autograd
end Runtime
