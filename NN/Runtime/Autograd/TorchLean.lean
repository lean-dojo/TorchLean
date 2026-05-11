/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Autodiff
public import NN.Runtime.Autograd.TorchLean.Backend
public import NN.Runtime.Autograd.TorchLean.Fft
public import NN.Runtime.Autograd.TorchLean.Fno1d
public import NN.Runtime.Autograd.TorchLean.Functional
public import NN.Runtime.Autograd.TorchLean.Loss
public import NN.Runtime.Autograd.TorchLean.Metrics
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.Runtime.Autograd.TorchLean.NN
public import NN.Runtime.Autograd.TorchLean.Norm
public import NN.Runtime.Autograd.TorchLean.Optim
public import NN.Runtime.Autograd.TorchLean.Session
public import NN.Runtime.Autograd.TorchLean.Training

/-!
# TorchLean

TorchLean is the runtime front-end for training and execution.

This module is the user-facing wrapper around the lower-level runtime session implementation:
- write a model/loss once over a small `Ops` interface,
- choose `backend := .eager` (dynamic tape) or `backend := .compiled` (typed SSA/DAG),
- run `forward`, `backward`, and `step` with the same call shape.

`Runtime.Autograd.TorchLean` is the stable runtime namespace re-exported by `NN.API.Runtime`.
`Runtime.Autograd.Torch` remains available as the lower-level session layer used internally by
TorchLean and by linked compiled sessions.

This umbrella deliberately does **not** own model catalogs or RL objectives. Reusable architecture
specifications live under `NN.GraphSpec.Models.TorchLean`, while differentiable PPO / actor-critic
loss helpers live under `NN.Runtime.RL.PolicyGradient.Autograd`. Keeping those out of the runtime
core makes the dependency graph easier to audit: this folder should provide tensors, ops, modules,
sessions, losses, optim/training glue, and executable autodiff utilities.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

export _root_.Runtime.Autograd.Torch
  (TList
   TensorRef Param AnyParam
   CompiledScalar compileScalar
   CompiledOut compileOut
   ParamList ScalarTrainer scalarTrainer)

-- Unified imperative session (choose eager vs compiled at `new` time):
-- `TorchLean.Session` is defined in `NN.Runtime.Autograd.TorchLean.Session` and is available
-- automatically via the import above.

/-! ## Training helpers -/

export _root_.Runtime.Autograd.Torch
  (scalarOf tlist1 tlist2 tlist3 tlist4 trainCycleSGD meanLoss)

namespace Init
export _root_.Runtime.Autograd.Torch.Init (Scheme tensor xavierW kaimingW)
end Init

namespace Samples
export _root_.Runtime.Autograd.Torch.Samples (vec1 vec2 affine2)
end Samples

namespace ScalarTrainer

export _root_.Runtime.Autograd.Torch.ScalarTrainer (forwardT backwardT stepT)

end ScalarTrainer

/-! ## Optimizers -/
export _root_.Runtime.Autograd.TorchLean.Optim (StateList Optimizer)
export _root_.Runtime.Autograd.TorchLean.Optim
  (sgd momentumSGD adagrad rmsprop adam adamw adadelta projectedSGD muon)

/-! ## Module wrappers (PyTorch-style) -/
export _root_.Runtime.Autograd.TorchLean.Module
  (ScalarModuleDef ScalarModule)
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModule
  (create forward backward step initOptim stepWith params)
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef
  (instantiate)

end TorchLean
end Autograd
end Runtime
