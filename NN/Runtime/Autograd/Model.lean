/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.Runtime.Autograd.Model.Autodiff -- shake: keep
public import NN.Runtime.Autograd.Model.Program -- shake: keep
public import NN.Runtime.Autograd.Model.Fft -- shake: keep
public import NN.Runtime.Autograd.Model.Fno -- shake: keep
public import NN.Runtime.Autograd.Model.Functional -- shake: keep
public import NN.Runtime.Autograd.Model.Functional.EinsumDynamic -- shake: keep
public import NN.Runtime.Autograd.Model.Loss -- shake: keep
public import NN.Runtime.Autograd.Model.Metrics -- shake: keep
public import NN.Runtime.Autograd.Model.Module -- shake: keep
public import NN.Runtime.Autograd.Model.Layers -- shake: keep
public import NN.Runtime.Autograd.Model.Norm -- shake: keep
public import NN.Runtime.Autograd.Model.Optim -- shake: keep
public import NN.Runtime.Autograd.Model.Session -- shake: keep
public import NN.Runtime.Autograd.Model.Training -- shake: keep
public import NN.Runtime.Autograd.Model.VqVae -- shake: keep

/-!
# TorchLean

TorchLean is the runtime front-end for training and execution.

This module is the user-facing wrapper around the lower-level runtime session implementation:
- write a model/loss once over a small `Ops` interface,
- choose `execution := .eager` (dynamic tape) or `execution := .typedGraph` (typed SSA/DAG),
- run `forward`, `backward`, and `step` with the same call shape.

`Runtime.Autograd.Model` is the stable runtime namespace re-exported by `NN.API.Runtime`.
`Runtime.Autograd.Torch` remains available as the lower-level session layer used internally by
TorchLean and by typed graph sessions.

This umbrella does **not** own model catalogs or RL objectives. Runnable models live under
`TorchLean.nn.models`, proof-oriented graph descriptions live under `NN.GraphSpec.Models`, and
differentiable PPO / actor-critic loss helpers live under
`NN.Runtime.RL.PolicyGradient.Autograd`. Keeping those out of the runtime core makes the dependency
graph easier to audit: this folder should provide tensors, ops, modules, sessions, losses,
optim/training glue, and executable autodiff utilities.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

export Runtime.Autograd.Torch
  (TensorRef Param AnyParam
   ParamList ScalarTrainer scalarTrainer)

-- Unified imperative session (choose eager or typed graph execution at `new` time):
-- `TorchLean.Session` is defined in `NN.Runtime.Autograd.Model.Session` and is available
-- automatically via the import above.

/-! ## Training helpers -/

export Runtime.Autograd.Torch
  (trainCycleSGD meanLoss)

namespace Init
export Runtime.Autograd.Torch.Init (Scheme tensor xavierUniform kaimingUniform)
end Init

namespace ScalarTrainer

export Runtime.Autograd.Torch.ScalarTrainer
  (runLoss runDiff runGrad runStep)

end ScalarTrainer

/-! ## Optimizers -/
export Runtime.Autograd.Model.Optim (StateList Optimizer)
export Runtime.Autograd.Model.Optim
  (sgd momentumSGD adagrad rmsprop adam adamw adadelta projectedSGD muon)

end Model
end Autograd
end Runtime
