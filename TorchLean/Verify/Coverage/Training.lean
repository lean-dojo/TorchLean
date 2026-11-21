/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

-- shake: keep-all

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public import TorchLean.NN.Optimizer
public import TorchLean.NN.Optimizer.AdamW
public import TorchLean.NN.Optimizer.Config
public import TorchLean.NN.Optimizer.Constructor
public import TorchLean.NN.Optimizer.GD
public import TorchLean.NN.Optimizer.GD.Convergence
public import TorchLean.NN.Optimizer.GD.StrongConvexity
public import TorchLean.NN.Optimizer.Muon
public import TorchLean.NN.Optimizer.Muon.Certificates
public import TorchLean.NN.Optimizer.Muon.NewtonSchulz
public import TorchLean.NN.Optimizer.Muon.QR
public import TorchLean.NN.Optimizer.TensorOptimizer
public import TorchLean.NN.Optimizer.Update
public import TorchLean.NN.RL
public import TorchLean.NN.RL.Backup
public import TorchLean.NN.RL.Bandits
public import TorchLean.NN.RL.Boundary
public import TorchLean.NN.RL.Boundary.Json
public import TorchLean.NN.RL.Boundary.Soundness
public import TorchLean.NN.RL.CLI
public import TorchLean.NN.RL.DQN
public import TorchLean.NN.RL.DQN.Autograd
public import TorchLean.NN.RL.DQN.SoftUpdate
public import TorchLean.NN.RL.Environment
public import TorchLean.NN.RL.Environment.Safety
public import TorchLean.NN.RL.Eval
public import TorchLean.NN.RL.FiniteMDP
public import TorchLean.NN.RL.FiniteMDP.Bellman
public import TorchLean.NN.RL.Gymnasium.Client
public import TorchLean.NN.RL.Gymnasium.Session
public import TorchLean.NN.RL.MDP
public import TorchLean.NN.RL.MDP.Bellman
public import TorchLean.NN.RL.Numerics.Advantage
public import TorchLean.NN.RL.Numerics.Intervals
public import TorchLean.NN.RL.Numerics.PPO
public import TorchLean.NN.RL.Numerics.Returns
public import TorchLean.NN.RL.Numerics.Types
public import TorchLean.NN.RL.PPO.ActorCritic
public import TorchLean.NN.RL.PPO.Collect
public import TorchLean.NN.RL.PPO.Rollout
public import TorchLean.NN.RL.PPO.Training
public import TorchLean.NN.RL.PolicyGradient
public import TorchLean.NN.RL.PolicyGradient.Autograd
public import TorchLean.NN.RL.PolicyGradient.Real
public import TorchLean.NN.RL.Replay
public import TorchLean.NN.RL.Replay.Capacity
public import TorchLean.NN.RL.Session
public import TorchLean.NN.RL.StochasticMDP
public import TorchLean.NN.RL.StochasticMDP.Bellman
public import TorchLean.NN.RL.Tabular
public import TorchLean.NN.RL.Transition
public import TorchLean.NN.RL.ValueLearning

/-!
# Roundoff coverage of the reinforcement-learning and optimizer functions

`#roundoff_coverage` runs `roundoff` on every public function of the modules under
`TorchLean/NN/RL/` and `TorchLean/NN/Optimizer/` whose value is the scalar or a tensor, as
`every` lists them (`moduleFunctions`): the losses, objectives, targets and backups of the
reinforcement-learning algorithms and the steps of gradient descent. Each route is analyzed for the
configured scalar of an arbitrary format, for native `Float32` and for the rounded reals
`FP32`, so each theorem holds at every argument and shape. The functions of `TorchLean/NN/Loss/`
return a computation on the tape (`m (Ref ...)`), so `every` finds none there. This module imports
every module under the two roots, so that `every` sees each of their functions.

FloatLib's `#roundoff_check` evaluates a mixed theorem of this module at given inputs, as
`#roundoff_check RL.tdResidual.value_mixedError (value := (1.5 : Binary32)) ...` does. Its reports
cover those inputs only. At such inputs it shows that the bound of `RL.tdResidual` does not fix
the sign where `reward + gamma * nextValue` is close to `value`, and that the bound of
`RL.PolicyGradient.weightedExpm1x` is its underflow term where the term is below half the least
subnormal; `TorchLean/Verify/RuntimeApprox/README.md` gives the inputs.
-/

public section

namespace TorchLean.Coverage.Training

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

#roundoff_coverage
  reinforcement := every TorchLean.NN.RL
  optimizer := every TorchLean.NN.Optimizer

end TorchLean.Coverage.Training
