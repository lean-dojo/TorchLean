/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Runtime.RL.Replay
public import NN.Runtime.RL.Algorithms
public import NN.Runtime.RL.Boundary
public import NN.Runtime.RL.Session
public import NN.Runtime.RL.Eval
public import NN.Runtime.RL.Gymnasium
public import NN.Runtime.RL.PPO
public import NN.Runtime.RL.PolicyGradient.Autograd
public import NN.Runtime.RL.Numerics
public import NN.Runtime.RL.Artifacts.DefaultPaths

/-!
# Reinforcement Learning Runtime Entrypoint

This is the runtime umbrella for typed reinforcement-learning utilities. The modules here are
executable infrastructure: they define small MDP/bandit interfaces, value-learning and
policy-gradient helpers, replay buffers, checked rollout boundaries, Gymnasium subprocess sessions,
and PPO rollout infrastructure.

The split is intentional:
- `Runtime.RL.Core` contains tensor-shaped rollout/loss helpers shared by algorithms;
- `Runtime.RL.Replay` contains typed bounded replay buffers for off-policy algorithms;
- `Runtime.RL.Algorithms.*` contains pure/mostly-pure bandit, tabular, value-learning, and
  policy-gradient equations;
- `Runtime.RL.Boundary` records host-side rollout contracts before converting observations and
  rewards into TorchLean tensors;
- `Runtime.RL.Gymnasium` is an external-process bridge and therefore a trust boundary;
- `Runtime.RL.PolicyGradient.Autograd` contains differentiable actor/critic losses over TorchLean
  refs; model architectures remain in GraphSpec/API model layers;
- `Runtime.RL.PPO` contains rollout/sample construction;
- `Runtime.RL.Numerics.*` contains optional checked float32 and interval diagnostics for RL
  recursions.
-/

@[expose] public section
