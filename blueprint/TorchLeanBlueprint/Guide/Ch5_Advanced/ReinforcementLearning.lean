import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Reinforcement Learning Stack" =>
%%%
tag := "reinforcement-learning"
%%%

Reinforcement learning is a good test for TorchLean because the model is only one part of the
system. A policy interacts with an environment, records transitions, computes returns and
advantages, and updates actor and critic networks. In mainstream code, many of these objects live
as mutable Python arrays. TorchLean gives them Lean-side names.

We built the TorchLean stack differently. We still want the familiar pieces: Gymnasium
environments, actor critic networks, replay buffers, PPO returns and advantages, CPU/CUDA execution,
and editor visualizations. The difference is that each piece has a Lean side name. The
[runtime RL API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL.lean), [specification RL API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL.lean), and
[RL proofs API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Core.lean) are deliberately separate: the split lets us say
precisely what is proved, what is checked at runtime, and what is still an external assumption.

# The Stack In One Picture

The useful mental model has three pieces:

$$`\text{Gymnasium or Lean environment}
\;\to\; \text{checked transition boundary}
\;\to\; \text{TorchLean rollout tensors}`

$$`\text{Spec.RL MDP semantics}
\;\longleftrightarrow\; \text{Runtime.RL algorithms}
\;\longleftrightarrow\; \text{Proofs.RL theorems}`

The left side is the world of interaction. The middle is the executable training path. The right
side is the mathematical account of the same objects. Mainstream ML systems often blur those
boundaries because that makes it easy to move fast. TorchLean keeps them visible because that makes
it possible to audit and prove claims later.

The operational pipeline is:

```
environment step
  -> checked transition
  -> rollout buffer
  -> return / advantage computation
  -> PPO loss
  -> policy and value update
  -> optional MDP theorem
```

The runtime umbrella [NN.Runtime.RL API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL.lean) collects the executable pieces:

- [Runtime.RL.Core](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Core.lean) contains typed transitions, returns, TD errors,
  action encodings, and rollout helpers with tensor shapes.
- [Runtime.RL.Replay](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Replay.lean) contains bounded FIFO replay buffers for
  off policy algorithms.
- [Runtime.RL.Boundary](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Boundary.lean) checks externally supplied observations,
  rewards, actions, and done flags before they enter training.
- [Runtime.RL.Gymnasium](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Gymnasium.lean) is the subprocess bridge to Python
  Gymnasium.
- [Runtime.RL.PPO](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/PPO.lean) contains PPO rollout and sample collection with a fixed horizon.
- [Runtime.RL.Numerics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Numerics.lean) contains checked float32 diagnostics for
  RL recurrences.

# MDP Semantics First

The spec side is intentionally not "whatever the Python environment did." In
[NN.Spec.RL API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL.lean), TorchLean collects several MDP vocabularies:

- [Spec.RL.Core](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/Core.lean) gives discounted backups, TD residuals, returns, and
  GAE style algebra over rollout data.
- [Spec.RL.Environment](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/Environment.lean) gives a pure Gymnasium style environment
  contract with explicit latent state.
- [Spec.RL.MDP](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/MDP.lean) gives deterministic finite MDPs over finite states and
  actions.
- [Spec.RL.FiniteStochasticMDP](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/FiniteStochasticMDP.lean) gives finite stochastic
  transition kernels.
- [Spec.RL.MarkovMDP](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/MarkovMDP.lean) gives the heavier measurable-space development.
- [Spec.RL.Envs.GridWorld](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/Envs/GridWorld.lean) gives a concrete finite GridWorld
  environment used by the PPO example.

This is the first practical difference from mainstream stacks. In PyTorch code, a Bellman backup
is usually a line in a training loop:

$$`target = reward + \gamma(1-done)\,nextValue`

In TorchLean, that line is also a named semantic object. The proof layer can state and prove facts
about the Bellman operator rather than reconstructing a tensor expression after the fact.
[NN.Proofs.RL.MDP API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/MDP.lean), for example, proves monotonicity and contraction
facts for finite discounted MDPs. That is the classic dynamic programming theorem, but attached to
the same finite state vocabulary used by the examples.

The theorem shape is the one from standard dynamic programming:

$$`\operatorname{BellmanPolicy}_{\gamma,\pi}(V)(s)
= r(s,\pi(s))+\gamma\,\mathbb{E}_{s'}[V(s')]`

$$`\operatorname{BellmanOptimality}_{\gamma}(V)(s)
= \max_a\left(r(s,a)+\gamma\,\mathbb{E}_{s'}[V(s')]\right)`

$$`0\le\gamma<1
\quad\Longrightarrow\quad
\operatorname{BellmanPolicy}_{\gamma,\pi}
\text{ and }
\operatorname{BellmanOptimality}_{\gamma}
\text{ are sup-norm contractions}`

The proof pages state this as `bellmanPolicy_contraction`,
`bellmanOptimality_contraction`, `bellmanPolicy_fixedPoint_unique`, and
`bellmanOptimality_fixedPoint_unique`. That is why the RL stack belongs in a verification guide:
the runtime PPO examples and the mathematical Bellman facts share the same vocabulary for states,
actions, rewards, and horizons.

# The Gymnasium Boundary

We still use Gymnasium because it is the ecosystem boundary most RL users know. CartPole and Atari
Pong should not require us to reimplement every environment in Lean. But the boundary is explicit.

The [Gymnasium client API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Gymnasium/Client.lean) talks to
`scripts/rl/gymnasium_server.py` with one JSON request/response per line. On startup,
the Lean client asks the server to describe the observation shape and action count, then checks
that they match the Lean types expected by the training code. On every reset and step, observations
and rewards are parsed into typed tensors and checked by the boundary contract.

The contract itself is in [NN.Runtime.RL.Boundary.Core API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Boundary/Core.lean).
It can require finite observations, finite rewards, observation ranges, reward ranges, and sensible
done flag behavior. The output type is not an untyped Python dictionary. It is a
`Spec.RL.ObservedTransition` with a `Fin nActions` action.

This is a boundary contract, not a full theorem about Gymnasium or ALE. It turns common deployment
assumptions into checked preconditions. If the server returns `NaN`, an action outside the valid
range, the wrong observation shape, or a RAM byte outside the declared range, the transition does
not silently enter the PPO batch.

For a proof handle on a checked transition, the wrapper in
[NN.Proofs.RL.Gymnasium API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Gymnasium.lean) returns the transition bundled with
a proof of `Runtime.RL.Boundary.ContractHolds`. The pattern is runtime checking first, then a small
theorem that converts the successful check into a proposition future proofs can consume.

# Replay Buffers Without Mystery Mutation

Off policy algorithms need replay. In mainstream code, replay buffers are often mutable Python
objects whose shape invariants are enforced by convention. TorchLean's
[NN.Runtime.RL.Replay API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Replay.lean) defines a bounded FIFO buffer over
typed transitions:

$$`\operatorname{Buffer}(\alpha,obsShape,nActions)`

The type says that every stored observation has the same `obsShape` and every action belongs to
`Fin nActions`. The buffer has a capacity, pushes drop the oldest item when full, and sampling is
deterministic from an explicit seed/counter pair when random sampling is requested.

The proof layer in [NN.Proofs.RL.Replay API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Replay.lean) certifies structural
facts such as:

- empty buffers have size zero,
- zero capacity buffers remain empty after a push,
- pushing with room increases size by one,
- pushing when full preserves capacity by evicting an old item.

Those facts are modest, but they matter. A DQN theorem should not have to assume that the storage
layer forgot neither shape nor capacity. The repository also includes
[NN.Examples.Models.RL.DQNReplay API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/DQNReplay.lean) as a small
small example that uses the replay layer.

# PPO Rollouts, Returns, And Advantages

PPO is the clearest full example because it exercises almost every boundary. The top level
import is [NN.Runtime.RL.PPO API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/PPO.lean), with the data layout for a fixed horizon
in [NN.Runtime.RL.PPO.Rollout API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/PPO/Rollout.lean).

The rollout record stores the same fields a PyTorch implementation would store:

- `state`,
- `action`,
- `oldLogProb`,
- `reward`,
- `done`,
- `value`,
- `nextValue`.

The difference is that the rollout carries the invariant that it has exactly the configured
horizon. Conversion to an actor critic minibatch is therefore a total typed operation, not a
collection of unchecked array reshapes. `Rollout.toActorCriticSample` builds:

- a state tensor of shape `horizon × obsShape`,
- one hot actions of shape `horizon × nActions`,
- old log probabilities,
- normalized advantages,
- value targets computed from returns.

The advantage and return equations point back to the same RL core semantics. Generalized Advantage
Estimation uses the familiar recursion:

$$`\delta_t
= r_t+\gamma(1-done_t)V(s_{t+1})-V(s_t)`

$$`A_t
= \delta_t+\gamma\lambda(1-done_t)A_{t+1}`

TorchLean keeps this recurrence as a Lean definition, feeds its tensorized result to the PPO
autograd loss, and exposes the intermediate objects in examples and widgets. That makes it much
easier to answer "which formula did we train with?" months later.

The clipped PPO objective is the next object in the chain:

$$`r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}`

$$`L^{\mathrm{CLIP}}(\theta)
=
\mathbb E_t\left[
\min\!\left(r_t(\theta)A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t\right)
\right].`

TorchLean's checked numerics for ratios, clipping, returns, and advantages are local pieces of that
larger PPO story.

# Checked Float32 Numerics

RL code is numerically touchy. Rewards, bootstraps, exponentiated log probability ratios, and
advantage normalization all create opportunities for `NaN`, `Inf`, or accidental semantic mismatch
between notes over the reals and float32 execution.

TorchLean's checked float32 helpers are collected in the
[RL float32 numerics API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Runtime/RL/Numerics/Float32). They use the executable
`IEEE32Exec` model and return `Except String ...` so finite path failures are visible. The core
pieces are:

- [Returns](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Numerics/Float32/Returns.lean): checked discounted backups and
  returns over a fixed horizon.
- [Advantage](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Numerics/Float32/Advantage.lean): checked TD residuals, GAE, and
  advantage normalization.
- [PPO](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Numerics/Float32/PPO.lean): checked importance ratios and clipped PPO
  surrogate pieces.
- [Intervals](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/RL/Numerics/Float32/Intervals.lean): interval diagnostics for
  RL scalar recurrences.

The proof bridge is in [NN.Proofs.RL.Floats.IEEE32Exec API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Floats/IEEE32Exec.lean)
and [NN.Proofs.RL.Floats.CheckedRuntime API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Floats/CheckedRuntime.lean). The
important theorem shape for users is:

$$`\operatorname{checkedRuntime}(inputs)=\operatorname{ok}(result)
\Longrightarrow
\text{finite hypotheses hold}
\Longrightarrow
\operatorname{toReal}(result)
\text{ matches round-after-each-primitive FP32 semantics}`

That is stronger than "we ran PPO in float32 and it seemed fine." The checked helpers give us a
precise finite path semantics for the scalar formulas; native backend correctness is a separate
runtime agreement.

# The Three PPO Examples

The examples are deliberately arranged to show three different environment interfaces with one PPO
shape.

| Path | Environment | What is checked |
|---|---|---|
| GridWorld | Lean native | MDP semantics and transition facts |
| CartPole | Gymnasium | observation, action, reward, and done-field contract |
| Pong RAM | Gymnasium/ALE | observation shape and byte range |
| DQN replay | Lean runtime | buffer shape and capacity facts |

## Lean Native GridWorld

[NN.Examples.Models.RL.PPOGridWorld API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOGridWorld.lean) trains an
actor critic on a GridWorld defined in Lean. It imports the spec environment and proof hooks:
[NN.Spec.RL.Envs.GridWorld API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/RL/Envs/GridWorld.lean) and
[NN.Proofs.RL.Envs.GridWorld API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RL/Envs/GridWorld.lean). The example proves that
the finite stochastic MDP view over the reals is valid, then runs the same PPO update path used by the
external examples.

This is the most proof friendly path: dynamics, observations, rewards, and MDP validity all have
Lean side names. The widget viewer
[NN.Examples.RL.PPOGridWorldView API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/RL/PPOGridWorldView.lean) reads the
training/path artifacts so we can inspect behavior in the editor.

Run:

```
lake exe torchlean ppo_gridworld
```

## Gymnasium CartPole

[NN.Examples.Models.RL.PPOCartPole API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOCartPole.lean) uses
Gymnasium `CartPole-v1` through the subprocess bridge. This is the familiar mainstream workflow:
a Python environment produces observations, Lean builds the actor and critic, PPO collects rollouts,
and the optimizer updates the model.

The TorchLean difference is the contract. CartPole observations have shape `4`, actions live in
`Fin 2`, and every transition must pass the boundary checker before becoming training data. The
viewer [NN.Examples.RL.PPOCartPoleView API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/RL/PPOCartPoleView.lean) displays the
training log artifact.

Run:

```
python3 -m pip install --user 'gymnasium>=1.0'
lake exe torchlean ppo_cartpole
```

## Atari Pong RAM

[NN.Examples.Models.RL.PPOPongRam API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/RL/PPOPongRam.lean) targets
`ALE/Pong-v5` with RAM observations. The example uses RAM rather than pixels because the JSON lines
bridge is meant to demonstrate a checked boundary, not to maximize frames per second. The contract
checks the `128`-entry observation shape and the expected byte range, while the PPO model sees a
typed tensor just like the other examples.

The viewer [NN.Examples.RL.PPOPongRamView API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/RL/PPOPongRamView.lean) displays
the training curve artifact.

Run:

```
python3 -m pip install --user 'gymnasium>=1.0' ale-py
lake exe torchlean ppo_pong_ram
```

# Interpreting A TorchLean RL Claim

When an RL demo succeeds, we try to be precise and fair about the claim:

- The executable ran the PPO/autograd program and produced artifacts.
- The rollout boundary checked shapes, actions, finite values, and configured ranges.
- If the environment is defined in Lean, its MDP semantics can be used directly by proofs.
- If the environment is Gymnasium/ALE, the external dynamics remain a named producer assumption.
- Checked float32 helpers can connect selected scalar recurrences to `IEEE32Exec` and FP32-style
  semantics with rounding after each primitive.
- MDP and replay proofs certify specific structural or dynamic programming facts, not global PPO
  convergence.

That phrasing is less flashy than "verified RL," but it is much more useful. We built the stack so
each claim can be upgraded independently: prove more about GridWorld, add stronger Gymnasium
contracts, extend checked PPO numerics, or connect a larger algorithm theorem to the same runtime
objects.
