# Reinforcement-Learning Specs

This folder contains TorchLean's pure reinforcement-learning semantics. It is where the project says
what an environment, return, Bellman backup, advantage estimate, and policy objective mean before any
runtime collector or optimizer is involved.

These definitions are the mathematical contracts implemented or approximated by runtime code. Rollout buffers,
Gymnasium sessions, CUDA kernels, logging, and optimizer state live under `NN/Runtime`; the objects in
this folder are the reference definitions those systems can be compared against.

## Layers

- `Core.lean`: Bellman style one step backups, TD residuals, discounted returns, and GAE.
- `Environment.lean`: a pure Gymnasium style environment interface with explicit latent state.
- `MDP.lean`: deterministic finite discounted MDPs over `Fin n` states/actions.
- `FiniteStochasticMDP.lean`: finite stochastic discounted MDPs with tensor transition rows.
- `MarkovMDP.lean`: measurable space discounted MDPs using mathlib Markov kernels.
- `Envs/GridWorld.lean`: a concrete finite GridWorld plus deterministic/stochastic MDP views.

The deterministic model is `Spec.RL.FiniteMDP`; the stochastic models are
`Spec.RL.FiniteStochastic.MDP` and `Spec.RL.Markov.MDP`. Both finite models use the tensor-backed
`Spec.RL.ValueFunction`, while `Spec.RL.Markov.ValueFunction` is a real-valued function on a
measurable state space. Their Bellman operators live in the corresponding namespaces.

## How It Connects To The Rest Of TorchLean

There are three different questions that should stay separate:

- What is the mathematical RL object? That belongs here.
- How do we collect samples or run an external environment? That belongs in `NN/Runtime/RL`.
- What do we prove about the resulting algorithm or boundary? That belongs in `NN/Proofs/RL` and
  `NN/MLTheory`.

For example, PPO code may use minibatches, clipping, logging, and runtime tensors. The spec layer
should still expose the idealized return and advantage quantities that make the objective meaningful.
Likewise, a Gymnasium adapter may talk to Python, but the mathematical environment interface here
defines its state transition, reward, and termination condition explicitly.

## What To Prove Here

Good spec-level lemmas are small and reusable:

- a discounted return recurrence agrees with its finite sum form;
- a TD residual expands to the expected Bellman expression;
- a deterministic GridWorld transition preserves the declared state space;
- a stochastic transition row is normalized before it is used as a probability distribution.

Those lemmas give runtime and verification code stable targets: the checked statements are about the
return, residual, transition, or probability row named by the spec, while full training claims can be
added as separate theorem or evaluation layers.

## References

- Bellman, *Dynamic Programming* (1957).
- Puterman, *Markov Decision Processes* (1994).
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.).
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015).
- Gymnasium API reference: <https://gymnasium.farama.org/>.
- TorchRL documentation: <https://pytorch.org/rl/>.
