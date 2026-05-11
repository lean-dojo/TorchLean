# Reinforcement-Learning Specs

This folder contains TorchLean's pure RL semantics. The intent is to keep the mathematical contract
small and auditable, while runtime training, collectors, CUDA, logging, and optimizer state live
under `NN/Runtime`.

## Layers

- `Core.lean`: Bellman style one step backups, TD residuals, discounted returns, and GAE.
- `Environment.lean`: a pure Gymnasium style environment interface with explicit latent state.
- `MDP.lean`: deterministic finite discounted MDPs over `Fin n` states/actions.
- `FiniteStochasticMDP.lean`: finite stochastic discounted MDPs with tensor transition rows.
- `MarkovMDP.lean`: measurable space discounted MDPs using mathlib Markov kernels.
- `Envs/GridWorld.lean`: a concrete finite GridWorld plus deterministic/stochastic MDP views.

The repeated names (`MDP`, `ValueFunction`, `bellmanPolicy`, etc.) are namespace-scoped on purpose:
`Spec.RL` is deterministic finite, `Spec.RL.FiniteStochastic` is finite stochastic, and
`Spec.RL.Markov` is measure-theoretic.

## References

- Bellman, *Dynamic Programming* (1957).
- Puterman, *Markov Decision Processes* (1994).
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.).
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015).
- Gymnasium API docs: <https://gymnasium.farama.org/>.
- TorchRL documentation: <https://pytorch.org/rl/>.
