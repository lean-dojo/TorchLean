/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis
public import NN.Proofs.Autograd.Core.RealCorrectness
public import NN.Proofs.Autograd.Core.SemiringCorrectness
public import NN.Proofs.Autograd.Coverage
public import NN.Proofs.Autograd.FDeriv.Core
public import NN.Proofs.Autograd.FDeriv.Elementwise
public import NN.Proofs.Autograd.FDeriv.MlpMse
public import NN.Proofs.Autograd.FDeriv.OpSpec
public import NN.Proofs.Autograd.FDeriv.Params
public import NN.Proofs.Autograd.Runtime.Link
public import NN.Proofs.Autograd.Tape.Algebra.Nodes
public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Proofs.Autograd.Tape.Core.FDeriv
public import NN.Proofs.Autograd.Tape.Core.Soundness
public import NN.Proofs.Autograd.Tape.Nodes
public import NN.Proofs.Autograd.Tape.Ops.Attention.MultiHeadSelfAttention
public import NN.Proofs.Autograd.Tape.Ops.Attention.ScaledDotProduct
public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot
public import NN.Proofs.Autograd.Tape.Ops.Conv.FDeriv
public import NN.Proofs.Autograd.Tape.Ops.Embedding.GatherRows
public import NN.Proofs.Autograd.Tape.Ops.Norm.BatchNormChannelFirst
public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm
public import NN.Proofs.Autograd.Training.StepAlgebra
public import NN.Proofs.Gradients.Activation
public import NN.Proofs.Gradients.Linear
public import NN.Proofs.Models
public import NN.Proofs.Probability
public import NN.Proofs.RuntimeApprox
public import NN.Proofs.RL.Boundary
public import NN.Proofs.RL.Gymnasium
public import NN.Proofs.RL.Core
public import NN.Proofs.RL.Replay
public import NN.Proofs.RL.Environment
public import NN.Proofs.RL.Envs.GridWorld
public import NN.Proofs.RL.Algorithms.DQN
public import NN.Proofs.RL.MDP
public import NN.Proofs.RL.MarkovMDP
public import NN.Proofs.RL.FiniteStochasticMDP
public import NN.Proofs.RL.Floats.IEEE32Exec
public import NN.Proofs.RL.Floats.CheckedRuntime
public import NN.Proofs.Tensor
public import NN.Proofs.Verification

/-!
# Proof entrypoint

Umbrella import for TorchLean's core proof infrastructure.

This is the set of proof modules that `NN.Library` considers part of the supported
library surface area (as opposed to tests, examples, or executable workflows).

Notes:
- This is intentionally *curated* (not “import everything under `NN/Proofs`”).
- If you add a new proof module that should be part of the stable surface,
  add it here.

Proof-facing landmarks:
- real-analysis and numerics-facing helper theorems: `NN.Proofs.Analysis`,
- analytic autograd correctness: `NN.Proofs.Autograd.FDeriv.*`,
- tape/DAG reverse-mode correctness: `NN.Proofs.Autograd.Tape.*`,
- model-level invariants: `NN.Proofs.Models`,
- probability-kernel facts: `NN.Proofs.Probability`,
- runtime-approximation bounds: `NN.Proofs.RuntimeApprox.*`,
- verification envelopes and backend bridges: `NN.Proofs.Verification`.

References / context:
- PyTorch autograd background:
  https://pytorch.org/docs/stable/autograd.html
- The theorem bundles re-exported here document the math/model references they rely on locally, so
  this file stays a thin, stable import surface rather than duplicating long bibliographies.
-/

@[expose] public section
