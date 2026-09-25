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
public import NN.Proofs.Autograd.Dual
public import NN.Proofs.Autograd.Dual.Domain
public import NN.Proofs.Autograd.DualTensor
public import NN.Proofs.Autograd.FDeriv.Core
public import NN.Proofs.Autograd.FDeriv.Elementwise
public import NN.Proofs.Autograd.FDeriv.HardMaskedAttention
public import NN.Proofs.Autograd.FDeriv.HardMaskedSoftmax
public import NN.Proofs.Autograd.FDeriv.MlpMse
public import NN.Proofs.Autograd.FDeriv.OpSpec
public import NN.Proofs.Autograd.FDeriv.Params
public import NN.Proofs.Autograd.FDeriv.PrimitiveArithmetic
public import NN.Proofs.Autograd.FDeriv.PrimitiveCoordinates
public import NN.Proofs.Autograd.FDeriv.PrimitiveExtrema
public import NN.Proofs.Autograd.FDeriv.PrimitiveSpecs
public import NN.Proofs.Autograd.FDeriv.Reindex
public import NN.Proofs.Autograd.FDeriv.SoftmaxAxis
public import NN.Proofs.Autograd.FDeriv.SoftmaxSpec
public import NN.Proofs.Autograd.FDeriv.TensorCoordinates
public import NN.Proofs.Autograd.Model
public import NN.Proofs.Autograd.Runtime.Link
public import NN.Proofs.Autograd.Tape.Algebra.Nodes
public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Proofs.Autograd.Tape.Core.FDeriv
public import NN.Proofs.Autograd.Tape.Core.Soundness
public import NN.Proofs.Autograd.Tape.Nodes
public import NN.Proofs.Autograd.Tape.Ops.Attention.MultiHeadSelfAttention
public import NN.Proofs.Autograd.Tape.Ops.Attention.ScaledDotProduct
public import NN.Proofs.Autograd.Tape.Ops.Attention.SpecBridge
public import NN.Proofs.Autograd.Tape.Ops.Conv.FDeriv
public import NN.Proofs.Autograd.Tape.Ops.Embedding.GatherRows
public import NN.Proofs.Autograd.Tape.Ops.Norm.BatchNorm
public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNorm
public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormBounds
public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormSecondBounds
public import NN.Proofs.Autograd.Tape.Ops.Norm.LayerNormRuntime
public import NN.Proofs.Autograd.Training.StepAlgebra
public import NN.Proofs.Backend
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
# Proofs

This module collects the maintained proof API: tensor facts, selected autograd correctness
theorems, runtime-approximation theorems, model proofs, and verification soundness results.

These are the proof modules included in the supported library, separate from tests,
examples, and executable workflows.

Proof landmarks:
- real-analysis and numerics helper theorems: `NN.Proofs.Analysis`,
- analytic autograd correctness fragments: `NN.Proofs.Autograd.FDeriv.*`,
- tape/DAG reverse-mode correctness fragments: `NN.Proofs.Autograd.Tape.*`,
- backend selection and lossless contract grouping: `NN.Proofs.Backend`,
- model-level invariants: `NN.Proofs.Models`,
- probability-kernel facts: `NN.Proofs.Probability`,
- runtime-approximation bounds: `NN.Proofs.RuntimeApprox.*`,
- verification envelopes: `NN.Proofs.Verification`.

Backend proofs establish properties of the production Lean planner and its declared contracts.
They do not prove native implementation refinements.

References:
- PyTorch autograd background:
  https://pytorch.org/docs/stable/autograd.html
- The theorem bundles re-exported here document the math/model references they rely on locally.
-/

@[expose] public section
