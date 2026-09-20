/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core

/-!
# α-CROWN configuration

This file defines data structures for $\alpha$-optimized CROWN bounds (as in $\alpha$-CROWN /
auto\_LiRPA): per-neuron relaxation parameters that can be tuned externally to tighten affine
relaxations.

This module is optional for the core bound-propagation development; it is grouped under
`NN/MLTheory/CROWN/Extras/` to keep the main entrypoints smaller.

The design mirrors the practical α-CROWN workflow: Lean stores the relaxation parameters and the
resulting transfer functions, while a separate optimizer may tune those parameters before a
certificate is replayed.

References:
- Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions" (CROWN), NeurIPS 2018.
- Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond"
  (auto_LiRPA), NeurIPS 2020.
- Xu et al., "Fast and Complete: Enabling Complete Neural Network Verification with Rapid and
  Massively Parallel Incomplete Verifiers" (α/β-CROWN), ICLR 2021.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.alpha

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Per-neuron optimizable alpha parameters. -/
structure NeuronAlpha (α : Type) where
  /-- Lower-envelope α parameter. For ReLU this is the candidate lower slope. -/
  lower : α
  /-- Upper-envelope α parameter, used by smooth relaxations that interpolate upper candidates. -/
  upper : α

/-- Layer-wise alpha configuration. -/
structure LayerAlpha (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of neurons in this activation layer. -/
  dim : Nat
  /-- Per-neuron α parameters, indexed by the layer dimension. -/
  alphas : Tensor (NeuronAlpha α) [dim]

/-- Full network alpha configuration. -/
structure NetworkAlpha (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of activation layers (not counting input/output) -/
  numLayers : Nat
  /-- Per-layer alpha values -/
  layers : Array (LayerAlpha α)

/-- Neuron status based on pre-activation bounds. -/
inductive NeuronStatus where
  | inactive  -- u ≤ 0: ReLU always outputs 0
  | active    -- l ≥ 0: ReLU always passes through
  | crossing  -- l < 0 < u: ReLU needs relaxation
  | unknown   -- Cannot determine
  deriving Repr, BEq

/-- Determine neuron status from bounds. -/
def neuronStatus (l u : α) : NeuronStatus :=
  if u < 0 then
    .inactive
  else if l > 0 then
    .active
  else if l < 0 ∧ u > 0 then
    .crossing
  else
    .unknown

/-- Initialize alpha for a crossing ReLU neuron.

The default lower slope is `0`, the conservative lower envelope `y ≥ 0`. -/
def defaultReLUAlpha : NeuronAlpha α :=
  { lower := 0
  , upper := 1 }

/-- Initialize alpha for active neuron (no relaxation needed). -/
def activeAlpha : NeuronAlpha α :=
  { lower := 1
  , upper := 1 }

/-- Initialize alpha for inactive neuron. -/
def inactiveAlpha : NeuronAlpha α :=
  { lower := 0
  , upper := 0 }

/-- Initialize alpha based on pre-activation bounds [l, u]. -/
def initAlpha (l u : α) : NeuronAlpha α :=
  match neuronStatus l u with
  | .inactive => inactiveAlpha
  | .active => activeAlpha
  | .crossing => defaultReLUAlpha
  | .unknown => defaultReLUAlpha

/-- Initialize layer alpha from pre-activation bounds box. -/
def initLayerAlpha (n : Nat) (preB : Box α (.dim n .scalar)) : LayerAlpha α :=
  let alphas := Tensor.dim (fun i : Fin n =>
    Tensor.scalar (initAlpha (α := α) (preB.lo.getScalar i) (preB.hi.getScalar i)))
  { dim := n, alphas := alphas }

/-- Project alpha to valid range [0, 1] for ReLU. -/
def projectReLUAlpha (a : NeuronAlpha α) : NeuronAlpha α :=
  let lo := if a.lower < 0 then 0
            else if a.lower > 1 then 1
            else a.lower
  let hi := if a.upper < 0 then 0
            else if a.upper > 1 then 1
            else a.upper
  { lower := lo, upper := hi }

/-- Compute a ReLU lower-envelope candidate using an optimized alpha.

For a crossing neuron with bounds `[l, u]`, the candidate is `y = α * x`. A sound replay theorem
should pair this executable rule with the usual α-range condition for the chosen scalar backend.
-/
def reluLowerWithAlpha (_l _u alphaLo : α) : α × α :=
  -- Lower bound: y = α·x
  -- Slope = α, bias = 0
  (alphaLo, 0)

/-- Compute the fixed triangular ReLU upper bound.

The line is `y = (u/(u-l)) * x - (u*l)/(u-l)` for a crossing interval `[l,u]`.
-/
def reluUpperFixed (l u : α) : α × α :=
  let denom := u - l
  let slope := u / denom
  let bias := -(u * l) / denom
  (slope, bias)

/-- Apply alpha-parameterized ReLU relaxation to get affine bounds.
    Returns (slope_lo, bias_lo, slope_hi, bias_hi). -/
def reluWithAlpha (l u : α) (alphas : NeuronAlpha α) : α × α × α × α :=
  match neuronStatus l u with
  | .inactive =>
    (0, 0, 0, 0)
  | .active =>
    (1, 0, 1, 0)
  | .crossing =>
    let (slo, blo) := reluLowerWithAlpha (α:=α) l u alphas.lower
    let (shi, bhi) := reluUpperFixed (α:=α) l u
    (slo, blo, shi, bhi)
  | .unknown =>
    if u > l then
      let (slo, blo) := reluLowerWithAlpha (α:=α) l u alphas.lower
      let (shi, bhi) := reluUpperFixed (α:=α) l u
      (slo, blo, shi, bhi)
    else
      -- The only valid ordered interval in this branch is `[0,0]`. Avoid the `0/0`
      -- denominator in the secant formula and use the exact zero map.
      (0, 0, 0, 0)

/-- Gradient of output bounds with respect to $\alpha$.

For the ReLU lower bound $y=\alpha x$, the derivative is
$\partial\,\mathrm{bound}/\partial\alpha=x$.
-/
structure AlphaGradient (α : Type) where
  /-- Gradient for lower alpha -/
  grad_lower : α
  /-- Gradient for upper alpha -/
  grad_upper : α

/-- Layer-wise alpha gradients. -/
structure LayerAlphaGrad (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Number of neurons in the activation layer. -/
  dim : Nat
  /-- Per-neuron gradients of the bound objective with respect to α parameters. -/
  grads : Tensor (AlphaGradient α) [dim]

/-- Configuration for alpha optimization. -/
structure AlphaOptConfig where
  /-- Learning rate for alpha updates -/
  learningRate : Float := 0.1
  /-- Number of optimization iterations -/
  numIterations : Nat := 20
  /-- Whether to optimize lower alphas -/
  optimizeLower : Bool := true
  /-- Whether to optimize upper alphas (for smooth activations) -/
  optimizeUpper : Bool := false

/-- Result of alpha optimization. -/
structure OptimizedAlpha (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Optimized network alpha configuration -/
  alphas : NetworkAlpha α
  /-- Final bound achieved -/
  finalBound : α
  /-- Number of iterations used -/
  iterations : Nat

end NN.MLTheory.CROWN.alpha
