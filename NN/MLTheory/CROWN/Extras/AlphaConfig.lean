/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor

/-!
# α-CROWN configuration

This file defines data structures for \(\alpha\)-optimized CROWN bounds (as in \(\alpha\)-CROWN /
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

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-- Per-neuron optimizable alpha parameters. -/
structure NeuronAlpha (α : Type) where
  /-- Lower-envelope α parameter. For ReLU this is the candidate lower slope. -/
  lower : α
  /-- Upper-envelope α parameter, used by smooth relaxations that interpolate upper candidates. -/
  upper : α

/-- Layer-wise alpha configuration. -/
structure LayerAlpha (α : Type) [Context α] where
  /-- Number of neurons in this activation layer. -/
  dim : Nat
  /-- Per-neuron α parameters, indexed by the layer dimension. -/
  alphas : Tensor (NeuronAlpha α) (.dim dim .scalar)

/-- Full network alpha configuration. -/
structure NetworkAlpha (α : Type) [Context α] where
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
  if u < Numbers.zero then
    .inactive
  else if l > Numbers.zero then
    .active
  else if l < Numbers.zero ∧ u > Numbers.zero then
    .crossing
  else
    .unknown

/-- Initialize alpha for a crossing ReLU neuron.

The default lower slope is `0`, the conservative lower envelope `y ≥ 0`. -/
def defaultReLUAlpha : NeuronAlpha α :=
  { lower := Numbers.zero
  , upper := Numbers.one }

/-- Initialize alpha for active neuron (no relaxation needed). -/
def activeAlpha : NeuronAlpha α :=
  { lower := Numbers.one
  , upper := Numbers.one }

/-- Initialize alpha for inactive neuron. -/
def inactiveAlpha : NeuronAlpha α :=
  { lower := Numbers.zero
  , upper := Numbers.zero }

/-- Initialize alpha based on pre-activation bounds [l, u]. -/
def initAlpha (l u : α) : NeuronAlpha α :=
  match neuronStatus l u with
  | .inactive => inactiveAlpha
  | .active => activeAlpha
  | .crossing => defaultReLUAlpha
  | .unknown => defaultReLUAlpha

/-- Initialize layer alpha from pre-activation bounds box. -/
def initLayerAlpha (n : Nat) (preB : Box α (.dim n .scalar)) : LayerAlpha α :=
  match preB.lo, preB.hi with
  | .dim lo, .dim hi =>
    let alphas := Tensor.dim (fun i : Fin n =>
      match lo i, hi i with
      | .scalar l, .scalar u => Tensor.scalar (initAlpha (α:=α) l u))
    { dim := n, alphas := alphas }

/-- Project alpha to valid range [0, 1] for ReLU. -/
def projectReLUAlpha (a : NeuronAlpha α) : NeuronAlpha α :=
  let lo := if a.lower < Numbers.zero then Numbers.zero
            else if a.lower > Numbers.one then Numbers.one
            else a.lower
  let hi := if a.upper < Numbers.zero then Numbers.zero
            else if a.upper > Numbers.one then Numbers.one
            else a.upper
  { lower := lo, upper := hi }

/-- Compute a ReLU lower-envelope candidate using an optimized alpha.

For a crossing neuron with bounds `[l, u]`, the candidate is `y = α * x`. A sound replay theorem
should pair this executable rule with the usual α-range condition for the chosen scalar backend.
-/
def reluLowerWithAlpha (_l _u alphaLo : α) : α × α :=
  -- Lower bound: y = α·x
  -- Slope = α, bias = 0
  (alphaLo, Numbers.zero)

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
    (Numbers.zero, Numbers.zero, Numbers.zero, Numbers.zero)
  | .active =>
    (Numbers.one, Numbers.zero, Numbers.one, Numbers.zero)
  | .crossing =>
    let (slo, blo) := reluLowerWithAlpha (α:=α) l u alphas.lower
    let (shi, bhi) := reluUpperFixed (α:=α) l u
    (slo, blo, shi, bhi)
  | .unknown =>
    let (slo, blo) := reluLowerWithAlpha (α:=α) l u alphas.lower
    let (shi, bhi) := reluUpperFixed (α:=α) l u
    (slo, blo, shi, bhi)

/-- Gradient of output bounds w.r.t. alpha (for optimization).
    For ReLU: ∂bound/∂α = x for lower bound (where y = αx). -/
structure AlphaGradient (α : Type) where
  /-- Gradient for lower alpha -/
  grad_lower : α
  /-- Gradient for upper alpha -/
  grad_upper : α

/-- Layer-wise alpha gradients. -/
structure LayerAlphaGrad (α : Type) [Context α] where
  /-- Number of neurons in the activation layer. -/
  dim : Nat
  /-- Per-neuron gradients of the bound objective with respect to α parameters. -/
  grads : Tensor (AlphaGradient α) (.dim dim .scalar)

/-- Local sigmoid approximation: σ(x) = 1 / (1 + exp(-x))
    Using polynomial approximation for the Context typeclass. -/
def sigmoidApprox (x : α) : α :=
  -- Approximate sigmoid using tanh: σ(x) ≈ 0.5 + 0.5*tanh(x/2)
  -- Or use linear approximation in typical range
  let y := x * Numbers.pointfive
  let y2 := y * y
  -- Approximation: 0.5 + 0.25*x - 0.02*x^3 (for small x)
  Numbers.pointfive + Numbers.pointfive * Numbers.pointfive * x -
    (Numbers.one / (Numbers.one + Numbers.one + Numbers.one + Numbers.one + Numbers.one) /
     (Numbers.one + Numbers.one + Numbers.one + Numbers.one + Numbers.one)) * y2 * y

/-- Local tanh approximation: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) -/
def tanhApprox (x : α) : α :=
  -- Approximate tanh using polynomial
  let x2 := x * x
  let x3 := x2 * x
  -- tanh(x) ≈ x - x^3/3 for small x
  x - x3 / Numbers.three

/-- Sigmoid relaxation with interpolation alpha.
    α ∈ [0,1] interpolates between tangent and secant for lower/upper. -/
def sigmoidWithAlpha (l u : α) (alphas : NeuronAlpha α) : α × α × α × α :=
  let σl := sigmoidApprox (α:=α) l
  let σu := sigmoidApprox (α:=α) u
  let mid := (l + u) * Numbers.pointfive
  let σmid := sigmoidApprox (α:=α) mid

  -- Tangent slopes: σ'(x) = σ(x) * (1 - σ(x))
  let slope_tan_l := σl * (Numbers.one - σl)
  let slope_tan_m := σmid * (Numbers.one - σmid)

  -- Secant slope
  let slope_sec := if u > l + Numbers.epsilon then (σu - σl) / (u - l) else slope_tan_m

  -- Lower: interpolate between tangent at l and tangent at mid
  let slope_lo := alphas.lower * slope_tan_l + (Numbers.one - alphas.lower) * slope_tan_m
  let bias_lo := σl - slope_lo * l

  -- Upper: always secant (not optimizable for convex hull)
  let bias_hi := σl - slope_sec * l

  (slope_lo, bias_lo, slope_sec, bias_hi)

/-- Tanh relaxation with interpolation alpha. -/
def tanhWithAlpha (l u : α) (alphas : NeuronAlpha α) : α × α × α × α :=
  let tl := tanhApprox (α:=α) l
  let tu := tanhApprox (α:=α) u
  let mid := (l + u) * Numbers.pointfive
  let tmid := tanhApprox (α:=α) mid

  -- Tangent slopes: tanh'(x) = 1 - tanh²(x)
  let slope_tan_l := Numbers.one - tl * tl
  let slope_tan_m := Numbers.one - tmid * tmid

  -- Secant slope
  let slope_sec := if u > l + Numbers.epsilon then (tu - tl) / (u - l) else slope_tan_m

  -- Lower: interpolate
  let slope_lo := alphas.lower * slope_tan_l + (Numbers.one - alphas.lower) * slope_tan_m
  let bias_lo := tl - slope_lo * l

  -- Upper: secant
  let bias_hi := tl - slope_sec * l

  (slope_lo, bias_lo, slope_sec, bias_hi)

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
structure OptimizedAlpha (α : Type) [Context α] where
  /-- Optimized network alpha configuration -/
  alphas : NetworkAlpha α
  /-- Final bound achieved -/
  finalBound : α
  /-- Number of iterations used -/
  iterations : Nat

namespace Theorems

/-- ReLU lower with alpha=0 gives zero slope. -/
theorem relu_lower_alpha_zero (l u : α) :
    let (slope, bias) := reluLowerWithAlpha (α:=α) l u Numbers.zero
    slope = Numbers.zero ∧ bias = Numbers.zero := by
  unfold reluLowerWithAlpha
  exact ⟨rfl, rfl⟩

/-- ReLU lower with alpha=1 gives unit slope. -/
theorem relu_lower_alpha_one (l u : α) :
    let (slope, _) := reluLowerWithAlpha (α:=α) l u Numbers.one
    slope = Numbers.one := by
  unfold reluLowerWithAlpha
  rfl

/-- Default ReLU alpha has zero lower slope. -/
theorem default_relu_alpha_lower :
    (defaultReLUAlpha (α:=α)).lower = Numbers.zero := by
  rfl

/-- Default ReLU alpha has unit upper value. -/
theorem default_relu_alpha_upper :
    (defaultReLUAlpha (α:=α)).upper = Numbers.one := by
  rfl

/-- Active alpha has unit slope for both. -/
theorem active_alpha_unit :
    (activeAlpha (α:=α)).lower = Numbers.one ∧ (activeAlpha (α:=α)).upper = Numbers.one := by
  exact ⟨rfl, rfl⟩

/-- Inactive alpha has zero slope for both. -/
theorem inactive_alpha_zero :
    (inactiveAlpha (α:=α)).lower = Numbers.zero ∧ (inactiveAlpha (α:=α)).upper = Numbers.zero := by
  exact ⟨rfl, rfl⟩

/-- Init layer alpha preserves dimension. -/
theorem init_layer_alpha_dim (n : Nat) (preB : Box α (.dim n .scalar)) :
    (initLayerAlpha n preB).dim = n := by
  simp only [initLayerAlpha]
  match preB.lo, preB.hi with
  | .dim _, .dim _ => rfl

end Theorems

end NN.MLTheory.CROWN.alpha
