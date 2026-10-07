/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Spec

/-!
# `NN.MLTheory.Robustness.Runtime`

Executable Float-specialized utilities for the robustness specifications in
`NN.MLTheory.Robustness.Spec`.
-/

@[expose] public section

open Spec TorchLean

namespace NN.MLTheory.Robustness.Runtime

/-!
# Robustness runtime utilities (Float)

This file specializes the polymorphic spec in `NN.MLTheory.Robustness.Spec` to `Float` and adds a
few executable helpers used by experiments, examples, and command-line diagnostics.

It includes:

- deterministic **specializations** (norms/distances/balls), and
- small **sampling-based helpers** that compute *empirical* quantities (e.g. “max observed ratio”
  over a chosen sample set).

The sampling helpers return `Except` results computed from finite samples. An error records absent
or non-finite evidence; a successful value is still empirical and does not provide a certificate
or `Prop`-level guarantee.
-/

/--
Runtime `L∞` norm, defined by specializing the polymorphic spec to `Float`.
-/
def tensorLinfNormFloat {s : Shape} (t : Tensor Float s) : Float :=
  NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := Float) (s := s) t

/--
Runtime `L2` norm, defined by specializing the polymorphic spec to `Float`.
-/
def tensorL2NormFloat {s : Shape} (t : Tensor Float s) : Float :=
  NN.MLTheory.Robustness.Spec.tensorL2Norm (α := Float) (s := s) t

/-- `L2` distance (specialization of `Robustness.Spec.tensorDistance`). -/
def tensorL2DistanceFloat {s : Shape} (t1 t2 : Tensor Float s) : Float :=
  NN.MLTheory.Robustness.Spec.tensorDistance
    (α := Float) (norm := tensorL2NormFloat) (s := s) t1 t2

/-- `L∞` distance (specialization of `Robustness.Spec.tensorDistance`). -/
def tensorLinfDistanceFloat {s : Shape} (t1 t2 : Tensor Float s) : Float :=
  NN.MLTheory.Robustness.Spec.tensorDistance
    (α := Float) (norm := tensorLinfNormFloat) (s := s) t1 t2

/--
Decide whether `t` lies in the closed `L∞`-ball of radius `ε` around `center`.
-/
def inLinfBallFloat {s : Shape} (center : Tensor Float s) (ε : Float) (t : Tensor Float s) :
    Bool :=
  tensorLinfDistanceFloat center t ≤ ε

/-! ## Empirical (sampling-based) helpers -/

namespace Empirical

/--
Maximum observed $L^2$ Lipschitz ratio over a given finite array of input pairs.

This computes:

$$
\max_{(x,y)\in\mathrm{pairs}}
\frac{\lVert f(x)-f(y)\rVert_2}{\lVert x-y\rVert_2}
$$

Factually: this is a maximum over the provided pairs only; it is not a certified global bound.

Empty input, non-finite distances or ratios, and collections containing no positive input distance
are reported as errors. Zero-distance pairs are ignored when another pair is informative.
-/
def maxL2LipschitzRatio {s₁ s₂ : Shape}
    (f : Tensor Float s₁ → Tensor Float s₂)
    (pairs : Array (Tensor Float s₁ × Tensor Float s₁)) : Except String Float := do
  if pairs.isEmpty then
    throw "empirical Lipschitz ratio requires at least one input pair"
  let mut foundInformative := false
  let mut maxRatio := 0.0
  for p in pairs do
    let (x, y) := p
    let inputDist := tensorL2DistanceFloat x y
    let outputDist := tensorL2DistanceFloat (f x) (f y)
    unless inputDist.isFinite && outputDist.isFinite do
      throw "empirical Lipschitz ratio encountered a non-finite distance"
    if inputDist < 0.0 || outputDist < 0.0 then
      throw "empirical Lipschitz ratio encountered a negative distance"
    if inputDist > 0.0 then
      let ratio := outputDist / inputDist
      unless ratio.isFinite do
        throw "empirical Lipschitz ratio overflowed to a non-finite value"
      foundInformative := true
      maxRatio := max maxRatio ratio
  unless foundInformative do
    throw "empirical Lipschitz ratio requires a pair with positive input distance"
  pure maxRatio

end Empirical

namespace Sampling

/--
Turn an array with at least two elements `#[x₀,x₁,…,x_{m-1}]` into adjacent pairs
`[(x₀,x₁),(x₁,x₂),…,(x_{m-1},x₀)]`.

This is a small combinator that is useful when you want to turn a sample array into a set of
“nearby pairs” for empirical ratio computations. Empty and singleton arrays return no pairs.
-/
def adjacentPairs {α : Type} (xs : Array α) : Array (α × α) :=
  if xs.size < 2 then #[]
  else
    -- `(i + 1) % xs.size` is always in bounds; the `getD` fallback only avoids a panic path.
    xs.mapIdx fun i x => (x, xs.getD ((i + 1) % xs.size) x)

end Sampling

end NN.MLTheory.Robustness.Runtime
