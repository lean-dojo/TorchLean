/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Robustness
public import TorchLean.Tensor.Reduce.Fold

/-!
# Executable robustness utilities

Float-specialized executable versions of the robustness specifications in
`TorchLean.Robustness`.
-/

@[expose] public section

namespace TorchLean.Robustness

/-!
# Robustness runtime utilities (Float)

This file holds executable `Float` helpers for experiments, examples and command-line diagnostics.
The L∞ norm and distance of `Float` tensors are `Tensor.maxAbs` and `Tensor.maxAbsDiff`. We define
the Euclidean norm `l2NormFloat` and sampling helpers such as
`maxL2LipschitzRatio`, the largest L2 Lipschitz ratio observed over a chosen set of input
pairs. For `[3, 4]` the L∞ norm is `4` and `l2NormFloat` gives `5`.

The sampling helpers return `Except` results computed from finite samples. An error records absent
or non-finite evidence. A successful value is empirical: it is neither a certificate nor a
`Prop`-level guarantee.
-/

/-- Runtime `L2` norm, computed by dividing every entry by the largest absolute entry before
squaring, so finite entries do not overflow or underflow merely because their squares do. A NaN
entry gives NaN, and an infinite entry or a final norm too large for `Float` gives infinity. -/
def l2NormFloat {s : Shape} (t : Tensor Float s) : Float :=
  let scale := Tensor.maxAbs t
  if scale == 0 || !scale.isFinite then scale else
    let squares := t.foldl (fun sum value =>
      let normalized := value / scale
      sum + normalized * normalized) 0
    scale * Float.sqrt squares

/-! ## Empirical (sampling-based) helpers -/

/--
Maximum observed $L^2$ Lipschitz ratio over a given finite array of input pairs:

$$
\max_{(x,y)\in\mathrm{pairs}}
\frac{\lVert f(x)-f(y)\rVert_2}{\lVert x-y\rVert_2}
$$

The maximum ranges over the given pairs only and certifies no global bound. Empty input,
non-finite distances or ratios, and collections containing no positive input distance are
reported as errors. Zero-distance pairs are ignored when another pair is informative.
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
    let inputDist := l2NormFloat (x - y)
    let outputDist := l2NormFloat (f x - f y)
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

/--
Turn an array with at least two elements `#[x₀,x₁,…,x_{m-1}]` into adjacent pairs
`#[(x₀,x₁),(x₁,x₂),…,(x_{m-1},x₀)]`.

The result serves as a set of “nearby pairs” for empirical ratio computations such as
`maxL2LipschitzRatio`. Empty and singleton arrays return no pairs.
-/
def adjacentPairs {α : Type} (xs : Array α) : Array (α × α) :=
  if xs.size < 2 then #[]
  else
    -- `(i + 1) % xs.size` is always in bounds, so the `getD` fallback only spares a panic path.
    xs.mapIdx fun i x => (x, xs.getD ((i + 1) % xs.size) x)

end TorchLean.Robustness
