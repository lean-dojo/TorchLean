/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.Tensor.Conversion
public import NN.Spec.Core.Tensor -- shake: keep

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
Decide whether `t` lies in the closed `L2`-ball of radius `ε` around `center`.
-/
def inL2BallFloat {s : Shape} (center : Tensor Float s) (ε : Float) (t : Tensor Float s) : Bool
  :=
  tensorL2DistanceFloat center t ≤ ε

/--
Decide whether `t` lies in the closed `L∞`-ball of radius `ε` around `center`.
-/
def inLinfBallFloat {s : Shape} (center : Tensor Float s) (ε : Float) (t : Tensor Float s) : Bool
  :=
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

/-!
## Sampling design (why these helpers look “complicated”)

The spec tensor representation `TorchLean.Tensor α s` is **shape-indexed** and is represented
functionally (`Fin n → ...`). That is great for proofs, but it is not the easiest shape to work
with when you want to build *concrete perturbations* of a fixed length at runtime.

For sampling, we therefore go through a standard interop path:

1. compute the scalar count `numel s` from the tensor shape,
2. construct a flat array `xs : Array Float` of that length, then
3. use the checked flat-data constructor to obtain a tensor of shape `s`.

Correctness (what is and is not guaranteed):

- Every point returned by `sampleL2Ball` **provably satisfies** the predicate
  `inL2BallFloat center ε` because we *filter* candidates using that very predicate.
- The sampler is **deterministic** (no `IO` randomness). You control variability via the `seed`
  input (here derived from the loop index).
- The sampler is **not** intended to approximate a uniform distribution on the ball, and it does
  not attempt to be “complete” in any verification sense: it is purely an empirical exploration
  tool to produce inputs for downstream checks/counterexamples.
-/

/--
Number of scalar elements (“numel”) in a tensor of shape `s`.
-/
def numel (s : Shape) : Nat :=
  s.size

/--
Deterministically generate a length-`n` direction vector from a `seed`.

This is **not** cryptographic and not intended to model a probabilistic distribution; it is a
deterministic way to generate reproducible, varied directions without introducing `IO`.
-/
def perturbDirection (n : Nat) (seed : Nat) : Array Float :=
  -- Deterministic “pseudo-random-ish” direction: entry `i` depends on `seed` and `i`.
  -- We avoid `IO` randomness here; callers can treat `seed` as a reproducibility knob.
  Array.ofFn fun i : Fin n =>
    let a : Float := Float.ofNat (seed + 1) * Float.ofNat (i.1 + 1)
    Float.sin a + 0.5 * Float.cos (a + 1.0)

/-- Euclidean norm of a flat runtime array. -/
def l2NormArray (xs : Array Float) : Float :=
  Float.sqrt (xs.foldl (fun acc x => acc + x * x) 0.0)

/--
Normalize a flat array to unit `L2` norm (when possible).

If the list has zero norm, we return it unchanged.
-/
def normalizeArray (xs : Array Float) : Array Float :=
  let n := l2NormArray xs
  if n > 0.0 then xs.map (fun x => x / n) else xs

/--
Unflatten a flat array of length `numel s` into a `TorchLean.Tensor Float s`.

The length proof is part of the interface to avoid “silent truncation/padding”.
-/
def unflattenToTensor {s : Shape} (xs : Array Float)
    (h : xs.size = s.size) : Tensor Float s :=
  (Tensor.from xs).reshape s (by simpa [Shape.size] using h)

/--
Build a perturbation tensor of (approximately) the given `radius`, deterministically from `seed`.

Construction:

1. Make a flat direction array `base` of length `numel s`.
2. Normalize it to unit norm (when nonzero).
3. Scale by `radius`.
4. Unflatten back into the tensor shape.

This ensures the perturbation has the right shape by construction.
-/
def perturbationTensor {s : Shape} (radius : Float) (seed : Nat) : Tensor Float s :=
  let n := numel s
  let base : Array Float := perturbDirection n seed
  let dir : Array Float := normalizeArray base
  -- Scale direction and unflatten back into the tensor shape.
  let xs : Array Float := dir.map (fun x => radius * x)
  have hlen : xs.size = s.size := by
    have hbase : base.size = n := by simp [base, perturbDirection]
    have hdir : dir.size = n := by
      dsimp [dir]
      by_cases h : l2NormArray base > 0.0
      · simp [normalizeArray, h, hbase]
      · simp [normalizeArray, h, hbase]
    calc
      xs.size = dir.size := by simp [xs]
      _ = n := hdir
      _ = s.size := by simp [numel, n]
  unflattenToTensor (s := s) xs hlen

/--
Generate candidate samples in the closed $L^2$ ball around `center` (deterministic sampler).

We generate `numSamples` candidates, each constructed as:

$\mathrm{center}+\delta_k$, where $\delta_k$ is a direction derived from $k$ and then scaled to a
radius in $[0,\varepsilon]$.

We then **filter** candidates using `inL2BallFloat center ε` to ensure that every returned
element satisfies the predicate under the same runtime distance function.

This “generate + filter” style is deliberate: it makes the only factual guarantee we claim
extremely clear (“every returned element lies in the ball”), independent of the details of the
direction generator.
-/
def sampleL2Ball {s : Shape} (center : Tensor Float s) (ε : Float) (numSamples : Nat) :
    Array (Tensor Float s) :=
  (Array.range numSamples).foldl (fun acc k =>
    -- Choose a deterministic radius in `(0, ε)`; we avoid `0` so that, for typical shapes, we
    -- don't only resample the center.
    --
    -- The exact schedule is not semantically important; what matters is that:
    -- - candidates are easy to reproduce, and
    -- - every returned element is checked with `inL2BallFloat`.
    let radius := ε * (Float.ofNat (k + 1) / Float.ofNat (numSamples + 1))
    let δ := perturbationTensor (s := s) radius k
    let x := TorchLean.Tensor.addSpec center δ
    if inL2BallFloat center ε x then acc.push x else acc) #[]

/--
Turn an array with at least two elements `#[x₀,x₁,…,x_{m-1}]` into adjacent pairs
`[(x₀,x₁),(x₁,x₂),…,(x_{m-1},x₀)]`.

This is a small combinator that is useful when you want to turn a sample array into a set of
“nearby pairs” for empirical ratio computations. Empty and singleton arrays return no pairs.
-/
def adjacentPairs {α : Type} [Inhabited α] (xs : Array α) : Array (α × α) :=
  if xs.size < 2 then #[]
  else
    (Array.range xs.size).map fun i =>
      (xs[i]!, xs[(i + 1) % xs.size]!)

end Sampling

/--
Empirical max “gain from the center” on a sampled $L^2$-ball neighborhood.

This computes:

$$
\max_{x\in\mathrm{samples}}
\frac{\lVert f(x)-f(x_0)\rVert_2}{\lVert x-x_0\rVert_2}
$$

where `samples` are generated by `Sampling.sampleL2Ball`. This is a maximum over those samples
only (not a certified global Lipschitz bound).
-/
def empiricalMaxL2GainFromSamples {s₁ s₂ : Shape}
    (f : Tensor Float s₁ → Tensor Float s₂)
    (x₀ : Tensor Float s₁) (ε : Float) (numSamples : Nat) : Except String Float :=
  let samples := Sampling.sampleL2Ball (center := x₀) ε numSamples
  -- We compare each sample to the center point `x₀`, because that is the common pattern in
  -- robustness debugging (“how much can the output move within an ε-neighborhood of x₀?”).
  let pairs := samples.map (fun x => (x₀, x))
  Empirical.maxL2LipschitzRatio (s₁ := s₁) (s₂ := s₂) f pairs

end NN.MLTheory.Robustness.Runtime
