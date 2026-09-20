/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Runtime

/-!
# `NN.MLTheory.Stability.Runtime`

Executable Float-specialized diagnostics for the stability specifications in
`NN.MLTheory.Stability.Spec`.
-/

@[expose] public section

open Spec TorchLean

namespace NN.MLTheory.Stability.Runtime

/-!
# Stability runtime utilities (Float)

This module provides executable, Float-specialized helpers for exploring stability properties of
discrete-time systems ($x_{t+1}=f(x_t)$).

All routines in this file are **runtime diagnostics**:

- they compute concrete trajectories and check concrete inequalities, and
- they return `Except String` so absent or non-finite evidence cannot pass vacuously.

They are factually correct as *computations*: “this inequality held on these sampled points for
this many steps.” The corresponding theorem statements live in the `Prop` definitions in
`NN.MLTheory.Stability.Spec`.

In other words:

- `.ok true` means the sampled runtime diagnostic passed;
- `.ok false` means “found a counterexample to the tested condition”; and
- `.error _` means the evidence was empty, non-finite, or otherwise inconclusive.
-/

open NN.MLTheory.Robustness.Runtime

/--
Apply `steps` transitions of a discrete-time system $x_{t+1}=f(x_t)$, starting at $x_0$.

The returned array has `steps + 1` states, including the initial state $x_0$.
-/
def generateTrajectory {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (x₀ : Tensor Float s)
    (steps : Nat) : Array (Tensor Float s) := Id.run do
  let mut trajectory := #[x₀]
  let mut x := x₀
  for _ in [0:steps] do
    x := f x
    trajectory := trajectory.push x
  return trajectory

/--
Empirical Lyapunov stability test:

For each initial point `x₀` in `initialPoints`, generate `maxIterations + 1` states and
check that every state stays within `tolerance` (in `L2` distance) of `equilibrium`.

This is a **bounded-time** and **finite-set** check; it does not certify Lyapunov stability.
-/
def testLyapunovStability {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (initialPoints : Array (Tensor Float s))
    (maxIterations : Nat)
    (tolerance : Float) : Except String Bool := do
  if initialPoints.isEmpty then
    throw "Lyapunov diagnostic requires at least one initial point"
  unless tolerance.isFinite && 0.0 ≤ tolerance do
    throw "Lyapunov diagnostic requires a finite nonnegative tolerance"
  let mut passed := true
  for x₀ in initialPoints do
    let trajectory := generateTrajectory f x₀ maxIterations
    for x in trajectory do
      let distance := tensorL2DistanceFloat equilibrium x
      unless distance.isFinite do
        throw "Lyapunov diagnostic encountered a non-finite distance"
      if !(distance ≤ tolerance) then
        passed := false
  pure passed

/--
Empirical asymptotic stability test (finite-horizon):

For each `x₀` in `initialPoints`, simulate `maxIterations` steps and check that the final state
is within `convergenceThreshold` of `equilibrium` (in `L2` distance).

This is a very coarse check: it only inspects the *last* iterate and does not quantify a rate.
-/
def testAsymptoticStability {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (initialPoints : Array (Tensor Float s))
    (maxIterations : Nat)
    (convergenceThreshold : Float) : Except String Bool := do
  if initialPoints.isEmpty then
    throw "asymptotic-stability diagnostic requires at least one initial point"
  unless convergenceThreshold.isFinite && 0.0 ≤ convergenceThreshold do
    throw "asymptotic-stability diagnostic requires a finite nonnegative threshold"
  let mut passed := true
  for x₀ in initialPoints do
    let trajectory := generateTrajectory f x₀ maxIterations
    let finalState := trajectory.getD (trajectory.size - 1) x₀
    let finalDistance := tensorL2DistanceFloat equilibrium finalState
    unless finalDistance.isFinite do
      throw "asymptotic-stability diagnostic encountered a non-finite distance"
    if !(finalDistance ≤ convergenceThreshold) then
      passed := false
  pure passed

/--
Empirical exponential decay test:

We simulate a trajectory, compute distances
$d_t=\lVert x_t-\mathrm{equilibrium}\rVert_2$, and check a simple
inequality of the form

$$
d_t\leq d_0 e^{-\mathrm{rate}\,t}
$$

for the given `expectedDecayRate`.

This is a heuristic diagnostic. A theorem about exponential stability should state the dynamical
hypotheses separately and use this run only as runtime evidence.
-/
def testExponentialStability {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (x₀ : Tensor Float s)
    (expectedDecayRate : Float)
    (maxIterations : Nat) : Except String Bool := do
  if maxIterations = 0 then
    throw "exponential-stability diagnostic requires at least one transition"
  unless expectedDecayRate.isFinite && 0.0 ≤ expectedDecayRate do
    throw "exponential-stability diagnostic requires a finite nonnegative decay rate"
  let trajectory := generateTrajectory f x₀ maxIterations
  let distances := trajectory.map (tensorL2DistanceFloat equilibrium)
  let some d₀ := distances[0]?
    | throw "exponential-stability diagnostic requires a nonempty trajectory"
  unless d₀.isFinite do
    throw "exponential-stability diagnostic encountered a non-finite distance"
  let mut passed := true
  for n in [1:distances.size] do
    let distance := distances[n]!
    let bound := d₀ * Float.exp (-expectedDecayRate * Float.ofNat n)
    unless distance.isFinite && bound.isFinite do
      throw "exponential-stability diagnostic encountered a non-finite value"
    if !(distance ≤ bound) then
      passed := false
  pure passed

/--
Empirical contractivity test on a finite list of input pairs.

Checks the inequality

$$
\frac{\lVert f(x)-f(y)\rVert_2}{\lVert x-y\rVert_2}
\leq \mathrm{expected\_contraction\_factor}
$$

for each pair `(x,y)`. Zero-distance pairs are ignored, but at least one positive-distance pair is
required.
-/
def testContractivity {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (testPairs : Array (Tensor Float s × Tensor Float s))
    (expectedContractionFactor : Float) : Except String Bool := do
  if testPairs.isEmpty then
    throw "contractivity diagnostic requires at least one input pair"
  unless expectedContractionFactor.isFinite do
    throw "contractivity diagnostic requires a finite contraction factor"
  let mut foundInformative := false
  let mut passed := true
  for (x, y) in testPairs do
    let inputDist := tensorL2DistanceFloat x y
    let outputDist := tensorL2DistanceFloat (f x) (f y)
    unless inputDist.isFinite && outputDist.isFinite do
      throw "contractivity diagnostic encountered a non-finite distance"
    if inputDist > 0.0 then
      let ratio := outputDist / inputDist
      unless ratio.isFinite do
        throw "contractivity diagnostic overflowed to a non-finite ratio"
      foundInformative := true
      if !(ratio ≤ expectedContractionFactor) then
        passed := false
    else if inputDist < 0.0 then
      throw "contractivity diagnostic encountered a negative input distance"
  unless foundInformative do
    throw "contractivity diagnostic requires a pair with positive input distance"
  pure passed

/--
Empirical BIBO stability test on a finite list of inputs.

For each test input $x$, if $\lVert x\rVert_2\leq\mathrm{input\_bound}$ then we check
$\lVert f(x)\rVert_2\leq\mathrm{output\_bound}$.
-/
def testBiboStability {s₁ s₂ : Shape}
    (f : Tensor Float s₁ → Tensor Float s₂)
    (testInputs : Array (Tensor Float s₁))
    (inputBound : Float)
    (outputBound : Float) : Except String Bool := do
  if testInputs.isEmpty then
    throw "BIBO diagnostic requires at least one test input"
  unless inputBound.isFinite && outputBound.isFinite &&
      0.0 ≤ inputBound && 0.0 ≤ outputBound do
    throw "BIBO diagnostic requires finite nonnegative bounds"
  let mut passed := true
  for x in testInputs do
    let inputNorm := tensorL2NormFloat x
    let outputNorm := tensorL2NormFloat (f x)
    unless inputNorm.isFinite && outputNorm.isFinite do
      throw "BIBO diagnostic encountered a non-finite norm"
    if inputNorm ≤ inputBound && !(outputNorm ≤ outputBound) then
      passed := false
  pure passed

/--
Empirical monotonic-loss check for a training log.

Returns `.ok true` if each consecutive loss satisfies
$\ell_{t+1}\leq\ell_t+\mathrm{tolerance}$. Fewer than two losses and non-finite values are
inconclusive errors.
-/
def testTrainingStability
    (lossSequence : Array Float)
    (tolerance : Float) : Except String Bool := do
  if lossSequence.size < 2 then
    throw "training-stability diagnostic requires at least two loss values"
  unless tolerance.isFinite && 0.0 ≤ tolerance do
    throw "training-stability diagnostic requires a finite nonnegative tolerance"
  let mut passed := true
  for i in [1:lossSequence.size] do
    let previous := lossSequence[i - 1]!
    let current := lossSequence[i]!
    let bound := previous + tolerance
    unless previous.isFinite && current.isFinite && bound.isFinite do
      throw "training-stability diagnostic encountered a non-finite loss or bound"
    if !(current ≤ bound) then
      passed := false
  pure passed

/--
Empirical estimate of a Lyapunov-style stability margin.

For each candidate radius `r` in `testRadii`, we generate a small finite set of points on a
coordinatewise cosine perturbation around `equilibrium` and check a bounded-horizon Lyapunov
test. The generated points are not generally on the L2 sphere of radius `r`.
The result is `some` maximum radius when at least one radius passes, and `none` when valid evidence
was collected but every radius failed. Empty or non-finite radius sets are errors.
-/
def estimateStabilityMargin {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (testRadii : Array Float)
    (maxIterations : Nat) : Except String (Option Float) := do
  if testRadii.isEmpty then
    throw "stability-margin diagnostic requires at least one candidate radius"
  let mut best : Option Float := none
  for r in testRadii do
    unless r.isFinite && 0.0 ≤ r do
      throw "stability-margin diagnostic requires finite nonnegative radii"
    let testPoints := generatePointsOnSphere equilibrium r 8
    if ← testLyapunovStability f equilibrium testPoints maxIterations r then
      best := some (best.elim r (max r))
  pure best
where
  generatePointsOnSphere {s : Shape} (center : Tensor Float s) (radius : Float) (count : Nat) :
    Array (Tensor Float s) :=
    Array.range count |>.map (fun i =>
      let angle := Float.ofNat i * 2.0 * 3.14159 / Float.ofNat count
      addSphericalPerturbation center radius angle)

  addSphericalPerturbation {s : Shape} (center : Tensor Float s) (radius : Float) (angle : Float)
    : Tensor Float s :=
    match s with
    | .scalar => TorchLean.Tensor.addSpec center (.scalar (radius * Float.cos angle))
    | .dim _ _ =>
      .dim (fun i =>
        let localAngle := angle + Float.ofNat i.val * 0.1
        addSphericalPerturbation (Tensor.unstack center i) radius localAngle)

/-- Results of a small battery of empirical stability diagnostics. -/
structure StabilityAnalysisResult where
  /-- Result of a finite-horizon Lyapunov test (`testLyapunovStability`). -/
  isLyapunovStable : Bool
  /-- Result of a finite-horizon asymptotic test (`testAsymptoticStability`). -/
  isAsymptoticallyStable : Bool
  /-- Result of an empirical contractivity test (`testContractivity`). -/
  isContractive : Bool
  /-- Result of a BIBO check (`testBiboStability`). -/
  isBiboStable : Bool
  /-- Empirical stability margin estimate (`estimateStabilityMargin`). -/
  stabilityMargin : Option Float
  /-- Empirical convergence-rate estimate (see `analyzeStability`). -/
  convergenceRate : Option Float
  deriving Repr

/--
Run a small collection of empirical stability diagnostics and summarize the results.
-/
def analyzeStability {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (testPoints : Array (Tensor Float s))
    (maxIterations : Nat) : Except String StabilityAnalysisResult := do
  let some x0 := testPoints[0]?
    | throw "stability analysis requires at least one test point"
  let testPairs := testPoints.mapIdx fun i x =>
    (x, testPoints.getD ((i + 1) % testPoints.size) x)
  let isLyapunovStable ← testLyapunovStability f equilibrium testPoints maxIterations 0.1
  let isAsymptoticallyStable ←
    testAsymptoticStability f equilibrium testPoints maxIterations 0.01
  let isContractive ← testContractivity f testPairs 0.9
  let isBiboStable ← testBiboStability (fun x => f x) testPoints 1.0 1.0
  let stabilityMargin ←
    estimateStabilityMargin f equilibrium #[0.01, 0.05, 0.1, 0.2] maxIterations
  let convergenceRate ← estimateConvergenceRate f equilibrium x0 maxIterations
  pure
    { isLyapunovStable
      isAsymptoticallyStable
      isContractive
      isBiboStable
      stabilityMargin
      convergenceRate }
where
  estimateConvergenceRate (f : Tensor Float s → Tensor Float s)
      (eq : Tensor Float s) (x₀ : Tensor Float s) (steps : Nat) : Except String (Option Float) := do
    let trajectory := generateTrajectory f x₀ steps
    let distances := trajectory.map (tensorL2DistanceFloat eq)
    match distances[0]?, distances[1]? with
    | some d₀, some d₁ =>
        unless d₀.isFinite && d₁.isFinite do
          throw "convergence-rate estimate encountered a non-finite distance"
        if d₀ > 0.0 && d₁ > 0.0 then
          let rate := -Float.log (d₁ / d₀)
          unless rate.isFinite do
            throw "convergence-rate estimate produced a non-finite rate"
          pure (some rate)
        else
          pure none
    | _, _ => pure none

end NN.MLTheory.Stability.Runtime
