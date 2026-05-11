/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Runtime
public import NN.MLTheory.LearningTheory.Stability.Dynamics.Spec
public import NN.Spec.Core.Context
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor
public import NN.Spec.Models.Mlp

/-!
# `NN.MLTheory.Stability.Runtime`

Executable Float-specialized diagnostics for the stability specifications in
`NN.MLTheory.Stability.Spec`.
-/

@[expose] public section

open Spec

namespace NN.MLTheory.Stability.Runtime

/-!
# Stability runtime utilities (Float)

This module provides executable, Float-specialized helpers for exploring stability properties of
discrete-time systems (`x_{t+1} = f x_t`).

All routines in this file are **runtime diagnostics**:

- they compute concrete trajectories and check concrete inequalities, and
- they return `Bool` for convenience in scripts/CLIs.

They are factually correct as *computations* (“this inequality held on these sampled points for
this many steps”), but they are **not proofs** of the corresponding `Prop` definitions in
`NN.MLTheory.Stability.Spec`.

In other words:

- `true` means “passed these tests”, not “theorem proved”;
- `false` means “found a counterexample to the tested condition”.
-/

open NN.MLTheory.Robustness.Runtime

/--
Empirical Lyapunov stability test:

For each initial point `x₀` in `initial_points`, generate a length-`max_iterations` trajectory and
check that every state stays within `tolerance` (in `L2` distance) of `equilibrium`.

This is a **bounded-time** and **finite-set** check; it does not certify Lyapunov stability.
-/
def testLyapunovStability {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (initial_points : List (Tensor Float s))
    (max_iterations : Nat)
    (tolerance : Float) : Bool :=
  initial_points.all (fun x₀ =>
    test_trajectory_bounded f equilibrium x₀ max_iterations tolerance)
where
  test_trajectory_bounded (f : Tensor Float s → Tensor Float s)
      (eq : Tensor Float s) (x₀ : Tensor Float s) (max_iter : Nat) (tol : Float) : Bool :=
    let trajectory := List.range max_iter |>.scanl (fun x _ => f x) x₀
    trajectory.all (fun x => tensorL2DistanceFloat eq x ≤ tol)

/--
Generate the first `steps` iterates of a discrete-time system `x_{t+1} = f x_t`, starting at `x₀`.

The returned list includes the initial state `x₀` as the first element (because we use `scanl`).
-/
def generateTrajectory {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (x₀ : Tensor Float s)
    (steps : Nat) : List (Tensor Float s) :=
  List.range steps |>.scanl (fun x _ => f x) x₀

/--
Empirical asymptotic stability test (finite-horizon):

For each `x₀` in `initial_points`, simulate `max_iterations` steps and check that the final state
is within `convergence_threshold` of `equilibrium` (in `L2` distance).

This is a very coarse check: it only inspects the *last* iterate and does not quantify a rate.
-/
def testAsymptoticStability {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (initial_points : List (Tensor Float s))
    (max_iterations : Nat)
    (convergence_threshold : Float) : Bool :=
  initial_points.all (fun x₀ =>
    let trajectory := generateTrajectory f x₀ max_iterations
    let final_distance := tensorL2DistanceFloat equilibrium (trajectory.getLast!)
    final_distance ≤ convergence_threshold)

/--
Empirical exponential decay test:

We simulate a trajectory, compute distances `d_t = ‖x_t - equilibrium‖₂`, and check a simple
inequality of the form

`d_t ≤ d_0 * exp(-rate * t)`

for the given `expected_decay_rate`.

This is a heuristic diagnostic; it is not a proof of exponential stability.
-/
def testExponentialStability {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (x₀ : Tensor Float s)
    (expected_decay_rate : Float)
    (max_iterations : Nat) : Bool :=
  let trajectory := generateTrajectory f x₀ max_iterations
  let distances := trajectory.map (tensorL2DistanceFloat equilibrium)
  test_exponential_decay distances expected_decay_rate
where
  test_exponential_decay (distances : List Float) (rate : Float) : Bool :=
    match distances with
    | [] => true
    | d₀ :: rest =>
      rest.zip (List.range rest.length) |>.all (fun (d, n) =>
        d ≤ d₀ * Float.exp (-rate * Float.ofNat (n + 1)))

/--
Empirical contractivity test on a finite list of input pairs.

Checks the inequality

`‖f x - f y‖₂ / ‖x - y‖₂ ≤ expected_contraction_factor`

for each pair `(x,y)` in `test_pairs`, ignoring pairs with zero input distance.
-/
def testContractivity {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (test_pairs : List (Tensor Float s × Tensor Float s))
    (expected_contraction_factor : Float) : Bool :=
  test_pairs.all (fun (x, y) =>
    let input_dist := tensorL2DistanceFloat x y
    let output_dist := tensorL2DistanceFloat (f x) (f y)
    if input_dist > 0.0 then
      output_dist / input_dist ≤ expected_contraction_factor
    else true)

/--
Empirical BIBO stability test on a finite list of inputs.

For each test input `x`, if `‖x‖₂ ≤ input_bound` then we check `‖f x‖₂ ≤ output_bound`.
-/
def testBiboStability {s₁ s₂ : Shape}
    (f : Tensor Float s₁ → Tensor Float s₂)
    (test_inputs : List (Tensor Float s₁))
    (input_bound : Float)
    (output_bound : Float) : Bool :=
  test_inputs.all (fun x =>
    let input_norm := tensorL2NormFloat x
    let output_norm := tensorL2NormFloat (f x)
    input_norm ≤ input_bound → output_norm ≤ output_bound)

/--
Empirical monotonic-loss check for a training log.

Returns `true` if each consecutive loss satisfies `l_{t+1} ≤ l_t + tolerance`.
-/
def testTrainingStability
    (loss_sequence : List Float)
    (tolerance : Float) : Bool :=
  match loss_sequence with
  | [] => true
  | [_] => true
  | l₁ :: l₂ :: rest =>
    (l₂ ≤ l₁ + tolerance) ∧ testTrainingStability (l₂ :: rest) tolerance

/--
Empirical estimate of a Lyapunov-style stability margin.

For each candidate radius `r` in `test_radii`, we generate a small finite set of points on a
synthetic “sphere” of radius `r` around `equilibrium` and check a bounded-horizon Lyapunov test.
The result is the maximum radius that passes these finite checks.
-/
def estimateStabilityMargin {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (test_radii : List Float)
    (max_iterations : Nat) : Float :=
  let stable_radii := test_radii.filter (fun r =>
    let test_points := generate_points_on_sphere equilibrium r 8
    testLyapunovStability f equilibrium test_points max_iterations r)
  stable_radii.foldl max 0.0
where
  generate_points_on_sphere {s : Shape} (center : Tensor Float s) (radius : Float) (count : Nat) :
    List (Tensor Float s) :=
    List.range count |>.map (fun i =>
      let angle := Float.ofNat i * 2.0 * 3.14159 / Float.ofNat count
      add_spherical_perturbation center radius angle)

  add_spherical_perturbation {s : Shape} (center : Tensor Float s) (radius : Float) (angle : Float)
    : Tensor Float s :=
    match s with
    | .scalar => Spec.Tensor.addSpec center (.scalar (radius * Float.cos angle))
    | .dim _ _ => match center with
      | .dim f => .dim (fun i =>
        let local_angle := angle + Float.ofNat i.val * 0.1
        add_spherical_perturbation (f i) radius local_angle)

/-- Results of a small battery of empirical stability diagnostics. -/
structure StabilityAnalysisResult where
  /-- Result of a finite-horizon Lyapunov test (`test_lyapunov_stability`). -/
  isLyapunovStable : Bool
  /-- Result of a finite-horizon asymptotic test (`test_asymptotic_stability`). -/
  isAsymptoticallyStable : Bool
  /-- Result of an empirical contractivity test (`test_contractivity`). -/
  isContractive : Bool
  /-- Result of a BIBO check (`test_bibo_stability`). -/
  isBiboStable : Bool
  /-- Empirical stability margin estimate (`estimate_stability_margin`). -/
  stabilityMargin : Float
  /-- Empirical convergence-rate estimate (see `analyze_stability`). -/
  convergence_rate : Float

/--
Run a small collection of empirical stability diagnostics and summarize the results.
-/
def analyzeStability {s : Shape}
    (f : Tensor Float s → Tensor Float s)
    (equilibrium : Tensor Float s)
    (test_points : List (Tensor Float s))
    (max_iterations : Nat) : StabilityAnalysisResult :=
  match test_points with
  | [] =>
      { isLyapunovStable := true
        isAsymptoticallyStable := true
        isContractive := true
        isBiboStable := true
        stabilityMargin := estimateStabilityMargin f equilibrium [0.01, 0.05, 0.1, 0.2]
          max_iterations
        convergence_rate := 0.0 }
  | x0 :: rest =>
      let test_pairs := List.zip (x0 :: rest) (rest ++ [x0])
      { isLyapunovStable := testLyapunovStability f equilibrium (x0 :: rest) max_iterations 0.1
        isAsymptoticallyStable := testAsymptoticStability f equilibrium (x0 :: rest)
          max_iterations 0.01
        isContractive := testContractivity f test_pairs 0.9
        isBiboStable := testBiboStability (fun x => f x) (x0 :: rest) 1.0 1.0
        stabilityMargin := estimateStabilityMargin f equilibrium [0.01, 0.05, 0.1, 0.2]
          max_iterations
        convergence_rate := estimate_convergence_rate f equilibrium x0 max_iterations }
where
  estimate_convergence_rate (f : Tensor Float s → Tensor Float s)
      (eq : Tensor Float s) (x₀ : Tensor Float s) (steps : Nat) : Float :=
    let trajectory := generateTrajectory f x₀ steps
    let distances := trajectory.map (tensorL2DistanceFloat eq)
    match distances with
    | d₀ :: d₁ :: _ => if d₀ > 0.0 ∧ d₁ > 0.0 then -Float.log (d₁ / d₀) else 0.0
    | _ => 0.0

end NN.MLTheory.Stability.Runtime
