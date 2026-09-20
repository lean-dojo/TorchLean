/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Stability.Dynamics.Runtime
public import NN.Tensor.Internal.Elab.TensorLiteral

/-!
# Empirical Diagnostic Regressions

Checks that empirical robustness and stability diagnostics distinguish failed finite checks from
empty or non-finite evidence.
-/

@[expose] public section

namespace NN.Tests.MLTheory.Diagnostics

open Spec TorchLean
open NN.MLTheory.Robustness.Runtime
open NN.MLTheory.Stability.Runtime

def expectError {α : Type} (label : String) : Except String α → IO Unit
  | .error _ => pure ()
  | .ok _ => throw <| IO.userError s!"{label}: expected an inconclusive/error result"

def expectBool (label : String) (expected : Bool) : Except String Bool → IO Unit
  | .ok actual =>
      unless actual = expected do
        throw <| IO.userError s!"{label}: expected {expected}, got {actual}"
  | .error message => throw <| IO.userError s!"{label}: unexpected error: {message}"

def run : IO Unit := do
  let zero : Tensor Float [] := Tensor.scalar 0.0
  let one : Tensor Float [] := Tensor.scalar 1.0
  let inf : Tensor Float [] := Tensor.scalar (1.0 / 0.0)
  let identity : Tensor Float [] → Tensor Float [] := fun x => x

  expectError "empty empirical Lipschitz evidence" <|
    Empirical.maxL2LipschitzRatio identity #[]
  expectError "zero-distance empirical Lipschitz evidence" <|
    Empirical.maxL2LipschitzRatio identity #[(zero, zero)]
  expectError "non-finite empirical Lipschitz evidence" <|
    Empirical.maxL2LipschitzRatio identity #[(zero, inf)]
  match Empirical.maxL2LipschitzRatio identity #[(zero, one)] with
  | .ok ratio =>
      unless ratio.isFinite && Float.abs (ratio - 1.0) ≤ 1e-6 do
        throw <| IO.userError s!"finite empirical Lipschitz ratio: expected 1, got {ratio}"
  | .error message =>
      throw <| IO.userError s!"finite empirical Lipschitz ratio: unexpected error: {message}"

  expectError "empty Lyapunov evidence" <|
    testLyapunovStability identity zero #[] 2 0.1
  expectError "non-finite Lyapunov evidence" <|
    testLyapunovStability identity inf #[zero] 2 0.1
  expectBool "finite Lyapunov counterexample" false <|
    testLyapunovStability identity zero #[one] 2 0.1
  expectError "empty asymptotic-stability evidence" <|
    testAsymptoticStability identity zero #[] 2 0.1
  expectError "empty exponential-stability horizon" <|
    testExponentialStability identity zero one 0.1 0
  expectError "empty contractivity evidence" <|
    testContractivity identity #[] 0.9
  expectError "zero-distance contractivity evidence" <|
    testContractivity identity #[(zero, zero)] 0.9
  expectBool "finite contractivity counterexample" false <|
    testContractivity identity #[(zero, one)] 0.9
  expectError "empty BIBO evidence" <|
    testBiboStability identity #[] 1.0 1.0
  expectError "short training-loss evidence" <|
    testTrainingStability #[1.0] 0.0
  expectError "non-finite training-loss evidence" <|
    testTrainingStability #[1.0, 0.0 / 0.0] 0.0
  expectError "empty aggregate stability evidence" <|
    analyzeStability identity zero #[] 2
  expectError "empty stability-margin evidence" <|
    estimateStabilityMargin identity zero #[] 2

end NN.Tests.MLTheory.Diagnostics
