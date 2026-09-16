/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.RL.Core
public import NN.Spec.Models.Gmm
public import NN.Tests.Utils

/-!
# Probability Model Contracts

Regression checks for GMM parameter validation at empty-data boundaries and categorical KL
normalization when its probability guard changes total mass.
-/

public section

namespace Tests.Floats.ProbabilityContracts

open TorchLean

private def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError label

private def emptyCalls {k p : Nat} (model : Spec.GMMSpec Float k p) (hk : k ≠ 0) :
    Array (String × Bool) :=
  let empty : Tensor Float [0, p] := Tensor.dim fun i => nomatch i
  #[("forward", (Spec.gmmBatchedForwardSpec model empty).isSome),
    ("responsibilities", (Spec.gmmResponsibilitiesBatchedSpec model empty hk).isSome),
    ("negative log likelihood", (Spec.gmmNegLogLikelihoodBatchedSpec model empty hk).isSome),
    ("EM step", (Spec.gmmEmStepSpec model empty hk).isSome),
    ("zero-epoch EM", (Spec.gmmEmTrainSpec 0 model empty hk).isSome)]

/-- Empty batches and zero epochs preserve valid GMMs but must not hide invalid parameters. -/
def checkGmmBoundaries : IO Unit := do
  let valid : Spec.GMMSpec Float 1 1 := Spec.gmmInitSpec
  let singular : Spec.GMMSpec Float 1 1 :=
    { valid with covariances := Tensor.full [1, 1, 1] 0 }
  let zeroWeight : Spec.GMMSpec Float 1 1 :=
    { valid with weights := Tensor.full [1] 0 }
  let unnormalized : Spec.GMMSpec Float 1 1 :=
    { valid with weights := Tensor.full [1] 2 }
  let noFeatures : Spec.GMMSpec Float 1 0 := Spec.gmmInitSpec
  let one : Tensor Float [1, 1] := Tensor.full [1, 1] 0
  let input : Tensor Float [1] := Tensor.full [1] 0
  for (label, model) in #[("singular", singular), ("zero-weight", zeroWeight),
      ("unnormalized", unnormalized)] do
    expect (label ++ ": single sample rejects invalid model")
      (Spec.gmmForwardSpec model input).isNone
    expect (label ++ ": nonempty batch rejects invalid model")
      (Spec.gmmBatchedForwardSpec model one).isNone
    for (operation, accepted) in emptyCalls model (by decide) do
      expect (label ++ ": empty " ++ operation ++ " rejects invalid model") (!accepted)
    expect (label ++ ": zero epochs with nonempty data reject invalid model")
      (Spec.gmmEmTrainSpec 0 model one (by decide)).isNone
  for (operation, accepted) in emptyCalls noFeatures (by decide) do
    expect ("zero features: empty " ++ operation ++ " rejects invalid model") (!accepted)
  let noComponents : Spec.GMMSpec Float 0 1 := Spec.gmmInitSpec
  let empty : Tensor Float [0, 1] := Tensor.dim fun i => nomatch i
  expect "zero components: empty forward rejects invalid model"
    (Spec.gmmBatchedForwardSpec noComponents empty).isNone
  for (operation, accepted) in emptyCalls valid (by decide) do
    expect ("valid empty " ++ operation ++ " succeeds") accepted
  expect "valid empty negative log likelihood is zero"
    ((Spec.gmmNegLogLikelihoodBatchedSpec valid empty (by decide)).getD 99 == 0)
  for result in #[Spec.gmmEmStepSpec valid empty (by decide),
      Spec.gmmEmTrainSpec 0 valid one (by decide)] do
    match result with
    | none => throw <| IO.userError "valid no-op training rejected"
    | some model =>
        expect "no-op weights preserved" (Tensor.getScalar model.weights ⟨0, by decide⟩ == 1)
        expect "no-op means preserved"
          (Tensor.getScalar (Tensor.unstack model.means ⟨0, by decide⟩) ⟨0, by decide⟩ == 0)
        expect "no-op covariances preserved"
          (Tensor.get2 (Tensor.unstack model.covariances ⟨0, by decide⟩)
            ⟨0, by decide⟩ ⟨0, by decide⟩ == 1)
  expect "valid nonempty forward succeeds" (Spec.gmmBatchedForwardSpec valid one).isSome
  expect "valid nonempty responsibilities succeed"
    (Spec.gmmResponsibilitiesBatchedSpec valid one (by decide)).isSome
  expect "valid nonempty likelihood succeeds"
    (Spec.gmmNegLogLikelihoodBatchedSpec valid one (by decide)).isSome
  expect "valid nonempty EM succeeds"
    (Spec.gmmEmTrainSpec 1 valid one (by decide)).isSome

/-- Guarded KL uses normalized distributions, including at support and dimension boundaries. -/
def checkCategoricalKL : IO Unit := do
  let old : Tensor Float [3] := [0.8, 0.1, 0.1]
  let new : Tensor Float [3] := [0.98, 0.01, 0.01]
  let score := rl.policy.categoricalKL old new 0.1
  -- The guarded distributions are (4/5, 1/10, 1/10) and (9/11, 1/11, 1/11).
  let expected := 0.8 * Float.log (44.0 / 45.0) + 0.2 * Float.log 1.1
  Tests.Utils.assertApprox "normalized guarded categorical KL" score expected 1e-12
  expect "unequal guarded distributions have positive KL" (score > 0)
  Tests.Utils.assertApprox "same guarded policy"
    (rl.policy.categoricalKL new new 0.1) 0 1e-12
  let interiorOld : Tensor Float [3] := [0.5, 0.25, 0.25]
  let interiorNew : Tensor Float [3] := [0.25, 0.5, 0.25]
  Tests.Utils.assertApprox "interior ordinary KL"
    (rl.policy.categoricalKL interiorOld interiorNew) (0.25 * Float.log 2) 1e-12
  let pointOld : Tensor Float [3] := [1, 0, 0]
  let pointNew : Tensor Float [3] := [0, 1, 0]
  Tests.Utils.assertApprox "disjoint support guarded KL"
    (rl.policy.categoricalKL pointOld pointNew 0.1) ((8.0 / 11.0) * Float.log 9) 1e-12
  let oldLogits : Tensor Float [3] :=
    [Float.log 0.8, Float.log 0.1, Float.log 0.1]
  let newLogits : Tensor Float [3] :=
    [Float.log 0.98, Float.log 0.01, Float.log 0.01]
  Tests.Utils.assertApprox "logits facade"
    (rl.policy.categoricalKLFromLogits oldLogits newLogits 0.1) expected 1e-12
  let extremeOld : Tensor Float [3] := [1000, -1000, -1000]
  let extremeNew : Tensor Float [3] := [-1000, 1000, -1000]
  Tests.Utils.assertApprox "extreme logits guard"
    (rl.policy.categoricalKLFromLogits extremeOld extremeNew 0.1)
    ((8.0 / 11.0) * Float.log 9) 1e-12
  let single : Tensor Float [1] := [1]
  Tests.Utils.assertApprox "single action KL" (rl.policy.categoricalKL single single) 0 1e-12
  let empty : Tensor Float [0] := Tensor.dim fun i => nomatch i
  Tests.Utils.assertApprox "empty action KL" (rl.policy.categoricalKL empty empty) 0 1e-12
  let old32 : Tensor Float32 [3] := [0.8, 0.1, 0.1]
  let new32 : Tensor Float32 [3] := [0.98, 0.01, 0.01]
  Tests.Utils.assertApprox "float32 normalized guarded KL"
    (rl.policy.categoricalKL old32 new32 0.1).toFloat expected 1e-6

/-- Run the probability-model boundary regressions in the maintained Float suite. -/
def run : IO Unit := do
  checkGmmBoundaries
  checkCategoricalKL
  IO.println "Probability model contract checks passed"

end Tests.Floats.ProbabilityContracts
