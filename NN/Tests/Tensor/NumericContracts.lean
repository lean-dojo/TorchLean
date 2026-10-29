/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context.Rational
public import NN.Spec.Core.FloatInstances
public import NN.Spec.Models.Svm
public import NN.Spec.Module.GruModels
public import NN.Runtime.Autograd.Model.Dual
public import NN.Tensor

/-!
# Numeric model contract regressions

These fixtures distinguish constant-construction overflow, activation argument order, loss
normalization, and a nonzero recurrent initial state. Expected gradients come from scalar
formulas and objective differences, independently of the backward implementations.
-/

@[expose] public section

namespace NN.Tests.Tensor.NumericContracts

open Spec TorchLean
open FloatLib.Floats
open scoped Spec.RationalAlgebraic

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"numeric contract: {label}"

/-- A finite logarithm and tangent must survive an unrepresentable intermediate sum.
The second format cannot represent `2`, so a repair cannot divide by a scalar `2` either. -/
def checkLog1pRange : IO Unit := do
  let value : ExecFloat.Binary 2 1 := Context.log1p 3
  expect "tiny log1p keeps the certified finite result"
    (ExecFloat.Binary.toRat? value == some (3 / 2 : Rat))
  let differentiated : Runtime.Autograd.Model.Dual (ExecFloat.Binary 2 1) := Context.log1p ⟨3, 2⟩
  expect "tiny log1p chain rule replays the complete denominator"
    (ExecFloat.Binary.toRat? differentiated.re == some (3 / 2 : Rat) &&
      ExecFloat.Binary.toRat? differentiated.du == some (1 / 2 : Rat))
  let biased : Runtime.Autograd.Model.Dual (ExecFloat.Binary 2 1 (bias := 2)) :=
    Context.log1p ⟨1, 1⟩
  expect "log1p handles a custom bias without representing two"
    (ExecFloat.Binary.toRat? biased.re == some (3 / 4 : Rat) &&
      ExecFloat.Binary.toRat? biased.du == some (1 / 2 : Rat))
  let small : Float32 := 0.00000001
  expect "log1p retains its small-argument path" ((Context.log1p small).toBits == small.toBits)
  expect "log1p keeps invalid real arguments invalid"
    (ExecFloat.Binary.isNaN (Context.log1p (-2 : ExecFloat.Binary 2 1)))

/-- Binary16 must round the coefficient itself, without first overflowing its denominator. -/
def checkGeluCoefficient : IO Unit := do
  let coefficient : ExecFloat.Binary 5 10 := Activation.Math.geluTanhCoeff
  expect "binary16 GELU coefficient bits" (ExecFloat.Binary.toNatBits coefficient == 0x29b9)
  expect "binary16 GELU coefficient exact value"
    (ExecFloat.Binary.toRat? coefficient == some (1465 / 32768 : Rat))
  let oldCoefficient : ExecFloat.Binary 5 10 :=
    ((44715 : Nat) : ExecFloat.Binary 5 10) / ((1000000 : Nat) : ExecFloat.Binary 5 10)
  expect "binary16 separate integer casts reproduce zero"
    (ExecFloat.Binary.toRat? oldCoefficient == some 0)
  let exact : Rat := Activation.Math.geluTanhCoeff
  expect "exact GELU coefficient" (exact == 44715 / 1000000)
  let native32 : Float32 := Activation.Math.geluTanhCoeff
  let native64 : Float := Activation.Math.geluTanhCoeff
  expect "native32 GELU coefficient preserves the previous result"
    (native32.toBits == (((44715 : Nat) : Float32) / ((1000000 : Nat) : Float32)).toBits)
  expect "native64 GELU coefficient preserves the previous result"
    (native64.toBits == (((44715 : Nat) : Float) / ((1000000 : Nat) : Float)).toBits)
  let value := Activation.Math.geluSpec (1 : ExecFloat.Binary 5 10)
  let derivative := Activation.Math.geluDerivSpec (1 : ExecFloat.Binary 5 10)
  expect "binary16 GELU forward retains the cubic correction"
    ((ExecFloat.Binary.toRat? value).any fun q => 835 / 1000 < q && q < 850 / 1000)
  expect "binary16 GELU derivative retains the cubic correction"
    ((ExecFloat.Binary.toRat? derivative).any fun q => 1075 / 1000 < q && q < 1095 / 1000)

/-- Distinct slopes and inputs expose accidental partial application in the tensor wrappers. -/
def checkActivationArguments : IO Unit := do
  let negative : Tensor Float [1] := [-2.0]
  let zero : Tensor Float [1] := [0.0]
  expect "leaky ReLU input before slope"
    ((Activation.leakyReluSpec negative 0.25).to (Array Float) == #[-0.5])
  expect "leaky ReLU derivative input before slope"
    ((Activation.leakyReluDerivSpec negative 0.25).to (Array Float) == #[0.25])
  expect "ELU zero input with alpha two"
    ((Activation.eluSpec zero 2.0).to (Array Float) == #[0.0])
  expect "ELU derivative zero input with alpha two"
    ((Activation.eluDerivSpec zero 2.0).to (Array Float) == #[2.0])

/-- A large configured epsilon must not change a dataset mean's gradient denominator. -/
def checkSvmDenominator : IO Unit :=
  letI : Context Rat :=
    { Spec.RationalAlgebraic.instContextRat with defaultEpsilon := 2 }
  do
    let model : LinearSVM 1 Rat := { w := [0], b := 0 }
    let inputs : Tensor Rat [1, 1] := [[1]]
    let labels : Tensor Rat [1] := [1]
    let (dw, db, _) := LinearSVM.backward 0 model inputs labels
    let base := model.objective 0 inputs labels
    let weightStep : LinearSVM 1 Rat := { model with w := [1 / 4] }
    let biasStep : LinearSVM 1 Rat := { model with b := 1 / 4 }
    expect "SVM exact active hinge objective" (base == 1)
    expect "SVM weight gradient matches objective difference"
      (dw.to (Array Rat) == #[-1] &&
        (weightStep.objective 0 inputs labels - base) / (1 / 4) == -1)
    expect "SVM bias gradient matches objective difference"
      (db == -1 && (biasStep.objective 0 inputs labels - base) / (1 / 4) == -1)
    let (_, _, dx) := LinearSVM.backward 0 weightStep inputs labels
    let inputStep : Tensor Rat [1, 1] := [[5 / 4]]
    expect "SVM input gradient matches objective difference"
      (dx.to (Array Rat) == #[-1 / 4] &&
        (weightStep.objective 0 inputStep labels - weightStep.objective 0 inputs labels) /
          (1 / 4) == -1 / 4)
    let emptyInputs : Tensor Rat [0, 1] := Tensor.zeros [0, 1]
    let emptyLabels : Tensor Rat [0] := Tensor.zeros [0]
    let (emptyDw, emptyDb, emptyDx) := LinearSVM.backward 0 model emptyInputs emptyLabels
    expect "SVM empty mean and gradients"
      (model.objective 0 emptyInputs emptyLabels == 0 &&
        emptyDw.to (Array Rat) == #[0] && emptyDb == 0 &&
        (emptyDx.to (Array Rat)).isEmpty)

def zeroGru : GRUSpec Float 1 1 :=
  { resetWeight := Tensor.zeros [1, 2]
    resetBias := Tensor.zeros [1]
    updateWeight := Tensor.zeros [1, 2]
    updateBias := Tensor.zeros [1]
    candidateWeight := Tensor.zeros [1, 2]
    candidateBias := Tensor.zeros [1] }

/-- With zero gate parameters and initial hidden one, the update-bias and candidate recurrent
weight gradients are both 1/4. -/
def checkGruInitialState : IO Unit := do
  let model : Gru.Model Float 1 1 1 :=
    { gru := zeroGru, outputLayer := { weights := [[1.0]], bias := [0.0] } }
  let inputs : Tensor Float [1, 1] := [[0.0]]
  let initial : Tensor Float [1] := [1.0]
  let outputGrad : Tensor Float [1, 1] := [[1.0]]
  let (hiddenStates, reset, update, candidate, _) :=
    gruExtractIntermediateValues model.gru inputs initial
  let (gradient, _) :=
    model.backward inputs hiddenStates outputGrad reset update candidate (by decide) initial
  expect "GRU nonzero initial forward state" (hiddenStates.to (Array Float) == #[0.5])
  expect "GRU nonzero initial update bias"
    (gradient.cell.updateBias.to (Array Float) == #[0.25])
  expect "GRU nonzero initial candidate recurrent weight"
    (gradient.cell.candidateWeight.to (Array Float) == #[0.0, 0.25])
  let step : Float := 0.0001
  let plus : Gru.Model Float 1 1 1 :=
    { model with gru := { model.gru with updateBias := [step] } }
  let minus : Gru.Model Float 1 1 1 :=
    { model with gru := { model.gru with updateBias := [-step] } }
  let plusOutput := (plus.forwardSequence inputs initial).1
  let minusOutput := (minus.forwardSequence inputs initial).1
  let numerical := ((plusOutput.to (Array Float))[0]! -
    (minusOutput.to (Array Float))[0]!) / (2 * step)
  expect "GRU update bias agrees with forward finite difference"
    ((numerical - 0.25).abs < 1e-8)
  let (zeroHidden, zeroReset, zeroUpdate, zeroCandidate, _) :=
    gruExtractIntermediateValues model.gru inputs (Tensor.zeros [1])
  let (zeroGradient, _) :=
    model.backward inputs zeroHidden outputGrad zeroReset zeroUpdate zeroCandidate (by decide)
  expect "GRU omitted initial state retains zero-state gradients"
    (zeroGradient.cell.updateBias.to (Array Float) == #[0.0] &&
      zeroGradient.cell.candidateWeight.to (Array Float) == #[0.0, 0.0])

def run : IO Unit := do
  checkLog1pRange
  checkGeluCoefficient
  checkActivationArguments
  checkSvmDenominator
  checkGruInitialState
  IO.println "  numeric model contracts: passed"

end NN.Tests.Tensor.NumericContracts
