/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Autodiff
public import NN.Runtime.RL.DQN.Autograd

/-!
# DQN Semi-gradient Loss Checks

Analytic checks cover the quadratic region, both linear tails, the join, target detachment, action
masking, and normalization over transitions. Float and Float32 exercise the same generic program.
-/

@[expose] public section

namespace Tests.Floats.DQN

open Spec TorchLean Runtime.Autograd.Model

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"DQN loss check failed: {label}"

def close {α : Type} [Context α] (actual expected : α) : Bool :=
  Context.gtBool (1 / 100000 : α) (MathFunctions.abs (actual - expected))

def vectorLoss (delta : Rat) (reduction : Loss.Reduction) :
    ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
      Program β [Shape.ofList [7], Shape.ofList [7]] [] :=
  fun {β} _ _ => fun {m} _ _ => fun prediction target =>
    Runtime.RL.DQN.Autograd.huberTDLoss (m := m) (α := β)
      prediction target (delta : β) reduction

def actionLoss :
    ∀ {β : Type}, [TorchLean.Storage β] → [Context β] →
      Program β [Shape.ofList [2, 3], Shape.ofList [2, 3], Shape.ofList [2]] [] :=
  fun {β} _ _ => fun {m} _ _ => fun prediction actions target =>
    Runtime.RL.DQN.Autograd.actionHuberLoss (m := m) (α := β)
      prediction actions target

def checkVector {α : Type} [TorchLean.Storage α] [Context α] : IO Unit := do
  let prediction : Tensor α [7] := Tensor.ofFn fun i =>
    ([(-4 : α), -2, -1, 0, 1, 2, 4] : List α)[i.val]!
  let target : Tensor α [7] := Tensor.zeros [7]
  for reduction in [Loss.Reduction.sum, .mean] do
    let divisor : α := if reduction == .mean then 7 else 1
    let graph ← Autodiff.lowerScalarToTypedGraph (α := α)
      (paramShapes := [Shape.ofList [7]]) (inputShapes := [Shape.ofList [7]])
      (vectorLoss 2 reduction)
    let (gradients, loss) ← Autodiff.Impl.vjpWithValue graph
      (.cons prediction (.cons target .nil)) (Tensor.scalar (1 : α))
    expect "Huber value with delta two" (close (loss.getFlat ⟨0, by decide⟩) (17 / divisor))
    let expected : Tensor α [7] := Tensor.ofFn fun i =>
      ([(-2 : α), -2, -1, 0, 1, 2, 2] : List α)[i.val]! / divisor
    for i in List.finRange 7 do
      expect "Huber prediction gradient"
        (close ((gradients.get ⟨0, by decide⟩).getFlat i) (expected.getFlat i))
      expect "Bellman target is detached" (close ((gradients.get ⟨1, by decide⟩).getFlat i) 0)

def checkActions {α : Type} [TorchLean.Storage α] [Context α] : IO Unit := do
  let prediction : Tensor α [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 =>
      ([(-3 : α), 100, -100, 100, 1 / 2, -100] : List α)[i.val]!).reshape [2, 3] (by decide)
  let actions : Tensor α [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 =>
      ([1, 0, 0, 0, 1, 0] : List α)[i.val]!).reshape [2, 3] (by decide)
  let target : Tensor α [2] := Tensor.zeros [2]
  let graph ← Autodiff.lowerScalarToTypedGraph (α := α)
    (paramShapes := [Shape.ofList [2, 3]])
    (inputShapes := [Shape.ofList [2, 3], Shape.ofList [2]]) actionLoss
  let (gradients, loss) ← Autodiff.Impl.vjpWithValue graph
    (.cons prediction (.cons actions (.cons target .nil))) (Tensor.scalar (1 : α))
  expect "mean over transitions" (close (loss.getFlat ⟨0, by decide⟩) (21 / 16))
  let expected : Tensor α [2, 3] :=
    (Tensor.ofFn fun i : Fin 6 =>
      ([(-1 / 2 : α), 0, 0, 0, 1 / 4, 0] : List α)[i.val]!).reshape [2, 3] (by decide)
  for i in List.finRange 6 do
    expect "only selected actions receive gradients"
      (close ((gradients.get ⟨0, by decide⟩).getFlat i) (expected.getFlat i))
    expect "action encoding is detached" (close ((gradients.get ⟨1, by decide⟩).getFlat i) 0)
  for i in List.finRange 2 do
    expect "selected-action target is detached" (close ((gradients.get ⟨2, by decide⟩).getFlat i) 0)

/-- Small quadratic-region residuals must retain their derivative without cancellation. -/
def checkSmallResidual {α : Type} [TorchLean.Storage α] [Context α] : IO Unit := do
  let small : α := 1 / 1073741824
  let prediction : Tensor α [7] := Tensor.ofFn fun _ => small
  let target : Tensor α [7] := Tensor.zeros [7]
  let (gradient, _) ← Autodiff.gradients (α := α)
    (paramShapes := [Shape.ofList [7]]) (inputShapes := [Shape.ofList [7]])
    (vectorLoss 1 .sum) (.cons prediction .nil) (.cons target .nil)
  for i in List.finRange 7 do
    expect "small residual derivative is preserved"
      ((gradient.get ⟨0, by decide⟩).getFlat i == small)

def run : IO Unit := do
  checkVector (α := Float)
  checkActions (α := Float)
  checkSmallResidual (α := Float)
  checkVector (α := Float32)
  checkActions (α := Float32)
  checkSmallResidual (α := Float32)
  IO.println "DQN semi-gradient Huber checks passed (Float, Float32)."

end Tests.Floats.DQN
