/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Autograd.Complex
public import NN.API.Checkpoint
public import NN.API.Neural.Training
public import NN.API.Seeded

/-!
# Learning a complex affine map from a real loss

Fit `z ↦ (2-i)*z + (1/2+3i/4)` at four points on the unit circle. The model, parameters,
predictions, and checkpoint remain complex; only the squared-residual objective is real.
`autograd.complex.grad` differentiates both coordinates and `nn.sgdStep` updates them together.
This small example uses the coordinate-forward reference algorithm, not a complex reverse engine.
-/

@[expose] public section

namespace NN.Examples.Models.Operators.ComplexRegression

open TorchLean

/-- An ordinary linear builder, evaluated with complex scalar arithmetic. -/
def model : nn.Sequential [1] [1] := nn.build 0 (nn.linear 1 1)

/-- Half the mean squared complex residual, returned in the real component type. -/
def objective {α : Type} [Storage α] [Context α] [Atan2 α]
    (state : nn.State (Complex α) (nn.stateShapes model)) : IO α := do
  let graph ← nn.lowerToTypedGraph model (α := Complex α)
  let inputs : Array (Complex α) := #[⟨1, 0⟩, ⟨0, 1⟩, ⟨-1, 0⟩, ⟨0, -1⟩]
  let mut loss : α := 0
  for z in inputs do
    let prediction := nn.TypedGraphModel.forward graph state (Tensor.full [1] z)
    let target := (⟨2, -1⟩ : Complex α) * z + ⟨1 / 2, 3 / 4⟩
    let residual := prediction.getScalar ⟨0, by decide⟩ - target
    loss := loss + residual.normSq / 8
  pure loss

/-- Train both complex components, optionally persisting the exact final state. -/
def run (steps : Nat := 50) (checkpoint : Option System.FilePath := none) : IO Unit := do
  let mut state : nn.State (Complex Float32) (nn.stateShapes model) := nn.State.zeros
  IO.println s!"initial real objective: {← objective state}"
  for _ in [:steps] do
    let gradient ← autograd.complex.grad objective state
    state ← IO.ofExcept (nn.sgdStep model (TorchLean.Complex.ofReal 0.2) state gradient)
  IO.println s!"final real objective: {← objective state}"
  let weight :=
    ((state.get ⟨0, by decide⟩).reshape [1] (by decide)).getScalar ⟨0, by decide⟩
  IO.println s!"weight: {weight}"
  IO.println s!"bias: {(state.get ⟨1, by decide⟩).getScalar ⟨0, by decide⟩}"
  let graph ← nn.lowerToTypedGraph model (α := Complex Float32)
  let probe : Complex Float32 := ⟨2, 3⟩
  let predicted := nn.TypedGraphModel.forward graph state (Tensor.full [1] probe)
  IO.println s!"held-out input {probe}: {predicted.getScalar ⟨0, by decide⟩}"
  IO.println "expected: (7.5 + 4.75i)"
  if let some path := checkpoint then
    Checkpoint.State.save model state path
    let restored ← Checkpoint.State.load (α := Complex Float32) model path
    let repeated := nn.TypedGraphModel.forward graph restored (Tensor.full [1] probe)
    unless repeated.to (Array (Complex Float32)) == predicted.to (Array (Complex Float32)) do
      throw <| IO.userError "checkpoint changed the complex prediction"
    IO.println s!"saved and checked both components: {path}"

/-- Run with an optional step count and checkpoint path. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println "usage: torchlean complex_regression [steps [checkpoint]]"
    return
  match args with
  | [] => run
  | count :: rest =>
      let some steps := count.toNat?
        | throw <| IO.userError "usage: torchlean complex_regression [steps [checkpoint]]"
      match rest with
      | [] => run steps
      | [path] => run steps (some path)
      | _ => throw <| IO.userError "usage: torchlean complex_regression [steps [checkpoint]]"

end NN.Examples.Models.Operators.ComplexRegression
