/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Autograd.Complex
public import TorchLean.NN.Checkpoint
public import TorchLean.NN.Sequential.Training
public import TorchLean.NN.Builder

/-!
# Learning a complex affine map from a real loss

Fit `z ↦ (2-i)*z + (1/2+3i/4)` at four points on the unit circle. The model, parameters,
predictions and checkpoint are complex, and only the squared-residual objective is real.
`autograd.complex.grad` differentiates both coordinates of every parameter, and `nn.sgdStep`
updates them together. The gradient comes from the coordinate-forward reference algorithm, two
dual-number passes per complex parameter entry, which is cheap for a model this small.
-/

@[expose] public section

namespace TorchLean.Examples.ComplexRegression

/-- An ordinary linear builder, evaluated with complex scalar arithmetic. -/
def model : nn.Sequential [1] [1] := nn.build 0 (nn.linear 1 1)

/-- Half the mean squared complex residual, returned in the real component type. -/
def objective {α : Type} [Storage α] [Context α] [InverseTrig α]
    (state : TensorPack (Complex α) (nn.Sequential.stateShapes model)) : IO α := do
  let graph ← nn.Sequential.lowerToTypedGraph model (α := Complex α)
  let inputs : Array (Complex α) := #[⟨1, 0⟩, ⟨0, 1⟩, ⟨-1, 0⟩, ⟨0, -1⟩]
  let mut loss : α := 0
  for z in inputs do
    let prediction := nn.TypedGraphModel.forward graph state (Tensor.full [1] z)
    let target := (⟨2, -1⟩ : Complex α) * z + ⟨1 / 2, 3 / 4⟩
    let residual := prediction[0] - target
    loss := loss + residual.normSq / 8
  pure loss

/--
Train both complex components from zero with SGD at learning rate `0.2`, then print the fit and
the prediction at the held-out input `2 + 3i`.

Given a checkpoint path, the final state is saved there.
-/
def run (steps : Nat) (checkpoint : Option System.FilePath) : IO Unit := do
  let mut state : TensorPack (Complex Float32) (nn.Sequential.stateShapes model) := .zero
  IO.println s!"initial real objective: {← objective state}"
  for _ in [:steps] do
    let gradient ← autograd.complex.grad objective state
    state ← IO.ofExcept (nn.sgdStep model (Complex.ofReal 0.2) state gradient)
  IO.println s!"final real objective: {← objective state}"
  let weight :=
    ((state.get ⟨0, by decide⟩).reshape [1] (by decide)).getScalar ⟨0, by decide⟩
  IO.println s!"weight: {weight}"
  IO.println s!"bias: {(state.get ⟨1, by decide⟩).getScalar ⟨0, by decide⟩}"
  let graph ← nn.Sequential.lowerToTypedGraph model (α := Complex Float32)
  let probe : Complex Float32 := ⟨2, 3⟩
  let predicted := nn.TypedGraphModel.forward graph state (Tensor.full [1] probe)
  IO.println s!"held-out input {probe}: {predicted[0]}"
  IO.println "expected: (7.5 + 4.75i)"
  if let some path := checkpoint then
    Checkpoint.State.save state path
    IO.println s!"saved both components: {path}"

/-- Run with `--steps N` (default 50) and an optional `--save-checkpoint PATH`. -/
def main (args : List String) : IO Unit := do
  let exeName := "complex_regression"
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println "usage: torchlean complex_regression [--steps N] [--save-checkpoint PATH]"
    return
  let (steps, args) ← CLI.orThrow exeName <| CLI.takeNatFlag args "steps" (default := 50)
  let (checkpoint, args) ← CLI.orThrow exeName <| CLI.takePathFlag? args "save-checkpoint"
  CLI.requireNoArgs exeName args
  run steps checkpoint

end TorchLean.Examples.ComplexRegression
