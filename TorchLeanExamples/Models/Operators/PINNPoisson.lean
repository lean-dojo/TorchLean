/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Autograd.Differential
public import TorchLean.NN.Builder
public import TorchLean.NN.Sequential.Training
public import TorchLean.Tensor.Operations

/-!
# Training a physics-informed neural field

Solve `u''(x) = -2` on `[-1, 1]`, with `u(-1) = u(1) = 0`. The exact solution is `1 - x²`, and
training sees only the equation and the boundary conditions. A tanh MLP with one hidden layer of
width 8 represents the field. The objective is half the mean squared equation residual plus half
the mean squared boundary residual. `autograd.model.derivativeVJP` pulls the equation residual
back through the second input derivative to every model parameter, and plain SGD updates the
typed state.

The example follows the residual method of Raissi, Perdikaris, and Karniadakis, "Physics-informed
neural networks", Journal of Computational Physics 378 (2019), 686–707, at a small scale on the
CPU. The residuals it prints are measured at the sample points only.
-/

@[expose] public section

namespace TorchLean.Examples.PINNPoisson

open Activation.Kind
open Activation.Kind (tanh)

/--
A smooth neural field. Tanh activations give it the nonzero second derivatives that the equation
residual needs.
-/
def model : nn.Sequential [1] [1] :=
  nn.build 7 (nn.mlp 1 1 { hiddenWidths := [8], activation := tanh })

/-- The equation is imposed at five interior collocation points. -/
def collocation : Tensor Float [5, 1] := [[-0.8], [-0.4], [0], [0.4], [0.8]]

/-- The field is zero at the two endpoints. -/
def boundary : Tensor Float [2, 1] := [[-1], [1]]

/-- Evaluate the residual objective and accumulate its parameter gradient. -/
def lossGradient (state : TensorPack Float (nn.Sequential.stateShapes model)) :
    IO (Float × TensorPack Float (nn.Sequential.stateShapes model)) := do
  let mut loss := 0.0
  let mut gradient : TensorPack Float (nn.Sequential.stateShapes model) := .zero
  let direction : Tensor Float [1] := [1]
  for i in List.finRange 5 do
    let input := collocation[i]
    let second ← autograd.model.derivative model state input [direction, direction]
    let residual := second[0] + 2
    loss := loss + residual * residual / (2 * (5 : Float))
    let (contribution, _) ← autograd.model.derivativeVJP model state input
      [direction, direction] (Tensor.full [1] (residual / (5 : Float)))
    gradient := gradient.zipWith Tensor.add contribution
  for i in List.finRange 2 do
    let input := boundary[i]
    let output ← autograd.model.derivative model state input (Tensor.zeros [0, 1])
    let residual := output[0]
    loss := loss + residual * residual / 4
    let (contribution, _) ← autograd.model.vjp model state input
      (Tensor.full [1] (residual / 2))
    gradient := gradient.zipWith Tensor.add contribution
  pure (loss, gradient)

/--
Fit the equation and boundary conditions with SGD at learning rate `0.01`, then print the
prediction against the exact solution `1 - x²` at seven points.
-/
def run (steps : Nat := 1500) : IO Unit := do
  let mut state : TensorPack Float (nn.Sequential.stateShapes model) := nn.initialState model
  let (before, _) ← lossGradient state
  IO.println s!"initial residual objective: {before}"
  for step in [:steps] do
    let (loss, gradient) ← lossGradient state
    unless loss.isFinite do throw <| IO.userError "PINN objective became nonfinite"
    state ← IO.ofExcept (nn.sgdStep model 0.01 state gradient)
    if (step + 1) % 50 == 0 then
      IO.println s!"step {step + 1}: {loss}"
  let (after, _) ← lossGradient state
  IO.println s!"final residual objective: {after}"
  let points : Tensor Float [7, 1] := [[-1], [-0.75], [-0.25], [0], [0.25], [0.75], [1]]
  for i in List.finRange 7 do
    let input := points[i]
    let x := input[0]
    let output ← autograd.model.derivative model state
      input (Tensor.zeros [0, 1])
    IO.println s!"x={x}: prediction={output[0]}, exact={1 - x*x}"

/-- Run with the default 1500 steps, or with the step count given as the only argument. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println "usage: torchlean pinn [steps]"
    return
  match args with
  | [] => run
  | [count] =>
      match count.toNat? with
      | some steps => run steps
      | none => throw <| IO.userError "usage: torchlean pinn [steps]"
  | _ => throw <| IO.userError "usage: torchlean pinn [steps]"

end TorchLean.Examples.PINNPoisson
