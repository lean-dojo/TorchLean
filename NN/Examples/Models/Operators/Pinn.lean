/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Autograd.Differential
public import NN.API.Seeded
public import NN.API.Neural.Training

/-!
# Training a physics-informed neural field

Solve `u''(x) = -2` on `[-1, 1]`, with `u(-1) = u(1) = 0`. The exact solution is `1 - x²`,
but training only sees the equation and boundary conditions, not samples of that solution.
A tanh MLP represents the field. The objective is half the mean squared equation residual plus
half the mean squared boundary residual. `derivativeVjp` differentiates the second coordinate
derivative with respect to all model parameters; ordinary SGD then updates the same typed state.

This is a small CPU example of the residual method of Raissi, Perdikaris, and Karniadakis,
"Physics-informed neural networks", Journal of Computational Physics 378 (2019), 686–707.
The reported residuals are sampled diagnostics, not a uniform PDE certificate.
-/

@[expose] public section

namespace NN.Examples.Models.Operators.Pinn

open TorchLean

/-- A smooth neural field; tanh permits the second derivatives in the equation residual. -/
def model : nn.Sequential [1] [1] :=
  nn.build 7 (nn.mlp 1 1 { hiddenWidths := [8], activation := .tanh })

/-- The equation is imposed at five interior collocation points. -/
def collocation : Array Float := #[-0.8, -0.4, 0, 0.4, 0.8]

/-- Evaluate the residual objective and accumulate its parameter gradient. -/
def lossGradient (state : nn.State Float (nn.stateShapes model)) :
    IO (Float × nn.State Float (nn.stateShapes model)) := do
  let mut loss := 0.0
  let mut gradient := nn.State.zeros
  let direction : Tensor Float [1] := [1]
  for x in collocation do
    let input : Tensor Float [1] := [x]
    let second ← autograd.model.derivative model state input [direction, direction]
    let residual := second.getScalar ⟨0, by decide⟩ + 2
    loss := loss + residual * residual / (2 * collocation.size.toFloat)
    let (contribution, _) ← autograd.model.derivativeVjp model state input
      [direction, direction] (Tensor.full [1] (residual / collocation.size.toFloat))
    gradient := gradient.zipWith contribution Tensor.add
  for x in #[-1.0, 1.0] do
    let input : Tensor Float [1] := [x]
    let output ← autograd.model.derivative model state input []
    let residual := output.getScalar ⟨0, by decide⟩
    loss := loss + residual * residual / 4
    let (contribution, _) ← autograd.model.vjp model state input
      (Tensor.full [1] (residual / 2))
    gradient := gradient.zipWith contribution Tensor.add
  pure (loss, gradient)

/-- Fit the equation and boundary conditions, then compare with the held-out exact solution. -/
def run (steps : Nat := 1500) : IO Unit := do
  let mut state : nn.State Float (nn.stateShapes model) := nn.initialState model
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
  for x in #[-1.0, -0.75, -0.25, 0.0, 0.25, 0.75, 1.0] do
    let output ← autograd.model.derivative model state ([x] : Tensor Float [1]) []
    IO.println s!"x={x}: prediction={output.getScalar ⟨0, by decide⟩}, exact={1 - x*x}"

/-- Run with the default step count or one explicit natural-number argument. -/
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

end NN.Examples.Models.Operators.Pinn
