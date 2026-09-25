/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.RL.Runtime
public import NN.API.Seeded

/-!
# One-element critic outputs

Scalar extraction must preserve the stored bits at any singleton shape. The state-bearing case
also checks that actor state is skipped and that the typed critic's derivatives are unchanged.
-/

@[expose] public section

namespace NN.Tests.API.RLCritic

open TorchLean

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"RL critic: {label}"

def checkShape (shape : Shape) (oneValue : shape.size = 1) : IO Unit := do
  let model : nn.Sequential shape shape := .id shape
  let graph ← nn.lowerToTypedGraph model (α := Float)
  let value := rl.ppo.criticValue graph model model nn.State.empty (oneValue := oneValue)
  for bits in (#[0x0000000000000000, 0x8000000000000000, 0x3ff4000000000000,
      0xc004000000000000, 0x7ff0000000000000, 0xfff0000000000000,
      0x7ff8000000000042, 0xfff8000000000042] : Array UInt64) do
    -- Float.ofBits canonicalizes NaNs; extraction must preserve the constructed scalar.
    let scalar := Float.ofBits bits
    let input := Tensor.full shape scalar
    let actual := (value input).toBits
    expect s!"shape {shape}: expected scalar bits {scalar.toBits}, got {actual}"
      (actual == scalar.toBits)
  let (_, inputGradient) := nn.TypedGraphModel.vjp graph nn.State.empty
    (Tensor.full shape 2) (Tensor.full shape 3)
  expect s!"shape {shape}: input VJP" (Tensor.to inputGradient (Array Float) == #[3.0])

def checkStateSplit : IO Unit := do
  let model := nn.build 0 (nn.linear 1 1)
  let graph ← nn.lowerToTypedGraph model (α := Float)
  let actorState : nn.State Float (nn.stateShapes model) := nn.State.full 50
  let criticState : nn.State Float (nn.stateShapes model) := nn.State.full 2
  let value := rl.ppo.criticValue graph model model (actorState.append criticState)
  let input : Tensor Float [1] := [4]
  expect "critic reads its state after the actor state" (value input == 10)
  let (gradient, inputGradient) :=
    nn.TypedGraphModel.vjp graph criticState input ([1] : Tensor Float [1])
  expect "critic weight VJP"
    (Tensor.to (gradient.get ⟨0, by decide⟩) (Array Float) == #[4.0])
  expect "critic bias VJP"
    (Tensor.to (gradient.get ⟨1, by decide⟩) (Array Float) == #[1.0])
  expect "critic input VJP" (Tensor.to inputGradient (Array Float) == #[2.0])

/-- The scalar adapter cannot accept empty or multi-element outputs. -/
example : ¬ (Shape.size [0] = 1) ∧ ¬ (Shape.size [1, 2, 1] = 1) := by decide

def run : IO Unit := do
  checkShape [] (by decide)
  checkShape [1] (by decide)
  checkShape [1, 1] (by decide)
  checkShape [1, 1, 1, 1, 1] (by decide)
  checkStateSplit
  IO.println "  RL singleton critics: passed"

end NN.Tests.API.RLCritic
