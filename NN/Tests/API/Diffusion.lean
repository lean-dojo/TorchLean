/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.Diffusion

/-!
# Diffusion API Tests

Regression checks for schedule boundaries and the shape-preserving diffusion helpers.
-/

@[expose] public section

namespace NN.Tests.API.Diffusion

open TorchLean

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"diffusion API check failed: {label}"

def close (actual expected : Float) (tolerance : Float := 1e-6) : Bool :=
  Float.abs (actual - expected) <= tolerance

def run : IO Unit := do
  let emptySchedule := diffusion.linearAlphaBars 0 0.1 0.2
  expect "zero-step schedule is empty"
    (Tensor.to emptySchedule (Array Float)).isEmpty
  expect "runnable schedule names its zero-step validation"
    (match diffusion.Schedule.linear 0 0.1 0.2 with
     | .error message =>
         message == "Diffusion.Schedule: must contain at least one step"
     | .ok _ => false)
  expect "linear schedule names its negative-start validation"
    (match diffusion.Schedule.linear 2 (-0.1) 0.2 with
     | .error message =>
         message == "Diffusion.Schedule: beta start must be in [0, 1)"
     | .ok _ => false)
  expect "linear schedule names its invalid-end validation"
    (match diffusion.Schedule.linear 2 0.1 1.0 with
     | .error message =>
         message == "Diffusion.Schedule: beta end must be in [0, 1)"
     | .ok _ => false)
  expect "linear schedule names its non-finite validation"
    (match diffusion.Schedule.linear 2
      (Float.ofBits 0x7ff0000000000000) 0.2 with
     | .error message =>
         message == "Diffusion.Schedule: beta endpoints must be finite"
     | .ok _ => false)
  expect "loaded schedules name invalid coefficients"
    (match diffusion.Schedule.from
      ([Float.ofBits 0x7ff8000000000000] : Tensor Float [1]) with
     | .error message =>
         message ==
           "Diffusion.Schedule: coefficients must be finite values in [0, 1]"
     | .ok _ => false)
  expect "loaded schedules reject coefficients outside the unit interval"
    (match diffusion.Schedule.from ([1.1] : Tensor Float [1]) with
     | .error _ => true
     | .ok _ => false)
  expect "loaded schedules name increasing cumulative coefficients"
    (match diffusion.Schedule.from ([0.8, 0.9] : Tensor Float [2]) with
     | .error message =>
         message ==
           "Diffusion.Schedule: cumulative coefficients must be nonincreasing"
     | .ok _ => false)
  expect "loaded schedules accept finite nonincreasing coefficients"
    (match diffusion.Schedule.from ([1.0, 0.9, 0.0] : Tensor Float [3]) with
     | .error _ => false
     | .ok _ => true)

  let oneStep : Fin 1 := ⟨0, by decide⟩
  expect "one-step schedule starts at betaStart"
    (close (diffusion.linearBeta 0.1 0.2 oneStep) 0.1)
  expect "one-step cumulative alpha uses betaStart"
    (close (diffusion.linearAlphaBar 0.1 0.2 oneStep) 0.9)

  let start : Fin 5 := ⟨0, by decide⟩
  let middle : Fin 5 := ⟨2, by decide⟩
  let finish : Fin 5 := ⟨4, by decide⟩
  expect "linear schedule preserves its start endpoint"
    (close (diffusion.linearBeta 0.1 0.2 start) 0.1)
  expect "linear schedule interpolates its midpoint"
    (close (diffusion.linearBeta 0.1 0.2 middle) 0.15)
  expect "linear schedule preserves its end endpoint"
    (close (diffusion.linearBeta 0.1 0.2 finish) 0.2)
  let longSchedule := diffusion.linearAlphaBars 257 0.0001 0.02
  expect "prefix schedule agrees with the pointwise definition"
    ((List.finRange 257).all fun step =>
      close (Tensor.getScalar longSchedule step)
        (diffusion.linearAlphaBar 0.0001 0.02 step))

  let spatial : Tensor Nat [1] := [2]
  let x0 : Tensor Float [2, 1, 2] := [[[1.0, 2.0]], [[3.0, 4.0]]]
  let eps : Tensor Float [2, 1, 2] := [[[5.0, 6.0]], [[7.0, 8.0]]]
  let alphaBars : Tensor Float [1] := [1.0]
  let schedule ←
    match diffusion.Schedule.from alphaBars with
    | .ok schedule => pure schedule
    | .error message => throw <| IO.userError message
  let sample :=
    diffusion.noisedSampleFromNoise [2] spatial schedule x0 eps 17
  expect "batched noising preserves samples and appends a zero time channel"
    (Tensor.to sample.input (Array Float) ==
      #[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0])
  expect "explicit diffusion noise is the supervised target"
    (Tensor.to sample.target (Array Float) == Tensor.to eps (Array Float))

  let degeneratePrevious :=
    diffusion.ddimPrev (shape := [2]) 1.0 0.0
      ([0.25, -0.25] : Tensor Float [2])
      ([0.5, -0.5] : Tensor Float [2])
  expect "DDIM update remains finite when alpha-bar is zero"
    ((Tensor.to degeneratePrevious (Array Float)).all Float.isFinite)

  IO.println "  diffusion API: passed"

end NN.Tests.API.Diffusion
