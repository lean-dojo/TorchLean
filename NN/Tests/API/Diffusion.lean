/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.Diffusion.Sampling
public import NN.MLTheory.Generative.Diffusion.ImageDDIM

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

/--
The fixed image formula pins the default's operation order independently of the configurable API.
Keep the explicit denominator branch and scalar dictionary: NaNs and signed zeros distinguish them.
-/
def defaultDdimFormula {shape : Shape} (abPrev ab : Float)
    (sample epsilon : Tensor Float shape) : Tensor Float shape :=
  let zero := @Zero.zero Float Context.toZero
  let one := @One.one Float Context.toOne
  let _ : Max Float := Context.toMax
  let _ : LT Float := Context.toLT
  let _ : DecidableRel ((· > ·) : Float → Float → Prop) := Context.decidableGT
  let sqrtAb := MathFunctions.sqrt (Max.max ab zero)
  let sqrtAbPrev := MathFunctions.sqrt (Max.max abPrev zero)
  let sqrtOneMinusAb := MathFunctions.sqrt (Max.max (one - ab) zero)
  let sqrtOneMinusAbPrev := MathFunctions.sqrt (Max.max (one - abPrev) zero)
  let reconstruction := Tensor.scale
    (Tensor.sub sample (Tensor.scale epsilon sqrtOneMinusAb))
    (one / (if sqrtAb > 1e-12 then sqrtAb else 1e-12))
  let clipped := Tensor.clamp reconstruction (-one) one
  Tensor.add (Tensor.scale clipped sqrtAbPrev) (Tensor.scale epsilon sqrtOneMinusAbPrev)

def checkDdim : IO Unit := do
  let negativeZero := Float.ofBits 0x8000000000000000
  let nan := Float.ofBits 0x7ff8000000000042
  let infinity := Float.ofBits 0x7ff0000000000000
  let sample : Tensor Float [8] := [2, -2, 0.25, -0.25, 0, negativeZero, infinity, nan]
  let epsilon : Tensor Float [8] := [0.5, -0.5, negativeZero, 0, 0, negativeZero, 0, 1]
  for previousAlpha in [0.0, negativeZero, 0.25, 1.0, nan, infinity] do
    for alpha in [0.0, negativeZero, 0.25, 1.0, -1.0, 2.0, nan, infinity] do
      let actual := diffusion.ddimPrev previousAlpha alpha sample epsilon
      let expected := defaultDdimFormula previousAlpha alpha sample epsilon
      expect "DDIM defaults preserve every Float bit"
        ((Tensor.to actual (Array Float)).map Float.toBits ==
          (Tensor.to expected (Array Float)).map Float.toBits)

  let input : Tensor Float [3] := [2, -2, 0.5]
  let zero : Tensor Float [3] := [0, 0, 0]
  expect "DDIM default clips reconstructed values"
    (Tensor.to (diffusion.ddimPrev 1 0.25 input zero) (Array Float) == #[1, -1, 1])
  expect "DDIM identity postprocessor retains unclipped values"
    (Tensor.to (diffusion.ddimPrev 1 0.25 input zero (postprocess := id))
      (Array Float) == #[4, -4, 1])
  expect "DDIM chosen floor controls reconstruction"
    (Tensor.to (diffusion.ddimPrev 1 0.25 input zero
      (postprocess := id) (denominatorFloor := 2)) (Array Float) == #[1, -1, 0.25])
  expect "DDIM equality at the floor uses the same denominator"
    (Tensor.to (diffusion.ddimPrev 1 0.25 input zero
      (postprocess := id) (denominatorFloor := 0.5)) (Array Float) == #[4, -4, 1])
  expect "DDIM does not replace a NaN floor with the root"
    ((Tensor.to (diffusion.ddimPrev 1 0.25 input zero
      (postprocess := id) (denominatorFloor := nan)) (Array Float)).all Float.isNaN)

  -- Mixing coordinates rules out a theorem restricted to elementwise postprocessing.
  let postprocess := fun x : Tensor Float [2, 1] =>
    Tensor.full [2, 1] (Spec.get2 x 0 0 - 2 * Spec.get2 x 1 0)
  let batched : Tensor Float [2, 1] := [[2], [-1]]
  let noise : Tensor Float [2, 1] := [[0.25], [-0.5]]
  for floor in [0.0, negativeZero, 0.125, 0.5, 2.0, -1.0, infinity, nan] do
    let actual := diffusion.ddimPrev 0.25 0.0625 batched noise postprocess floor
    let expected := Generative.Diffusion.ImageDDIM.stepFromEps
      floor 0.25 0.0625 batched noise postprocess
    expect "DDIM custom nonlocal postprocessor agrees with the spec bitwise"
      ((Tensor.to actual (Array Float)).map Float.toBits ==
        (Tensor.to expected (Array Float)).map Float.toBits)

/-- The API-to-spec theorem applies to arbitrary tensor transforms and arbitrary Float floors. -/
example {shape : Shape} (previousAlpha alpha floor : Float)
    (sample epsilon : Tensor Float shape)
    (postprocess : Tensor Float shape → Tensor Float shape) :
    diffusion.ddimPrev previousAlpha alpha sample epsilon postprocess floor =
      Generative.Diffusion.ImageDDIM.stepFromEps
        floor previousAlpha alpha sample epsilon postprocess :=
  Generative.Diffusion.ImageDDIM.ddimPrev_eq_stepFromEps
    previousAlpha alpha sample epsilon postprocess floor

def checkDdimSampling : IO Unit := do
  let coefficients : Tensor Float [3] := [0.25, 0.0625, 0.015625]
  let initial : Tensor Float [1] := [0.25]
  let calls ← IO.mkRef (#[] : Array (Nat × Float))
  let predict := fun (index : Fin 3) (sample : Tensor Float [1]) => do
    calls.modify (·.push (index.val, sample[0]))
    pure ([0] : Tensor Float [1])
  let postprocess := fun x : Tensor Float [1] => Tensor.add x ([1] : Tensor Float [1])
  let sampled ← diffusion.reverseDdim predict coefficients initial postprocess 0.5
  expect "DDIM sampling threads floor and postprocessor through every step"
    (Tensor.to sampled (Array Float) == #[2.75])
  expect "DDIM predicts once per step on the updated sample"
    ((← calls.get) == #[(2, 0.25), (1, 0.375), (0, 0.875)])
  calls.set #[]
  let partialSample ← diffusion.reverseDdimFrom predict coefficients ⟨1, by decide⟩
    initial postprocess 0.5
  expect "DDIM partial reverse uses the chosen starting index"
    (Tensor.to partialSample (Array Float) == #[2.5])
  expect "DDIM partial reverse preserves predictor order"
    ((← calls.get) == #[(1, 0.25), (0, 0.75)])
  let specSample := Generative.Diffusion.ImageDDIM.sample 0.5 coefficients
    (fun _ _ => ([0] : Tensor Float [1])) initial postprocess
  expect "DDIM full sampling follows the spec traversal"
    ((Tensor.to sampled (Array Float)).map Float.toBits ==
      (Tensor.to specSample (Array Float)).map Float.toBits)
  let untouched ← diffusion.reverseDdim
    (fun (_ : Fin 0) (_ : Tensor Float [1]) =>
      throw <| IO.userError "empty DDIM schedule evaluated its predictor")
    (Tensor.full [0] 0.0) initial
    (postprocess := fun _ => ([999] : Tensor Float [1])) (denominatorFloor := 2)
  expect "empty DDIM schedule leaves the input unchanged"
    ((Tensor.to untouched (Array Float)).map Float.toBits ==
      (Tensor.to initial (Array Float)).map Float.toBits)

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

  checkDdim
  checkDdimSampling
  IO.println "  diffusion API: passed"

end NN.Tests.API.Diffusion
