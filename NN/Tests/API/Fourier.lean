/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.FNO
public import NN.API.Autograd.Model

/-!
# Multidimensional Fourier and FNO regressions

Direct phase sums provide an independent reference for odd/even spatial grids. Round trips check
both complex coordinates, and the adjoint check uses the normalized inverse of an arbitrary seed.
FNO checks compare identical parameters through full-grid DFT and separable FFT paths, including
parameter and input gradients. CUDA runs also inspect the tape to require actual FFT nodes.
-/

@[expose] public section

namespace NN.Tests.API.Fourier

open TorchLean
open Runtime.Autograd

/-- Compare finite tensors without relying on their printed decimal precision. -/
def check {shape : Shape} (label : String) (actual expected : Tensor Float shape)
    (tolerance : Float) : IO Unit := do
  let a := actual.to (Array Float)
  let b := expected.to (Array Float)
  for index in List.finRange a.size do
    let expected := b[index.val]!
    unless a[index].isFinite && Float.abs (a[index] - expected) ≤ tolerance do
      throw <| IO.userError s!"{label}[{index.val}]: {a[index]} vs {expected}"

/-- Evaluate a model and its full seeded pullback on one eager interpreter. -/
def evaluate {σ τ : Shape} (device : NN.Backend.Device) (model : nn.Sequential σ τ)
    (state : nn.State Float (nn.stateShapes model)) (input : Tensor Float σ)
    (seed : Tensor Float τ) :
    IO (Tensor Float τ × nn.State Float (nn.stateShapes model) × Tensor Float σ × Nat) := do
  let session ← Torch.Internal.EagerSession.new (α := Float) { device, execution := .eager }
  try
    let rec register : {shapes : List Shape} → TensorPack Float shapes →
        IO (Torch.RefList (Torch.TensorRef Float) shapes)
      | [], .nil => pure .nil
      | _ :: _, .cons value rest => do
          pure (.cons (← session.input value (requiresGrad := true)) (← register rest))
    let refs ← register (nn.State.Internal.toTensorPack state)
    let inputRef ← session.input input (requiresGrad := true)
    let outputRef ← (Model.Layers.Seq.forwardState model (α := Float)
      (m := Torch.Internal.EagerM Float)
      .eval refs inputRef) session
    let output ← session.getValue outputRef
    let nativeCount := ((← session.cudaTape.get).nodes.filter
      fun node => node.name == some "rfft1d").size
    let gradients ← session.backwardDenseAll outputRef seed
    let rec collect : {shapes : List Shape} → Torch.RefList (Torch.TensorRef Float) shapes →
        IO (TensorPack Float shapes)
      | [], .nil => pure .nil
      | _ :: _, .cons ref rest => do
          pure (.cons (← Torch.Internal.EagerSession.grad gradients ref) (← collect rest))
    pure (output, nn.State.Internal.fromTensorPack (← collect refs),
      ← Torch.Internal.EagerSession.grad gradients inputRef, nativeCount)
  finally session.resetTape

/-- Pack two spatial component planes around the public complex FFT operation. -/
def transform (spatial : List Nat) (positive : 0 < spatial.prod) (channels : Nat)
    (inverse roundtrip : Bool := false) :
    nn.Sequential [2, spatial.prod, channels] [2, spatial.prod, channels] :=
  nn.Sequential.fromLayer
    { kind := "Fourier"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      forward := fun _ {α} _ _ {m} _ _ => fun input =>
       (show m (Model.RefTy m α [2, spatial.prod, channels]) from do
        let realPart : Model.RefTy m α [spatial.prod, channels] ←
          Model.select 0 input ⟨0, by change 0 < 2; decide⟩
        let imagPart : Model.RefTy m α [spatial.prod, channels] ←
          Model.select 0 input ⟨1, by change 1 < 2; decide⟩
        let (realPart, imagPart) ← nn.functional.fft spatial positive realPart imagPart inverse
        let (realPart, imagPart) ← if roundtrip then
            nn.functional.fft spatial positive realPart imagPart (!inverse)
          else pure (realPart, imagPart)
        let realPlane ← Model.reshape (s₂ := [1, spatial.prod, channels]) realPart
          (by simp [Shape.size])
        let imagPlane ← Model.reshape (s₂ := [1, spatial.prod, channels]) imagPart
          (by simp [Shape.size])
        Model.concatLeadingAxis realPlane imagPlane) }

/-- Direct complex phase sum; no FFT, reflection, or axis-permutation implementation is reused. -/
def reference (spatial : List Nat) (channels : Nat)
    (input : Tensor Float [2, spatial.prod, channels]) (inverse : Bool) :
    Tensor Float [2, spatial.prod, channels] :=
  let values := input.to (Array Float)
  Tensor.generateFlat [2, spatial.prod, channels] fun index => Id.run do
    let part := index / (spatial.prod * channels)
    let frequency := index / channels % spatial.prod
    let channel := index % channels
    let mut result := 0.0
    for time in [:spatial.prod] do
      let phase := Model.Layers.FNO.Internal.phase (α := Float) spatial time frequency
      let angle := (if inverse then 2 else -2) * (MathFunctions.pi : Float) * phase
      let realPart := values[time * channels + channel]!
      let imagPart := values[(spatial.prod + time) * channels + channel]!
      result := result + if part == 0 then
        realPart * Float.cos angle - imagPart * Float.sin angle
      else realPart * Float.sin angle + imagPart * Float.cos angle
    pure (if inverse then result / spatial.prod.toFloat else result)

/-- Check direct sums, inverse normalization, and the real-coordinate adjoint on one grid. -/
def checkTransform (device : NN.Backend.Device) (spatial : List Nat)
    (positive : 0 < spatial.prod) (channels : Nat := 2) : IO Unit := do
  let input : Tensor Float [2, spatial.prod, channels] :=
    Tensor.generateFlat _ fun i => (i % 11).toFloat / 7 - 0.6
  let seed : Tensor Float [2, spatial.prod, channels] :=
    Tensor.generateFlat _ fun i => (i % 5).toFloat / 3 - 0.4
  let tolerance := if device == .cuda then 3e-4 else 1e-10
  for inverse in [false, true] do
    let model := transform spatial positive channels inverse
    let (actual, _, gradient, nativeCount) ← evaluate device model nn.State.empty input seed
    check s!"FFT {spatial}" actual (reference spatial channels input inverse) tolerance
    let adjoint := (reference spatial channels seed (!inverse)).map fun x =>
      if inverse then x / spatial.prod.toFloat else x * spatial.prod.toFloat
    check s!"FFT adjoint {spatial}" gradient adjoint tolerance
    if device == .cuda && !spatial.isEmpty && nativeCount != 2 * spatial.length then
      throw <| IO.userError s!"expected native per-axis FFTs, got {nativeCount}"
  let roundtrip := transform spatial positive channels (roundtrip := true)
  let (actual, _, gradient, _) ← evaluate device roundtrip nn.State.empty input seed
  check s!"FFT roundtrip {spatial}" actual input tolerance
  check s!"FFT roundtrip adjoint {spatial}" gradient seed tolerance

/-- Compare an FNO's complete state and input pullbacks through both spectral implementations. -/
def checkFno {rank : Nat} (device : NN.Backend.Device) (spatial modes : Tensor Nat [rank]) :
    IO Unit := do
  let config : nn.models.FNO.Config rank :=
    { spatial, modes, width := 2, layerCount := 2, activation := .tanh }
  let fast := nn.build 17 (nn.models.fno config [2, 1])
  let dense := nn.build 17 (nn.models.fno { config with spectralPath := .denseReference } [2, 1])
  if same : nn.stateShapes fast = nn.stateShapes dense then
    let state := nn.initialState fast
    let input : Tensor Float (config.inputShape [2, 1]) :=
      Tensor.generateFlat _ fun i => (i % 9).toFloat / 7 - 0.5
    let seed : Tensor Float (config.outputShape [2, 1]) :=
      Tensor.generateFlat _ fun i => (i % 5).toFloat / 3 - 0.2
    let (actual, gradient, inputGrad, nativeCount) ← evaluate device fast state input seed
    let (expected, referenceGrad, referenceInputGrad, _) ←
      evaluate .cpu dense (state.cast same) input seed
    let tolerance := if device == .cuda then 1e-4 else 1e-10
    check "FNO forward" actual expected tolerance
    check "FNO input gradient" inputGrad referenceInputGrad tolerance
    let referenceGrad := referenceGrad.cast same.symm
    for index in Array.finRange (nn.stateShapes fast).length do
      check s!"FNO parameter {index.val}" (gradient.get index) (referenceGrad.get index) tolerance
    if device == .cuda && nativeCount == 0 then
      throw <| IO.userError "FNO automatic path did not record native FFTs"
  else throw <| IO.userError "FNO spectral path changed the checkpoint layout"

/-- Run the same contracts on CPU or, in the CUDA suite, through actual cuFFT nodes. -/
def run (device : NN.Backend.Device := .cpu) : IO Unit := do
  for spatial in [[], [1], [5], [2, 3], [2, 3, 2], [1, 2, 1, 3]] do
    if positive : 0 < spatial.prod then checkTransform device spatial positive
    else throw <| IO.userError "invalid Fourier test grid"
  checkTransform device [2, 3] (by decide) 0
  checkFno device ([2, 3] : Tensor Nat [2]) [1, 2]
  checkFno device ([2, 2, 3] : Tensor Nat [3]) [1, 1, 2]
  checkFno device ([2, 3] : Tensor Nat [2]) [0, 0]
  IO.println s!"  multidimensional FFT/FNO values and gradients ({reprStr device}): passed"

end NN.Tests.API.Fourier
