/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- Check that the application umbrella exposes the complete typed update path.
public import NN.API
public import NN.Examples.Quickstart.TypedTraining

/-!
# Typed SGD consumer regressions

Expected parameter updates are independent exact rationals. Wide formats retain both an initial
coefficient and SGD increments beyond binary64 precision. Native controls exercise nonzero updates.
The public squared-error example is checked separately from the constant-gradient update cases.
-/

@[expose] public section

namespace NN.Tests.API.TypedTraining

open TorchLean
open FloatLib.Floats

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"typed training: {label}"

def unwrap {α : Type} (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message => throw <| IO.userError s!"typed training: {message}"

abbrev Binary128 := ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)
abbrev Binary256 := ExecFloat.Binary (exponentBits := 19) (fractionBits := 236)

def affine : nn.Sequential [1] [1] :=
  NN.Examples.Quickstart.TypedTraining.model

/-- Freeze the bias of the single affine layer used below. -/
def freezeBias {σ τ : Shape} : nn.Sequential σ τ → nn.Sequential σ τ
  | .id shape => .id shape
  | .cons layer rest =>
      .cons { layer with
        requiresGrad := layer.requiresGrad.mapIdx fun index flag =>
          flag && index != 1 } (freezeBias rest)

/-- A persistent read-only buffer with no mode-dependent update hook. -/
def bufferLayer : nn.Layer [1] [1] :=
  { kind := "ReadOnlyBuffer"
    stateShapes := [[1]]
    initState := .cons (Tensor.zeros [1]) .nil
    requiresGrad := #[false]
    forward := fun _ {α} _ _ {m} _ _ =>
      fun (_buffer input : Runtime.Autograd.Model.RefTy (m := m) (α := α) [1]) =>
        (pure input : m (Runtime.Autograd.Model.RefTy (m := m) (α := α) [1])) }

def maskedAffine : nn.Sequential [1] [1] :=
  nn.compose (freezeBias affine) bufferLayer

def normalized : nn.Sequential [1, 1] [1, 1] :=
  nn.build 0 (nn.batchNorm ([1] : Tensor Nat [1]) (channels := 1))

/--
For the scalar objective `w*2+b`, the VJP is `(2,1)`. Each step subtracts `(2*lr,lr)`.
Both the original coefficient and the learning rate are supplied as exact rational casts.
-/
def checkSmallUpdates {α : Type} [Storage α] [Context α]
    (label : String) (decode : α → Option Rat) (gapExponent rateExponent : Nat) :
    IO Unit := do
  let source : Rat := 1 + 1 / (2 ^ gapExponent : Nat)
  let rate : Rat := 1 / (2 ^ rateExponent : Nat)
  let initial : nn.State α (nn.stateShapes affine) := nn.State.full (Rat.cast source)
  let input : Tensor α [1] := Tensor.full [1] 2
  let graph ← nn.lowerToTypedGraph affine (α := α) (mode := .train)
  let learningRate : α := Rat.cast rate
  expect (label ++ "/exact learning rate") (decode learningRate == some rate)
  let mut state := initial
  for step in [1:4] do
    let (gradient, _) := nn.TypedGraphModel.vjp graph state input (Tensor.full [1] 1)
    expect (label ++ "/weight VJP")
      (((gradient.get ⟨0, by decide⟩).to (Array α)).map decode == #[some 2])
    expect (label ++ "/bias VJP")
      (((gradient.get ⟨1, by decide⟩).to (Array α)).map decode == #[some 1])
    state ← unwrap (nn.sgdStep affine learningRate state gradient)
    let count : Rat := step
    expect s!"{label}/weight after step {step}"
      (((state.get ⟨0, by decide⟩).to (Array α)).map decode ==
        #[some (source - count * 2 * rate)])
    expect s!"{label}/bias after step {step}"
      (((state.get ⟨1, by decide⟩).to (Array α)).map decode ==
        #[some (source - count * rate)])
  expect (label ++ "/original state unchanged")
    (((initial.get ⟨0, by decide⟩).to (Array α)).map decode == #[some source])

/--
For input 2, target 0 and rate 1/8, two half-squared-error steps from `(a,a)` produce
`(-a/32,31*a/64)`. The prediction is `27*a/64`, down from `3*a`.
-/
def checkSquaredErrorExample {α : Type} [Storage α] [Context α]
    (label : String) (decode : α → Option Rat) (gapExponent : Nat) : IO Unit := do
  let source : Rat := 1 + 1 / (2 ^ gapExponent : Nat)
  let initial : nn.State α (nn.stateShapes affine) := nn.State.full (Rat.cast source)
  let input : Tensor α [1] := Tensor.full [1] 2
  let trained ← NN.Examples.Quickstart.TypedTraining.fit initial input
    (Tensor.zeros [1]) (Rat.cast (1 / 8 : Rat)) 2
  expect (label ++ "/squared-error weight")
    (((trained.get ⟨0, by decide⟩).to (Array α)).map decode == #[some (-source / 32)])
  expect (label ++ "/squared-error bias")
    (((trained.get ⟨1, by decide⟩).to (Array α)).map decode == #[some (31 * source / 64)])
  let graph ← nn.lowerToTypedGraph affine (α := α)
  let prediction := nn.TypedGraphModel.forward graph trained input
  expect (label ++ "/squared-error prediction")
    (decode (prediction.getScalar ⟨0, by decide⟩) == some (27 * source / 64))

def checkMasks {α : Type} [Storage α] [Context α]
    (label : String) (decode : α → Option Rat) (gapExponent : Nat) : IO Unit := do
  let source : Rat := 1 + 1 / (2 ^ gapExponent : Nat)
  let state : nn.State α (nn.stateShapes maskedAffine) := nn.State.full (Rat.cast source)
  expect (label ++ "/original model flags") (nn.requiresGrad maskedAffine == #[true, false, false])
  -- Deliberately nonzero gradients in frozen slots must not update either frozen entry.
  let gradient : nn.State α (nn.stateShapes maskedAffine) := nn.State.full 1
  let result ← unwrap (nn.sgdStep maskedAffine (Rat.cast (1 / 8 : Rat)) state gradient)
  expect (label ++ "/trainable weight updated")
    (((result.get ⟨0, by decide⟩).to (Array α)).map decode == #[some (source - 1 / 8)])
  expect (label ++ "/frozen bias preserved")
    (((result.get ⟨1, by decide⟩).to (Array α)).map decode == #[some source])
  expect (label ++ "/persistent buffer preserved")
    (((result.get ⟨2, by decide⟩).to (Array α)).map decode == #[some source])
  expect (label ++ "/masked input unchanged")
    (((state.get ⟨0, by decide⟩).to (Array α)).map decode == #[some source])

def checkRejectedModels : IO Unit := do
  let state : nn.State Binary128 (nn.stateShapes normalized) := nn.State.full 1
  match nn.sgdStep normalized (Rat.cast (1 / 8 : Rat)) state nn.State.zeros with
  | .ok _ => throw <| IO.userError "typed training: accepted a BatchNorm buffer-update hook"
  | .error message =>
      expect "visible buffer-update restriction"
        (message == "nn.sgdStep: models with buffer-update hooks are unsupported")
  let malformed := nn.Sequential.fromLayer { bufferLayer with requiresGrad := #[] }
  let badState : nn.State Binary128 (nn.stateShapes malformed) := nn.State.full 1
  expect "malformed trainable mask rejected"
    (!(nn.sgdStep malformed 1 badState nn.State.zeros).isOk)
  expect "empty state accepted"
    ((nn.sgdStep (.id [1]) (1 : Binary128) nn.State.empty nn.State.empty).isOk)

def checkBackend {α : Type} [Storage α] [Context α]
    (label : String) (decode : α → Option Rat) (gapExponent rateExponent : Nat) :
    IO Unit := do
  checkSmallUpdates (α := α) label decode gapExponent rateExponent
  checkSquaredErrorExample (α := α) label decode gapExponent
  checkMasks (α := α) label decode gapExponent

/-- Maintained public training checks for configured binary128/binary256 and both native types. -/
def run : IO Unit := do
  checkBackend (α := Binary128) "binary128" ExecFloat.Binary.toRat? 100 110
  checkBackend (α := Binary256) "binary256" ExecFloat.Binary.toRat? 200 230
  checkBackend (α := Float) "Float"
    (fun value => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofFloat value)) 40 48
  checkBackend (α := Float32) "Float32"
    (fun value => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofFloat32 value)) 10 18
  checkRejectedModels

end NN.Tests.API.TypedTraining
