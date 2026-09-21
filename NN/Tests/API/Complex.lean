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
public import NN.Spec.Core.FloatInstances.Angle

/-!
# Real-coordinate complex gradients and exact checkpoints

Analytic derivatives distinguish real-loss differentiation from a complex-linear pullback.
An affine model checks the complete gradient-to-update path, and checkpoint round trips check
both coordinates without converting them to a narrower scalar format.
-/

@[expose] public section

namespace NN.Tests.API.Complex

open TorchLean

/-- Compare a measured real coordinate with its independent analytic value. -/
def checkClose (label : String) (actual expected : Float) : IO Unit := do
  unless actual.isFinite && Float.abs (actual - expected) < 1e-9 do
    throw <| IO.userError s!"{label}: expected {expected}, got {actual}"

/-- A single trainable complex scalar, with no dependence on the component representation. -/
def scalarState {α : Type} [Storage α] (z : Complex α) : nn.State (Complex α) [[1]] :=
  nn.State.empty.push (Tensor.full [1] z)

/-- Test the two coordinate derivatives of an explicitly real scalar objective. -/
def checkScalar (label : String) (objective : autograd.complex.Objective [[1]])
    (z expected : Complex Float) : IO Unit := do
  let gradient ← autograd.complex.grad objective (scalarState z)
  let actual := (gradient.get ⟨0, by decide⟩).getScalar ⟨0, by decide⟩
  checkClose s!"{label} real" actual.re expected.re
  checkClose s!"{label} imaginary" actual.im expected.im

/-- The affine model has a complex weight and bias, both trained by the same real objective. -/
def model : nn.Sequential [1] [1] := nn.build 0 (nn.linear 1 1)

/-- Half the squared residual for input `2+i` and target `-1+2i`. -/
def objective : autograd.complex.Objective (nn.stateShapes model) :=
  fun {α} _ _ _ state => do
    let graph ← nn.lowerToTypedGraph model (α := Complex α)
    let prediction := nn.TypedGraphModel.forward graph state
      (Tensor.full [1] (⟨2, 1⟩ : Complex α))
    let residual := prediction.getScalar ⟨0, by decide⟩ - (⟨-1, 2⟩ : Complex α)
    pure (residual.normSq / 2)

/-- Check scalar rules, a trainable affine graph, and shape-preserving checkpoint encodings. -/
def run : IO Unit := do
  checkScalar "conjugation" (fun state =>
    pure ((state.get ⟨0, by decide⟩).getScalar ⟨0, by decide⟩).conj.im)
    ⟨3, 4⟩ ⟨0, -1⟩
  checkScalar "magnitude" (fun state =>
    pure (MathFunctions.abs ((state.get ⟨0, by decide⟩).getScalar ⟨0, by decide⟩)).re)
    ⟨3, 4⟩ ⟨0.6, 0.8⟩
  checkScalar "sqrt imaginary tangent" (fun state =>
    pure (MathFunctions.sqrt ((state.get ⟨0, by decide⟩).getScalar ⟨0, by decide⟩)).im)
    ⟨4, 0⟩ ⟨0, 0.25⟩
  checkScalar "log magnitude" (fun state =>
    pure (MathFunctions.log ((state.get ⟨0, by decide⟩).getScalar ⟨0, by decide⟩)).re)
    ⟨1, 1⟩ ⟨0.5, 0.5⟩
  checkScalar "log argument" (fun state =>
    pure (MathFunctions.log ((state.get ⟨0, by decide⟩).getScalar ⟨0, by decide⟩)).im)
    ⟨1, 1⟩ ⟨-0.5, 0.5⟩
  let state : nn.State (Complex Float) (nn.stateShapes model) :=
    (nn.State.full ⟨3, -1⟩).set ⟨0, by decide⟩ (Tensor.full [1, 1] ⟨1, 2⟩)
  let result : nn.State (Complex Float) (nn.stateShapes model) × Float ←
    autograd.complex.grad objective state (value := true)
  let gradient := result.1
  checkClose "affine loss" result.2 10
  let dw := ((gradient.get ⟨0, by decide⟩).to (Array (Complex Float)))[0]!
  let db := (gradient.get ⟨1, by decide⟩).getScalar ⟨0, by decide⟩
  checkClose "weight real" dw.re 10
  checkClose "weight imaginary" dw.im 0
  checkClose "bias real" db.re 4
  checkClose "bias imaginary" db.im 2
  let updated ← IO.ofExcept (nn.sgdStep model (TorchLean.Complex.ofReal 0.1) state gradient)
  checkClose "updated loss" (← objective updated) 1.6
  let (_, emptyLoss) ← autograd.complex.grad (shapes := []) (fun _ => pure 7)
    (nn.State.empty (α := Complex Float)) (value := true)
  checkClose "empty state objective" emptyLoss 7
  let (primal, tangent) ← autograd.complex.jvp objective state
    (nn.State.full (⟨2, -3⟩ : Complex Float))
  checkClose "affine JVP primal" primal 10
  checkClose "affine JVP tangent" tangent 22
  let emptyGradient ← autograd.complex.grad (shapes := [[0]]) (fun _ => pure 5)
    (nn.State.zeros (α := Complex Float))
  unless ((emptyGradient.get ⟨0, by decide⟩).to (Array (Complex Float))).isEmpty do
    throw <| IO.userError "gradient changed empty tensor shape"
  IO.FS.withTempFile fun _ path => do
    Checkpoint.State.save model updated path
    let restored ← Checkpoint.State.load (α := Complex Float) model path
    checkClose "restored objective" (← objective restored) 1.6
    for index in Array.finRange (nn.stateShapes model).length do
      let original := (updated.get index).to (Array (Complex Float))
      let loaded := (restored.get index).to (Array (Complex Float))
      unless (original.map fun z => (z.re.toBits, z.im.toBits)) ==
          (loaded.map fun z => (z.re.toBits, z.im.toBits)) do
        throw <| IO.userError "complex checkpoint changed component bits"
    let rejected ← try
      let _ ← Checkpoint.State.load (α := Complex Float32) model path
      pure false
    catch _ => pure true
    unless rejected do throw <| IO.userError "checkpoint accepted a different component format"
  let malformed := Checkpoint.Encoding.decode (α := Complex Float) (Lean.Json.arr #[])
  if let .ok _ := malformed then
    throw <| IO.userError "accepted missing complex coordinates"
  let overflow := Checkpoint.Encoding.decode (α := Float) (Lean.toJson (2 ^ 64 : Nat))
  if let .ok _ := overflow then
    throw <| IO.userError "checkpoint silently truncated scalar bits"
  let missingValues := Runtime.Autograd.Model.StateIO.tensorFromJsonBits
    (α := Complex Float) "test" [0] (Lean.Json.mkObj [("shape", Lean.toJson ([0] : List Nat))])
  if let .ok _ := missingValues then
    throw <| IO.userError "empty checkpoint tensor accepted missing values"
  for bits in #[0x8000000000000000, 0x7ff8000000000042, 1] do
    let original := Float.ofBits bits
    let restored ← IO.ofExcept
      (Checkpoint.Encoding.decode (α := Float) (Checkpoint.Encoding.encode original))
    unless restored.toBits == original.toBits do
      throw <| IO.userError (s!"checkpoint scalar bits: input={bits}, " ++
        s!"before={original.toBits}, after={restored.toBits}")
  let noncanonical := Checkpoint.Encoding.decode (α := Float)
    (Lean.toJson (0x7ff8000000000042 : Nat))
  if let .ok _ := noncanonical then
    throw <| IO.userError "checkpoint silently canonicalized a native NaN payload"
  let wide : Complex (FloatLib.Floats.ExecFloat.Binary 15 112) :=
    ⟨FloatLib.Floats.ExecFloat.Binary.ofNatBits
        (FloatLib.Floats.ExecFloat.Binary.toNatBits
          (3 : FloatLib.Floats.ExecFloat.Binary 15 112) + 1),
      FloatLib.Floats.ExecFloat.Binary.ofNatBits (2 ^ 127)⟩
  let binary : Complex (FloatLib.Floats.ExecFloat.Binary 8 23) := ⟨3, 4⟩
  let binaryGrad ← autograd.complex.grad (fun state =>
    pure ((state.get ⟨0, by decide⟩).getScalar ⟨0, by decide⟩).normSq) (scalarState binary)
  let dz := (binaryGrad.get ⟨0, by decide⟩).getScalar ⟨0, by decide⟩
  unless FloatLib.Floats.ExecFloat.Binary.toRat? dz.re == some 6 &&
      FloatLib.Floats.ExecFloat.Binary.toRat? dz.im == some 8 do
    throw <| IO.userError "binary32 complex gradient"
  let encoded := Checkpoint.Encoding.encode wide
  let decoded ← IO.ofExcept (Checkpoint.Encoding.decode
    (α := Complex (FloatLib.Floats.ExecFloat.Binary 15 112)) encoded)
  unless FloatLib.Floats.ExecFloat.Binary.toNatBits decoded.re ==
      FloatLib.Floats.ExecFloat.Binary.toNatBits wide.re &&
      FloatLib.Floats.ExecFloat.Binary.toNatBits decoded.im ==
      FloatLib.Floats.ExecFloat.Binary.toNatBits wide.im do
    throw <| IO.userError "binary128 complex checkpoint"
  IO.println "  complex real-loss gradients, SGD, and exact checkpoints: passed"

end NN.Tests.API.Complex
