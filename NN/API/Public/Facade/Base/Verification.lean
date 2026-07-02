/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.NN
public import NN.API.Public.TensorPack
public import NN.API.Public.Seeded
public import NN.API.Public.Autograd
public import NN.API.Data
public import NN.API.Data.Transforms
public import NN.API.Runtime
public import NN.API.Models
public import NN.API.Public.NN.Transformer
public import NN.API.RL
public import NN.API.Rand
public import NN.API.Samples.Bands
public import NN.API.Text.Bpe
public import NN.MLTheory.CROWN.Flatbox
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.TorchLean.Compile
public import NN.API.Public.Facade.Base.Root

/-!
# TorchLean Verification Helpers

Convenience names for compiling TorchLean models into IBP/CROWN checks.
-/

@[expose] public section

namespace TorchLean

namespace Verification

/-- TorchLean-to-verifier compiled IR bundle. -/
abbrev CompiledIR := NN.Verification.TorchLean.CompiledIR

/-- Verifier-side parameter store used by IBP/CROWN workflows. -/
abbrev ParamStore := NN.MLTheory.CROWN.Graph.ParamStore

/-- Flat input box used to seed verifier input bounds. -/
abbrev FlatBox := NN.MLTheory.CROWN.FlatBox

/-- Input-node metadata required by affine/CROWN verifier passes. -/
abbrev AffineCtx := NN.MLTheory.CROWN.Graph.AffineCtx

/-- Output of the forward affine pass for one verifier node. -/
abbrev FlatAffine := NN.MLTheory.CROWN.Graph.FlatAffine

/-- Output of the CROWN forward pass for one verifier node. -/
abbrev FlatAffineBounds := NN.MLTheory.CROWN.Graph.FlatAffineBounds

/--
Compile a sequential TorchLean model into verifier IR with one distinguished input.

This is the usual “train a model, then run IBP/CROWN on its forward pass” path.
-/
def compileForward1 {α : Type} [NN.API.Semantics.Scalar α] [DecidableEq Shape]
    {σ τ : Shape}
    (model : SequentialModel σ τ)
    (params : ParamTensors α (modelParamShapes model)) :
    Except String (CompiledIR α) :=
  NN.Verification.TorchLean.compileForward1
    (α := α) (paramShapes := modelParamShapes model)
    (inShape := σ) (outShape := τ)
    (NN.API.nn.program (model := model) (α := α)) params

/--
Compile a custom TorchLean forward program into verifier IR with one distinguished input.

Use this when the target is not a plain `nn.Sequential`, for example a hand-written loss program or
an attention fragment built directly from `TorchLean.Ops`.
-/
def compileProgram1 {α : Type} [NN.API.Semantics.Scalar α] [DecidableEq Shape]
    {paramShapes : List Shape} {σ τ : Shape}
    (program : _root_.Runtime.Autograd.TorchLean.Program α (paramShapes ++ [σ]) τ)
    (params : ParamTensors α paramShapes) :
    Except String (CompiledIR α) :=
  NN.Verification.TorchLean.compileForward1
    (α := α) (paramShapes := paramShapes)
    (inShape := σ) (outShape := τ)
    program params

/--
Seed the verifier input with an explicit input box.

Call this after `compileForward1`, then hand the returned store to IBP/CROWN passes.
-/
def seedInputBox {α : Type} [NN.API.Semantics.Scalar α]
    (compiled : CompiledIR α) (xB : FlatBox α) : ParamStore α :=
  compiled.seedInputBox xB

/--
Flatten a center tensor and radius tensor into the `FlatBox` expected by IBP/CROWN.

Use this for a shaped TorchLean input with a shaped perturbation radius.
-/
def lInfBox {α : Type} [NN.API.Semantics.Scalar α] {s : Shape}
    (center radius : _root_.Spec.Tensor α s) : FlatBox α :=
  NN.Verification.TorchLean.lInfBox (α := α) center radius

/--
Build a uniform `ℓ∞` box around a shaped TorchLean input tensor.

This fills the input shape with the scalar radius `eps`, then flattens it into a verifier box.
-/
def lInfBall {α : Type} [NN.API.Semantics.Scalar α] {s : Shape}
    (center : _root_.Spec.Tensor α s) (eps : α) : FlatBox α :=
  NN.Verification.TorchLean.lInfBall (α := α) center eps

/--
Seed the compiled verifier input with a uniform `ℓ∞` box around a shaped TorchLean input tensor.
-/
def seedLInfBall {α : Type} [NN.API.Semantics.Scalar α] {s : Shape}
    (compiled : CompiledIR α) (center : _root_.Spec.Tensor α s) (eps : α) : ParamStore α :=
  compiled.seedLInfBall center eps

/-- Shape of the distinguished verifier input node. -/
def inputShape? {α : Type} [Context α] (compiled : CompiledIR α) : Except String Shape :=
  compiled.inputShape?

/-- Flattened dimension of the distinguished verifier input node. -/
def inputDim? {α : Type} [Context α] (compiled : CompiledIR α) : Except String Nat :=
  compiled.inputDim?

/-- Checked affine context for the distinguished verifier input. -/
def affineCtx? {α : Type} [Context α] (compiled : CompiledIR α) :
    Except String AffineCtx :=
  compiled.affineCtx?

/-- Compatibility affine context for the distinguished verifier input. -/
def affineCtx {α : Type} [Context α] (compiled : CompiledIR α) : AffineCtx :=
  compiled.affineCtx

/-- Run IBP on a compiled verifier graph. -/
def runIBP {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) : Array (Option (FlatBox α)) :=
  compiled.runIBP ps

/-- Read the verifier output box from an IBP result array. -/
def outputBox? {α : Type} [Context α] (compiled : CompiledIR α)
    (boxes : Array (Option (FlatBox α))) : Except String (FlatBox α) := do
  compiled.outputBox? boxes

/-- Read the verifier output box, throwing an `IO.userError` if it is missing. -/
def outputBoxOrThrow {α : Type} [Context α] (compiled : CompiledIR α)
    (boxes : Array (Option (FlatBox α))) : IO (FlatBox α) :=
  compiled.outputBoxOrThrow boxes

/-- Run the forward affine pass on a compiled verifier graph. -/
def runAffine {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Array (Option (FlatAffine α)) :=
  NN.MLTheory.CROWN.Graph.runAffine (α := α) compiled.graph ps (affineCtx compiled) ibp

/-- Checked variant of `runAffine` that validates the compiled input node first. -/
def runAffine? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Except String (Array (Option (FlatAffine α))) := do
  let ctx ← affineCtx? compiled
  pure <| NN.MLTheory.CROWN.Graph.runAffine (α := α) compiled.graph ps ctx ibp

/-- Read the verifier output affine form from a forward affine result array. -/
def outputAffine? {α : Type} [Context α] (compiled : CompiledIR α)
    (affs : Array (Option (FlatAffine α))) : Except String (FlatAffine α) := do
  compiled.outputAffine? affs

/-- Run CROWN on a compiled verifier graph. -/
def runCROWN {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Array (Option (FlatAffineBounds α)) :=
  NN.MLTheory.CROWN.Graph.runCROWN (α := α) compiled.graph ps (affineCtx compiled) ibp

/-- Checked variant of `runCROWN` that validates the compiled input node first. -/
def runCROWN? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Except String (Array (Option (FlatAffineBounds α))) := do
  let ctx ← affineCtx? compiled
  pure <| NN.MLTheory.CROWN.Graph.runCROWN (α := α) compiled.graph ps ctx ibp

/-- Read the verifier output CROWN bounds from a CROWN result array. -/
def outputCROWN? {α : Type} [Context α] (compiled : CompiledIR α)
    (bounds : Array (Option (FlatAffineBounds α))) : Except String (FlatAffineBounds α) := do
  compiled.outputCROWN? bounds

/-- Run forward CROWN and evaluate the verifier output bounds on the compiled input box. -/
def outputBoxCROWN? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) (xB : FlatBox α) :
    Except String (FlatBox α) :=
  compiled.outputBoxCROWN? ps xB

/-- Run forward CROWN and return the evaluated verifier output box, throwing on failure. -/
def outputBoxCROWNOrThrow {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) (xB : FlatBox α) : IO (FlatBox α) :=
  compiled.outputBoxCROWNOrThrow ps xB

/-- Run backward CROWN for a scalar objective and evaluate it on the compiled input box. -/
def backwardObjectiveBox? {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (xB : FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatVec α) :
    Except String (FlatBox α) :=
  compiled.backwardObjectiveBox? ps ibp xB obj

/-- `IO` version of `backwardObjectiveBox?`. -/
def backwardObjectiveBoxOrThrow {α : Type} [Context α] [NN.MLTheory.CROWN.BoundOps α]
    (compiled : CompiledIR α) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (xB : FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatVec α) : IO (FlatBox α) :=
  compiled.backwardObjectiveBoxOrThrow ps ibp xB obj

/--
Compute the conservative two-class margin lower bound `lo[class0] - hi[class1]`.

If this is positive, class `class0` is certified against `class1` over the input box.
-/
def twoClassMarginLowerBound {α : Type} [Context α] {n : Nat}
    (lo hi : _root_.Spec.Tensor α (.dim n .scalar)) (class0 class1 : Fin n) : α :=
  Spec.Tensor.vecGet lo class0 - Spec.Tensor.vecGet hi class1

/-- Decide whether the two-class margin lower bound is strictly positive. -/
def certifiesTwoClassMargin {α : Type} [Context α] {n : Nat}
    (lo hi : _root_.Spec.Tensor α (.dim n .scalar)) (class0 class1 : Fin n) : Bool :=
  Context.gtBool
    (twoClassMarginLowerBound (α := α) lo hi class0 class1) (0 : α)

end Verification


end TorchLean
