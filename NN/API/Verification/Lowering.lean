/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.API.Neural.Execution -- shake: keep
public import NN.MLTheory.CROWN.Graph -- shake: keep
public import NN.Verification.Builtin.Lowering -- shake: keep

/-!
# Explicit Verification Lowering

Low-level access to TorchLean's verifier graph. Most applications should import
`NN.API.Verification`, train with `trainer.train`, then call
`trained.verify center (radius := r) (norm := .inf)`; import this module only when constructing or
inspecting graph-level IBP/CROWN workflows directly.
-/

@[expose] public section

namespace TorchLean.Verification

export NN.Verification.Builtin (LoweredIR)
export NN.MLTheory.CROWN (FlatBox)
export NN.MLTheory.CROWN.Graph (ParamStore AffineCtx FlatAffine FlatAffineBounds)

/-- Lower a sequential model into checked verifier IR with one distinguished input. -/
def lowerForwardToIR {α : Type} [TorchLean.Storage α] [Context α]
    {σ τ : Shape}
    (model : TorchLean.nn.Sequential σ τ)
    (state : TorchLean.nn.State α (TorchLean.nn.stateShapes model)) :
    Except String (LoweredIR α) :=
  NN.Verification.Builtin.lowerForwardToIR
    (α := α) (paramShapes := TorchLean.nn.stateShapes model)
    (inShape := σ) (outShape := τ)
    (TorchLean.nn.forward model (α := α))
    (TorchLean.nn.State.Internal.toTensorPack state)

/-- Lower a custom forward program into checked verifier IR with one distinguished input. -/
def lowerProgramToIR {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {σ τ : Shape}
    (forwardProgram : Runtime.Autograd.Model.Program α (paramShapes ++ [σ]) τ)
    (state : TorchLean.nn.State α paramShapes) :
    Except String (LoweredIR α) :=
  NN.Verification.Builtin.lowerForwardToIR
    (α := α) (paramShapes := paramShapes)
    (inShape := σ) (outShape := τ)
    forwardProgram (TorchLean.nn.State.Internal.toTensorPack state)

/-- Dimensions of the distinguished verifier input node. -/
def inputShape? {α : Type} [TorchLean.Storage α] [Context α]
    (lowered : LoweredIR α) : Except String Shape :=
  lowered.inputShape?

/-- Compute upper affine bounds after validating the lowered verifier input.

The engine retains both sides internally so negative coefficients use the correct parent bound.
-/
def runAffine {α : Type} [TorchLean.Storage α] [Context α]
    [NN.MLTheory.CROWN.BoundOps α]
    (lowered : LoweredIR α) (parameters : ParamStore α)
    (intervalBounds : Array (Option (FlatBox α))) :
    Except String (Array (Option (FlatAffine α))) := do
  let affineContext ← lowered.affineCtx?
  pure <| NN.MLTheory.CROWN.Graph.runAffine
    (α := α) lowered.graph parameters affineContext intervalBounds

/-- Compute nodewise CROWN bounds after validating the lowered verifier input. -/
def runCROWN {α : Type} [TorchLean.Storage α] [Context α]
    [NN.MLTheory.CROWN.BoundOps α] [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α) (parameters : ParamStore α)
    (intervalBounds : Array (Option (FlatBox α))) :
    Except String (Array (Option (FlatAffineBounds α))) := do
  let affineContext ← lowered.affineCtx?
  pure <| NN.MLTheory.CROWN.Graph.runCROWN
    (α := α) lowered.graph parameters affineContext intervalBounds

end TorchLean.Verification
