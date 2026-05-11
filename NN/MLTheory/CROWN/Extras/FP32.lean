/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32
public import NN.MLTheory.CROWN.Graph

/-!
# FP32

FP32-specialized entrypoints for the CROWN/LiRPA graph engine.

This is *not* an executable backend (it is `noncomputable` in general, because `FP32` is modeled on
`ℝ`). It exists so proofs can state “sound w.r.t. float32 semantics” without mentioning Lean’s
builtin `Float`.

This module is an optional convenience layer and lives under `NN/MLTheory/CROWN/Extras/`.
-/

@[expose] public section


namespace NN.MLTheory.CROWN

/-! ## FP32 entrypoints -/

/-- FP32 scalar type used for FP32-specialized CROWN/LiRPA statements. -/
abbrev FP32 := TorchLean.Floats.FP32

namespace FP32

open NN.MLTheory.CROWN.Graph

/-- Run IBP over `FP32` graph semantics. -/
noncomputable def runIBP (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32) :
    Array (Option (_root_.NN.MLTheory.CROWN.FlatBox FP32)) :=
  NN.MLTheory.CROWN.Graph.runIBP (α := FP32) g ps

/-- Run the one-dimensional derivative IBP pass over `FP32` graph semantics. -/
noncomputable def runDeriv1D (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ibp : Array (Option (_root_.NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (_root_.NN.MLTheory.CROWN.FlatBox FP32)) :=
  NN.MLTheory.CROWN.Graph.runDeriv1D (α := FP32) g ps ibp

/-- Run the second-derivative IBP pass over `FP32` graph semantics. -/
noncomputable def runDeriv2D (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ibp : Array (Option (_root_.NN.MLTheory.CROWN.FlatBox FP32)))
    (d1 : Array (Option (_root_.NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (_root_.NN.MLTheory.CROWN.FlatBox FP32)) :=
  NN.MLTheory.CROWN.Graph.runDeriv2D (α := FP32) g ps ibp d1

/-- Run the forward affine CROWN pass over `FP32` graph semantics. -/
noncomputable def runAffine (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ctx : NN.MLTheory.CROWN.Graph.AffineCtx)
    (ibp : Array (Option (_root_.NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (NN.MLTheory.CROWN.Graph.FlatAffine FP32)) :=
  NN.MLTheory.CROWN.Graph.runAffine (α := FP32) g ps ctx ibp

/-- Run the forward CROWN lower/upper affine-bounds pass over `FP32` graph semantics. -/
noncomputable def runCROWN (g : Graph) (ps : NN.MLTheory.CROWN.Graph.ParamStore FP32)
    (ctx : NN.MLTheory.CROWN.Graph.AffineCtx)
    (ibp : Array (Option (_root_.NN.MLTheory.CROWN.FlatBox FP32))) :
    Array (Option (NN.MLTheory.CROWN.Graph.FlatAffineBounds FP32)) :=
  NN.MLTheory.CROWN.Graph.runCROWN (α := FP32) g ps ctx ibp

end FP32

end NN.MLTheory.CROWN
