/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeERealTotal
public import NN.Floats.IEEEExec.BridgeFP32
public import NN.Floats.IEEEExec.BridgeFP32Expr
public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Floats.IEEEExec.BridgeInitFloat32
public import NN.Floats.IEEEExec.DirectedRoundingSoundness
public import NN.Floats.IEEEExec.ErrorBounds
public import NN.Floats.IEEEExec.ERealSemantics
public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.IEEEExec.Notation
public import NN.Floats.IEEEExec.OpSandwich
public import NN.Floats.IEEEExec.Reductions
public import NN.Floats.IEEEExec.RoundQuotEvenBounds
public import NN.Floats.IEEEExec.SpecialRules
public import NN.Floats.IEEEExec.TranscendentalRules
public import NN.Floats.IEEEExec.TrigRules
public import NN.Floats.IEEEExec.TrigBounds

/-!
# `NN.Floats.IEEEExec`

This is TorchLean’s execution-aware float32 layer. We use it when we want runs inside Lean to have a
precise, platform-independent meaning (including NaN/Inf and signed-zero corner cases):

- `IEEE32Exec`: an executable, bit-level IEEE-754 binary32 kernel,
- companion lemmas about special values,
- bridge theorems connecting `IEEE32Exec` back to the proof-oriented `FP32` model.

Suggested entry points:
- `NN.Floats.IEEEExec.Exec32` for the executable kernel and the core instances,
- `NN.Floats.IEEEExec.SpecialRules` for NaN/Inf propagation rules,
- `NN.Floats.IEEEExec.BridgeFP32` and `...BridgeFP32Total` for refinement into the real-valued
  `FP32` model,
- `NN.Floats.IEEEExec.ERealSemantics` and `...MinMaxERealSoundness` for interval-style reasoning,
- `NN.Floats.IEEEExec.OpSandwich` for nearest-even-vs-directed-rounding operation sandwiches,
- `NN.Floats.IEEEExec.Notation` for the scoped syntax used in docs and proofs.
-/

@[expose] public section
