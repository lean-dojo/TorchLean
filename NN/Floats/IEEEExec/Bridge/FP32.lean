/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.Compare
public import NN.Floats.IEEEExec.Bridge.FP32.Core
public import NN.Floats.IEEEExec.Bridge.FP32.DyadicRounding
public import NN.Floats.IEEEExec.Bridge.FP32.NearestEven
public import NN.Floats.IEEEExec.Bridge.FP32.Ops
public import NN.Floats.IEEEExec.Bridge.FP32.Sqrt
public import NN.Floats.IEEEExec.Bridge.FP32.RatBounds
public import NN.Floats.IEEEExec.Bridge.FP32.RoundDyadic
public import NN.Floats.IEEEExec.Bridge.FP32.RoundRat
public import NN.Floats.IEEEExec.Bridge.FP32.Ulp

/-!
Bridge lemmas between executable IEEE32 arithmetic and the FP32-facing API.

The submodules expose operation-level, rounding, and totality facts for verification code that
uses binary32 semantics rather than Lean's host `Float` behavior.
-/
