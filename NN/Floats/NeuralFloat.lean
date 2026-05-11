/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Conversion
public import NN.Floats.NeuralFloat.Core
public import NN.Floats.NeuralFloat.ErrorBounds
public import NN.Floats.NeuralFloat.Formats
public import NN.Floats.NeuralFloat.NF
public import NN.Floats.NeuralFloat.NNOps
public import NN.Floats.NeuralFloat.Rounding

/-!
# `NN.Floats.NeuralFloat`

This is the “rounding-on-ℝ” side of TorchLean’s float model stack. We use it when we want to talk about
formats and rounding generically, without committing to a concrete IEEE bit encoding:

- generic format/rounding infrastructure,
- the rounded scalar type `NF`,
- ULP-style error bounds and small calculation libraries.

Most files that are *not* bit-level IEEE execution live under `NN/Floats/NeuralFloat/`.
-/

@[expose] public section
