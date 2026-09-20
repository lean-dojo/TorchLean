/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all

public import FloatLib.Floats
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
public import NN.Core.Numeric
public import NN.Floats.Arb
public import NN.Floats.FP32
public import FloatLib.Floats.Formats.BinaryInterchange
public import NN.Floats.Interval
public import NN.Floats.Quantization

/-!
# Numerical formats used by TorchLean

FloatLib supplies generic formats, arithmetic, rounding, and interval theory. This umbrella adds
TorchLean's rounded-real specialization, interval endpoint contracts, quantization imports, and
external Arb interface. Use FloatLib's configured formats and scalar operations directly.
-/

@[expose] public section
