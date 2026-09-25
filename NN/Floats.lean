/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module -- shake: keep-all

public import FloatLib.Floats
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
public import NN.Core.Numeric
public import NN.Floats.Arb.Oracle
public import NN.Floats.FP32
public import FloatLib.Floats.Formats.BinaryInterchange
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.Formats.Flocq.Theory.Rounding.Affine

/-!
# Numerical formats used by TorchLean

FloatLib supplies generic formats, arithmetic, rounding, interval theory, and the affine
quantizer. This umbrella adds TorchLean's rounded-real `FP32` specialization and the external Arb
oracle. Use FloatLib's configured formats and scalar operations directly.
-/

@[expose] public section
