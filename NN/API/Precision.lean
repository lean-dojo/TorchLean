/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This facade exposes the upstream scalar and the existing typed model operations to consumers.
module -- shake: keep-downstream

public import NN.Spec.Core.FloatInstances -- shake: keep
public import NN.API.Seeded -- shake: keep
public import NN.API.Neural.Execution -- shake: keep
public import NN.API.Neural.Training -- shake: keep

/-!
# Selecting precision for tensors and models

Use FloatLib's scalar type directly, for example
`FloatLib.Floats.ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)`.
The fraction width excludes the leading significand bit of a normal number. Widths are static
type parameters; storage and execution costs grow with the selected width and tensor size.
The format requires `2 ≤ exponentBits`, `0 < fractionBits`, and
`0 < bias ≤ encoding.maxFiniteExponent exponentBits`; these are proof arguments in FloatLib's
type constructor. Its defaults use IEEE encoding and the encoding's standard bias.

This import supplies `Context` and the public typed tensor/model operations:

* Construct `Tensor α shape` and `nn.State α shapes` directly in the selected scalar type.
  Literals, rational casts, and `ExecFloat.Binary.parse` avoid a binary64 intermediate.
* Call `nn.lowerToTypedGraph model (α := α)` once, then `nn.TypedGraphModel.forward`,
  `nn.TypedGraphModel.jvp`, or `nn.TypedGraphModel.vjp` with typed state and inputs.
  Parameters, outputs, and derivatives retain `α`.
  Checked nested quotient replay is enabled for IEEE encodings; finite-only encodings retain
   their saturation semantics and do not receive that derivative-replay contract.
* Apply `nn.sgdStep model learningRate state stateGradient` to update trainable state with
  coefficients in the selected type. It preserves frozen entries and rejects models with
  buffer-update hooks; it does not silently omit running-statistics updates.
* Inspect finite results with `ExecFloat.Binary.toRat?`, or preserve the entire encoding with
  `ExecFloat.Binary.toNatBits`. Converting through `Float` can discard additional precision.

This path executes through CPU typed graphs. It does not select arbitrary-precision CUDA kernels.
The supervised trainer's input/report/checkpoint boundary and the default model initializers use
`Float`; use explicit typed state when additional input precision matters. FloatLib's elementary
approximations do not gain a general error theorem merely by selecting a wider format.

`Context.defaultEpsilon` rounds the exact rational `1/1000000` once. If it becomes zero, the
adapter uses the smallest positive subnormal. This can be large in a tiny format (for widths
`2` and `1`, it is `1/2`); it is a safeguard for guarded formulas, not machine epsilon or a
promise about numerical accuracy. Pass a tolerance suitable for the format and problem where the
operation allows it.

Normalization has a separate default, `TorchLean.normalizationEpsilon`: it casts the exact
rational `1/100000` once, retaining a nonzero binary16 value. It has no minimum-subnormal fallback;
if this tolerance rounds to zero in a tiny format, pass an explicit positive normalization epsilon.
For example, with three exponent bits and two fraction bits, use
`Spec.layerNorm ... (epsilon := Rat.cast (1 / 16 : Rat))`, or configure the model with
`nn.layerNorm ... (eps := (1 / 16 : Rat))` before lowering it. The default instead gives a zero
denominator for constant rows and can produce NaNs in typed-graph execution. Model validation checks
the rational epsilon's positivity, not positivity after scalar conversion. A valid format does not
guarantee finite or accurate results for every operation.
-/

@[expose] public section
