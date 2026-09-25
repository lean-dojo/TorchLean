/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This facade exposes the upstream scalar and the existing typed model operations to consumers.
module -- shake: keep-downstream

public import NN.API.Neural.Training -- shake: keep
public import NN.API.Trainer.Session -- shake: keep

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
* Open `trainer.openTyped (α := α) (initialState? := some state)` to train on CPU with typed
  samples. Session inputs, predictions, state, losses, and `Result.report.loss` retain `α`.
  `Session.save`, `Session.load`, and `Result.save` reuse `Checkpoint.Encoding α` and preserve
  exact bits. A finished result keeps its own state snapshot while the session continues training.
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

Typed sessions use the maintained CPU runtime, in eager or graph mode. They reject CUDA, other
devices, and custom backend profiles explicitly. The ordinary `trainer.open` and `trainer.train`
paths retain their runtime-selected binary32 execution and `Float` boundary. Typed results do not
provide a verifier; calling `Result.verify` fails explicitly.

`nn.initialState model (α := α)` converts the model's stored seeded `Float` values. It does not
generate extra random precision; supply `initialState?` to preserve exact wider parameters.
The same typed override is available in `Module.instantiate`, `nn.Module.instantiate`, and
`nn.IndexedModule.instantiate`. Trainer optimizer and scheduler coefficients remain configured in
`Float` and are validated after conversion into `α`; use `nn.sgdStep` when the coefficient itself
must be specified in `α`. Dataset constructors backed by Float observations
also retain that source precision; typed samples supplied to a session avoid this boundary.
Structured typed reports preserve their losses; the existing `Report.toTrainLog` renderer accepts
`Float` reports. FloatLib's elementary approximations do not gain a general error theorem merely by
selecting a wider format.

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
