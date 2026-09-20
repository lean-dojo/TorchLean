/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals

/-!
# Configured binary scalars for TorchLean specifications

FloatLib owns the binary formats, representations, arithmetic, and elementary-function class.
This module supplies the `Context` dictionary used by TorchLean's tensors and models. Select a
format directly with `FloatLib.Floats.ExecFloat.Binary`, including formats wider than binary64.
The descriptor carries proofs that the exponent width is at least two, the fraction width and bias
are positive, and the bias is at most the encoding's largest finite exponent field. Concrete valid
widths discharge these requirements automatically; parameterized formats must provide the proofs.

Integer and rational casts round their exact value once in the destination format. The `pow`
field lifts FloatLib's model operation without converting through a native `Float`. FloatLib's
transcendentals are deterministic approximations; this dictionary adds no general accuracy or
correct-rounding theorem for them.

Configured values use CPU software arithmetic. They do not acquire a native CUDA tensor
representation, a `LawfulContext`, or a trainer/checkpoint encoding from this instance.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/--
Use any configured binary format as a tensor/model scalar.

The default safeguard is `1e-6`, rounded once. If a coarse format rounds it to zero, use its
smallest positive subnormal (encoding word `1`) instead. For example, exponent width `3` and
fraction width `2` use `1/16`; the minimal default-bias format with widths `2` and `1` uses `1/2`.
Such a large safeguard can materially change a guarded formula. This is a nonzero default, not
machine epsilon or an error bound; callers should choose tolerances for their format and problem.
-/
instance configuredBinaryContext : Context Value where
  natCast n := ExecFloat.Binary.ofModel (Model.roundRatQ format (n : Rat))
  ratCast value := ExecFloat.Binary.ofModel (Model.roundRatQ format value)
  pow x y := ExecFloat.Binary.ofModel
    (Model.pow (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y))
  defaultEpsilon :=
    let candidate : Value := ExecFloat.Binary.ofModel
      (Model.roundRatQ format (1 / 1000000 : Rat))
    if ExecFloat.Binary.isZero candidate then ExecFloat.Binary.ofNatBits 1 else candidate
  decidableGT := fun x y => inferInstanceAs (Decidable (x > y))

end TorchLean
