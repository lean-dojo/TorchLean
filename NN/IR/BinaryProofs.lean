/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Binary
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Proof
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof

/-!
# Binary IR conversion semantics

The interpreter's conversion returns FloatLib's specified directed rounding of
the exact source encoding. This statement includes exceptional encodings; it does
not assign a real value to NaN or infinity. FloatLib supplies the finite rounding
and error theorems applicable when their additional hypotheses hold.
-/

@[expose] public section

namespace NN.IR.Binary

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/-- A successful interpreter cast delivers exactly FloatLib's directed binary conversion. -/
theorem cast_value (source target : FloatFormat) (rounding : Model.IEEERoundingMode)
    (value : Scalar source) (converted : Scalar target) (flags : ExecFloat.ConversionStatus)
    (h : cast (.binary source) (.binary target) rounding value = .ok (converted, flags)) :
    ExecFloat.Binary.toModel converted =
      Model.cast source target (ExecFloat.Binary.toModel value) (mode := rounding) := by
  simp only [cast, Except.ok.injEq] at h
  have hValue := congrArg (fun pair => ExecFloat.Binary.toModel pair.1) h
  dsimp only at hValue
  simp only [ExecFloat.Binary.toModel_ofModel] at hValue
  exact hValue.symm.trans
    (Model.value_cast source target (ExecFloat.Binary.toModel value) rounding)

end NN.IR.Binary
