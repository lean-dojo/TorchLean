/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RL.Floats.IEEE32Exec
public import NN.Runtime.RL.Numerics.Float32.Advantage

/-!
# Runtime Checked Preconditions → Float32 Semantics Theorems

`NN.Proofs.RL.Floats.IEEE32Exec` proves refinement theorems for RL formulas in the executable
`ExecFloat.Binary 8 23` float32 semantics, but those theorems are intentionally stated with explicit
`isFinite … = true` hypotheses for each intermediate.

In the runtime layer, TorchLean typically enforces these hypotheses by *checked preconditions*:
`Runtime.RL.Numerics.Float32.*Checked` returns `Except String …` and fails fast if any
intermediate becomes NaN/Inf.

This file is the glue: it turns “the runtime checker returned `.ok`” into the proof hypotheses
needed by the refinement theorem, yielding a user-facing statement:

`checked boundary ⇒ theorem applies`.

References:

- IEEE 754-2019 (binary32 arithmetic): https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic” (1991):
  https://doi.org/10.1145/103162.103163
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace Proofs
namespace RL
namespace Float32Exec

open Spec.RL

open TorchLean.Floats.IEEE754

open IEEE32Exec

/--
If `Runtime.RL.Numerics.Float32.discountedBackupChecked` returns `.ok`, then the decoded real
meaning of the result agrees with the standard “real-op + round-to-float32” model (`fp32Round`)
at each primitive operation.

This is the direct `checked boundary ⇒ semantics theorem applies` wrapper.
-/
theorem toReal_discountedBackupChecked_eq_fp32Round_chain
    (reward gamma bootstrap : ExecFloat.Binary 8 23) (done : Bool)
    (out : ExecFloat.Binary 8 23)
    (h : Runtime.RL.Numerics.Float32.discountedBackupChecked reward gamma bootstrap done
      = .ok out) :
    (ExecFloat.Binary.toModel out).toReal =
      fp32Round
        ((ExecFloat.Binary.toModel reward).toReal +
          fp32Round
            (fp32Round
                ((ExecFloat.Binary.toModel gamma).toReal *
                  (ExecFloat.Binary.toModel (continueMask (α := (ExecFloat.Binary 8 23))
                    done)).toReal) *
              (ExecFloat.Binary.toModel bootstrap).toReal)) := by
  obtain ⟨h₁, h₂, h₃, hout⟩ :=
    Runtime.RL.Numerics.Float32.discountedBackup_eq_ok
      (reward := reward) (gamma := gamma) (bootstrap := bootstrap) (done := done) (out := out) h
  -- Reduce to the spec-layer refinement theorem.
  rw [hout]
  exact
    (toReal_discountedBackup_eq_fp32Round_chain_of_isFinite
      (reward := reward) (gamma := gamma) (bootstrap := bootstrap) (done := done)
      (h₁ := h₁) (h₂ := h₂) (h₃ := h₃))

/--
If `Runtime.RL.Numerics.Float32.tdResidualChecked` returns `.ok`, then the decoded real meaning of
the result agrees with the standard “real-op + round-to-float32” model (`fp32Round`) at each
primitive operation.

This is the `checked boundary ⇒ semantics theorem applies` wrapper for TD residuals.
-/
theorem toReal_tdResidualChecked_eq_fp32Round_chain
    (value reward gamma nextValue : ExecFloat.Binary 8 23) (done : Bool)
    (out : ExecFloat.Binary 8 23)
    (h : Runtime.RL.Numerics.Float32.tdResidualChecked value reward gamma nextValue done
      = .ok out) :
    (ExecFloat.Binary.toModel out).toReal =
      fp32Round
        (fp32Round
            ((ExecFloat.Binary.toModel reward).toReal +
              fp32Round
                (fp32Round ((ExecFloat.Binary.toModel gamma).toReal * (ExecFloat.Binary.toModel
                  (continueMask (α := (ExecFloat.Binary 8 23)) done)).toReal) *
                  (ExecFloat.Binary.toModel nextValue).toReal)) -
          (ExecFloat.Binary.toModel value).toReal) := by
  obtain ⟨h₁, h₂, h₃, hval, hsub, hout⟩ :=
    Runtime.RL.Numerics.Float32.tdResidual_eq_ok
      (value := value) (reward := reward) (gamma := gamma) (nextValue := nextValue) (done := done)
      (out := out)
      h
  -- Reduce to the spec-layer refinement theorem.
  rw [hout]
  exact
    (toReal_tdResidual_eq_fp32Round_chain_of_isFinite
      (value := value) (reward := reward) (gamma := gamma) (nextValue := nextValue) (done := done)
      (h₁ := h₁) (h₂ := h₂) (h₃ := h₃) (hval := hval) (hsub := hsub))

end Float32Exec
end RL
end Proofs
