/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Numerics.Float32
public import NN.Proofs.RL.Floats.IEEE32Exec

/-!
# Runtime Checked Preconditions → Float32 Semantics Theorems

`NN.Proofs.RL.Floats.IEEE32Exec` proves refinement theorems for RL formulas in the executable
`IEEE32Exec` float32 semantics, but those theorems are intentionally stated with explicit
`isFinite … = true` hypotheses for each intermediate.

In the runtime layer, TorchLean typically enforces these hypotheses by *checked preconditions*:
`Runtime.RL.Numerics.Float32.*Checked` returns `Except String …` and fails fast if any intermediate becomes
NaN/Inf.

This file is the glue: it turns “the runtime checker returned `.ok`” into the proof hypotheses
needed by the refinement theorem, yielding a user-facing statement:

`checked boundary ⇒ theorem applies`.

References:

- IEEE 754-2019 (binary32 arithmetic): https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg, “What Every Computer Scientist Should Know About Floating-Point Arithmetic” (1991):
  https://doi.org/10.1145/103162.103163
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Float32Exec

open Spec.RL

open TorchLean.Floats.IEEE754

open IEEE32Exec

/--
If `Runtime.RL.Numerics.Float32.discountedBackupIEEE32ExecChecked` returns `.ok`, then the decoded real
meaning of the result agrees with the standard “real-op + round-to-float32” model (`fp32Round`)
at each primitive operation.

This is the `checked boundary ⇒ semantics theorem applies` wrapper most users want.
-/
theorem toReal_discountedBackupIEEE32ExecChecked_eq_fp32Round_chain
    (reward gamma bootstrap : TorchLean.Floats.IEEE754.IEEE32Exec) (done : Bool)
    (out : TorchLean.Floats.IEEE754.IEEE32Exec)
    (h : Runtime.RL.Numerics.Float32.discountedBackupIEEE32ExecChecked reward gamma bootstrap done = .ok out) :
    toReal out =
      fp32Round
        (toReal reward +
          fp32Round
            (fp32Round
                (toReal gamma *
                  toReal (continueMask (α := TorchLean.Floats.IEEE754.IEEE32Exec) done)) *
              toReal bootstrap)) := by
  obtain ⟨h₁, h₂, h₃, hout⟩ :=
    Runtime.RL.Numerics.Float32.discountedBackupIEEE32ExecChecked_eq_ok
      (reward := reward) (gamma := gamma) (bootstrap := bootstrap) (done := done) (out := out) h
  -- Reduce to the spec-layer refinement theorem.
  rw [hout]
  exact
    (toReal_discountedBackup_eq_fp32Round_chain_of_isFinite
      (reward := reward) (gamma := gamma) (bootstrap := bootstrap) (done := done)
      (h₁ := h₁) (h₂ := h₂) (h₃ := h₃))

/--
If `Runtime.RL.Numerics.Float32.tdResidualIEEE32ExecChecked` returns `.ok`, then the decoded real meaning of
the result agrees with the standard “real-op + round-to-float32” model (`fp32Round`) at each
primitive operation.

This is the `checked boundary ⇒ semantics theorem applies` wrapper for TD residuals.
-/
theorem toReal_tdResidualIEEE32ExecChecked_eq_fp32Round_chain
    (value reward gamma nextValue : TorchLean.Floats.IEEE754.IEEE32Exec) (done : Bool)
    (out : TorchLean.Floats.IEEE754.IEEE32Exec)
    (h : Runtime.RL.Numerics.Float32.tdResidualIEEE32ExecChecked value reward gamma nextValue done = .ok out) :
    toReal out =
      fp32Round
        (fp32Round
            (toReal reward +
              fp32Round
                (fp32Round (toReal gamma * toReal (continueMask (α := IEEE32Exec) done)) *
                  toReal nextValue)) -
          toReal value) := by
  obtain ⟨h₁, h₂, h₃, hval, hsub, hout⟩ :=
    Runtime.RL.Numerics.Float32.tdResidualIEEE32ExecChecked_eq_ok
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
