/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32
public import NN.Floats.IEEEExec.BridgeFP32
public import NN.Proofs.Verification.ODE.Enclosure

/-!
# Backend Views for ODE Enclosures

`Enclosure.lean` proves the ODE corridor theorems over exact real-valued functions. The executable
verification pipeline, however, often evaluates candidate PINN corridors in concrete numeric
backends:

- `TorchLean.Floats.FP32`, our proof-level binary32-style rounded real model;
- `TorchLean.Floats.IEEE754.IEEE32Exec`, the executable IEEE-754 binary32 bridge.

This file does **not** prove that those backends are numerically sound by itself. Instead, it gives
the clean final adapters: if the backend-valued functions satisfy the real hypotheses after applying
their `toReal` interpretation, then the real ODE enclosure theorem applies.

That is the same trust split used by Tanaka and Yatabe's learn-and-verify framework
(arXiv:2601.19818): interval/CROWN/IBP machinery verifies inequalities for a concrete producer,
while the ODE comparison theorem consumes the resulting real inequalities.
-/

@[expose] public section


namespace NN.Proofs.Verification.ODE.Enclosure

open scoped Topology
open Set

namespace Backend

open TorchLean.Floats

noncomputable section

/-- Interpret a backend-valued trajectory `g : ℝ → α` as a real-valued function via `toReal`. -/
abbrev realView {α : Type} (toReal : α → ℝ) (g : ℝ → α) : ℝ → ℝ :=
  fun t => toReal (g t)

namespace LocalCorridor

/-! ## Local corridor wrapper -/

/--
Generic backend wrapper for the local corridor theorem.

The hypotheses are deliberately phrased over `realView toReal ...`: a caller must already have
translated backend computations into real continuity, derivative, and inequality facts. Given those
facts, the result is the same enclosure + unclamping statement for the backend trajectory's real
interpretation.
-/
theorem fromRealView
    {α : Type} (toReal : α → ℝ)
    {T : ℝ} (hT : 0 ≤ T) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → α} {a : ℝ}
    (hu_cont : ContinuousOn (realView toReal u) (Icc 0 T))
    (hu_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal u)
          (f t (clampToCorridor
            (realView toReal uL) (realView toReal uU) t ((realView toReal u) t)))
            (Ici t) t)
    (hu0 : (realView toReal u) 0 = a)
    (hL_cont : ContinuousOn (realView toReal uL) (Icc 0 T))
    (hL_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal uL) ((realView toReal uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (realView toReal uL') t ≤ f t ((realView toReal uL) t))
    (hL0 : (realView toReal uL) 0 ≤ a)
    (hU_cont : ContinuousOn (realView toReal uU) (Icc 0 T))
    (hU_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal uU) ((realView toReal uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((realView toReal uU) t) ≤ (realView toReal uU') t)
    (hU0 : a ≤ (realView toReal uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (realView toReal uL) t ≤ (realView toReal uU) t) :
    (∀ t ∈ Icc 0 T,
        (realView toReal uL) t ≤ (realView toReal u) t ∧
          (realView toReal u) t ≤ (realView toReal uU) t) ∧
      (∀ t ∈ Ico 0 T, HasDerivWithinAt (realView toReal u) (f t ((realView toReal u) t)) (Ici t)
        t) := by
  simpa [realView] using
    (localSolutionEnclosed_fromClampedDynamics (T := T) hT (f := f)
      (u := realView toReal u)
      (uL := realView toReal uL) (uU := realView toReal uU)
      (uL' := realView toReal uL') (uU' := realView toReal uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU)

/-- Specialization of `fromRealView` to TorchLean's proof-level `FP32` model. -/
theorem forFP32
    {T : ℝ} (hT : 0 ≤ T) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → FP32} {a : ℝ}
    (hu_cont : ContinuousOn (realView FP32.toReal u) (Icc 0 T))
    (hu_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal u)
          (f t (clampToCorridor (realView FP32.toReal uL) (realView FP32.toReal uU) t ((realView
            FP32.toReal u) t)))
          (Ici t) t)
    (hu0 : (realView FP32.toReal u) 0 = a)
    (hL_cont : ContinuousOn (realView FP32.toReal uL) (Icc 0 T))
    (hL_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal uL) ((realView FP32.toReal uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (realView FP32.toReal uL') t ≤ f t ((realView FP32.toReal uL) t))
    (hL0 : (realView FP32.toReal uL) 0 ≤ a)
    (hU_cont : ContinuousOn (realView FP32.toReal uU) (Icc 0 T))
    (hU_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal uU) ((realView FP32.toReal uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((realView FP32.toReal uU) t) ≤ (realView FP32.toReal uU') t)
    (hU0 : a ≤ (realView FP32.toReal uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (realView FP32.toReal uL) t ≤ (realView FP32.toReal uU) t) :
    (∀ t ∈ Icc 0 T,
        (realView FP32.toReal uL) t ≤ (realView FP32.toReal u) t ∧
          (realView FP32.toReal u) t ≤ (realView FP32.toReal uU) t) ∧
      (∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal u) (f t ((realView FP32.toReal u) t)) (Ici t) t)
          := by
  simpa using
    (fromRealView (α := FP32) FP32.toReal (T := T) hT (f := f)
      (u := u) (uL := uL) (uU := uU) (uL' := uL') (uU' := uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU)

end LocalCorridor

namespace ConstantExtension

/-! ## Constant-extension wrapper -/

/--
Generic backend wrapper for the constant-extension theorem.

The corridor is interpreted as real-valued first and then frozen after `T` with
`constantExtensionAfter`. This mirrors the paper's global step while keeping all backend arithmetic
outside this proof.
-/
theorem fromRealView
    {α : Type} (toReal : α → ℝ)
    {T τ : ℝ} (hT : 0 ≤ T) (hτ : T ≤ τ) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → α} {a : ℝ}
    (hu_cont : ContinuousOn (realView toReal u) (Icc 0 τ))
    (hu_der : ∀ t ∈ Ico 0 τ,
      HasDerivWithinAt (realView toReal u)
        (f t (clampToCorridor
          (constantExtensionAfter T (realView toReal uL))
          (constantExtensionAfter T (realView toReal uU)) t
          ((realView toReal u) t))) (Ici t) t)
    (hu0 : (realView toReal u) 0 = a)
    (hL_cont : ContinuousOn (realView toReal uL) (Icc 0 T))
    (hL_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal uL) ((realView toReal uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (realView toReal uL') t ≤ f t ((realView toReal uL) t))
    (hL0 : (realView toReal uL) 0 ≤ a)
    (hU_cont : ContinuousOn (realView toReal uU) (Icc 0 T))
    (hU_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView toReal uU) ((realView toReal uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((realView toReal uU) t) ≤ (realView toReal uU') t)
    (hU0 : a ≤ (realView toReal uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (realView toReal uL) t ≤ (realView toReal uU) t)
    (hLower :
      ∀ t, T < t →
        0 ≤ f T ((realView toReal uL) T) ∧ f T ((realView toReal uL) T) ≤ f t ((realView
          toReal uL) T))
    (hUpper :
      ∀ t, T < t →
        f t ((realView toReal uU) T) ≤ f T ((realView toReal uU) T) ∧ f T ((realView toReal
          uU) T) ≤ 0) :
    (∀ t ∈ Icc 0 τ,
        constantExtensionAfter T (realView toReal uL) t ≤ (realView toReal u) t ∧
          (realView toReal u) t ≤ constantExtensionAfter T (realView toReal uU) t) ∧
      (∀ t ∈ Ico 0 τ,
        HasDerivWithinAt (realView toReal u) (f t ((realView toReal u) t)) (Ici t) t) := by
  simpa [realView] using
    (extendedSolutionEnclosed_fromClampedDynamics (T := T) (τ := τ) hT hτ (f := f)
      (u := realView toReal u)
      (uL := realView toReal uL) (uU := realView toReal uU)
      (uL' := realView toReal uL') (uU' := realView toReal uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU hLower hUpper)

/-- Specialization of `fromRealView` to TorchLean's proof-level `FP32` model. -/
theorem forFP32
    {T τ : ℝ} (hT : 0 ≤ T) (hτ : T ≤ τ) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → FP32} {a : ℝ}
    (hu_cont : ContinuousOn (realView FP32.toReal u) (Icc 0 τ))
    (hu_der : ∀ t ∈ Ico 0 τ,
      HasDerivWithinAt (realView FP32.toReal u)
        (f t
          (clampToCorridor
            (constantExtensionAfter T (realView FP32.toReal uL))
            (constantExtensionAfter T (realView FP32.toReal uU)) t
            ((realView FP32.toReal u) t)))
        (Ici t) t)
    (hu0 : (realView FP32.toReal u) 0 = a)
    (hL_cont : ContinuousOn (realView FP32.toReal uL) (Icc 0 T))
    (hL_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal uL) ((realView FP32.toReal uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (realView FP32.toReal uL') t ≤ f t ((realView FP32.toReal uL) t))
    (hL0 : (realView FP32.toReal uL) 0 ≤ a)
    (hU_cont : ContinuousOn (realView FP32.toReal uU) (Icc 0 T))
    (hU_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (realView FP32.toReal uU) ((realView FP32.toReal uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((realView FP32.toReal uU) t) ≤ (realView FP32.toReal uU') t)
    (hU0 : a ≤ (realView FP32.toReal uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (realView FP32.toReal uL) t ≤ (realView FP32.toReal uU) t)
    (hLower :
      ∀ t, T < t →
        0 ≤ f T ((realView FP32.toReal uL) T) ∧
          f T ((realView FP32.toReal uL) T) ≤ f t ((realView FP32.toReal uL) T))
    (hUpper :
      ∀ t, T < t →
        f t ((realView FP32.toReal uU) T) ≤ f T ((realView FP32.toReal uU) T) ∧
          f T ((realView FP32.toReal uU) T) ≤ 0) :
    (∀ t ∈ Icc 0 τ,
        constantExtensionAfter T (realView FP32.toReal uL) t ≤ (realView FP32.toReal u) t ∧
          (realView FP32.toReal u) t ≤ constantExtensionAfter T (realView FP32.toReal uU) t) ∧
      (∀ t ∈ Ico 0 τ,
        HasDerivWithinAt (realView FP32.toReal u) (f t ((realView FP32.toReal u) t)) (Ici t) t)
          := by
  simpa using
    (fromRealView (α := FP32) FP32.toReal (T := T) (τ := τ) hT hτ (f := f)
      (u := u) (uL := uL) (uU := uU) (uL' := uL') (uU' := uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU hLower hUpper)

end ConstantExtension

/-! ## IEEE32Exec wrappers -/

open TorchLean.Floats.IEEE754

/--
Real interpretation of an executable IEEE-754 binary32 trajectory.

This is intentionally just `toReal`. Rounding/error guarantees have to be proved before these
wrappers are invoked; this abbreviation only keeps theorem statements readable.
-/
abbrev ieee32RealView (g : ℝ → IEEE32Exec) : ℝ → ℝ := fun t => IEEE32Exec.toReal (g t)

namespace LocalCorridor

/-- Local corridor theorem specialized to the executable IEEE-754 binary32 backend. -/
theorem forIEEE32Exec
    {T : ℝ} (hT : 0 ≤ T) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → IEEE32Exec} {a : ℝ}
    (hu_cont : ContinuousOn (ieee32RealView u) (Icc 0 T))
    (hu_der :
      ∀ t ∈ Ico 0 T,
        HasDerivWithinAt (ieee32RealView u)
          (f t (clampToCorridor (ieee32RealView uL) (ieee32RealView uU) t
            ((ieee32RealView u) t))) (Ici t) t)
    (hu0 : (ieee32RealView u) 0 = a)
    (hL_cont : ContinuousOn (ieee32RealView uL) (Icc 0 T))
    (hL_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt (ieee32RealView uL) ((ieee32RealView uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (ieee32RealView uL') t ≤ f t ((ieee32RealView uL) t))
    (hL0 : (ieee32RealView uL) 0 ≤ a)
    (hU_cont : ContinuousOn (ieee32RealView uU) (Icc 0 T))
    (hU_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt (ieee32RealView uU) ((ieee32RealView uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((ieee32RealView uU) t) ≤ (ieee32RealView uU') t)
    (hU0 : a ≤ (ieee32RealView uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (ieee32RealView uL) t ≤ (ieee32RealView uU) t) :
    (∀ t ∈ Icc 0 T,
        (ieee32RealView uL) t ≤ (ieee32RealView u) t ∧
          (ieee32RealView u) t ≤ (ieee32RealView uU) t) ∧
      (∀ t ∈ Ico 0 T, HasDerivWithinAt (ieee32RealView u) (f t ((ieee32RealView u) t)) (Ici t)
        t) := by
  simpa [ieee32RealView] using
    (localSolutionEnclosed_fromClampedDynamics (T := T) hT (f := f)
      (u := ieee32RealView u) (uL := ieee32RealView uL) (uU := ieee32RealView uU)
      (uL' := ieee32RealView uL') (uU' := ieee32RealView uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU)

end LocalCorridor

namespace ConstantExtension

/-- Constant-extension theorem specialized to the executable IEEE-754 binary32 backend. -/
theorem forIEEE32Exec
    {T τ : ℝ} (hT : 0 ≤ T) (hτ : T ≤ τ) {f : ℝ → ℝ → ℝ}
    {u uL uU uL' uU' : ℝ → IEEE32Exec} {a : ℝ}
    (hu_cont : ContinuousOn (ieee32RealView u) (Icc 0 τ))
    (hu_der : ∀ t ∈ Ico 0 τ,
      HasDerivWithinAt (ieee32RealView u)
        (f t (clampToCorridor
          (constantExtensionAfter T (ieee32RealView uL))
          (constantExtensionAfter T (ieee32RealView uU)) t ((ieee32RealView u) t))) (Ici t) t)
    (hu0 : (ieee32RealView u) 0 = a)
    (hL_cont : ContinuousOn (ieee32RealView uL) (Icc 0 T))
    (hL_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt (ieee32RealView uL) ((ieee32RealView uL') t) (Ici t) t)
    (hL_sub : ∀ t ∈ Ico 0 T, (ieee32RealView uL') t ≤ f t ((ieee32RealView uL) t))
    (hL0 : (ieee32RealView uL) 0 ≤ a)
    (hU_cont : ContinuousOn (ieee32RealView uU) (Icc 0 T))
    (hU_der : ∀ t ∈ Ico 0 T, HasDerivWithinAt (ieee32RealView uU) ((ieee32RealView uU') t) (Ici t) t)
    (hU_sup : ∀ t ∈ Ico 0 T, f t ((ieee32RealView uU) t) ≤ (ieee32RealView uU') t)
    (hU0 : a ≤ (ieee32RealView uU) 0)
    (hLU : ∀ t ∈ Icc 0 T, (ieee32RealView uL) t ≤ (ieee32RealView uU) t)
    (hLower :
      ∀ t, T < t →
        0 ≤ f T ((ieee32RealView uL) T) ∧ f T ((ieee32RealView uL) T) ≤ f t ((ieee32RealView uL) T))
    (hUpper :
      ∀ t, T < t →
        f t ((ieee32RealView uU) T) ≤ f T ((ieee32RealView uU) T) ∧ f T ((ieee32RealView uU) T) ≤ 0) :
    (∀ t ∈ Icc 0 τ,
        constantExtensionAfter T (ieee32RealView uL) t ≤ (ieee32RealView u) t ∧
          (ieee32RealView u) t ≤ constantExtensionAfter T (ieee32RealView uU) t) ∧
      (∀ t ∈ Ico 0 τ, HasDerivWithinAt (ieee32RealView u) (f t ((ieee32RealView u) t)) (Ici t) t)
        := by
  simpa [ieee32RealView] using
    (extendedSolutionEnclosed_fromClampedDynamics (T := T) (τ := τ) hT hτ (f := f)
      (u := ieee32RealView u)
      (uL := ieee32RealView uL) (uU := ieee32RealView uU)
      (uL' := ieee32RealView uL') (uU' := ieee32RealView uU') (a := a)
      hu_cont hu_der hu0 hL_cont hL_der hL_sub hL0 hU_cont hU_der hU_sup hU0 hLU hLower hUpper)

end ConstantExtension

end

end Backend

end NN.Proofs.Verification.ODE.Enclosure
