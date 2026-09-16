/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Interval.IEEEExec32
public import NN.MLTheory.Proofs.Approximation.FloatInterval.Semantics
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# Constant rounded targets over `Interval32`

Exact interval-image theorem for constant rounded targets over `ExecFloat.Binary 8 23`.

This file packages the finite-float base case for the concrete `IEEE32Exec.Interval32` interval
type: a constant rounded target has exact interval semantics given by the point interval `[c,c]` on
every valid input box.

The companion file `FloatInterval.Semantics` develops the same idea for the abstract interval
domain `I` used by the MLP interval evaluator. We keep this file separate because it is the
direct `Interval32` statement used at the low-level rounded-target boundary, while
`FloatInterval.Semantics` is the reusable abstract-domain semantics.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.MLTheory.Proofs.UniversalApproximation

open TorchLean.Floats.IEEE754
open FloatLib.Floats.Formats.BinaryInterchange

namespace FloatIntervalApprox.ConstantTarget

open IEEE32Exec

-- The scalar type and the extremum vocabulary are the companion file's; only the interval type
-- differs here, because this file works over the concrete `Interval32` rather than the abstract
-- domain. Borrowing them keeps one definition of "minimum on a set" for the whole development.
open FloatIntervalApprox (F)
open FloatIntervalApprox.ExactImage (Icc IsMinOn IsMaxOn)

noncomputable section

/-- Shorthand for the float32 interval type `IEEE32Exec.Interval32`. -/
abbrev I : Type := IEEE32Exec.Interval32

/-- Product box of float32 intervals. -/
abbrev Box (d : Nat) : Type := Fin d → I

/-- Concretization of a float32 interval to a set of float32 values. -/
def γI (J : I) : Set F := fun x => x ∈ J

/-- Concretization of a product box to a set of float32 vectors. -/
def γ {d : Nat} (B : Box d) : Set (Fin d → F) := fun x => ∀ i, x i ∈ B i

/-- Basic “well-formedness” predicate for product boxes. -/
def BoxValid {d : Nat} (B : Box d) : Prop := ∀ i, Interval32.Valid (B i)

/-- `B` is a box contained in `[-1,1]^d`. -/
def BoxInCube {d : Nat} (B : Box d) : Prop :=
  ∀ i, ((-1) : F) ≤ (B i).lo ∧ (B i).hi ≤ (1 : F)

/--
Exact interval-image property, phrased as:
for every valid input box `B`, the interval semantics `nuInt(B)`’s concretization is exactly the
float interval between the min/max of the target’s direct image on `γ(B)`.

We phrase extrema relationally, rather than through a chosen float `min`/`max` operator, because
NaN-aware binary32 orders need their edge cases stated explicitly.
-/
def ExactIntervalImage {d : Nat} (g : (Fin d → F) → F) (_ν : (Fin d → F) → F)
    (nuInt : Box d → I) : Prop :=
  ∀ B, BoxValid B →
    ∃ m M,
      IsMinOn g (γ (d := d) B) m ∧
      IsMaxOn g (γ (d := d) B) M ∧
      γI (nuInt B) = Icc m M

/--
Generic exact-interval-image statement shape for `ExecFloat.Binary 8 23` rounded targets.
-/
def RoundedTargetExactIntervalImageStatement (d : Nat) : Prop :=
  ∀ (fHat : (Fin d → F) → F),
    (∀ x, ExecFloat.Binary.isNaN (fHat x) = false) →
    ∃ (_ν : (Fin d → F) → F) (nuInt : Box d → I),
      (∀ B, BoxValid B → BoxInCube (d := d) B →
        ∃ m M,
          IsMinOn fHat (γ (d := d) B) m ∧
          IsMaxOn fHat (γ (d := d) B) M ∧
          γI (nuInt B) = Icc m M)

/-- Float comparison is reflexive on finite values. Not a `Preorder` instance, because `NaN` is not
comparable to itself and IEEE 754 order is genuinely partial. -/
theorem le_refl_of_isFinite (x : F) (hx : ExecFloat.Binary.isFinite x = true) : x ≤ x := by
  apply FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel.mpr
  exact (Model.Interval.le_iff_toReal_le_of_isFinite
    (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel x) hx hx).mpr le_rfl

/-- A valid box is nonempty, witnessed by its own lower corner.

The exact-image statements quantify over nonempty concretizations, so this is what discharges that
hypothesis for any box the checker actually produces. -/
theorem gamma_nonempty_of_BoxValid {d : Nat} {B : Box d} (hB : BoxValid B) :
    (γ (d := d) B).Nonempty := by
  refine ⟨fun i => (B i).lo, ?_⟩
  intro i
  have hv : Interval32.Valid (B i) := hB i
  have hlelo : (B i).lo ≤ (B i).lo := le_refl_of_isFinite (x := (B i).lo) hv.1
  exact And.intro (FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel.mp hlelo) hv.2.2

/--
Base case: a constant target `g(x) = c` has an exact interval-image witness given by the constant
network and the point interval `[c,c]`.
-/
theorem exactIntervalImage_constant {d : Nat} (c : F) (hc : ExecFloat.Binary.isFinite c = true) :
    ExactIntervalImage (d := d) (g := fun _ => c) (_ν := fun _ => c)
      (nuInt := fun _ => Interval32.point c) := by
  intro B hB
  refine ⟨c, c, ?_, ?_, ?_⟩
  · -- `IsMinOn`
    have hn : (γ (d := d) B).Nonempty := gamma_nonempty_of_BoxValid (d := d) hB
    rcases hn with ⟨x0, hx0⟩
    refine And.intro ?_ ?_
    · exact ⟨x0, hx0, rfl⟩
    · intro y hy
      rcases hy with ⟨x, hx, hgy⟩
      subst hgy
      simpa using (le_refl_of_isFinite (x := c) hc)
  · -- `IsMaxOn`
    have hn : (γ (d := d) B).Nonempty := gamma_nonempty_of_BoxValid (d := d) hB
    rcases hn with ⟨x0, hx0⟩
    refine And.intro ?_ ?_
    · exact ⟨x0, hx0, rfl⟩
    · intro y hy
      rcases hy with ⟨x, hx, hgy⟩
      subst hgy
      simpa using (le_refl_of_isFinite (x := c) hc)
  · -- `γ([c,c]) = Icc c c`
    ext x
    dsimp [γI, Icc]
    change Model.Interval.mem (Interval32.toModel (Interval32.point c))
      (ExecFloat.Binary.toModel x) ↔ (c ≤ x ∧ x ≤ c)
    rw [Interval32.toModel_point]
    change (Model.le (ExecFloat.Binary.toModel c) (ExecFloat.Binary.toModel x) ∧
      Model.le (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel c)) ↔ (c ≤ x ∧ x ≤ c)
    simp only [FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel]

end

end FloatIntervalApprox.ConstantTarget

end NN.MLTheory.Proofs.UniversalApproximation
