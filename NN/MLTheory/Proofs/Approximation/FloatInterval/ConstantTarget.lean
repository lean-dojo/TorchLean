/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Proofs.Approximation.FloatInterval.Semantics
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# Constant rounded targets over binary32 intervals

Exact interval-image theorem for constant rounded targets over `Binary 8 23`.

This file packages the finite-float base case for FloatLib's binary32 intervals: a constant
rounded target has exact interval semantics given by the point interval `[c,c]` on every valid
input box.

The companion file `FloatInterval.Semantics` develops the same idea for the abstract interval
domain `I` used by the MLP interval evaluator. This file states the result directly for binary32
endpoints at the rounded-target boundary.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.MLTheory.Proofs.UniversalApproximation

open FloatLib.Floats.Formats.BinaryInterchange

namespace FloatIntervalApprox.ConstantTarget


-- The scalar type and the extremum vocabulary are the companion file's; only the interval type
-- differs here, because this file uses FloatLib's concrete endpoint intervals.
open FloatIntervalApprox (F)
open FloatIntervalApprox.ExactImage (Icc IsMinOn IsMaxOn)

noncomputable section

/-- Product box of float32 intervals. -/
abbrev Box (d : Nat) : Type := Fin d → Interval F

/-- Concretization of a float32 interval to a set of float32 values. -/
def γI (J : Interval F) : Set F := fun x => x ∈ J

/-- Concretization of a product box to a set of float32 vectors. -/
def γ {d : Nat} (B : Box d) : Set (Fin d → F) := fun x => ∀ i, x i ∈ B i

/-- Basic “well-formedness” predicate for product boxes. -/
def BoxValid {d : Nat} (B : Box d) : Prop := ∀ i, Binary.Interval.Valid (B i)

/--
Exact interval-image property, phrased as:
for every valid input box `B`, the interval semantics `nuInt(B)`’s concretization is exactly the
float interval between the min/max of the target’s direct image on `γ(B)`.

We phrase extrema relationally, rather than through a chosen float `min`/`max` operator, because
NaN-aware binary32 orders need their edge cases stated explicitly.
-/
def ExactIntervalImage {d : Nat} (g : (Fin d → F) → F)
    (nuInt : Box d → Interval F) : Prop :=
  ∀ B, BoxValid B →
    ∃ m M,
      IsMinOn g (γ (d := d) B) m ∧
      IsMaxOn g (γ (d := d) B) M ∧
      γI (nuInt B) = Icc m M

/-- Float comparison is reflexive on finite values. -/
private theorem le_self_of_isFinite (x : F) (hx : ExecFloat.Binary.isFinite x = true) : x ≤ x :=
  ExecLemmas.le_self_of_isNaN_false x (Model.isNaN_eq_false_of_isFinite_eq_true _ hx)

/-- A valid box is nonempty, witnessed by its own lower corner.

The exact-image statements quantify over nonempty concretizations, so this is what discharges that
hypothesis for any box the checker actually produces. -/
theorem gamma_nonempty_of_BoxValid {d : Nat} {B : Box d} (hB : BoxValid B) :
    (γ (d := d) B).Nonempty := by
  refine ⟨fun i => (B i).lo, ?_⟩
  intro i
  have hv : Binary.Interval.Valid (B i) := hB i
  have hlelo : (B i).lo ≤ (B i).lo := le_self_of_isFinite (x := (B i).lo) hv.1
  exact And.intro (FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel.mp hlelo) hv.2.2

/--
Base case: a constant target `g(x) = c` has an exact interval-image witness given by the constant
network and the point interval `[c,c]`.
-/
theorem exactIntervalImage_constant {d : Nat} (c : F) (hc : ExecFloat.Binary.isFinite c = true) :
    ExactIntervalImage (d := d) (g := fun _ => c)
      (nuInt := fun _ => Binary.Interval.point c) := by
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
      simpa using (le_self_of_isFinite (x := c) hc)
  · -- `IsMaxOn`
    have hn : (γ (d := d) B).Nonempty := gamma_nonempty_of_BoxValid (d := d) hB
    rcases hn with ⟨x0, hx0⟩
    refine And.intro ?_ ?_
    · exact ⟨x0, hx0, rfl⟩
    · intro y hy
      rcases hy with ⟨x, hx, hgy⟩
      subst hgy
      simpa using (le_self_of_isFinite (x := c) hc)
  · -- `γ([c,c]) = Icc c c`
    ext x
    dsimp [γI, Icc]
    change Model.Interval.mem (Binary.Interval.toModel (Binary.Interval.point c))
      (ExecFloat.Binary.toModel x) ↔ (c ≤ x ∧ x ≤ c)
    simp only [Binary.Interval.toModel_point]
    change (Model.le (ExecFloat.Binary.toModel c) (ExecFloat.Binary.toModel x) ∧
      Model.le (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel c)) ↔ (c ≤ x ∧ x ≤ c)
    simp only [FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel]

end

end FloatIntervalApprox.ConstantTarget

end NN.MLTheory.Proofs.UniversalApproximation
