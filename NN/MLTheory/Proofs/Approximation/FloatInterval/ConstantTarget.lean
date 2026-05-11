/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Fin.Tuple.Basic
public import Mathlib.Data.Set.Image
public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Floats.Interval.IEEEExec32

/-!
# Constant rounded targets over `Interval32`

Exact interval-image theorem for constant rounded targets over `IEEE32Exec`.

This file packages the finite-float base case for the concrete `IEEE32Exec.Interval32` interval
type: a constant rounded target has exact interval semantics given by the point interval `[c,c]` on
every valid input box.

The companion file `FloatInterval.Semantics` develops the same idea for the abstract interval
domain `I` used by the MLP interval evaluator. We keep this file separate because it is the
direct `Interval32` statement used at the low-level rounded-target boundary, while
`FloatInterval.Semantics` is the reusable abstract-domain semantics.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.UniversalApproximation

open TorchLean.Floats.IEEE754

namespace FloatIntervalApprox.ConstantTarget

open IEEE32Exec

noncomputable section

/-- Shorthand for the float32 executable type `IEEE32Exec`. -/
abbrev F : Type := IEEE32Exec
/-- Shorthand for the float32 interval type `IEEE32Exec.Interval32`. -/
abbrev I : Type := IEEE32Exec.Interval32

/-- Product box of float32 intervals. -/
abbrev Box (d : Nat) : Type := Fin d → I

/-- Float interval set `{x | a ≤ x ∧ x ≤ b}` (avoids requiring `Preorder`). -/
def Icc (a b : F) : Set F := fun x => a ≤ x ∧ x ≤ b

/-- Concretization of a float32 interval to a set of float32 values. -/
def γI (J : I) : Set F := fun x => x ∈ J

/-- Concretization of a product box to a set of float32 vectors. -/
def γ {d : Nat} (B : Box d) : Set (Fin d → F) := fun x => ∀ i, x i ∈ B i

/-- Basic “well-formedness” predicate for product boxes. -/
def BoxValid {d : Nat} (B : Box d) : Prop := ∀ i, Interval32.Valid (B i)

/-- `B` is a box contained in `[-1,1]^d`. -/
def BoxInCube {d : Nat} (B : Box d) : Prop :=
  ∀ i, (Numbers.neg_one : F) ≤ (B i).lo ∧ (B i).hi ≤ (Numbers.one : F)

/-- `m` is a minimum of `g` on the set `S`, stated without choosing a canonical `min`. -/
def IsMinOn {X : Type} (g : X → F) (S : Set X) (m : F) : Prop :=
  (∃ x, x ∈ S ∧ g x = m) ∧ ∀ y, (∃ x, x ∈ S ∧ g x = y) → m ≤ y

/-- `M` is a maximum of `g` on the set `S`, stated without choosing a canonical `max`. -/
def IsMaxOn {X : Type} (g : X → F) (S : Set X) (M : F) : Prop :=
  (∃ x, x ∈ S ∧ g x = M) ∧ ∀ y, (∃ x, x ∈ S ∧ g x = y) → y ≤ M

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
Generic exact-interval-image statement shape for `IEEE32Exec` rounded targets.
-/
def RoundedTargetExactIntervalImageStatement (d : Nat) : Prop :=
  ∀ (fHat : (Fin d → F) → F),
    (∀ x, isNaN (fHat x) = false) →
    ∃ (_ν : (Fin d → F) → F) (nuInt : Box d → I),
      (∀ B, BoxValid B → BoxInCube (d := d) B →
        ∃ m M,
          IsMinOn fHat (γ (d := d) B) m ∧
          IsMaxOn fHat (γ (d := d) B) M ∧
          γI (nuInt B) = Icc m M)

theorem le_refl_of_isFinite (x : F) (hx : isFinite x = true) : x ≤ x := by
  have hcmp : compare x x = some .eq := by
    have h :=
      (compare_eq_some_eq_iff_toReal_eq_of_isFinite (x := x) (y := x) hx hx)
    exact h.mpr rfl
  change IEEE32Exec.le x x
  simp [IEEE32Exec.le, hcmp]

theorem gamma_nonempty_of_BoxValid {d : Nat} {B : Box d} (hB : BoxValid B) :
    (γ (d := d) B).Nonempty := by
  refine ⟨fun i => (B i).lo, ?_⟩
  intro i
  have hv : Interval32.Valid (B i) := hB i
  have hlelo : (B i).lo ≤ (B i).lo := le_refl_of_isFinite (x := (B i).lo) hv.1
  exact And.intro hlelo hv.2.2

/--
Base case: a constant target `g(x) = c` has an exact interval-image witness given by the constant network and the
point interval `[c,c]`.
-/
theorem exactIntervalImage_constant {d : Nat} (c : F) (hc : isFinite c = true) :
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
    dsimp [Interval32.point]
    change IEEE32Exec.Interval32.mem { lo := c, hi := c } x ↔ (c ≤ x ∧ x ≤ c)
    dsimp [IEEE32Exec.Interval32.mem]
    change (IEEE32Exec.le c x ∧ IEEE32Exec.le x c) ↔ (IEEE32Exec.le c x ∧ IEEE32Exec.le x c)
    exact Iff.rfl

end

end FloatIntervalApprox.ConstantTarget

end NN.MLTheory.Proofs.UniversalApproximation
