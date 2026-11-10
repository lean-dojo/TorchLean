/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Approximation.FloatInterval.Semantics

/-!
# Constant targets over binary32 interval boxes

We prove the base case of exact interval images for FloatLib's binary32 intervals
`Interval Binary32`. A box of valid intervals has a member, its lower corner, and a finite constant
`c` has exact interval images under the constant evaluator `Binary.Interval.point c`.

The companion module `FloatInterval.Semantics` works with the abstract domain `I` of the network
evaluator, whose top element contains NaN. A FloatLib interval holds no NaN, since NaN is unordered.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat Binary32)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.Formats.BinaryInterchange (Model)

namespace TorchLean.UniversalApproximation

namespace FloatIntervalApprox.ConstantTarget

/-- A box of valid binary32 intervals has a member, its lower corner. -/
theorem box_nonempty_of_valid {d : Nat} {B : Fin d → Interval Binary32}
    (hB : ∀ i, Binary.Interval.Valid (B i)) :
    {x : Fin d → Binary32 | ∀ i, x i ∈ B i}.Nonempty := by
  refine ⟨fun i => (B i).lo, ?_⟩
  intro i
  have hv : Binary.Interval.Valid (B i) := hB i
  have hlo : (B i).lo ≤ (B i).lo :=
    (ExecFloat.Binary.le_self_iff _).mpr (Model.isNaN_eq_false_of_isFinite_eq_true _ hv.1)
  exact And.intro (ExecFloat.Binary.le_iff_le_toModel.mp hlo) hv.2.2

/-- A finite constant `c` has exact interval images under the constant evaluator
`Binary.Interval.point c`. -/
theorem exactIntervalImage_constant {d : Nat} (c : Binary32)
    (hc : ExecFloat.Binary.isFinite c = true) :
    ExactIntervalImage (d := d) (fun _ => c) (fun _ => Binary.Interval.point c) := by
  intro B hne
  have hcc : c ≤ c :=
    (ExecFloat.Binary.le_self_iff c).mpr (Model.isNaN_eq_false_of_isFinite_eq_true _ hc)
  obtain ⟨x0, hx0⟩ := hne
  refine ⟨c, c, ⟨⟨x0, hx0, rfl⟩, ?_⟩, ⟨⟨x0, hx0, rfl⟩, ?_⟩, ?_⟩
  · rintro _ ⟨_, _, rfl⟩
    exact hcc
  · rintro _ ⟨_, _, rfl⟩
    exact hcc
  · ext x
    change Model.Interval.mem (Binary.Interval.toModel (Binary.Interval.point c))
      (ExecFloat.Binary.toModel x) ↔ (c ≤ x ∧ x ≤ c)
    simp only [Binary.Interval.toModel_point]
    change (Model.le (ExecFloat.Binary.toModel c) (ExecFloat.Binary.toModel x) ∧
      Model.le (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel c)) ↔ (c ≤ x ∧ x ≤ c)
    simp only [ExecFloat.Binary.le_iff_le_toModel]

end FloatIntervalApprox.ConstantTarget

end TorchLean.UniversalApproximation
