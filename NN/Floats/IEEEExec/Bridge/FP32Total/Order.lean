/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.Compare
public import NN.Floats.IEEEExec.Bridge.FP32Total.Core

/-!
# Total FP32 Bridge: Ordering

This module relates executable IEEE comparisons to real order when the operands are finite. It
keeps unordered NaN cases explicit rather than coercing every bit pattern into `ℝ`; signed zeros,
which have distinct encodings but equal real values, are handled by the IEEE comparison rules.

See `FP32Total.Core` for the total bridge convention and references.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-- On finite values, IEEE-754 `minimum` agrees with real `min`. -/
theorem toReal_minimum_eq_min_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal (minimum x y) = min (toReal x) (toReal y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact toReal_minimum_eq_min (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, IEEE-754 `maximum` agrees with real `max`. -/
theorem toReal_maximum_eq_max_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    toReal (maximum x y) = max (toReal x) (toReal y) := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact toReal_maximum_eq_max (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .lt` iff `toReal x < toReal y`. -/
theorem compare_eq_some_lt_iff_toReal_lt_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .lt ↔ toReal x < toReal y := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact compare_eq_some_lt_iff_toReal_lt (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .eq` iff `toReal x = toReal y`. -/
theorem compare_eq_some_eq_iff_toReal_eq_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .eq ↔ toReal x = toReal y := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact compare_eq_some_eq_iff_toReal_eq (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

/-- On finite values, `compare x y = some .gt` iff `toReal y < toReal x`. -/
theorem compare_eq_some_gt_iff_toReal_gt_of_isFinite (x y : IEEE32Exec)
    (hx : isFinite x = true) (hy : isFinite y = true) :
    compare x y = some .gt ↔ toReal y < toReal x := by
  classical
  cases hdx : toDyadic? x with
  | none =>
      have : isFinite x = false := isFinite_eq_false_of_toDyadic?_eq_none (x := x) hdx
      have : False := by
        simp [hx]  at this
      exact this.elim
  | some dx =>
      cases hdy : toDyadic? y with
      | none =>
          have : isFinite y = false := isFinite_eq_false_of_toDyadic?_eq_none (x := y) hdy
          have : False := by
            simp [hy]  at this
          exact this.elim
      | some dy =>
          exact compare_eq_some_gt_iff_toReal_gt (x := x) (y := y) (dx := dx) (dy := dy) hdx hdy

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
