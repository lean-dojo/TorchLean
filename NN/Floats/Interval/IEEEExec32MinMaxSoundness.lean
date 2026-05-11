/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32Total
public import NN.Floats.Interval.IEEEExec32

/-!
# `min4`/`max4` real semantics (finite regime)

`NN/Floats/Interval/IEEEExec32.lean` exposes helper combinators:

- `IEEE32Exec.Interval32.min4`
- `IEEE32Exec.Interval32.max4`

used to implement the classical 4-corner multiplication enclosure rule.

This file provides small lemmas that compute the `toReal` semantics of these helpers when all
arguments are finite.

These lemmas are intended as building blocks for a larger `Interval32.mul` soundness theorem.
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

namespace Interval32

noncomputable section

/--
Real semantics of `Interval32.min4` in the finite regime.

If all four arguments are finite, then `toReal (min4 a b c d)` is the corresponding nested `min`
of the four real values.
-/
theorem toReal_min4_eq_min_of_isFinite (a b c d : IEEE32Exec)
    (ha : isFinite a = true) (hb : isFinite b = true) (hc : isFinite c = true) (hd : isFinite d =
      true) :
    toReal (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.min4 a b c d) =
      min (min (toReal a) (toReal b)) (min (toReal c) (toReal d)) := by
  have hab : toReal (minimum a b) = min (toReal a) (toReal b) :=
    toReal_minimum_eq_min_of_isFinite (x := a) (y := b) ha hb
  have hcd : toReal (minimum c d) = min (toReal c) (toReal d) :=
    toReal_minimum_eq_min_of_isFinite (x := c) (y := d) hc hd
  have hfin_ab : isFinite (minimum a b) = true := isFinite_minimum_of_isFinite (x := a) (y := b) ha
    hb
  have hfin_cd : isFinite (minimum c d) = true := isFinite_minimum_of_isFinite (x := c) (y := d) hc
    hd
  have houter :
      toReal (minimum (minimum a b) (minimum c d)) =
        min (toReal (minimum a b)) (toReal (minimum c d)) :=
    toReal_minimum_eq_min_of_isFinite (x := minimum a b) (y := minimum c d) hfin_ab hfin_cd
  -- Avoid `simp` rewriting `toReal` into its `toDyadic?` match form too early, which makes
  -- rewriting with `hab`/`hcd` brittle.
  have houter' :
      toReal (minimum (minimum a b) (minimum c d)) =
        min (min (toReal a) (toReal b)) (min (toReal c) (toReal d)) := by
    calc
      toReal (minimum (minimum a b) (minimum c d))
          = min (toReal (minimum a b)) (toReal (minimum c d)) := houter
      _ = min (min (toReal a) (toReal b)) (min (toReal c) (toReal d)) := by
          rw [hab, hcd]
  simpa [TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.min4] using houter'

/--
Real semantics of `Interval32.max4` in the finite regime.

If all four arguments are finite, then `toReal (max4 a b c d)` is the corresponding nested `max`
of the four real values.
-/
theorem toReal_max4_eq_max_of_isFinite (a b c d : IEEE32Exec)
    (ha : isFinite a = true) (hb : isFinite b = true) (hc : isFinite c = true) (hd : isFinite d =
      true) :
    toReal (TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.max4 a b c d) =
      max (max (toReal a) (toReal b)) (max (toReal c) (toReal d)) := by
  have hab : toReal (maximum a b) = max (toReal a) (toReal b) :=
    toReal_maximum_eq_max_of_isFinite (x := a) (y := b) ha hb
  have hcd : toReal (maximum c d) = max (toReal c) (toReal d) :=
    toReal_maximum_eq_max_of_isFinite (x := c) (y := d) hc hd
  have hfin_ab : isFinite (maximum a b) = true := isFinite_maximum_of_isFinite (x := a) (y := b) ha
    hb
  have hfin_cd : isFinite (maximum c d) = true := isFinite_maximum_of_isFinite (x := c) (y := d) hc
    hd
  have houter :
      toReal (maximum (maximum a b) (maximum c d)) =
        max (toReal (maximum a b)) (toReal (maximum c d)) :=
    toReal_maximum_eq_max_of_isFinite (x := maximum a b) (y := maximum c d) hfin_ab hfin_cd
  have houter' :
      toReal (maximum (maximum a b) (maximum c d)) =
        max (max (toReal a) (toReal b)) (max (toReal c) (toReal d)) := by
    calc
      toReal (maximum (maximum a b) (maximum c d))
          = max (toReal (maximum a b)) (toReal (maximum c d)) := houter
      _ = max (max (toReal a) (toReal b)) (max (toReal c) (toReal d)) := by
          rw [hab, hcd]
  simpa [TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.max4] using houter'

end

end Interval32

end IEEE32Exec

end TorchLean.Floats.IEEE754
