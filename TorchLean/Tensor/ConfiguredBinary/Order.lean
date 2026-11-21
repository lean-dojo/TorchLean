/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Reduce.Order
public import FloatLib.Floats.Formats.BinaryInterchange.Operations.Compare.EReal

/-!
# Running maxima and minima of configured binary values in every order

`key` maps a configured binary value to its extended real and a NaN to `⊤`, so both zeros have the
key `0`. The test of `Tensor.cummax` moves from `p` to `q` exactly when `key p ≤ key q`
(`move_max`): to a later entry at a tie, `-0` and `+0` included, and to a NaN, and from a NaN only
to a NaN. The test of `Tensor.cummin` moves exactly when the key of `q` is at most that of `p` in
the reversed order of the extended reals, with `⊤` still last (`move_min`). So at every configured
binary format, in every order that covers the length of the axis, `Tensor.cummaxIn` and
`Tensor.cumminIn` give the values and indices of `Tensor.cummax` and `Tensor.cummin`, bit for bit
(`cummaxIn_eq`, `cumminIn_eq`).
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange
open Tensor (ReduceOrder)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code] {s : Shape}

local notation "Value" => ExecFloat (Configured.Family format code plan)

namespace ConfiguredBinary

/-- The extended real of a configured binary value, `⊤` at a NaN. -/
noncomputable def key (x : Value) : WithTop EReal :=
  Model.toEReal? (ExecFloat.Binary.toModel x)

private theorem compare_of_key {x y : Value} {a b : EReal} (hx : key x = a) (hy : key y = b) :
    ExecFloat.compare x y = some (compare a b) := by
  rw [ExecFloat.Binary.compare_eq_compare_toModel]
  exact Model.compare_eq_of_toEReal? hx hy

private theorem beq_self_of_key_top {x : Value} (h : key x = ⊤) : (x == x) = false := by
  show (ExecFloat.compare x x == some Ordering.eq) = false
  rw [ExecFloat.Binary.compare_eq_compare_toModel, (Model.compare_eq_none_iff _ _).2
    (Or.inl ((Model.toEReal?_eq_none_iff_isNaN_eq_true _).1 h))]
  rfl

private theorem beq_self_of_key_coe {x : Value} {a : EReal} (h : key x = a) : (x == x) = true := by
  show (ExecFloat.compare x x == some Ordering.eq) = true
  rw [compare_of_key h h]
  simp

private theorem gt_iff_of_key {p q : Value} {a b : EReal} (hp : key p = a) (hq : key q = b) :
    p > q ↔ b < a := by
  show (ExecFloat.compare q p == some Ordering.lt) = true ↔ b < a
  rw [compare_of_key hq hp]
  simp [compare_lt_iff_lt]

/-- The test of `Tensor.cummax` moves from `p` to `q` exactly when `key p ≤ key q`. -/
theorem move_max (p q : Value) :
    (!(q == q) || ((p == p) && !decide (p > q))) = decide (key p ≤ key q) := by
  by_cases hq : key q = ⊤
  · simp [beq_self_of_key_top hq, hq]
  obtain ⟨b, hb⟩ := WithTop.ne_top_iff_exists.1 hq
  by_cases hp : key p = ⊤
  · simp [beq_self_of_key_coe hb.symm, beq_self_of_key_top hp, hp, ← hb]
  obtain ⟨a, ha⟩ := WithTop.ne_top_iff_exists.1 hp
  simp [beq_self_of_key_coe hb.symm, beq_self_of_key_coe ha.symm, gt_iff_of_key ha.symm hb.symm,
    ← ha, ← hb]
  rw [Bool.eq_iff_iff]
  simp [not_le]

/-- The test of `Tensor.cummin` moves from `p` to `q` exactly when the key of `q`, with the
extended reals reversed and `⊤` still last, is at least that of `p`. -/
theorem move_min (p q : Value) :
    (!(q == q) || ((p == p) && !decide (q > p))) =
      decide ((key p).map OrderDual.toDual ≤ (key q).map OrderDual.toDual) := by
  by_cases hq : key q = ⊤
  · simp [beq_self_of_key_top hq, hq]
  obtain ⟨b, hb⟩ := WithTop.ne_top_iff_exists.1 hq
  by_cases hp : key p = ⊤
  · simp [beq_self_of_key_coe hb.symm, beq_self_of_key_top hp, hp, ← hb]
  obtain ⟨a, ha⟩ := WithTop.ne_top_iff_exists.1 hp
  simp [beq_self_of_key_coe hb.symm, beq_self_of_key_coe ha.symm, gt_iff_of_key hb.symm ha.symm,
    ← ha, ← hb]
  rw [Bool.eq_iff_iff]
  simp [not_le]

/-- At a configured binary format, running maxima in an order that covers the length of the axis
are `Tensor.cummax`, bit for bit and with their indices. -/
theorem cummaxIn_eq {order : ReduceOrder} (x : Tensor Value s) (dim : Fin s.rank)
    (hcov : order.Covers (s.extent dim)) : Tensor.cummaxIn order x dim = Tensor.cummax x dim :=
  (Tensor.cummaxIn_eq_leftFold key move_max x dim hcov).trans (Tensor.cummaxIn_leftFold x dim)

/-- At a configured binary format, running minima in an order that covers the length of the axis
are `Tensor.cummin`, bit for bit and with their indices. -/
theorem cumminIn_eq {order : ReduceOrder} (x : Tensor Value s) (dim : Fin s.rank)
    (hcov : order.Covers (s.extent dim)) : Tensor.cumminIn order x dim = Tensor.cummin x dim :=
  (Tensor.cumminIn_eq_leftFold _ move_min x dim hcov).trans (Tensor.cumminIn_leftFold x dim)

end ConfiguredBinary

end TorchLean
