/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Op.Pointwise
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Op.Table.Elementwise

/-!
# The derivative certificates and the reconstruction of `frexp`

Over the reals the mantissa output of `Op.frexp` is `frexpMantissa x = x / 2 ^ frexpExponent x`
at each entry, and its pullback multiplies the cotangent of the mantissa by `2 ^ (-e)` at the
exponent `e = frexpExponent x` of the input (`frexp_backward_eq`, at every input). The exponent
output is an integer word and has no cotangent. Away from zero and from the powers of two in
absolute value the exponent is constant near each entry, so the mantissa has the derivative
`1 / 2 ^ e` there (`hasDerivAt_frexpMantissa`), and the pullback is the adjoint of the derivative
of the mantissa (`frexp_backward_eq_adjoint`). At zero the stored pullback is the cotangent itself,
since the exponent of zero is `0`. Where the exponent of every entry lies in `[-2³¹, 2³¹)`, the
int32 exponent output does not wrap, and the mantissa output times two to its signed value is the
input (`frexp_forward_reconstruct`).

At every scalar whose capability record offers `Frexp` as `i`, `Op.frexp` has a rule whose outputs
at each entry are the mantissa of `i.frexp` and the int32 word of its exponent
(`frexp_forward_of_frexp?`), and every rule of `Op.frexp` is one of these (`frexp_forward_of_mem`).
In a configured binary format that is IEEE with exponent bias at least two, and whose exponents of
finite nonzero values lie in `[-2³¹, 2³¹)`, the mantissa output times two to the signed value of
the exponent output is the input at every finite entry (`frexp_forward_reconstruct_configured`).
These are the formats where the capability offers `Frexp`, so every rule of `Op.frexp` at a
configured format reconstructs its finite inputs (`frexp_forward_reconstruct_of_mem`).
-/

@[expose] public section

namespace TorchLean.Op

open TorchLean.Tensor

variable {s : Shape}

/-- Where no entry is zero or a power of two in absolute value, the real mantissa of `Op.frexp`
has the slope map of `1 / 2 ^ e` at the exponents `e` of the entries as derivative, and its
pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.frexp derivative]
theorem frexp_backward_eq_adjoint (x δ : Tensor ℝ s)
    (hx : ∀ c, x c ≠ 0 ∧ ∀ k : ℤ, |x c| ≠ 2 ^ k) :
    HasFDerivAt (fun y => (((Op.frexp (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, ())).1)
        (slopeCLM fun c => 1 / 2 ^ frexpExponent (x c)) x ∧
      (((Op.frexp (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
        (fderiv ℝ (fun y => (((Op.frexp (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => frexpMantissa)
    (fun c => 1 / 2 ^ frexpExponent (x c)) x δ _
    (fun y c => Tensor.map_apply _ y c)
    (fun c => hasDerivAt_frexpMantissa (hx c).1 (hx c).2)
    (fun c => by
      show (Tensor.zipWith _ x δ) c = _
      simp only [Tensor.zipWith_apply]
      show δ c * (2 : ℝ) ^ (-frexpExponent (x c)) = _
      rw [zpow_neg, one_div, mul_comm])

/-- At every real input, entry `c` of the stored pullback of `Op.frexp` is `δ c / 2 ^ e` at the
exponent `e` of `x c`, so `δ c` where `x c` is zero. -/
@[op_cert TorchLean.Op.frexp convention]
theorem frexp_backward_eq (x δ : Tensor ℝ s) :
    (((Op.frexp (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, (), ())).1 =
      Tensor.zipWith (fun t d => d / 2 ^ frexpExponent t) x δ := by
  show Tensor.zipWith (fun t d : ℝ => d * (2 : ℝ) ^ (-frexpExponent t)) x δ = _
  simp only [zpow_neg, div_eq_mul_inv]

/-- Where the exponent of every entry lies in `[-2³¹, 2³¹)`, entry `c` of the real mantissa output
of `Op.frexp` times two to the signed value of the int32 exponent output is `x c`. -/
theorem frexp_forward_reconstruct (x : Tensor ℝ s)
    (hx : ∀ c, -2 ^ 31 ≤ frexpExponent (x c) ∧ frexpExponent (x c) < 2 ^ 31) (c) :
    (((Op.frexp (s := s)).rule? (α := ℝ) |>.get rfl).forward (x, ())).1 c *
        2 ^ ((((Op.frexp (s := s)).rule? (α := ℝ) |>.get rfl).forward (x, ())).2.1 c).toInt =
      x c := by
  have hm : (((Op.frexp (s := s)).rule? (α := ℝ) |>.get rfl).forward (x, ())).1 c =
      frexpMantissa (x c) := Tensor.map_apply _ x c
  have he : (((Op.frexp (s := s)).rule? (α := ℝ) |>.get rfl).forward (x, ())).2.1 c =
      BitVec.ofInt 32 (frexpExponent (x c)) := Tensor.map_apply _ x c
  rw [hm, he, BitVec.toInt_ofInt_eq_self (by norm_num) (hx c).1 (hx c).2, ← frexp_reconstruct]

section Capability

variable {α : Type} [Storage α] [Context α]

/-- Where the capability record `caps` offers `Frexp` as `i`, `Op.frexp` has a rule whose outputs
at entry `c` are the mantissa of `i.frexp (x c)` and the int32 word of its exponent. -/
theorem frexp_forward_of_frexp? (caps : Op.Capabilities α) (i : Frexp α)
    (h : caps.frexp? = some i) (x : Tensor α s) :
    ∃ rule ∈ @Op.rule? α _ _ caps _ _ (Op.frexp (s := s)), ∀ c,
      (rule.forward (x, ())).1 c = (i.frexp (x c)).1 ∧
        (rule.forward (x, ())).2.1 c = BitVec.ofInt 32 (i.frexp (x c)).2 := by
  rcases caps with ⟨_, _, _, _, _, _, _, f⟩
  obtain rfl : f = some i := h
  exact ⟨_, rfl, fun c => ⟨Tensor.map_apply _ x c, Tensor.map_apply _ x c⟩⟩

/-- Every rule of `Op.frexp` comes from the capability record: where `Op.frexp` has the rule
`rule`, the record offers `Frexp` as some `i`, and at every input `x` the outputs of `rule` at
entry `c` are the mantissa of `i.frexp (x c)` and the int32 word of its exponent. -/
theorem frexp_forward_of_mem (caps : Op.Capabilities α) {rule}
    (h : rule ∈ @Op.rule? α _ _ caps _ _ (Op.frexp (s := s))) :
    ∃ i, caps.frexp? = some i ∧ ∀ (x : Tensor α s) c,
      (rule.forward (x, ())).1 c = (i.frexp (x c)).1 ∧
        (rule.forward (x, ())).2.1 c = BitVec.ofInt 32 (i.frexp (x c)).2 := by
  rcases hcap : caps.frexp? with _ | i
  · rcases caps with ⟨_, _, _, _, _, _, _, f⟩
    obtain rfl : f = none := hcap
    exact (Option.some_ne_none _ (show (none : Option _) = some rule from h).symm).elim
  · refine ⟨i, rfl, fun x c => ?_⟩
    obtain ⟨rule', hrule', hfwd⟩ := frexp_forward_of_frexp? caps i hcap x
    obtain rfl := Option.mem_unique h hrule'
    exact hfwd c

end Capability

section Configured

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- In an IEEE format of exponent bias at least two, where the exponents `e` with
`minSubnormalExponent + 1 ≤ e ≤ maxNormalExponent + 1` fit in an int32, `Op.frexp` has a rule
whose mantissa output times two to the signed value of its exponent output is the input at every
finite entry. -/
theorem frexp_forward_reconstruct_configured (hformat : format.isIEEE = true)
    (hbias : 2 ≤ format.exponentBias) (hlow : -2 ^ 31 ≤ format.minSubnormalExponent + 1)
    (hhigh : format.maxNormalExponent + 1 < 2 ^ 31) (x : Tensor Value s) :
    ∃ rule ∈ (Op.frexp (s := s)).rule? (α := Value), ∀ c,
      ExecFloat.Binary.isFinite (x c) = true →
        ExecFloat.Binary.toReal ((rule.forward (x, ())).1 c) *
            2 ^ ((rule.forward (x, ())).2.1 c).toInt = ExecFloat.Binary.toReal (x c) := by
  have hcap : Op.Capabilities.frexp? (α := Value) =
      some ⟨ConfiguredBinary.frexp, fun x n => ExecFloat.Binary.scale x n⟩ := by
    show (if _ then _ else none) = _
    rw [hformat, decide_eq_true hbias, decide_eq_true hlow, decide_eq_true hhigh]
    rfl
  obtain ⟨rule, hrule, hfwd⟩ := frexp_forward_of_frexp? _ _ hcap x
  refine ⟨rule, hrule, fun c hx => ?_⟩
  rw [(hfwd c).1, (hfwd c).2]
  exact ConfiguredBinary.toReal_frexp_mul_toInt hformat hbias hlow hhigh hx

/-- Every rule of `Op.frexp` at a configured binary format reconstructs the input: its mantissa
output times two to the signed value of its exponent output is the input at every finite entry.
The capability offers `Frexp` only where the premises of `frexp_forward_reconstruct_configured`
hold, so none is left here. -/
theorem frexp_forward_reconstruct_of_mem {rule}
    (h : rule ∈ (Op.frexp (s := s)).rule? (α := Value)) (x : Tensor Value s) (c)
    (hx : ExecFloat.Binary.isFinite (x c) = true) :
    ExecFloat.Binary.toReal ((rule.forward (x, ())).1 c) *
        2 ^ ((rule.forward (x, ())).2.1 c).toInt = ExecFloat.Binary.toReal (x c) := by
  obtain ⟨i, hcap, hfwd⟩ := frexp_forward_of_mem _ h
  change (if _ then _ else none) = _ at hcap
  split at hcap
  next hcond =>
    rw [Bool.and_eq_true, Bool.and_eq_true, Bool.and_eq_true, decide_eq_true_eq,
      decide_eq_true_eq, decide_eq_true_eq] at hcond
    obtain ⟨⟨⟨hformat, hbias⟩, hlow⟩, hhigh⟩ := hcond
    obtain rfl := Option.some.inj hcap
    rw [(hfwd x c).1, (hfwd x c).2]
    exact ConfiguredBinary.toReal_frexp_mul_toInt hformat hbias hlow hhigh hx
  next => exact (Option.some_ne_none _ hcap.symm).elim

end Configured

end TorchLean.Op
