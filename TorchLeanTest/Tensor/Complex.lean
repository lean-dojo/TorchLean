/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Op.Complex
public import TorchLean.NN.Op.ComplexRound
public import TorchLean.NN.PyTorch.Export.IR
public import TorchLean.Tensor.ConfiguredBinary
public import TorchLean.Tensor.Op.Need
public import TorchLean.Tensor.Op.Table.Elementwise
public import FloatLib.Tactic.FloatDecide
public meta import TorchLean.NN.PyTorch.Export.IR
public meta import TorchLean.Tensor.ConfiguredBinary
public meta import TorchLean.Tensor.Op.Need

/-!
# The complex functions on pairs of parts

In binary32 the rule of `Op.complex` applies FloatLib's `ExecComplex` functions to the pair of parts
at each entry: `exp (0 + 0i)` is `1 + 0i`, and the pullback of the seed `1 + 0i` is the conjugate of
the derivative `exp z`, here `1 + 0i`. On the branch cut the sign of the zero imaginary part selects
the side (C23 7.3.3): `sqrt (-4 ± 0i)` is `0 ± 2i` and `log (-1 ± 0i)` is `0 ± πi`, with `π` rounded
to binary32. In every scalar type with `ComplexFunctions` the forward applies the function to the
pair at each coordinate (`complex_forward_pairAt`); `Float` has none, so the operator has no rule
there. Its wire record is tagged `complex` with the `jax.lax` name of the function and the shape (it
reads back, `Op.Wire.decode_encode_complex`). Over the reals, at the shape `[3]`, the pullback of
`Op.complex .exp` is the adjoint of the derivative at every input (`complex_exp_pullback`), that of
`Op.complex .log` at the input whose entries are all `1 + 0i` (`complex_log_pullback`), and that
of `Op.complex .tan` at the input whose entries are all `0` (`complex_tan_pullback`), that of
`Op.complex .atanh` at the input whose entries are all `1 / 2` (`complex_atanh_pullback`) and that
of `Op.complex .asinh` at the input whose entries are all `1 + 2i` (`complex_asinh_pullback`):
instances of `Op.complex_domain_backward_eq_adjoint` on the whole plane, the slit plane, off the
poles, between the two cuts of `atanh` on the real axis and off the cuts of `asinh` above `i`.

In binary32 the rule of `Op.complexPow` applies FloatLib's `ExecComplex.pow`, each part of the
exact power rounded once where it is certified: `(-4 ± 0i) ^ (1 / 2)` is `0 ± 2i`, `(1 + i) ^ 2` is
`2i`, `(1 + 2i) ^ (3 - i)` is the exact value rounded once (mpmath), `NaN ^ 0` is `1`, `0 ^ (-1)`
is `∞ + NaN i` and `0 ^ (2 + i)` is `0`. At the base `2` and the exponent `1` the pullbacks of the
seed `1 + 0i` are `1` to the base and `2 log 2` to the exponent. In every scalar type with
`ComplexFunctions` the forward applies `ComplexFunctions.pow` to the pairs at each coordinate
(`complexPow_forward_pairAt`), and its wire record is tagged `complex_pow` with the shape. Over the
reals the pullback is the adjoint of the derivative in both inputs at the base entries `2` and
every exponent (`complex_pow_pullback`), and in the exponent at the base entries `-1`, on the cut
(`complex_pow_pullback_right`).

In binary32, at the shape `[3]`, each part of each entry of the result is the part of the exact
value rounded once to nearest even: of `exp z` at every finite entry `z` of every input
(`complex_exp_round`), of `log 1 = 0` and `acos 1 = 0` at the input whose entries are all `1 + 0i`
(`complex_log_round`, `complex_acos_round`), of `acosh 1 = 0` at the input whose entries are all
`1 - 0i`, with the imaginary sign bit set (`complex_acosh_round`), of `acos 2 = i log (2 + √3)` at
`2 - 0i`, on the cut from below (`complex_acos_cut_round`), of `asinh (-1) = log (√2 - 1)` at
`-1 + 0i` (`complex_asinh_round`) and `asin (-1) = -π / 2` at `-1 - 0i` (`complex_asin_round`),
where FloatLib reflects the first quadrant, and of `2 ^ 1 = 2` for `Op.complexPow` at the base
entries `2 + 0i` and the exponent entries `1 + 0i` (`complex_pow_round`). These are instances of
`Op.complex_forward_value` and `Op.complexPow_forward_value`, whose premises the kernel decides at
those words.
-/

@[expose] public section

namespace TorchLean.Test.Complex

open FloatLib.Floats

/-- The words of the parts of the result and of the pullback of the seed `1 + 0i` that the rule of
`Op.complex f` gives in binary32 at the pairs of words `ws`. -/
def complexWords (f : ComplexFn) (ws : Array (Nat × Nat)) :
    Option (Array (Nat × Nat) × Array (Nat × Nat)) := do
  let b (w : Nat) : Binary32 := ExecFloat.Binary.ofNatBits w
  let x : Tensor Binary32 ([ws.size] ++ [2]) := Tensor.ofPairs fun c => (b ws[c.1].1, b ws[c.1].2)
  let rule ← (Op.complex f (s := [ws.size])).rule? (α := Binary32)
  let y := (rule.forward (x, ())).1
  let g := (rule.backward (x, ()) (Tensor.ofPairs (fun _ => (b 0x3f800000, b 0)), ())).1
  let words (t : Tensor Binary32 ([ws.size] ++ [2])) := (Array.finRange ws.size).map fun i =>
    let p := t.pairAt (i, ⟨⟩)
    (ExecFloat.Binary.toNatBits p.1, ExecFloat.Binary.toNatBits p.2)
  pure (words y, words g)

#guard (complexWords .exp #[(0, 0)]) == some (#[(0x3f800000, 0)], #[(0x3f800000, 0)])
#guard (complexWords .sqrt #[(0xc0800000, 0), (0xc0800000, 0x80000000)]).map (·.1) ==
  some #[(0, 0x40000000), (0, 0xc0000000)]
#guard (complexWords .log #[(0xbf800000, 0), (0xbf800000, 0x80000000)]).map (·.1) ==
  some #[(0, 0x40490fdb), (0, 0xc0490fdb)]

/-- In every scalar type with `ComplexFunctions`, `Op.complex f` has a rule, and its forward applies
`ComplexFunctions.apply f` to the pair of parts at every coordinate. -/
theorem complex_forward_pairAt {α : Type} [Storage α] [Context α] [ComplexFunctions α]
    (f : ComplexFn) {s : Shape} (x : Tensor α (s ++ [2])) (c : s.Coord) :
    ∃ rule, (Op.complex f (s := s)).rule? (α := α) = some rule ∧
      (rule.forward (x, ())).1.pairAt c = ComplexFunctions.apply f (x.pairAt c) :=
  ⟨_, Op.rule?_complex f, Tensor.pairAt_ofPairs _ _⟩

#guard ((Op.complex .exp (s := [1])).rule? (α := Float)).isNone
#guard (Op.complex .exp (s := [1])).need == .complex
#guard Op.encode ⟨_, _, Op.complex .acos (s := [2, 3])⟩ ==
  ⟨Op.Wire.formatVersion, "complex", [.name "acos", .shape [2, 3]]⟩

/-- The words of the parts of the result and of the pullbacks of the seed `1 + 0i` to the base and
to the exponent that the rule of `Op.complexPow` gives in binary32 at the pairs of words of the base
and the exponent in `ps`. -/
def powWords (ps : Array ((Nat × Nat) × (Nat × Nat))) :
    Option (Array (Nat × Nat) × Array (Nat × Nat) × Array (Nat × Nat)) := do
  let b (w : Nat) : Binary32 := ExecFloat.Binary.ofNatBits w
  let x : Tensor Binary32 ([ps.size] ++ [2]) :=
    Tensor.ofPairs fun c => (b ps[c.1].1.1, b ps[c.1].1.2)
  let y : Tensor Binary32 ([ps.size] ++ [2]) :=
    Tensor.ofPairs fun c => (b ps[c.1].2.1, b ps[c.1].2.2)
  let rule ← (Op.complexPow (s := [ps.size])).rule? (α := Binary32)
  let r := (rule.forward (x, y, ())).1
  let g := rule.backward (x, y, ()) (Tensor.ofPairs (fun _ => (b 0x3f800000, b 0)), ())
  let words (t : Tensor Binary32 ([ps.size] ++ [2])) := (Array.finRange ps.size).map fun i =>
    let p := t.pairAt (i, ⟨⟩)
    (ExecFloat.Binary.toNatBits p.1, ExecFloat.Binary.toNatBits p.2)
  pure (words r, words g.1, words g.2.1)

#guard (powWords #[((0xc0800000, 0), (0x3f000000, 0)), ((0xc0800000, 0x80000000), (0x3f000000, 0)),
    ((0x3f800000, 0x3f800000), (0x40000000, 0)),
    ((0x3f800000, 0x40000000), (0x40400000, 0xbf800000))]).map (·.1) ==
  some #[(0, 0x40000000), (0, 0xc0000000), (0, 0x40000000), (0xc1db7db6, 0x419e5082)]
#guard (powWords #[((0x7fc00000, 0), (0, 0)), ((0, 0), (0xbf800000, 0)),
    ((0, 0), (0x40000000, 0x3f800000))]).map (·.1) ==
  some #[(0x3f800000, 0), (0x7f800000, 0x7fc00000), (0, 0)]
#guard powWords #[((0x40000000, 0), (0x3f800000, 0))] ==
  some (#[(0x40000000, 0)], #[(0x3f800000, 0)], #[(0x3fb17218, 0)])

/-- In every scalar type with `ComplexFunctions`, `Op.complexPow` has a rule, and its forward
applies `ComplexFunctions.pow` to the pairs of the base and the exponent at every coordinate. -/
theorem complexPow_forward_pairAt {α : Type} [Storage α] [Context α] [ComplexFunctions α]
    {s : Shape} (x y : Tensor α (s ++ [2])) (c : s.Coord) :
    ∃ rule, (Op.complexPow (s := s)).rule? (α := α) = some rule ∧
      (rule.forward (x, y, ())).1.pairAt c = ComplexFunctions.pow (x.pairAt c) (y.pairAt c) :=
  ⟨_, Op.rule?_complexPow, Tensor.pairAt_ofPairs _ _⟩

#guard ((Op.complexPow (s := [1])).rule? (α := Float)).isNone
#guard (Op.complexPow (s := [1])).need == .complex
#guard Op.encode ⟨_, _, Op.complexPow (s := [2, 3])⟩ ==
  ⟨Op.Wire.formatVersion, "complex_pow", [.shape [2, 3]]⟩

section Round

open FloatLib.Floats.Formats.BinaryInterchange

/-- The binary32 pair of parts of `1 + 0i`. -/
def one32 : Binary32 × Binary32 :=
  (ExecFloat.Binary.ofNatBits 0x3f800000, ExecFloat.Binary.ofNatBits 0)

/-- The binary32 pair of parts of `1 - 0i`. -/
def oneNeg32 : Binary32 × Binary32 :=
  (ExecFloat.Binary.ofNatBits 0x3f800000, ExecFloat.Binary.ofNatBits 0x80000000)

/-- The binary32 pair of parts of `2 + 0i`. -/
def two32 : Binary32 × Binary32 :=
  (ExecFloat.Binary.ofNatBits 0x40000000, ExecFloat.Binary.ofNatBits 0)

/-- The binary32 pair of parts of `2 - 0i`. -/
def twoNeg32 : Binary32 × Binary32 :=
  (ExecFloat.Binary.ofNatBits 0x40000000, ExecFloat.Binary.ofNatBits 0x80000000)

/-- The binary32 pair of parts of `-1 + 0i`. -/
def minusOne32 : Binary32 × Binary32 :=
  (ExecFloat.Binary.ofNatBits 0xbf800000, ExecFloat.Binary.ofNatBits 0)

/-- The binary32 pair of parts of `-1 - 0i`. -/
def minusOneNeg32 : Binary32 × Binary32 :=
  (ExecFloat.Binary.ofNatBits 0xbf800000, ExecFloat.Binary.ofNatBits 0x80000000)

/-- A complex number whose parts decode to the rationals `a` and `b` is `a + b i`. -/
theorem toComplex_of_toRat? {format : FloatFormat} {z : ExecComplex format} {a b : ℚ} {w : ℂ}
    (ha : Model.toRat? z.re = some a) (hb : Model.toRat? z.im = some b)
    (hw : (⟨a, b⟩ : ℂ) = w := by apply Complex.ext <;> simp) :
    ExecComplex.toComplex z = w := by
  rw [← hw]
  simp [ExecComplex.toComplex, Model.Algebraic.toReal_of_toRat?_eq_some ha,
    Model.Algebraic.toReal_of_toRat?_eq_some hb]

/-- The complex number of `one32` is `1`. -/
theorem toComplex_one32 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex one32) = 1 :=
  toComplex_of_toRat? (a := 1) (b := 0) (by float_decide [one32]) (by float_decide [one32])

/-- The complex number of `oneNeg32` is `1`. -/
theorem toComplex_oneNeg32 :
    ExecComplex.toComplex (ConfiguredBinary.toExecComplex oneNeg32) = 1 :=
  toComplex_of_toRat? (a := 1) (b := 0) (by float_decide [oneNeg32]) (by float_decide [oneNeg32])

/-- The complex number of `two32` is `2`. -/
theorem toComplex_two32 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex two32) = 2 :=
  toComplex_of_toRat? (a := 2) (b := 0) (by float_decide [two32]) (by float_decide [two32])

/-- The complex number of `twoNeg32` is `2`. -/
theorem toComplex_twoNeg32 :
    ExecComplex.toComplex (ConfiguredBinary.toExecComplex twoNeg32) = 2 :=
  toComplex_of_toRat? (a := 2) (b := 0) (by float_decide [twoNeg32]) (by float_decide [twoNeg32])

/-- The complex number of `minusOne32` is `-1`. -/
theorem toComplex_minusOne32 :
    ExecComplex.toComplex (ConfiguredBinary.toExecComplex minusOne32) = -1 :=
  toComplex_of_toRat? (a := -1) (b := 0) (by float_decide [minusOne32])
    (by float_decide [minusOne32])

/-- The complex number of `minusOneNeg32` is `-1`. -/
theorem toComplex_minusOneNeg32 :
    ExecComplex.toComplex (ConfiguredBinary.toExecComplex minusOneNeg32) = -1 :=
  toComplex_of_toRat? (a := -1) (b := 0) (by float_decide [minusOneNeg32])
    (by float_decide [minusOneNeg32])

/-- In binary32, at every input of shape `[3]` and each entry `z` with finite parts, each part of
the result of `Op.complex .exp` is the part of `exp z` rounded once to nearest even. -/
theorem complex_exp_round (x : Tensor Binary32 ([3] ++ [2])) (c : Shape.Coord [3])
    (hz : ExecComplex.isFinite (ConfiguredBinary.toExecComplex (x.pairAt c)) = true) :
    ∃ rule, (Op.complex .exp (s := [3])).rule? (α := Binary32) = some rule ∧
      ∃ e : ExecComplex.Exact,
        e.value = Complex.exp (ExecComplex.toComplex (ConfiguredBinary.toExecComplex
          (x.pairAt c))) ∧
        ConfiguredBinary.toExecComplex ((rule.forward (x, ())).1.pairAt c) =
          e.round _ .nearest :=
  Op.complex_forward_value rfl .exp x c hz trivial trivial

/-- In binary32, at the input of shape `[3]` whose entries are all `1 + 0i`, each part of each
entry of the result of `Op.complex .log` is the part of `log 1 = 0` rounded once to nearest even:
the entries meet `complexRegular` and `complexPrincipal`. -/
theorem complex_log_round (c : Shape.Coord [3]) :
    let x : Tensor Binary32 ([3] ++ [2]) := Tensor.ofPairs fun _ => one32
    ∃ rule, (Op.complex .log (s := [3])).rule? (α := Binary32) = some rule ∧
      ∃ e : ExecComplex.Exact, e.value = 0 ∧
        ConfiguredBinary.toExecComplex ((rule.forward (x, ())).1.pairAt c) =
          e.round _ .nearest := by
  intro x
  have hx : x.pairAt c = one32 := Tensor.pairAt_ofPairs _ _
  have h1 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex (x.pairAt c)) = 1 := by
    rw [hx, toComplex_one32]
  obtain ⟨rule, hrule, e, he, hround⟩ := Op.complex_forward_value rfl .log x c
    (by rw [hx]; decide) (by show ExecComplex.toComplex _ ≠ 0; rw [h1]; exact one_ne_zero)
    (Or.inl (by rw [hx]; decide))
  exact ⟨rule, hrule, e, by rw [he, h1]; exact Complex.log_one, hround⟩

/-- In binary32, at the input of shape `[3]` whose entries are all `1 + 0i`, each part of each
entry of the result of `Op.complex .acos` is the part of `acos 1 = π / 2 - asin 1 = 0` rounded once
to nearest even: the entries are off `{x : 1 < |x|}`. -/
theorem complex_acos_round (c : Shape.Coord [3]) :
    let x : Tensor Binary32 ([3] ++ [2]) := Tensor.ofPairs fun _ => one32
    ∃ rule, (Op.complex .acos (s := [3])).rule? (α := Binary32) = some rule ∧
      ∃ e : ExecComplex.Exact, e.value = 0 ∧
        ConfiguredBinary.toExecComplex ((rule.forward (x, ())).1.pairAt c) =
          e.round _ .nearest := by
  intro x
  have hx : x.pairAt c = one32 := Tensor.pairAt_ofPairs _ _
  have h1 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex (x.pairAt c)) = 1 := by
    rw [hx, toComplex_one32]
  obtain ⟨rule, hrule, e, he, hround⟩ := Op.complex_forward_value rfl .acos x c
    (by rw [hx]; decide) trivial (Or.inr (by rw [h1]; simp))
  refine ⟨rule, hrule, e, ?_, hround⟩
  rw [he, h1]
  show (Real.pi : ℂ) / 2 + _root_.Complex.I *
    Complex.log (_root_.Complex.I * 1 + (1 - 1 ^ 2) ^ (2⁻¹ : ℂ)) = 0
  rw [one_pow, sub_self, Complex.zero_cpow (by norm_num), mul_one, add_zero, _root_.Complex.log_I]
  linear_combination ((Real.pi : ℂ) / 2) * _root_.Complex.I_sq

/-- In binary32, at the input of shape `[3]` whose entries are all `2 - 0i`, each part of each
entry of the result of `Op.complex .acos` is the part of `acos 2 = i log (2 + √3)`, the limit from
below the cut, rounded once to nearest even: the sign bits differ. -/
theorem complex_acos_cut_round (c : Shape.Coord [3]) :
    let x : Tensor Binary32 ([3] ++ [2]) := Tensor.ofPairs fun _ => twoNeg32
    ∃ rule, (Op.complex .acos (s := [3])).rule? (α := Binary32) = some rule ∧
      ∃ e : ExecComplex.Exact, e.value = _root_.Complex.I * (Real.log (2 + Real.sqrt 3) : ℝ) ∧
        ConfiguredBinary.toExecComplex ((rule.forward (x, ())).1.pairAt c) =
          e.round _ .nearest := by
  intro x
  have hx : x.pairAt c = twoNeg32 := Tensor.pairAt_ofPairs _ _
  have h2 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex (x.pairAt c)) = 2 := by
    rw [hx, toComplex_twoNeg32]
  obtain ⟨rule, hrule, e, he, hround⟩ := Op.complex_forward_value rfl .acos x c
    (by rw [hx]; decide) trivial (Or.inl (by rw [hx]; decide))
  refine ⟨rule, hrule, e, ?_, hround⟩
  have h3 : (1 : ℂ) - 2 ^ 2 = (_root_.Complex.I * (Real.sqrt 3 : ℝ)) ^ 2 := by
    rw [mul_pow, _root_.Complex.I_sq, ← Complex.ofReal_pow, Real.sq_sqrt (by norm_num)]
    norm_num
  have hsum : _root_.Complex.I * 2 + _root_.Complex.I * (Real.sqrt 3 : ℝ) =
      ((2 + Real.sqrt 3 : ℝ) : ℂ) * _root_.Complex.I := by
    push_cast
    ring
  rw [he, h2]
  show (Real.pi : ℂ) / 2 + _root_.Complex.I *
    Complex.log (_root_.Complex.I * 2 + (1 - 2 ^ 2) ^ (2⁻¹ : ℂ)) = _
  rw [h3, ExecComplex.sq_cpow_inv_two (by simp) (Or.inr (by simp [Real.sqrt_nonneg])), hsum,
    Complex.log_ofReal_mul (by positivity) _root_.Complex.I_ne_zero, _root_.Complex.log_I]
  linear_combination ((Real.pi : ℂ) / 2) * _root_.Complex.I_sq

/-- In binary32, at the input of shape `[3]` whose entries are all `-1 + 0i`, each part of each
entry of the result of `Op.complex .asinh` is the part of `asinh (-1) = log (√2 - 1)` rounded once
to nearest even: the entries are off `{i y : 1 < |y|}`, where FloatLib's `asinh` at mixed sign
bits is the principal expression. -/
theorem complex_asinh_round (c : Shape.Coord [3]) :
    let x : Tensor Binary32 ([3] ++ [2]) := Tensor.ofPairs fun _ => minusOne32
    ∃ rule, (Op.complex .asinh (s := [3])).rule? (α := Binary32) = some rule ∧
      ∃ e : ExecComplex.Exact, e.value = (Real.log (Real.sqrt 2 - 1) : ℝ) ∧
        ConfiguredBinary.toExecComplex ((rule.forward (x, ())).1.pairAt c) =
          e.round _ .nearest := by
  intro x
  have hx : x.pairAt c = minusOne32 := Tensor.pairAt_ofPairs _ _
  have h1 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex (x.pairAt c)) = -1 := by
    rw [hx, toComplex_minusOne32]
  obtain ⟨rule, hrule, e, he, hround⟩ := Op.complex_forward_value rfl .asinh x c
    (by rw [hx]; decide) trivial (Or.inr (by rw [h1]; simp))
  refine ⟨rule, hrule, e, ?_, hround⟩
  have h2 : (1 : ℂ) + (-1) ^ 2 = (Real.sqrt 2 : ℂ) ^ 2 := by
    rw [← Complex.ofReal_pow, Real.sq_sqrt (by norm_num)]
    norm_num
  have hpos : 0 ≤ Real.sqrt 2 - 1 := by
    nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 2 by norm_num), Real.sqrt_nonneg 2]
  rw [he, h1]
  show Complex.log (-1 + (1 + (-1) ^ 2) ^ (2⁻¹ : ℂ)) = _
  rw [h2, ExecComplex.sq_cpow_inv_two (by simp [Real.sqrt_nonneg]) (Or.inr (by simp)),
    Complex.ofReal_log hpos, Complex.ofReal_sub, Complex.ofReal_one, neg_add_eq_sub]

/-- In binary32, at the input of shape `[3]` whose entries are all `-1 - 0i`, each part of each
entry of the result of `Op.complex .asin` is the part of `asin (-1) = -π / 2` rounded once to
nearest even: the entries are off `{x : 1 < |x|}`, where FloatLib's `asin` with both sign bits
set is the principal expression. -/
theorem complex_asin_round (c : Shape.Coord [3]) :
    let x : Tensor Binary32 ([3] ++ [2]) := Tensor.ofPairs fun _ => minusOneNeg32
    ∃ rule, (Op.complex .asin (s := [3])).rule? (α := Binary32) = some rule ∧
      ∃ e : ExecComplex.Exact, e.value = -((Real.pi : ℂ) / 2) ∧
        ConfiguredBinary.toExecComplex ((rule.forward (x, ())).1.pairAt c) =
          e.round _ .nearest := by
  intro x
  have hx : x.pairAt c = minusOneNeg32 := Tensor.pairAt_ofPairs _ _
  have h1 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex (x.pairAt c)) = -1 := by
    rw [hx, toComplex_minusOneNeg32]
  obtain ⟨rule, hrule, e, he, hround⟩ := Op.complex_forward_value rfl .asin x c
    (by rw [hx]; decide) trivial (Or.inr (by rw [h1]; simp))
  refine ⟨rule, hrule, e, ?_, hround⟩
  rw [he, h1]
  show -_root_.Complex.I *
    Complex.log (_root_.Complex.I * -1 + (1 - (-1) ^ 2) ^ (2⁻¹ : ℂ)) = _
  rw [neg_one_sq, sub_self, Complex.zero_cpow (by norm_num), add_zero, mul_neg_one,
    _root_.Complex.log_neg_I]
  linear_combination ((Real.pi : ℂ) / 2) * _root_.Complex.I_sq

/-- In binary32, at the input of shape `[3]` whose entries are all `1 - 0i`, each part of each
entry of the result of `Op.complex .acosh` is the part of `acosh 1 = 0` rounded once to nearest
even: the imaginary sign bit is set and the entries are off `(-∞, 1)`. -/
theorem complex_acosh_round (c : Shape.Coord [3]) :
    let x : Tensor Binary32 ([3] ++ [2]) := Tensor.ofPairs fun _ => oneNeg32
    ∃ rule, (Op.complex .acosh (s := [3])).rule? (α := Binary32) = some rule ∧
      ∃ e : ExecComplex.Exact, e.value = 0 ∧
        ConfiguredBinary.toExecComplex ((rule.forward (x, ())).1.pairAt c) =
          e.round _ .nearest := by
  intro x
  have hx : x.pairAt c = oneNeg32 := Tensor.pairAt_ofPairs _ _
  have h1 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex (x.pairAt c)) = 1 := by
    rw [hx, toComplex_oneNeg32]
  obtain ⟨rule, hrule, e, he, hround⟩ := Op.complex_forward_value rfl .acosh x c
    (by rw [hx]; decide) trivial (Or.inr (by rw [h1]; simp))
  refine ⟨rule, hrule, e, ?_, hround⟩
  rw [he, h1]
  show 2 * Complex.log (((1 + 1) / 2) ^ (2⁻¹ : ℂ) + ((1 - 1) / 2) ^ (2⁻¹ : ℂ)) = 0
  rw [show ((1 : ℂ) + 1) / 2 = 1 by norm_num, Complex.one_cpow, sub_self, zero_div,
    Complex.zero_cpow (by norm_num), add_zero, Complex.log_one, mul_zero]

/-- In binary32, at the base of shape `[3]` whose entries are all `2 + 0i` and the exponent whose
entries are all `1 + 0i`, each part of each entry of the result of `Op.complexPow` is the part of
`2 ^ 1 = 2` rounded once to nearest even: FloatLib certifies the power there. -/
theorem complex_pow_round (c : Shape.Coord [3]) :
    let x : Tensor Binary32 ([3] ++ [2]) := Tensor.ofPairs fun _ => two32
    let y : Tensor Binary32 ([3] ++ [2]) := Tensor.ofPairs fun _ => one32
    ∃ rule, (Op.complexPow (s := [3])).rule? (α := Binary32) = some rule ∧
      ∃ e : ExecComplex.Exact, e.value = 2 ∧
        ConfiguredBinary.toExecComplex ((rule.forward (x, y, ())).1.pairAt c) =
          e.round _ .nearest := by
  intro x y
  have hx : x.pairAt c = two32 := Tensor.pairAt_ofPairs _ _
  have hy : y.pairAt c = one32 := Tensor.pairAt_ofPairs _ _
  have h2 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex (x.pairAt c)) = 2 := by
    rw [hx, toComplex_two32]
  have h1 : ExecComplex.toComplex (ConfiguredBinary.toExecComplex (y.pairAt c)) = 1 := by
    rw [hy, toComplex_one32]
  obtain ⟨rule, hrule, e, he, hround⟩ := Op.complexPow_forward_value rfl x y c
    (by rw [hx]; decide) (by rw [hy]; decide) (by rw [h2]; exact two_ne_zero)
    (by rw [hx, hy]; float_decide [one32, two32]) (Or.inl (by rw [hx]; decide))
  exact ⟨rule, hrule, e, by rw [he, h1, h2, Complex.cpow_one], hround⟩

end Round

/-- Over the reals the pullback of `Op.complex .exp` at the shape `[3]` is the adjoint of its
derivative at every input. -/
theorem complex_exp_pullback (x δ : Tensor ℝ ([3] ++ [2])) :
    (((Op.complex .exp (s := [3])).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun y => (((Op.complex .exp (s := [3])).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x).adjoint δ :=
  (Op.complex_domain_backward_eq_adjoint .exp x δ fun _ => Set.mem_univ _).2

/-- Over the reals the pullback of `Op.complex .log` at the shape `[3]` is the adjoint of its
derivative at the input whose entries are all `1 + 0i`. -/
theorem complex_log_pullback (δ : Tensor ℝ ([3] ++ [2])) :
    let x : Tensor ℝ ([3] ++ [2]) := Tensor.ofPairs fun _ => (1, 0)
    (((Op.complex .log (s := [3])).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun y => (((Op.complex .log (s := [3])).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x).adjoint δ := by
  intro x
  exact (Op.complex_domain_backward_eq_adjoint .log x δ fun c => by
    simp [x, Op.pairValue, ComplexFn.domain, Complex.mem_slitPlane_iff]).2

/-- Over the reals the pullback of `Op.complex .tan` at the shape `[3]` is the adjoint of its
derivative at the input whose entries are all `0`, where `cos` is `1`. -/
theorem complex_tan_pullback (δ : Tensor ℝ ([3] ++ [2])) :
    let x : Tensor ℝ ([3] ++ [2]) := Tensor.ofPairs fun _ => (0, 0)
    (((Op.complex .tan (s := [3])).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun y => (((Op.complex .tan (s := [3])).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x).adjoint δ := by
  intro x
  exact (Op.complex_domain_backward_eq_adjoint .tan x δ fun c => by
    simp [x, Op.pairValue, ComplexFn.domain, Complex.mk_eq_add_mul_I]).2

/-- Over the reals the pullback of `Op.complex .atanh` at the shape `[3]` is the adjoint of its
derivative at the input whose entries are all `1 / 2`, between the cuts `(-∞, -1]` and `[1, ∞)`. -/
theorem complex_atanh_pullback (δ : Tensor ℝ ([3] ++ [2])) :
    let x : Tensor ℝ ([3] ++ [2]) := Tensor.ofPairs fun _ => (1 / 2, 0)
    (((Op.complex .atanh (s := [3])).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun y => (((Op.complex .atanh (s := [3])).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x).adjoint δ := by
  intro x
  exact (Op.complex_domain_backward_eq_adjoint .atanh x δ fun c =>
    ComplexFn.one_sub_sq_mem_slitPlane.2 (Or.inr (by rw [abs_lt]; norm_num [x, Op.pairValue]))).2

/-- Over the reals the pullback of `Op.complex .asinh` at the shape `[3]` is the adjoint of its
derivative at the input whose entries are all `1 + 2i`, off the cut on the imaginary axis. -/
theorem complex_asinh_pullback (δ : Tensor ℝ ([3] ++ [2])) :
    let x : Tensor ℝ ([3] ++ [2]) := Tensor.ofPairs fun _ => (1, 2)
    (((Op.complex .asinh (s := [3])).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun y => (((Op.complex .asinh (s := [3])).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x).adjoint δ := by
  intro x
  exact (Op.complex_domain_backward_eq_adjoint .asinh x δ fun c =>
    ComplexFn.one_add_sq_mem_slitPlane.2 (Or.inl (by norm_num [x, Op.pairValue]))).2

/-- Over the reals the pullback of `Op.complexPow` at the shape `[3]` is the adjoint of its
derivative in the base and the exponent at every input whose base entries are all `2`. -/
theorem complex_pow_pullback (y δ : Tensor ℝ ([3] ++ [2])) :
    let x : Tensor ℝ ([3] ++ [2]) := Tensor.ofPairs fun _ => (2, 0)
    WithLp.toLp 2
        ((((Op.complexPow (s := [3])).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
        (((Op.complexPow (s := [3])).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ ([3] ++ [2]) × Tensor ℝ ([3] ++ [2])) =>
        (((Op.complexPow (s := [3])).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (x, y))).adjoint δ := by
  intro x
  exact (Op.complexPow_backward_eq_adjoint x y δ fun c => by
    simp [x, Op.pairValue, Complex.mem_slitPlane_iff]).2

/-- Over the reals the pullback of `Op.complexPow` at the shape `[3]` to the exponent is the
adjoint of its derivative in the exponent at every input whose base entries are all `-1`, on the
cut of the base. -/
theorem complex_pow_pullback_right (y δ : Tensor ℝ ([3] ++ [2])) :
    let x : Tensor ℝ ([3] ++ [2]) := Tensor.ofPairs fun _ => (-1, 0)
    (((Op.complexPow (s := [3])).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1 =
      (fderiv ℝ (fun y' => (((Op.complexPow (s := [3])).rule? (α := ℝ) |>.get rfl).forward
        (x, y', ())).1) y).adjoint δ := by
  intro x
  exact (Op.complexPow_backward_eq_adjoint_right x y δ fun c => Or.inl (by
    simp [x, Op.pairValue, Complex.ext_iff])).2

end TorchLean.Test.Complex
