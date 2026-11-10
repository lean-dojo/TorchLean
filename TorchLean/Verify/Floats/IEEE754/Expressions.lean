/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Floats.IEEE754

/-!
# Scalar expression refinement

`Expr.toReal_eval` composes FloatLib's arithmetic refinement results along an
`Expr` tree. `Expr.eval` evaluates with the encoded operations of a configured binary format,
such as binary32 `ExecFloat.Binary 8 23`, and `Expr.evalReal` rounds each real operation with
`Model.roundAt` in the same format. Fused multiply-add rounds the exact `a * b + c` once.

The theorem needs a `FiniteEval` witness: every intermediate result is finite, the leaves
included, and every divisor is nonzero. Subnormals and underflow to zero are allowed. The equality
is between decoded reals, so it identifies `+0` with `-0`, and native CPU or GPU kernels are
outside it.

`abs_evalReal_sub_evalExact_le` bounds the distance from `Expr.evalReal` to the exact real
evaluation `evalExact` by `errorBound`, computed along the tree from the exact values with the unit
roundoff `Model.unitRoundoffAt` and the underflow allowance `Model.underflowErrorAt`. It holds in
every format when every divisor exceeds its error bound (`DivisorsExceedError`), and
`abs_toReal_eval_sub_evalExact_le` gives the same bound for the runtime evaluation in an
IEEE format under a `FiniteEval` witness and the same divisor condition.
`errorBound_sqrt_le_sqrt` and `errorBound_sqrt_le_div` bound the square-root term of
`errorBound` by `Real.sqrt e` and, for a positive exact operand `t` and a nonnegative `e`, by
`e / Real.sqrt t`, where `e` is the operand's bound.

We wrote `errorBound` by hand, one rounding per operation in Higham's standard model plus an
absolute term for gradual underflow. FloatLib's `Model.mixedError_roundAt` puts a rounded `t` within
`u * |t| + η` of `t`, which is `2^-24 * |t| + 2^-150` in binary32. The bound recurses on the exact
intermediate values and the operands' bounds, with no closed-form `γₙ` constant.

References for the rounding model:
- IEEE 754-2019: https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991): https://doi.org/10.1145/103162.103163
- Flocq (Boldo and Melquiond, 2011): https://doi.org/10.1109/ARITH.2011.40
- Higham (2002), Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM:
  https://doi.org/10.1137/1.9780898718027
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean.IEEE754

noncomputable section

open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-!
## A small scalar expression language

`Expr` is a small scalar AST for straight-line floating-point computations. It exists to state
refinement theorems for whole expressions and is separate from TorchLean's graph IRs.
-/

/-- A compact AST for scalar expressions with constants of type `α`, such as
`ExecFloat.Binary 8 23`. -/
inductive Expr (α : Type) where
  /-- The variable with the given index. -/
  | var : Nat → Expr α
  /-- A constant. -/
  | const : α → Expr α
  /-- The sum of two expressions. -/
  | add : Expr α → Expr α → Expr α
  /-- The difference of two expressions. -/
  | sub : Expr α → Expr α → Expr α
  /-- The product of two expressions. -/
  | mul : Expr α → Expr α → Expr α
  /-- The quotient of two expressions. -/
  | div : Expr α → Expr α → Expr α
  /-- The fused multiply-add of three expressions: `fma a b c` is `a * b + c`. -/
  | fma : Expr α → Expr α → Expr α → Expr α
  /-- The square root of an expression. -/
  | sqrt : Expr α → Expr α
  deriving Repr

/-- Real semantics for the compact scalar expression language, rounding in the constants' format. -/
def Expr.evalReal (env : Nat → ℝ) : Expr Value → ℝ
  | .var i => env i
  | .const x => (toModel x).toReal
  | .add a b => Model.roundAt format (evalReal env a + evalReal env b)
  | .sub a b => Model.roundAt format (evalReal env a - evalReal env b)
  | .mul a b => Model.roundAt format (evalReal env a * evalReal env b)
  | .div a b => Model.roundAt format (evalReal env a / evalReal env b)
  | .fma a b c => Model.roundAt format
    (evalReal env a * evalReal env b + evalReal env c)
  | .sqrt a => Model.roundAt format (Real.sqrt (evalReal env a))

/-!
## Forward error against the exact real evaluation

`errorBound` adds at each operation the error `p` that its operands' errors cause before rounding
and the rounding error `u * (|t| + p) + η` around the exact result `t`.
-/

/-- Real semantics for the compact scalar expression language without rounding: each operation is
the exact real operation, and the constants are their real values. -/
def evalExact (env : Nat → ℝ) : Expr Value → ℝ
  | .var i => env i
  | .const x => (toModel x).toReal
  | .add a b => evalExact env a + evalExact env b
  | .sub a b => evalExact env a - evalExact env b
  | .mul a b => evalExact env a * evalExact env b
  | .div a b => evalExact env a / evalExact env b
  | .fma a b c => evalExact env a * evalExact env b + evalExact env c
  | .sqrt a => Real.sqrt (evalExact env a)

/-- The forward error bound of `Expr.evalReal` against `evalExact`: an operation with exact result
`t`, whose operands' errors move the unrounded result by at most `p`, gets
`u * |t| + ((1 + u) * p + η)` with `u = Model.unitRoundoffAt format` and
`η = Model.underflowErrorAt format`. At a division the bound holds when the divisor's bound is below
its magnitude (`DivisorsExceedError`). -/
def errorBound (env : Nat → ℝ) : Expr Value → ℝ
  | .var _ => 0
  | .const _ => 0
  | .add a b => Model.unitRoundoffAt format * |evalExact env a + evalExact env b| +
      ((1 + Model.unitRoundoffAt format) * (errorBound env a + errorBound env b) +
        Model.underflowErrorAt format)
  | .sub a b => Model.unitRoundoffAt format * |evalExact env a - evalExact env b| +
      ((1 + Model.unitRoundoffAt format) * (errorBound env a + errorBound env b) +
        Model.underflowErrorAt format)
  | .mul a b => Model.unitRoundoffAt format * |evalExact env a * evalExact env b| +
      ((1 + Model.unitRoundoffAt format) *
          (|evalExact env a| * errorBound env b + |evalExact env b| * errorBound env a +
            errorBound env a * errorBound env b) +
        Model.underflowErrorAt format)
  | .div a b => Model.unitRoundoffAt format * |evalExact env a / evalExact env b| +
      ((1 + Model.unitRoundoffAt format) *
          ((errorBound env a + |evalExact env a / evalExact env b| * errorBound env b) /
            (|evalExact env b| - errorBound env b)) +
        Model.underflowErrorAt format)
  | .fma a b c =>
      Model.unitRoundoffAt format * |evalExact env a * evalExact env b + evalExact env c| +
        ((1 + Model.unitRoundoffAt format) *
            (|evalExact env a| * errorBound env b + |evalExact env b| * errorBound env a +
              errorBound env a * errorBound env b + errorBound env c) +
          Model.underflowErrorAt format)
  | .sqrt a => Model.unitRoundoffAt format * Real.sqrt (evalExact env a) +
      ((1 + Model.unitRoundoffAt format) *
          max (Real.sqrt (evalExact env a + errorBound env a) - Real.sqrt (evalExact env a))
            (Real.sqrt (evalExact env a) - Real.sqrt (evalExact env a - errorBound env a)) +
        Model.underflowErrorAt format)

/-- Every division in the expression has a divisor whose error bound `errorBound env` is below the
absolute value of its exact value `evalExact env`. -/
def DivisorsExceedError (env : Nat → ℝ) : Expr Value → Prop
  | .var _ => True
  | .const _ => True
  | .add a b => DivisorsExceedError env a ∧ DivisorsExceedError env b
  | .sub a b => DivisorsExceedError env a ∧ DivisorsExceedError env b
  | .mul a b => DivisorsExceedError env a ∧ DivisorsExceedError env b
  | .div a b => DivisorsExceedError env a ∧ DivisorsExceedError env b ∧
      errorBound env b < |evalExact env b|
  | .fma a b c =>
      DivisorsExceedError env a ∧ DivisorsExceedError env b ∧ DivisorsExceedError env c
  | .sqrt a => DivisorsExceedError env a

/-- A product of approximations with errors `ex` and `ey` is within
`|x₀| * ey + |y₀| * ex + ex * ey` of the exact product. -/
private theorem abs_mul_sub_mul_le {x y x₀ y₀ ex ey : ℝ} (hx : |x - x₀| ≤ ex)
    (hy : |y - y₀| ≤ ey) : |x * y - x₀ * y₀| ≤ |x₀| * ey + |y₀| * ex + ex * ey := by
  rw [show x * y - x₀ * y₀ = x₀ * (y - y₀) + y₀ * (x - x₀) + (x - x₀) * (y - y₀) by ring]
  refine (abs_add_three _ _ _).trans ?_
  rw [abs_mul, abs_mul, abs_mul]
  exact add_le_add (add_le_add (mul_le_mul_of_nonneg_left hy (abs_nonneg _))
    (mul_le_mul_of_nonneg_left hx (abs_nonneg _)))
    (mul_le_mul hx hy (abs_nonneg _) ((abs_nonneg _).trans hx))

/-- `Expr.evalReal env e` is within `errorBound env e` of `evalExact env e` when every divisor of
`e` exceeds its error bound, in every format. -/
theorem abs_evalReal_sub_evalExact_le (env : Nat → ℝ) {e : Expr Value}
    (h : DivisorsExceedError env e) :
    |Expr.evalReal env e - evalExact env e| ≤ errorBound env e := by
  induction e with
  | var i => simp [Expr.evalReal, evalExact, errorBound]
  | const x => simp [Expr.evalReal, evalExact, errorBound]
  | add a b iha ihb =>
      obtain ⟨ha, hb⟩ := h
      have hp : |(Expr.evalReal env a + Expr.evalReal env b) - (evalExact env a +
        evalExact env b)| ≤
          errorBound env a + errorBound env b := by
        rw [show (Expr.evalReal env a + Expr.evalReal env b) - (evalExact env a + evalExact env b) =
          (Expr.evalReal env a - evalExact env a) + (Expr.evalReal env b - evalExact env b) by ring]
        exact (abs_add_le _ _).trans (add_le_add (iha ha) (ihb hb))
      exact ((Model.mixedError_roundAt format _).trans_of_abs_sub_le hp).bound
  | sub a b iha ihb =>
      obtain ⟨ha, hb⟩ := h
      have hp : |(Expr.evalReal env a - Expr.evalReal env b) - (evalExact env
        a - evalExact env b)| ≤
          errorBound env a + errorBound env b := by
        rw [show (Expr.evalReal env a - Expr.evalReal env b) - (evalExact env a - evalExact env b) =
          (Expr.evalReal env a - evalExact env a) - (Expr.evalReal env b - evalExact env b) by ring]
        exact (abs_sub _ _).trans (add_le_add (iha ha) (ihb hb))
      exact ((Model.mixedError_roundAt format _).trans_of_abs_sub_le hp).bound
  | mul a b iha ihb =>
      obtain ⟨ha, hb⟩ := h
      exact ((Model.mixedError_roundAt format _).trans_of_abs_sub_le
        (abs_mul_sub_mul_le (iha ha) (ihb hb))).bound
  | div a b iha ihb =>
      obtain ⟨ha, hb, hlt⟩ := h
      have h1 := iha ha
      have h2 := ihb hb
      have hpos : 0 < |evalExact env b| - errorBound env b := sub_pos.2 hlt
      have hlow : |evalExact env b| - errorBound env b ≤ |Expr.evalReal env b| := by
        have := abs_sub_abs_le_abs_sub (evalExact env b) (Expr.evalReal env b)
        rw [abs_sub_comm] at this
        linarith
      have hb' : Expr.evalReal env b ≠ 0 := abs_pos.1 (hpos.trans_le hlow)
      have hb₀ : evalExact env b ≠ 0 := by
        intro h0
        rw [h0, abs_zero] at hlt
        linarith [(abs_nonneg _).trans h2]
      have hp : |Expr.evalReal env a / Expr.evalReal env b - evalExact env a / evalExact env b| ≤
          (errorBound env a + |evalExact env a / evalExact env b| * errorBound env b) /
            (|evalExact env b| - errorBound env b) := by
        rw [show Expr.evalReal env a / Expr.evalReal env b - evalExact env a / evalExact env b =
            ((Expr.evalReal env a - evalExact env a) -
              evalExact env a / evalExact env b * (Expr.evalReal env b - evalExact env b)) /
                Expr.evalReal env b by field_simp; ring, abs_div]
        have hnum : |(Expr.evalReal env a - evalExact env a) -
            evalExact env a / evalExact env b * (Expr.evalReal env b - evalExact env b)| ≤
              errorBound env a + |evalExact env a / evalExact env b| * errorBound env b := by
          refine (abs_sub _ _).trans (add_le_add h1 ?_)
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_left h2 (abs_nonneg _)
        calc _ ≤ (errorBound env a + |evalExact env a / evalExact env b| * errorBound env b) /
              |Expr.evalReal env b| := div_le_div_of_nonneg_right hnum (abs_nonneg _)
          _ ≤ _ := div_le_div_of_nonneg_left ((abs_nonneg _).trans hnum) hpos hlow
      exact ((Model.mixedError_roundAt format _).trans_of_abs_sub_le hp).bound
  | fma a b c iha ihb ihc =>
      obtain ⟨ha, hb, hc⟩ := h
      have hp : |(Expr.evalReal env a * Expr.evalReal env b + Expr.evalReal env c) -
          (evalExact env a * evalExact env b + evalExact env c)| ≤
            |evalExact env a| * errorBound env b + |evalExact env b| * errorBound env a +
              errorBound env a * errorBound env b + errorBound env c := by
        rw [show (Expr.evalReal env a * Expr.evalReal env b + Expr.evalReal env c) -
            (evalExact env a * evalExact env b + evalExact env c) =
              (Expr.evalReal env a * Expr.evalReal env b - evalExact env a * evalExact env b) +
                (Expr.evalReal env c - evalExact env c) by ring]
        exact (abs_add_le _ _).trans
          (add_le_add (abs_mul_sub_mul_le (iha ha) (ihb hb)) (ihc hc))
      exact ((Model.mixedError_roundAt format _).trans_of_abs_sub_le hp).bound
  | sqrt a iha =>
      have h1 := abs_le.1 (iha h)
      have hp : |Real.sqrt (Expr.evalReal env a) - Real.sqrt (evalExact env a)| ≤
          max (Real.sqrt (evalExact env a + errorBound env a) - Real.sqrt (evalExact env a))
            (Real.sqrt (evalExact env a) - Real.sqrt (evalExact env a - errorBound env a)) := by
        refine abs_sub_le_iff.2 ⟨le_max_of_le_left ?_, le_max_of_le_right ?_⟩
        · exact sub_le_sub_right (Real.sqrt_le_sqrt (by linarith [h1.2])) _
        · exact sub_le_sub_left (Real.sqrt_le_sqrt (by linarith [h1.1])) _
      have hb := ((Model.mixedError_roundAt format _).trans_of_abs_sub_le hp).bound
      rwa [abs_of_nonneg (Real.sqrt_nonneg _)] at hb

/-- At a square root, `errorBound` is at most `u * Real.sqrt t + ((1 + u) * Real.sqrt e + η)` for
the exact operand `t = evalExact env a`, its bound `e = errorBound env a` and `u`, `η` as in
`errorBound`. -/
theorem errorBound_sqrt_le_sqrt (env : Nat → ℝ) (a : Expr Value) :
    errorBound env (.sqrt a) ≤ Model.unitRoundoffAt format * Real.sqrt (evalExact env a) +
      ((1 + Model.unitRoundoffAt format) * Real.sqrt (errorBound env a) +
        Model.underflowErrorAt format) := by
  have hu := Model.unitRoundoffAt_pos format
  have key : ∀ t e : ℝ,
      max (Real.sqrt (t + e) - Real.sqrt t) (Real.sqrt t - Real.sqrt (t - e)) ≤ Real.sqrt e := by
    intro t e
    rcases lt_or_ge e 0 with he | he
    · refine (max_le ?_ ?_).trans (Real.sqrt_nonneg e)
      · linarith [Real.sqrt_le_sqrt (by linarith : t + e ≤ t)]
      · linarith [Real.sqrt_le_sqrt (by linarith : t ≤ t - e)]
    · have hsub : ∀ x : ℝ, Real.sqrt (x + e) ≤ Real.sqrt x + Real.sqrt e := by
        intro x
        rw [Real.sqrt_le_left (add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))]
        nlinarith [Real.sq_sqrt' (x := x), Real.sq_sqrt he, le_max_left x 0,
          mul_nonneg (Real.sqrt_nonneg x) (Real.sqrt_nonneg e)]
      refine max_le (by linarith [hsub t]) ?_
      have h := hsub (t - e)
      rw [sub_add_cancel] at h
      linarith
  simp only [errorBound]
  have h := mul_le_mul_of_nonneg_left (key (evalExact env a) (errorBound env a))
    (by linarith : (0 : ℝ) ≤ 1 + Model.unitRoundoffAt format)
  linarith

/-- At a square root whose exact operand `t = evalExact env a` is positive and whose operand bound
`e = errorBound env a` is nonnegative, `errorBound` is at most
`u * Real.sqrt t + ((1 + u) * (e / Real.sqrt t) + η)` with `u`, `η` as in `errorBound`. -/
theorem errorBound_sqrt_le_div (env : Nat → ℝ) (a : Expr Value) (ht : 0 < evalExact env a)
    (he : 0 ≤ errorBound env a) :
    errorBound env (.sqrt a) ≤ Model.unitRoundoffAt format * Real.sqrt (evalExact env a) +
      ((1 + Model.unitRoundoffAt format) * (errorBound env a / Real.sqrt (evalExact env a)) +
        Model.underflowErrorAt format) := by
  have hu := Model.unitRoundoffAt_pos format
  have key : ∀ t e : ℝ, 0 < t → 0 ≤ e →
      max (Real.sqrt (t + e) - Real.sqrt t) (Real.sqrt t - Real.sqrt (t - e)) ≤
        e / Real.sqrt t := by
    intro t e ht he
    have hs : 0 < Real.sqrt t := Real.sqrt_pos.2 ht
    have hst := Real.sq_sqrt ht.le
    refine max_le ?_ ?_ <;> rw [le_div_iff₀ hs]
    · nlinarith [Real.sq_sqrt (by linarith : 0 ≤ t + e),
        sq_nonneg (Real.sqrt (t + e) - Real.sqrt t)]
    · have h1 : Real.sqrt (t - e) ≤ Real.sqrt t := Real.sqrt_le_sqrt (by linarith)
      nlinarith [Real.sq_sqrt' (x := t - e), le_max_left (t - e) 0,
        mul_nonneg (Real.sqrt_nonneg (t - e)) (sub_nonneg.2 h1)]
  simp only [errorBound]
  have h := mul_le_mul_of_nonneg_left (key (evalExact env a) (errorBound env a) ht he)
    (by linarith : (0 : ℝ) ≤ 1 + Model.unitRoundoffAt format)
  linarith

section Runtime

variable [ExecFloat.Backend.PolicyFor (Configured.Family format code plan)]
    [addCap : ExecFloat.Add (Configured.Family format code plan)]
    [subCap : ExecFloat.Sub (Configured.Family format code plan)]
    [mulCap : ExecFloat.Mul (Configured.Family format code plan)]
    [divCap : ExecFloat.Div (Configured.Family format code plan)]
    [fmaCap : ExecFloat.Fma (Configured.Family format code plan)]
    [sqrtCap : ExecFloat.Sqrt (Configured.Family format code plan)]

/-- Evaluate an `Expr` with the executable operations of its format (`ExecFloat.Binary 8 23` for
binary32). -/
def Expr.eval (env : Nat → Value) : Expr Value → Value
  | .var i => env i
  | .const x => x
  | .add a b => ExecFloat.add (eval env a) (eval env b)
  | .sub a b => ExecFloat.sub (eval env a) (eval env b)
  | .mul a b => ExecFloat.mul (eval env a) (eval env b)
  | .div a b => ExecFloat.div (eval env a) (eval env b)
  | .fma a b c =>
      (ExecFloat.Binary.fma (mode := .nearest))
        (eval env a) (eval env b) (eval env c)
  | .sqrt a => (ExecFloat.Binary.sqrt (mode := .nearest)) (eval env a)

/-!
## "Finite evaluation" witnesses (finite at every intermediate node)

The finite bridge relates a finite executable result to real rounding in the same format, so a
statement about a whole expression needs every intermediate value to be finite.

`FiniteEval env e d` records the decoded dyadic `d` of the result together with witnesses for
every subtree. A successful `toDyadic?` rules out NaN and infinity.
-/

/-- Finite-evaluation witness for the compact scalar expression language. -/
inductive FiniteEval (env : Nat → Value) : Expr Value → FloatLib.Numerics.Dyadic → Prop
  where
  /-- A variable whose value decodes to the dyadic `d`. -/
  | var (i : Nat) (d : FloatLib.Numerics.Dyadic) (h : (toModel (env i)).toDyadic? = some d) :
      FiniteEval env (.var i) d
  /-- A constant that decodes to the dyadic `d`. -/
  | const (x : Value) (d : FloatLib.Numerics.Dyadic) (h : (toModel x).toDyadic? =
    some d) :
      FiniteEval env (.const x) d
  /-- A sum of finite subtrees whose float result decodes to the dyadic `dout`. -/
  | add {a b : Expr Value} {da db dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : (toModel (ExecFloat.add (Expr.eval env a) (Expr.eval env b))).toDyadic? = some
        dout) :
      FiniteEval env (.add a b) dout
  /-- A difference of finite subtrees whose float result decodes to the dyadic `dout`. -/
  | sub {a b : Expr Value} {da db dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : (toModel (ExecFloat.sub (Expr.eval env a) (Expr.eval env b))).toDyadic? = some
        dout) :
      FiniteEval env (.sub a b) dout
  /-- A product of finite subtrees whose float result decodes to the dyadic `dout`. -/
  | mul {a b : Expr Value} {da db dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : (toModel (ExecFloat.mul (Expr.eval env a) (Expr.eval env b))).toDyadic? = some
        dout) :
      FiniteEval env (.mul a b) dout
  /-- A quotient of finite subtrees, with a divisor of nonzero significand, whose float result
  decodes to the dyadic `dout`. -/
  | div {a b : Expr Value} {da db dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hden : db.significand ≠ 0)
      (hout : (toModel (ExecFloat.div (Expr.eval env a) (Expr.eval env b))).toDyadic? = some
        dout) :
      FiniteEval env (.div a b) dout
  /-- A fused multiply-add of finite subtrees whose float result decodes to the dyadic `dout`. -/
  | fma {a b c : Expr Value} {da db dc dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db) (hc : FiniteEval env c dc)
      (hout : (toModel ((ExecFloat.Binary.fma (mode := .nearest))
        (Expr.eval env a) (Expr.eval env b) (Expr.eval env c))).toDyadic? = some dout) :
      FiniteEval env (.fma a b c) dout
  /-- A square root of a finite subtree whose float result decodes to the dyadic `dout`. -/
  | sqrt {a : Expr Value} {da dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da)
      (hout : (toModel ((ExecFloat.Binary.sqrt (mode := .nearest))
        (Expr.eval env a))).toDyadic? = some dout) :
      FiniteEval env (.sqrt a) dout

namespace FiniteEval

/-- Extract the decoded dyadic of the runtime evaluation from a `FiniteEval` witness. -/
theorem toDyadic?_eq_some {env : Nat → Value} {e : Expr Value}
    {d : FloatLib.Numerics.Dyadic} :
    FiniteEval env e d → (toModel (Expr.eval env e)).toDyadic? = some d := by
  intro h
  cases h with
  | var i d h =>
      simpa [Expr.eval] using h
  | const x d h =>
      simpa [Expr.eval] using h
  | add ha hb hout =>
      simpa [Expr.eval] using hout
  | sub ha hb hout =>
      simpa [Expr.eval] using hout
  | mul ha hb hout =>
      simpa [Expr.eval] using hout
  | div ha hb hden hout =>
      simpa [Expr.eval] using hout
  | fma ha hb hc hout =>
      simpa [Expr.eval] using hout
  | sqrt ha hout =>
      simpa [Expr.eval] using hout

end FiniteEval

/-!
## Whole-expression refinement

Induction on `FiniteEval` applies the arithmetic refinement theorem at each operation.
-/

/-- If `FiniteEval env e d` holds, the runtime value of `e` decodes to the real value that
`Expr.evalReal` computes from the decoded environment. The format must be IEEE (`hformat`, proved by
`ieee_format` by default). -/
theorem Expr.toReal_eval
    [ExecFloat.Capability.HasSpec addCap Configured.Spec.add]
    [ExecFloat.Capability.HasSpec subCap Configured.Spec.sub]
    [ExecFloat.Capability.HasSpec mulCap Configured.Spec.mul]
    [ExecFloat.Capability.HasSpec divCap Configured.Spec.div]
    [ExecFloat.Capability.HasSpec fmaCap Configured.Spec.fma]
    [ExecFloat.Capability.HasSpec sqrtCap Configured.Spec.sqrt]
    (env : Nat → Value) {e : Expr Value} {d : FloatLib.Numerics.Dyadic}
    (h : FiniteEval env e d) (hformat : format.isIEEE = true := by ieee_format) :
    (toModel (eval env e)).toReal = evalReal (fun i => (toModel (env i)).toReal) e := by
  induction h with
  | var i d h =>
      simp [eval, evalReal]
  | const x d h =>
      simp [eval, evalReal]
  | add ha hb hout iha ihb =>
      rename_i a b da db dout
      have hfin : isFinite (ExecFloat.add (eval env a) (eval env b)) = true :=
        Model.isFinite_eq_true_of_toDyadic?_some hout
      rw [eval, evalReal, ← iha, ← ihb]
      exact ExecFloat.Binary.toReal_add_eq_roundAt hfin hformat
  | sub ha hb hout iha ihb =>
      rename_i a b da db dout
      have hfin : isFinite (ExecFloat.sub (eval env a) (eval env b)) = true :=
        Model.isFinite_eq_true_of_toDyadic?_some hout
      rw [eval, evalReal, ← iha, ← ihb]
      exact ExecFloat.Binary.toReal_sub_eq_roundAt hfin hformat
  | mul ha hb hout iha ihb =>
      rename_i a b da db dout
      have hfin : isFinite (ExecFloat.mul (eval env a) (eval env b)) = true :=
        Model.isFinite_eq_true_of_toDyadic?_some hout
      rw [eval, evalReal, ← iha, ← ihb]
      exact ExecFloat.Binary.toReal_mul_eq_roundAt hfin hformat
  | div ha hb hden hout iha ihb =>
      rename_i a b da db dout
      have hfin : isFinite (ExecFloat.div (eval env a) (eval env b)) = true :=
        Model.isFinite_eq_true_of_toDyadic?_some hout
      rw [eval, evalReal, ← iha, ← ihb]
      exact toReal_div_eq_round hfin hformat
  | fma ha hb hc hout iha ihb ihc =>
      rename_i a b c da db dc dout
      let xa := eval env a
      let xb := eval env b
      let xc := eval env c
      have hfin :
          isFinite
            ((ExecFloat.Binary.fma (mode := .nearest)) xa xb xc) = true :=
        Model.isFinite_eq_true_of_toDyadic?_some hout
      have href :
          (toModel ((ExecFloat.Binary.fma (mode := .nearest))
            xa xb xc)).toReal =
              Model.roundAt format
                ((toModel xa).toReal * (toModel xb).toReal + (toModel xc).toReal) :=
        ExecFloat.Binary.toReal_fma_eq_roundAt hfin hformat
      simpa [xa, xb, xc, eval, evalReal, iha, ihb, ihc] using href
  | sqrt ha hout iha =>
      rename_i a da dout
      let xa := eval env a
      have hfin :
          isFinite ((ExecFloat.Binary.sqrt (mode := .nearest)) xa) = true :=
        Model.isFinite_eq_true_of_toDyadic?_some hout
      have href :
          (toModel ((ExecFloat.Binary.sqrt (mode := .nearest)) xa)).toReal =
            Model.roundAt format (Real.sqrt ((toModel xa).toReal)) :=
        toReal_sqrt_eq_round hfin hformat
      simpa [xa, eval, evalReal, iha] using href

/-- In an IEEE format, the real value of a finite runtime evaluation is within `errorBound` of
the exact evaluation `evalExact` at the real values of `env`, when every divisor exceeds its error
bound (`hdiv`). -/
theorem abs_toReal_eval_sub_evalExact_le
    [ExecFloat.Capability.HasSpec addCap Configured.Spec.add]
    [ExecFloat.Capability.HasSpec subCap Configured.Spec.sub]
    [ExecFloat.Capability.HasSpec mulCap Configured.Spec.mul]
    [ExecFloat.Capability.HasSpec divCap Configured.Spec.div]
    [ExecFloat.Capability.HasSpec fmaCap Configured.Spec.fma]
    [ExecFloat.Capability.HasSpec sqrtCap Configured.Spec.sqrt]
    (env : Nat → Value) {e : Expr Value} {d : FloatLib.Numerics.Dyadic}
    (h : FiniteEval env e d) (hdiv : DivisorsExceedError (fun i => (toModel (env i)).toReal) e)
    (hformat : format.isIEEE = true := by ieee_format) :
    |(toModel (Expr.eval env e)).toReal - evalExact (fun i => (toModel (env i)).toReal) e| ≤
      errorBound (fun i => (toModel (env i)).toReal) e := by
  rw [Expr.toReal_eval env h hformat]
  exact abs_evalReal_sub_evalExact_le _ hdiv

end Runtime

end

end TorchLean.IEEE754
