/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the elementwise operators

Each lemma here states one arm of `Op.rule?` for an operator that acts entry by entry: the
activations, the arithmetic, transcendental and rounding operators, `clamp`, `select`, `detach` and
`nanToNum`. An arithmetic operator at integer words has a lemma of its own (`Op.rule?_add_integer`),
and a lemma whose rule reads a capability of the scalar (`Rounding`, `Erf`, `InverseTrig`, `Cbrt`,
`NextAfter`) assumes it.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.relu`. -/
theorem rule?_relu [Op.Capabilities α] {s : Shape} :
    (Op.relu (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map Activation.relu xs.1, ())
        backward := fun xs g =>
          (Tensor.map Activation.reluDeriv xs.1 * g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.map Activation.reluDeriv xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.sigmoid`. -/
theorem rule?_sigmoid [Op.Capabilities α] {s : Shape} :
    (Op.sigmoid (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map Context.sigmoid xs.1, ())
        backward := fun xs g => (Tensor.sigmoidBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.sigmoidBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.tanh`. -/
theorem rule?_tanh [Op.Capabilities α] {s : Shape} :
    (Op.tanh (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map MathFunctions.tanh xs.1, ())
        backward := fun xs g => (Tensor.tanhBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.tanhBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.exp`. -/
theorem rule?_exp [Op.Capabilities α] {s : Shape} :
    (Op.exp (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.exp xs.1, ())
        backward := fun xs g => (Tensor.zipWith Context.expMul xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith Context.expMul xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.abs`. -/
theorem rule?_abs [Op.Capabilities α] {s : Shape} :
    (Op.abs (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.abs xs.1, ())
        backward := fun xs g => (Tensor.sign xs.1 * g.1, ())
        jvp? := some fun xs dxs => (Tensor.sign xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.sqrt`: IEEE's root, with pullback `g / (2 |y|)` at the root `y`. -/
theorem rule?_sqrt [Op.Capabilities α] {s : Shape} :
    (Op.sqrt (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.sqrt xs.1, ())
        backward := fun xs g => (Tensor.zipWith (fun value seed =>
          Numeric.sqrtBackward (MathFunctions.abs (MathFunctions.sqrt value))
            seed) xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith (fun value seed =>
          Numeric.sqrtBackward (MathFunctions.abs (MathFunctions.sqrt value))
            seed) xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.sqrt .clampNegative`: the root `y` of `max x 0`, with pullback and tangent
`g / (2 y)` where `x > 0` and `0 * g` elsewhere. At a NaN `x` the root is a NaN and `0 * g` is a
zero with the sign of a finite `g` and a NaN at an infinite or NaN `g`. -/
theorem rule?_sqrt_clampNegative [Op.Capabilities α] {s : Shape} :
    (Op.sqrt (s := s) .clampNegative).rule? (α := α) = some
      { forward := fun xs => (Tensor.sqrt xs.1 .clampNegative, ())
        backward := fun xs g => (Tensor.zipWith (fun value seed =>
          if value > 0 then
            Numeric.sqrtBackward (MathFunctions.sqrt value) seed
          else (0 : α) * seed) xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith (fun value seed =>
          if value > 0 then
            Numeric.sqrtBackward (MathFunctions.sqrt value) seed
          else (0 : α) * seed) xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.sin`. -/
theorem rule?_sin [Op.Capabilities α] {s : Shape} :
    (Op.sin (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.sin xs.1, ())
        backward := fun xs g => (Tensor.cos xs.1 * g.1, ())
        jvp? := some fun xs dxs => (Tensor.cos xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.cos`. -/
theorem rule?_cos [Op.Capabilities α] {s : Shape} :
    (Op.cos (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.cos xs.1, ())
        backward := fun xs g => (Tensor.neg (Tensor.sin xs.1) * g.1, ())
        jvp? := some fun xs dxs => (Tensor.neg (Tensor.sin xs.1) * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.neg`. -/
theorem rule?_neg [Op.Capabilities α] {s : Shape} :
    (Op.neg (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.neg xs.1, ())
        backward := fun _ g => (Tensor.neg g.1, ())
        jvp? := some fun _ dxs => (Tensor.neg dxs.1, ()) } :=
  rfl

/-- The rule of `Op.neg` at integer words: two's-complement negation. -/
theorem rule?_neg_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.neg (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map (- ·) xs.1, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.add`. -/
theorem rule?_add [Op.Capabilities α] {s : Shape} :
    (Op.add (s := s)).rule? (α := α) = some
      { forward := fun xs => (xs.1 + xs.2.1, ())
        backward := fun _ g => (g.1, g.1, ())
        jvp? := some fun _ dxs => (dxs.1 + dxs.2.1, ()) } :=
  rfl

/-- The rule of `Op.add` at integer words, wrapping. -/
theorem rule?_add_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.add (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (· + ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.mul`. -/
theorem rule?_mul [Op.Capabilities α] {s : Shape} :
    (Op.mul (s := s)).rule? (α := α) = some
      { forward := fun xs => (xs.1 * xs.2.1, ())
        backward := fun xs g => (g.1 * xs.2.1, g.1 * xs.1, ())
        jvp? := some fun xs dxs => ((dxs.1 * xs.2.1) + (xs.1 * dxs.2.1), ()) } :=
  rfl

/-- The rule of `Op.mul` at integer words, wrapping. -/
theorem rule?_mul_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.mul (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (· * ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.max`. -/
theorem rule?_max [Op.Capabilities α] {s : Shape} :
    (Op.max (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.maximum xs.1 xs.2.1, ())
        backward := fun xs g =>
          let half : α := Rat.cast (1 / 2 : Rat)
          (Tensor.zipWith3 (fun a b seed =>
              if a > b then seed else if b > a then (0 : α) else half * seed) xs.1 xs.2.1 g.1,
            Tensor.zipWith3 (fun a b seed =>
              if a > b then seed else if b > a then (0 : α) else half * seed) xs.2.1 xs.1 g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            if xs.1 c > xs.2.1 c then dxs.1 c else if xs.2.1 c > xs.1 c then dxs.2.1 c
            else Numeric.average (dxs.1 c) (dxs.2.1 c), ())
        alias? := some
          { first := ⟨0, by simp⟩
            second := ⟨1, by simp⟩
            distinct := by simp
            same := rfl
            enabled := (inferInstance : Numeric.QuotientArithmetic α).supported
            prepare := fun xs flags => ⟨(Tensor.maximum xs.1 xs.2.1, ()), fun g =>
              (if flags.1 then some (Tensor.map (fun gradient =>
                Numeric.average gradient gradient) g.1)
              else none, none, ())⟩
            prepare_eq := by intro _ ⟨_, b, _⟩ _; cases b <;> rfl } } :=
  rfl

/-- The rule of `Op.max` at integer words: the larger word in the type's order. -/
theorem rule?_max_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.max (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.maximum signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.log`. -/
theorem rule?_log [Op.Capabilities α] {s : Shape} :
    (Op.log (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.log xs.1, ())
        backward := fun xs g => (g.1 / xs.1, ())
        jvp? := some fun xs dxs => (dxs.1 / xs.1, ()) } :=
  rfl

/-- The rule of `Op.detach`. -/
theorem rule?_detach [Op.Capabilities α] {s : Shape} :
    (Op.detach (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.detach xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full s 0, ()) } :=
  rfl

/-- The rule of `Op.softplus`. -/
theorem rule?_softplus [Op.Capabilities α] {s : Shape} :
    (Op.softplus (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map Activation.softplus xs.1, ())
        backward := fun xs g =>
          (Tensor.map Context.sigmoid xs.1 * g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.map Context.sigmoid xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.silu`. -/
theorem rule?_silu [Op.Capabilities α] {s : Shape} :
    (Op.silu (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map Activation.silu xs.1, ())
        backward := fun xs g =>
          (Tensor.map Activation.siluDeriv xs.1 * g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.map Activation.siluDeriv xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.square`. -/
theorem rule?_square [Op.Capabilities α] {s : Shape} :
    (Op.square (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.square xs.1, ())
        backward := fun xs g =>
          ((Tensor.full _ 2 * xs.1) * g.1, ())
        jvp? := some fun xs dxs => ((Tensor.full _ 2 * xs.1) * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.reciprocal`. -/
theorem rule?_reciprocal [Op.Capabilities α] {s : Shape} :
    (Op.reciprocal (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.reciprocal xs.1, ())
        backward := fun xs g =>
          ((Tensor.neg g.1 / xs.1) / xs.1, ())
        jvp? := some fun xs dxs => ((Tensor.neg dxs.1 / xs.1) / xs.1, ()) } :=
  rfl

/-- The rule of `Op.sub`. -/
theorem rule?_sub [Op.Capabilities α] {s : Shape} :
    (Op.sub (s := s)).rule? (α := α) = some
      { forward := fun xs => (xs.1 - xs.2.1, ())
        backward := fun _ g => (g.1, Tensor.neg g.1, ())
        jvp? := some fun _ dxs => (dxs.1 - dxs.2.1, ()) } :=
  rfl

/-- The rule of `Op.sub` at integer words, wrapping. -/
theorem rule?_sub_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.sub (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (· - ·) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.div` at true division. -/
theorem rule?_div [Op.Capabilities α] {s : Shape} :
    (Op.div (s := s)).rule? (α := α) = some
      { forward := fun xs => (xs.1 / xs.2.1, ())
        backward := fun xs g =>
          (g.1 / xs.2.1, Tensor.neg (g.1 * ((xs.1 / xs.2.1) / xs.2.1)), ())
        jvp? := some fun xs dxs =>
          ((dxs.1 / xs.2.1) - (dxs.2.1 * ((xs.1 / xs.2.1) / xs.2.1)), ()) } :=
  rfl

/-- The rule of `Op.div .trunc`, where the scalar has `Rounding`: the flooring division of `|a|` by
`|b|` with the sign of `a / b` (`trunc (a / b)` over the reals,
`Tensor.copysign_floorDivide_abs_real`), with zero pullbacks and a zero tangent, as PyTorch
differentiates a rounded division. -/
theorem rule?_div_trunc [Rounding α] {s : Shape} :
    (Op.div (s := s) .trunc).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.copysign (Tensor.floorDivide (Tensor.abs xs.1) (Tensor.abs xs.2.1))
            (xs.1 / xs.2.1), ())
        backward := fun _ g =>
          (Tensor.map (fun _ => (0 : α)) g.1, Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) } :=
  rfl

/-- The rule of `Op.div .floor`, where the scalar has `Rounding`: the flooring division
`Tensor.floorDivide`, with zero pullbacks and a zero tangent. -/
theorem rule?_div_floor [Rounding α] {s : Shape} :
    (Op.div (s := s) .floor).rule? (α := α) = some
      { forward := fun xs => (Tensor.floorDivide xs.1 xs.2.1, ())
        backward := fun _ g =>
          (Tensor.map (fun _ => (0 : α)) g.1, Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) } :=
  rfl

/-- `Op.div` has no rule at integer words for true division. -/
theorem rule?_div_integer_eq_none [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.div (s := s) .none (.integer signed width)).rule? (α := α) = none :=
  rfl

/-- The rule of `Op.div .trunc` at integer words: the quotient rounded toward zero, after the check
that no divisor is zero. -/
theorem rule?_div_trunc_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.div (s := s) .trunc (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.divTrunc signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ())
        validate := fun xs => Op.checkDivisors "div" xs.2.1 } :=
  rfl

/-- The rule of `Op.div .floor` at integer words: the quotient rounded toward negative infinity,
after the check that no divisor is zero. -/
theorem rule?_div_floor_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.div (s := s) .floor (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.divFloor signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ())
        validate := fun xs => Op.checkDivisors "div" xs.2.1 } :=
  rfl

/-- The rule of `Op.min`. -/
theorem rule?_min [Op.Capabilities α] {s : Shape} :
    (Op.min (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.minimum xs.1 xs.2.1, ())
        backward := fun xs g =>
          let half : α := Rat.cast (1 / 2 : Rat)
          (Tensor.zipWith3 (fun a b seed =>
              if b > a then seed else if a > b then (0 : α) else half * seed) xs.1 xs.2.1 g.1,
            Tensor.zipWith3 (fun a b seed =>
              if b > a then seed else if a > b then (0 : α) else half * seed) xs.2.1 xs.1 g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            if xs.1 c < xs.2.1 c then dxs.1 c else if xs.2.1 c < xs.1 c then dxs.2.1 c
            else Numeric.average (dxs.1 c) (dxs.2.1 c), ())
        alias? := some
          { first := ⟨0, by simp⟩
            second := ⟨1, by simp⟩
            distinct := by simp
            same := rfl
            enabled := (inferInstance : Numeric.QuotientArithmetic α).supported
            prepare := fun xs flags => ⟨(Tensor.minimum xs.1 xs.2.1, ()), fun g =>
              (if flags.1 then some (Tensor.map (fun gradient =>
                Numeric.average gradient gradient) g.1)
              else none, none, ())⟩
            prepare_eq := by intro _ ⟨_, b, _⟩ _; cases b <;> rfl } } :=
  rfl

/-- The rule of `Op.min` at integer words: the smaller word in the type's order. -/
theorem rule?_min_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.min (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.minimum signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.safeLog`. -/
theorem rule?_safeLog [Op.Capabilities α] {s : Shape} :
    (Op.safeLog (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.map (fun x => Activation.safeLog x xs.2.1.item) xs.1, ())
        backward := fun xs g =>
          ((Tensor.map (fun x => Activation.safeLogDeriv x xs.2.1.item)
            xs.1) * g.1, (), ())
        jvp? := some fun xs dxs =>
          ((Tensor.map (fun x => Activation.safeLogDeriv x xs.2.1.item)
            xs.1) * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.clamp`. -/
theorem rule?_clamp [Op.Capabilities α] {s : Shape} :
    (Op.clamp (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.clamp xs.1 (Tensor.full _ xs.2.1.item)
          (Tensor.full _ xs.2.2.1.item) .interior, ())
        backward := fun xs g =>
          (Tensor.clampDerivative xs.1 (Tensor.full _ xs.2.1.item)
              (Tensor.full _ xs.2.2.1.item) * g.1, (), (),
            ())
        jvp? := some fun xs dxs =>
          (Tensor.clampDerivative xs.1 (Tensor.full _ xs.2.1.item)
              (Tensor.full _ xs.2.2.1.item) * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.scale`. -/
theorem rule?_scale [Op.Capabilities α] {s : Shape} :
    (Op.scale (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.scale xs.1 xs.2.1.item, ())
        backward := fun xs g => (Tensor.scale g.1 xs.2.1.item, (), ())
        jvp? := some fun xs dxs => (Tensor.scale dxs.1 xs.2.1.item, ()) } :=
  rfl

/-- The rule of `Op.elu`. -/
theorem rule?_elu [Op.Capabilities α] {s : Shape} :
    (Op.elu (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          let alpha := xs.2.1.item
          (Tensor.map (fun x => Activation.elu x alpha) xs.1, ())
        backward := fun xs g =>
          let alpha := xs.2.1.item
          (Tensor.map (fun x => Activation.eluDeriv x alpha) xs.1 * g.1, (), ())
        jvp? := some fun xs dxs =>
          let alpha := xs.2.1.item
          (Tensor.map (fun x => Activation.eluDeriv x alpha) xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.leakyReLU`. -/
theorem rule?_leakyReLU [Op.Capabilities α] {s : Shape} :
    (Op.leakyReLU (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          let slope := xs.2.1.item
          (Tensor.map (fun x => Activation.leakyReLU x slope) xs.1, ())
        backward := fun xs g =>
          let slope := xs.2.1.item
          (Tensor.map (fun x => Activation.leakyReLUDeriv x slope) xs.1 * g.1, (), ())
        jvp? := some fun xs dxs =>
          let slope := xs.2.1.item
          (Tensor.map (fun x => Activation.leakyReLUDeriv x slope) xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.sign`. -/
theorem rule?_sign [Op.Capabilities α] {s : Shape} :
    (Op.sign (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.sign xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) } :=
  rfl

/-- The rule of `Op.floor`, where the scalar has `Rounding`. -/
theorem rule?_floor [Rounding α] {s : Shape} :
    (Op.floor (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.floor xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) } :=
  rfl

/-- The rule of `Op.ceil`, where the scalar has `Rounding`. -/
theorem rule?_ceil [Rounding α] {s : Shape} :
    (Op.ceil (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.ceil xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) } :=
  rfl

/-- The rule of `Op.round`, where the scalar has `Rounding`. -/
theorem rule?_round [Rounding α] {s : Shape} :
    (Op.round (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.round xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) } :=
  rfl

/-- The rule of `Op.trunc`, where the scalar has `Rounding`. -/
theorem rule?_trunc [Rounding α] {s : Shape} :
    (Op.trunc (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.trunc xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : α)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : α)) dxs.1, ()) } :=
  rfl

/-- The rule of `Op.frac`, where the scalar has `Rounding`. -/
theorem rule?_frac [Rounding α] {s : Shape} :
    (Op.frac (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.frac xs.1, ())
        backward := fun _ g => (g.1, ())
        jvp? := some fun _ dxs => (dxs.1, ()) } :=
  rfl

/-- The rule of `Op.rsqrt`. -/
theorem rule?_rsqrt [Op.Capabilities α] {s : Shape} :
    (Op.rsqrt (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.rsqrt xs.1, ())
        backward := fun xs g => (Tensor.rsqrtBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.rsqrtBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.exp2`. -/
theorem rule?_exp2 [Op.Capabilities α] {s : Shape} :
    (Op.exp2 (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.exp2 xs.1, ())
        backward := fun xs g => (Tensor.exp2Backward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.exp2Backward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.expm1`. -/
theorem rule?_expm1 [Op.Capabilities α] {s : Shape} :
    (Op.expm1 (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.expm1 xs.1, ())
        backward := fun xs g => (Tensor.zipWith Context.expMul xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith Context.expMul xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.log2`. -/
theorem rule?_log2 [Op.Capabilities α] {s : Shape} :
    (Op.log2 (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.log2 xs.1, ())
        backward := fun xs g => (Tensor.log2Backward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.log2Backward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.log10`. -/
theorem rule?_log10 [Op.Capabilities α] {s : Shape} :
    (Op.log10 (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.log10 xs.1, ())
        backward := fun xs g => (Tensor.log10Backward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.log10Backward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.log1p`. -/
theorem rule?_log1p [Op.Capabilities α] {s : Shape} :
    (Op.log1p (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.log1p xs.1, ())
        backward := fun xs g => (Tensor.log1pBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.log1pBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.tan`. -/
theorem rule?_tan [Op.Capabilities α] {s : Shape} :
    (Op.tan (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.tan xs.1, ())
        backward := fun xs g => (Tensor.tanBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.tanBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.asinh`. -/
theorem rule?_asinh [Op.Capabilities α] {s : Shape} :
    (Op.asinh (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.asinh xs.1, ())
        backward := fun xs g => (Tensor.asinhBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.asinhBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.acosh`. -/
theorem rule?_acosh [Op.Capabilities α] {s : Shape} :
    (Op.acosh (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.acosh xs.1, ())
        backward := fun xs g => (Tensor.acoshBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.acoshBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.atanh`. -/
theorem rule?_atanh [Op.Capabilities α] {s : Shape} :
    (Op.atanh (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.atanh xs.1, ())
        backward := fun xs g => (Tensor.atanhBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.atanhBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.atan`, where the scalar has `InverseTrig`. -/
theorem rule?_atan [InverseTrig α] {s : Shape} :
    (Op.atan (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.atan xs.1, ())
        backward := fun xs g => (Tensor.atanBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.atanBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.asin`, where the scalar has `InverseTrig`. -/
theorem rule?_asin [InverseTrig α] {s : Shape} :
    (Op.asin (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.asin xs.1, ())
        backward := fun xs g => (Tensor.asinBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.asinBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.acos`, where the scalar has `InverseTrig`. -/
theorem rule?_acos [InverseTrig α] {s : Shape} :
    (Op.acos (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.acos xs.1, ())
        backward := fun xs g => (Tensor.acosBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.acosBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.erf`, where the scalar has `Erf`. -/
theorem rule?_erf [Erf α] {s : Shape} :
    (Op.erf (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.erf xs.1, ())
        backward := fun xs g => (Tensor.erfBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.erfBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.atan2`, where the scalar has `InverseTrig`. -/
theorem rule?_atan2 [InverseTrig α] {s : Shape} :
    (Op.atan2 (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.atan2 xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.atan2Backward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            Numeric.angleTangent (xs.2.1 c) (xs.1 c) (dxs.2.1 c) (dxs.1 c), ()) } :=
  rfl

/-- The rule of `Op.fmod`, where the scalar has `Rounding`. -/
theorem rule?_fmod [Rounding α] {s : Shape} :
    (Op.fmod (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.fmod xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.fmodBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.fmodBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.fmodBackward xs.1 xs.2.1 dxs.2.1).2, ()) } :=
  rfl

/-- The rule of `Op.remainder`, where the scalar has `Rounding`. -/
theorem rule?_remainder [Rounding α] {s : Shape} :
    (Op.remainder (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.remainder xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.remainderBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.remainderBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.remainderBackward xs.1 xs.2.1 dxs.2.1).2, ()) } :=
  rfl

/-- The rule of `Op.remainder` at integer words: PyTorch's remainder, with the divisor's sign, after
the check that no divisor is zero. -/
theorem rule?_remainder_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.remainder (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.remainder signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ())
        validate := fun xs => Op.checkDivisors "remainder" xs.2.1 } :=
  rfl

/-- The rule of `Op.hypot`. -/
theorem rule?_hypot [Op.Capabilities α] {s : Shape} :
    (Op.hypot (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.hypot xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.hypotBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.hypotBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.hypotBackward xs.1 xs.2.1 dxs.2.1).2, ()) } :=
  rfl

/-- The rule of `Op.copysign`. -/
theorem rule?_copysign [Op.Capabilities α] {s : Shape} :
    (Op.copysign (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.copysign xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.copysignBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.copysignBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.copysignBackward xs.1 xs.2.1 dxs.2.1).2, ()) } :=
  rfl

/-- The rule of `Op.fmax`. -/
theorem rule?_fmax [Op.Capabilities α] {s : Shape} :
    (Op.fmax (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.fmax xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.fmaxBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.fmaxBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.fmaxBackward xs.1 xs.2.1 dxs.2.1).2, ()) } :=
  rfl

/-- The rule of `Op.fmin`. -/
theorem rule?_fmin [Op.Capabilities α] {s : Shape} :
    (Op.fmin (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.fmin xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.fminBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.fminBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.fminBackward xs.1 xs.2.1 dxs.2.1).2, ()) } :=
  rfl

/-- The rule of `Op.nanToNum`. -/
theorem rule?_nanToNum [Op.Capabilities α] {s : Shape} :
    (Op.nanToNum (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.nanToNum xs.1 xs.2.1.item xs.2.2.1.item xs.2.2.2.1.item, ())
        backward := fun xs g => (Tensor.nanToNumBackward xs.1 g.1, (), (), (), ())
        jvp? := some fun xs dxs => (Tensor.nanToNumBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.select`. -/
theorem rule?_select [Op.Capabilities α] {s : Shape} :
    (Op.select (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.select xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun xs g =>
          let (dx, dy) := Tensor.selectBackward xs.1 g.1
          ((), dx, dy, ())
        jvp? := some fun xs dxs => (Tensor.select xs.1 dxs.2.1 dxs.2.2.1, ()) } :=
  rfl

/-- The rule of `Op.select` at Booleans. -/
theorem rule?_select_bool [Op.Capabilities α] {s : Shape} :
    (Op.select (s := s) .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.select xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun _ _ => ((), (), (), ()) } :=
  rfl

/-- The rule of `Op.select` at integer words. -/
theorem rule?_select_integer [Op.Capabilities α] {s : Shape} {signed : Bool} {width : Nat} :
    (Op.select (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.select xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun _ _ => ((), (), (), ()) } :=
  rfl

/-- The rule of `Op.lerp`. -/
theorem rule?_lerp [Op.Capabilities α] {s : Shape} :
    (Op.lerp (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.lerp xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun xs g =>
          let (da, db, dw) := Tensor.lerpBackward xs.1 xs.2.1 xs.2.2.1 g.1
          (da, db, dw, ())
        jvp? := some fun xs dxs =>
          let pullback := Tensor.lerpBackward xs.1 xs.2.1 xs.2.2.1
          (((pullback dxs.1).1 + (pullback dxs.2.1).2.1) + (pullback dxs.2.2.1).2.2, ()) } :=
  rfl

/-- The rule of `Op.addcmul`. -/
theorem rule?_addcmul [Op.Capabilities α] {s : Shape} :
    (Op.addcmul (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.addcmul xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item, ())
        backward := fun xs g =>
          let (dx, da, db) := Tensor.addcmulBackward xs.2.1 xs.2.2.1 xs.2.2.2.1.item g.1
          (dx, da, db, (), ())
        jvp? := some fun xs dxs =>
          let pullback := Tensor.addcmulBackward xs.2.1 xs.2.2.1 xs.2.2.2.1.item
          (((pullback dxs.1).1 + (pullback dxs.2.1).2.1) + (pullback dxs.2.2.1).2.2, ()) } :=
  rfl

/-- The rule of `Op.addcdiv`. -/
theorem rule?_addcdiv [Op.Capabilities α] {s : Shape} :
    (Op.addcdiv (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.addcdiv xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1.item, ())
        backward := fun xs g =>
          let (dx, da, db) := Tensor.addcdivBackward xs.2.1 xs.2.2.1 xs.2.2.2.1.item g.1
          (dx, da, db, (), ())
        jvp? := some fun xs dxs =>
          let pullback := Tensor.addcdivBackward xs.2.1 xs.2.2.1 xs.2.2.2.1.item
          (((pullback dxs.1).1 + (pullback dxs.2.1).2.1) + (pullback dxs.2.2.1).2.2, ()) } :=
  rfl

/-- The rule of `Op.gelu` at the tanh formula. -/
theorem rule?_gelu [Op.Capabilities α] {s : Shape} :
    (Op.gelu (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map Activation.geluTanh xs.1, ())
        backward := fun xs g =>
          (Tensor.map Activation.geluTanhDeriv xs.1 * g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.map Activation.geluTanhDeriv xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.gelu .none`, where the scalar has `Erf`: the exact GELU and its pullback. -/
theorem rule?_gelu_none [Erf α] {s : Shape} :
    (Op.gelu (s := s) .none).rule? (α := α) = some
      { forward := fun xs => (Tensor.gelu xs.1, ())
        backward := fun xs g => (Tensor.geluBackward xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.geluBackward xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.sinh`. -/
theorem rule?_sinh [Op.Capabilities α] {s : Shape} :
    (Op.sinh (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.sinh xs.1, ())
        backward := fun xs g => (Tensor.map MathFunctions.cosh xs.1 * g.1, ())
        jvp? := some fun xs dxs => (Tensor.map MathFunctions.cosh xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.cosh`. -/
theorem rule?_cosh [Op.Capabilities α] {s : Shape} :
    (Op.cosh (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.cosh xs.1, ())
        backward := fun xs g => (Tensor.map MathFunctions.sinh xs.1 * g.1, ())
        jvp? := some fun xs dxs => (Tensor.map MathFunctions.sinh xs.1 * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.safeInv`. -/
theorem rule?_safeInv [Op.Capabilities α] {s : Shape} :
    (Op.safeInv (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.safeDiv (Tensor.full _ (1 : α)) xs.1, ())
        backward := fun xs g =>
          ((Tensor.full _ (-(1 : α)) *
            (Tensor.square (Tensor.map (fun y =>
                (1 : α) / (y + Context.defaultEpsilon)) xs.1))) * g.1, ())
        jvp? := some fun xs dxs =>
          ((Tensor.full _ (-(1 : α)) *
            (Tensor.square (Tensor.map (fun y =>
                (1 : α) / (y + Context.defaultEpsilon)) xs.1))) * dxs.1, ()) } :=
  rfl

/-- The rule of `Op.smoothAbs`. -/
theorem rule?_smoothAbs [Op.Capabilities α] {s : Shape} :
    (Op.smoothAbs (s := s)).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.map (fun x => Activation.smoothAbs x xs.2.1.item) xs.1, ())
        backward := fun xs g =>
          (Tensor.map (fun x => Activation.smoothAbsDeriv x xs.2.1.item) xs.1 *
            g.1, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.map (fun x => Activation.smoothAbsDeriv x xs.2.1.item) xs.1 *
            dxs.1, ()) } :=
  rfl

/-- The rule of `Op.safeDiv`. -/
theorem rule?_safeDiv [Op.Capabilities α] {s : Shape} :
    (Op.safeDiv (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.safeDiv xs.1 xs.2.1, ())
        backward := fun xs g => (Tensor.safeDiv g.1 xs.2.1, (), ())
        jvp? := some fun xs dxs => (Tensor.safeDiv dxs.1 xs.2.1, ()) } :=
  rfl

/-- The rule of `Op.pow`. -/
theorem rule?_pow [Op.Capabilities α] {s : Shape} :
    (Op.pow (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.pow xs.1 xs.2.1, ())
        backward := fun xs g =>
          (Tensor.Rep.ofFlatFn fun index =>
            Numeric.powerBackward Context.isZero
              (Tensor.Rep.getFlat xs.1 index) (Tensor.Rep.getFlat xs.2.1 index)
              (Tensor.Rep.getFlat g.1 index), (), ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFlatFn fun index =>
            Numeric.powerBackward Context.isZero
              (Tensor.Rep.getFlat xs.1 index) (Tensor.Rep.getFlat xs.2.1 index)
              (Tensor.Rep.getFlat dxs.1 index), ()) } :=
  rfl

/-- The rule of `Op.pow` at integer words: PyTorch's integer power. -/
theorem rule?_pow_integer [Op.Capabilities α] {signed : Bool} {width : Nat} {s : Shape} :
    (Op.pow (s := s) (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith (Word.pow signed) xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.cbrt`, where the scalar has `Cbrt`. -/
theorem rule?_cbrt [Cbrt α] {s : Shape} :
    (Op.cbrt (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.map Cbrt.cbrt xs.1, ())
        backward := fun xs g => (Tensor.zipWith (fun x seed =>
          let y := Cbrt.cbrt x
          seed * (((1 / 3 : Rat) : α) * (1 / (y * y)))) xs.1 g.1, ())
        jvp? := some fun xs dxs => (Tensor.zipWith (fun x dx =>
          let y := Cbrt.cbrt x
          dx * (((1 / 3 : Rat) : α) * (1 / (y * y)))) xs.1 dxs.1, ()) } :=
  rfl

/-- The rule of `Op.nextafter`, where the scalar has `NextAfter`. -/
theorem rule?_nextafter [NextAfter α] {s : Shape} :
    (Op.nextafter (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.zipWith NextAfter.nextafter xs.1 xs.2.1, ())
        backward := fun _ _ => ((), (), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  rfl

end Op

end TorchLean
