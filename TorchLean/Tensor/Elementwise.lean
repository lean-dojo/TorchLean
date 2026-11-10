/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Numeric.InverseTrig
public import TorchLean.Tensor.Arithmetic

/-!
# Elementwise operators of core ATen and `jax.lax`

Each operator applies one scalar function at every coordinate, through `map`, `zipWith` or
`zipWith3`, and every operand has the result's shape. A broadcast operand is written with
`Tensor.broadcast`. The scalar functions are fields of `Context` (`copysign`, `isNaN`, `tan`, ...),
fields of `Rounding` (`floor`, `ceil`, `trunc`, `round`, `fmod`), or are defined below from those,
following PyTorch's CPU kernels where the choice matters.

| Operator | PyTorch | `jax.lax` | Policy |
| --- | --- | --- | --- |
| `sign` | `sign` | `sign` | `x` at a NaN and both zeros, as `lax.sign` (`torch.sign` gives `0`) |
| `floor`, `ceil`, `trunc` | same | `floor`, `ceil` | exact, with infinities, NaNs and zeros kept |
| `round` | `round` | `round` (even) | ties to even, with `round (-0.5) = -0` |
| `frac` | `frac` | | `x - trunc x`, `+0` at an integral `x` and a NaN at an infinity |
| `fmod`, `remainder` | same | `rem` | sign of `x` (C `fmod`), of `y` (Python's `%`, zeros too) |
| `maximum`, `minimum` | same | `max`, `min` | IEEE 754-2019 9.6 `maximum`: a quiet NaN, `+0 > -0` |
| `fmax`, `fmin` | same | | 9.6 `maximumNumber`: a NaN ignored against a number, `+0 > -0` |
| `clamp` | `clamp` | `clamp` | `minimum (maximum x lo) hi` |
| `copysign` | same | | the sign bit of `y` in IEEE formats |
| `eq` ... `ge` | same | same | false at a NaN, except `ne` |
| `isNaN`, `isInf`, `isFinite` | `isnan` ... | `is_finite` | |
| `select` | `where` | `select_n` | `where` is a Lean keyword |
| `lerp` | `lerp` | | ATen's two-sided formula, split at `|w| < 1/2` |
| `nanToNum` | `nan_to_num` | | replacement values are arguments |

So `clamp` gives `hi` when `lo > hi` with no NaN, and a NaN at a NaN bound. PyTorch's defaults for
`nan_to_num` are `0` and the largest finite magnitudes of the format. `maximum`, `minimum`, `fmax`
and `fmin` return the values of IEEE 754-2019 9.6's `maximum`, `minimum`, `maximumNumber` and
`minimumNumber`: `+0` is above `-0`, and a NaN result is quiet (`TorchLean.Context.quietNaN`), the
first operand's when both are NaNs. The scalar interface has no exception flags, so the invalid
operation a signaling NaN signals, also where `maximumNumber` returns the other operand, is not
represented. On equal zeros of opposite signs, PyTorch 2.13's CPU `maximum` and `minimum` return
the first operand in the scalar loop and the second in its vectorized blocks, so the sign of the
zero depends on the entry's position. We measured these policies on x86-64 in binary64 and
binary32. `sign` gives a NaN at a NaN and keeps the sign of a zero, as `jax.lax.sign` does, while
`torch.sign` gives `0` at both. `remainder` gives a zero result the divisor's sign, as Python's `%`
and NumPy do and as `torch.remainder` documents. PyTorch 2.13 and JAX 0.12 return the dividend's.

We wrote the scalar functions by hand against the PyTorch 2.8 docs and `jax.lax`, and `lerp` from
ATen's `Lerp.h`. Torch 2.8.0, the version the parity checks pin, behaves as measured above. On our
AVX-512 machine its binary32 `torch.maximum` of `-0` and `0` gives `-0` at each of 8 entries and
`0` at the first 32 of 33. `torch.remainder` of `-4` by `2` is `-0`, and `torch.sign` maps a NaN
and `-0` to `0`. Its `fmax` of `-0` and `0` is `-0` at 1, 16 and 33 entries, where ours is `0`.
The capture tests `test_exact_roundings_match_eager_words` and
`test_binary_functions_match_eager_words` in `scripts/checks/test_ir_capture.py` compare the
roundings, `frac`, `copysign` and `fmod` with eager PyTorch, and `fmax` and `fmin` away from tied
zeros. `test_remainder_is_python_modulus` compares `remainder` with Python's `%`.

The integral roundings and remainders need a `Rounding` instance, `atan2`, `atan`, `asin` and
`acos` an `InverseTrig` instance, which supplies each function, and `erf` an `Erf` instance. A
scalar type without the instance does not have these operators.

## References

- PyTorch Core ATen IR, "IRs" (PyTorch 2.8 docs):
  `https://docs.pytorch.org/docs/2.8/torch.compiler_ir.html`
- `jax.lax`: `https://docs.jax.dev/en/latest/jax.lax.html`
- `torch.sign`, `torch.remainder`, `torch.clamp`, `torch.lerp` and `torch.nan_to_num` (PyTorch 2.8
  docs), at `https://docs.pytorch.org/docs/2.8/generated/torch.sign.html` and the pages of the
  same form
- ATen's `lerp` and `is_lerp_weight_small`, `aten/src/ATen/native/Lerp.h` at v2.8.0:
  `https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/Lerp.h`
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Context

variable {α : Type} [Context α]

/-- Neither a NaN nor an infinity. -/
def isFinite (x : α) : Bool := !isNaN x && !isInf x

/-- Succeeds exactly when `x` meets each given bound, `lo ≤ x` and `x ≤ hi` (strictly unless
`closed`), and is finite unless both bounds are given (`requireWithin_eq_ok_iff`). Otherwise it
fails with a message naming `label`, the range and `x`. -/
@[inline] def requireWithin [DecidableLE α] [ToString α] (label : String) (lo hi : Option α) (x : α)
    (closed : Bool := true) : Except String Unit :=
  let below (a b : α) : Bool := if closed then a ≤ b else a < b
  if lo.all (below · x) && hi.all (below x ·) && (lo.isSome && hi.isSome || isFinite x) then
    .ok ()
  else
    let range := match lo, hi with
      | some l, some h => if closed then s!"lie in [{l}, {h}]" else s!"lie in ({l}, {h})"
      | some l, none => s!"be finite and {if closed then "at least" else "above"} {l}"
      | none, some h => s!"be finite and {if closed then "at most" else "below"} {h}"
      | none, none => "be finite"
    .error s!"{label} must {range}, got {x}"

/-- The success condition of `requireWithin`: each given bound holds, strictly unless `closed`, and
`x` is finite unless both bounds are given. -/
theorem requireWithin_eq_ok_iff [DecidableLE α] [ToString α] {label : String} {lo hi : Option α}
    {x : α} {closed : Bool} :
    requireWithin label lo hi x closed = .ok () ↔
      (∀ l ∈ lo, if closed then l ≤ x else l < x) ∧ (∀ h ∈ hi, if closed then x ≤ h else x < h) ∧
        (lo.isSome ∧ hi.isSome ∨ isFinite x) := by
  cases closed <;> cases lo <;> cases hi <;> simp [requireWithin]

/-- `1` above zero, `-1` below, and otherwise `x` itself without differentiation metadata: a zero
keeps its sign and a NaN stays a NaN, as `jax.lax.sign` computes it. -/
def sign (x : α) : α := if x > 0 then 1 else if 0 > x then -1 else stopGradient x

/-- `1 / x`. -/
def reciprocal (x : α) : α := 1 / x

/-- IEEE 754-2019 9.6 `maximum`: the larger operand, `+0` above `-0` (`copysign` reads the signs)
and `x` when neither is larger, or a quiet NaN when an operand is a NaN, `x`'s before `y`'s. A
result that neither operand wins strictly goes through `extremaTie`, the identity except on dual
numbers, which average the two tangents there. -/
def maximum (x y : α) : α :=
  if isNaN x then extremaTie x y (quietNaN x) else if isNaN y then extremaTie x y (quietNaN y)
  else if y > x then y else if x > y then x
  else extremaTie x y (if 0 > copysign 1 x ∧ copysign 1 y > 0 then y else x)

/-- IEEE 754-2019 9.6 `minimum`: the smaller operand, `-0` below `+0` (`copysign` reads the signs)
and `x` when neither is smaller, or a quiet NaN when an operand is a NaN, `x`'s before `y`'s. A
result that neither operand wins strictly goes through `extremaTie`, as in `maximum`. -/
def minimum (x y : α) : α :=
  if isNaN x then extremaTie x y (quietNaN x) else if isNaN y then extremaTie x y (quietNaN y)
  else if x > y then y else if y > x then x
  else extremaTie x y (if copysign 1 x > 0 ∧ 0 > copysign 1 y then y else x)

/-- Whether `fmax x y` is `x` (its quiet form when both are NaNs): where `y` is a NaN, and where
neither is a NaN and `x` is not below `y`, `+0` counting as above `-0`. -/
def fmaxTakesFirst (x y : α) : Bool :=
  if isNaN y then true else if isNaN x || gtBool y x then false else if gtBool x y then true
  else !(gtBool 0 (copysign 1 x) && gtBool (copysign 1 y) 0)

/-- IEEE 754-2019 9.6 `maximumNumber`: the larger operand, `+0` above `-0` and `x` when neither is
larger, the number against a NaN, or a quiet NaN, `x`'s, when both are NaNs. It returns the operand
`fmaxTakesFirst` names, so that dual numbers carry that operand's tangent. -/
def fmax (x y : α) : α :=
  if isNaN x && isNaN y then quietNaN x else if fmaxTakesFirst x y then x else y

/-- Whether `fmin x y` is `x` (its quiet form when both are NaNs): where `y` is a NaN, and where
neither is a NaN and `x` is not above `y`, `-0` counting as below `+0`. -/
def fminTakesFirst (x y : α) : Bool :=
  if isNaN y then true else if isNaN x || gtBool x y then false else if gtBool y x then true
  else !(gtBool (copysign 1 x) 0 && gtBool 0 (copysign 1 y))

/-- IEEE 754-2019 9.6 `minimumNumber`: the smaller operand, `-0` below `+0` and `x` when neither is
smaller, the number against a NaN, or a quiet NaN, `x`'s, when both are NaNs. It returns the operand
`fminTakesFirst` names, so that dual numbers carry that operand's tangent. -/
def fmin (x y : α) : α :=
  if isNaN x && isNaN y then quietNaN x else if fminTakesFirst x y then x else y

/-- `x` limited to `[lo, hi]`: `minimum (maximum x lo) hi`. -/
def clamp (x lo hi : α) : α := minimum (maximum x lo) hi

/-- `x` with a NaN replaced by `nan` and an infinity by `posinf` or `neginf`. -/
def nanToNum (x nan posinf neginf : α) : α :=
  if isNaN x then nan else if isInf x then (if x > 0 then posinf else neginf) else x

/-- ATen's linear interpolation: `a + w (b - a)` when `|w| < 1/2`, else `b - (b - a)(1 - w)`, with
`b - a` rounded first, so finite endpoints whose difference overflows give an infinity or a NaN. -/
def lerp (a b w : α) : α :=
  if ((1 / 2 : Rat) : α) > MathFunctions.abs w then a + w * (b - a) else b - (b - a) * (1 - w)

/-- `x + value * a * b`, PyTorch's `addcmul`. -/
def addcmul (x a b value : α) : α := x + value * a * b

/-- `x + value * (a / b)`, PyTorch's `addcdiv`. -/
def addcdiv (x a b value : α) : α := x + value * (a / b)

/-- The tangent of `copysign x y` along `x`: zero at a zero `x`, else `dx` where `copysign 1 x`
and `copysign 1 y` agree and `-dx` where they differ, also at an infinite or NaN `x`. -/
def copysignTangent (x y dx : α) : α :=
  if x == 0 then 0 else if copysign 1 x == copysign 1 y then dx else -dx

/-- The exact GELU `x Φ(x)`, `x * ½ * (1 + erf (x * √½))` in PyTorch's order with `√½` the scalar's
square root, for an error function with values in the monad `m`, `Id` for `Tensor.gelu`. At `±∞` it
is the limit, `+∞` or `-0`, without reading `erf`, since at `-∞` that order would compute
`-∞ · 0`, a NaN. -/
def geluErf {m : Type → Type} [Monad m] (erf : α → m α) (x : α) : m α :=
  if isInf x then pure (if x > 0 then x else -0) else do
    let half : α := ((1 / 2 : Rat) : α)
    let value ← erf (x * MathFunctions.sqrt half)
    pure (x * half * (1 + value))

/-- The truncation `n` of the exact quotient `x / y`, so that `fmod x y = x - n y`, computed as
`round ((x - fmod x y) / y)` with the sign of `x / y`, and a NaN where `fmod x y` is one. In a
binary format of precision `p` it is exact while `|n| < 2 ^ (p - 2)` (`2 ^ 51` in binary64,
`2 ^ 22` in binary32), and otherwise within a relative `2 ^ (2 - p)` of `n` or infinite where it
overflows. -/
def fmodQuotient [Rounding α] (x y : α) : α :=
  copysign (Rounding.round ((x - Rounding.fmod x y) / y)) (x / y)

/-- The floor of the exact quotient `x / y`: `fmodQuotient x y`, less one where `fmod x y` is
nonzero and its sign differs from `y`'s, the test with which `remainderSchedule` moves it by
`y`. It is a NaN where `fmod x y` is one and exact where `fmodQuotient x y` is. -/
def remainderQuotient [Rounding α] (x y : α) : α :=
  let n := fmodQuotient x y
  let r := Rounding.fmod x y
  if !(r == 0) && decide (r < 0) != decide (y < 0) then n - 1 else n

/-- The negation of `x == y`. In IEEE formats it holds wherever an operand is a NaN and fails at
`-0` against `+0`. -/
def neBool (x y : α) : Bool := !(x == y)

/-- `x < y` as a `Bool`. -/
def ltBool (x y : α) : Bool := gtBool y x

/-- `x < y` or `x == y`, which is false at a NaN. -/
def leBool (x y : α) : Bool := gtBool y x || x == y

/-- `x > y` or `x == y`, which is false at a NaN. -/
def geBool (x y : α) : Bool := gtBool x y || x == y

end TorchLean.Context

namespace TorchLean

/-- On a linear order whose `Context` comparison is the order's `<`, with `isNaN` always `false`
and `extremaTie x y r = r`, `maximum` is the order's `max`. -/
theorem Context.maximum_eq_max {α : Type} [LinearOrder α] {C : Context α}
    (hnan : ∀ x : α, isNaN x = false) (htie : ∀ x y r : α, extremaTie x y r = r)
    (hlt : ∀ x y : α, @LT.lt α C.toLT x y ↔ @LT.lt α Preorder.toLT x y) (x y : α) :
    maximum x y = @Max.max α LinearOrder.toMax x y := by
  unfold maximum
  simp only [hnan, htie, Bool.false_eq_true, ↓reduceIte]
  split_ifs with h₁ h₂
  · exact (max_eq_right ((hlt x y).mp h₁).le).symm
  · exact (max_eq_left ((hlt y x).mp h₂).le).symm
  all_goals
    obtain rfl : x = y :=
      le_antisymm (not_lt.mp (mt (hlt y x).mpr h₂)) (not_lt.mp (mt (hlt x y).mpr h₁))
    exact (max_self x).symm

/-- On a linear order whose `Context` comparison is the order's `<`, with `isNaN` always `false`
and `extremaTie x y r = r`, `minimum` is the order's `min`. -/
theorem Context.minimum_eq_min {α : Type} [LinearOrder α] {C : Context α}
    (hnan : ∀ x : α, isNaN x = false) (htie : ∀ x y r : α, extremaTie x y r = r)
    (hlt : ∀ x y : α, @LT.lt α C.toLT x y ↔ @LT.lt α Preorder.toLT x y) (x y : α) :
    minimum x y = @Min.min α LinearOrder.toMin x y := by
  unfold minimum
  simp only [hnan, htie, Bool.false_eq_true, ↓reduceIte]
  split_ifs with h₁ h₂
  · exact (min_eq_right ((hlt y x).mp h₁).le).symm
  · exact (min_eq_left ((hlt x y).mp h₂).le).symm
  all_goals
    obtain rfl : x = y :=
      le_antisymm (not_lt.mp (mt (hlt y x).mpr h₁)) (not_lt.mp (mt (hlt x y).mpr h₂))
    exact (min_self x).symm

namespace Tensor

/-- `if c then x else y` at every coordinate, PyTorch's `where` and `jax.lax.select_n`. -/
def select {α : Type} [Storage α] {s : Shape} (c : Tensor Bool s) (x y : Tensor α s) :
    Tensor α s :=
  zipWith3 (fun b u v => if b then u else v) c x y

/-- Entry `i` of `select c x y` is `x i` where `c i` holds and `y i` elsewhere. -/
@[simp] theorem select_apply {α : Type} [Storage α] {s : Shape} (c : Tensor Bool s)
    (x y : Tensor α s) (i : s.Coord) : select c x y i = if c i then x i else y i :=
  zipWith3_apply _ c x y i

/-- Logical negation of a Boolean tensor. -/
def logicalNot {s : Shape} (x : Tensor Bool s) : Tensor Bool s := map (!·) x

/-- Logical conjunction of two Boolean tensors. -/
def logicalAnd {s : Shape} (x y : Tensor Bool s) : Tensor Bool s := zipWith (· && ·) x y

/-- Logical disjunction of two Boolean tensors. -/
def logicalOr {s : Shape} (x y : Tensor Bool s) : Tensor Bool s := zipWith (· || ·) x y

/-- Exclusive disjunction of two Boolean tensors. -/
def logicalXor {s : Shape} (x y : Tensor Bool s) : Tensor Bool s := zipWith xor x y

/-- Entry `i` of `logicalNot x` is `!x i`. -/
@[simp] theorem logicalNot_apply {s : Shape} (x : Tensor Bool s) (i : s.Coord) :
    logicalNot x i = !x i := Tensor.map_apply _ x i

/-- Entry `i` of `logicalAnd x y` is `x i && y i`. -/
@[simp] theorem logicalAnd_apply {s : Shape} (x y : Tensor Bool s) (i : s.Coord) :
    logicalAnd x y i = (x i && y i) := zipWith_apply _ x y i

/-- Entry `i` of `logicalOr x y` is `x i || y i`. -/
@[simp] theorem logicalOr_apply {s : Shape} (x y : Tensor Bool s) (i : s.Coord) :
    logicalOr x y i = (x i || y i) := zipWith_apply _ x y i

/-- Entry `i` of `logicalXor x y` is `xor (x i) (y i)`. -/
@[simp] theorem logicalXor_apply {s : Shape} (x y : Tensor Bool s) (i : s.Coord) :
    logicalXor x y i = xor (x i) (y i) := zipWith_apply _ x y i

variable {α : Type} [Storage α] [Context α] {s : Shape}

/-- `select c x y` (PyTorch's `where`), whose pullback sends the gradient to `x` where `c` holds and
to `y` elsewhere. -/
def selectBackward (c : Tensor Bool s) (g : Tensor α s) : Tensor α s × Tensor α s :=
  (select c g (map (fun _ => (0 : α)) g), select c (map (fun _ => (0 : α)) g) g)

/-- `Context.sign` at every coordinate: `1` above zero, `-1` below, a NaN and a zero kept
(`jax.lax.sign`). -/
def sign (x : Tensor α s) : Tensor α s := map Context.sign x

/-- Entry `i` of `sign x` is `Context.sign (x i)`. -/
@[simp] theorem sign_apply (x : Tensor α s) (i : s.Coord) : sign x i = Context.sign (x i) :=
  Tensor.map_apply _ x i

/-- `1 / x` at every coordinate (`torch.reciprocal`). -/
def reciprocal (x : Tensor α s) : Tensor α s := map Context.reciprocal x

/-- Entry `i` of `reciprocal x` is `Context.reciprocal (x i)`. -/
@[simp] theorem reciprocal_apply (x : Tensor α s) (i : s.Coord) :
    reciprocal x i = Context.reciprocal (x i) :=
  Tensor.map_apply _ x i

/-- The item of the reciprocal of a scalar tensor is `1 /` its item. -/
@[simp] theorem toScalar_reciprocal (x : Tensor α .scalar) :
    (reciprocal x).item = 1 / x.item := by
  simp [reciprocal, Context.reciprocal, Tensor.item]

/-- What `sqrt` computes at each entry. -/
inductive SqrtPolicy
  /-- `MathFunctions.sqrt x` (`torch.sqrt`): a NaN below zero in IEEE formats. -/
  | raw
  /-- `MathFunctions.sqrt (Max.max x 0)`, with the scalar's `Max` deciding zeros and NaNs: zero at
  a negative number. -/
  | clampNegative
  deriving DecidableEq, Repr, Inhabited

/-- The square root at every coordinate, by `policy`. -/
def sqrt (x : Tensor α s) (policy : SqrtPolicy := .raw) : Tensor α s :=
  match policy with
  | .raw => map MathFunctions.sqrt x
  | .clampNegative => map (fun x => MathFunctions.sqrt (Max.max x 0)) x

/-- `sqrt` at `.raw` maps the scalar root. -/
theorem sqrt_raw (x : Tensor α s) : sqrt x = map MathFunctions.sqrt x := rfl

/-- `sqrt` at `.clampNegative` maps the root of `max x 0`. -/
theorem sqrt_clampNegative (x : Tensor α s) :
    sqrt x .clampNegative = map (fun x => MathFunctions.sqrt (Max.max x 0)) x := rfl

/-- Entry `i` of `sqrt x` is `MathFunctions.sqrt (x i)`. -/
@[simp] theorem sqrt_apply (x : Tensor α s) (i : s.Coord) :
    (sqrt x) i = MathFunctions.sqrt (x i) :=
  Tensor.map_apply _ x i

/-- Entry `i` of `sqrt x .clampNegative` is the square root of `max (x i) 0`. -/
@[simp] theorem sqrt_clampNegative_apply (x : Tensor α s) (i : s.Coord) :
    sqrt x .clampNegative i = MathFunctions.sqrt (Max.max (x i) 0) :=
  Tensor.map_apply _ x i

/-- `Context.rsqrt` at every coordinate (`torch.rsqrt`), `1 / √x` with two roundings in a binary
format. -/
def rsqrt (x : Tensor α s) : Tensor α s := map Context.rsqrt x

/-- Entry `i` of `rsqrt x` is `Context.rsqrt (x i)`. -/
@[simp] theorem rsqrt_apply (x : Tensor α s) (i : s.Coord) : rsqrt x i = Context.rsqrt (x i) :=
  Tensor.map_apply _ x i

/-- `rsqrt` with pullback `-(1/2) g y³` at the result `y`, with `y³` rounded before the seed scales
it, so the pullback is infinite or a NaN where `y³` overflows and loses bits or is a zero where `y³`
underflows. -/
def rsqrtBackward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g =>
    let y := Context.rsqrt x; -((1 / 2 : Rat) : α) * g * (y * y * y)) x g

omit [Context α] in
/-- The largest integer not above each entry (`torch.floor`). -/
def floor [Rounding α] (x : Tensor α s) : Tensor α s := map Rounding.floor x

omit [Context α] in
/-- Entry `i` of `floor x` is `Rounding.floor (x i)`. -/
@[simp] theorem floor_apply [Rounding α] (x : Tensor α s) (i : s.Coord) :
    floor x i = Rounding.floor (x i) :=
  Tensor.map_apply _ x i

omit [Context α] in
/-- The smallest integer not below each entry (`torch.ceil`). -/
def ceil [Rounding α] (x : Tensor α s) : Tensor α s := map Rounding.ceil x

omit [Context α] in
/-- Entry `i` of `ceil x` is `Rounding.ceil (x i)`. -/
@[simp] theorem ceil_apply [Rounding α] (x : Tensor α s) (i : s.Coord) :
    ceil x i = Rounding.ceil (x i) :=
  Tensor.map_apply _ x i

omit [Context α] in
/-- Each entry rounded to the nearest integer, a tie to the even one (`torch.round`). -/
def round [Rounding α] (x : Tensor α s) : Tensor α s := map Rounding.round x

omit [Context α] in
/-- Entry `i` of `round x` is `Rounding.round (x i)`. -/
@[simp] theorem round_apply [Rounding α] (x : Tensor α s) (i : s.Coord) :
    round x i = Rounding.round (x i) :=
  Tensor.map_apply _ x i

omit [Context α] in
/-- Each entry rounded toward zero (`torch.trunc`). -/
def trunc [Rounding α] (x : Tensor α s) : Tensor α s := map Rounding.trunc x

omit [Context α] in
/-- Entry `i` of `trunc x` is `Rounding.trunc (x i)`. -/
@[simp] theorem trunc_apply [Rounding α] (x : Tensor α s) (i : s.Coord) :
    trunc x i = Rounding.trunc (x i) :=
  Tensor.map_apply _ x i

/-- Flooring division at every coordinate (`torch.floor_divide`, `torch.div` with
`rounding_mode='floor'`): `divFloorSchedule` over `Rounding`'s `fmod` and `floor`, the floor
of `a / b` over the reals (`TorchLean.divFloorSchedule_real`). In a binary format a nonzero finite
`a` over an infinite `b` of the other sign gives `-1`, as Python's `//` does, and an infinite
dividend over a finite divisor gives the infinite quotient. -/
def floorDivide [Rounding α] (a b : Tensor α s) : Tensor α s :=
  zipWith (divFloorSchedule Rounding.fmod Rounding.floor (Context.copysign 0)
    ((1 / 2 : Rat) : α)) a b

/-- `x - trunc x` at every coordinate (`torch.frac`). -/
def frac [Rounding α] (x : Tensor α s) : Tensor α s := map Context.frac x

/-- Entry `i` of `frac x` is `Context.frac (x i)`. -/
@[simp] theorem frac_apply [Rounding α] (x : Tensor α s) (i : s.Coord) :
    frac x i = Context.frac (x i) :=
  Tensor.map_apply _ x i

/-- `2 ^ x` at every coordinate (`torch.exp2`). -/
def exp2 (x : Tensor α s) : Tensor α s := map Context.exp2 x

/-- Entry `i` of `exp2 x` is `Context.exp2 (x i)`. -/
@[simp] theorem exp2_apply (x : Tensor α s) (i : s.Coord) : exp2 x i = Context.exp2 (x i) :=
  Tensor.map_apply _ x i

/-- `exp2` with pullback `g · 2ˣ · log 2`. -/
def exp2Backward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g => g * Context.exp2 x * MathFunctions.log (1 + 1)) x g

/-- `exp x - 1` at every coordinate (`torch.expm1`). -/
def expm1 (x : Tensor α s) : Tensor α s := map Context.expm1 x

/-- Entry `i` of `expm1 x` is `Context.expm1 (x i)`. -/
@[simp] theorem expm1_apply (x : Tensor α s) (i : s.Coord) : expm1 x i = Context.expm1 (x i) :=
  Tensor.map_apply _ x i

/-- The base-two logarithm at every coordinate (`torch.log2`). -/
def log2 (x : Tensor α s) : Tensor α s := map Context.log2 x

/-- Entry `i` of `log2 x` is `Context.log2 (x i)`. -/
@[simp] theorem log2_apply (x : Tensor α s) (i : s.Coord) : log2 x i = Context.log2 (x i) :=
  Tensor.map_apply _ x i

/-- `log2` with pullback `g / (x log 2)`, the product `x log 2` rounded first, so the pullback loses
bits where that product is subnormal. -/
def log2Backward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g => g / (x * MathFunctions.log (1 + 1))) x g

/-- The base-ten logarithm at every coordinate (`torch.log10`). -/
def log10 (x : Tensor α s) : Tensor α s := map Context.log10 x

/-- Entry `i` of `log10 x` is `Context.log10 (x i)`. -/
@[simp] theorem log10_apply (x : Tensor α s) (i : s.Coord) : log10 x i = Context.log10 (x i) :=
  Tensor.map_apply _ x i

/-- `log10` with pullback `g / (x log 10)`, the product `x log 10` rounded first, so the pullback
is a zero at a finite seed where that product overflows and loses bits where it is subnormal. -/
def log10Backward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g => g / (x * MathFunctions.log ((10 : Nat) : α))) x g

/-- `log (1 + x)` at every coordinate (`torch.log1p`). -/
def log1p (x : Tensor α s) : Tensor α s := map Context.log1p x

/-- Entry `i` of `log1p x` is `Context.log1p (x i)`. -/
@[simp] theorem log1p_apply (x : Tensor α s) (i : s.Coord) : log1p x i = Context.log1p (x i) :=
  Tensor.map_apply _ x i

/-- `log1p` with pullback `g / (1 + x)` (`Numeric.log1pBackward`), which on a supported carrier
rounds the exact quotient of finite operands once where `1 + x` or the quotient loses range. -/
def log1pBackward (x g : Tensor α s) : Tensor α s :=
  zipWith Numeric.log1pBackward x g

/-- The tangent at every coordinate, in radians (`torch.tan`). -/
def tan (x : Tensor α s) : Tensor α s := map Context.tan x

/-- Entry `i` of `tan x` is `Context.tan (x i)`. -/
@[simp] theorem tan_apply (x : Tensor α s) (i : s.Coord) : tan x i = Context.tan (x i) :=
  Tensor.map_apply _ x i

/-- `tan` with pullback `g (1 + y²)` at the result `y`. -/
def tanBackward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g => let y := Context.tan x; g * (1 + y * y)) x g

/-- The inverse hyperbolic sine at every coordinate (`torch.asinh`). -/
def asinh (x : Tensor α s) : Tensor α s := map Context.asinh x

/-- Entry `i` of `asinh x` is `Context.asinh (x i)`. -/
@[simp] theorem asinh_apply (x : Tensor α s) (i : s.Coord) : asinh x i = Context.asinh (x i) :=
  Tensor.map_apply _ x i

/-- `asinh` with pullback `g / √(x² + 1)`, the root computed as `hypot x 1`
(`Numeric.asinhBackward`). -/
def asinhBackward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g => Numeric.asinhBackward (Context.hypot x 1) g) x g

/-- The inverse hyperbolic cosine at every coordinate (`torch.acosh`). -/
def acosh (x : Tensor α s) : Tensor α s := map Context.acosh x

/-- Entry `i` of `acosh x` is `Context.acosh (x i)`. -/
@[simp] theorem acosh_apply (x : Tensor α s) (i : s.Coord) : acosh x i = Context.acosh (x i) :=
  Tensor.map_apply _ x i

/-- `acosh` with pullback `g / (√(x - 1) √(x + 1))` (`Numeric.acoshBackward`), which is
`g / √(x² - 1)` above one. -/
def acoshBackward (x g : Tensor α s) : Tensor α s :=
  zipWith Numeric.acoshBackward x g

/-- The inverse hyperbolic tangent at every coordinate (`torch.atanh`). -/
def atanh (x : Tensor α s) : Tensor α s := map Context.atanh x

/-- Entry `i` of `atanh x` is `Context.atanh (x i)`. -/
@[simp] theorem atanh_apply (x : Tensor α s) (i : s.Coord) : atanh x i = Context.atanh (x i) :=
  Tensor.map_apply _ x i

/-- `atanh` with pullback `g / ((1 - x) (1 + x))` (`Numeric.atanhBackward`). -/
def atanhBackward (x g : Tensor α s) : Tensor α s :=
  zipWith Numeric.atanhBackward x g

/-- Whether each entry is a NaN (`torch.isnan`). -/
def isNaN (x : Tensor α s) : Tensor Bool s := map Context.isNaN x

/-- Entry `i` of `isNaN x` is `Context.isNaN (x i)`. -/
@[simp] theorem isNaN_apply (x : Tensor α s) (i : s.Coord) : isNaN x i = Context.isNaN (x i) :=
  Tensor.map_apply _ x i

/-- Whether each entry is an infinity (`torch.isinf`). -/
def isInf (x : Tensor α s) : Tensor Bool s := map Context.isInf x

/-- Entry `i` of `isInf x` is `Context.isInf (x i)`. -/
@[simp] theorem isInf_apply (x : Tensor α s) (i : s.Coord) : isInf x i = Context.isInf (x i) :=
  Tensor.map_apply _ x i

/-- Whether each entry is neither a NaN nor an infinity (`torch.isfinite`). -/
def isFinite (x : Tensor α s) : Tensor Bool s := map Context.isFinite x

/-- Entry `i` of `isFinite x` is `Context.isFinite (x i)`. -/
@[simp] theorem isFinite_apply (x : Tensor α s) (i : s.Coord) :
    isFinite x i = Context.isFinite (x i) :=
  Tensor.map_apply _ x i

/-- The larger entry at every coordinate, `Context.maximum` (IEEE 754-2019 9.6 `maximum`): a NaN
where either is one, and `+0` above `-0` (`torch.maximum`). -/
def maximum (x y : Tensor α s) : Tensor α s := zipWith Context.maximum x y

/-- `maximum` is `zipWith Context.maximum`. -/
theorem maximum_def (x y : Tensor α s) : maximum x y = zipWith Context.maximum x y := rfl

/-- Entry `i` of `maximum x y` is `Context.maximum (x i) (y i)`. -/
@[simp] theorem maximum_apply (x y : Tensor α s) (i : s.Coord) :
    (maximum x y) i = Context.maximum (x i) (y i) :=
  zipWith_apply _ x y i

/-- The smaller entry at every coordinate, `Context.minimum` (IEEE 754-2019 9.6 `minimum`): a NaN
where either is one, and `-0` below `+0` (`torch.minimum`). -/
def minimum (x y : Tensor α s) : Tensor α s := zipWith Context.minimum x y

/-- `minimum` is `zipWith Context.minimum`. -/
theorem minimum_def (x y : Tensor α s) : minimum x y = zipWith Context.minimum x y := rfl

/-- Entry `i` of `minimum x y` is `Context.minimum (x i) (y i)`. -/
@[simp] theorem minimum_apply (x y : Tensor α s) (i : s.Coord) :
    (minimum x y) i = Context.minimum (x i) (y i) :=
  zipWith_apply _ x y i

/-- The larger entry at every coordinate, ignoring a NaN against a number and with `+0` above
`-0` (`torch.fmax`, IEEE 754-2019 `maximumNumber`). -/
def fmax (x y : Tensor α s) : Tensor α s := zipWith Context.fmax x y

/-- Entry `i` of `fmax x y` is `Context.fmax (x i) (y i)`. -/
@[simp] theorem fmax_apply (x y : Tensor α s) (i : s.Coord) :
    fmax x y i = Context.fmax (x i) (y i) :=
  zipWith_apply _ x y i

/-- The NaN-ignoring maximum, whose gradient goes to the operand `fmax` returns: to `a` where
`Context.fmaxTakesFirst a b`, else to `b`. PyTorch's goes to `a` where `a ≥ b` or `b` is a NaN,
which differs only at `(a, b) = (-0, +0)`. -/
def fmaxBackward (a b g : Tensor α s) : Tensor α s × Tensor α s :=
  (zipWith3 (fun a b g => if Context.fmaxTakesFirst a b then g else 0) a b g,
    zipWith3 (fun a b g => if Context.fmaxTakesFirst a b then 0 else g) a b g)

/-- The smaller entry at every coordinate, ignoring a NaN against a number and with `-0` below
`+0` (`torch.fmin`, IEEE 754-2019 `minimumNumber`). -/
def fmin (x y : Tensor α s) : Tensor α s := zipWith Context.fmin x y

/-- Entry `i` of `fmin x y` is `Context.fmin (x i) (y i)`. -/
@[simp] theorem fmin_apply (x y : Tensor α s) (i : s.Coord) :
    fmin x y i = Context.fmin (x i) (y i) :=
  zipWith_apply _ x y i

/-- The NaN-ignoring minimum, whose gradient goes to the operand `fmin` returns: to `a` where
`Context.fminTakesFirst a b`, else to `b`. PyTorch's goes to `a` where `a ≤ b` or `b` is a NaN,
which differs only at `(a, b) = (+0, -0)`. -/
def fminBackward (a b g : Tensor α s) : Tensor α s × Tensor α s :=
  (zipWith3 (fun a b g => if Context.fminTakesFirst a b then g else 0) a b g,
    zipWith3 (fun a b g => if Context.fminTakesFirst a b then 0 else g) a b g)

omit [Context α] in
/-- C's `fmod` at every coordinate, with the sign of `x` (`torch.fmod`). -/
def fmod [Rounding α] (x y : Tensor α s) : Tensor α s := zipWith Rounding.fmod x y

omit [Context α] in
/-- Entry `i` of `fmod x y` is `Rounding.fmod (x i) (y i)`. -/
@[simp] theorem fmod_apply [Rounding α] (x y : Tensor α s) (i : s.Coord) :
    fmod x y i = Rounding.fmod (x i) (y i) :=
  zipWith_apply _ x y i

/-- C's `fmod`, with pullbacks `g` and `-g n` for `n` the truncated exact quotient
`Context.fmodQuotient a b`, a NaN where `fmod a b` is one. PyTorch multiplies by the truncated
rounded quotient: by `10` where `n` is `9` at `(1, 0.1)`, and by an infinity at a nonzero `a` over
a zero `b` and at an infinite `a` over a finite `b`. -/
def fmodBackward [Rounding α] (a b g : Tensor α s) : Tensor α s × Tensor α s :=
  (g,
    zipWith3 (fun a b g => -(g * Context.fmodQuotient a b)) a b g)

/-- Python's `%` at every coordinate, with the sign of `y`, a zero included (`torch.remainder`). -/
def remainder [Rounding α] (x y : Tensor α s) : Tensor α s := zipWith Context.remainder x y

/-- Entry `i` of `remainder x y` is `Context.remainder (x i) (y i)`. -/
@[simp] theorem remainder_apply [Rounding α] (x y : Tensor α s) (i : s.Coord) :
    remainder x y i = Context.remainder (x i) (y i) :=
  zipWith_apply _ x y i

/-- Python's `%`, with pullbacks `g` and `-g q` for `q` the floored exact quotient
`Context.remainderQuotient a b`: `9` at `(1, 0.1)`, where the rounded quotient is `10`, and a NaN
where `fmod a b` is one. PyTorch's derivative multiplies by its flooring division, which is the
infinity `a / 0` at a nonzero `a` over a zero `b`. -/
def remainderBackward [Rounding α] (a b g : Tensor α s) : Tensor α s × Tensor α s :=
  (g, zipWith3 (fun a b g => -(g * Context.remainderQuotient a b)) a b g)

/-- The magnitude of `x` with the sign of `y` at every coordinate (`torch.copysign`). -/
def copysign (x y : Tensor α s) : Tensor α s := zipWith Context.copysign x y

/-- Entry `i` of `copysign x y` is `Context.copysign (x i) (y i)`. -/
@[simp] theorem copysign_apply (x y : Tensor α s) (i : s.Coord) :
    copysign x y i = Context.copysign (x i) (y i) :=
  zipWith_apply _ x y i

/-- `copysign`, whose pullback to `a` is `Context.copysignTangent a b g` (`±g`, zero at `a = 0`).
The pullback to `b` is zero, as in PyTorch. -/
def copysignBackward (a b g : Tensor α s) : Tensor α s × Tensor α s :=
  (zipWith3 Context.copysignTangent a b g, map (fun _ => (0 : α)) g)

/-- `√(x² + y²)` at every coordinate (`torch.hypot`). -/
def hypot (x y : Tensor α s) : Tensor α s := zipWith Context.hypot x y

/-- Entry `i` of `hypot x y` is `Context.hypot (x i) (y i)`. -/
@[simp] theorem hypot_apply (x y : Tensor α s) (i : s.Coord) :
    hypot x y i = Context.hypot (x i) (y i) :=
  zipWith_apply _ x y i

/-- `√(a² + b²)`, with pullbacks `g a / h` and `g b / h` at the result `h`, zero at the origin
(`Numeric.hypotBackward`), each a zero where `h` overflows and its product does not. -/
def hypotBackward (a b g : Tensor α s) : Tensor α s × Tensor α s :=
  (zipWith3 (fun a b g => Numeric.hypotBackward a (Context.hypot a b) g) a b g,
    zipWith3 (fun a b g => Numeric.hypotBackward b (Context.hypot a b) g) a b g)

/-- Whether the entries are equal (`torch.eq`). In IEEE formats a NaN equals nothing, itself
included, and `-0` equals `+0`. -/
def eq (x y : Tensor α s) : Tensor Bool s := zipWith (· == ·) x y

/-- Entry `i` of `eq x y` is `x i == y i`. -/
@[simp] theorem eq_apply (x y : Tensor α s) (i : s.Coord) :
    eq x y i = (x i == y i) :=
  zipWith_apply _ x y i

/-- Whether the entries compare unequal, `Context.neBool` at every coordinate (`torch.ne`). In IEEE
formats it holds where an entry is a NaN and fails at `-0` against `+0`. -/
def ne (x y : Tensor α s) : Tensor Bool s := zipWith Context.neBool x y

/-- Entry `i` of `ne x y` is `Context.neBool (x i) (y i)`. -/
@[simp] theorem ne_apply (x y : Tensor α s) (i : s.Coord) :
    ne x y i = Context.neBool (x i) (y i) :=
  zipWith_apply _ x y i

/-- Whether `x < y` at every coordinate (`torch.lt`). -/
def lt (x y : Tensor α s) : Tensor Bool s := zipWith Context.ltBool x y

/-- Entry `i` of `lt x y` is `Context.ltBool (x i) (y i)`. -/
@[simp] theorem lt_apply (x y : Tensor α s) (i : s.Coord) :
    lt x y i = Context.ltBool (x i) (y i) :=
  zipWith_apply _ x y i

/-- Whether `x ≤ y` at every coordinate (`torch.le`). -/
def le (x y : Tensor α s) : Tensor Bool s := zipWith Context.leBool x y

/-- Entry `i` of `le x y` is `Context.leBool (x i) (y i)`. -/
@[simp] theorem le_apply (x y : Tensor α s) (i : s.Coord) :
    le x y i = Context.leBool (x i) (y i) :=
  zipWith_apply _ x y i

/-- Whether `x > y` at every coordinate (`torch.gt`). -/
def gt (x y : Tensor α s) : Tensor Bool s := zipWith Context.gtBool x y

/-- Entry `i` of `gt x y` is `Context.gtBool (x i) (y i)`. -/
@[simp] theorem gt_apply (x y : Tensor α s) (i : s.Coord) :
    gt x y i = Context.gtBool (x i) (y i) :=
  zipWith_apply _ x y i

/-- Whether `x ≥ y` at every coordinate (`torch.ge`). -/
def ge (x y : Tensor α s) : Tensor Bool s := zipWith Context.geBool x y

/-- Entry `i` of `ge x y` is `Context.geBool (x i) (y i)`. -/
@[simp] theorem ge_apply (x y : Tensor α s) (i : s.Coord) :
    ge x y i = Context.geBool (x i) (y i) :=
  zipWith_apply _ x y i

/-- What `clamp` computes. -/
inductive ClampPolicy
  /-- `Context.clamp`, that is `minimum (maximum x lo) hi` with the IEEE `maximum` and `minimum`,
  the formula `torch.clamp` documents. In IEEE formats it gives `+0` at `x = -0` against a lower
  bound `+0`, where eager PyTorch keeps `-0` for a scalar bound. -/
  | propagate
  /-- The value of `Context.clamp`, with scalar tangents cleared (`Context.stopGradient`) wherever
  `lo < x < hi` fails, endpoints and NaNs included. Dual-number execution then has the zero
  coefficient that `clampDerivative` gives there: the tangent is cleared, where `0 * t` would be a
  NaN for a nonfinite tangent `t`. -/
  | interior
  deriving DecidableEq, Repr, Inhabited

/-- Each entry limited to `[lo, hi]` with tensor bounds, by `policy`. A scalar bound is written as
`Tensor.full`. -/
def clamp (x lo hi : Tensor α s) (policy : ClampPolicy := .propagate) : Tensor α s :=
  match policy with
  | .propagate => zipWith3 Context.clamp x lo hi
  | .interior => zipWith3 (fun v lo hi =>
      let clipped := Context.clamp v lo hi
      if v > lo ∧ hi > v then clipped else Context.stopGradient clipped) x lo hi

/-- `clamp` at `.propagate` is `zipWith3 Context.clamp`. -/
theorem clamp_propagate (x lo hi : Tensor α s) : clamp x lo hi = zipWith3 Context.clamp x lo hi :=
  rfl

/-- `clamp` at `.interior` is the `zipWith3` of `Context.clamp` that clears tangents wherever
`lo < x < hi` fails. -/
theorem clamp_interior (x lo hi : Tensor α s) :
    clamp x lo hi .interior = zipWith3 (fun v lo hi =>
      let clipped := Context.clamp v lo hi
      if v > lo ∧ hi > v then clipped else Context.stopGradient clipped) x lo hi := rfl

/-- Entry `i` of `clamp x lo hi` is `Context.clamp (x i) (lo i) (hi i)`. -/
@[simp] theorem clamp_apply (x lo hi : Tensor α s) (i : s.Coord) :
    (clamp x lo hi) i = Context.clamp (x i) (lo i) (hi i) :=
  zipWith3_apply _ x lo hi i

/-- Entry `i` of `clamp x lo hi .interior` is the clamped value strictly between `lo i` and
`hi i`, and its `Context.stopGradient` elsewhere. -/
@[simp] theorem clamp_interior_apply (x lo hi : Tensor α s) (i : s.Coord) :
    clamp x lo hi .interior i =
      let clipped := Context.clamp (x i) (lo i) (hi i)
      if x i > lo i ∧ hi i > x i then clipped else Context.stopGradient clipped :=
  zipWith3_apply _ x lo hi i

/-- The derivative mask of `clamp x lo hi .interior`: one strictly inside `(lo, hi)`, zero
elsewhere. -/
def clampDerivative (x lo hi : Tensor α s) : Tensor α s :=
  zipWith3 (fun v lo hi => if v > lo ∧ v < hi then 1 else 0) x lo hi

/-- Entry `i` of `clampDerivative x lo hi` is one when `x i` lies strictly between `lo i` and
`hi i`, and zero elsewhere. -/
@[simp] theorem clampDerivative_apply (x lo hi : Tensor α s) (i : s.Coord) :
    clampDerivative x lo hi i = if x i > lo i ∧ x i < hi i then 1 else 0 :=
  zipWith3_apply _ x lo hi i

/-- ATen's linear interpolation from `a` to `b` with weights `w` (`torch.lerp`). -/
def lerp (a b w : Tensor α s) : Tensor α s := zipWith3 Context.lerp a b w

/-- Entry `i` of `lerp a b w` is `Context.lerp (a i) (b i) (w i)`. -/
@[simp] theorem lerp_apply (a b w : Tensor α s) (i : s.Coord) :
    lerp a b w i = Context.lerp (a i) (b i) (w i) :=
  zipWith3_apply _ a b w i

/-- `lerp a b w`, with pullbacks `g (1 - w)`, `g w` and `g (b - a)`, the last infinite or a NaN
where `b - a` overflows. -/
def lerpBackward (a b w g : Tensor α s) : Tensor α s × Tensor α s × Tensor α s :=
  (zipWith (fun w g => g * (1 - w)) w g, zipWith (fun w g => g * w) w g,
    zipWith3 (fun a b g => g * (b - a)) a b g)

/-- `x + value * a * b` at every coordinate (`torch.addcmul`). -/
def addcmul (x a b : Tensor α s) (value : α) : Tensor α s :=
  zipWith3 (fun u v w => Context.addcmul u v w value) x a b

/-- Entry `i` of `addcmul x a b value` is `Context.addcmul (x i) (a i) (b i) value`. -/
@[simp] theorem addcmul_apply (x a b : Tensor α s) (value : α) (i : s.Coord) :
    addcmul x a b value i = Context.addcmul (x i) (a i) (b i) value :=
  zipWith3_apply _ x a b i

/-- `addcmul x a b value`, with pullbacks `g`, `g b value` and `g a value`. -/
def addcmulBackward (a b : Tensor α s) (value : α) (g : Tensor α s) :
    Tensor α s × Tensor α s × Tensor α s :=
  (g, zipWith (fun b g => g * b * value) b g, zipWith (fun a g => g * a * value) a g)

/-- `x + value * (a / b)` at every coordinate (`torch.addcdiv`). -/
def addcdiv (x a b : Tensor α s) (value : α) : Tensor α s :=
  zipWith3 (fun u v w => Context.addcdiv u v w value) x a b

/-- Entry `i` of `addcdiv x a b value` is `Context.addcdiv (x i) (a i) (b i) value`. -/
@[simp] theorem addcdiv_apply (x a b : Tensor α s) (value : α) (i : s.Coord) :
    addcdiv x a b value i = Context.addcdiv (x i) (a i) (b i) value :=
  zipWith3_apply _ x a b i

/-- `addcdiv x a b value`, with pullbacks `g`, `g value / b` and `-g value a / b²`. The last is
`div`'s pullback to its divisor at the seed `g value`, which overflows or underflows where
`g value`, `a / b`, `(a / b) / b` or the result does. -/
def addcdivBackward (a b : Tensor α s) (value : α) (g : Tensor α s) :
    Tensor α s × Tensor α s × Tensor α s :=
  (g, zipWith (fun b g => g * value / b) b g,
    zipWith3 (fun a b g => -(g * value * ((a / b) / b))) a b g)

/-- Each NaN replaced by `nan` and each infinity by `posinf` or `neginf` (`torch.nan_to_num`). -/
def nanToNum (x : Tensor α s) (nan posinf neginf : α) : Tensor α s :=
  map (fun v => Context.nanToNum v nan posinf neginf) x

/-- Entry `i` of `nanToNum x nan posinf neginf` is `Context.nanToNum (x i) nan posinf neginf`. -/
@[simp] theorem nanToNum_apply (x : Tensor α s) (nan posinf neginf : α) (i : s.Coord) :
    nanToNum x nan posinf neginf i = Context.nanToNum (x i) nan posinf neginf :=
  Tensor.map_apply _ x i

/-- `nanToNum` with pullback `g` at a finite entry and `0` elsewhere, as PyTorch's. -/
def nanToNumBackward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g => if Context.isFinite x then g else 0) x g

section Angle

variable [InverseTrig α]

omit [Context α] in
/-- The polar angle of `(x, y)` at every coordinate, in `(-π, π]` over ℝ (`torch.atan2`). -/
def atan2 (y x : Tensor α s) : Tensor α s := zipWith InverseTrig.atan2 y x

omit [Context α] in
/-- Entry `i` of `atan2 y x` is `InverseTrig.atan2 (y i) (x i)`. -/
@[simp] theorem atan2_apply (y x : Tensor α s) (i : s.Coord) :
    atan2 y x i = InverseTrig.atan2 (y i) (x i) :=
  zipWith_apply _ y x i

omit [InverseTrig α] in
/-- `atan2 y x`, with pullbacks `g x / (y² + x²)` to `y` and `-g y / (y² + x²)` to `x`, both from
`Numeric.angleTangent`, which gives zero slopes at the origin and at an infinite
coordinate. -/
def atan2Backward (y x g : Tensor α s) : Tensor α s × Tensor α s :=
  (zipWith3 (fun y x g => Numeric.angleTangent x y 0 g) y x g,
    zipWith3 (fun y x g => Numeric.angleTangent x y g 0) y x g)

omit [Context α] in
/-- The arctangent at every coordinate (`torch.atan`). -/
def atan (x : Tensor α s) : Tensor α s := map InverseTrig.atan x

omit [Context α] in
/-- Entry `i` of `atan x` is `InverseTrig.atan (x i)`. -/
@[simp] theorem atan_apply (x : Tensor α s) (i : s.Coord) : atan x i = InverseTrig.atan (x i) :=
  Tensor.map_apply _ x i

omit [InverseTrig α] in
/-- `atan` with pullback `g / (x² + 1)`, by `Numeric.atanBackward`. -/
def atanBackward (x g : Tensor α s) : Tensor α s :=
  zipWith Numeric.atanBackward x g

omit [Context α] in
/-- The arcsine at every coordinate (`torch.asin`). -/
def asin (x : Tensor α s) : Tensor α s := map InverseTrig.asin x

omit [Context α] in
/-- Entry `i` of `asin x` is `InverseTrig.asin (x i)`. -/
@[simp] theorem asin_apply (x : Tensor α s) (i : s.Coord) : asin x i = InverseTrig.asin (x i) :=
  Tensor.map_apply _ x i

omit [InverseTrig α] in
/-- `asin` with pullback `g / √(1 - x²)`. -/
def asinBackward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g => g / MathFunctions.sqrt ((1 - x) * (1 + x))) x g

omit [Context α] in
/-- The arccosine at every coordinate (`torch.acos`). -/
def acos (x : Tensor α s) : Tensor α s := map InverseTrig.acos x

omit [Context α] in
/-- Entry `i` of `acos x` is `InverseTrig.acos (x i)`. -/
@[simp] theorem acos_apply (x : Tensor α s) (i : s.Coord) : acos x i = InverseTrig.acos (x i) :=
  Tensor.map_apply _ x i

omit [InverseTrig α] in
/-- `acos` with pullback `-g / √(1 - x²)`. -/
def acosBackward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g => -(g / MathFunctions.sqrt ((1 - x) * (1 + x)))) x g

end Angle

omit [Context α] in
/-- The error function at every coordinate (`torch.erf`). -/
def erf [Erf α] (x : Tensor α s) : Tensor α s := map Erf.erf x

omit [Context α] in
/-- Entry `i` of `erf x` is `Erf.erf (x i)`. -/
@[simp] theorem erf_apply [Erf α] (x : Tensor α s) (i : s.Coord) : erf x i = Erf.erf (x i) :=
  Tensor.map_apply _ x i

/-- `erf` with pullback `2 / √π · exp (-x²) · g`. -/
def erfBackward (x g : Tensor α s) : Tensor α s :=
  zipWith (fun x g =>
    (1 + 1) / MathFunctions.sqrt (MathFunctions.pi : α) * MathFunctions.exp (-(x * x)) * g)
    x g

/-- The exact GELU at every coordinate (`torch.nn.functional.gelu` with its default
`approximate='none'`): `Context.geluErf` with the scalar's error function. -/
def gelu [Erf α] (x : Tensor α s) : Tensor α s := map (Context.geluErf (m := Id) Erf.erf) x

/-- `gelu` with pullback `g * (cdf + x * pdf)`, where `cdf = ½ (1 + erf (x * √½))`,
`pdf = β * exp ((x * x) * -½)` and `β = 2 / √π * √½ * ½`: the tensor expression of PyTorch's
decomposition (`torch._decomp`), with `β` and `√½` evaluated by the scalar instead of taken as
precomputed constants, so it is not PyTorch's floating-point schedule (nor its fused CPU kernel).
At `±∞` the slope is its limit, `1` or `-0`, where `x * pdf` is `±∞ · 0`. -/
def geluBackward [Erf α] (x g : Tensor α s) : Tensor α s :=
  let half : α := ((1 / 2 : Rat) : α)
  let beta : α :=
    (1 + 1) / MathFunctions.sqrt (MathFunctions.pi : α) * MathFunctions.sqrt half * half
  zipWith (fun x g =>
    let cdf := half * (1 + Erf.erf (x * MathFunctions.sqrt half))
    let pdf := beta * MathFunctions.exp ((x * x) * -half)
    g * (if Context.isInf x then (if x > 0 then 1 else -0) else cdf + x * pdf)) x g

end Tensor

end TorchLean
