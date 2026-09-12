/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32.Compare

/-!
Executable binary32 transcendental approximations.

The functions in this file provide deterministic `exp`, `log`, and related operations for the
IEEE32 executable model. Stronger libm-style correctness claims live outside this layer.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754
namespace IEEE32Exec

namespace Transcendentals

/-- Fixed-point scale (in bits) used by the integer-only `exp`/`log` approximations. -/
def fixedScale : Nat := 48

/-- `fixedScale` as an `Int`. -/
def fixedScaleInt : Int := Int.ofNat fixedScale

/-- Fixed-point encoding of $1$ at scale `fixedScale` (that is, $2^{\mathtt{fixedScale}}$). -/
def fixedOne : Int := Int.ofNat (pow2 fixedScale)

/-- Integer power of two: $\mathtt{pow2Int}(k)=2^k$, represented as an `Int`. -/
def pow2Int (k : Nat) : Int := Int.ofNat (pow2 k)

/--
Round an integer quotient $\mathtt{num}/\mathtt{den}$ to the nearest integer, ties-to-even.

Assumes $\mathtt{den}>0$.
-/
def roundQuotEvenInt (num den : Int) : Int :=
  -- Round `num/den` to nearest, ties-to-even (assumes `den > 0`).
  let q := Int.ediv num den
  let r := Int.emod num den
  let twice := 2 * r
  if twice < den then q
  else if twice > den then q + 1
  else
    if q % 2 == 0 then q else q + 1

/-- Divide by $2^{\mathtt{shift}}$, rounding to nearest with ties-to-even. -/
def roundDivPow2EvenInt (n : Int) (shift : Nat) : Int :=
  if shift == 0 then n else roundQuotEvenInt n (pow2Int shift)

/--
Shift by a power of two: multiply when $k\ge 0$, divide when $k<0$.

Division uses ties-to-even rounding.
-/
def shiftPow2EvenInt (n : Int) (k : Int) : Int :=
  match k with
  | .ofNat sh => n * pow2Int sh
  | .negSucc sh => roundDivPow2EvenInt n (sh + 1)

/-- Fixed-point multiplication at scale `fixedScale` (ties-to-even). -/
def fixedMul (a b : Int) : Int :=
  roundDivPow2EvenInt (a * b) fixedScale

/--
Fixed-point division at scale `fixedScale` (ties-to-even).

If `a` and `b` are fixed-point at scale `fixedScale`, the result is at the same scale.
-/
def fixedDiv (a b : Int) : Int :=
  -- `a` and `b` are fixedpoint at scale `fixedScale`; result is fixedpoint at the same scale.
  roundQuotEvenInt (a * fixedOne) b

/-- Divide by a natural number, rounding to nearest with ties-to-even. -/
def fixedDivByNat (a : Int) (n : Nat) : Int :=
  roundQuotEvenInt a (Int.ofNat n)

/-- Convert a dyadic number to a signed fixed-point integer at scale `fixedScale`. -/
def fixedOfDyadic (d : Dyadic) : Int :=
  let signedMant : Int := if d.sign then -(Int.ofNat d.mant) else (Int.ofNat d.mant)
  shiftPow2EvenInt signedMant (d.exp + fixedScaleInt)

/-- Convert a signed fixed-point integer at scale `fixedScale` to a dyadic number. -/
def fixedToDyadic (x : Int) : Dyadic :=
  { sign := x < 0, mant := Int.natAbs x, exp := -fixedScaleInt }

/-- Fixed-point approximation to $\log 2$ at scale `fixedScale`. -/
def fixedLn2 : Int := 195103586505167     -- round(ln2 * 2^48)
/-- Fixed-point approximation to $1/\log 2$ at scale `fixedScale`. -/
def fixedInvLn2 : Int := 406082553034800  -- round((1/ln2) * 2^48)

-- Coefficients for `2^x` on `[-1/2, 1/2]` using the Taylor series:
--   2^x = Σ (ln 2)^n / n! * x^n
-- Each coefficient is rounded to scale `2^48`.
/-- Fixed-point Taylor coefficients (highest degree first) for $2^x$ on
$[-\tfrac12,\tfrac12]$. -/
def exp2PolyCoeffsDesc : Array Int :=
  #[1985781
  , 28648765
  , 371982884
  , 4293262892
  , 43357083587
  , 375306296874
  , 2707262666570
  , 15623017693776
  , 67617750451595
  , 195103586505167
  , 281474976710656
  ]

/-- Evaluate the fixed-point $2^x$ polynomial approximation using Horner’s method. -/
def evalExp2Poly (xFixed : Int) : Int :=
  match exp2PolyCoeffsDesc[0]? with
  | none => fixedOne
  | some c0 =>
      -- Horner: p = cN; p = c + x*p
      (exp2PolyCoeffsDesc.extract 1 exp2PolyCoeffsDesc.size).foldl
        (fun p c => c + fixedMul xFixed p) c0

end Transcendentals

/-- Deterministic `exp` (no delegation to `Float`): range-reduced $2^{x/\log 2}$ with a
fixed-point polynomial. -/
def exp (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then quietNaN x
  else if isInf x then
    if signBit x then posZero else posInf
  else
    match toDyadic? x with
    | none => canonicalNaN
    | some dx =>
        let xFixed := Transcendentals.fixedOfDyadic dx
        let yFixed := Transcendentals.fixedMul xFixed Transcendentals.fixedInvLn2
        -- k = round(y), f = y - k in [-1/2, 1/2].
        let k : Int := Transcendentals.roundDivPow2EvenInt yFixed Transcendentals.fixedScale
        let fFixed : Int := yFixed - k * Transcendentals.pow2Int Transcendentals.fixedScale
        let pFixed : Int := Transcendentals.evalExp2Poly fFixed
        if pFixed ≤ 0 then
          posZero
        else
          roundDyadicToIEEE32
            { sign := false
              , mant := Int.natAbs pFixed
              , exp := k - Transcendentals.fixedScaleInt }

/-- Deterministic `log` (no delegation to `Float`): normalize $x=m2^k$ and use an
atanh-series for $\log m$. -/
def log (x : IEEE32Exec) : IEEE32Exec :=
  if isNaN x then quietNaN x
  else if isInf x then
    if signBit x then canonicalNaN else posInf
  else if isZero x then
    negInf
  else if signBit x then
    -- We follow common libm behavior: log(negative) = NaN (including log(-0) already handled
    -- above).
    canonicalNaN
  else
    match toDyadic? x with
    | none => canonicalNaN
    | some dx =>
        if dx.mant == 0 then
          negInf
        else
          let k : Int := (Int.ofNat (Nat.log2 dx.mant)) + dx.exp
          -- m = x / 2^k ∈ [1,2)
          let mFixed : Int :=
            Transcendentals.fixedOfDyadic { sign := false, mant := dx.mant, exp := dx.exp - k }
          let u : Int := mFixed - Transcendentals.fixedOne
          let v : Int := mFixed + Transcendentals.fixedOne
          let t : Int := Transcendentals.fixedDiv u v
          let t2 : Int := Transcendentals.fixedMul t t
          -- log(m) = 2 * (t + t^3/3 + t^5/5 + ...), convergent for m ∈ [1,2).
          let term3 : Int := Transcendentals.fixedMul t t2
          let term5 : Int := Transcendentals.fixedMul term3 t2
          let term7 : Int := Transcendentals.fixedMul term5 t2
          let term9 : Int := Transcendentals.fixedMul term7 t2
          let term11 : Int := Transcendentals.fixedMul term9 t2
          let term13 : Int := Transcendentals.fixedMul term11 t2
          let term15 : Int := Transcendentals.fixedMul term13 t2
          let sum : Int :=
            t
            + Transcendentals.fixedDivByNat term3 3
            + Transcendentals.fixedDivByNat term5 5
            + Transcendentals.fixedDivByNat term7 7
            + Transcendentals.fixedDivByNat term9 9
            + Transcendentals.fixedDivByNat term11 11
            + Transcendentals.fixedDivByNat term13 13
            + Transcendentals.fixedDivByNat term15 15
          let logmFixed : Int := 2 * sum
          let kLn2Fixed : Int := k * Transcendentals.fixedLn2
          let logxFixed : Int := logmFixed + kLn2Fixed
          roundDyadicToIEEE32 (Transcendentals.fixedToDyadic logxFixed)

/-- Deterministic `sinh` (no delegation to `Float`): defined via `exp`. -/
def sinh (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        x
      else
        -- For small `|x|`, `0.5*(exp(x)-exp(-x))` suffers cancellation. A short Taylor polynomial
        -- is both deterministic and more accurate near 0.
        let ax := abs x
        let half : IEEE32Exec := ofBits 0x3F000000
        match compare ax half with
        | some .lt =>
            -- `sinh x ≈ x + x^3/3! + x^5/5!` for `|x| < 0.5`.
            let x2 := mul x x
            let x3 := mul x2 x
            let x5 := mul x3 x2
            let six : IEEE32Exec := ofBits 0x40C00000      -- 6.0
            let oneTwenty : IEEE32Exec := ofBits 0x42F00000 -- 120.0
            add (add x (div x3 six)) (div x5 oneTwenty)
        | _ =>
            mul (sub (exp x) (exp (neg x))) half

/-- Deterministic `cosh` (no delegation to `Float`): defined via `exp`. -/
def cosh (x : IEEE32Exec) : IEEE32Exec :=
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        posInf
      else
        let half : IEEE32Exec := ofBits 0x3F000000
        mul (add (exp x) (exp (neg x))) half

/--
Deterministic `tanh` without host `Float` delegation.

For $|x|\le\tfrac14$, the degree-nine Taylor polynomial

$$
\frac{x\left(2835-945x^2+378x^4-153x^6+62x^8\right)}{2835}
$$

is evaluated as one exact dyadic rational and rounded only at the end. This avoids cancellation and
preserves every tiny binary32 input, including the minimum subnormal. Larger finite inputs use the
bounded identity `1 - 2/(exp(2|x|)+1)`. The outer branch is clamped to the shared boundary value
because the two independently rounded approximations can otherwise reverse adjacent outputs at the
switch.
-/
def tanh (x : IEEE32Exec) : IEEE32Exec :=
  let tanhTaylorSmall (x : IEEE32Exec) : IEEE32Exec :=
    match toDyadic? x with
    | none => canonicalNaN
    | some dx =>
        let multiply (a b : Dyadic) : Dyadic :=
          { sign := Bool.xor a.sign b.sign
            mant := a.mant * b.mant
            exp := a.exp + b.exp }
        let ofInt (c : Int) : Dyadic :=
          { sign := c < 0, mant := c.natAbs, exp := 0 }
        let z := multiply dx dx
        -- Horner evaluation of `62*z⁴ - 153*z³ + 378*z² - 945*z + 2835`.
        let p := addDyadic (multiply (ofInt 62) z) (ofInt (-153))
        let p := addDyadic (multiply p z) (ofInt 378)
        let p := addDyadic (multiply p z) (ofInt (-945))
        let p := addDyadic (multiply p z) (ofInt 2835)
        let numerator := multiply dx p
        if numerator.mant == 0 then
          if numerator.sign then negZero else posZero
        else
          match numerator.exp with
          | .ofNat shift =>
              roundRatToIEEE32 numerator.sign (Nat.shiftLeft numerator.mant shift) 2835
          | .negSucc shift =>
              roundRatToIEEE32 numerator.sign numerator.mant (2835 * pow2 (shift + 1))
  match chooseNaN1 x with
  | some nan => nan
  | none =>
      if isInf x then
        if signBit x then ofBits 0xBF800000 else ofBits 0x3F800000
      else if isZero x then
        x
      else
        let one : IEEE32Exec := ofBits 0x3F800000
        let two : IEEE32Exec := ofBits 0x40000000
        let ax : IEEE32Exec := abs x
        let quarter : IEEE32Exec := ofBits 0x3E800000
        match compare ax quarter with
        | some .lt =>
            tanhTaylorSmall x
        | some .eq =>
            tanhTaylorSmall x
        | _ =>
            let s : Bool := signBit x
            let e : IEEE32Exec := exp (add ax ax)
            let raw : IEEE32Exec := sub one (div two (add e one))
            -- Nearest binary32 value of the once-rounded Taylor approximation at `1/4`.
            let tanhAtQuarter : IEEE32Exec := ofBits 0x3E7ACBF5
            let tpos : IEEE32Exec :=
              match compare raw tanhAtQuarter with
              | some .lt => tanhAtQuarter
              | _ => raw
            if s then neg tpos else tpos

namespace Trig

/-!
Deterministic `sin`/`cos`
========================

Unlike `exp`/`log`, `sin` and `cos` are used by the runtime FFT layer (`NN.Runtime.*.Fft`) to build
twiddle factors. Delegating to the host `Float` implementation makes results platform-dependent.

We implement `sin`/`cos` purely inside Lean:

1. reduce the exact binary32 input by a 256-bit fixed-point approximation of `pi/2`, obtaining a
   quadrant and a remainder in approximately `[-pi/4, pi/4]`,
2. approximate the sine and cosine of that remainder by exact Taylor partial sums (degree 13 / 12),
3. restore the quadrant using exact sign changes and swaps.

This is **deterministic** and uses only the IEEE32Exec kernel ops (`roundRatToIEEE32`, `add/mul/sub`,
etc.). We do not claim correctly-rounded libm behavior; reproducible execution is the contract.
-/

/-- Exact dyadic multiplication: xor the signs, multiply the mantissas, add the exponents.

No rounding happens here, which is the point: the series evaluations below stay exact until a single
final `roundRatToIEEE32`. -/
@[inline] def mulDyadic (a b : Dyadic) : Dyadic :=
  { sign := Bool.xor a.sign b.sign, mant := a.mant * b.mant, exp := a.exp + b.exp }

/-- Dyadic `1`. -/
@[inline] def oneDyadic : Dyadic :=
  { sign := false, mant := 1, exp := 0 }

/--
Round the exact rational $d/\mathtt{den}$ to binary32, where $d$ is the exact dyadic
$\mathtt{mant}\,2^{\mathtt{exp}}$.

We package the dyadic exponent into a rational numerator/denominator and call `roundRatToIEEE32`.
-/
def roundDyadicDivNat (d : Dyadic) (den : Nat) : IEEE32Exec :=
  if d.mant == 0 then
    if d.sign then negZero else posZero
  else
    match d.exp with
    | .ofNat sh =>
        roundRatToIEEE32 d.sign (Nat.shiftLeft d.mant sh) den
    | .negSucc sh =>
        roundRatToIEEE32 d.sign d.mant (den * pow2 (sh + 1))

/-!
### Taylor partial sums on `|y| < 1/2`

We encode the partial sums using a common factorial denominator so the coefficients are *exact*
integers, not approximations.

For $z=y^2$:

* $\displaystyle
  \sin y=\sum_{i=0}^{6}\frac{(-1)^i y^{2i+1}}{(2i+1)!}+R_7(y)$,
  where the polynomial part can be written as
  $\displaystyle
  \frac{y}{13!}\sum_{i=0}^{6}(-1)^i\frac{13!}{(2i+1)!}z^i$.

* $\displaystyle
  \cos y=\sum_{i=0}^{6}\frac{(-1)^i y^{2i}}{(2i)!}+R'_7(y)$,
  whose polynomial part is
  $\displaystyle
  \frac1{12!}\sum_{i=0}^{6}(-1)^i\frac{12!}{(2i)!}z^i$.
-/

/-- Common denominator `13!` of the sine polynomial, so all its coefficients are integers. -/
def sinDen : Nat := 6227020800   -- 13!
/-- Common denominator `12!` of the cosine polynomial. -/
def cosDen : Nat := 479001600    -- 12!

/-- Integer coefficient `(-1)^i * 13! / (2i+1)!` of the scaled sine polynomial in `z = y²`.

Written out as literals rather than computed from factorials so that the kernel never has to
evaluate a factorial during rounding, and so the values are auditable by eye. -/
def sinCoeff (i : Nat) : Int :=
  match i with
  | 0 => 6227020800
  | 1 => -1037836800
  | 2 => 51891840
  | 3 => -1235520
  | 4 => 17160
  | 5 => -156
  | _ => 1

/-- Integer coefficient `(-1)^i * 12! / (2i)!` of the scaled cosine polynomial in `z = y²`. -/
def cosCoeff (i : Nat) : Int :=
  match i with
  | 0 => 479001600
  | 1 => -239500800
  | 2 => 19958400
  | 3 => -665280
  | 4 => 11880
  | 5 => -132
  | _ => 1

/-- View an integer coefficient as an exact dyadic with exponent zero. -/
def coeffToDyadic (c : Int) : Dyadic :=
  if c < 0 then
    { sign := true, mant := Int.natAbs c, exp := 0 }
  else
    { sign := false, mant := Int.natAbs c, exp := 0 }

/-- Evaluate `Σ_{i<7} coeff i * z ^ i` exactly over the dyadics.

Every step is an exact dyadic multiply-add, so the only rounding in the whole sine and cosine path
happens once at the end, in the final division by `13!` or `12!`. That is what keeps the error
analysis to a single rounding plus the Taylor remainder. -/
def evalPolyNumerator (coeff : Nat → Int) (z : Dyadic) : Dyadic :=
  -- Degree 6 polynomial: Σ_{i=0..6} coeff(i) * z^i, evaluated by successive multiplication.
  let step (st : Dyadic × Dyadic) (i : Nat) : Dyadic × Dyadic :=
    let pow := st.1
    let acc := st.2
    let term := mulDyadic (coeffToDyadic (coeff i)) pow
    let acc' := addDyadic acc term
    let pow' := mulDyadic pow z
    (pow', acc')
  -- Start with `pow = 1`, `acc = 0`, and fold over `i = 0..6`.
  let acc0 : Dyadic := { sign := false, mant := 0, exp := 0 }
  ((List.range 7).foldl step (oneDyadic, acc0)).2

/-- Sine and cosine of a reduced argument `|y| ≤ π/4`, as a pair of binary32 values.

Both polynomials use the same exact square `z = y²`. Each polynomial is evaluated separately,
and each result is rounded once to binary32. -/
def sinCosTaylorSmall (y : Dyadic) : IEEE32Exec × IEEE32Exec :=
  -- z = y^2
  let z : Dyadic :=
    { sign := false, mant := y.mant * y.mant, exp := y.exp + y.exp }
  let sinNum : Dyadic := evalPolyNumerator sinCoeff z
  let cosNum : Dyadic := evalPolyNumerator cosCoeff z
  let sinDy : Dyadic := mulDyadic y sinNum
  let s : IEEE32Exec := roundDyadicDivNat sinDy sinDen
  let c : IEEE32Exec := roundDyadicDivNat cosNum cosDen
  (s, c)

/-- Precision used for trigonometric argument reduction. It exceeds the binary32 exponent range. -/
def reductionScale : Nat := 256

/-- `reductionScale` as an integer. -/
def reductionScaleInt : Int := Int.ofNat reductionScale

/-- $\operatorname{round}((\pi/2)2^{256})$, used for deterministic Payne–Hanek-style quadrant
reduction. -/
def halfPiFixed : Int :=
  181885788445883162140117471388864931326696688664196214979386075558969447233093

/-- Exact conversion of any binary32 dyadic to the trigonometric fixed-point scale. -/
def reductionFixedOfDyadic (d : Dyadic) : Int :=
  let signedMant : Int := if d.sign then -(Int.ofNat d.mant) else Int.ofNat d.mant
  Transcendentals.shiftPow2EvenInt signedMant (d.exp + reductionScaleInt)

/-- Convert a signed trigonometric fixed-point remainder back to a dyadic. -/
def reductionFixedToDyadic (x : Int) : Dyadic :=
  { sign := x < 0, mant := Int.natAbs x, exp := -reductionScaleInt }

/-- Reduce an exact finite binary32 dyadic to one quadrant and evaluate the small-angle kernels. -/
def sinCosScaled (dx : Dyadic) : IEEE32Exec × IEEE32Exec :=
  let xFixed := reductionFixedOfDyadic dx
  let quadrant := Transcendentals.roundQuotEvenInt xFixed halfPiFixed
  let remainder := xFixed - quadrant * halfPiFixed
  let sc := sinCosTaylorSmall (reductionFixedToDyadic remainder)
  match quadrant % 4 with
  | 0 => sc
  | 1 => (sc.2, neg sc.1)
  | 2 => (neg sc.1, neg sc.2)
  | _ => (neg sc.2, sc.1)

/-- Joint deterministic `sin`/`cos` computation for `IEEE32Exec`. -/
def sinCos (x : IEEE32Exec) : IEEE32Exec × IEEE32Exec :=
  if isNaN x then
    let q := quietNaN x
    (q, q)
  else if isInf x then
    (canonicalNaN, canonicalNaN)
  else if isZero x then
    -- Preserve signed zero for `sin` and return `cos(±0) = 1`.
    (x, ofBits 0x3F800000)
  else
    match toDyadic? x with
    | none => (canonicalNaN, canonicalNaN)
    | some dx => sinCosScaled dx

/-- Deterministic `sin` implementation (shared core via `sinCos`). -/
def sin (x : IEEE32Exec) : IEEE32Exec :=
  (sinCos x).1

/-- Deterministic `cos` implementation (shared core via `sinCos`). -/
def cos (x : IEEE32Exec) : IEEE32Exec :=
  (sinCos x).2

end Trig

/-!
### Public `sin` / `cos`

We expose `sin`/`cos` as executable ops on `IEEE32Exec` using the deterministic implementation
above, together with standard IEEE special-case conventions.
-/

/-- Deterministic `sin` for `IEEE32Exec`. -/
def sin (x : IEEE32Exec) : IEEE32Exec :=
  Trig.sin x

/-- Deterministic `cos` for `IEEE32Exec`. -/
def cos (x : IEEE32Exec) : IEEE32Exec :=
  Trig.cos x


end IEEE32Exec

end TorchLean.Floats.IEEE754
