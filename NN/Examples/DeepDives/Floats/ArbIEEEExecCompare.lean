/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Interval.Arb
public import NN.Floats.Interval.Comparison
public import NN.API.CLI
public import Std
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Transcendentals
/-!
# Arb vs configured binary32 interval tutorial
This tutorial prints side-by-side enclosures for a few unary functions over a one-dimensional input
interval:
- **Arb** (`python-flint` / Arb ball arithmetic): rigorous real enclosures at chosen precision.
- **configured binary32**: executable float32 evaluation on the endpoints (not a proved
outward-rounded interval rule for transcendentals).
- **Float32 baseline**: ordinary runtime `Float32` endpoint arithmetic, included to show why
  directed rounding matters.
- **Rational baseline**: exact `Rat` interval arithmetic for small polynomial/reference checks.
NumPy / PyTorch analogue:
```python
import numpy as np

lo, hi = np.float32(-0.5), np.float32(0.5)
endpoint_box = (np.tanh(lo), np.tanh(hi))   # common fast check, not a rigorous enclosure
```

TorchLean's lesson is more explicit: endpoint evaluation is useful for debugging, but rigorous
transcendental enclosures need a trusted real enclosure source (here Arb) plus outward rounding back
to the binary32 grid.

Implementation note: the reusable baseline interval helpers live in
`NN.Floats.Interval.Comparison`; this file only chooses tutorial cases and prints their results.

Run:

```bash
scripts/lake.sh exe torchlean floats_arb_ieee_compare
```

If Arb is not installed, the tutorial still prints the configured binary32 side and reports the Arb
failure.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofBits32 ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


open Std

namespace NN.Examples.DeepDives.Floats.ArbIEEEExecCompare

open TorchLean.Floats.Arb
open TorchLean.Floats.Interval.Comparison
open FloatLib.Numerics (RationalInterval)

/-- JSON expression for $x^2+0.1x-0.5$, in the safe Arb expression language. -/
def polynomialExpr : Lean.Json :=
  Lean.Json.mkObj [
    ("op", Lean.Json.str "sub"),
    ("args", Lean.Json.arr #[
      Lean.Json.mkObj [
        ("op", Lean.Json.str "add"),
        ("args", Lean.Json.arr #[
          Lean.Json.mkObj [
            ("op", Lean.Json.str "mul"),
            ("args", Lean.Json.arr #[Lean.Json.mkObj [("var", Lean.Json.str "x")],
              Lean.Json.mkObj [("var", Lean.Json.str "x")]])
          ],
          Lean.Json.mkObj [
            ("op", Lean.Json.str "mul"),
            ("args", Lean.Json.arr #[Lean.Json.mkObj [("const", Lean.Json.str "0.1")],
              Lean.Json.mkObj [("var", Lean.Json.str "x")]])
          ]
        ])
      ],
      Lean.Json.mkObj [("const", Lean.Json.str "0.5")]
    ])
  ]

/--
Run one tutorial comparison.

The output has four conceptual rows:

- `Arb`: rigorous real interval from the external oracle when available;
- `Binary 8 23`: endpoint evaluation using FloatLib's configured binary32 arithmetic;
- `Float32`: ordinary runtime endpoint evaluation;
- `configured binary32+Arb`: Arb real enclosure rounded outward to binary32 endpoints.
-/
def compare (func : String) (lo hi : Float) (precBits digits : Nat) : IO Unit := do
  let loF32 := Float.toFloat32 lo
  let hiF32 := Float.toFloat32 hi
  let lo32 := ofBits32 loF32.toBits
  let hi32 := ofBits32 hiF32.toBits

  IO.println s!"func={func}, x∈[{lo}, {hi}] (as Float32 bits: lo={loF32.toBits}, hi={hiF32.toBits})"

  -- Arb (rigorous real enclosure / ball enclosure).
  if func = "poly" then
    -- Use the safe `expr` request format (not the unary `--func` mode).
    let xVar := ("x", (toString lo, toString hi))
    try
      let r ← TorchLean.Floats.Arb.runExpr (vars := #[xVar]) (expr := polynomialExpr)
        (precBits := precBits) (digits := digits)
      IO.println s!"  Arb(expr):[{r.outLo}, {r.outHi}] (precBits={r.precBits})"
    catch e =>
      IO.println s!"  Arb(expr):(failed) {e.toString}"
  else
    try
      let q : Query :=
        { func := func
          lo := toString lo
          hi := toString hi
          precBits := precBits
          digits := digits }
      let r ← TorchLean.Floats.Arb.run q
      IO.println s!"  Arb:      [{r.outLo}, {r.outHi}] (precBits={r.precBits})"
    catch e =>
      IO.println s!"  Arb:      (failed) {e.toString}"

  -- configured binary32 (endpoint evaluation).
  if func = "tanh" ∨ func = "exp" ∨ func = "log" ∨ func = "sqrt" then
    /-
    PyTorch / NumPy analogue:

    ```python
    np.array([f(lo), f(hi)], dtype=np.float32)
    ```

    That is a useful consistency check, but it is not generally an interval proof for nonlinear
    functions. The Arb-backed line below is the rigorous path when Arb is installed.
    -/
    let I :=
      match func with
      | "tanh" => endpoints FloatLib.Floats.ExecFloat.Binary.tanh lo32 hi32
      | "exp"  => endpoints FloatLib.Floats.ExecFloat.Binary.exp lo32 hi32
      | "log"  => endpoints FloatLib.Floats.ExecFloat.Binary.log lo32 hi32
      | "sqrt" =>
          endpoints (Binary.sqrtWithRounding (rounding := .nearestEven)) lo32 hi32
      | _      => ⟨(Binary.canonicalNaN : Binary 8 23), (Binary.canonicalNaN : Binary 8 23)⟩
    IO.println s!"  configured binary32:{formatConfigured I}"

    -- Native runtime Float32 (endpoint evaluation).
    let If32 :=
      match func with
      | "tanh" => endpoints Float32.tanh loF32 hiF32
      | "exp"  => endpoints Float32.exp  loF32 hiF32
      | "log"  => endpoints Float32.log  loF32 hiF32
      | "sqrt" => endpoints Float32.sqrt loF32 hiF32
      | _      => ⟨Naive.posZero, Naive.posZero⟩
    IO.println s!"  Float32:  {format (formatValue Float32.toBits) If32}"

    -- configured binary32 endpoints, but with Arb-provided *rigorous* real enclosure rounded
    -- outward to float32.
    let X : Interval (Binary 8 23) := ⟨lo32, hi32⟩
    try
      let Iarb ← TorchLean.Floats.Interval.Arb.unary func X
        (precBits := precBits) (digits := digits)
      IO.println s!"  configured binary32+Arb:{formatConfigured Iarb}"

      -- Check whether endpoint-evaluation enclosures contain the Arb-rounded outward enclosure.
      match toRational? Binary.toRat? I, toRational? Binary.toRat? Iarb with
      | some Ir, some Iar =>
        IO.println
          s!"  contains(binary32 endpoints ⊇ binary32+Arb)? {Rational.contains Ir Iar}"
      | _, _ =>
        IO.println
          s!"  contains(configured binary32 endpoints ⊇ configured binary32+Arb)? (n/a: non-finite)"
      match toRational? (fun x : Float32 => Binary.toRat? (ofBits32 x.toBits)) If32,
          toRational? Binary.toRat? Iarb with
      | some Ir, some Iar =>
        IO.println
          s!"  contains(Float32 endpoints ⊇ configured binary32+Arb)? {Rational.contains Ir Iar}"
      | _, _ =>
        IO.println s!"  contains(Float32 endpoints ⊇ configured binary32+Arb)? (n/a: non-finite)"
    catch e =>
      IO.println s!"  configured binary32+Arb:(failed) {e.toString}"

  -- A small polynomial using executable directed-rounding interval arithmetic (add/mul only).
  if func = "poly" then
    /-
    NumPy analogue:

    ```python
    x = np.array([lo, hi], dtype=np.float32)
    p = x*x + np.float32(0.1)*x - np.float32(0.5)
    ```

    FloatLib uses directed rounding for each binary32 interval arithmetic step.
    Its `0.1` coefficient is rounded to binary32, while the rational and Arb rows use exact `1/10`.
    The rows therefore differ in their represented coefficient as well as their arithmetic.
    -/
    let X : Interval (Binary 8 23) := ⟨lo32, hi32⟩
    let coefficient (value : Float) : Interval (Binary 8 23) :=
      Binary.Interval.point (ofModel (Model.cast .binary64 .binary32
        (toModel (Binary.ofFloat value))))
    let c01 := coefficient 0.1
    let c05 := coefficient 0.5
    -- p(x) = x*x + 0.1*x - 0.5
    let x2 := Binary.Interval.mul X X
    let t1 := Binary.Interval.mul c01 X
    let p := Binary.Interval.sub (Binary.Interval.add x2 t1) c05
    IO.println s!"  poly(x)=x^2+0.1x-0.5: {formatConfigured p}"

    -- Real interval arithmetic baseline, using exact rationals (and exact `0.1 = 1/10`).
    let Xr? : Option RationalInterval := do
      let loR ← Binary.toRat? (ofBits32 loF32.toBits)
      let hiR ← Binary.toRat? (ofBits32 hiF32.toBits)
      pure ⟨loR, hiR⟩
    let c01r : RationalInterval := RationalInterval.point (Rat.normalize 1 10)
    let c05r : RationalInterval := RationalInterval.point (Rat.normalize 1 2)
    match Xr? with
    | none =>
      IO.println s!"  RealIA:   (n/a: non-finite input endpoints)"
    | some Xr =>
      let x2r := Rational.mul Xr Xr
      let t1r := Rational.mul c01r Xr
      let pr := RationalInterval.sub (RationalInterval.add x2r t1r) c05r
      IO.println s!"  RealIA:   {Rational.format pr}"

      -- Native Float32 interval arithmetic baseline (no directed rounding).
      let Xf : Interval Float32 := ⟨loF32, hiF32⟩
      let c01f : Interval Float32 := Interval.point (Float.toFloat32
        0.1)
      let c05f : Interval Float32 := Interval.point (Float.toFloat32
        0.5)
      let x2f := Naive.mul Xf Xf
      let t1f := Naive.mul c01f Xf
      let pf := Naive.sub (Naive.add x2f t1f) c05f
      IO.println s!"  Float32IA:{format (formatValue Float32.toBits) pf}"

      -- Containment checks against the real-interval baseline.
      match toRational? Binary.toRat? p with
      | some pR =>
        IO.println s!"  contains(configured binary32 IA ⊇ RealIA)? {Rational.contains pR pr}"
      | none =>
        IO.println s!"  contains(configured binary32 IA ⊇ RealIA)? (n/a: non-finite)"
      match toRational? (fun x : Float32 => Binary.toRat? (ofBits32 x.toBits)) pf with
      | some pR =>
        IO.println s!"  contains(Float32 IA ⊇ RealIA)? {Rational.contains pR pr}"
      | none =>
        IO.println s!"  contains(Float32 IA ⊇ RealIA)? (n/a: non-finite)"

  IO.println ""

/--
The classic round-to-nearest-even tie: `1 + 2^-24` in binary32.

The exact sum needs 25 significand bits, so it sits exactly halfway between `1` and the next float.
Round-to-nearest-even therefore returns `1`, not the next value up. Comparing the naive interval
arithmetic against the directed `addDown`/`addUp` pair here is the point: only the directed version
still encloses the exact rational sum.
-/
def tie : IO Unit := do
  let one : Float32 := Float.toFloat32 1.0
  -- Exact `2^-24` as a float32 bit pattern.
  let halfUlp : Float32 := Float32.ofBits (0x33800000 : UInt32)
  let A : Interval Float32 := ⟨one, one⟩
  let B : Interval Float32 := ⟨halfUlp, halfUlp⟩
  let sumF32 := Naive.add A B

  let one32 : Binary 8 23 := ofBits32 one.toBits
  let halfUlp32 : Binary 8 23 := ofBits32 halfUlp.toBits
  let A32 : Interval (Binary 8 23) := ⟨one32, one32⟩
  let B32 : Interval (Binary 8 23) := ⟨halfUlp32, halfUlp32⟩
  let sum32 : Interval (Binary 8 23) := Binary.Interval.add A32 B32

  IO.println "func=add_tie (round-to-nearest-even stress)"
  IO.println ("  PyTorch analogue: torch.tensor(1.0, dtype=torch.float32) "
    ++ "+ torch.tensor(2**-24, dtype=torch.float32)")
  IO.println s!"  a=[{formatValue Float32.toBits one}, {formatValue Float32.toBits one}]"
  IO.println s!"  b=[{formatValue Float32.toBits halfUlp}, {formatValue Float32.toBits halfUlp}]"
  IO.println s!"  Float32 add (naive IA): {format (formatValue Float32.toBits) sumF32}"
  IO.println s!"  configured binary32 addDown/addUp: {formatConfigured sum32}"

  -- Real reference: exact dyadic sum a + b.
  let ref? : Option RationalInterval := do
    let aR ← Binary.toRat? (ofBits32 one.toBits)
    let bR ← Binary.toRat? (ofBits32 halfUlp.toBits)
    pure <| RationalInterval.point (aR + bR)
  match ref? with
  | none =>
    IO.println s!"  Real ref: (n/a)"
  | some ref =>
    IO.println s!"  Real ref: {Rational.format ref}"
    match toRational? (fun x : Float32 => Binary.toRat? (ofBits32 x.toBits)) sumF32 with
    | some sR => IO.println s!"  contains(Float32 naive ⊇ Real ref)? {Rational.contains sR ref}"
    | none => IO.println s!"  contains(Float32 naive ⊇ Real ref)? (n/a)"
    match toRational? Binary.toRat? sum32 with
    | some sR =>
        IO.println s!"  contains(configured binary32 dir ⊇ Real ref)? {Rational.contains sR ref}"
    | none => IO.println s!"  contains(configured binary32 dir ⊇ Real ref)? (n/a)"

  IO.println ""

/--
Division by negative zero, which IEEE 754 defines as `-∞` rather than an error.

The sign of zero is observable precisely through operations like this one, which is why the float32
model keeps `+0` and `-0` distinct instead of collapsing them.
-/
def signedZero : IO Unit := do
  let one : Float32 := Float.toFloat32 1.0
  let negZ : Float32 := Naive.negZero
  let denom : Interval Float32 := ⟨negZ, negZ⟩
  let numer : Interval Float32 := ⟨one, one⟩
  let qF32 := Naive.div numer denom
  let qPoint := Float32.div one negZ

  let one32 : Binary 8 23 := ofBits32 one.toBits
  let negZ32 : Binary 8 23 := ofBits32 negZ.toBits
  let denom32 : Interval (Binary 8 23) := ⟨negZ32, negZ32⟩
  let numer32 : Interval (Binary 8 23) := ⟨one32, one32⟩
  let q32 := Binary.Interval.div numer32 denom32
  let qPoint32 := ExecFloat.div one32 negZ32
  let hostQuotient := Binary.toFloat (ofModel (Model.cast .binary32 .binary64 (toModel qPoint32)))

  IO.println "func=div_signed_zero (widening from signed-zero containment)"
  IO.println ("  PyTorch analogue: torch.tensor(1.0, dtype=torch.float32) "
    ++ "/ torch.tensor(-0.0, dtype=torch.float32)")
  IO.println s!"  point Float32: 1/(-0.0) = {formatValue Float32.toBits qPoint}"
  IO.println
    s!"  point configured binary32: 1/(-0.0) = {hostQuotient} \
      (bits={Binary.toBits32 qPoint32})"
  IO.println s!"  Float32 IA div: {format (formatValue Float32.toBits) qF32}"
  IO.println s!"  configured binary32 IA div: {formatConfigured q32}"
  IO.println ""

/-- Run a small fixed set of comparisons (unary funcs + a polynomial + some edge cases). -/
def run : IO UInt32 := do
  let precBits := 200
  let digits := 50

  IO.println "Arb vs configured binary32 comparison (unary funcs + one polynomial)"
  IO.println "(Arb is rigorous; configured binary32 is endpoint evaluation for transcendentals.)"
  IO.println ""

  -- Choose dyadic-friendly endpoints so the float32 endpoints are exact.
  compare "tanh" (-0.5) 0.5 precBits digits
  compare "exp" (-1.0)  1.0 precBits digits
  compare "exp" 80.0   90.0 precBits digits
  compare "log"  0.5   2.0 precBits digits
  compare "sqrt" 0.0   2.0 precBits digits
  compare "poly" (-0.5) 0.5 precBits digits
  tie
  signedZero
  pure 0

/-- Command-line help for the Arb-vs-IEEE32 interval tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean Arb vs configured binary32 interval tutorial"
    , ""
    , "Usage:"
    , "  scripts/lake.sh exe torchlean floats_arb_ieee_compare"
    , ""
    , "This command runs a fixed set of interval comparisons. It has no tutorial-specific flags."
    ]

/-- Entrypoint: run the Arb-vs-`Binary 8 23` interval tutorial. -/
def main (args : List String) : IO UInt32 := do
  let args := TorchLean.CLI.dropDashDash args
  if TorchLean.CLI.hasHelp args then
    IO.println usage
    return 0
  TorchLean.CLI.requireNoArgs "floats_arb_ieee_compare" args
  run

end NN.Examples.DeepDives.Floats.ArbIEEEExecCompare
