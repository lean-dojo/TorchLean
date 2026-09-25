/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Interval.IEEEExec32ArbTrans
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
outward-rounded
  interval rule for transcendentals).
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
lake exe torchlean floats_arb_ieee_compare
```

If Arb is not installed, the tutorial still prints the configured binary32 side and reports the Arb
failure.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open TorchLean.Floats.IEEE754
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
def runOne (func : String) (lo hi : Float) (precBits digits : Nat) : IO Unit := do
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
      | "tanh" => intervalUnaryEndpoints FloatLib.Floats.ExecFloat.Binary.tanh lo32 hi32
      | "exp"  => intervalUnaryEndpoints FloatLib.Floats.ExecFloat.Binary.exp lo32 hi32
      | "log"  => intervalUnaryEndpoints FloatLib.Floats.ExecFloat.Binary.log lo32 hi32
      | "sqrt" => intervalUnaryEndpoints (Binary.sqrt (rounding := .nearestEven)) lo32 hi32
      | _      => ⟨(Binary.canonicalNaN : Binary 8 23), (Binary.canonicalNaN : Binary 8 23)⟩
    IO.println s!"  configured binary32:{showInterval32 I}"

    -- Native runtime Float32 (endpoint evaluation).
    let If32 :=
      match func with
      | "tanh" => intervalUnaryEndpointsF32 Float32.tanh loF32 hiF32
      | "exp"  => intervalUnaryEndpointsF32 Float32.exp  loF32 hiF32
      | "log"  => intervalUnaryEndpointsF32 Float32.log  loF32 hiF32
      | "sqrt" => intervalUnaryEndpointsF32 Float32.sqrt loF32 hiF32
      | _      => ⟨Float32Interval.IntervalF32.posZero, Float32Interval.IntervalF32.posZero⟩
    IO.println s!"  Float32:  {showIntervalF32 If32}"

    -- configured binary32 endpoints, but with Arb-provided *rigorous* real enclosure rounded
    -- outward to
    -- float32.
    let X : Interval (Binary 8 23) := ⟨lo32, hi32⟩
    try
      let Iarb ←
        match func with
        | "tanh" => IEEE32Exec.tanhArb X (precBits := precBits) (digits := digits)
        | "exp"  => IEEE32Exec.expArb  X (precBits := precBits) (digits := digits)
        | "log"  => IEEE32Exec.logArb  X (precBits := precBits) (digits := digits)
        | "sqrt" => IEEE32Exec.sqrtArb X (precBits := precBits) (digits := digits)
        | _      => pure ⟨(Binary.canonicalNaN : Binary 8 23), (Binary.canonicalNaN : Binary 8 23)⟩
      IO.println s!"  configured binary32+Arb:{showInterval32 Iarb}"

      -- Check whether endpoint-evaluation enclosures contain the Arb-rounded outward enclosure.
      match interval32ToRat? I, interval32ToRat? Iarb with
      | some Ir, some Iar =>
        IO.println
          s!"  contains(binary32 endpoints ⊇ binary32+Arb)? {Rational.contains Ir Iar}"
      | _, _ =>
        IO.println
          s!"  contains(configured binary32 endpoints ⊇ configured binary32+Arb)? (n/a: non-finite)"
      match intervalF32ToRat? If32, interval32ToRat? Iarb with
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
    The exact-rational row is a small reference check for this polynomial case.
    -/
    let X : Interval (Binary 8 23) := ⟨lo32, hi32⟩
    let c01 : Interval (Binary 8 23) := Binary.Interval.point
      ((fun x => (ofModel (Model.cast .binary64 .binary32
      (toModel (Binary.ofFloat x))) : Binary 8 23)) 0.1)
    let c05 : Interval (Binary 8 23) := Binary.Interval.point
      ((fun x => (ofModel (Model.cast .binary64 .binary32
      (toModel (Binary.ofFloat x))) : Binary 8 23)) 0.5)
    -- p(x) = x*x + 0.1*x - 0.5
    let x2 := Binary.Interval.mul X X
    let t1 := Binary.Interval.mul c01 X
    let p := Binary.Interval.sub (Binary.Interval.add x2 t1) c05
    IO.println s!"  poly(x)=x^2+0.1x-0.5: {showInterval32 p}"

    -- Real interval arithmetic baseline, using exact rationals (and exact `0.1 = 1/10`).
    let Xr? : Option RationalInterval := do
      let loR ← float32ToRat? loF32
      let hiR ← float32ToRat? hiF32
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
      let Xf : Float32Interval.IntervalF32 := ⟨loF32, hiF32⟩
      let c01f : Float32Interval.IntervalF32 := Float32Interval.IntervalF32.point (Float.toFloat32
        0.1)
      let c05f : Float32Interval.IntervalF32 := Float32Interval.IntervalF32.point (Float.toFloat32
        0.5)
      let x2f := Float32Interval.IntervalF32.mul Xf Xf
      let t1f := Float32Interval.IntervalF32.mul c01f Xf
      let pf := Float32Interval.IntervalF32.sub (Float32Interval.IntervalF32.add x2f t1f) c05f
      IO.println s!"  Float32IA:{showIntervalF32 pf}"

      -- Containment checks against the real-interval baseline.
      let prR := pr
      match interval32ToRat? p with
      | some pR =>
        IO.println s!"  contains(configured binary32 IA ⊇ RealIA)? {Rational.contains pR prR}"
      | none =>
        IO.println s!"  contains(configured binary32 IA ⊇ RealIA)? (n/a: non-finite)"
      match intervalF32ToRat? pf with
      | some pR =>
        IO.println s!"  contains(Float32 IA ⊇ RealIA)? {Rational.contains pR prR}"
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
def runAddTie : IO Unit := do
  let one : Float32 := Float.toFloat32 1.0
  -- Exact `2^-24` as a float32 bit pattern.
  let halfUlp : Float32 := Float32.ofBits (0x33800000 : UInt32)
  let A : Float32Interval.IntervalF32 := ⟨one, one⟩
  let B : Float32Interval.IntervalF32 := ⟨halfUlp, halfUlp⟩
  let sumF32 := Float32Interval.IntervalF32.add A B

  let one32 : Binary 8 23 := ofBits32 one.toBits
  let halfUlp32 : Binary 8 23 := ofBits32 halfUlp.toBits
  let A32 : Interval (Binary 8 23) := ⟨one32, one32⟩
  let B32 : Interval (Binary 8 23) := ⟨halfUlp32, halfUlp32⟩
  let sum32 : Interval (Binary 8 23) := Binary.Interval.add A32 B32

  IO.println "func=add_tie (round-to-nearest-even stress)"
  IO.println ("  PyTorch analogue: torch.tensor(1.0, dtype=torch.float32) "
    ++ "+ torch.tensor(2**-24, dtype=torch.float32)")
  IO.println s!"  a=[{showFloat32 one}, {showFloat32 one}]"
  IO.println s!"  b=[{showFloat32 halfUlp}, {showFloat32 halfUlp}]"
  IO.println s!"  Float32 add (naive IA): {showIntervalF32 sumF32}"
  IO.println s!"  configured binary32 addDown/addUp: {showInterval32 sum32}"

  -- Real reference: exact dyadic sum a + b.
  let ref? : Option RationalInterval := do
    let aR ← float32ToRat? one
    let bR ← float32ToRat? halfUlp
    pure <| RationalInterval.point (aR + bR)
  match ref? with
  | none =>
    IO.println s!"  Real ref: (n/a)"
  | some ref =>
    IO.println s!"  Real ref: {Rational.format ref}"
    match intervalF32ToRat? sumF32 with
    | some sR => IO.println s!"  contains(Float32 naive ⊇ Real ref)? {Rational.contains sR ref}"
    | none => IO.println s!"  contains(Float32 naive ⊇ Real ref)? (n/a)"
    match interval32ToRat? sum32 with
    | some sR =>
        IO.println s!"  contains(configured binary32 dir ⊇ Real ref)? {Rational.contains sR ref}"
    | none => IO.println s!"  contains(configured binary32 dir ⊇ Real ref)? (n/a)"

  IO.println ""

/--
Division by negative zero, which IEEE 754 defines as `-∞` rather than an error.

The sign of zero is observable precisely through operations like this one, which is why the float32
model keeps `+0` and `-0` distinct instead of collapsing them.
-/
def runSignedZeroDiv : IO Unit := do
  let one : Float32 := Float.toFloat32 1.0
  let negZ : Float32 := Float32Interval.IntervalF32.negZero
  let denom : Float32Interval.IntervalF32 := ⟨negZ, negZ⟩
  let numer : Float32Interval.IntervalF32 := ⟨one, one⟩
  let qF32 := Float32Interval.IntervalF32.div numer denom
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
  IO.println s!"  point Float32: 1/(-0.0) = {showFloat32 qPoint}"
  IO.println
    s!"  point configured binary32: 1/(-0.0) = {hostQuotient} \
      (bits={Binary.toBits32 qPoint32})"
  IO.println s!"  Float32 IA div: {showIntervalF32 qF32}"
  IO.println s!"  configured binary32 IA div: {showInterval32 q32}"
  IO.println ""

/-- Run a small fixed set of comparisons (unary funcs + a polynomial + some edge cases). -/
def run : IO UInt32 := do
  let precBits := 200
  let digits := 50

  IO.println "Arb vs configured binary32 comparison (unary funcs + one polynomial)"
  IO.println "(Arb is rigorous; configured binary32 is endpoint evaluation for transcendentals.)"
  IO.println ""

  -- Choose dyadic-friendly endpoints so the float32 endpoints are exact.
  runOne "tanh" (-0.5) 0.5 precBits digits
  runOne "exp" (-1.0)  1.0 precBits digits
  runOne "exp" 80.0   90.0 precBits digits
  runOne "log"  0.5   2.0 precBits digits
  runOne "sqrt" 0.0   2.0 precBits digits
  runOne "poly" (-0.5) 0.5 precBits digits
  runAddTie
  runSignedZeroDiv
  pure 0

/-- Command-line help for the Arb-vs-IEEE32 interval tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean Arb vs configured binary32 interval tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean floats_arb_ieee_compare"
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
