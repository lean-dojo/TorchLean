/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Arb.Oracle
public import NN.Floats.Interval.IEEEExec32ArbTrans
public import NN.Floats.Interval.Comparison
public import Std

/-!
# Arb vs IEEE32Exec interval tutorial

This tutorial prints side-by-side enclosures for a few unary functions over a one-dimensional input
interval:

- **Arb** (`python-flint` / Arb ball arithmetic): rigorous real enclosures at chosen precision.
- **IEEE32Exec**: executable float32 evaluation on the endpoints (not a proved outward-rounded
  interval rule for transcendentals).
- **Float32 baseline**: ordinary runtime `Float32` endpoint arithmetic, included to show why
  directed rounding matters.
- **Rational baseline**: exact `Rat` interval arithmetic for small polynomial/reference checks.

NumPy / PyTorch analogue:

```python
import numpy as np

lo, hi = np.float32(-0.5), np.float32(0.5)
endpoint_box = (np.tanh(lo), np.tanh(hi))   # common quick check, not a rigorous enclosure
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

If Arb is not installed, the tutorial still prints the IEEE32Exec side and reports the Arb failure.
-/

@[expose] public section


open Std

namespace TorchLean.Floats.Interval.ComparisonTutorial

open TorchLean.Floats.Arb
open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.IEEE32Exec
open TorchLean.Floats.Interval.Comparison
open TorchLean.Floats.Interval.Comparison.RealInterval

/-- JSON expression for `x^2 + 0.1*x - 0.5`, in the safe Arb expression language. -/
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
- `IEEE32Exec`: endpoint evaluation using TorchLean's deterministic binary32 executable model;
- `Float32`: ordinary runtime endpoint evaluation;
- `IEEE32Exec+Arb`: Arb real enclosure rounded outward to binary32 endpoints.
-/
def runOne (func : String) (lo hi : Float) (precBits digits : Nat) : IO Unit := do
  let loF32 := Float.toFloat32 lo
  let hiF32 := Float.toFloat32 hi
  let lo32 := IEEE32Exec.ofBits loF32.toBits
  let hi32 := IEEE32Exec.ofBits hiF32.toBits

  IO.println s!"func={func}, x∈[{lo}, {hi}] (as Float32 bits: lo={loF32.toBits}, hi={hiF32.toBits})"

  -- Arb (rigorous real enclosure / ball enclosure).
  if func = "poly" then
    -- Use the safe `expr` request format (not the unary `--func` mode).
    let xVar := ("x", (toString lo, toString hi))
    try
      let r ← TorchLean.Floats.Arb.runExpr (vars := [xVar]) (expr := polynomialExpr)
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

  -- IEEE32Exec (endpoint evaluation).
  if func = "tanh" ∨ func = "exp" ∨ func = "log" ∨ func = "sqrt" then
    /-
    PyTorch / NumPy analogue:

    ```python
    np.array([f(lo), f(hi)], dtype=np.float32)
    ```

    That is a useful sanity check, but it is not generally an interval proof for nonlinear
    functions. The Arb-backed line below is the rigorous path when Arb is installed.
    -/
    let I :=
      match func with
      | "tanh" => intervalUnaryEndpoints IEEE32Exec.tanh lo32 hi32
      | "exp"  => intervalUnaryEndpoints IEEE32Exec.exp  lo32 hi32
      | "log"  => intervalUnaryEndpoints IEEE32Exec.log  lo32 hi32
      | "sqrt" => intervalUnaryEndpoints IEEE32Exec.sqrt lo32 hi32
      | _      => ⟨IEEE32Exec.canonicalNaN, IEEE32Exec.canonicalNaN⟩
    IO.println s!"  IEEE32Exec:{showInterval32 I}"

    -- Native runtime Float32 (endpoint evaluation).
    let If32 :=
      match func with
      | "tanh" => intervalUnaryEndpointsF32 Float32.tanh loF32 hiF32
      | "exp"  => intervalUnaryEndpointsF32 Float32.exp  loF32 hiF32
      | "log"  => intervalUnaryEndpointsF32 Float32.log  loF32 hiF32
      | "sqrt" => intervalUnaryEndpointsF32 Float32.sqrt loF32 hiF32
      | _      => ⟨Float32Interval.IntervalF32.posZero, Float32Interval.IntervalF32.posZero⟩
    IO.println s!"  Float32:  {showIntervalF32 If32}"

    -- IEEE32Exec endpoints, but with Arb-provided *rigorous* real enclosure rounded outward to
    -- float32.
    let X : Interval32 := ⟨lo32, hi32⟩
    try
      let Iarb ←
        match func with
        | "tanh" => IEEE32Exec.Interval32.tanhArb X (precBits := precBits) (digits := digits)
        | "exp"  => IEEE32Exec.Interval32.expArb  X (precBits := precBits) (digits := digits)
        | "log"  => IEEE32Exec.Interval32.logArb  X (precBits := precBits) (digits := digits)
        | "sqrt" => IEEE32Exec.Interval32.sqrtArb X (precBits := precBits) (digits := digits)
        | _      => pure ⟨IEEE32Exec.canonicalNaN, IEEE32Exec.canonicalNaN⟩
      IO.println s!"  IEEE32Exec+Arb:{showInterval32 Iarb}"

      -- Check whether endpoint-evaluation enclosures contain the Arb-rounded outward enclosure.
      match interval32ToRat? I, interval32ToRat? Iarb with
      | some Ir, some Iar =>
        IO.println
          s!"  contains(IEEE32Exec endpoints ⊇ IEEE32Exec+Arb)? {IntervalRat.contains Ir Iar}"
      | _, _ =>
        IO.println s!"  contains(IEEE32Exec endpoints ⊇ IEEE32Exec+Arb)? (n/a: non-finite)"
      match intervalF32ToRat? If32, interval32ToRat? Iarb with
      | some Ir, some Iar =>
        IO.println s!"  contains(Float32 endpoints ⊇ IEEE32Exec+Arb)? {IntervalRat.contains Ir Iar}"
      | _, _ =>
        IO.println s!"  contains(Float32 endpoints ⊇ IEEE32Exec+Arb)? (n/a: non-finite)"
    catch e =>
      IO.println s!"  IEEE32Exec+Arb:(failed) {e.toString}"

  -- A small polynomial using executable directed-rounding interval arithmetic (add/mul only).
  if func = "poly" then
    /-
    NumPy analogue:

    ```python
    x = np.array([lo, hi], dtype=np.float32)
    p = x*x + np.float32(0.1)*x - np.float32(0.5)
    ```

    TorchLean's `Interval32` version uses directed rounding for each interval arithmetic step.
    The exact-rational row is a small reference check for this polynomial case.
    -/
    let X : Interval32 := ⟨lo32, hi32⟩
    let c01 : Interval32 := Interval32.point (IEEE32Exec.ofFloat 0.1)
    let c05 : Interval32 := Interval32.point (IEEE32Exec.ofFloat 0.5)
    -- p(x) = x*x + 0.1*x - 0.5
    let x2 := Interval32.mul X X
    let t1 := Interval32.mul c01 X
    let p := Interval32.sub (Interval32.add x2 t1) c05
    IO.println s!"  poly(x)=x^2+0.1x-0.5: {showInterval32 p}"

    -- Real interval arithmetic baseline, using exact rationals (and exact `0.1 = 1/10`).
    let Xr? : Option IntervalRat := do
      let loR ← float32ToRat? loF32
      let hiR ← float32ToRat? hiF32
      pure ⟨loR, hiR⟩
    let c01r : IntervalRat := IntervalRat.point (Rat.normalize 1 10)
    let c05r : IntervalRat := IntervalRat.point (Rat.normalize 1 2)
    match Xr? with
    | none =>
      IO.println s!"  RealIA:   (n/a: non-finite input endpoints)"
    | some Xr =>
      let x2r := IntervalRat.mul Xr Xr
      let t1r := IntervalRat.mul c01r Xr
      let pr := IntervalRat.sub (IntervalRat.add x2r t1r) c05r
      IO.println s!"  RealIA:   {showIntervalRat pr}"

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
        IO.println s!"  contains(IEEE32Exec IA ⊇ RealIA)? {IntervalRat.contains pR prR}"
      | none =>
        IO.println s!"  contains(IEEE32Exec IA ⊇ RealIA)? (n/a: non-finite)"
      match intervalF32ToRat? pf with
      | some pR =>
        IO.println s!"  contains(Float32 IA ⊇ RealIA)? {IntervalRat.contains pR prR}"
      | none =>
        IO.println s!"  contains(Float32 IA ⊇ RealIA)? (n/a: non-finite)"

  IO.println ""

def runAddTie : IO Unit := do
  let one : Float32 := Float.toFloat32 1.0
  -- Exact `2^-24` as a float32 bit pattern.
  let halfUlp : Float32 := Float32.ofBits (0x33800000 : UInt32)
  let A : Float32Interval.IntervalF32 := ⟨one, one⟩
  let B : Float32Interval.IntervalF32 := ⟨halfUlp, halfUlp⟩
  let sumF32 := Float32Interval.IntervalF32.add A B

  let one32 : IEEE32Exec := IEEE32Exec.ofBits one.toBits
  let halfUlp32 : IEEE32Exec := IEEE32Exec.ofBits halfUlp.toBits
  let A32 : Interval32 := ⟨one32, one32⟩
  let B32 : Interval32 := ⟨halfUlp32, halfUlp32⟩
  let sum32 : Interval32 := Interval32.add A32 B32

  IO.println "func=add_tie (round-to-nearest-even stress)"
  IO.println "  PyTorch analogue: torch.tensor(1.0, dtype=torch.float32) + torch.tensor(2**-24, dtype=torch.float32)"
  IO.println s!"  a=[{showFloat32 one}, {showFloat32 one}]"
  IO.println s!"  b=[{showFloat32 halfUlp}, {showFloat32 halfUlp}]"
  IO.println s!"  Float32 add (naive IA): {showIntervalF32 sumF32}"
  IO.println s!"  IEEE32Exec addDown/addUp: {showInterval32 sum32}"

  -- Real reference: exact dyadic sum a + b.
  let ref? : Option IntervalRat := do
    let aR ← float32ToRat? one
    let bR ← float32ToRat? halfUlp
    pure <| IntervalRat.point (aR + bR)
  match ref? with
  | none =>
    IO.println s!"  Real ref: (n/a)"
  | some ref =>
    IO.println s!"  Real ref: {showIntervalRat ref}"
    match intervalF32ToRat? sumF32 with
    | some sR => IO.println s!"  contains(Float32 naive ⊇ Real ref)? {IntervalRat.contains sR ref}"
    | none => IO.println s!"  contains(Float32 naive ⊇ Real ref)? (n/a)"
    match interval32ToRat? sum32 with
    | some sR => IO.println s!"  contains(IEEE32Exec dir ⊇ Real ref)? {IntervalRat.contains sR ref}"
    | none => IO.println s!"  contains(IEEE32Exec dir ⊇ Real ref)? (n/a)"

  IO.println ""

def runSignedZeroDiv : IO Unit := do
  let one : Float32 := Float.toFloat32 1.0
  let negZ : Float32 := Float32Interval.IntervalF32.negZero
  let denom : Float32Interval.IntervalF32 := ⟨negZ, negZ⟩
  let numer : Float32Interval.IntervalF32 := ⟨one, one⟩
  let qF32 := Float32Interval.IntervalF32.div numer denom
  let qPoint := Float32.div one negZ

  let one32 : IEEE32Exec := IEEE32Exec.ofBits one.toBits
  let negZ32 : IEEE32Exec := IEEE32Exec.ofBits negZ.toBits
  let denom32 : Interval32 := ⟨negZ32, negZ32⟩
  let numer32 : Interval32 := ⟨one32, one32⟩
  let q32 := Interval32.div numer32 denom32
  let qPoint32 := IEEE32Exec.div one32 negZ32

  IO.println "func=div_signed_zero (widening from signed-zero containment)"
  IO.println "  PyTorch analogue: torch.tensor(1.0, dtype=torch.float32) / torch.tensor(-0.0, dtype=torch.float32)"
  IO.println s!"  point Float32: 1/(-0.0) = {showFloat32 qPoint}"
  IO.println s!"  point IEEE32Exec: 1/(-0.0) = {IEEE32Exec.toFloat qPoint32} (bits={qPoint32.bits})"
  IO.println s!"  Float32 IA div: {showIntervalF32 qF32}"
  IO.println s!"  IEEE32Exec IA div: {showInterval32 q32}"
  IO.println ""

/-- Run a small fixed set of comparisons (unary funcs + a polynomial + some edge cases). -/
def main (_args : List String) : IO UInt32 := do
  let precBits := 200
  let digits := 50

  IO.println "Arb vs IEEE32Exec quick comparison (unary funcs + one polynomial)"
  IO.println "(Arb is rigorous; IEEE32Exec is endpoint evaluation for transcendentals.)"
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

end TorchLean.Floats.Interval.ComparisonTutorial

namespace NN.Examples.Advanced.Floats.ArbIEEEExecCompare

/-- Entrypoint: run the Arb-vs-`IEEE32Exec` interval tutorial. -/
def main (args : List String) : IO UInt32 :=
  TorchLean.Floats.Interval.ComparisonTutorial.main args

end NN.Examples.Advanced.Floats.ArbIEEEExecCompare
