/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Util.Json
public import NN.Tensor.Conversion
public import NN.Floats.Interval.IEEEExec32ArbTrans
public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Conversion.Cast.Runtime
public import FloatLib.Floats.Formats.BinaryInterchange.Model.RealSemantics
public import FloatLib.Floats.Formats.BinaryInterchange.Model.ERealSemantics
public import FloatLib.Floats.Formats.IEEE754.Native
public import FloatLib.Floats.Formats.BinaryInterchange.DirectedSemantics.Rational.Conversion

/-!
# Piecewise polynomial certificates

This module implements a Lean checker for one-dimensional piecewise-polynomial certificates.

Why this exists:
- In TorchLean’s “external producer, Lean checker” workflow, an external tool (Julia, Python, ...)
  can fit or search for a piecewise-polynomial surrogate and export a compact JSON artifact.
- Lean then checks basic algebraic consistency conditions on that artifact.

Our checker is conservative: it checks *exact* equalities over `Rat` with no floating-point
tolerance.

Optionally, you can also run an *executable float32* cross-check via `checkJsonIEEE32ExecExact`:
if every rational in the certificate is exactly representable as a finite IEEE-754 binary32
(`ExecFloat.Binary 8 23`), we re-run the endpoint equalities under `ExecFloat.Binary 8 23`
arithmetic. This is useful
when the producer is actually operating in float32 and you want to confirm the certificate’s
equalities hold under the same semantics.

What is checked in `format = "piecewise_poly_v0"`:
- `xs` is strictly increasing,
- `ys` has the same length as `xs`,
- there is one `piece` per interval $[\mathrm{xs}[i],\mathrm{xs}[i+1]]$,
- each piece’s `(lo, hi)` matches the corresponding interval endpoints,
- the local-coordinate polynomial interpolates the endpoints exactly:
  $p_i(\mathrm{xs}[i])=\mathrm{ys}[i]$ and
  $p_i(\mathrm{xs}[i+1])=\mathrm{ys}[i+1]$.

This certificate format focuses on endpoint and piece consistency.  It does **not** claim global
range bounds for higher-degree polynomials; those require additional machinery such as Bernstein
bounds or interval arithmetic with derivative-based splitting.

References (background):
- de Boor, *A Practical Guide to Splines* (1978).
- Certificate checking pattern: export an artifact, then check it in a small Lean checker core.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace NN.Verification.Splines.PiecewisePolyCert

open Lean
open Json
open NN.Verification.Json
open Spec TorchLean

/-!
## Utilities: exact rationals in JSON
-/

/--
Parse a rational in the format emitted by TorchLean’s Arb helpers:

- integer: `"5"`, `"-3"`
- fraction: `"5/2"`, `"-7/10"`

We avoid JSON numbers here because they are stored as `Scientific` and are not guaranteed to
round-trip exactly for large integers.
-/
def parseRatString (s : String) : Except String Rat := do
  let s := s.trimAscii.toString
  if s.isEmpty then
    throw "empty rational string"
  match s.splitOn "/" with
  | [numStr] =>
      match numStr.toInt? with
      | some n => pure (Rat.ofInt n)
      | none => throw s!"invalid integer rational: '{s}'"
  | [numStr, denStr] =>
      let n ←
        match numStr.toInt? with
        | some n => pure n
        | none => throw s!"invalid numerator: '{numStr}'"
      let d ←
        match denStr.toNat? with
        | some d => pure d
        | none => throw s!"invalid denominator (expected Nat): '{denStr}'"
      if d = 0 then
        throw "invalid rational: denominator is 0"
      pure (Rat.ofInt n / Rat.ofInt (Int.ofNat d))
  | _ =>
      throw s!"invalid rational (expected n or n/d): '{s}'"

/-- Parse a `Rat` from a JSON string field with context. -/
def parseRat (ctx : String) (j : Json) : IO Rat := do
  let kind : String :=
    match j with
    | .null => "null"
    | .bool _ => "bool"
    | .num _ => "number"
    | .str _ => "string"
    | .arr _ => "array"
    | .obj _ => "object"
  let s ←
    match j with
    | .str s => pure s
    | _ => throw <| IO.userError s!"{ctx}: expected string encoding of a rational, got {kind}"
  match parseRatString s with
  | .ok q => pure q
  | .error e => throw <| IO.userError s!"{ctx}: {e}"

/-!
## Certificate schema
-/

/-- One polynomial segment on an interval $[\mathrm{lo},\mathrm{hi}]$, using local coordinate
$t=x-\mathrm{lo}$. -/
structure PolynomialPiece where
  lo : Rat
  hi : Rat
  coeffs : Array Rat
  deriving Repr, Inhabited

/--
Parsed certificate payload for `piecewise_poly_v0`.

`degree` is redundant (it should equal `coeffs.size - 1`), but it is useful for fast validation.
-/
structure PiecewisePolyCertificate where
  /-- Polynomial degree `d` (so each piece has `d+1` coefficients). -/
  degree : Nat
  /-- Number of knot points. -/
  n : Nat
  /-- Knot x-coordinates as a rank-one tensor of length `n`. -/
  xs : Tensor Rat [n]
  /-- Knot y-values as a rank-one tensor of length `n`. -/
  ys : Tensor Rat [n]
  /-- One piece per interval $[\mathrm{xs}[i],\mathrm{xs}[i+1]]$. -/
  pieces : Array PolynomialPiece

/-!
## Polynomial evaluation
-/

/--
Evaluate a polynomial in local coordinate $t$ using Horner’s rule.

Given coefficients $[a_0,a_1,\ldots,a_d]$ representing:

$$
a_0+a_1t+a_2t^2+\cdots+a_dt^d.
$$

This helper is generic so we can reuse it for `Rat` (exact checking) and for `ExecFloat.Binary 8 23`
(executable float32 semantics).
-/
def evalPolyHorner {α : Type} [Zero α] [Add α] [Mul α] (coeffs : Array α) (t : α) : α :=
  coeffs.foldr (fun a acc => acc * t + a) 0

/-!
## Parsing
-/

namespace Internal

/-- Parse an array of rational strings, attaching `ctx` to any error message. -/
def parseRatArray (ctx : String) (j : Json) : IO (Array Rat) := do
  match j with
  | .arr xs => xs.mapM (parseRat ctx)
  | _ => throw <| IO.userError s!"{ctx}: expected JSON array"

/-- Parse one polynomial segment from the certificate JSON object. -/
def parsePolynomialPiece (ctx : String) (j : Json) : IO PolynomialPiece := do
  let o ← expectObject j ctx
  let loJ ← expectField o "lo" ctx
  let hiJ ← expectField o "hi" ctx
  let coeffsJ ← expectField o "coeffs" ctx
  let lo ← parseRat (ctx := s!"{ctx}.lo") loJ
  let hi ← parseRat (ctx := s!"{ctx}.hi") hiJ
  let coeffs ← parseRatArray (ctx := s!"{ctx}.coeffs") coeffsJ
  pure { lo, hi, coeffs }

end Internal

/-- Require an array entry at a certificate-checking boundary, with a schema-oriented error. -/
def requireArrayEntry {α : Type} (ctx : String) (xs : Array α) (i : Nat) : IO α := do
  match xs[i]? with
  | some x => pure x
  | none => throw <| IO.userError s!"{ctx}: missing entry at index {i} (size={xs.size})"

/-- Parse the `piecewise_poly_v0` JSON payload into a structured certificate. -/
def parsePiecewisePolyCertificate (j : Json) : IO PiecewisePolyCertificate := do
  let top ← expectObject j "top-level"
  expectFormat top "piecewise_poly_v0"
  let degree ← expectFieldNat top "degree" "top-level"

  let xsArr ← Internal.parseRatArray (ctx := "top-level.xs") (← expectField top "xs" "top-level")
  let ysArr ← Internal.parseRatArray (ctx := "top-level.ys") (← expectField top "ys" "top-level")
  let piecesJ ← expectField top "pieces" "top-level"
  let pieces ←
    match piecesJ with
    | .arr ps => ps.mapIdxM (fun i pj => Internal.parsePolynomialPiece (ctx := s!"pieces[{i}]") pj)
    | _ => throw <| IO.userError "top-level.pieces: expected array"

  let n := xsArr.size
  if n < 2 then
    throw <| IO.userError "xs must have length ≥ 2"
  if hYsSize : ysArr.size = n then
    if pieces.size != n - 1 then
      throw <|
        IO.userError
          s!"pieces length mismatch: pieces.size={pieces.size}, expected {n - 1} (=xs.size-1)"

    -- Validate the dynamic arrays once, then enter the canonical shape-indexed tensor API.
    let xs : Tensor Rat [n] := Tensor.from xsArr
    let ys : Tensor Rat [n] := hYsSize ▸ Tensor.from ysArr

    pure { degree, n, xs, ys, pieces }
  else
    throw <| IO.userError s!"ys length mismatch: ys.size={ysArr.size}, xs.size={n}"

/-!
## Checking
-/

/--
Check a piecewise-polynomial certificate **exactly** over `Rat`.

This is the smallest checker core for this format: it uses exact rational arithmetic with no
tolerances, so a passing check means the certificate's equalities hold exactly as stated.
-/
def checkCertificateRat (cert : PiecewisePolyCertificate) : IO Unit := do
  let n := cert.n
  let xs := Tensor.to cert.xs (Array ℚ)
  let ys := Tensor.to cert.ys (Array ℚ)

  for i in [0:n - 1] do
    let a ← requireArrayEntry "xs" xs i
    let b ← requireArrayEntry "xs" xs (i + 1)
    unless a < b do
      throw <| IO.userError s!"xs not strictly increasing at i={i}: {a} !< {b}"

  for i in [0:cert.pieces.size] do
    let p ← requireArrayEntry "pieces" cert.pieces i
    let lo ← requireArrayEntry "xs" xs i
    let hi ← requireArrayEntry "xs" xs (i + 1)
    unless p.lo == lo do
      throw <| IO.userError s!"piece[{i}].lo mismatch: {p.lo} ≠ xs[{i}]={lo}"
    unless p.hi == hi do
      throw <| IO.userError s!"piece[{i}].hi mismatch: {p.hi} ≠ xs[{i+1}]={hi}"
    if p.coeffs.size != cert.degree + 1 then
      throw <|
        IO.userError
          s!"piece[{i}].coeffs length mismatch: {p.coeffs.size} ≠ degree+1={cert.degree + 1}"

    let yLo ← requireArrayEntry "ys" ys i
    let yHi ← requireArrayEntry "ys" ys (i + 1)
    let tHi : Rat := hi - lo
    let pLo := evalPolyHorner p.coeffs 0
    let pHi := evalPolyHorner p.coeffs tHi
    unless pLo == yLo do
      throw <| IO.userError s!"endpoint mismatch at i={i}: p(lo)={pLo} ≠ y={yLo}"
    unless pHi == yHi do
      throw <| IO.userError s!"endpoint mismatch at i={i}: p(hi)={pHi} ≠ yNext={yHi}"

/--
Check a `piecewise_poly_v0` certificate payload.

Raises `IO.userError` on the first mismatch; prints a short success message otherwise.
-/
def checkJson (j : Json) : IO Unit := do
  let cert ← parsePiecewisePolyCertificate j
  checkCertificateRat cert
  IO.println "Piecewise polynomial certificate verified."

/--
Attempt to convert a rational to an *exact* finite `ExecFloat.Binary 8 23` value.

This is used for optional “float32 semantics” checking: we only accept values that are exactly
representable on the binary32 grid, so equality checks are meaningful at the executable IEEE level.

Implementation note:
- FloatLib rounds the rational outward in its binary32 model; packing that model preserves its
  bits. We accept only when the lower and upper rounding compare equal.
-/
def ratToIEEE32ExecExact (ctx : String) (q : Rat) : IO (ExecFloat.Binary 8 23) := do
  let lo : ExecFloat.Binary 8 23 := ExecFloat.Binary.ofModel <|
    FloatLib.Floats.Formats.BinaryInterchange.Model.roundRatQDown
      FloatLib.Floats.Formats.BinaryInterchange.FloatFormat.binary32 q
  let hi : ExecFloat.Binary 8 23 := ExecFloat.Binary.ofModel <|
    FloatLib.Floats.Formats.BinaryInterchange.Model.roundRatQUp
      FloatLib.Floats.Formats.BinaryInterchange.FloatFormat.binary32 q
  unless lo == hi do
    throw <|
      IO.userError
        (s!"{ctx}: rational is not exactly representable as binary32 (lo={lo}, hi={hi}).\n" ++
         "If you intend a float32-valued certificate, prefer dyadic rationals (n/2^k) " ++
         "or emit raw bits.")
  unless ExecFloat.Binary.isFinite lo do
    throw <| IO.userError s!"{ctx}: value is not finite in binary32 semantics (NaN/Inf)."
  pure lo

/--
Check the same endpoint equalities as `checkCertificateRat`, but under `ExecFloat.Binary 8 23`
arithmetic.

This is an *additional* (optional) check that answers: “if we interpret the certificate data as
binary32 values and run the polynomial evaluation with executable IEEE-754 ops, do we still hit
the claimed endpoints?”

The check is deliberately strict:
- every rational in the cert must be exactly representable as a finite `ExecFloat.Binary 8 23`,
- comparisons use IEEE-style `BEq` (so `+0 == -0`, and NaNs never compare equal).
-/
def checkCertificateIEEE32ExecExact (cert : PiecewisePolyCertificate) : IO Unit := do
  let xsQ := Tensor.to cert.xs (Array ℚ)
  let ysQ := Tensor.to cert.ys (Array ℚ)

  let xs ← xsQ.mapIdxM (fun i q => ratToIEEE32ExecExact (ctx := s!"xs[{i}]") q)
  let ys ← ysQ.mapIdxM (fun i q => ratToIEEE32ExecExact (ctx := s!"ys[{i}]") q)

  for i in [0:cert.pieces.size] do
    let p ← requireArrayEntry "pieces" cert.pieces i
    let lo32 ← requireArrayEntry "xs" xs i
    let hi32 ← requireArrayEntry "xs" xs (i + 1)
    let yLo32 ← requireArrayEntry "ys" ys i
    let yHi32 ← requireArrayEntry "ys" ys (i + 1)

    let coeffs32 ←
      p.coeffs.mapIdxM (fun k q => ratToIEEE32ExecExact (ctx := s!"pieces[{i}].coeffs[{k}]") q)

    let tHi32 : ExecFloat.Binary 8 23 := hi32 - lo32
    let pLo32 := evalPolyHorner coeffs32 0
    let pHi32 := evalPolyHorner coeffs32 tHi32

    unless pLo32 == yLo32 do
      throw <|
        IO.userError
          (s!"IEEE32Exec endpoint mismatch at i={i}: p(lo)={pLo32} ≠ y={yLo32} " ++
            s!"(bits {ExecFloat.Binary.toBits32 pLo32} vs " ++
            s!"{ExecFloat.Binary.toBits32 yLo32})")
    unless pHi32 == yHi32 do
      throw <|
        IO.userError
          (s!"IEEE32Exec endpoint mismatch at i={i}: p(hi)={pHi32} ≠ yNext={yHi32} " ++
            s!"(bits {ExecFloat.Binary.toBits32 pHi32} vs " ++
            s!"{ExecFloat.Binary.toBits32 yHi32})")

  IO.println "IEEE32Exec semantics check verified (exact representability + endpoint equalities)."

/-- Check a piecewise-polynomial certificate stored as JSON, including `ExecFloat.Binary 8 23`
semantics. -/
def checkJsonIEEE32ExecExact (j : Json) : IO Unit := do
  let cert ← parsePiecewisePolyCertificate j
  checkCertificateRat cert
  IO.println "Piecewise polynomial certificate verified."
  checkCertificateIEEE32ExecExact cert

/-- Check a piecewise-polynomial certificate stored as a JSON file. -/
def checkFile (path : String) : IO Unit := do
  let j ← readJsonFile path
  checkJson j

end NN.Verification.Splines.PiecewisePolyCert
