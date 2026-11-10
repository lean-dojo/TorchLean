/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.JSON
public import TorchLean.Tensor.Conversion
public import FloatLib.Floats.Formats.BinaryInterchange.Configured

/-!
# Piecewise polynomial certificates

A Lean checker for one-dimensional piecewise-polynomial certificates.

An external tool fits a piecewise-polynomial surrogate and exports its knots and coefficients.
The checker validates their algebraic consistency with exact `Rat` arithmetic and no tolerance.

The optional *executable float32* cross-check `checkJsonBinary32` requires every rational
in the certificate to be exactly representable as a finite IEEE-754 binary32
(`Binary32`) and re-runs the endpoint equalities under `Binary32`
arithmetic. The replay follows the Horner order of `evalPolyHorner`, so its verdict covers that
order alone, and producer code with another order or a fused GPU evaluation can round
differently.

What is checked in `format = "piecewise_poly_v0"`:
- `xs` has at least two entries and is strictly increasing,
- `ys` has the same length as `xs`,
- there is one `piece` per interval $[\mathrm{xs}[i],\mathrm{xs}[i+1]]$,
- each piece’s `(lo, hi)` matches the corresponding interval endpoints,
- each piece has `degree + 1` coefficients,
- the local-coordinate polynomial interpolates the endpoints exactly:
  $p_i(\mathrm{xs}[i])=\mathrm{ys}[i]$ and
  $p_i(\mathrm{xs}[i+1])=\mathrm{ys}[i+1]$.

The format certifies endpoint and piece consistency. Global range bounds for higher-degree
polynomials lie outside it, since they need further machinery such as Bernstein bounds or
interval arithmetic with derivative-based splitting.

References (background):
- de Boor, *A Practical Guide to Splines* (1978).
- Certificate checking pattern: export an artifact, then check it in a small Lean checker core.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat Binary32)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Examples.Splines

open Lean
open TorchLean.Json Lean.Json

/-!
## Utilities: exact rationals in JSON
-/

/--
Parse a `Rat` from a JSON string `n` or `n/d`, prefixing any error with `ctx`. JSON numbers and
other non-string values are rejected.
-/
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
  /-- Left endpoint of the segment. -/
  lo : Rat
  /-- Right endpoint of the segment. -/
  hi : Rat
  /-- Coefficients `[a₀, …, a_d]` of the polynomial in the local coordinate `t = x - lo`. -/
  coeffs : Array Rat
  deriving Repr, Inhabited

/--
Parsed certificate payload for `piecewise_poly_v0`.

`degree` is redundant, since it should equal `coeffs.size - 1` for every piece, and it serves
validation: `Certificate.checkStructure` rejects any piece with another coefficient count.
-/
structure Certificate where
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
Evaluate a polynomial in local coordinate $t$ using Horner’s rule, given coefficients
$[a_0,a_1,\ldots,a_d]$ representing

$$
a_0+a_1t+a_2t^2+\cdots+a_dt^d.
$$

The helper needs only `Zero`, `Add` and `Mul`, so `Rat` (exact checking) and
`Binary32` (executable float32 semantics) share it.
-/
def evalPolyHorner {α : Type} [Zero α] [Add α] [Mul α] (coeffs : Array α) (t : α) : α :=
  coeffs.foldr (fun a acc => acc * t + a) 0

/-!
## Parsing
-/

/-- Parse an array of rational strings, attaching `ctx` to any error message. -/
def parseRatArray (ctx : String) (j : Json) : IO (Array Rat) := do
  match j with
  | .arr xs => xs.mapM (parseRat ctx)
  | _ => throw <| IO.userError s!"{ctx}: expected JSON array"

/-- Parse one polynomial segment from the certificate JSON object. -/
def PolynomialPiece.parse (ctx : String) (j : Json) : IO PolynomialPiece := do
  let o ← IO.ofExcept <| Json.object ctx j
  let loJ ← IO.ofExcept <| Json.field ctx "lo" o
  let hiJ ← IO.ofExcept <| Json.field ctx "hi" o
  let coeffsJ ← IO.ofExcept <| Json.field ctx "coeffs" o
  let lo ← parseRat (ctx := s!"{ctx}.lo") loJ
  let hi ← parseRat (ctx := s!"{ctx}.hi") hiJ
  let coeffs ← parseRatArray (ctx := s!"{ctx}.coeffs") coeffsJ
  pure { lo, hi, coeffs }

/-- Read entry `i` of `xs`, raising an error that names `ctx`, the index and the array size when
the entry is missing. -/
def requireArrayEntry {α : Type} (ctx : String) (xs : Array α) (i : Nat) : IO α := do
  match xs[i]? with
  | some x => pure x
  | none => throw <| IO.userError s!"{ctx}: missing entry at index {i} (size={xs.size})"

/-- Parse a `piecewise_poly_v0` JSON payload into a structured certificate, requiring at least two
knots, as many `ys` as `xs` and one piece per knot interval. -/
def Certificate.parse (j : Json) : IO Certificate := do
  let top ← IO.ofExcept <| Json.object "top-level" j
  let format ← IO.ofExcept <|
    Json.field "top-level" "format" top >>= Json.string "top-level.format"
  IO.ofExcept <| expectFormat format "piecewise_poly_v0"
  let degree ← IO.ofExcept <|
    Json.field "top-level" "degree" top >>= Json.expectNat "top-level.degree"

  let xsArr ← parseRatArray (ctx := "top-level.xs")
    (← IO.ofExcept (Json.field "top-level" "xs" top))
  let ysArr ← parseRatArray (ctx := "top-level.ys")
    (← IO.ofExcept (Json.field "top-level" "ys" top))
  let piecesJ ← IO.ofExcept <| Json.field "top-level" "pieces" top
  let pieces ←
    match piecesJ with
    | .arr ps => ps.mapIdxM (fun i pj => PolynomialPiece.parse (ctx := s!"pieces[{i}]") pj)
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

/-- Validate the knot partition and polynomial metadata: at least two knots, strictly increasing
`xs`, one piece per knot interval with matching endpoints and `degree + 1` coefficients per piece.
In-memory callers that skip `Certificate.parse` get the same validation. -/
def Certificate.checkStructure (cert : Certificate) : IO Unit := do
  let n := cert.n
  let xs := Tensor.to cert.xs (Array Rat)
  if n < 2 then
    throw <| IO.userError "xs must have length ≥ 2"
  if cert.pieces.size != n - 1 then
    throw <|
      IO.userError
        s!"pieces length mismatch: pieces.size={cert.pieces.size}, expected {n - 1} (=xs.size-1)"

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

/--
Check a piecewise-polynomial certificate **exactly** over `Rat`.

The smallest checker core of the format validates the structure and evaluates every piece at both
endpoints in exact rational arithmetic with no tolerance, so a passing check means the endpoint
equalities hold exactly as stated.
-/
def Certificate.checkRat (cert : Certificate) : IO Unit := do
  checkStructure cert
  let xs := Tensor.to cert.xs (Array Rat)
  let ys := Tensor.to cert.ys (Array Rat)
  for i in [0:cert.pieces.size] do
    let p ← requireArrayEntry "pieces" cert.pieces i
    let lo ← requireArrayEntry "xs" xs i
    let hi ← requireArrayEntry "xs" xs (i + 1)
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
Check a `piecewise_poly_v0` certificate payload exactly over `Rat`.

The first parse error or mismatch raises `IO.userError`, and a passing check prints a short
success message.
-/
def checkJson (j : Json) : IO Unit := do
  let cert ← Certificate.parse j
  Certificate.checkRat cert
  IO.println "Piecewise polynomial certificate verified."

/--
Convert a rational to an *exact* finite `Binary32` value, raising an error otherwise.

The optional float32 check accepts only values on the binary32 grid, so its equality checks are
meaningful at the executable IEEE level.

Implementation note: FloatLib rounds the rational down and up in its binary32 model, and packing
that model preserves its bits. The value is accepted only when the two roundings compare equal and
the result is finite.
-/
def ratToBinary32 (ctx : String) (q : Rat) : IO Binary32 := do
  let lo : Binary32 :=
    ExecFloat.Binary.ofModel (Model.ofRat FloatFormat.binary32 q (mode := .down))
  let hi : Binary32 :=
    ExecFloat.Binary.ofModel (Model.ofRat FloatFormat.binary32 q (mode := .up))
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
Check the endpoint equalities of `checkRat` under `Binary32` arithmetic.

The optional extra check reads the certificate data as binary32 values, runs the polynomial
evaluation with executable IEEE-754 operations and tests whether the results hit the claimed
endpoints. The local coordinate `hi - lo` comes from a binary32 subtraction, which can round.

The check is deliberately strict:
- the knot partition and polynomial metadata must be structurally valid,
- every rational in the cert must be exactly representable as a finite `Binary32`,
- comparisons use IEEE-style `BEq` (so `+0 == -0`, and NaNs never compare equal).
-/
def Certificate.checkBinary32 (cert : Certificate) : IO Unit := do
  checkStructure cert
  let xsQ := Tensor.to cert.xs (Array Rat)
  let ysQ := Tensor.to cert.ys (Array Rat)

  let xs ← xsQ.mapIdxM (fun i q => ratToBinary32 (ctx := s!"xs[{i}]") q)
  let ys ← ysQ.mapIdxM (fun i q => ratToBinary32 (ctx := s!"ys[{i}]") q)

  for i in [0:cert.pieces.size] do
    let p ← requireArrayEntry "pieces" cert.pieces i
    let lo32 ← requireArrayEntry "xs" xs i
    let hi32 ← requireArrayEntry "xs" xs (i + 1)
    let yLo32 ← requireArrayEntry "ys" ys i
    let yHi32 ← requireArrayEntry "ys" ys (i + 1)

    let coeffs32 ←
      p.coeffs.mapIdxM (fun k q => ratToBinary32 (ctx := s!"pieces[{i}].coeffs[{k}]") q)

    let tHi32 : Binary32 := hi32 - lo32
    let pLo32 := evalPolyHorner coeffs32 0
    let pHi32 := evalPolyHorner coeffs32 tHi32

    unless pLo32 == yLo32 do
      throw <|
        IO.userError
          (s!"binary32 endpoint mismatch at i={i}: p(lo)={pLo32} ≠ y={yLo32} " ++
            s!"(bits {ExecFloat.Binary.toBits32 pLo32} vs " ++
            s!"{ExecFloat.Binary.toBits32 yLo32})")
    unless pHi32 == yHi32 do
      throw <|
        IO.userError
          (s!"binary32 endpoint mismatch at i={i}: p(hi)={pHi32} ≠ yNext={yHi32} " ++
            s!"(bits {ExecFloat.Binary.toBits32 pHi32} vs " ++
            s!"{ExecFloat.Binary.toBits32 yHi32})")

  IO.println "binary32 check verified (exact representability + endpoint equalities)."

/-- Check a piecewise-polynomial certificate stored as JSON exactly over `Rat`, then replay its
endpoint equalities under `Binary32` arithmetic. -/
def checkJsonBinary32 (j : Json) : IO Unit := do
  let cert ← Certificate.parse j
  Certificate.checkRat cert
  IO.println "Piecewise polynomial certificate verified."
  Certificate.checkBinary32 cert

end TorchLean.Examples.Splines
