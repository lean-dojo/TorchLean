/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.Runtime.External.Process
import Lean

/-!
# Arb oracle (python-flint) integration

This module provides a small Lean wrapper around an external Arb/FLINT process:

- spawn `python` to run `NN/Floats/Arb/arb_oracle.py`,
- parse its JSON output,
- expose the result in a structured way (including an exact `Rat` enclosure reconstructed from
  Arb's `mid_rad_10exp` integers).

Trust boundary: this is an **oracle**. The Lean side parses and packages the returned enclosure; it
does not prove the Arb computation itself.

## Notes on design

We expose two layers:
- a *raw* layer returning `Lean.Json` (`runJson`, `runRequestJson`), and
- a *typed* layer with small parsing functions (`parseResult`) and convenience wrappers (`run`,
  `runExpr`, `runMLP`).

The raw layer is useful for debugging and for experimenting with new request schemas, while the
typed layer is nicer for examples and "certificate-like" workflows.
-/

@[expose] public section


namespace TorchLean.Floats.Arb

open Lean

/-!
## Unary queries (compatibility)

The original Lean integration called the oracle in “unary” mode via `--func`, `--lo`, `--hi`.
This API remains because it is convenient for quick demos.

For more general evaluation (expression ASTs / small MLPs), use the request API further below.
-/

/--
Unary interval query sent to the Python Arb oracle.

In words: evaluate `func` over the input interval `[lo, hi]` using Arb ball arithmetic at
the requested precision, and return a rigorous enclosure of the output.

Why endpoints are strings:
- we pass them straight through to Python/Arb without any intermediate parsing in Lean, and
- it avoids accidental loss of precision from `Float` parsing when writing demos.
-/
structure Query where
  /-- Unary function name (e.g. `"tanh"`, `"exp"`, `"log"`). -/
  func : String
  /-- Lower endpoint of the input interval, as a decimal string. -/
  lo : String
  /-- Upper endpoint of the input interval, as a decimal string. -/
  hi : String
  /-- Working precision (in bits) used inside Arb. -/
  precBits : Nat := 200
  /-- Number of decimal digits used in printed endpoints. -/
  digits : Nat := 50
  deriving Repr, Inhabited

/--
Arb `mid_rad_10exp` ball encoding.

In words: this represents the *exact* rational enclosure
`[(mid - rad) * 10^exp, (mid + rad) * 10^exp]`.

The Python oracle emits `mid`, `rad`, and `exp` as strings (then we parse them into `Int`) to avoid
JSON integer-size limitations.
-/
structure MidRad10Exp where
  /-- Midpoint integer. -/
  mid : Int
  /-- Radius integer (nonnegative for well-formed payloads). -/
  rad : Int
  /-- Base-10 exponent scaling both `mid` and `rad`. -/
  exp : Int
  deriving Repr, Inhabited

/--
Typed result for a unary query.

We keep both:
- the raw decimal endpoints (`outLo`/`outHi`) for logs, and
- the exact `MidRad10Exp` ball (`outputBall`) that can be converted into exact `Rat` bounds.
-/
structure Result where
  /-- Echoed `func` name. -/
  func : String
  /-- Working precision (bits) used by Arb. -/
  precBits : Nat
  /-- Decimal digits used in printed endpoints. -/
  digits : Nat
  /-- Input interval as an Arb ball (exact integer encoding). -/
  inputBall : MidRad10Exp
  /-- Output interval as an Arb ball (exact integer encoding). -/
  outputBall : MidRad10Exp
  /-- Lower endpoint as a decimal string (human-friendly). -/
  outLo : String
  /-- Upper endpoint as a decimal string (human-friendly). -/
  outHi : String
  deriving Repr, Inhabited

/-!
## General request API (`kind = "expr"` / `"mlp"`)

The Python oracle supports JSON requests (see `NN/Floats/Arb/arb_oracle.py`):

- `kind = "expr"`: evaluate a small, whitelisted expression language over Arb balls.
- `kind = "mlp"`: evaluate a small feedforward MLP given weights/biases/activations, over a box
  input.

These are still **oracle calls**; they let us move beyond one unary op while keeping the external
computation boundary explicit.
-/

/--
Typed result for an `expr` request.

In words: an enclosure of a small expression AST evaluated over a box of interval variables.
-/
structure ExprResult where
  /-- Working precision (bits) used by Arb. -/
  precBits : Nat
  /-- Decimal digits used in printed endpoints. -/
  digits : Nat
  /-- Output enclosure encoded as `mid_rad_10exp`. -/
  outputBall : MidRad10Exp
  /-- Lower decimal endpoint of the enclosure. -/
  outLo : String
  /-- Upper decimal endpoint of the enclosure. -/
  outHi : String
  deriving Repr, Inhabited

/--
Typed result for an `mlp` request.

The output is a vector of per-output-coordinate enclosures `(ball, lo, hi)`.
-/
structure MLPResult where
  /-- Working precision (bits) used by Arb. -/
  precBits : Nat
  /-- Decimal digits used in printed endpoints. -/
  digits : Nat
  /-- Per-output enclosures, each paired with human-readable endpoints. -/
  output : Array (MidRad10Exp × String × String)
  deriving Repr, Inhabited

/-- Relative path to the Python oracle entrypoint script (used in all invocation modes). -/
def oracleScriptPath : String :=
  "NN/Floats/Arb/arb_oracle.py"

/--
Resolve which Python executable to use.

In words: if the environment variable `TORCHLEAN_ARB_PY` is set, use it; otherwise fall
back to the provided `pythonCmd` (default: `"python3"`).
-/
def resolvePythonCmd (pythonCmd : String) : IO String := do
  -- Allow a project-wide override for non-default python environments.
  Runtime.External.Process.resolveCmdFromEnv "TORCHLEAN_ARB_PY" pythonCmd

/--
Run the oracle script as a subprocess and parse its stdout as JSON.

This is the shared IO boundary used by both unary queries and general JSON requests.
-/
def runPythonJson (pythonCmd : String) (args : Array String) : IO Json := do
  let pythonCmd ← resolvePythonCmd pythonCmd
  Runtime.External.Process.runJsonStdoutChecked (ctx := "Arb oracle")
    (cmd := pythonCmd) (args := args) (cwd := some ".")

/-- Internal JSON helper: interpret a JSON string as a `String`, or return an error. -/
def jsonToString (j : Json) : Except String String :=
  match j with
  | .str s => .ok s
  | _ => .error s!"Expected string, got {j}"

/--
Internal JSON helper: interpret a JSON number as a `Nat`, or return an error.

The oracle schema uses Nats only for metadata fields such as precision/digits.
-/
def jsonToNat (j : Json) : Except String Nat :=
  match j with
  | .num n =>
    -- `Json.num` stores a `Scientific` number. The oracle schema uses natural numbers here, so
    -- parsing from the decimal representation is sufficient (and keeps this helper
    -- dependency-free).
    match n.toString.toNat? with
    | some k => .ok k
    | none => .error s!"Expected Nat, got number {n}"
  | _ => .error s!"Expected number, got {j}"

/--
Internal JSON helper: read an integer encoded as a decimal *string* from the given object field.

The oracle emits `mid`/`rad`/`exp` as strings to avoid JSON integer-size limitations.
-/
def jsonToIntFromStringKey (o : Json) (k : String) : Except String Int := do
  let s ← o.getObjVal? k >>= jsonToString
  match s.toInt? with
  | some i => .ok i
  | none => .error s!"Expected Int string at key '{k}', got '{s}'"

/--
Parse an Arb `mid_rad_10exp` object into `MidRad10Exp`.

In words: this reads the exact integer encoding of an interval ball.
-/
def parseMidRad10Exp (j : Json) : Except String MidRad10Exp := do
  let mid ← jsonToIntFromStringKey j "mid"
  let rad ← jsonToIntFromStringKey j "rad"
  let exp ← jsonToIntFromStringKey j "exp"
  pure { mid, rad, exp }

/--
Convert a `MidRad10Exp` ball into an exact rational enclosure.

In words: if the payload encodes `mid ± rad` scaled by `10^exp`, then
`toRatBounds` returns the pair
`((mid - rad) * 10^exp, (mid + rad) * 10^exp)` as exact rationals.
-/
def MidRad10Exp.toRatBounds (m : MidRad10Exp) : Rat × Rat :=
  let pow10Int (n : Nat) : Int :=
    Int.ofNat (Nat.pow 10 n)
  let scale (z : Int) : Rat :=
    if m.exp ≥ 0 then
      let e : Nat := Int.toNat m.exp
      Rat.ofInt (z * pow10Int e)
    else
      let e : Nat := Int.toNat (-m.exp)
      (Rat.ofInt z) / (Rat.ofInt (pow10Int e))
  let loZ := m.mid - m.rad
  let hiZ := m.mid + m.rad
  (scale loZ, scale hiZ)

/--
Render a unary `Query` into the CLI arguments expected by `arb_oracle.py` in unary mode.

This is separated out so tools can log or override arguments more easily.
-/
def Query.toArgs (q : Query) : Array String :=
  #[
    oracleScriptPath,
    "--func", q.func,
    s!"--lo={q.lo}",
    s!"--hi={q.hi}",
    "--prec-bits", toString q.precBits,
    "--digits", toString q.digits
  ]

/--
Run a unary `Query` via the Python oracle and return the raw JSON payload.

This only checks that:
- the process exits successfully, and
- the output parses as JSON.

Use `parseResult` if you want the typed `Result`.
-/
def runJson (q : Query) (pythonCmd : String := "python3") : IO Json := do
  runPythonJson pythonCmd q.toArgs

/--
Parse the `ctx` object shared by all oracle responses.

In words: extract `(precBits, digits)` from the payload metadata.
-/
def parseCtx (j : Json) : Except String (Nat × Nat) := do
  let ctx ← j.getObjVal? "ctx"
  let precBits ← ctx.getObjVal? "prec_bits" >>= jsonToNat
  let digits ← ctx.getObjVal? "digits" >>= jsonToNat
  pure (precBits, digits)

/--
Internal helper: parse a `ball` field (in `mid_rad_10exp` format) from an object.

This is used by both unary results and request-mode results.
-/
def parseBallField (o : Json) : Except String MidRad10Exp := do
  let ballJson ← o.getObjVal? "ball"
  parseMidRad10Exp ballJson

/--
Internal helper: parse `(lo, hi)` endpoint strings from an object.

The oracle includes both human-friendly decimal endpoints and exact integer ball encodings.
-/
def parseLoHi (o : Json) : Except String (String × String) := do
  let lo ← o.getObjVal? "lo" >>= jsonToString
  let hi ← o.getObjVal? "hi" >>= jsonToString
  pure (lo, hi)

/--
Internal helper: parse a `ball` plus decimal `(lo, hi)` endpoints from an object.

The oracle includes both:
- human-friendly decimal endpoints (`lo`/`hi`), and
- an exact `mid_rad_10exp` ball encoding (`ball`).
-/
def parseBallLoHi (o : Json) : Except String (MidRad10Exp × String × String) := do
  let ball ← parseBallField o
  let (lo, hi) ← parseLoHi o
  pure (ball, lo, hi)

/--
Parse a unary-mode JSON payload into a typed `Result`.

In words: if the JSON matches the schema emitted by `arb_oracle.py --func ...`, then
`parseResult` extracts the input/output enclosures and context parameters; otherwise it returns a
human-readable error message.
-/
def parseResult (j : Json) : Except String Result := do
  let func ← j.getObjVal? "func" >>= jsonToString
  let (precBits, digits) ← parseCtx j
  let input ← j.getObjVal? "input"
  let inputBall ← parseBallField input
  let output ← j.getObjVal? "output"
  let (outputBall, outLo, outHi) ← parseBallLoHi output
  pure { func, precBits, digits, inputBall, outputBall, outLo, outHi }

/--
Run a unary `Query` and parse the result.

This is the convenience wrapper most demos should use.

It respects `TORCHLEAN_ARB_PY` (if set) to choose the Python executable.
-/
def run (q : Query) (pythonCmd : String := "python3") : IO Result := do
  let j ← runJson q pythonCmd
  match parseResult j with
  | .ok r => pure r
  | .error msg => throw <| IO.userError s!"Arb oracle result parse error: {msg}\njson:\n{j}"

/--
Ensure the temp directory used for request files exists.

We write request JSON payloads to `.lake/build/tmp/` so they are available for debugging if the
oracle subprocess fails or its output cannot be parsed. On success, we delete the file unless
`TORCHLEAN_ARB_KEEP_TMP` is set.
-/
def ensureTmpDir : IO Unit :=
  IO.FS.createDirAll ".lake/build/tmp"

/-- Return `true` iff request payload files should be kept on disk even on success. -/
def keepTmpRequests : IO Bool := do
  pure <| (← IO.getEnv "TORCHLEAN_ARB_KEEP_TMP") |>.isSome

/-- Best-effort file removal helper (ignore errors). -/
def tryRemoveFile (path : System.FilePath) : IO Unit := do
  try
    IO.FS.removeFile path
  catch _ =>
    pure ()

/--
Generate a fresh temporary filepath for a request payload.

In words: use the current monotone time in milliseconds plus a random suffix to reduce
collisions across concurrent processes.
-/
def freshReqPath : IO System.FilePath := do
  let t ← IO.monoMsNow
  let r ← IO.rand 0 (Nat.pow 2 63 - 1)
  pure <| System.FilePath.mk s!".lake/build/tmp/arb_req_{t}_{r}.json"

/--
Run the oracle with a general `--request <file.json>` payload (returns raw JSON).

This is the entrypoint for the richer request schemas supported by `arb_oracle.py`, such as:
- `kind = "expr"` (expression AST evaluation), and
- `kind = "mlp"` (small feedforward MLP evaluation).

This respects `TORCHLEAN_ARB_PY` (if set) to choose the Python executable.
-/
def runRequestJson (req : Json) (precBits : Nat := 200) (digits : Nat := 50)
    (pythonCmd : String := "python3") : IO Json := do
  ensureTmpDir
  let path ← freshReqPath
  IO.FS.writeFile path req.pretty
  try
    let j ← runPythonJson pythonCmd #[
      oracleScriptPath,
      "--request", path.toString,
      "--prec-bits", toString precBits,
      "--digits", toString digits
    ]
    if !(← keepTmpRequests) then
      tryRemoveFile path
    pure j
  catch e =>
    -- Keep the request payload on disk for debugging.
    throw e

/--
Parse an `expr` response payload into `ExprResult`.

In words: extract the output enclosure and context metadata from the response JSON.
-/
def parseExprResult (j : Json) : Except String ExprResult := do
  let (precBits, digits) ← parseCtx j
  let output ← j.getObjVal? "output"
  let (outputBall, outLo, outHi) ← parseBallLoHi output
  pure { precBits, digits, outputBall, outLo, outHi }

/--
Parse an `mlp` response payload into `MLPResult`.

In words: extract the vector of per-coordinate output enclosures and context metadata.
-/
def parseMLPResult (j : Json) : Except String MLPResult := do
  let (precBits, digits) ← parseCtx j
  let output ← j.getObjVal? "output"
  let vec ← output.getObjVal? "vector"
  let arr ← vec.getArr?
  let out ← arr.mapM (fun yi => do
    let (ball, lo, hi) ← parseBallLoHi yi
    pure (ball, lo, hi)
  )
  pure { precBits, digits, output := out }

/--
Run an `expr` request and parse the output interval.

In words: evaluate an expression AST over a box of interval variables using Arb ball
arithmetic, returning an enclosure of the result.

This is a convenience wrapper around `runRequestJson` with `kind = "expr"`.
-/
def runExpr (vars : List (String × (String × String))) (expr : Json)
    (precBits : Nat := 200) (digits : Nat := 50) (pythonCmd : String := "python3") : IO ExprResult
      := do
  let varsObj : List (String × Json) :=
    vars.map (fun (name, (lo, hi)) =>
      (name, Json.mkObj [("lo", Json.str lo), ("hi", Json.str hi)]))
  let req :=
    Json.mkObj [
      ("kind", Json.str "expr"),
      ("vars", Json.mkObj varsObj),
      ("expr", expr)
    ]
  let j ← runRequestJson req (precBits := precBits) (digits := digits) pythonCmd
  match parseExprResult j with
  | .ok r => pure r
  | .error msg => throw <| IO.userError s!"Arb oracle expr parse error: {msg}\njson:\n{j}"

/--
Run an `mlp` request (box input + layers) and parse the output vector intervals.

In words: run a small feedforward MLP using ball arithmetic, where the input is a box
interval, and return per-output-coordinate enclosures.

This is a convenience wrapper around `runRequestJson` with `kind = "mlp"`.
-/
def runMLP (req : Json) (precBits : Nat := 200) (digits : Nat := 50) (pythonCmd : String :=
  "python3") : IO MLPResult := do
  let j ← runRequestJson req (precBits := precBits) (digits := digits) pythonCmd
  match parseMLPResult j with
  | .ok r => pure r
  | .error msg => throw <| IO.userError s!"Arb oracle mlp parse error: {msg}\njson:\n{j}"

end TorchLean.Floats.Arb
