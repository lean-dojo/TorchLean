/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Util.Json

/-!
# VerifyMarginCert

Lean checker for per-example logit-margin certificates (`robust_margin_cert_v0_1`).

This checker is intentionally lightweight: it does **not** re-run bound propagation.
Instead, it verifies that the exported logit bounds imply the certified margin predicate:

  `logits_lo[label] > max_{j≠label} logits_hi[j]`

Run:
  `python3 scripts/verification/robustness/export_margin_cert.py`
  `lake exe verify -- margin-cert`
-/

@[expose] public section


namespace NN.Examples.Verification.Robustness.VerifyMarginCert

open Lean
open Data
open NN.Verification.Json

/-- Boolean `<` on `Float`. -/
def ltBool (x y : Float) : Bool := decide (x < y)
/-- Boolean `≤` on `Float`. -/
def leBool (x y : Float) : Bool := decide (x ≤ y)

/--
Check the logit-margin predicate from exported bounds.

Returns `true` if `lo[label] > max_{j≠label} hi[j]`.
-/
def certifiedTop1 (lo hi : Array Float) (label : Nat) : Bool :=
  if lo.size = hi.size then
    if label < lo.size then
      let loY := lo[label]!
      let maxOther? :=
        (List.range lo.size).foldl (fun (acc : Option Float) (i : Nat) =>
          if i = label then acc
          else
            match acc with
            | none => some hi[i]!
            | some m => some (max m hi[i]!)) none
      match maxOther? with
      | none => true
      | some m => ltBool m loY
    else
      false
  else
    false

/-- Running counters for reporting nominal vs certified accuracy. -/
structure Counters where
  /-- total. -/
  total : Nat := 0
  /-- nominal Ok. -/
  nominalOk : Nat := 0
  /-- certified Ok. -/
  certifiedOk : Nat := 0
  deriving Repr

/-!
Internal helpers.
-/
def checkOneExample (numClasses : Nat) (ex : Json) : IO (Bool × Bool) := do
  let exObj ← expectObj ex "example"
  let label ← expectFieldNat exObj "label" "example"
  let lo ← expectFieldFloatArray exObj "logits_lo" "example"
  let hi ← expectFieldFloatArray exObj "logits_hi" "example"
  if lo.size ≠ numClasses || hi.size ≠ numClasses then
    throw <| IO.userError s!"example logits length mismatch (expected {numClasses})"
  if !all2 lo hi leBool then
    throw <| IO.userError "example has invalid bounds (lo ≤ hi violated)"
  let cert := certifiedTop1 lo hi label

  -- Optional: cross-check exported 'certified' flag if present.
  match ← optionalFieldBool? exObj "certified" "example" with
  | some b =>
      if b != cert then
        throw <| IO.userError "example.certified does not match margin predicate"
  | none => pure ()

  -- Optional: nominal correctness from exported 'pred'.
  let nominalOk :=
    match ← optionalFieldNat? exObj "pred" "example" with
    | some p => decide (p = label)
    | none => false

  pure (nominalOk, cert)

/--
Check a `robust_margin_cert_v0_1` JSON certificate file.

If `timing=true`, prints per-example timings every `timingEvery` examples.
-/
def checkWithTiming (path : String) (timing : Bool) (timingEvery : Nat) : IO Unit := do
  let topObj ← readJsonObjectFile path
  expectFormat topObj "robust_margin_cert_v0_1"
  let numClasses ← expectFieldNat topObj "num_classes" "top-level"
  let examples ← expectFieldArray topObj "examples" "top-level"

  let timeMs {α : Type} (act : IO α) : IO (α × Float) := do
    let t0 ← IO.monoNanosNow
    let a ← act
    let t1 ← IO.monoNanosNow
    let ms := (t1 - t0).toFloat / 1_000_000.0
    pure (a, ms)

  let mut counts : Counters := {}
  let mut totalMs : Float := 0.0
  let mut maxMs : Float := 0.0
  for ex in examples do
    if timing then
      let ((nominalOk, cert), ms) ← timeMs (checkOneExample numClasses ex)
      counts := { counts with
        total := counts.total + 1
        nominalOk := counts.nominalOk + (if nominalOk then 1 else 0)
        certifiedOk := counts.certifiedOk + (if cert then 1 else 0) }
      totalMs := totalMs + ms
      if ms > maxMs then
        maxMs := ms
      if timingEvery > 0 && counts.total % timingEvery == 0 then
        IO.println s!"[margin cert] example {counts.total}: {ms} ms"
    else
      let (nominalOk, cert) ← checkOneExample numClasses ex
      counts := { counts with
        total := counts.total + 1
        nominalOk := counts.nominalOk + (if nominalOk then 1 else 0)
        certifiedOk := counts.certifiedOk + (if cert then 1 else 0) }

  IO.println s!"[margin cert] examples={counts.total}"
  IO.println s!"[margin cert] nominal_ok={counts.nominalOk} (requires 'pred' in examples)"
  IO.println s!"[margin cert] certified_ok={counts.certifiedOk}"
  if timing then
    let avgMs := if counts.total == 0 then 0.0 else totalMs / counts.total.toFloat
    IO.println s!"[margin cert] timing avg_ms={avgMs} max_ms={maxMs}"

  -- Optional: verify summary fields if present.
  match ← optionalField? topObj "summary" "top-level" with
  | none => pure ()
  | some summaryJ =>
      let summaryObj ← expectObj summaryJ "summary"
      let checkNatField (k : String) (v : Nat) : IO Unit := do
        match ← optionalFieldNat? summaryObj k "summary" with
        | none => pure ()
        | some n =>
            if n != v then
              throw <| IO.userError s!"summary.{k} mismatch (expected {v}, got {n})"
      checkNatField "examples" counts.total
      checkNatField "nominal_ok" counts.nominalOk
      checkNatField "certified_ok" counts.certifiedOk

/-- Check a margin cert file with timing disabled (thin wrapper around `checkWithTiming`). -/
def check (path : String) : IO Unit :=
  checkWithTiming path false 0

/--
CLI entry point: `lake exe verify -- margin-cert [cert.json]`.

If `--timing` is passed, prints timing information (optionally controlled by `--timing-every=N`).
-/
def run (args : List String) : IO Unit := do
  let defaultPath := "NN/Examples/Verification/Robustness/digits_linear_margin_cert.json"

  let usage :=
    String.intercalate "\n" [
      "Usage:",
      "  lake exe verify -- margin-cert [<path/to/cert.json>]",
      "",
      "If no path is provided, uses the bundled digits cert:",
      s!"  {defaultPath}",
      "",
      "To (re)generate the cert:",
      "  python3 scripts/verification/robustness/export_margin_cert.py"
    ]

  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return

  let timing := args.contains "--timing"
  let timingEvery :=
    match (args.find? (fun a => a.startsWith "--timing-every=")) with
    | some a =>
        match (a.drop 16).toNat? with
        | some n => n
        | none => 0
    | none => 0
  let args' := args.filter (fun a => !(a == "--timing" || a.startsWith "--timing-every="))
  let path :=
    match args' with
    | "--" :: rest => rest.getD 0 defaultPath
    | _ => args'.getD 0 defaultPath
  checkWithTiming path timing timingEvery

end NN.Examples.Verification.Robustness.VerifyMarginCert
