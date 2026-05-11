/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Util.Json

/-!
# AbCrown Leaf Certificate

Alpha-beta-CROWN (AbCrown) leaf-certificate checker.

This module checks a small JSON certificate format (`abcrown_leaf_cert_v0_1`) exported by a Python
verification pipeline. It does **not** run bound propagation itself; instead it sanity-checks that:
- each leaf input box is nested inside the declared root input box, and
- each leaf contains a witness that refutes the unsafe threshold (`lb[i] > threshold[i]` for some
  `i`).

This is useful for:
- regression testing JSON export/import paths, and
- reviewer-friendly certificate validation workflows.

References:
- beta-CROWN paper (NeurIPS 2021): `https://arxiv.org/abs/2103.06624`
- alpha-beta-CROWN implementation: `https://github.com/Verified-Intelligence/alpha-beta-CROWN`

Run:
`lake exe verify -- abcrown-leaf [path/to/cert.json]`
-/

@[expose] public section


namespace NN.Verification.Cert.AbCrownLeafCert

open Lean
open Data
open NN.Verification.Json

/-- Boolean `≤` on `Float` (used for array-wise box comparisons). -/
def leBool (x y : Float) : Bool := decide (x ≤ y)
/-- Boolean `<` on `Float` (used for strict refutation checks). -/
def ltBool (x y : Float) : Bool := decide (x < y)

/--
Check that a leaf box `[lo, hi]` is nested within a declared root box `[rootLo, rootHi]`.

All arrays are interpreted pointwise.
-/
def boxWithin (rootLo rootHi lo hi : Array Float) : Bool :=
  all2 rootLo lo leBool && all2 lo hi leBool && all2 hi rootHi leBool

/--
Check that a leaf is verified by a lower-bound vector `lb` against an unsafe threshold `thr`.

Interpretation: `lb` is a certified lower bound on some "violation score". If `lb[i] > thr[i]` for
some index `i`, then that unsafe constraint is refuted.
-/
def leafVerified (lb thr : Array Float) : Bool :=
  any2 lb thr (fun l t => ltBool t l)  -- ∃i, lb[i] > threshold[i]

/--
Like `leafVerified`, but first try a supplied witness index.

If the witness index is out of range, this returns `false` (callers usually fall back to
`leafVerified`).
-/
def leafVerifiedAt (lb thr : Array Float) (witnessIdx : Nat) : Bool :=
  if witnessIdx < lb.size ∧ witnessIdx < thr.size then
    ltBool thr[witnessIdx]! lb[witnessIdx]!
  else
    false

/--
Parse and validate a `abcrown_leaf_cert_v0_1` JSON certificate.

On failure this throws `IO.userError` with a brief message.
-/
def checkAbCrownLeafCertV01 (path : String) : IO Unit := do
  let topObj ← readJsonObjectFile path
  expectFormat topObj "abcrown_leaf_cert_v0_1"

  let rootObj ← expectFieldObj topObj "root" "top-level"
  let rootLo ← expectFieldFloatArray rootObj "lo" "root"
  let rootHi ← expectFieldFloatArray rootObj "hi" "root"

  let leaves ← expectFieldArray topObj "leaves" "top-level"
  if leaves.isEmpty then
    IO.println "[cert] Warning: leaves list is empty (nothing to check)"

  let mut okCount := 0
  let mut badCount := 0
  for leaf in leaves do
    let leafObj ← expectObj leaf "leaf"
    let lo ← expectFieldFloatArray leafObj "lo" "leaf"
    let hi ← expectFieldFloatArray leafObj "hi" "leaf"
    let lb ← expectFieldFloatArray leafObj "lb" "leaf"
    let thr ← expectFieldFloatArray leafObj "threshold" "leaf"

    let within := boxWithin rootLo rootHi lo hi
    let verified :=
      match ← optionalFieldNat? leafObj "witness_idx" "leaf" with
      | some wi => leafVerifiedAt lb thr wi || leafVerified lb thr
      | none => leafVerified lb thr
    if within && verified then
      okCount := okCount + 1
    else
      badCount := badCount + 1

  IO.println s!"[cert] Checked {leaves.size} leaves: ok={okCount}, bad={badCount}"
  if badCount > 0 then
    throw <| IO.userError s!"Certificate failed checks for {badCount} leaves"

/--
CLI entry point: `lake exe verify -- abcrown-leaf [cert.json]`.

If no path is provided, checks a small bundled sample certificate under
`NN/Examples/Verification/AbCrown/`.
-/
def run (args : List String) : IO Unit := do
  let defaultCertPath :=
    "NN/Examples/Verification/AbCrown/sample_abcrown_leaf_cert_v0_1.json"

  let usage :=
    String.intercalate "\n" [
      "Usage:",
      "  lake exe verify -- abcrown-leaf [<path/to/cert.json>]",
      "",
      "If no path is provided, runs a small bundled sample cert:",
      s!"  {defaultCertPath}"
    ]

  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return

  let path := args.getD 0 defaultCertPath
  checkAbCrownLeafCertV01 path

end NN.Verification.Cert.AbCrownLeafCert
