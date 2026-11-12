/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

meta import TorchLean.Verify.RuntimeApprox.Roundoff.Baseline

/-!
# The comparison of the roundoff baseline

Each case runs `TorchLean.Roundoff.Baseline.diff` on rows of the file and rows of the environment
that differ in one way, and checks the lines it gives. A lower status, a larger `r`, `a` or `ε`
at a format, also behind a term with the same hash, a smaller `r` with a larger `a`, a changed
budget without values, a budget without values whose definitions changed, a new premise, a new
caveat, a changed reference and a gone analysis are regressions. A new analysis, a better status,
removed premises and a budget whose term or definitions changed with the same values make the
file stale. Equal rows give nothing. The last cases check that a row read back from its line in
the file equals it, that the meaning of a constant follows the meanings of the constants it uses,
the values of the unit roundoff and the underflow error at binary32 and binary64, and the line of
the check without a file or with an unknown mode, which writes no file.
-/

meta section

namespace TorchLeanTest.RoundoffBaseline

open Lean Elab Command TorchLean.Roundoff TorchLean.Roundoff.Baseline
open FloatLib.Floats.Formats.BinaryInterchange

-- A zero budget against a reference that keeps a rounded branch is not an a priori bound.
run_cmd liftTermElabM do
  let check (termSyntax : Syntax) (conditional : Bool) : TermElabM Unit := do
    let computation ← Term.elabTerm termSyntax none
    let proof ← generateProof computation false
    let entry ← describeTheorem
      { family := `Audit, operator := `selection, route := "value", backend := "configured",
        mode := "absolute", status := .conditional }
      2 (← Meta.inferType proof) proof #[]
    unless (entry.status == .conditional) == conditional do
      throwError "the rounded branch was classified incorrectly: {entry.reference}"
  check (← `(fun (x y : FloatLib.Floats.Binary32) =>
    FloatLib.Floats.ExecFloat.Binary.toReal (if x < y then x else y))) false
  check (← `(fun (x y : FloatLib.Floats.Binary32) =>
    FloatLib.Floats.ExecFloat.Binary.toReal (if x < -1 then x else y))) false
  check (← `(fun (x y : FloatLib.Floats.Binary32) =>
    FloatLib.Floats.ExecFloat.Binary.toReal (if x * x < y then x else y))) true
  check (← `(fun (x y : FloatLib.Floats.Binary32) =>
    FloatLib.Floats.ExecFloat.Binary.toReal (if x * x = y then x else y))) true
  check (← `(fun (x y : FloatLib.Floats.Binary32) =>
    FloatLib.Floats.ExecFloat.Binary.toReal
      (if FloatLib.Floats.ExecFloat.Binary.isNaN (x * x) then x else y))) true

/-- The budget `name` with the hash `hash` and the values `values` at binary16, bfloat16,
binary32 and binary64. -/
def budget (name : String) (hash : UInt64) (values : Array Rat) : Budget :=
  { name, hash, term := s!"{name}{hash}"
    values := (formats.map (·.1)).zip (values.map some) }

/-- A conditional mixed analysis with one premise and one caveat. -/
def mixed : Row :=
  { family := "Pointwise", operator := "exp", route := "value", backend := "configured",
    mode := "mixed", status := .conditional, reference := 7, referenceSize := 3,
    budgets := #[budget "r" 1 #[4, 2, 1, 1/2], budget "a" 2 #[4, 2, 1, 1/2]],
    premises := #[(5, "x ≤ 1")], caveats := #["the input is finite"] }

/-- The absolute analysis of the same route, without premises or caveats. -/
def absolute : Row :=
  { mixed with
    mode := "absolute", status := .proved, budgets := #[budget "ε" 3 #[4, 2, 1, 1/2]],
    premises := #[], caveats := #[] }

/-- Log the differences between the rows `old` of the file and the rows `new`, one line each,
marked `regression` or `stale`. -/
def report (old new : Array Row) : CommandElabM Unit := do
  let lines := (diff old new).map fun difference =>
    (if difference.regression then "regression " else "stale ") ++ difference.line
  unless lines.isEmpty do logInfo ("\n".intercalate lines.toList)

/-- info: regression roundoff baseline: exp [value] configured mixed: status conditional → failed -/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with
  status := .failed, reference := 0, referenceSize := 0, budgets := #[], premises := #[],
  caveats := #[] }]

/-- info: regression roundoff baseline: exp [value] configured mixed: budget r weaker at binary32:
1 → 2 -/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with budgets := #[budget "r" 11 #[4, 2, 2, 1/2], budget "a" 2
  #[4, 2, 1, 1/2]] }]

/--
info: regression roundoff baseline: exp [value] configured mixed: budget a weaker at bfloat16:
2 → 4
regression roundoff baseline: exp [value] configured mixed: budget a weaker at binary64: 1/2 → 1
-/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with budgets := #[budget "r" 1 #[4, 2, 1, 1/2], budget "a" 12
  #[4, 4, 1, 1]] }]

/--
info: regression roundoff baseline: exp [value] configured absolute: budget ε weaker at binary16:
4 → 8
-/
#guard_msgs in
run_cmd report #[absolute] #[{ absolute with budgets := #[budget "ε" 13 #[8, 2, 1, 1/2]] }]

/--
info: stale roundoff baseline: exp [value] configured mixed: improved (budget r smaller at
binary16, bfloat16, binary32, binary64); update the baseline with
TORCHLEAN_ROUNDOFF_BASELINE=update
regression roundoff baseline: exp [value] configured mixed: budget a weaker at binary64: 1/2 → 1
-/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with budgets := #[budget "r" 11 #[2, 1, 1/2, 1/4], budget "a" 12
  #[4, 2, 1, 1]] }]

/--
info: regression roundoff baseline: exp [value] configured mixed: budget r changed and is not
comparable: B * u → 2 * B * u
-/
#guard_msgs in
run_cmd do
  let values : Array (String × Option Rat) := (formats.map (·.1)).map (·, none)
  report #[{ mixed with budgets := #[{ name := "r", hash := 1, term := "B * u", values }] }]
    #[{ mixed with budgets := #[{ name := "r", hash := 21, term := "2 * B * u", values }] }]

/--
info: stale roundoff baseline: exp [value] configured mixed: budget r changed with the same values;
update the baseline with TORCHLEAN_ROUNDOFF_BASELINE=update
-/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with budgets := #[budget "r" 11 #[4, 2, 1, 1/2], budget "a" 2
  #[4, 2, 1, 1/2]] }]

-- A larger value is a regression also when the term has the same hash, as when a definition it
-- uses changed.
/-- info: regression roundoff baseline: exp [value] configured mixed: budget r weaker at binary32:
1 → 2 -/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with budgets := #[budget "r" 1 #[4, 2, 2, 1/2], budget "a" 2
  #[4, 2, 1, 1/2]] }]

/--
info: stale roundoff baseline: exp [value] configured mixed: budget r changed with the same values;
update the baseline with TORCHLEAN_ROUNDOFF_BASELINE=update
-/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with budgets := #[{ budget "r" 1 #[4, 2, 1, 1/2] with
  definitions := 9 }, budget "a" 2 #[4, 2, 1, 1/2]] }]

-- A budget without values is compared by the hashes of its term and of its definitions.
/--
info: regression roundoff baseline: exp [value] configured mixed: budget r: a definition that its
term B * u uses changed, and it is not comparable
-/
#guard_msgs in
run_cmd do
  let values : Array (String × Option Rat) := (formats.map (·.1)).map (·, none)
  let symbolic (definitions : UInt64) : Row :=
    { mixed with budgets := #[{ name := "r", hash := 1, definitions, term := "B * u", values }] }
  report #[symbolic 4] #[symbolic 4]
  report #[symbolic 4] #[symbolic 5]

/-- info: regression roundoff baseline: exp [value] configured mixed: new premise: y ≤ 1 -/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with premises := #[(5, "x ≤ 1"), (6, "y ≤ 1")] }]

/--
info: stale roundoff baseline: exp [value] configured mixed: improved (premises removed: 1);
update the baseline with TORCHLEAN_ROUNDOFF_BASELINE=update
-/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with premises := #[] }]

/--
info: regression roundoff baseline: exp [value] configured mixed: new caveat: the sum is bounded
-/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with caveats := #["the input is finite", "the sum is bounded"] }]

/-- info: regression roundoff baseline: exp [value] configured mixed: reference changed -/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with reference := 8, budgets := #[budget "r" 11 #[8, 8, 8, 8]] }]

/-- info: regression roundoff baseline: exp [value] configured absolute: analysis gone -/
#guard_msgs in
run_cmd report #[mixed, absolute] #[mixed]

/--
info: stale roundoff baseline: exp [value] configured absolute: not in the baseline; update the
baseline with TORCHLEAN_ROUNDOFF_BASELINE=update
-/
#guard_msgs in
run_cmd report #[mixed] #[mixed, absolute]

/--
info: stale roundoff baseline: exp [value] configured mixed: improved (status conditional →
proved); update the baseline with TORCHLEAN_ROUNDOFF_BASELINE=update
-/
#guard_msgs in
run_cmd report #[mixed] #[{ mixed with status := .proved, premises := #[], caveats := #[] }]

#guard_msgs in
run_cmd report #[mixed, absolute] #[mixed, absolute]

-- A row read back from its line in the file has the same line and no difference with it.
#guard_msgs in
run_cmd do
  for row in #[mixed, absolute, { absolute with
      index := 1, reason := "no rule", usesIEEE := true, classes := #[("range", 2)] }] do
    let read ← ofExcept (Json.parse row.line >>= Row.ofJson?)
    unless read.line == row.line do throwError "{row.line} reads back as {read.line}"
    report #[read] #[row]

/-- A constant for the test of `meaning`. -/
def helper : Nat := 1

/-- A constant that uses `helper`. -/
def wrapper : Nat := helper + 1

-- The meaning of a constant follows the meanings of the constants it uses. The constants of the
-- current file are definitions of a budget, and those of core are not.
#guard_msgs in
run_cmd liftTermElabM do
  let fresh : IO.Ref (Std.HashMap Name UInt64) ← IO.mkRef {}
  let seeded : IO.Ref (Std.HashMap Name UInt64) ← IO.mkRef (Std.HashMap.emptyWithCapacity.insert
    ``helper 7)
  if (← meaning fresh ``wrapper) == (← meaning seeded ``wrapper) then
    throwError "the meaning of wrapper does not follow that of helper"
  unless (← mixUses fresh 0 #[mkConst ``Nat.succ]) == 0 do
    throwError "a constant of core counts as a definition"
  if (← mixUses fresh 0 #[mkConst ``wrapper]) == 0 then
    throwError "a constant of the current file does not count as a definition"

-- The unit roundoff at binary32 and the underflow error at binary64 evaluate to `2⁻²⁴` and
-- `2⁻¹⁰⁷⁵`.
#guard_msgs in
run_cmd liftTermElabM do
  for (constant, format, value) in
      [(``Model.unitRoundoffAt, ``FloatFormat.binary32, mkRat 1 (2 ^ 24)),
        (``Model.underflowErrorAt, ``FloatFormat.binary64, mkRat 1 (2 ^ 1075))] do
    let term := mkApp (mkConst constant) (mkConst format)
    unless (← value? term (some format)) == some value do
      throwError "{term} does not evaluate to {value}"

-- Without a file, and with an unknown mode, the check gives one line and writes no file.
/--
info: roundoff baseline: no-such-baseline.jsonl is missing; write it with
TORCHLEAN_ROUNDOFF_BASELINE=update
roundoff baseline: TORCHLEAN_ROUNDOFF_BASELINE is bogus; it can be update or accept
-/
#guard_msgs in
run_cmd do
  let path : System.FilePath := "no-such-baseline.jsonl"
  let lines := (← checkRows none #[mixed] path) ++ (← checkRows (some "bogus") #[mixed] path)
  if (← path.pathExists) then throwError "{path.toString} was written"
  logInfo ("\n".intercalate lines.toList)

end TorchLeanTest.RoundoffBaseline
