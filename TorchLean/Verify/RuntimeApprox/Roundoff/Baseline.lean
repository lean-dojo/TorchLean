/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.Roundoff.Coverage
public meta import Mathlib.Tactic.NormNum.Basic
public meta import Mathlib.Tactic.NormNum.Inv
public meta import Mathlib.Tactic.NormNum.Pow

/-!
# The baseline of the roundoff coverage

`scripts/checks/roundoff_baseline.jsonl` records the outcome of each analysis registered with
`#roundoff_coverage`, one JSON object per line, sorted by key. The key is the family, the
operator, the route, the scalar and the mode, and an index among the analyses with the same key.
The library check `scripts/checks/Library.lean` computes the same rows from the environment with
`rows` and compares them with the file with `diff`. For one key, these are regressions:

- a status lower in the order proved, conditional, unsupported, failed;
- a changed reference: another `String.hash` of the pretty-printed reference or another number of
  its distinct subterms. The budgets are then not compared;
- a budget that is larger at one of binary16, bfloat16, binary32 and binary64. The budgets are
  `r` and `a` of a mixed bound and `ε` of an absolute one, evaluated by `norm_num` after the unit
  roundoff and the underflow error of the format are unfolded, and compared whenever both have
  values at the same formats, whatever their terms. A budget without a value is compared by a hash
  of its term and a hash of the definitions it uses (`mixUses`: the types, values and constructors
  of the constants of TorchLean and FloatLib it uses, and of those these use in turn), and a
  change of either is a regression;
- at an equal status, a new premise, compared by a hash of its statement, or a new caveat;
- a key that is gone.

A new key, a better status, a smaller budget, a removed premise or caveat, and a budget whose term
or definitions changed with equal values make the file stale. The check fails on each difference,
with one line each, and on each line of the file that is not a row. With
`TORCHLEAN_ROUNDOFF_BASELINE=update` it writes the rows to the file instead, unless there is a
regression or such a line, and with `TORCHLEAN_ROUNDOFF_BASELINE=accept` it writes them in any
case and prints each regression. The rows are written in the same order with the same text each
time. The evidence, the class counts
of the premises, `usesIEEE` and the reason are recorded for information and not compared.

The budget comparison is sound only at the four formats. If `r' ≤ r` and `a' ≤ a` at a format,
then `|x - x₀| ≤ r' * |x₀| + a'` implies `|x - x₀| ≤ r * |x₀| + a` there, so the new bound implies
the old one at the four formats and at no other. Budgets without values count as equal when their
terms and definitions have the same hashes; a change in Mathlib or core behind the same term is
not seen.
-/

public meta section

namespace TorchLean.Roundoff.Baseline

open Lean Meta Elab Command
open FloatLib.Floats.Formats.BinaryInterchange
open FloatLib.Floats.Formats.BinaryInterchange.Roundoff (conjuncts)

/-- The formats at which budgets are evaluated, with their names in the file. -/
def formats : Array (String × Name) :=
  #[("binary16", ``FloatFormat.binary16), ("bfloat16", ``FloatFormat.bfloat16),
    ("binary32", ``FloatFormat.binary32), ("binary64", ``FloatFormat.binary64)]

/-- The name in the file of the value of a budget whose theorem has no format. -/
def fixed : String := "value"

/-- The name of a status in the file. -/
def statusName : CoverageStatus → String
  | .proved => "proved"
  | .conditional => "conditional"
  | .unsupported => "unsupported"
  | .failed => "failed"

/-- The rank of a status: 3 for proved, 2 for conditional, 1 for unsupported and 0 for failed. -/
def rank : CoverageStatus → Nat
  | .proved => 3
  | .conditional => 2
  | .unsupported => 1
  | .failed => 0

/-- The status that `statusName` names `name`. -/
def status? (name : String) : Option CoverageStatus :=
  [CoverageStatus.proved, .conditional, .unsupported, .failed].find? (statusName · == name)

/-- `text` with each run of spaces and line breaks replaced by one space, and cut after `limit`
characters, an ellipsis marking the cut. -/
def shorten (text : String) (limit : Nat := 160) : String :=
  let characters := text.toList.foldl (init := #[]) fun out character =>
    let character := if character == '\n' then ' ' else character
    if character == ' ' && out.back? == some ' ' then out else out.push character
  if characters.size ≤ limit then String.ofList characters.toList
  else String.ofList (characters.toList.take (limit - 1)) ++ "…"

/-- A budget of a generated bound. -/
structure Budget where
  /-- `r` or `a` of a mixed bound, `ε` of an absolute one. -/
  name : String
  /-- The `Expr.hash` of the term abstracted over the binders of the theorem. -/
  hash : UInt64
  /-- The hash of the definitions of TorchLean and FloatLib that the term uses, as `mixUses`
  computes it, `0` when it uses none. -/
  definitions : UInt64 := 0
  /-- The term, pretty-printed and shortened. -/
  term : String := ""
  /-- The values at `formats`, or at `fixed` when the theorem has no format, each `none` when the
  term does not evaluate to a rational number. -/
  values : Array (String × Option Rat) := #[]

/-- One analysis as the file records it. -/
structure Row where
  /-- The family the operator is registered in. -/
  family : String
  /-- The registered declaration. -/
  operator : String
  /-- The route, empty when the declaration is not analyzed. -/
  route : String
  /-- The scalar the route was analyzed for. -/
  backend : String
  /-- `absolute` or `mixed`, empty without a route. -/
  mode : String
  /-- The number of earlier analyses with the same key, in registration order. -/
  index : Nat := 0
  /-- The outcome. -/
  status : CoverageStatus
  /-- The generated theorem, for information. -/
  evidence : String := ""
  /-- The `String.hash` of the pretty-printed reference, `0` without a reference. -/
  reference : UInt64 := 0
  /-- The number of distinct subterms of the reference, `0` without a reference. -/
  referenceSize : Nat := 0
  /-- The budgets of the generated bound. -/
  budgets : Array Budget := #[]
  /-- The premises: the `Expr.hash` of each conjunct of the conditions, abstracted over the
  binders, and the `String.hash` of `the contract of f` for each function `f` whose contract the
  theorem assumes, sorted, each with its statement shortened, or `""` in a row read from the
  file. -/
  premises : Array (UInt64 × String) := #[]
  /-- The number of premises of each class, for information. -/
  classes : Array (String × Nat) := #[]
  /-- Why the theorem is weaker than an a priori bound. -/
  caveats : Array String := #[]
  /-- Whether the proof uses the hypothesis that the format is IEEE, for information. -/
  usesIEEE : Bool := false
  /-- The analysis' message for an unsupported or failed route, shortened, for information. -/
  reason : String := ""

/-- The key of a row: family, operator, route, scalar, mode and index. -/
def Row.key (row : Row) : String × String × String × String × String × Nat :=
  (row.family, row.operator, row.route, row.backend, row.mode, row.index)

/-- The order of the rows in the file: by family, operator, route, scalar, mode, then index. -/
def Row.order (row other : Row) : Ordering :=
  (compare row.family other.family).then <| (compare row.operator other.operator).then <|
    (compare row.route other.route).then <| (compare row.backend other.backend).then <|
    (compare row.mode other.mode).then (compare row.index other.index)

/-- The analysis in messages: operator, route in brackets, scalar and mode, then `#index` when
the index is not `0`. -/
def Row.label (row : Row) : String :=
  let route := if row.route.isEmpty then "" else s!"[{row.route}]"
  " ".intercalate ([row.operator, route, row.backend, row.mode].filter (!·.isEmpty)) ++
    if row.index == 0 then "" else s!" #{row.index}"

/-! ## The file -/

/-- The JSON text of an object with the fields `fields`, given as JSON texts, in this order. -/
def objectText (fields : List (String × String)) : String :=
  "{" ++ ",".intercalate (fields.map fun (key, value) => (Json.str key).compress ++ ":" ++ value)
    ++ "}"

/-- The JSON text of an array with the elements `elements`, given as JSON texts. -/
def arrayText (elements : List String) : String :=
  "[" ++ ",".intercalate elements ++ "]"

/-- The JSON text of a string. -/
def stringText (text : String) : String :=
  (Json.str text).compress

/-- The JSON text of a value: `p/q` or `p` as a string, `null` without a value. -/
def valueText : Option Rat → String
  | some value => stringText (toString value)
  | none => "null"

/-- The rational number written `p/q` or `p`. -/
def rat? (text : String) : Option Rat :=
  match text.splitOn "/" with
  | [numerator] => numerator.toInt?.map (mkRat · 1)
  | [numerator, denominator] => do
    let denominator ← denominator.toNat?
    if denominator == 0 then none else (numerator.toInt?).map (mkRat · denominator)
  | _ => none

/-- The JSON text of a budget: its name, hash, definitions, term and values, in this order. -/
def Budget.text (budget : Budget) : String :=
  objectText [("name", stringText budget.name), ("hash", (toJson budget.hash).compress),
    ("definitions", (toJson budget.definitions).compress), ("term", stringText budget.term),
    ("values", objectText (budget.values.toList.map fun (format, value) =>
      (format, valueText value)))]

/-- The line of the file that records `row`: its fields in a fixed order, those with their default
value left out but for the key and the status. -/
def Row.line (row : Row) : String :=
  let optional (present : Bool) (pair : String × String) := if present then [pair] else []
  objectText <|
    [("family", stringText row.family), ("operator", stringText row.operator),
      ("route", stringText row.route), ("backend", stringText row.backend),
      ("mode", stringText row.mode)] ++
    optional (row.index != 0) ("index", toString row.index) ++
    [("status", stringText (statusName row.status))] ++
    optional (!row.evidence.isEmpty) ("evidence", stringText row.evidence) ++
    optional (row.referenceSize != 0) ("reference", (toJson row.reference).compress) ++
    optional (row.referenceSize != 0) ("referenceSize", toString row.referenceSize) ++
    optional (!row.budgets.isEmpty) ("budgets", arrayText (row.budgets.toList.map (·.text))) ++
    optional (!row.premises.isEmpty)
      ("premises", (toJson (row.premises.map (·.1))).compress) ++
    optional (!row.classes.isEmpty) ("classes", (toJson row.classes).compress) ++
    optional (!row.caveats.isEmpty) ("caveats", (toJson row.caveats).compress) ++
    optional row.usesIEEE ("usesIEEE", "true") ++
    optional (!row.reason.isEmpty) ("reason", stringText row.reason)

/-- The field `key` of the object `json`, or `default` when it has none. -/
def field {α : Type} [FromJson α] (json : Json) (key : String) (default : α) : Except String α :=
  match json.getObjVal? key with
  | .ok value => fromJson? value
  | .error _ => pure default

/-- The budget that `Budget.text` writes as `json`. Its values are read at `formats` and at
`fixed`, in this order. -/
def Budget.ofJson? (json : Json) : Except String Budget := do
  let values ← json.getObjVal? "values"
  let mut read := #[]
  for format in (formats.map (·.1)).push fixed do
    match values.getObjVal? format with
    | .error _ => pure ()
    | .ok .null => read := read.push (format, none)
    | .ok (.str text) =>
      let some value := rat? text | throw s!"the value {text} is not a rational number"
      read := read.push (format, some value)
    | .ok other => throw s!"the value {other} is not a rational number"
  return { name := ← json.getObjValAs? String "name", hash := ← json.getObjValAs? UInt64 "hash"
           definitions := ← field json "definitions" 0, term := ← field json "term" "",
           values := read }

/-- The row that `Row.line` writes as `json`. Its premises have no statements. -/
def Row.ofJson? (json : Json) : Except String Row := do
  let name ← json.getObjValAs? String "status"
  let some status := status? name | throw s!"unknown status {name}"
  let budgets ← (← field json "budgets" (#[] : Array Json)).mapM Budget.ofJson?
  let premises ← field json "premises" (#[] : Array UInt64)
  return {
    family := ← json.getObjValAs? String "family"
    operator := ← json.getObjValAs? String "operator"
    route := ← json.getObjValAs? String "route"
    backend := ← json.getObjValAs? String "backend"
    mode := ← json.getObjValAs? String "mode"
    index := ← field json "index" 0
    status
    evidence := ← field json "evidence" ""
    reference := ← field json "reference" 0
    referenceSize := ← field json "referenceSize" 0
    budgets
    premises := premises.map (·, "")
    classes := ← field json "classes" #[]
    caveats := ← field json "caveats" #[]
    usesIEEE := ← field json "usesIEEE" false
    reason := ← field json "reason" "" }

/-! ## The rows of the environment -/

/-- Run `k` on the binders of `type` up to its conclusion, through the `let`s between them, and on
the conclusion. -/
partial def binders {α : Type} (type : Expr) (k : Array Expr → Expr → MetaM α)
    (outer : Array Expr := #[]) : MetaM α :=
  forallTelescope type fun inner body => do
    let body := Einops.instantiateOuterLets body
    if body.isForall then binders body k (outer ++ inner) else k (outer ++ inner) body

/-- The simp set that unfolds the unit roundoff and the underflow error into powers of two, and,
when `format?` is a format, that format and its exponents. -/
def unfolding (format? : Option Name) : MetaM Simp.Context := do
  let mut theorems : SimpTheorems := {}
  for name in [``Model.unitRoundoffAt_eq, ``Model.underflowErrorAt_eq] do
    theorems ← theorems.addConst name
  for name in [``FloatFormat.minSubnormalExponent, ``FloatFormat.minNormalExponent] ++
      format?.toList do
    theorems ← theorems.addDeclToUnfold name
  Simp.mkContext {} #[theorems] {}

/-- The value of the real term `term` that `norm_num` computes after `unfolding format?`, or `none`
when the term has free variables or `norm_num` gives no rational number. Exponents up to 4096 are
evaluated, enough for the underflow error of binary64. -/
def value? (term : Expr) (format? : Option Name) : MetaM (Option Rat) := do
  if term.hasFVar || term.hasMVar then return none
  try
    withOptions (exponentiation.threshold.set · 4096) do
      let unfolded ← try
          pure (← Mathlib.Meta.NormNum.deriveSimp (← unfolding format?) #[] true term).expr
        catch _ => pure term
      let result ← Mathlib.Meta.NormNum.derive (u := Level.zero) (α := mkConst ``Real) unfolded
      return result.toRat
  catch _ => return none

/-- Whether the constants of the module `module` count as definitions of a budget: those of
TorchLean and FloatLib. -/
def tracked (module : Name) : Bool :=
  module.getRoot.toString.startsWith "TorchLean" || module.getRoot == `FloatLib

/-- The constants that `terms` use, each once, in the order of their first use. -/
def uses (terms : Array Expr) : Array Name :=
  terms.foldl (init := #[]) fun names term =>
    term.getUsedConstants.foldl (init := names) fun names name =>
      if names.contains name then names else names.push name

mutual

/-- The hash of what the constant `name` means: its name, its type, its value when it is a
definition and the types of its constructors when it is an inductive type, mixed by `mixUses` with
the meaning of the constants these use. The proofs of theorems are left out. `cache` holds the
meanings computed so far. -/
partial def meaning (cache : IO.Ref (Std.HashMap Name UInt64)) (name : Name) : MetaM UInt64 := do
  if let some hash := (← cache.get)[name]? then return hash
  -- a constant whose meaning uses itself, which only unsafe code does, sees `0` for itself
  cache.modify (·.insert name 0)
  let some info := (← getEnv).find? name | return 0
  let (parts, block) : Array Expr × Array Name ← match info with
    | .defnInfo definition => pure (#[info.type, definition.value], #[])
    | .opaqueInfo definition => pure (#[info.type, definition.value], #[])
    | .inductInfo type => do
      let constructors ← type.ctors.toArray.mapM fun c => return (← getConstInfo c).type
      pure (#[info.type] ++ constructors, type.all.toArray)
    | _ => pure (#[info.type], #[])
  let hash ← mixUses cache (parts.foldl (mixHash · ·.hash) (Hashable.hash name)) parts
    (block.push name)
  cache.modify (·.insert name hash)
  return hash

/-- `hash` mixed with the `meaning` of each constant that `terms` use, in the order of their first
use, but those of `skip` and those that come neither from a module of TorchLean or FloatLib nor
from the current file. -/
partial def mixUses (cache : IO.Ref (Std.HashMap Name UInt64)) (hash : UInt64)
    (terms : Array Expr) (skip : Array Name := #[]) : MetaM UInt64 := do
  let env ← getEnv
  let mut hash := hash
  for used in uses terms do
    if skip.contains used then continue
    unless (env.getModuleIdxFor? used).all (tracked env.allImportedModuleNames[·]!) do continue
    hash := mixHash hash (← meaning cache used)
  return hash

end

/-- `term` pretty-printed, visiting at most 200 of its subterms, and shortened. -/
def shortText (term : Expr) : MetaM String := do
  return shorten (← withOptions (pp.maxSteps.set · 200) (ppExpr term)).pretty

/-- The row of `entry`, the analysis with `index` earlier analyses of the same key. The budgets
and premises come from the conclusion and conditions of its generated theorem. `cache` holds the
meanings of the constants computed so far. -/
def row (cache : IO.Ref (Std.HashMap Name UInt64)) (entry : CoverageEntry) (index : Nat) :
    MetaM Row := do
  let base : Row := {
    family := entry.family.toString, operator := entry.operator.toString, route := entry.route
    backend := entry.backend, mode := entry.mode, index, status := entry.status
    evidence := if entry.evidence.isAnonymous then "" else entry.evidence.toString
    reference := if entry.referenceSize == 0 then 0 else entry.reference.hash
    referenceSize := entry.referenceSize, classes := entry.premises, caveats := entry.caveats
    usesIEEE := entry.usesIEEE, reason := shorten entry.reason }
  if entry.evidence.isAnonymous then return base
  let some info := (← getEnv).find? entry.evidence | return base
  binders info.type fun arguments conclusion => do
    let some (_, terms) := boundParts conclusion | return base
    let names := if terms.size == 2 then #["r", "a"] else #["ε"]
    let format? ← arguments.findM? fun argument =>
      return (← inferType argument).isConstOf ``FloatFormat
    let budgets ← (names.zip terms).mapM fun (name, term) => do
      let values ← match format? with
        | some format => formats.mapM fun (label, constant) => do
            return (label, ← value? (term.replaceFVar format (mkConst constant)) (some constant))
        | none => pure #[(fixed, ← value? term none)]
      let definitions ← mixUses cache 0 #[term]
      let text ← shortText term
      return ({ name, hash := (term.abstract arguments).hash, definitions, term := text, values }
        : Budget)
    -- The conditions are the last binder when the theorem has premises besides contracts.
    let facts ← if entry.premises.any (·.1 != "function contract") && !arguments.isEmpty then
        pure (conjuncts (← inferType arguments.back!))
      else pure #[]
    let conditions ← facts.mapM fun fact => do
      return ((fact.abstract arguments).hash, ← shortText fact)
    let contracts := entry.functions.map fun function =>
      let text := s!"the contract of {function}"
      (text.hash, text)
    let premises := (conditions ++ contracts).qsort (·.1 < ·.1)
    return { base with budgets, premises }

/-- The rows of `entries`, sorted. Each entry is analyzed in its own `liftTermElabM`, so its own
heartbeat limit bounds it. -/
def rows (entries : Array CoverageEntry) : CommandElabM (Array Row) := do
  let cache ← IO.mkRef {}
  let mut seen : Std.HashMap (Name × Name × String × String × String) Nat := {}
  let mut result := #[]
  for entry in entries do
    let key := (entry.family, entry.operator, entry.route, entry.backend, entry.mode)
    let index := seen.getD key 0
    seen := seen.insert key (index + 1)
    result := result.push (← liftTermElabM (row cache entry index))
  return result.qsort (·.order · == .lt)

/-! ## The comparison -/

/-- A difference between the file and the environment. -/
structure Difference where
  /-- The message. -/
  line : String
  /-- Whether the outcome is worse than the file's: a regression, rather than a stale file. -/
  regression : Bool

/-- The regression `what` of the analysis `row`. -/
def worse (row : Row) (what : String) : Difference :=
  { line := s!"roundoff baseline: {row.label}: {what}", regression := true }

/-- The difference `what` of the analysis `row` that makes the file stale. -/
def stale (row : Row) (what : String) : Difference :=
  { line := s!"roundoff baseline: {row.label}: {what}; update the baseline with \
      TORCHLEAN_ROUNDOFF_BASELINE=update", regression := false }

/-- Where a value is in a message: ` at binary32` at a format, nothing at `fixed`. -/
def atFormat (format : String) : String :=
  if format == fixed then "" else s!" at {format}"

/-- The differences between the budget `old` of the file and the budget `new` of the analysis
`row` with the same name. They are compared by their values when both have values at the same
formats, and by the hashes of their terms and definitions otherwise. -/
def compareBudget (row : Row) (old new : Budget) : Array Difference := Id.run do
  let same := old.hash == new.hash && old.definitions == new.definitions
  let valued := !old.values.isEmpty && old.values.map (·.1) == new.values.map (·.1) &&
    (old.values ++ new.values).all (·.2.isSome)
  unless valued do
    if same && old.values == new.values then return #[]
    if old.hash == new.hash then
      return #[worse row s!"budget {new.name}: a definition that its term {new.term} uses \
        changed, and it is not comparable"]
    return #[worse row s!"budget {new.name} changed and is not comparable: {old.term} → \
      {new.term}"]
  let mut weaker := #[]
  let mut smaller := #[]
  for ((format, before), (_, after)) in old.values.zip new.values do
    let (some before, some after) := (before, after) | continue
    if before < after then
      weaker := weaker.push
        (worse row s!"budget {new.name} weaker{atFormat format}: {before} → {after}")
    else if after < before then
      smaller := smaller.push format
  unless weaker.isEmpty do return weaker
  if smaller.isEmpty then
    if same then return #[]
    return #[stale row s!"budget {new.name} changed with the same values"]
  let formats := if smaller == #[fixed] then "" else s!" at {", ".intercalate smaller.toList}"
  return #[stale row s!"improved (budget {new.name} smaller{formats})"]

/-- The differences between the row `old` of the file and the row `new` of the environment with
the same key. Premises and caveats are compared at an equal status only, and budgets only when
the reference did not change. -/
def compareRow (old new : Row) : Array Difference := Id.run do
  let mut out := #[]
  let same := old.status == new.status
  if rank new.status < rank old.status then
    out := out.push (worse new s!"status {statusName old.status} → {statusName new.status}")
  else if rank old.status < rank new.status then
    out := out.push (stale new s!"improved (status {statusName old.status} → \
      {statusName new.status})")
  let referenced (row : Row) := row.referenceSize != 0
  let changed := (referenced old && referenced new || same && referenced old != referenced new) &&
    (old.reference, old.referenceSize) != (new.reference, new.referenceSize)
  if changed then
    out := out.push (worse new "reference changed")
  else
    for before in old.budgets do
      match new.budgets.find? (·.name == before.name) with
      | some after => out := out ++ compareBudget new before after
      | none => if same then out := out.push (worse new s!"budget {before.name} gone")
    for after in new.budgets do
      if same && !old.budgets.any (·.name == after.name) then
        out := out.push (stale new s!"budget {after.name} is new")
  if same then
    for (hash, statement) in new.premises do
      unless old.premises.any (·.1 == hash) do
        out := out.push (worse new s!"new premise: {statement}")
    let removed := old.premises.filter fun (hash, _) => !new.premises.any (·.1 == hash)
    unless removed.isEmpty do
      out := out.push (stale new s!"improved (premises removed: {removed.size})")
    for caveat in new.caveats do
      unless old.caveats.contains caveat do
        out := out.push (worse new s!"new caveat: {caveat}")
    for caveat in old.caveats do
      unless new.caveats.contains caveat do
        out := out.push (stale new s!"improved (caveat removed: {caveat})")
  return out

/-- The differences between the rows `old` of the file and the rows `new` of the environment: those
of each analysis of `new`, in its order, then the analyses of `old` that are gone. -/
def diff (old new : Array Row) : Array Difference := Id.run do
  let byKey : Std.HashMap _ Row := old.foldl (fun rows row => rows.insert row.key row) {}
  let keys : Std.HashSet _ := new.foldl (fun keys row => keys.insert row.key) {}
  let mut out := #[]
  for row in new do
    match byKey.get? row.key with
    | some before => out := out ++ compareRow before row
    | none => out := out.push (stale row "not in the baseline")
  for row in old do
    unless keys.contains row.key do
      out := out.push (worse row "analysis gone")
  return out

/-! ## The check -/

/-- The rows of the file at `path`, `none` when there is no file, and a message for each line
that is not a row. -/
def read (path : System.FilePath) : IO (Option (Array Row) × Array String) := do
  unless ← path.pathExists do return (none, #[])
  let mut rows := #[]
  let mut unreadable := #[]
  let mut number := 0
  for line in ← IO.FS.lines path do
    number := number + 1
    if line.isEmpty then continue
    match Json.parse line >>= Row.ofJson? with
    | .ok row => rows := rows.push row
    | .error message => unreadable := unreadable.push s!"roundoff baseline: {path}:{number}: \
        {message}"
  return (some rows, unreadable)

/-- Write `rows` to `path`, one line each. -/
def write (path : System.FilePath) (rows : Array Row) : IO Unit :=
  IO.FS.writeFile path (String.join (rows.toList.map (·.line ++ "\n")))

/-- Compare the rows `new` with the file at `path` and return one line for each difference and
each unreadable line, or one line when there is no file. `mode` is the value of
`TORCHLEAN_ROUNDOFF_BASELINE`. With `update`, it writes `new` to `path` instead, unless there is a
regression or an unreadable line, whose lines it then returns. With `accept`, it writes `new` in
any case, logs each regression and returns nothing. -/
def checkRows (mode : Option String) (new : Array Row) (path : System.FilePath) :
    CommandElabM (Array String) := do
  let (old?, unreadable) ← read path
  let differences := diff (old?.getD #[]) new
  let regressions := differences.filter (·.regression)
  match mode with
  | none =>
    if old?.isNone then
      return #[s!"roundoff baseline: {path} is missing; write it with \
        TORCHLEAN_ROUNDOFF_BASELINE=update"]
    return unreadable ++ differences.map (·.line)
  | some "update" =>
    unless regressions.isEmpty && unreadable.isEmpty do
      return unreadable ++ regressions.map (·.line) ++
        #[s!"roundoff baseline: {path} is not updated while there are regressions or unreadable \
          lines; accept the regressions with TORCHLEAN_ROUNDOFF_BASELINE=accept"]
    write path new
    logInfo m!"roundoff baseline: wrote the {new.size} analyses to {path}, with \
      {differences.size} differences"
    return #[]
  | some "accept" =>
    write path new
    for regression in regressions do
      logInfo m!"accepted {regression.line}"
    logInfo m!"roundoff baseline: wrote the {new.size} analyses to {path}, with \
      {differences.size} differences, {regressions.size} of them regressions"
    return #[]
  | some other =>
    return #[s!"roundoff baseline: TORCHLEAN_ROUNDOFF_BASELINE is {other}; it can be update or \
      accept"]

/-- `checkRows` on the rows of `entries`, in the mode that `TORCHLEAN_ROUNDOFF_BASELINE` gives. -/
def check (entries : Array CoverageEntry) (path : System.FilePath) :
    CommandElabM (Array String) := do
  checkRows (← IO.getEnv "TORCHLEAN_ROUNDOFF_BASELINE") (← rows entries) path

end TorchLean.Roundoff.Baseline
