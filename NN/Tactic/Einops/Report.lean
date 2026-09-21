/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import NN.Tactic.Einops.Report.Analysis
public meta import ProofWidgets.Component.HtmlDisplay
public import NN.Tactic.Einops.Report.Analysis.Render
public import ProofWidgets.Component.HtmlDisplay

/-!
# InfoView reports for verified tensor transformations

`einops?` inspects the current target without changing it. Recognized
operations appear in a compact InfoView panel. Checked types and the principal
shape transformation remain visible. Independent disclosure sections expose
the checked axes, proof obligations, lowering stages, generated execution,
static work estimates, and correctness theorems on demand.

The tactic never evaluates or benchmarks user terms during elaboration.
-/

public meta section

namespace TorchLean.Tensor.Internal

open Lean Elab Tactic Meta
open ProofWidgets

/--
Split a recognized lowering from arguments subsequently applied to its tensor
result.

Tensors coerce to functions, so a term such as `(repeat tensor "...") index`
has one more application argument than `repeatTensor` itself. Checked
certificates are skipped, while runtime tensor inputs are followed to find
nested lowerings such as `unpack (pack ...)`.
-/
private def reportedApplication? (head : Expr) (headName? : Option Name)
    (arguments : Array Expr) :
    Option (Expr × Array Expr × Array Expr) :=
  let split (arity : Nat) (tensorInputs : Array Expr) :=
    some (mkAppN head (arguments.extract 0 arity), tensorInputs,
      arguments.extract arity arguments.size)
  if headName? == some ``Lowering.transformTensorFused &&
      arguments.size >= 9 then
    split 9 #[arguments[8]!]
  else if (headName? == some ``Lowering.rearrangeTensor ||
      headName? == some ``Lowering.repeatTensor) &&
      arguments.size >= 5 then
    split 5 #[arguments[4]!]
  else if headName? == some ``Lowering.reduceFoldTensor &&
      arguments.size >= 11 then
    split 11 #[arguments[10]!]
  else if headName? == some ``Lowering.reduceNonemptyFoldTensor &&
      arguments.size >= 7 then
    split 7 #[arguments[6]!]
  else if headName? == some ``Lowering.reduceTensor &&
      arguments.size >= 8 then
    split 8 #[arguments[7]!]
  else if headName? == some ``Lowering.reduceNonemptyTensor &&
      arguments.size >= 9 then
    split 9 #[arguments[8]!]
  else if headName? == some ``Lowering.einsumTensorKernel &&
      arguments.size >= 12 then
    split 12 #[arguments[7]!]
  else if headName? == some ``Lowering.einsumTensor &&
      arguments.size >= 8 then
    split 8 #[arguments[7]!]
  else if (headName? == some ``Lowering.packTensor ||
      headName? == some ``Lowering.unpackTensor) &&
      arguments.size >= 4 then
    split 4 #[arguments[3]!]
  else if headName? == some ``Check.checkParseShape &&
      arguments.size >= 2 then
    split 2 #[]
  else
    none

/--
Select explicit arguments from a reflected application so report discovery
does not recurse into typeclass dictionaries or implicit certificates.
-/
private def explicitApplicationArguments (function : Expr)
    (arguments : Array Expr) : MetaM (Array Expr) := do
  let mut functionType ← inferType function
  let mut explicitArguments : Array Expr := #[]
  for argument in arguments do
    functionType ← withTransparency .reducible <| whnf functionType
    match functionType with
    | .forallE _ _ body binderInfo =>
        if binderInfo.isExplicit then
          explicitArguments := explicitArguments.push argument
        functionType := body.instantiate1 argument
    | _ =>
        -- A malformed partial application should not occur in an elaborated
        -- target. Inspecting the argument is the conservative fallback.
        explicitArguments := explicitArguments.push argument
  return explicitArguments

/--
Traverse a target for distinct recognized lowerings and visible constants,
following runtime tensor inputs while skipping certificate internals.
-/
private partial def collectTargetReportData (expression : Expr)
    (found : Array Expr × Array Name := (#[], #[])) :
    MetaM (Array Expr × Array Name) := do
  let expression := expression.consumeMData
  let head := expression.getAppFn
  let headName? :=
    match head with
    | .const name _ => some name
    | _ => none
  let arguments := expression.getAppArgs
  let found :=
    match headName? with
    | some name =>
        if found.2.contains name then found
        else (found.1, found.2.push name)
    | none => found
  if headName? == some ``Elab.Impl.nativeTensorKernel &&
      arguments.size >= 8 then
    let mut found ← collectTargetReportData arguments[3]! found
    for argument in arguments.extract 8 arguments.size do
      found ← collectTargetReportData argument found
    return found
  match reportedApplication? head headName? arguments with
  | some (application, tensorInputs, resultArguments) =>
      let mut found :=
        if found.1.any fun existing => existing == application then found
        else (found.1.push application, found.2)
      for tensorInput in tensorInputs do
        found ← collectTargetReportData tensorInput found
      for argument in resultArguments do
        found ← collectTargetReportData argument found
      return found
  | none =>
      match expression with
      | .forallE name domain body binderInfo =>
          withLocalDecl name binderInfo domain fun localVariable =>
            collectTargetReportData (body.instantiate1 localVariable) found
      | .lam name domain body binderInfo =>
          withLocalDecl name binderInfo domain fun localVariable =>
            collectTargetReportData (body.instantiate1 localVariable) found
      | .letE name _ value body _ =>
          let found ←
            if name == `parseShapeStructuralCheck then
              collectTargetReportData value found
            else
              pure found
          collectTargetReportData (body.instantiate1 value) found
      | .app _ _ =>
          let mut found ← collectTargetReportData head found
          for argument in ←
              explicitApplicationArguments head arguments do
            found ← collectTargetReportData argument found
          return found
      | .mdata _ body =>
          collectTargetReportData body found
      | .proj _ _ body =>
          collectTargetReportData body found
      | _ => return found

/-- Read the operation name from the first line of a rendered report. -/
private def reportOperation (report : String) : String :=
  (report.splitOn "\n").headD "einops operation"

/-- Select the tensor-level type information useful in the compact view. -/
private def compactTypeLine (line : String) : Bool :=
  line.startsWith "Input tensor:" ||
    line.startsWith "Output tensor:" ||
    line.startsWith "Operand " ||
    line.startsWith "Component " ||
    line.startsWith "Rep input tensor:" ||
    line.startsWith "Rep output tensor:" ||
    line.startsWith "Input component family:" ||
    line.startsWith "Output component family:" ||
    line.startsWith "Generated output tensor:" ||
    line.startsWith "Kernel result:" ||
    line.startsWith "Structural shape metadata:" ||
    line.startsWith "Public return metadata:"

/-- Extract a bounded tensor signature from the report's type-check section. -/
private def compactTypeSummary (report : String) : List String :=
  let afterHeader :=
    ((report.splitOn "\n").dropWhile fun line =>
      line != "  Type checks:").drop 1
  let entries :=
    (afterHeader.takeWhile fun line => line.startsWith "    ")
      |>.map (fun line => (line.drop 4).toString)
      |>.filter compactTypeLine
  let visibleEntries := entries.take 5
  visibleEntries ++
    if entries.length > visibleEntries.length then ["..."] else []

/-- Keep one operation-specific shape fact in the compact view. -/
private def compactShapeSummary (report : String) : List String :=
  let relevantLines :=
    (report.splitOn "\n").filter fun line =>
      line.startsWith "  Normalized axes:" ||
        line.startsWith "  Contracted axes:" ||
        line.startsWith "  Fixed prefix / suffix:" ||
        line.startsWith "  Expanded positions:" ||
        line.startsWith "  Concrete shapes, axes, and lengths remain symbolic"
  (relevantLines.take 1).map fun line => (line.drop 2).toString

/-- State the verified boundary accurately for executable and metadata reports. -/
private def compactVerificationSummary (operation : String) : String :=
  if operation.startsWith "parse_shape" then
    "Verified: grammar, rank, dimensions, and returned metadata."
  else
    "Verified: checked plan, native lowering, and semantic correctness."

/-- Section headings emitted by the report analyzer, in display order. -/
private def reportSectionTitles : List String :=
  ["Type checks",
   "Discharged obligations",
   "Verified logical stages",
   "Generated execution strategy",
   "Shape-derived work estimate",
   "Correctness",
   "Performance note"]

/--
Extract one analyzer section and remove its report-level indentation.

The type-check section intentionally includes the following unheaded shape
facts, stopping at the first proof-obligation heading.
-/
private def reportSectionLines (report title : String) : List String :=
  let header := s!"  {title}:"
  match (report.splitOn "\n").dropWhile fun line => line != header with
  | [] => []
  | _ :: lines =>
      (lines.takeWhile fun line =>
        !(reportSectionTitles.any fun candidate =>
          line == s!"  {candidate}:")).map fun line =>
        if line.startsWith "    " then
          (line.drop 4).toString
        else if line.startsWith "  " then
          (line.drop 2).toString
        else
          line

/-- Remove analyzer-supplied numbering before rendering an HTML ordered list. -/
private def dropReportNumber (line : String) : String :=
  match line.splitOn ". " with
  | [] | [_] => line
  | _ :: remainder => String.intercalate ". " remainder

/-- Render one independently expandable part of an operation audit. -/
private def reportSectionHtml (report reportTitle displayTitle : String)
    (ordered : Bool := false) (showCount : Bool := false) : Option Html :=
  let lines := reportSectionLines report reportTitle
  if lines.isEmpty then
    none
  else
    let visibleLines :=
      if ordered then lines.map dropReportNumber else lines
    let summary :=
      if showCount then s!"{displayTitle} ({lines.length})" else displayTitle
    let listTag := if ordered then "ol" else "ul"
    some <| Html.element "details" #[
      ("style", json% {
        "border-top":
          "1px solid var(--vscode-panel-border, rgba(128,128,128,0.35))",
        "padding": "0.45em 0"
      })
    ] #[
      Html.element "summary" #[
        ("className", json% "pointer"),
        ("style", json% {
          "font-size": "0.9em",
          "font-weight": "600",
          "line-height": "1.4"
        })
      ] #[Html.text summary],
      Html.element listTag #[
        ("style", json% {
          "font-size": "0.86em",
          "line-height": "1.45",
          "margin": "0.45em 0 0.15em 0",
          "overflow-wrap": "anywhere",
          "padding-left": "1.5em"
        })
      ] <| visibleLines.toArray.map fun line =>
        Html.element "li" #[
          ("style", json% {"margin": "0.22em 0"})
        ] #[Html.text line]
    ]

/-- Render one operation as a compact summary followed by focused audit sections. -/
private def transformationReportEntryHtml (report : String) : Html :=
  let operation := reportOperation report
  let signature :=
    String.intercalate "\n" <|
      compactTypeSummary report ++ compactShapeSummary report
  let sections :=
    ([
      reportSectionHtml report "Type checks" "Checked tensors and axes",
      reportSectionHtml report "Verified logical stages"
        "Transformation stages" (ordered := true) (showCount := true),
      reportSectionHtml report "Discharged obligations"
        "Proof obligations" (showCount := true),
      reportSectionHtml report "Generated execution strategy"
        "Native execution" (ordered := true) (showCount := true),
      reportSectionHtml report "Shape-derived work estimate" "Static cost",
      reportSectionHtml report "Correctness" "Correctness",
      reportSectionHtml report "Performance note" "Benchmark note"
    ] : List (Option Html)).filterMap id |>.toArray
  Html.element "div" #[
    ("style", json% {
      "border-left": "3px solid var(--vscode-testing-iconPassed, #3aa675)",
      "margin": "0.65em 0",
      "padding": "0.2em 0 0.1em 0.75em"
    })
  ] <| #[
    Html.element "div" #[
      ("style", json% {
        "align-items": "baseline",
        "display": "flex",
        "gap": "0.7em",
        "justify-content": "space-between",
        "margin": "0.1em 0 0.3em 0"
      })
    ] #[
      Html.element "strong" #[
        ("className", json% "font-code"),
        ("style", json% {
          "font-size": "0.95em",
          "overflow-wrap": "anywhere"
        })
      ] #[Html.text operation],
      Html.element "span" #[
        ("style", json% {
          "color": "var(--vscode-testing-iconPassed, #3aa675)",
          "font-size": "0.82em",
          "font-weight": "600"
        })
      ] #[Html.text "Verified"]
    ],
    Html.element "pre" #[
      ("className", json% "font-code"),
      ("style", json% {
        "font-size": "0.88em",
        "line-height": "1.4",
        "margin": "0.2em 0",
        "overflow-x": "auto",
        "white-space": "pre-wrap"
      })
    ] #[Html.text signature],
    Html.element "div" #[
      ("style", json% {
        "font-size": "0.86em",
        "line-height": "1.4",
        "margin": "0.3em 0 0.45em 0"
      })
    ] #[Html.text (compactVerificationSummary operation)]
  ] ++ sections

/-- Present all discovered operations in one layered InfoView panel. -/
private def transformationReportHtml (reports : Array String) : Html :=
  Html.element "div" #[("className", json% "mv2")] <| #[
    Html.element "strong" #[] #[Html.text "Verified einops analysis"]
  ] ++ reports.map transformationReportEntryHtml

/--
Report checked types and shapes, verified lowering stages, generated
execution, static work estimates, and compiler-correctness theorems found in
the current target.

The tactic is observational: it does not change the target, add local facts,
evaluate user terms, or run benchmarks. Symbolic certificates receive exact
formulas without invented concrete dimensions.
-/
elab (name := einopsSuggestionTactic) token:"einops?" : tactic =>
    withMainContext do
      let target ← instantiateMVars (← getMainTarget)
      let (applications, visibleConstants) :=
        ← collectTargetReportData target
      let mut reports : Array String := #[]
      for application in applications do
        let report ← Report.renderApplication application
        unless reports.any fun existing => existing == report do
          reports := reports.push report
      unless reports.isEmpty do
        let operations :=
          String.intercalate ", " <| reports.toList.map reportOperation
        logInfoAt token
          s!"Verified einops analysis: {operations}.\n\
            Open the InfoView panel for checked types, proof stages, cost, \
            and correctness."
        let html := transformationReportHtml reports
        Widget.savePanelWidgetInfo
          (hash HtmlDisplay.javascript)
          (return json% {
            html: $(← Server.RpcEncodable.rpcEncode html)
          })
          token

      let containsAny (names : List Name) : Bool :=
        names.any visibleConstants.contains
      let mut families : List String := []
      if containsAny
          [``Lowering.rearrangeTensor,
            ``Semantics.denoteRearrange] then
        families := families.concat "rearrange"
      if containsAny
          [``Lowering.repeatTensor,
            ``Semantics.denoteRepeat] then
        families := families.concat "repeat"
      if containsAny
          [``Lowering.reduceFoldTensor, ``Lowering.reduceNonemptyFoldTensor,
            ``Lowering.reduceTensor,
            ``Lowering.reduceNonemptyTensor,
            ``Semantics.denoteOrderedReduce,
            ``Semantics.denoteOrderedReduceNonempty,
            ``Semantics.denoteReduce,
            ``Semantics.denoteReduceNonempty] then
        families := families.concat "reduction"
      if containsAny
          [``Lowering.einsumTensor, ``Lowering.einsumTensorKernel,
            ``Semantics.einsumProductTensor,
            ``Semantics.denoteEinsum] then
        families := families.concat "einsum"
      if containsAny
          [``Lowering.packTensor, ``Lowering.unpackTensor,
            ``Semantics.denotePack, ``Semantics.denoteUnpack] then
        families := families.concat "pack/unpack"

      let primitiveCandidates : List (String × List Name) := [
        ("reshape", [``Rep.reshape]),
        ("reindex", [``Rep.reindex]),
        ("broadcast", [``Rep.broadcast]),
        ("segment concatenation", [``Rep.concatenateAxes]),
        ("axis splitting", [``Rep.splitAxis]),
        ("pull", [``Rep.pull]),
        ("fiber push", [``Rep.push]),
        ("fiber reduction", [``Rep.reduce, ``Rep.reduceNonempty]),
        ("pointwise map", [``Rep.map]),
        ("pointwise binary operator", [``Rep.zipWith]),
        ("finite tensor pairing", [``Rep.dot])]
      let primitives :=
        primitiveCandidates.filterMap fun (label, names) =>
          if containsAny names then some label else none

      if families.isEmpty then
        if reports.isEmpty then
          let primitiveMessage :=
            if primitives.isEmpty then
              "No supported einops operation or tensor primitive occurs in the target."
            else
              s!"No lowered einops operation occurs in the target.\n\
                Rep primitives: {String.intercalate ", " primitives}."
          logInfoAt token primitiveMessage
      else
        let primitiveMessage :=
          if primitives.isEmpty then ""
          else
            s!"\nTensor primitives visible in the target: \
              {String.intercalate ", " primitives}."
        let suggestion ← `(tactic| einops)
        Lean.Meta.Tactic.TryThis.addSuggestion token suggestion
          (origSpan? := some token)
          (header :=
            s!"Detected einops families: {String.intercalate ", " families}.\
              {primitiveMessage}\n\nTry this:")

end TorchLean.Tensor.Internal
