/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Exception
public meta import TorchLean.NN.Widget.UI
public meta import Lean -- shake: keep
public import TorchLean.NN.Widget.UI
import ProofWidgets.Component.HtmlDisplay

/-!
# Declaration Documentation

`#tl_doc f` prints the type and docstring of the declaration `f` as an info message, and
`#tl_doc_view f` shows the same two pieces as a card in the infoview. Both commands elaborate `f`,
take its head constant and read the docstring and the pretty-printed type from the environment.
Import `TorchLean.NN.Widget.Docs` to use them.
-/

public meta section

open Lean Elab Command Meta Term
open scoped ProofWidgets.Jsx

namespace TorchLean.Widget

section
open UI

/-- Monospaced `<pre>` block that wraps long lines, for signatures and docstrings. -/
def preWrap (s : String) : ProofWidgets.Html :=
  <pre style={json% {
    "margin": "0",
    "white-space": "pre-wrap",
    "word-break": "break-word",
    "font-family":
      "var(--vscode-editor-font-family, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \
      monospace)"
  }}>{.text s}</pre>

/-- One declaration rendered as a card: name, type, and docstring, themed to the editor. -/
def docPanel (declName typeStr docStr : String) : ProofWidgets.Html :=
  <div style={json% {
    "padding": "12px 12px 10px 12px",
    "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, #fff)"
  }}>
    <div style={json% {"display": "flex", "flex-wrap": "wrap", "gap": "8px",
        "align-items": "center"}}>
      {pill "TorchLean docs"} {monospace declName}
    </div>
    <div style={json% {"margin-top": "10px"}}>
      <details «open»={true}>
        <summary>{.text "Type"}</summary>
        <div style={json% {"margin-top": "6px"}}>{preWrap typeStr}</div>
      </details>
    </div>
    <div style={json% {"margin-top": "10px"}}>
      <details «open»={true}>
        <summary>{.text "Docs"}</summary>
        <div style={json% {"margin-top": "6px"}}>{preWrap docStr}</div>
      </details>
    </div>
  </div>


/-- Return the head constant name of an application, if one exists. -/
def termConstName? (e : Expr) : Option Name :=
  match e.getAppFn with
  | .const n _ => some n
  | _ => none

/-- The docstring with trailing whitespace trimmed, or `(no docstring found)` when it is missing. -/
def renderDocString (doc? : Option String) : String :=
  match doc? with
  | some s => s.trimAsciiEnd.toString
  | none => "(no docstring found)"

/-- Type of the declaration `n`, pretty-printed with `ppExpr`. -/
def ppConstType (n : Name) : CommandElabM String := do
  let fmt ← liftTermElabM do
    let info ← getConstInfo n
    ppExpr info.type
  pure fmt.pretty

/-- Elaborate a term and return its head constant, failing at the term when there is none. -/
def resolveConstFromTerm (t : Syntax) : CommandElabM Name := do
  let e ← liftTermElabM do
    Term.elabTerm t none
  match termConstName? e with
  | some n => pure n
  | none =>
      throwErrorAt t
        "expected a constant (e.g. `nn.linear`), but got an expression that does not resolve to a \
        declaration"

end

/-!
## Commands
-/

/-- `#tl_doc f` prints the type and docstring of the head constant of `f` as an info message. -/
syntax (name := tlDocCmd) "#tl_doc " term : command

/-- `#tl_doc_view f` shows the type and docstring of the head constant of `f` as a card. -/
syntax (name := tlDocViewCmd) "#tl_doc_view " term : command

elab_rules : command
  | `(#tl_doc $t) => do
      let declName ← resolveConstFromTerm t
      let env ← getEnv
      let doc? ← liftIO <| Lean.findDocString? env declName
      let ty ← ppConstType declName
      let docStr := renderDocString doc?
      logInfo m!"{declName} : {ty}\n\n{docStr}"

elab_rules : command
  | `(#tl_doc_view $t) => do
      let declName ← resolveConstFromTerm t
      let env ← getEnv
      let doc? ← liftIO <| Lean.findDocString? env declName
      let ty ← ppConstType declName
      let docStr := renderDocString doc?
      let declStr := toString declName

      let declLit := Syntax.mkStrLit declStr
      let tyLit := Syntax.mkStrLit ty
      let docLit := Syntax.mkStrLit docStr

      -- Mark the generated command canonical so the infoview anchors the panel to this command.
      let cmd ← (UI.canonicalCommand <$> `(
        #html (docPanel $declLit $tyLit $docLit)
      ))
      elabCommand cmd

end TorchLean.Widget
