/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Exception
public meta import NN.Widgets.Core.UI
public meta import Lean -- shake: keep
public import NN.Widgets.Core.UI
import ProofWidgets.Component.HtmlDisplay

/-!
# Docs

Small helpers for surfacing docstrings in the InfoView.

This is available through the widget entrypoint (`import NN.Widgets`) or
directly as `import NN.Widgets.Core.Docs`, since it depends on ProofWidgets.

Commands:
- `#tl_doc f` prints the type + docstring for `f` as an info message.
- `#tl_doc_view f` renders the type + docstring for `f` as a rich HTML panel in the InfoView.

Tip: if you already have an identifier in the InfoView (e.g. under “Expected type”), you can also
hover it to see its type + docstring. InfoView hover tooltips can be toggled in VS Code under
`Lean 4 > Infoview: Show Tooltip On Hover`.

## Main definitions

- `#tl_doc f`: print `f`'s type and docstring in a plain info message.
- `#tl_doc_view f`: show the same information in a richer HTML panel.
-/

public meta section

open Lean Elab Command Meta Term
open scoped ProofWidgets.Jsx

namespace NN.Widgets

namespace DocsInternal
open UI

/-- Monospaced block that wraps long lines instead of scrolling, for signatures and docstrings. -/
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

/-- Normalize optional docs into user-facing text. -/
def renderDocString (doc? : Option String) : String :=
  match doc? with
  | some s => s.trimAsciiEnd.toString
  | none => "(no docstring found)"

/-- Pretty-print a declaration type using Lean's normal pretty-printer. -/
def ppConstType (n : Name) : CommandElabM String := do
  let fmt ← liftTermElabM do
    let info ← getConstInfo n
    ppExpr info.type
  pure fmt.pretty

/-- Resolve an input term to a declaration name for doc lookup. -/
def resolveConstFromTerm (t : Syntax) : CommandElabM Name := do
  let e ← liftTermElabM do
    Term.elabTerm t none
  match termConstName? e with
  | some n => pure n
  | none =>
      throwErrorAt t
        "expected a constant (e.g. `nn.linear`), but got an expression that does not resolve to a \
        declaration"

end DocsInternal

/-!
## Commands
-/

syntax (name := tlDocCmd) "#tl_doc " term : command
syntax (name := tlDocViewCmd) "#tl_doc_view " term : command

elab_rules : command
  | `(#tl_doc $t) => do
      let declName ← DocsInternal.resolveConstFromTerm t
      let env ← getEnv
      let doc? ← liftIO <| Lean.findDocString? env declName
      let ty ← DocsInternal.ppConstType declName
      let docStr := DocsInternal.renderDocString doc?
      logInfo m!"{declName} : {ty}\n\n{docStr}"

elab_rules : command
  | `(#tl_doc_view $t) => do
      let declName ← DocsInternal.resolveConstFromTerm t
      let env ← getEnv
      let doc? ← liftIO <| Lean.findDocString? env declName
      let ty ← DocsInternal.ppConstType declName
      let docStr := DocsInternal.renderDocString doc?
      let declStr := toString declName

      let declLit := Syntax.mkStrLit declStr
      let tyLit := Syntax.mkStrLit ty
      let docLit := Syntax.mkStrLit docStr

      -- Attach to a canonical syntax node so the infoview can anchor the panel reliably.
      let cmd ← (UI.canonicalCommand <$> `(
        #html (NN.Widgets.DocsInternal.docPanel $declLit $tyLit $docLit)
      ))
      elabCommand cmd

end NN.Widgets
