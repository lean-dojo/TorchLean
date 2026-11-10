/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json.Elab
public import ProofWidgets.Data.Html
public meta import ProofWidgets.Component.HtmlDisplay -- shake: keep

/-!
# Widgets UI helpers

Shared HTML formatting, status badges and source-location handling for ProofWidgets viewers.
Everything here is meta code, evaluated by the elaborator when a widget command runs.

References:
- ProofWidgets: https://github.com/leanprover-community/ProofWidgets4
-/

public meta section

open scoped ProofWidgets.Jsx

namespace TorchLean.Widget

namespace UI

/-- Mark the synthetic source info of a macro-generated command as canonical, so the infoview
anchors its widget panel at the command. -/
def canonicalCommand (stx : Lean.TSyntax `command) : Lean.TSyntax `command :=
  ⟨match stx.raw with
    | .missing => .missing
    | .node info kind args => .node (canonicalInfo info) kind args
    | .atom info value => .atom (canonicalInfo info) value
    | .ident info raw value pre => .ident (canonicalInfo info) raw value pre⟩
where
  canonicalInfo : Lean.SourceInfo → Lean.SourceInfo
    | .synthetic start stop _ => .synthetic start stop true
    | info => info

/-- Render a string as monospace code, using VS Code theme fonts when available. -/
def monospace (s : String) : ProofWidgets.Html :=
  <code style={json% {
    "font-family":
      "var(--vscode-editor-font-family, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \
      monospace)"
  }}>{.text s}</code>

/-- Render a small “pill” badge (used for compact key/value metadata). -/
def pill (s : String) : ProofWidgets.Html :=
  <span
    style={json% {
      "display": "inline-block",
      "padding": "2px 8px",
      "border-radius": "999px",
      "background": "var(--vscode-badge-background, #f2f2f2)",
      "color": "var(--vscode-badge-foreground, inherit)",
      "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
      "font-size": "12px",
      "line-height": "18px"
    }}>{.text s}</span>

/-- Badge layout shared by `okBadge`, `warnBadge` and `errBadge`, which pass a tinted background
and a theme text color. -/
private def statusBadge (s background color : String) : ProofWidgets.Html :=
  <span style={json% {
    "display": "inline-block",
    "padding": "2px 8px",
    "border-radius": "999px",
    "background": $(Lean.toJson background),
    "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
    "font-size": "12px",
    "line-height": "18px",
    "color": $(Lean.toJson color),
    "font-weight": 600
  }}>{.text s}</span>

/-- Render a success badge. -/
def okBadge (s : String) : ProofWidgets.Html :=
  statusBadge s "rgba(0, 200, 120, 0.16)" "var(--vscode-testing-iconPassed, #0a7)"

/-- Render a warning badge. -/
def warnBadge (s : String) : ProofWidgets.Html :=
  statusBadge s "rgba(255, 200, 0, 0.20)" "var(--vscode-editorWarning-foreground, #b36200)"

/-- Render an error badge. -/
def errBadge (s : String) : ProofWidgets.Html :=
  statusBadge s "rgba(255, 80, 80, 0.18)" "var(--vscode-errorForeground, #c00)"

/-- Render a boolean flag: green when set, muted otherwise. -/
def flagBadge (name : String) (isSet : Bool) : ProofWidgets.Html :=
  if isSet then okBadge name else pill name

end UI

end TorchLean.Widget
