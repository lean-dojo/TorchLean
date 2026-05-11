/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# Widgets UI helpers

TorchLean has multiple ProofWidgets-based viewers (IR graphs, autograd tapes, RL rollouts, etc.).
Several widgets share the same compact HTML helpers (`monospace`, `pill`, status badges, DOT label
escaping).

This file centralizes those helpers so:
- widget modules stay consistent,
- style tweaks happen in one place, and
- we avoid “private def …” duplication across many files.

Design notes:
- This is intentionally lightweight and depends only on ProofWidgets.
- These helpers are **meta** only (used for infoview UI), not part of the executable runtime.

References:
- ProofWidgets: https://github.com/leanprover-community/ProofWidgets4
- GraphViz DOT label escaping rules: https://graphviz.org/doc/info/lang.html
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

namespace UI

/-- Render a string as monospace code, using VS Code theme fonts when available. -/
def monospace (s : String) : ProofWidgets.Html :=
  <code style={json% {
    "font-family":
      "var(--vscode-editor-font-family, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace)"
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

/-- Render an “OK” badge with accent color. -/
def okBadge (s : String) : ProofWidgets.Html :=
  <span style={json% {
    "display": "inline-block",
    "padding": "2px 8px",
    "border-radius": "999px",
    "background": "rgba(0, 200, 120, 0.16)",
    "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
    "font-size": "12px",
    "line-height": "18px",
    "color": "var(--vscode-testing-iconPassed, #0a7)",
    "font-weight": 600
  }}>{.text s}</span>

/-- Render a warning badge. -/
def warnBadge (s : String) : ProofWidgets.Html :=
  <span style={json% {
    "display": "inline-block",
    "padding": "2px 8px",
    "border-radius": "999px",
    "background": "rgba(255, 200, 0, 0.20)",
    "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
    "font-size": "12px",
    "line-height": "18px",
    "color": "var(--vscode-editorWarning-foreground, #b36200)",
    "font-weight": 600
  }}>{.text s}</span>

/-- Render an error badge. -/
def errBadge (s : String) : ProofWidgets.Html :=
  <span style={json% {
    "display": "inline-block",
    "padding": "2px 8px",
    "border-radius": "999px",
    "background": "rgba(255, 80, 80, 0.18)",
    "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
    "font-size": "12px",
    "line-height": "18px",
    "color": "var(--vscode-errorForeground, #c00)",
    "font-weight": 600
  }}>{.text s}</span>

/-- Render a boolean "flag" badge: green when `true`, muted when `false`. -/
def flagBadge (name : String) (isSet : Bool) : ProofWidgets.Html :=
  if isSet then
    <span style={json% {
      "display": "inline-block",
      "padding": "2px 8px",
      "border-radius": "999px",
      "background": "rgba(0, 200, 120, 0.16)",
      "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
      "font-size": "12px",
      "line-height": "18px",
      "color": "var(--vscode-testing-iconPassed, #0a7)"
    }}>{.text name}</span>
  else
    <span style={json% {
      "display": "inline-block",
      "padding": "2px 8px",
      "border-radius": "999px",
      "background": "var(--vscode-badge-background, #f7f7f7)",
      "border": "1px solid var(--vscode-panel-border, #e0e0e0)",
      "font-size": "12px",
      "line-height": "18px",
      "color": "var(--vscode-descriptionForeground, #777)"
    }}>{.text name}</span>

/-- Escape a string so it is safe to embed inside a double-quoted DOT node label. -/
def escapeDotLabel (s : String) : String :=
  -- DOT labels are double-quoted; keep widget-generated DOT robust.
  let s := s.replace "\\" "\\\\"
  let s := s.replace "\"" "\\\""
  s.replace "\n" "\\n"

end UI

end NN.Widgets
