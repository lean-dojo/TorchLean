/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import NN.Runtime.Context
public meta import NN.Widgets.Core.Tensor
public meta import NN.Widgets.Core.UI
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Demos.Macro

/-!
# RuntimeCtx

Runtime context viewer widget (variables + gradients).

When stepping through a training loop, the most common debugging questions are:
- Which parameters exist and what shapes do they have?
- Which variables have gradients, and do their shapes match the corresponding values?

This widget renders `Runtime.RuntimeContext` as two tables:
- `var_registry` (values),
- `gradients` (accumulated gradients).

## Main definitions

- `runtimeCtxHtml`: top-level context renderer.
- `sectionHtml`: reusable expandable section builder.
- `kvRow`: per-variable row with tensor preview.
- `#runtime_ctx_view`: command entry point.

## Implementation notes

- We split variables and gradients into separate sections because that mirrors how people debug
  training state ("what exists?" vs "what got grad?").
- Expanded rows show tensor previews directly, which helps identify shape
  issues without leaving the infoview.
- This viewer is intentionally read-only and does not mutate runtime state.

## References

- [ProofWidgets](https://github.com/leanprover-community/ProofWidgets4)
- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)

## Tags

runtime-context, gradients, variables, debugging, proofwidgets
-/

public meta section

open scoped ProofWidgets.Jsx

namespace NN.Widgets

open Runtime
open UI

/-- Render one named runtime tensor entry with shape metadata and preview. -/
private def kvRow {α : Type} [ToString α] (name : String) (v : AnyTensor α) : ProofWidgets.Html :=
  <details style={json% {"margin": "6px 0"}}>
    <summary>
      {monospace name} {pill s!"shape={Spec.Shape.pretty v.s}"} {pill
        s!"size={Spec.Shape.size v.s}"}
    </summary>
    <div style={json% {"margin-top": "8px", "padding-left": "10px"}}>
      {anyTensorHtml (α := α) v (maxRows := 10) (maxCols := 12) (maxElems := 64)}
    </div>
  </details>

/-- Render a titled expandable section for named tensor lists. -/
private def sectionHtml {α : Type} [ToString α] (title : String) (xs : List (String × AnyTensor α))
  : ProofWidgets.Html :=
  <details «open»={false}>
    <summary>
      {pill title} {pill s!"count={xs.length}"}
    </summary>
    <div style={json% {"margin-top": "8px"}}>
      {... xs.toArray.map (fun (k, v) => kvRow (α := α) k v)}
    </div>
  </details>

/-- Render a runtime context as a rich HTML panel. -/
def runtimeCtxHtml {α : Type} [ToString α] (ctx : RuntimeContext α) : ProofWidgets.Html :=
  <div style={json% {
    "padding": "10px",
    "border": "1px solid var(--vscode-panel-border, #e5e5e5)",
    "border-radius": "10px",
    "background": "var(--vscode-editor-background, transparent)",
    "color": "var(--vscode-editor-foreground, inherit)"
  }}>
    <div style={json% {"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin-bottom":
      "10px"}}>
      {pill "RuntimeContext"} {pill s!"vars={ctx.var_registry.length}"} {pill
        s!"grads={ctx.gradients.length}"} {pill s!"next_id={ctx.next_id}"}
    </div>
    {sectionHtml (α := α) "var_registry" ctx.var_registry}
    <div style={json% {"height": "8px"}}></div>
    {sectionHtml (α := α) "gradients" ctx.gradients}
  </div>

/-!
## Commands
-/

syntax (name := runtimeCtxViewCmd) "#runtime_ctx_view " term : command

macro "#runtime_ctx_view " ctx:term : command =>
  Lean.TSyntax.mkInfoCanonical <$> `(#html (runtimeCtxHtml $ctx))

end NN.Widgets
