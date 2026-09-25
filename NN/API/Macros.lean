/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders -- shake: keep

/-!
# Sequential Model Literals

TorchLean sequential models are shape-indexed (`Sequential σ τ`), so a plain `List` of layers
cannot describe a model the way PyTorch's `nn.Sequential([...])` does: every element would need
the same type. This file provides two list-shaped spellings that expand to ordinary composition
through `TorchLean.nn.compose`:

- `nn.Sequential![a, b, c]` runs each entry as a monadic layer builder and returns the composed
  model in that monad. This is the spelling used with the seeded builders of `NN.API.Seeded`.
- `nn.compose![a, b, c]` composes already-built layers or sequential models without any monad.

Both are scoped syntax in the `TorchLean` namespace, so they become available after
`open TorchLean`. The `!` suffix keeps `nn.Sequential` itself usable as a type name in expressions
such as `nn.Sequential σ τ`.

When adjacent entries have incompatible shapes, the diagnostic identifies their positions in
the literal and reports the preceding output shape and the following input shape.
-/

public section

namespace TorchLean

/--
`nn.Sequential![a, b, c]` builds a sequential model from monadic layer builders.

Each entry is run in order in the ambient monad and the results are composed left to right with
`TorchLean.nn.compose`, so the output shape of every entry must match the input shape of
the next. A single entry is converted to a `Sequential` model with
`TorchLean.nn.AsSequential.asSequential`. Entries may be layers or sequential models.

Example:
```lean
-- As close to `torch.nn.Sequential([...])` as a shape-indexed model can get: the entries are
-- builders, they run in order, and the shapes have to line up or the model does not compile.
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```
-/
scoped syntax (name := nnSequentialBangLit) "nn.Sequential!" "[" term,+ "]" : term

/--
`nn.compose![a, b, c]` composes already-built layers and sequential models without a monad.

Entries are composed left to right with `TorchLean.nn.compose`, so the output shape of
every entry must match the input shape of the next. A single entry is returned unchanged.

Example:
```lean
-- The same list spelling for models that are already built, with no monad in the way.
def stack (first : nn.Sequential [4] [8]) (second : nn.Sequential [8] [2]) :
    nn.Sequential [4] [2] :=
  nn.compose![first, second]
```
-/
scoped syntax (name := nnComposeBangLit) "nn.compose!" "[" term,+ "]" : term

meta section

open Lean Meta Elab Term

-- The expected right-operand type in `compose` is `G output result`. Read its input
-- index only after checking fails; infer the actual input through the user's AsSequential.
private def shapeMismatch? (expectedType value : Expr) : MetaM (Option (Format × Format)) := do
  let saved ← Meta.saveState
  try
    let .app (.app _ output) _ ← instantiateMVars expectedType | return none
    let model ← mkAppM ``nn.AsSequential.asSequential #[value]
    let type ← instantiateMVars (← whnf (← inferType model))
    unless type.isAppOfArity ``Runtime.Autograd.Model.Layers.Seq 2 do
      return none
    let input := type.getAppArgs[0]!
    let output ← instantiateMVars output
    let input ← instantiateMVars input
    if output.hasExprMVar || input.hasExprMVar then
      return none
    if ← isDefEq output input then
      return none
    return some (← ppExpr (← reduce output), ← ppExpr (← reduce input))
  catch _ =>
    return none
  finally
    saved.restore

private def compositionError (literal : String) (index : Nat)
    (expectedType value : Expr) : MetaM MessageData := do
  if let some (output, input) ← shapeMismatch? expectedType value then
    return m!"{literal}: layer {index + 1} expects input shape {input}, \
      but layer {index} outputs {output}.\n\
      Change layer {index + 1}'s input shape or insert a layer that converts \
      {output} to {input}."
  mkTypeMismatchError none value (← inferType value) expectedType

private def withComposition (literal : String) (index : Nat) (nextEntry : Syntax)
    (left right : Term) (continuation : Term → TermElabM Expr) : TermElabM Expr := do
  -- Coercion resolution can be deferred. Both callbacks run only on failure, so
  -- ordinary inference and successful coercions keep their original behavior.
  elabToSyntax (ref := nextEntry) (fun expectedType? => do
    let value ← elabTerm right expectedType? (catchExPostpone := false)
    ensureHasTypeWithErrorMsgs expectedType? value
      (fun _ => compositionError literal index)
      (fun _ => compositionError literal index)) fun checkedRight => do
        continuation (← `(TorchLean.nn.compose $left $checkedRight))

-- Preserve the original right-associated expansion, including each bind and pure.
-- elabToSyntax adds a diagnostic hook without introducing another surface syntax.
private partial def expandSequential (entries : List Term) (index : Nat)
    (continuation : Term → TermElabM Expr) : TermElabM Expr := do
  match entries with
  | [entry] =>
    continuation (← `(do
      let a ← ($entry)
      pure (TorchLean.nn.AsSequential.asSequential a)))
  | [first, second] =>
    withComposition "nn.Sequential!" index second (← `(a)) (← `(b)) fun composition => do
      continuation (← `(do
        let a ← ($first)
        let b ← ($second)
        pure $composition))
  | first :: second :: rest =>
    expandSequential (second :: rest) (index + 1) fun tail => do
      withComposition "nn.Sequential!" index second (← `(a)) (← `(bc)) fun composition => do
        continuation (← `(do
          let a ← ($first)
          let bc ← ($tail)
          pure $composition))
  | [] => throwUnsupportedSyntax

private partial def expandCompose (entries : List Term) (index : Nat)
    (continuation : Term → TermElabM Expr) : TermElabM Expr := do
  match entries with
  | [entry] => continuation entry
  | [first, second] => withComposition "nn.compose!" index second first second continuation
  | first :: second :: rest =>
    expandCompose (second :: rest) (index + 1) fun tail =>
      withComposition "nn.compose!" index second first tail continuation
  | [] => throwUnsupportedSyntax

namespace nn.Internal

/-- Elaborate monadic model literals with diagnostics at adjacent entry boundaries. -/
@[term_elab TorchLean.nnSequentialBangLit]
def elabSequentialLiteral : TermElab := fun stx expectedType? => do
  let `(nn.Sequential![$entries:term,*]) := stx | throwUnsupportedSyntax
  expandSequential entries.getElems.toList 1 fun expanded =>
    withMacroExpansion stx expanded <| elabTerm expanded expectedType?

/-- Elaborate pure model literals with diagnostics at adjacent entry boundaries. -/
@[term_elab TorchLean.nnComposeBangLit]
def elabComposeLiteral : TermElab := fun stx expectedType? => do
  let `(nn.compose![$entries:term,*]) := stx | throwUnsupportedSyntax
  expandCompose entries.getElems.toList 1 fun expanded =>
    withMacroExpansion stx expanded <| elabTerm expanded expectedType?

end nn.Internal

end

end TorchLean
