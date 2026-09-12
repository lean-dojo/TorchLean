/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import Lean.Attributes
public meta import Lean.Meta.MatchUtil
public import Lean.Exception

/-!
# Certified tensor-view equations

The `einops_view` attribute records equations that expose the tensor view
implemented by an otherwise opaque named function. Consumer compilers use
only these kernel-checked equations; they never inspect or trust an opaque
implementation.
-/

public meta section

namespace TorchLean.Tensor.Internal.Elab.ViewRegistry

open Lean

/-- Reject an `[einops_view]` tag on anything whose conclusion is not an equation. -/
private def validateViewTheorem (declarationName : Name) : AttrM Unit := do
  let declaration ← getConstVal declarationName
  Lean.Meta.MetaM.run' do
    let (_, _, conclusion) ←
      Meta.forallMetaTelescopeReducing declaration.type
    unless (← Meta.matchEq? conclusion).isSome do
      throwError
        "invalid `[einops_view]` theorem `{declarationName}`: \
          the conclusion must be an equality"

/--
Register a theorem that exposes an opaque tensor-producing function as a
coordinate-preserving tensor view.

The theorem should be oriented from the named function application to a
supported `rearrange`, `expand`, `Rep.pull`, or `Rep.reindex`
expression. The compiler applies only the tagged equality itself; it does not
add the theorem to the global simplifier or inspect the named implementation.
-/
public initialize einopsViewAttribute : TagAttribute ←
  registerTagAttribute `einops_view
    "register a kernel-checked tensor-view equation for Einops fusion"
    validateViewTheorem

end TorchLean.Tensor.Internal.Elab.ViewRegistry
