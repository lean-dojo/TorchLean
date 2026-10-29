/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public meta import Lean.LabelAttribute

/-!
# Numerical automation rules

The attribute is separate from the analyzer so implementations can register existing numerical
bounds, tensor-read equalities and shape laws without importing proof search. Matching uses the
entire theorem, including its arithmetic capabilities and side conditions. Imported theorems can be
registered directly, and local registrations stay in their scope.
Compiler replacements carrying both `csimp` and `roundoff_rule` normalize optimized programs
back to their proved source code.
-/

public meta section

namespace Proofs.RuntimeApprox.Roundoff

/-- A scalar or tensor observation, shape equality, error bound, or proved compiler replacement.
Exact observations contribute zero local error and retain all their original premises. -/
public initialize roundoffRules : Lean.LabelExtension ←
  Lean.registerLabelAttr `roundoff_rule "register a proved numerical observation or error bound"

initialize Lean.registerTraceClass `TorchLean.roundoff

end Proofs.RuntimeApprox.Roundoff
