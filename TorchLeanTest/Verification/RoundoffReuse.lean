/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

import TorchLean.Verify.Coverage.TensorComposites.Normalization
meta import TorchLean.Verify.Coverage.TensorComposites.Normalization

/-!
# The reuse of an imported propagation theorem

`TorchLean/Verify/Coverage/TensorComposites/Normalization.lean` declares the mixed propagation
theorem of `TorchLean.Normalization.normalize`, whose proof this module cannot read. The first check
records an analysis with its statement whose proof applies another propagation theorem and another
primitive theorem: the analysis is applied under the imported name, whose record, from which
`describeTheorem` reads the contracts of a composite applying it, keeps the bounds, propagation
theorems and primitive theorems the declaring module recorded, and whose closure does not reach the
other propagation theorem.
The second runs `#roundoff_propagation` on `normalize` in this module, which applies the imported
theorem and declares no constant, neither a theorem of its own nor a lemma of its analysis.
-/

meta section

namespace TorchLeanTest.RoundoffReuse

open Lean Elab Command Meta TorchLean.Roundoff
open FloatLib.Floats.Formats.BinaryInterchange.Roundoff (primitiveRoundings primitiveEnclosures)

/-- The mixed propagation theorem of `normalize` that the coverage declares. -/
def normalizeMixed : Name :=
  `TorchLean.Coverage.TensorComposites.Normalization.normalize.value_mixed_error_of_input_bounds

-- An analysis with the statement of `normalizeMixed` through another arm and primitive theorem.
run_cmd liftTermElabM do
  let environment ← getEnv
  let some info := environment.find? normalizeMixed | throwError "no {normalizeMixed}"
  let some declared := (propagationExtension.getState environment).find? (·.name == normalizeMixed)
    | throwError "{normalizeMixed} is not recorded as a declared propagation theorem"
  let some primitive := (primitiveRoundings ++ primitiveEnclosures).find?
      (!declared.primitives.contains ·) |
    throwError "{normalizeMixed} uses every primitive theorem"
  let other := `TorchLeanTest.RoundoffReuse.other
  let applied ← recordPropagation `TorchLeanTest.RoundoffReuse.normalize
    { type := info.type, proof := mkConst primitive, bounds := #[], arms := #[other],
      auxiliary := #[], primitives := #[primitive] }
  unless applied == normalizeMixed do throwError "the analysis is applied under {applied}"
  let some record ← propagationTheorem? applied | throwError "no record of {applied}"
  unless record.bounds == declared.bounds && record.arms == declared.arms &&
      record.primitives == declared.primitives do
    throwError "the record of {applied} is the analysis's: bounds {record.bounds}, arms \
      {record.arms}, primitive theorems {record.primitives}"
  let closure ← armClosure #[mkConst applied]
  unless !closure.contains other && declared.arms.all closure.contains &&
      closure.back? == some applied do
    throwError "the closure of {applied} is {closure}"

-- `#roundoff_propagation` in this module applies `normalizeMixed` and declares nothing.
run_cmd do
  let count := fun (environment : Environment) =>
    environment.constants.foldStage2 (fun count _ _ => count + 1) (0 : Nat)
  let before := count (← getEnv)
  elabCommand (← `(command| #roundoff_propagation
    $(mkIdent `mixed) := $(mkIdent `TorchLean.Normalization.normalize)))
  let after := count (← getEnv)
  unless after == before do
    throwError "#roundoff_propagation declared {after - before} constants"

end TorchLeanTest.RoundoffReuse

end
