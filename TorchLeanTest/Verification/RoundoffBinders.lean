/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

import TorchLean.Verify.Coverage.Pointwise.Selection
import TorchLean.Verify.Coverage.TapeOperations
import TorchLean.Verify.Coverage.TensorComposites.Normalization
meta import TorchLean.Verify.Coverage.TensorComposites.Normalization

/-!
# The binders of generated roundoff theorems

`#roundoff_coverage` analyzes a configured route under the hypothesis that the format is IEEE, and
the theorem it declares keeps that hypothesis, the scalar's storage and the other instances and
hypotheses of the route only where its statement or proof uses them (`dropUnused`). The first check
runs `dropUnused` on a proof that uses one of two hypotheses and not the instance before them. The
second checks, for every configured route of the imported coverage modules, that its theorem takes
the IEEE hypothesis exactly when the record says the proof uses it, and that the absolute theorem of
`Tensor.fmax` and the mixed propagation theorem of `Tensor.detach` do not take it while the absolute
theorem of `SomeTensor.add` does. The last applies the theorem of `Tensor.fmax` at OCP's E4M3
format, whose encoding is not IEEE.
-/

meta section

namespace TorchLeanTest.RoundoffBinders

open Lean Meta Elab Command TorchLean.Roundoff
open FloatLib.Floats.Formats.BinaryInterchange

-- `fun n [Inhabited Nat] (first : n = n) (second : 0 < n) => second` keeps `n` and `second`.
run_meta do
  let nat := mkConst ``Nat
  withLocalDeclD `n nat fun n => do
  withLocalDecl `inst .instImplicit (← mkAppM ``Inhabited #[nat]) fun inst => do
  withLocalDeclD `first (← mkEq n n) fun first => do
  withLocalDeclD `second (← mkAppM ``LT.lt #[mkNatLit 0, n]) fun second => do
    let arguments := #[n, inst, first, second]
    let proof ← mkLambdaFVars arguments second
    let optional ← optionalArguments arguments
    unless optional == #[1, 2, 3] do throwError "the optional arguments are {optional}"
    let (proof, type, kept, labels) ←
      dropUnused proof (← inferType proof) 4 optional #[(2, "first"), (3, "second")]
    unless kept == #[0, 3] && labels == #["second"] do
      throwError "dropUnused keeps {kept} with the contracts {labels}"
    let expected ← mkForallFVars #[n, second] (← inferType second)
    unless ← isDefEq type expected <&&> isDefEq (← inferType proof) expected do
      throwError "dropUnused gives {proof} : {type}"

/-- Whether the theorem `name` takes the hypothesis that the format is IEEE. -/
def takesIEEE (name : Name) : MetaM Bool := do
  let some info := (← getEnv).find? name | throwError "no {name}"
  forallTelescope info.type fun binders _ => binders.anyM fun binder => do
    return ((← inferType binder).eq?.map fun (_, left, _) =>
      left.isAppOf ``FloatFormat.isIEEE).getD false

-- Each configured theorem takes the IEEE hypothesis exactly when its proof uses it.
run_meta do
  let entries := (coverageExtension.getState (← getEnv)).filter fun entry =>
    entry.backend == "configured" && !entry.evidence.isAnonymous
  if entries.isEmpty then throwError "no configured coverage theorem is imported"
  for entry in entries do
    unless (← takesIEEE entry.evidence) == entry.usesIEEE do
      throwError "{entry.evidence} takes the IEEE hypothesis: {!entry.usesIEEE}, and its record \
        says the proof uses it: {entry.usesIEEE}"
  for (name, expected) in [
      (`TorchLean.Coverage.Pointwise.Tensor.fmax.value_error, false),
      (`TorchLean.Coverage.TensorComposites.Tensor.detach.value_mixed_error_of_input_bounds, false),
      (`TorchLean.Coverage.TapeOperations.SomeTensor.add.value_error, true)] do
    unless (← takesIEEE name) == expected do
      throwError "{name} takes the IEEE hypothesis: {!expected}"

end TorchLeanTest.RoundoffBinders

end

open FloatLib.Floats.Formats.BinaryInterchange

/-- OCP's E4M3 format is not IEEE. -/
example : FloatFormat.e4m3.isIEEE = false := by decide

/-- The absolute bound of `Tensor.fmax` for the configured scalar of OCP's E4M3 format, stored in
a byte. -/
example := @TorchLean.Coverage.Pointwise.Tensor.fmax.value_error FloatFormat.e4m3
  (Configured.StoragePlan.forKnownWidth FloatFormat.e4m3 8 rfl)
  (Configured.Code (Configured.StoragePlan.forKnownWidth FloatFormat.e4m3 8 rfl)) _
