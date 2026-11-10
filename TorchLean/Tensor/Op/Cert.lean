/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule
public meta import Lean.ScopedEnvExtension
public meta import Lean.Elab.Command
public meta import Lean.Meta.Basic

/-!
# Certificates of registered operators

`@[op_cert Op.<operator> <kind>]` on a theorem records that the theorem is a correctness
certificate of that operator, and `@[op_cert Op.<operator> <kind> 2]` records that it concerns
only the listed input positions, the others held fixed. Each kind names what the theorem states:

* `algebraic`: the pullback is the adjoint of a selected tangent map, over a commutative semiring
  or over the reals, with no claim that the tangent map is a derivative.
* `derivative`: the tangent map is the Fréchet derivative and the pullback its adjoint, on the
  domain and shapes the theorem states.
* `convention`: a tangent rule or pullback fixed by an AD convention, not a derivative (`detach`,
  a subgradient).
* `prepared`: a prepared or compact pullback returns the dense one, beyond the `prepare_eq` field
  every rule carries.
* `roundoff`: a floating-point error bound of a route.
* `enclosure`: the interval transfer of the graph numerical certificate encloses the operator's
  real forward value.

`op_cert` checks that the operator is a constructor of `TorchLean.Op`, that the kind is one of
these and, when the operator's input list reduces to a literal list, that every listed position
is an input of it. It also checks that the
declaration is a theorem whose statement mentions the operator's rule (`TorchLean.Op.rule?` at that
constructor), but not that the statement says what its kind says. A certificate of an existing
record, for instance, is a theorem identifying the record's operator with the arm (as in
`TorchLean.NN.Op.Certificates`). `Op.certificates` lists the registrations of one operator
and `#op_certs Op.<operator>` prints them.
-/

public meta section

open Lean Elab Command

namespace TorchLean.Op

/-- The certificate kinds of `op_cert`. -/
def certKinds : List Name :=
  [`algebraic, `derivative, `convention, `prepared, `roundoff, `enclosure]

/-- One registration: the operator constructor, the kind and the theorem. -/
structure CertEntry where
  /-- The constructor of `TorchLean.Op`. -/
  operator : Name
  /-- One of `certKinds`. -/
  kind : Name
  /-- The registered theorem. -/
  certificate : Name
  /-- The input positions it concerns, the others held fixed. An empty list means every input. -/
  inputs : List Nat := []
  deriving Inhabited, BEq, Repr

/-- The registered certificates, in registration order. -/
initialize certExtension : SimplePersistentEnvExtension CertEntry (Array CertEntry) ←
  registerSimplePersistentEnvExtension {
    addEntryFn := Array.push
    addImportedFn := fun modules => modules.foldl (· ++ ·) #[] }

/-- The length of a list of ports after reducing it to its constructors, or `none` when it does not
reduce to them (as for `Op.cat` and `Op.custom`). -/
partial def portCount (ports : Expr) : Meta.MetaM (Option Nat) := do
  let ports ← Meta.whnf ports
  if ports.isAppOfArity ``List.cons 3 then
    return (← portCount ports.getAppArgs[2]!).map (1 + ·)
  if ports.isAppOfArity ``List.nil 1 then return some 0
  return none

/-- Register a theorem as a certificate of a registered operator. -/
syntax (name := opCertAttr) "op_cert " ident ident (ppSpace num)* : attr

initialize registerBuiltinAttribute {
  name := `opCertAttr
  descr := "a correctness certificate of a registered operator"
  applicationTime := .afterTypeChecking
  add := fun declaration stx _ => do
    let `(attr| op_cert $operator:ident $kind:ident $positions*) := stx |
      throwError "op_cert: expected `op_cert Op.<operator> <kind> <position>...`"
    let inputs := positions.toList.map (·.getNat)
    let env ← getEnv
    let operator ← realizeGlobalConstNoOverloadWithInfo operator
    let some (.ctorInfo info) := env.find? operator |
      throwError "op_cert: {operator} is not a registered operator"
    unless info.induct == ``TorchLean.Op do
      throwError "op_cert: {operator} is not a constructor of TorchLean.Op"
    unless certKinds.contains kind.getId do
      throwError "op_cert: the kind {kind.getId} is not one of {certKinds}"
    let info ← getConstInfo declaration
    unless wasOriginallyTheorem (← getEnv) declaration do
      throwError "op_cert: {declaration} is not a theorem"
    let mentions := info.type.find? fun e =>
      e.isAppOf ``Op.rule? && e.getAppNumArgs > 0 &&
        e.getAppArgs.back!.getAppFn.constName? == some operator
    unless mentions.isSome do
      throwError "op_cert: the statement of {declaration} does not mention the rule of {operator}"
    let inputCount ← Meta.MetaM.run' <| Meta.forallTelescope (← getConstInfo operator).type
      fun _ body => do portCount (← Meta.whnf body).getAppArgs[0]!
    if let some inputCount := inputCount then
      for position in inputs do
        unless position < inputCount do
          throwError "op_cert: {operator} has {inputCount} inputs, not an input at {position}"
    modifyEnv fun env => certExtension.addEntry env
      { operator, kind := kind.getId, certificate := declaration, inputs } }

/-- The certificates registered for `operator`, in registration order. -/
def certificates (env : Environment) (operator : Name) : Array CertEntry :=
  (certExtension.getState env).filter (·.operator == operator)

/-- Print the certificates registered for an operator. -/
syntax (name := opCertsCommand) "#op_certs " ident : command

/-- Elaborate `#op_certs`. -/
@[command_elab opCertsCommand] def elabOpCerts : CommandElab := fun stx => do
  let `(#op_certs $operator:ident) := stx | throwUnsupportedSyntax
  let operator ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo operator
  let entries := certificates (← getEnv) operator
  if entries.isEmpty then
    logInfo m!"{operator}: no certificates"
  else
    logInfo m!"{operator}: {entries.toList.map fun e =>
      if e.inputs.isEmpty then s!"{e.kind} {e.certificate}"
      else s!"{e.kind} {e.certificate} (inputs {e.inputs})"}"

end TorchLean.Op
