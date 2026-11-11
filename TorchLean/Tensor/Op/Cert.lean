/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule
public meta import Lean.ScopedEnvExtension
public meta import Lean.Elab.Command
public meta import Lean.Elab.Tactic.Basic
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
is an input of it. It also checks that the declaration is a theorem whose statement mentions the
operator's rule (`TorchLean.Op.rule?` at that constructor). For the kinds other than `derivative`
it does not check that the statement says what its kind says. A certificate of an existing
record, for instance, is a theorem identifying the record's operator with the arm (as in
`TorchLean.NN.Op.Certificates`).

For `derivative` it checks the statement against one proposition, `TorchLean.DerivCert f x δ v`
(`TorchLean.NN.FDeriv.DerivCert`): `f` is the real forward of the operator in the listed inputs,
the value outputs collected in their L2 product and the other inputs fixed; `x` is the point of
the listed inputs in the L2 product of their tangent spaces; `δ` is a cotangent of the value
outputs; and `v` is the operator's backward at those inputs and `δ`. With no positions listed, the
listed inputs are the value inputs; a listed input that is not a value port is rejected. When the
inputs are a mapped list `l.map g`, as for `Op.cat`, they vary together and the point is their
family by position (`TorchLean.TProd.parts`). A statement `HasFDerivAt f D x ∧ R` counts as
`DifferentiableAt ℝ f x ∧ R`. The match is up to definitional unfolding, and the kernel confirms
it. Every input and `δ` must be a distinct variable of the statement that does not occur in the
operator, and every other variable an argument of the operator or a proposition. The rules on the
propositions, the proofs the operator takes included, are: (a) none mentions `δ`, (b) each
mentions a variable, (c) none is `False` or an equation with a varying input as one side, and (d)
`decide`, `simp_all` or `omega` does not refute them together. A proposition that is not an
argument of the operator is a domain premise. The match, the refutation and the kernel's check
are bounded in heartbeats. A failure is an error naming the theorem, the operator, the positions
and what failed. Arguments of the operator fixed to constants or to each other are allowed here;
the library check counts only certificates that fix none.

`Op.certificates` lists the registrations of one operator and `#op_certs Op.<operator>` prints
them. `Op.derivEntry?` gives what the `derivative` check found: the varying positions, the domain
premises, and whether the statement gives the derivative itself.
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

/-- What the `derivative` check of `op_cert` found in a registered certificate. -/
structure DerivEntry where
  /-- The registered theorem. -/
  certificate : Name
  /-- The input positions in which the forward is differentiated, in increasing order; empty when
  the inputs are a mapped list, which vary together. -/
  varying : List Nat
  /-- The names of the domain premises. -/
  premises : List Name
  /-- Whether the statement gives the derivative itself, as `HasFDerivAt f D x ∧ …`. -/
  strong : Bool

/-- What the `derivative` check found in the registered `derivative` certificates, in registration
order. -/
initialize derivExtension : SimplePersistentEnvExtension DerivEntry (Array DerivEntry) ←
  registerSimplePersistentEnvExtension {
    addEntryFn := Array.push
    addImportedFn := fun modules => modules.foldl (· ++ ·) #[] }

/-- What the `derivative` check found in `certificate`, if it is a registered `derivative`
certificate. -/
def derivEntry? (env : Environment) (certificate : Name) : Option DerivEntry :=
  (derivExtension.getState env).find? (·.certificate == certificate)

/-- The ports of a list of ports after reducing it to its constructors, or `none` when it does not
reduce to them (as for `Op.cat` and `Op.custom`). -/
partial def literalPorts (ports : Expr) : Meta.MetaM (Option (List Expr)) := do
  let ports ← Meta.whnf ports
  if ports.isAppOfArity ``List.cons 3 then
    return (← literalPorts ports.appArg!).map (ports.appFn!.appArg! :: ·)
  if ports.isAppOfArity ``List.nil 1 then return some []
  return none

namespace DerivCheck

open Meta

/-- The heartbeats, in thousands, of the match with the canonical proposition and of the kernel's
check of it. -/
def matchHeartbeats : Nat := 100000

/-- The heartbeats, in thousands, of the attempt to refute the premises. -/
def refuteHeartbeats : Nat := 4000

/-- `x` with at most `heartbeats` thousand heartbeats from now, or `none` when it throws or runs
out, the state restored. -/
def bounded {α : Type} (heartbeats : Nat) (x : MetaM α) : MetaM (Option α) := do
  let s ← saveState
  tryCatchRuntimeEx
    (withTheReader Core.Context (fun c => { c with maxHeartbeats := heartbeats * 1000 }) <|
      withCurrHeartbeats do return some (← x))
    fun _ => do s.restore; return none

/-- Whether a port is a value port, and its kind. -/
def portKind (port : Expr) : MetaM (Bool × Expr) := do
  let kind ← whnf (mkApp (mkConst ``TorchLean.Port.kind) port)
  return (kind.isConstOf ``TorchLean.Port.Kind.value, kind)

/-- The L2 product `WithLp 2 (t × …)` of `types`, right-nested, the last factor unwrapped. -/
def nestType (two : Expr) : List Expr → MetaM Expr
  | [] => throwError "op_cert: an empty product"
  | [t] => pure t
  | t :: ts => do mkAppM `WithLp #[two, ← mkAppM ``Prod #[t, ← nestType two ts]]

/-- The point of `nestType` with the coordinates `xs`. -/
def nestPoint (two : Expr) : List Expr → MetaM Expr
  | [] => throwError "op_cert: an empty product"
  | [x] => pure x
  | x :: xs => do mkAppM `WithLp.toLp #[two, ← mkAppM ``Prod.mk #[x, ← nestPoint two xs]]

/-- Coordinate `j` of a point `p` of the `nestType` of `n` factors. -/
def nestProj (p : Expr) : Nat → Nat → MetaM Expr
  | 0, _ | 1, _ => pure p
  | _ + 1, 0 => mkAppM `WithLp.fst #[p]
  | n + 1, j + 1 => do nestProj (← mkAppM `WithLp.snd #[p]) n j

/-- Component `i` of a tuple of nested pairs. -/
def tupleProj (t : Expr) : Nat → MetaM Expr
  | 0 => mkAppM ``Prod.fst #[t]
  | i + 1 => do tupleProj (← mkAppM ``Prod.snd #[t]) i

/-- The tuple of nested pairs of `xs`, ending in `()`. -/
def mkTuple : List Expr → MetaM Expr
  | [] => pure (mkConst ``Unit.unit)
  | x :: xs => do mkAppM ``Prod.mk #[x, ← mkTuple xs]

/-- The error of a statement that is not the canonical proposition. -/
def reject {α : Type} (declaration operator : Name) (positions : List Nat) (why : MessageData) :
    MetaM α :=
  throwError "op_cert: {declaration} is not a derivative certificate of {operator} at the \
    inputs {toString positions}: {why}"

/-- An instance of the class `cls` at the reals. -/
def realInstance (cls : Expr) : MetaM Expr := do
  let goal := mkApp cls (mkConst `Real)
  let some inst ← synthInstance? goal |
    throwError "op_cert: no instance {goal} for a derivative certificate"
  return inst

/-- Check that the statement of `declaration` is the canonical proposition `DerivCert f x δ v` of
`operator` at the input positions `positions` (the value inputs when empty), with domain premises
only, and return what it found. -/
def check (declaration operator : Name) (positions : List Nat) : MetaM DerivEntry := do
  unless (← getEnv).contains `TorchLean.DerivCert do
    throwError "op_cert: import TorchLean.NN.FDeriv.DerivCert to register a derivative \
      certificate"
  let real := mkConst `Real
  let instS ← realInstance (mkConst ``TorchLean.Storage [Level.zero])
  let instC ← realInstance (mkConst ``TorchLean.Context)
  let instCap ← realInstance (mkConst ``TorchLean.Op.Capabilities)
  let two ← mkNumeral (mkConst `ENNReal) 2
  let info ← getConstInfo declaration
  forallTelescopeReducing info.type fun bs body => do
  let fail {α : Type} (positions : List Nat) (why : MessageData) : MetaM α :=
    reject declaration operator positions why
  let some app := body.find? fun e =>
      e.isAppOfArity ``TorchLean.Op.rule? 7 && e.appArg!.getAppFn.isConstOf operator |
    fail positions m!"its conclusion does not mention the rule of {operator}"
  let op := app.appArg!
  if op.hasLooseBVars then
    fail positions m!"the operator {op} depends on a bound variable"
  let opType ← whnf (← inferType op)
  unless opType.isAppOfArity ``TorchLean.Op 2 do
    fail positions m!"the operator has the type {opType}"
  let ins := opType.appFn!.appArg!
  let some outPorts ← literalPorts opType.appArg! |
    fail positions m!"its output ports {opType.appArg!} are not a literal list"
  let values ← (List.range outPorts.length).filterM fun o => return (← portKind outPorts[o]!).1
  if values.isEmpty then
    fail positions m!"{operator} has no value output"
  let tangent (port : Expr) : MetaM Expr :=
    whnf (mkApp3 (mkConst ``TorchLean.Port.Tangent) real instS port)
  let carrier (port : Expr) : MetaM Expr :=
    whnf (mkApp3 (mkConst ``TorchLean.Port.Carrier) real instS port)
  -- the cotangent of the value outputs, and the tangents of the outputs it gives
  let δ ← mkFreshExprMVar (← nestType two (← values.mapM fun o => tangent outPorts[o]!))
  let cot ← mkTuple (← (List.range outPorts.length).mapM fun o =>
    match values.idxOf? o with
    | some j => nestProj δ values.length j
    | none => pure (mkConst ``Unit.unit))
  let ruleOpt := mkAppN (mkConst ``TorchLean.Op.rule?) #[real, instS, instC, instCap, ins,
    opType.appArg!, op]
  let isSome ← mkEq (← mkAppM ``Option.isSome #[ruleOpt]) (mkConst ``Bool.true)
  let rule ← mkAppOptM ``Option.get #[none, ruleOpt, ← mkFreshExprMVar isSome]
  let output (y : Expr) : MetaM Expr := do nestPoint two (← values.mapM (tupleProj y ·))
  -- the varying positions, the input variables, the point, the forward and the pullback
  let (S, inputs, varying, p₀, f, v) ← match ← literalPorts ins with
    | some inPorts => do
      let S ← if positions.isEmpty then
          (List.range inPorts.length).filterM fun i => return (← portKind inPorts[i]!).1
        else pure positions
      unless S.Pairwise (· < ·) do
        fail positions m!"the positions are not increasing"
      if S.isEmpty then
        fail positions m!"{operator} has no value input"
      for i in S do
        let some port := inPorts[i]? |
          fail positions m!"{operator} has {inPorts.length} inputs, not an input at {i}"
        let (isValue, kind) ← portKind port
        unless isValue do
          fail positions m!"input {i} is a {kind} port, not a value port"
      let xs ← inPorts.mapM fun port => do mkFreshExprMVar (← carrier port)
      let E ← nestType two (← S.mapM fun i => tangent inPorts[i]!)
      let f ← withLocalDeclD `p E fun p => do
        let t ← mkTuple (← (List.range xs.length).mapM fun i =>
          match S.idxOf? i with
          | some j => nestProj p S.length j
          | none => pure xs[i]!)
        mkLambdaFVars #[p] (← output (← mkAppM ``TorchLean.Op.Rule.forward #[rule, t]))
      let pullback ← mkAppM ``TorchLean.Op.Rule.backward #[rule, ← mkTuple xs, cot]
      pure (S, xs, S.map (xs[·]!), ← nestPoint two (S.map (xs[·]!)), f,
        ← nestPoint two (← S.mapM (tupleProj pullback ·)))
    | none => do
      let mapped ← whnf ins
      unless mapped.isAppOfArity ``List.map 4 do
        fail positions m!"its input ports {ins} are neither a literal list nor a mapped list"
      unless positions.isEmpty do
        fail positions m!"its input ports {ins} are a mapped list, whose inputs vary together"
      let g := mapped.appFn!.appArg!
      let l := mapped.appArg!
      let family ← withLocalDeclD `i (mkApp (mkConst ``Fin) (← mkAppM ``List.length #[l]))
        fun i => do
          let port := (mkApp g (← mkAppM ``List.get #[l, i])).headBeta
          let (isValue, kind) ← portKind port
          unless isValue do
            fail positions m!"its mapped input ports are {kind} ports, not value ports"
          mkLambdaFVars #[i] (← tangent port)
      let E ← mkAppM `PiLp #[two, family]
      let xs ← mkFreshExprMVar (mkApp3 (mkConst ``TorchLean.Values) real instS ins)
      let parts (β tuple : Expr) : MetaM Expr := do
        mkAppM `WithLp.toLp #[two, ← mkAppOptM `TorchLean.TProd.parts #[none, none, β, g, l, tuple]]
      let f ← withLocalDeclD `p E fun p => do
        let t ← mkAppOptM `TorchLean.TProd.ofParts
          #[none, none, mkApp2 (mkConst ``TorchLean.Port.Carrier) real instS, g, l,
            ← mkAppM `WithLp.ofLp #[p]]
        mkLambdaFVars #[p] (← output (← mkAppM ``TorchLean.Op.Rule.forward #[rule, t]))
      let pullback ← mkAppM ``TorchLean.Op.Rule.backward #[rule, xs, cot]
      pure ([], [xs], [xs], ← parts (mkApp2 (mkConst ``TorchLean.Port.Carrier) real instS) xs, f,
        ← parts (mkApp2 (mkConst ``TorchLean.Port.Tangent) real instS) pullback)
  let expected ← mkAppM `TorchLean.DerivCert #[f, p₀, δ, v]
  -- the rule (d) goal, in the context of the binders only
  let refutation ← mkFreshExprMVar (mkConst ``False)
  withLocalDeclD `h body fun h => do
  let strong := body.isAppOfArity ``And 2 && body.appFn!.appArg!.isAppOf `HasFDerivAt
  let (target, proof) ← if strong then do
      let d ← mkAppM `HasFDerivAt.differentiableAt #[← mkAppM ``And.left #[h]]
      let rest ← mkAppM ``And.right #[h]
      pure (mkAnd (← inferType d) (← inferType rest), ← mkAppM ``And.intro #[d, rest])
    else pure (body, h)
  let some true ← bounded matchHeartbeats (withTransparency .default (isDefEq target expected)) |
    fail S m!"its statement is not `DerivCert f x δ v` with `f` the operator's forward in these \
      inputs, `x` a point of them, `δ` a cotangent of its value outputs and `v` its pullback of \
      `δ` at these inputs"
  -- every input and the cotangent are distinct variables of the statement
  let mut seen : Array Expr := #[]
  for (x, what) in inputs.zipIdx.map (fun (x, i) => (x, m!"input {i}")) ++
      [(δ, m!"the cotangent")] do
    let x ← instantiateMVars x
    unless x.isFVar && bs.contains x do
      fail S m!"{what} is {x}, not a variable of the statement"
    if seen.contains x then
      fail S m!"{what} is {x}, a variable already used for another input"
    if op.containsFVar x.fvarId! then
      fail S m!"{what} is {x}, which occurs in the operator"
    seen := seen.push x
  let δ ← instantiateMVars δ
  let varying ← varying.mapM instantiateMVars
  -- every other binder is an argument of the operator or a proposition; every proposition passes
  -- the premise rules, and the domain premises are those that are not arguments of the operator
  let arguments := op.getAppArgs
  let mut propositions := #[]
  let mut premises := #[]
  for b in bs do
    if seen.contains b then continue
    let type ← inferType b
    let name ← b.fvarId!.getUserName
    unless ← isProp type do
      if arguments.contains b then continue
      fail S m!"the binder {name} is not a proposition, an argument of the operator, an input or \
        the cotangent"
    let premise {α : Type} (rule : String) (why : MessageData) : MetaM α :=
      fail S m!"the premise {name} is rejected by rule ({rule}): {why}"
    if type.containsFVar δ.fvarId! then premise "a" m!"it mentions the cotangent"
    if (← whnfR type).isConstOf ``False then premise "c" m!"it is False"
    if let some (_, a, c) := type.eq? then
      if varying.contains a || varying.contains c then
        premise "c" m!"it fixes a varying input"
    unless type.hasFVar do premise "b" m!"it mentions no variable"
    propositions := propositions.push name
    unless arguments.contains b do premises := premises.push name
  unless propositions.isEmpty do
    let s ← saveState
    let tactic ← `(tactic| first | decide | simp_all | omega)
    let refuted ← bounded refuteHeartbeats do
      let goals ← Term.TermElabM.run' <|
        Tactic.run refutation.mvarId! (Tactic.evalTactic tactic)
      return goals.isEmpty
    s.restore
    if refuted == some true then
      fail S m!"the premises {toString propositions.toList} are rejected by rule (d): `first | \
        decide | simp_all | omega` refutes them"
  -- the kernel confirms the match, on a copy of the environment that is then dropped
  let type ← instantiateMVars (← mkForallFVars (bs.push h) expected)
  let value ← instantiateMVars (← mkLambdaFVars (bs.push h) proof)
  if type.hasMVar || value.hasMVar then
    fail S m!"its statement leaves part of the canonical proposition undetermined"
  let decl := Declaration.thmDecl
    { name := declaration ++ `opCertCheck, levelParams := info.levelParams, type, value }
  match (← getEnv).addDeclCore (matchHeartbeats * 1000).toUSize
      (maxRecDepth.get (← getOptions)).toUSize decl none with
  | .ok _ => pure ()
  | .error e =>
    fail S m!"the kernel does not confirm the match: {e.toMessageData (← getOptions)}"
  return { certificate := declaration, varying := S, premises := premises.toList, strong }

end DerivCheck

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
    let inputPorts ← Meta.MetaM.run' <| Meta.forallTelescope (← getConstInfo operator).type
      fun _ body => do literalPorts (← Meta.whnf body).getAppArgs[0]!
    if let some inputPorts := inputPorts then
      for position in inputs do
        unless position < inputPorts.length do
          throwError "op_cert: {operator} has {inputPorts.length} inputs, not an input at \
            {position}"
    if kind.getId == `derivative then
      let entry ← Meta.MetaM.run' <| DerivCheck.check declaration operator inputs
      modifyEnv fun env => derivExtension.addEntry env entry
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
