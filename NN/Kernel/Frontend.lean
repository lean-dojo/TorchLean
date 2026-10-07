/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Program
public meta import NN.Kernel.Expr
public meta import Lean.Meta.Match.MatcherApp
public meta import Lean.Elab.Tactic.Split
public meta import Lean.Elab.Tactic.Omega

/-!
# Lean-source custom operations

`Program.of` elaborates a Lean function before recognizing its supported computations. The result
retains that function and a kernel-checked equality with the expression evaluator. Unsupported
functions are rejected, rather than executed on the host or silently treated as GPU primitives.
-/

public section

namespace NN.Kernel

/-- Compile an elementwise scalar function, or an indexed function with checked input reads.

`Program.of (fun (x : Float32) => x * x + 1)` reads each element of input zero. Indexed functions
take a `Reader` and `UInt64` index and return `Except Error` explicitly. Both forms retain a
kernel-checked equality with their reference calculation; unsupported operations are rejected.
-/
scoped syntax (name := programStx) "Program.of " term : term

end NN.Kernel

public meta section

namespace NN.Kernel.Frontend

open Lean Lean.Meta Lean.Elab
open Lean.Elab.Term hiding mkConst

/-- Local source variables, newest first, with their expression-language types. -/
private abbrev Context := List (Lean.Expr × Ty)

private def typeExpr : Ty → Lean.Expr
  | .scalar => mkConst ``Ty.scalar
  | .index => mkConst ``Ty.index
  | .predicate => mkConst ``Ty.predicate

private def contextExpr (ctx : Context) : MetaM Lean.Expr :=
  mkListLit (mkConst ``Ty) (ctx.map fun entry => typeExpr entry.2)

private def valueType (α : Lean.Expr) (type : Lean.Expr) : MetaM Ty := do
  if ← isDefEq type α then return .scalar
  if ← isDefEq type (mkConst ``UInt64) then return .index
  if ← isDefEq type (mkConst ``Bool) then return .predicate
  throwError "Program.of: unsupported local type {type}; expected the scalar type, UInt64, or Bool"

private def address (ctx : Context) (value : Lean.Expr) : MetaM Lean.Expr := do
  match ctx with
  | [] => throwError "Program.of: unbound source variable {value}"
  | (entry, type) :: rest =>
    if entry == value then
      mkAppOptM ``Var.zero #[some (typeExpr type), some (← contextExpr rest)]
    else
      mkAppOptM ``Var.succ #[some (← contextExpr rest), none, some (typeExpr type),
        some (← address rest value)]

/-- Recognition has a finite traversal budget; it never unfolds arbitrary recursive definitions. -/
private def reify (α read : Lean.Expr) (ctx : Context) (type : Ty) :
    Nat → Lean.Expr → TermElabM Lean.Expr
  | 0, _ => throwError "Program.of: source exceeds the expression traversal limit"
  | fuel + 1, source => do
    let source := source.consumeMData.headBeta
    let Γ ← contextExpr ctx
    let construct (name : Name) (args : Array Lean.Expr) :=
      mkAppOptM name (#[some α, some Γ] ++ args.map some)
    if source.isFVar then
      return ← mkAppOptM ``Expr.var
        #[some α, some Γ, some (typeExpr type), some (← address ctx source)]
    if source.isLet then
      let .letE name localType value body _ := source | unreachable!
      -- Monadic branches introduce lambda-valued join points, not runtime scalar locals.
      if value.isLambda && localType.isForall then
        return ← reify α read ctx type fuel (body.instantiate1 value)
      let localTy ← valueType α localType
      let value ← reify α read ctx localTy fuel value
      return ← withLocalDeclD name localType fun entry => do
        let body ← reify α read ((entry, localTy) :: ctx) type fuel (body.instantiate1 entry)
        mkAppOptM ``Expr.letIn
          #[some α, some Γ, some (typeExpr localTy), some (typeExpr type), some value, some body]
    let args := source.getAppArgs
    let head := source.getAppFn
    if head == read && args.size == 2 then
      unless type == .scalar do throwError "Program.of: an input read must return a scalar"
      let some operand ← getNatValue? args[0]! |
        throwError "Program.of: input numbers must be literal natural numbers"
      return ← construct ``Expr.load #[mkNatLit operand,
        ← reify α read ctx .index fuel args[1]!]
    let name := head.constName?
    if name == some ``ite then
      let predicate ← mkAppOptM ``decide #[some args[1]!, none]
      let predicate ← reify α read ctx .predicate fuel predicate
      let yes ← reify α read ctx type fuel args[3]!
      let no ← reify α read ctx type fuel args[4]!
      return ← mkAppOptM ``Expr.cond
        #[some α, some Γ, some (typeExpr type), some predicate, some yes, some no]
    if let some matcher ← matchMatcherApp? source (alsoCasesOn := true) then
      if matcher.discrs.size == 1 && matcher.alts.size == 2 &&
          (← isDefEq (← inferType matcher.discrs[0]!) (mkConst ``Bool)) then
        let discr := matcher.discrs[0]!
        let branch (value : Bool) : TermElabM Lean.Expr := do
          let replaced := source.replace fun expr =>
            if expr == discr then some (mkConst (if value then ``Bool.true else ``Bool.false))
            else none
          reify α read ctx type fuel (← withTransparency .all (whnf replaced))
        let yes ← branch true
        let no ← branch false
        return ← mkAppOptM ``Expr.cond
          #[some α, some Γ, some (typeExpr type),
            some (← reify α read ctx .predicate fuel discr), some yes, some no]
    let comparisonSource := if name == some ``decide then args[0]! else source
    let comparisonArgs := comparisonSource.getAppArgs
    let comparisonName := comparisonSource.getAppFn.constName?
    let comparisons := [( ``BEq.beq, "eq"), (``Eq, "eq"), (``LT.lt, "lt"), (``LE.le, "le")]
    if type == .predicate then
      if let some (_, op) := comparisons.find? fun entry => comparisonName == some entry.1 then
        let x := comparisonArgs[comparisonArgs.size - 2]!
        let y := comparisonArgs.back!
        let inputType ← valueType α (← inferType x)
        -- Lean elaborates `if b then ...` as a proposition asserting `b = true`.
        if inputType == .predicate && comparisonName == some ``Eq && y.isConstOf ``Bool.true then
          return ← reify α read ctx .predicate fuel x
        unless inputType == .scalar || inputType == .index do
          throwError "Program.of: comparisons require scalar or index operands"
        return ← construct (if inputType == .scalar then ``Expr.compare else ``Expr.indexCompare)
          #[mkConst ((``Compare).str op), ← reify α read ctx inputType fuel x,
            ← reify α read ctx inputType fuel y]
    if name == some ``Pure.pure || name == some ``Except.ok then
      return ← reify α read ctx type fuel args.back!
    if name == some ``Bind.bind then
      let bound := args[args.size - 2]!
      let continuation := args.back!
      return ← lambdaTelescope continuation fun locals body => do
        unless locals.size == 1 do throwError "Program.of: unsupported bind continuation"
        let entry := locals[0]!
        let localTy ← valueType α (← inferType entry)
        let value ← reify α read ctx localTy fuel bound
        let body ← reify α read ((entry, localTy) :: ctx) type fuel body
        mkAppOptM ``Expr.letIn
          #[some α, some Γ, some (typeExpr localTy), some (typeExpr type), some value, some body]
    if name == some ``iterate then
      unless type == .scalar && args.size == 5 do
        throwError "Program.of: unsupported bounded fold signature"
      let count := args[2]!
      let count ← if count.isAppOf ``UInt64.toNat then
          reify α read ctx .index fuel count.getAppArgs.back!
        else if let some n ← getNatValue? count then do
          if n ≥ 2 ^ 64 then throwError "Program.of: fold count exceeds the UInt64 range"
          construct ``Expr.index #[← mkAppM ``UInt64.ofNat #[mkNatLit n]]
        else
          throwError "Program.of: fold counts must be literal naturals or UInt64 values with .toNat"
      let zero ← elabTerm (← `((0 : UInt64))) none
      unless ← withTransparency .all (isDefEq args[3]! zero) do
        throwError "Program.of: bounded folds must start at index zero"
      let initial ← reify α read ctx .scalar fuel args[4]!
      return ← lambdaTelescope args[1]! fun locals body => do
        unless locals.size == 2 do throwError "Program.of: unsupported fold step"
        unless (← isDefEq (← inferType locals[0]!) (mkConst ``UInt64)) &&
            (← isDefEq (← inferType locals[1]!) α) do
          throwError "Program.of: fold steps take an UInt64 index and scalar accumulator"
        let body ← reify α read ((locals[1]!, .scalar) :: (locals[0]!, .index) :: ctx)
          .scalar fuel body
        construct ``Expr.fold #[count, initial, body]
    let operations := [( ``HAdd.hAdd, "add"), (``HSub.hSub, "sub"),
      (``HMul.hMul, "mul"), (``HDiv.hDiv, "div"), (``HMod.hMod, "mod")]
    if let some (_, op) := operations.find? fun entry => name == some entry.1 then
      unless type == .scalar || type == .index do
        throwError "Program.of: arithmetic requires scalar or index operands"
      if type == .scalar && op == "mod" then
        throwError "Program.of: scalar remainder is not supported"
      let operation := mkConst ((if type == .scalar then ``ScalarOp else ``IndexOp).str op)
      return ← construct (if type == .scalar then ``Expr.binary else ``Expr.indexBinary)
        #[operation, ← reify α read ctx type fuel args[args.size - 2]!,
          ← reify α read ctx type fuel args.back!]
    if name == some ``Neg.neg && type == .scalar then
      return ← construct ``Expr.neg #[← reify α read ctx type fuel args.back!]
    if name == some ``OfNat.ofNat || name == some ``OfScientific.ofScientific ||
        source.nat?.isSome || name == some ``Bool.true || name == some ``Bool.false then
      unless !source.hasFVar && !source.hasMVar do
        throwError "Program.of: numeric literals must be closed"
      if type == .index then
        if let some (n, _) ← getOfNatValue? source ``UInt64 then
          unless n < 2 ^ 64 do
            throwError "Program.of: index literal {n} exceeds the UInt64 range"
      return ← construct (match type with
        | .scalar => ``Expr.scalar | .index => ``Expr.index | .predicate => ``Expr.predicate)
        #[source]
    throwError "Program.of: unsupported computation {source}"

/-- Recognize a source function and construct its checked expression representation. -/
def elaborate (source : Syntax) (expectedType? : Option Lean.Expr) : TermElabM Lean.Expr := do
  let reference ← elabTerm source none
  synthesizeSyntheticMVarsNoPostponing
  let reference ← instantiateMVars reference
  let reference ← whnf reference
  let source ← exprToSyntax reference
  let reference ← lambdaTelescope reference fun locals body => do
    if locals.size != 1 then return reference
    let α ← inferType locals[0]!
    let binary32 ← isDefEq α (mkConst ``Float32)
    unless binary32 || (← isDefEq α (mkConst ``Float)) do
      throwError "Program.of: elementwise functions take Float32 or Float"
    unless ← isDefEq (← inferType body) α do
      throwError "Program.of: an elementwise function must return its input scalar type"
    let scalar ← if binary32 then `(Float32) else `(Float)
    elabTerm (← `(fun (read : NN.Kernel.Reader $scalar) (i : UInt64) => do
      let x ← read 0 i
      pure ($source x))) none
  synthesizeSyntheticMVarsNoPostponing
  let reference ← instantiateMVars reference
  lambdaTelescope reference fun locals body => do
    unless locals.size == 2 do
      throwError
        "Program.of: expected a scalar function, or an input reader and UInt64 output index"
    let read := locals[0]!
    let index := locals[1]!
    unless ← isDefEq (← inferType index) (mkConst ``UInt64) do
      throwError "Program.of: the output index must have type UInt64"
    let readerType ← whnf (← inferType read)
    let α ← forallTelescope readerType fun _ result => do
      let args := result.getAppArgs
      unless result.isAppOf ``Except && args.size == 2 do
        throwError "Program.of: expected NN.Kernel.Reader Float32 or NN.Kernel.Reader Float"
      pure args[1]!
    let binary32 ← isDefEq α (mkConst ``Float32)
    unless binary32 || (← isDefEq α (mkConst ``Float)) do
      throwError "Program.of: supported scalar types are Float32 and Float"
    unless ← isDefEq (← inferType read) (← mkAppM ``Reader #[α]) do
      throwError "Program.of: the first argument must be NN.Kernel.Reader"
    let resultType ← mkAppM ``Except #[mkConst ``NN.Kernel.Error, α]
    unless ← isDefEq (← inferType body) resultType do
      throwError "Program.of: the function must return {resultType}, got {← inferType body}"
    let expression ← reify α read [(index, .index)] .scalar 4096 body
    let arithmeticType ← mkAppM ``Arithmetic #[α]
    let arithmetic ← elabTermEnsuringType (← `(Arithmetic.ofOps)) (some arithmeticType)
    synthesizeSyntheticMVarsNoPostponing
    let arithmetic ← instantiateMVars arithmetic
    let evaluation ← mkAppM ``evaluate #[arithmetic, expression, read, index]
    let equality ← mkEq evaluation body
    let evidence ← if ← isDefEq evaluation body then
        mkExpectedTypeHint (← mkEqRefl evaluation) equality
      else do
        let goal ← mkFreshExprSyntheticOpaqueMVar equality
        let remaining ← Lean.Elab.Tactic.run goal.mvarId! do
          Lean.Elab.Tactic.evalTactic (← `(tactic|
            simp [evaluate, Expr.eval, Env.output, Env.push, Arithmetic.ofOps,
              Compare.index, IndexOp.eval, Bind.bind, Pure.pure,
              Except.instMonad, Except.bind, Except.pure, BEq.beq, instBEqOfDecidableEq,
              Bool.cond_decide, apply_ite] <;>
              repeat' first
              | simp_all [UInt64.lt_iff_toNat_lt, UInt64.le_iff_toNat_le]
              | split
              | omega))
        unless remaining.isEmpty do
          let diagnostic ← Lean.Meta.ppGoal remaining.head!
          throwError "Program.of: could not certify source equivalence; {diagnostic}"
        instantiateMVars goal
    let proof ← mkLambdaFVars locals evidence
    let format := mkConst (if binary32 then ``Cuda.Format.binary32 else ``Cuda.Format.binary64)
    let bits ← if binary32 then
        elabTerm (← `(fun (x : Float32) => x.toBits.toUInt64)) none
      else elabTerm (← `(Float.toBits)) none
    let result ← mkAppM ``Program.mk #[format, bits, arithmetic, reference, expression, proof]
    if let some expectedType := expectedType? then
      unless ← isDefEq (← inferType result) expectedType do
        throwError "Program.of: expected {expectedType}, produced {← inferType result}"
    return result

/-- Elaborate the explicit indexed-program constructor. -/
@[term_elab NN.Kernel.programStx]
def elabProgram : TermElab := fun stx expectedType? => withRef stx do
  let `(Program.of $source:term) := stx | throwUnsupportedSyntax
  elaborate source expectedType?

end NN.Kernel.Frontend
