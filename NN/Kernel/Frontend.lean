/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Program
public meta import NN.Kernel.Expr
public meta import Lean.Elab.Tactic.Omega
public meta import Lean.Elab.Tactic.Split
public meta import Lean.Meta.Closure
public meta import Lean.Meta.Eqns
public meta import Lean.Meta.Match.MatcherApp
public meta import Lean.Meta.Tactic.Rewrite

/-!
# Lean-source custom operations

`Program.of` elaborates a Lean function before recognizing its supported computations. The result
retains that function and a kernel-checked equality with the expression evaluator. Unsupported
functions are rejected, rather than executed on the host or silently treated as GPU primitives.
Captured scalar, unsigned-index and Boolean values become constants shared by all output lanes;
calculations involving the reader or any lane-local variable remain in the generated expression.
Fixed-parameter scalar recurrences use the equation compiler's theorems to produce checked loops.
-/

public section

namespace NN.Kernel

/-- Compile an elementwise scalar function, or an indexed function with checked input reads.

`Program.of (fun (x : Float32) => x * x + 1)` reads each element of input zero. Indexed functions
take a `Reader` and `UInt64` index and return `Except Error` explicitly. Both forms retain a
kernel-checked equality with their reference calculation; unsupported operations are rejected.
Captured values can parameterize scalar arithmetic, branches and bounded-fold lengths. They are
evaluated on the host and emitted as constants when GPU source is rendered.
Named recursive functions may have a final natural-number argument, with zero and successor
equations and recursive calls on the predecessor at fixed parameters. The count must be a literal
below `2^64` or a `UInt64.toNat` expression. A proved recurrence bridge preserves read failures
and the arithmetic order while reusing the loop backend.
-/
scoped syntax (name := programStx) "Program.of " term : term

end NN.Kernel

public meta section

namespace NN.Kernel.Frontend

open Lean Lean.Meta Lean.Elab
open Lean.Elab.Term hiding mkConst

/-- Local source variables, newest first, with their expression-language types. -/
private abbrev Context := List (Lean.Expr × Ty)

/-- Semantic bridges for recursive definitions discovered during source recognition. -/
private abbrev ReifyM := StateT (Array Lean.Expr) TermElabM

/-- Use the equation compiler's theorem, rather than unfolding its recursive implementation. -/
def equationBody (source : Lean.Expr) (equations : Array Name) :
    TermElabM (Lean.Expr × Name) := do
  let goal ← mkFreshExprMVar (mkConst ``True)
  for equation in equations do
    let saved ← saveState
    try
      let result ← goal.mvarId!.rewrite source (← mkConstWithFreshMVarLevels equation)
      unless result.mvarIds.isEmpty do throwError "recursive equation has unresolved premises"
      return (← instantiateMVars result.eNew, equation)
    catch _ => saved.restore
  throwError "Program.of: no constructor equation for {source}"

/-- Recognize a fixed-parameter scalar recurrence from its zero and successor equations.

The recursive argument must be last. Every recursive call in the successor equation must use
the predecessor with unchanged parameters; tree recursion and changing recursive state are not
silently turned into this single-state loop. The generated bridge is checked by Lean.
-/
private def recurrence? (α : Lean.Expr) (source : Lean.Expr) :
    TermElabM (Option (Lean.Expr × Lean.Expr)) := do
  let head := source.getAppFn
  let args := source.getAppArgs
  let some name := head.constName? | return none
  if args.isEmpty then return none
  let count := args.back!
  unless ← isDefEq (← inferType count) (mkConst ``Nat) do return none
  unless count.isAppOf ``UInt64.toNat || (← getNatValue? count).isSome do return none
  let some equations ← getEqnsFor? name | return none
  let resultType ← inferType source
  let pureResult ← isDefEq resultType α
  unless pureResult ||
      (← isDefEq resultType (← mkAppM ``Except #[mkConst ``NN.Kernel.Error, α])) do
    return none
  let apply (n : Lean.Expr) := mkAppN head (args.set! (args.size - 1) n)
  let wrap (value : Lean.Expr) : MetaM Lean.Expr :=
    if pureResult then mkAppOptM ``Except.ok #[some (mkConst ``NN.Kernel.Error), some α,
      some value] else pure value
  let (zero, zeroEquation) ← equationBody (apply (mkNatLit 0)) equations
  withLocalDeclD `n (mkConst ``Nat) fun n => do
    let previous := apply n
    let (successor, succEquation) ← equationBody (apply (← mkAppM ``Nat.succ #[n])) equations
    -- Nonrecursive helper functions use ordinary unfolding, not the recurrence bridge.
    unless (successor.find? (·.isConstOf name)).isSome do return none
    withLocalDeclD `i (mkConst ``UInt64) fun i => do
      withLocalDeclD `acc α fun acc => do
        let replacement ← if pureResult then pure acc else
          mkAppOptM ``Except.ok #[some (mkConst ``NN.Kernel.Error), some α, some acc]
        let predecessor ← mkAppM ``UInt64.toNat #[i]
        let body := successor.replace fun term =>
          if term == previous then some replacement
          else if (term.isAppOf ``Nat.toUInt64 || term.isAppOf ``UInt64.ofNat) &&
              term.getAppArgs.back! == n then some i
          else if term == n then some predecessor else none
        if (body.find? (·.isConstOf name)).isSome then
          throwError "Program.of: recursive calls must use the predecessor and fixed parameters"
        let step ← mkLambdaFVars #[i, acc] (← wrap body)
        let reference ← mkLambdaFVars #[n] (← wrap previous)
        let initial ← wrap zero
        let referenceStx ← exprToSyntax reference
        let stepStx ← exprToSyntax step
        let initialStx ← exprToSyntax initial
        let zeroEquation := mkIdent zeroEquation
        let succEquation := mkIdent succEquation
        let bridge ← elabTerm (← `(
          NN.Kernel.iterate_eq_recurrence $referenceStx $stepStx $initialStx
            (by simp only [$zeroEquation:term])
            (by
              intro n
              simp only [$succEquation:term]
              simp only [Bind.bind, Except.bind, Pure.pure, Except.pure,
                Nat.toUInt64, UInt64.ofNat_toNat]))) none
        synthesizeSyntheticMVarsNoPostponing
        let bridge ← instantiateMVars bridge
        if bridge.hasSorry then
          throwError "Program.of: recursive equations do not give a recurrence"
        -- Normalize only monadic plumbing. The rewrite must see the same step function as the
        -- expression evaluator, without unfolding scalar operations or changing their order.
        let unfolds ← #[``Bind.bind, ``Pure.pure, ``Except.bind, ``Except.pure].foldlM
          (fun rules name => rules.addDeclToUnfold name) ({} : SimpTheorems)
        let simpContext ← Simp.mkContext {} #[unfolds]
        let (bridgeType, _) ← dsimp (← inferType bridge) simpContext
        let bridge ← mkExpectedTypeHint bridge bridgeType
        let bridge ← withTransparency .reducible do
          mkAuxTheorem bridgeType bridge
            (kind? := (← getMainModule) ++ `_recurrence)
        let scalar ← exprToSyntax α
        let count ← exprToSyntax count
        let lowered ← elabTerm (← `(do
          let value : $scalar ← ($initialStx)
          NN.Kernel.iterate $stepStx $count 0 value)) none
        synthesizeSyntheticMVarsNoPostponing
        let lowered ← instantiateMVars lowered
        return some (lowered, bridge)

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
    Nat → Lean.Expr → ReifyM Lean.Expr
  | 0, _ => throwError "Program.of: source exceeds the expression traversal limit"
  | fuel + 1, source => do
    let source := source.consumeMData.headBeta
    let Γ ← contextExpr ctx
    let construct (name : Name) (args : Array Lean.Expr) :=
      mkAppOptM name (#[some α, some Γ] ++ args.map some)
    -- Captured values are uniform across output lanes. Never turn a read or a loop-local
    -- calculation into a host constant: those must remain inside the checked expression.
    let dependsOnLane := source.hasAnyFVar fun id =>
      (read.isFVar && read.fvarId! == id) || ctx.any (fun entry => entry.1.fvarId! == id)
    if !dependsOnLane && !source.hasLooseBVars && !source.hasMVar then
      let expected := match type with
        | .scalar => α
        | .index => mkConst ``UInt64
        | .predicate => mkConst ``Bool
      if ← isDefEq (← inferType source) expected then
        if type == .index then
          if let some (n, _) ← getOfNatValue? source ``UInt64 then
            unless n < 2 ^ 64 do
              throwError "Program.of: index literal {n} exceeds the UInt64 range"
        return ← construct (match type with
          | .scalar => ``Expr.scalar | .index => ``Expr.index | .predicate => ``Expr.predicate)
          #[source]
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
    if type == .index && (head.isConstOf ``Nat.toUInt64 || head.isConstOf ``UInt64.ofNat) &&
        args.size == 1 && args[0]!.isAppOf ``UInt64.toNat then
      return ← reify α read ctx .index fuel args[0]!.getAppArgs.back!
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
        let branch (value : Bool) : ReifyM Lean.Expr := do
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
    if type == .scalar then
      if let some (lowered, bridge) ← recurrence? α source then
        let bridgeType ← inferType bridge
        let entries := (#[read] ++ ctx.reverse.toArray.map (·.1)).filter fun entry =>
          bridgeType.hasAnyFVar (· == entry.fvarId!)
        let bridge ← mkLambdaFVars entries bridge
        modify (·.push bridge)
        return ← reify α read ctx type fuel lowered
    if let some name := name then
      if ← isRecursiveDefinition name then
        throwError "Program.of: unsupported recursion; expected a fixed-parameter scalar recurrence"
    if let some unfolded ← unfoldDefinition? source then
      unless unfolded == source do return ← reify α read ctx type fuel unfolded
    throwError "Program.of: unsupported computation {source}"

/-- Check literal-depth programs as specializations of a universally checked unsigned depth.
Keeping the literal out of the proof body prevents kernel conversion from evaluating an entire
recurrence while checking the compiler's certificate. The generated GPU count is still literal.
-/
private partial def withDepths (body : Lean.Expr) (build : Lean.Expr → TermElabM Lean.Expr) :
    TermElabM Lean.Expr := do
  let recursiveNames ← body.getUsedConstants.filterM fun name => isRecursiveDefinition name
  let search : StateT (Option (Lean.Expr × Nat)) TermElabM Unit := body.forEach' fun term => do
    if (← get).isSome then return false
    let args := term.getAppArgs
    if !args.isEmpty && recursiveNames.contains (term.getAppFn.constName?.getD .anonymous) then
      if let some count ← getNatValue? args.back! then
        set (some (term, count))
        return false
    return true
  let (_, candidate) ← search.run none
  let some (candidate, count) := candidate | return ← build body
  unless count < UInt64.size do throwError "Program.of: recursive depth exceeds the UInt64 range"
  withLocalDeclD `depth (mkConst ``UInt64) fun depth => do
    let natural ← mkAppM ``UInt64.toNat #[depth]
    let args := candidate.getAppArgs
    let replacement := mkAppN candidate.getAppFn (args.set! (args.size - 1) natural)
    let body := body.replace fun term => if term == candidate then some replacement else none
    let result ← withDepths body build
    let result ← mkLambdaFVars #[depth] result
    return mkApp result (← mkAppM ``UInt64.ofNat #[mkNatLit count])

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
    unless ← isDefEq (← inferType body) α do
      throwError "Program.of: an elementwise function must return its input scalar type"
    let scalar ← exprToSyntax α
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
        throwError "Program.of: expected NN.Kernel.Reader"
      pure args[1]!
    let scalar ← synthInstance (← mkAppM ``Scalar #[α])
    unless ← isDefEq (← inferType read) (← mkAppM ``Reader #[α]) do
      throwError "Program.of: the first argument must be NN.Kernel.Reader"
    let resultType ← mkAppM ``Except #[mkConst ``NN.Kernel.Error, α]
    unless ← isDefEq (← inferType body) resultType do
      throwError "Program.of: the function must return {resultType}, got {← inferType body}"
    withDepths body fun body => do
      let reference ← mkLambdaFVars locals body
      let (expression, bridges) ← (reify α read [(index, .index)] .scalar 4096 body).run #[]
      let arithmeticType ← mkAppM ``Arithmetic #[α]
      let arithmetic ← elabTermEnsuringType (← `(Arithmetic.ofOps)) (some arithmeticType)
      synthesizeSyntheticMVarsNoPostponing
      let arithmetic ← instantiateMVars arithmetic
      let evaluation ← mkAppM ``evaluate #[arithmetic, expression, read, index]
      let equality ← mkEq evaluation body
      -- Recursive equations provide the proof without speculative definitional evaluation.
      let defeq ← if bridges.isEmpty then isDefEq evaluation body else pure false
      let evidence ← if defeq then
          mkExpectedTypeHint (← mkEqRefl evaluation) equality
        else do
          let goal ← mkFreshExprSyntheticOpaqueMVar equality
          let bridges ← bridges.mapM fun bridge => do
            let term ← exprToSyntax bridge
            `(Parser.Tactic.simpLemma| $term:term)
          let remaining ← withTransparency .reducible <| Lean.Elab.Tactic.run goal.mvarId! do
            Lean.Elab.Tactic.evalTactic (← `(tactic|
              simp (config := { implicitDefEqProofs := false })
                [evaluate, Expr.eval, Env.output, Env.push, Arithmetic.ofOps,
                Compare.index, IndexOp.eval, Bind.bind, Pure.pure,
                Except.instMonad, Except.bind, Except.pure, BEq.beq, instBEqOfDecidableEq,
                Bool.cond_decide, apply_ite, $[$bridges],*] <;>
                repeat' first
                | simp_all [UInt64.lt_iff_toNat_lt, UInt64.le_iff_toNat_le]
                | split
                | omega))
          unless remaining.isEmpty do
            let diagnostic ← Lean.Meta.ppGoal remaining.head!
            throwError "Program.of: could not certify source equivalence; {diagnostic}"
          instantiateMVars goal
      let evidence ← mkExpectedTypeHint evidence equality
      let proof ← mkLambdaFVars locals evidence
      let proofType ← mkForallFVars locals equality
      -- Seal before literal-depth specialization: later elaboration must apply the universal
      -- certificate, not inline its body and ask kernel conversion to evaluate the recurrence.
      -- Closing captured let-values avoids repeated checks of open proof subterms; the resulting
      -- auxiliary theorem is still checked by the kernel.
      let proof ← if bridges.isEmpty then pure proof else withTransparency .reducible do
        mkAuxTheorem proofType proof (zetaDelta := true)
          (kind? := (← getMainModule) ++ `_recurrence)
      let result ← mkAppM ``Program.mk #[scalar, arithmetic, reference, expression, proof]
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
