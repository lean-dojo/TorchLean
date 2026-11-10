/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Fold
public import TorchLean.CUDA.Tree.Derive
public import TorchLean.CUDA.Kernel.Elab
public meta import TorchLean.CUDA.Kernel.Expr
public meta import Lean.Elab.Tactic.Omega
public meta import Lean.Elab.Tactic.Split
public meta import Lean.Meta.Eqns
public meta import Lean.Meta.Match.MatcherApp
public meta import Lean.Meta.Tactic.Rewrite
public meta import Lean.Meta.Transform

/-!
# Folds from Lean functions

`Fold.of f` turns a Lean function `f` over a tree type into a `Fold` whose certificate the kernel
checks. The tree type `T` must derive `Sig.Inductive` with one sort, and its scalar type `α` must
have one of TorchLean's named `Scalar` instances (`Float`, `Float32` and the configured binary
formats). Types with list fields and mutual blocks are rejected, and their folds are written by
hand with `Fold.select` and `Fold.correct_select`.

## The accepted language

The source `f` has one of the forms `T → α`, `T → Except Error α` and
`Reader α → T → Except Error α`. Up to eta, `f` is a definition `g` applied to fixed leading
arguments, then to the reader in the third form, then to the tree. `g` is either structurally or
well-founded recursive with one equation per constructor, or a definition by `match` on the
constructor without recursion. The leading arguments may mention the variables of the context
in which `Fold.of` is elaborated, as in `def scaledOf (c : Float) := Fold.of (scaled c)`, and
they become constants of the step.

The right-hand side of each equation may use `let`, `if` on one comparison (`<`, `≤`, `>`, `≥`,
`==` on scalars, and also `=` on index fields), `if h : c then .. else ..` whose branches do not
use `h`, `match` on a `Bool`, `+`, `-`, `*` and `/` on scalars, `+`, `-`, `*`, `/` and `%` on
indices, negation, literals, the scalar and index fields of the constructor, and closed terms of
the scalar or index type, which become constants. A reader function reads with `read n i` for a
literal `n` below `256` written in the function, and binds the results of reads and recursive
calls with `←` in `do` notation. `Expr` has no `throw`, so a function may fail only where it
reads an input or binds the result of a child. Children may only be passed to recursive calls
with the same leading arguments. Conditions joined by `∧`, `∨`, `¬`, `&&`, `||` and `!`, `match`
on anything but the constructor or a `Bool`, `throw`, and operations outside `Elab.reify` are
rejected with a message that names the term.

## How the fold is built

For every constructor we rewrite `g` with its equation on fresh variables for the fields.
Scalar and index fields stand for their slots of the row. A recursive call on a child field
becomes a read of input `inputs` at the child's row pointer, where `inputs` is one more than the
largest input number that the function reads. In a pure function we first read the results of
all children that the case uses, then compute the body. `Elab.reify` turns the case into a
kernel `Expr` within a depth of `4096`, and a tactic proof of `Fold.Case` relates the two, so the
step selected by the tag agrees with `f` on every layer. The proof unfolds `Expr.eval` and the
equation, pulls `Except.ok` out of every `if` (`Fold.ite_ok`), and closes the goal by `rfl`, so
that its cost is linear in the size of the case. Only a `match` on a `Bool` needs a case split.
-/

public section

namespace TorchLean.Kernel

/-- Compile a function over a tree type into a certified `Fold`, as in `Fold.of size` for
`size : BTree Float → Float`. The function may also return `Except Error α` and take a
`Reader α` before the tree. -/
scoped syntax (name := foldStx) "Fold.of " term : term

/-- An `if` whose branches both succeed is the successful `if` of their values. The certificate
of `Fold.of` rewrites the evaluation of a conditional step into this form, which is the form of
a Lean `if` inside an arithmetic expression. -/
theorem Fold.ite_ok {ε β : Type} (p : Prop) [Decidable p] (a b : β) :
    (if p then (Except.ok a : Except ε β) else Except.ok b) = Except.ok (if p then a else b) := by
  split <;> rfl

end TorchLean.Kernel

public meta section

namespace TorchLean.Kernel.Elab

open Lean Lean.Meta Lean.Elab
open Lean.Elab.Term hiding mkConst

/-- The largest input number plus one that a function may read. -/
private def maxInputs : Nat := 256

/-- The three accepted shapes of a source function. -/
private inductive FoldForm where
  | pure
  | except
  | reader
  deriving BEq, Inhabited

/-- A recognized source function with the data of its tree type. -/
private structure FoldSource where
  form : FoldForm
  source : Lean.Expr
  tree : Lean.Expr
  scalar : Lean.Expr
  sig : Lean.Expr
  family : Lean.Expr
  inst : Lean.Expr
  fn : Lean.Expr
  leading : Array Lean.Expr
  equations : Array Name
  ctors : Array Derive.CtorShape

/-- The recursive function applied to its leading arguments, the reader if it takes one, and
`t`. -/
private def FoldSource.call (src : FoldSource) (read t : Lean.Expr) : Lean.Expr :=
  mkAppN src.fn (src.leading ++ if src.form == .reader then #[read, t] else #[t])

/-- The reference result of the recursive function on `t`, in `Except Error α`. -/
private def FoldSource.result (src : FoldSource) (read t : Lean.Expr) : MetaM Lean.Expr := do
  let call := src.call read t
  if src.form == .pure then
    mkAppOptM ``Except.ok #[some (mkConst ``Kernel.Error), some src.scalar, some call]
  else
    return call

/-- The constant of an expression-language type. -/
private def tyConst : Ty → Lean.Expr
  | .scalar => mkConst ``Ty.scalar
  | .index => mkConst ``Ty.index
  | .predicate => mkConst ``Ty.predicate

/-- The list literal of expression-language types. -/
private def tyList (types : List Ty) : MetaM Lean.Expr :=
  mkListLit (mkConst ``Ty) (types.map tyConst)

/-- Whether the definition `name` applied to `args` is a `match` on its last argument. -/
private def casesOnLast (name : Name) (levels : List Level) (args : Array Lean.Expr) :
    MetaM Bool := do
  let some info := (← getEnv).find? name | return false
  let some value := info.value? | return false
  if args.isEmpty then return false
  let body := (value.instantiateLevelParams info.levelParams levels).beta args
  let some matcher ← matchMatcherApp? body (alsoCasesOn := true) | return false
  return matcher.discrs.contains args.back!

/-- Unfold `body` until its head is a recursive definition with equation lemmas, or a definition
by cases on its last argument, whose unfolding equation we use. -/
private def recursiveHead (body : Lean.Expr) : TermElabM (Lean.Expr × Array Name) := do
  let mut current := body
  for _ in [0:9] do
    current := current.headBeta
    if let some name := current.getAppFn.constName? then
      if ← isRecursiveDefinition name then
        if let some equations ← getEqnsFor? name then
          return (current, equations)
      else if ← casesOnLast name current.getAppFn.constLevels! current.getAppArgs then
        if let some equation ← getUnfoldEqnFor? name (nonRec := true) then
          return (current, #[equation])
    let some unfolded ← unfoldDefinition? current |
      throwError "Fold.of: we expected a recursive definition or a definition by cases on the \
        tree, got{indentExpr body}"
    current := unfolded
  throwError "Fold.of: we unfolded the function nine times without reaching a recursive \
    definition or a definition by cases on the tree:{indentExpr body}"

/-- Write `a > b` as `b < a` and `a ≥ b` as `b ≤ a`, the comparisons that `reify` knows. -/
private partial def flipComparisons (e : Lean.Expr) : Lean.Expr :=
  e.replace fun e =>
    if e.isAppOfArity ``GT.gt 4 || e.isAppOfArity ``GE.ge 4 then
      let name := if e.isAppOfArity ``GT.gt 4 then ``LT.lt else ``LE.le
      let args := e.getAppArgs
      some (mkApp4 (mkConst name e.getAppFn.constLevels!) args[0]! args[1]!
        (flipComparisons args[3]!) (flipComparisons args[2]!))
    else none

/-- Write a dependent `if` whose branches do not use the proof as an `if`, the form that `reify`
knows. -/
private partial def plainIte (e : Lean.Expr) : Lean.Expr :=
  e.replace fun e =>
    if e.isAppOfArity ``dite 5 then
      let args := e.getAppArgs
      let yes := args[3]!
      let no := args[4]!
      if yes.isLambda && no.isLambda && !yes.bindingBody!.hasLooseBVar 0 &&
          !no.bindingBody!.hasLooseBVar 0 then
        some (mkApp5 (mkConst ``ite e.getAppFn.constLevels!) args[0]! args[1]! args[2]!
          (plainIte (yes.bindingBody!.lowerLooseBVars 1 1))
          (plainIte (no.bindingBody!.lowerLooseBVars 1 1)))
      else none
    else none

/-- Whether the type `type` is a tree type with a derived signature. -/
private def hasSignature (type : Lean.Expr) : TermElabM Bool := do
  let some name := type.getAppFn.constName? | return false
  unless ← isInductive name do return false
  let block ← try Derive.analyze name catch _ => return false
  return (← getEnv).contains (Derive.sigName block)

/-- Recognize the form, the tree type, its signature and scalar type, and the recursive
definition of a source function. -/
private def recognizeFold (f : Lean.Expr) : TermElabM FoldSource := do
  let fType ← instantiateMVars (← inferType f)
  forallTelescopeReducing fType fun xs result => do
    unless xs.size == 1 || xs.size == 2 do
      throwError "Fold.of: we expected a function of a tree, or of a reader and a tree, \
        got{indentExpr fType}"
    let t := xs.back!
    let tree ← instantiateMVars (← inferType t)
    unless ← hasSignature tree do
      if ← xs.anyM fun x => do hasSignature (← inferType x) then
        throwError "Fold.of: the tree must be the last argument of the function, \
          got{indentExpr fType}"
    let some typeName := tree.getAppFn.constName? |
      throwError "Fold.of: the argument type{indentExpr tree}\nis not an inductive type"
    unless ← isInductive typeName do
      throwError "Fold.of: the argument type{indentExpr tree}\nis not an inductive type"
    let block ← Derive.analyze typeName
    unless block.types.size == 1 do
      throwError "Fold.of: {.ofConstName typeName} is part of a mutual block, which we do not \
        support"
    let sigName := Derive.sigName block
    unless (← getEnv).contains sigName do
      throwError "Fold.of: {.ofConstName typeName} has no derived signature. Derive \
        `Sig.Inductive` for it."
    let sig := mkConst sigName
    let family := Lean.mkLambda `s .default (mkApp (mkConst ``Fin)
      (mkApp (mkConst ``Sig.sorts) sig)) tree
    let scalar ← mkFreshExprMVar (mkSort Level.one)
    let inst ← try synthInstance (mkApp3 (mkConst ``Sig.Inductive) sig scalar family) catch _ =>
      throwError "Fold.of: no `Sig.Inductive {sigName}` instance for{indentExpr tree}"
    let scalar ← instantiateMVars scalar
    let exceptType ← mkAppM ``Except #[mkConst ``Kernel.Error, scalar]
    let form ← if xs.size == 2 then
        unless ← isDefEq (← inferType xs[0]!) (← mkAppM ``Reader #[scalar]) do
          throwError "Fold.of: the first of two arguments must be a `Reader {scalar}`, got\
            {indentExpr (← inferType xs[0]!)}"
        pure FoldForm.reader
      else if ← isDefEq result scalar then pure FoldForm.pure
      else pure FoldForm.except
    unless form == .pure || (← isDefEq result exceptType) do
      throwError "Fold.of: the function must return {scalar} or {exceptType}, where {scalar} is \
        the scalar type of {tree}, got{indentExpr result}"
    let lctx := (← getLCtx).setUserName t.fvarId! `t
    let lctx := if form == .reader then lctx.setUserName xs[0]!.fvarId! `read else lctx
    withLCtx lctx (← getLocalInstances) do
    let (body, equations) ← recursiveHead (mkAppN f xs)
    let args := body.getAppArgs
    let fixed := if form == .reader then 2 else 1
    unless args.size ≥ fixed && args.back! == t &&
        (form != .reader || args[args.size - 2]! == xs[0]!) do
      throwError "Fold.of: the tree{if form == .reader then " and the reader" else ""} must be \
        the last argument of the recursive function, got{indentExpr body}"
    let leading := args.extract 0 (args.size - fixed)
    if leading.any (·.hasAnyFVar fun id => xs.any (·.fvarId! == id)) then
      throwError "Fold.of: the leading arguments of the recursive function must not depend on \
        the tree or the reader, got{indentExpr body}"
    if block.ctors.isEmpty then
      throwError "Fold.of: {.ofConstName typeName} has no constructors"
    return { form, source := f, tree, scalar, sig, family, inst, fn := body.getAppFn, leading,
             equations, ctors := block.ctors }

/-- The text of a message of `reify` without its `Program.of` prefix. -/
private def stripPrefix (msg : MessageData) : MetaM String := do
  let s ← msg.toString
  return if s.startsWith "Program.of: " then (s.drop 12).toString else s

/-- The first subterm of `e` that satisfies `p`, outside the subterms equal to one of `stop`. -/
private partial def findOutside (stop : Array Lean.Expr) (p : Lean.Expr → Bool) (e : Lean.Expr) :
    Option Lean.Expr :=
  if stop.contains e then none
  else if p e then some e
  else match e with
    | .app f a => findOutside stop p f <|> findOutside stop p a
    | .lam _ t b _ | .forallE _ t b _ => findOutside stop p t <|> findOutside stop p b
    | .letE _ t v b _ => findOutside stop p t <|> findOutside stop p v <|> findOutside stop p b
    | .mdata _ b | .proj _ _ b => findOutside stop p b
    | _ => none

/-- Whether `e` is a read `R n i` of the reader `R`. -/
private def isRead (R e : Lean.Expr) : Bool :=
  e.isApp && e.getAppFn == R && e.getAppNumArgs == 2

/-- The literal input number of a read `R n i`. -/
private def readNumber? (e : Lean.Expr) : Option Nat :=
  let n := e.appFn!.appArg!
  n.nat? <|> n.rawNatLit?

/-- The input numbers of the reads of `body` with a literal input number, in order. -/
private partial def readNumbers (R body : Lean.Expr) (acc : Array Nat := #[]) : Array Nat :=
  if isRead R body then
    match readNumber? body with
    | some n => acc.push n
    | none => acc
  else match body with
    | .app f a => readNumbers R a (readNumbers R f acc)
    | .lam _ t b _ | .forallE _ t b _ => readNumbers R b (readNumbers R t acc)
    | .letE _ t v b _ => readNumbers R b (readNumbers R v (readNumbers R t acc))
    | .mdata _ b | .proj _ _ b => readNumbers R b acc
    | _ => acc

/-- The heads of conditions that join comparisons. -/
private def joinedConditions : List Name :=
  [``Not, ``And, ``Or, ``Iff, ``and, ``or, ``not, ``bne, ``Bool.and, ``Bool.or, ``Bool.not]

/-- The heads of the terms that throw an error. -/
private def throwers : List Name :=
  [``throw, ``throwThe, ``MonadExcept.throw, ``MonadExceptOf.throw, ``Except.error]

/-- The condition of an `if` that is not one comparison, if `e` is such an `if`. -/
private def joinedCondition? (e : Lean.Expr) : Option Lean.Expr :=
  if e.isAppOfArity ``ite 5 || e.isAppOfArity ``dite 5 then
    let c := e.getArg! 1
    let b := if c.isAppOfArity ``Eq 3 && (c.getArg! 2).isConstOf ``Bool.true then c.getArg! 1
      else c
    if joinedConditions.any b.isAppOf then some c else none
  else none

/-- Run `k` on the fields of constructor `ctor`, the right-hand side of the equation of the
function on it and the name of that equation. Fields written with arrows get readable names. -/
private def withEquation {β : Type} [Inhabited β] (src : FoldSource) (ctor : Derive.CtorShape)
    (R : Lean.Expr) (k : Array Lean.Expr → Lean.Expr → Name → TermElabM β) : TermElabM β := do
  let ctorFn := mkAppN (mkConst ctor.name src.tree.getAppFn.constLevels!) src.tree.getAppArgs
  forallTelescopeReducing (← inferType ctorFn) fun fields _ => do
  unless fields.size == ctor.fields.size do
    throwError "Fold.of: the fields of {.ofConstName ctor.name} differ from its signature"
  let mut lctx ← getLCtx
  for field in fields, i in [0:fields.size] do
    if (lctx.getFVar! field).userName.hasMacroScopes then
      lctx := lctx.setUserName field.fvarId! ((`field).appendIndexAfter i)
  withLCtx lctx (← getLocalInstances) do
  -- Variables stand for the leading arguments, so that an equation rewrites the whole case and
  -- not a call inside a leading argument.
  let leadDecls ← src.leading.mapIdxM fun i a => return ((`lead).appendIndexAfter i, ← inferType a)
  withLocalDeclsDND leadDecls fun leads => do
  let term := mkAppN src.fn (leads ++ if src.form == .reader then #[R, mkAppN ctorFn fields]
    else #[mkAppN ctorFn fields])
  let (body, equation) ← try equationBody term src.equations catch ex =>
    let message ← ex.toMessageData.toString
    unless message.startsWith "Program.of: no constructor equation" do throw ex
    throwError "Fold.of: no equation of {src.fn} covers the whole case\
      {indentExpr (src.call R (mkAppN ctorFn fields))}\nThe function must match on the \
      constructor only, and children may only be passed to recursive calls."
  let body := (← instantiateMVars body).replaceFVars leads src.leading
  -- The unfolding equation of a definition by cases leaves a `match` on the constructor.
  let body ← Meta.transform body fun e => do
    match ← reduceMatcher? e with
    | .reduced e => return .visit e.headBeta
    | _ => return .continue
  k fields body equation

/-- Run `k` on the row context of constructor `ctor` and on the source of its case, ready for
`reify`, together with the equation that unfolds it and the input numbers that the case reads.
Index and scalar fields are their own slots of the row, and child results are read as input
`operand`. -/
private def withCase {β : Type} [Inhabited β] (src : FoldSource) (ctor : Derive.CtorShape)
    (n m operand : Nat) (R : Lean.Expr)
    (k : Context → Lean.Expr → Name → Array Nat → TermElabM β) : TermElabM β := do
  withEquation src ctor R fun fields body equation => do
  let mut slots : Array (Option Lean.Expr) := #[]
  let mut scalars := #[]
  let mut children : Array (Lean.Expr × Nat) := #[]
  for field in fields, shape in ctor.fields do
    match shape with
    | .scalar => scalars := scalars.push field
    | .index => slots := slots.push (some field)
    | .child _ =>
      children := children.push (field, slots.size)
      slots := slots.push none
    | .list _ =>
      throwError "Fold.of: the list field {field} of {.ofConstName ctor.name} is not supported"
  let index := mkConst ``UInt64
  let mut childSlots := #[]
  for (child, _) in children do
    childSlots := childSlots.push ((← child.fvarId!.getUserName).appendAfter "_index", index)
  withLocalDeclsDND childSlots fun childIdx => do
  withLocalDeclsDND ((Array.range (n - slots.size)).map fun i =>
      ((`idx).appendIndexAfter (slots.size + i), index)) fun extraIdx => do
  withLocalDeclsDND ((Array.range (m - scalars.size)).map fun j =>
      ((`sc).appendIndexAfter (scalars.size + j), src.scalar)) fun extraSc => do
  withLocalDeclD `tag index fun tag => do
    let mut idx := #[]
    let mut next := 0
    for slot in slots do
      if let some field := slot then
        idx := idx.push field
      else
        idx := idx.push childIdx[next]!
        next := next + 1
    idx := idx ++ extraIdx
    let ctx : Context := (tag, .index) :: idx.toList.map (·, .index) ++
      (scalars ++ extraSc).toList.map (·, .scalar)
    let name := src.fn.constName!
    let pre := src.leading ++ if src.form == .reader then #[R] else #[]
    let call? (e : Lean.Expr) : Option (Lean.Expr × Nat) :=
      if e.isAppOfArity name (pre.size + 1) && e.getAppArgs.extract 0 pre.size == pre then
        children.find? (·.1 == e.appArg!)
      else none
    let reads := readNumbers R body
    -- The checks skip the leading arguments, which may mention anything closed.
    let check (body : Lean.Expr) : TermElabM Unit := do
      if let some bad := findOutside src.leading (·.getAppFn.isConstOf name) body then
        throwError "Fold.of: in the case {.ofConstName ctor.name}, the recursive call\
          {indentExpr bad}\nmust pass the same leading arguments and a child of the constructor"
      for (child, _) in children do
        if body.containsFVar child.fvarId! then
          let use := (body.find? fun e => e.isApp && e.getAppArgs.contains child).getD body
          throwError "Fold.of: in the case {.ofConstName ctor.name}, the child {child} is used \
            as a value in{indentExpr use}\nChildren may only be passed to recursive calls."
      if let some bad := findOutside src.leading (fun e => isRead R e && (readNumber? e).isNone)
          body then
        throwError "Fold.of: in the case {.ofConstName ctor.name}, the read{indentExpr bad}\n\
          must use a literal input number"
      if let some bad := findOutside src.leading
          (fun e => isRead R e && (readNumber? e).any (· ≥ maxInputs)) body then
        throwError "Fold.of: in the case {.ofConstName ctor.name}, the read{indentExpr bad}\n\
          uses an input number of {maxInputs} or more, which we do not support"
      if let some bad := findOutside src.leading (joinedCondition? · |>.isSome) body then
        throwError "Fold.of: in the case {.ofConstName ctor.name}, the condition\
          {indentExpr (joinedCondition? bad).get!}\nmust be one comparison. Nest the `if`s \
          instead of joining their conditions."
      if let some bad := findOutside src.leading (fun e => throwers.any e.isAppOf) body then
        throwError "Fold.of: in the case {.ofConstName ctor.name}, the function throws\
          {indentExpr bad}\nA fold may fail only where it reads an input or the result of a \
          child."
    let load (slot : Nat) := mkApp2 R (mkNatLit operand) idx[slot]!
    if src.form != .pure then
      let body := body.replace fun e => (call? e).map fun (_, slot) => load slot
      check body
      return ← k ctx (plainIte (flipComparisons body)) equation reads
    let mut names := #[]
    for (child, _) in children do
      names := names.push ((← child.fvarId!.getUserName).appendAfter "_result", src.scalar)
    withLocalDeclsDND names fun ys => do
      let body := body.replace fun e => (call? e).map fun (child, _) =>
        ys[(children.findIdx? (·.1 == child)).getD 0]!
      check body
      let body := plainIte (flipComparisons body)
      let mut term ← mkAppOptM ``Except.ok
        #[some (mkConst ``Kernel.Error), some src.scalar, some body]
      for (y, (_, slot)) in (ys.zip children).reverse do
        if term.containsFVar y.fvarId! then
          term ← mkAppM ``Bind.bind #[load slot, ← mkLambdaFVars #[y] term]
      k ctx term equation reads

/-- The operands of the loads in a reified expression. -/
private partial def loadOperands (e : Lean.Expr) (acc : Array Nat := #[]) : Array Nat :=
  let acc := if e.isAppOfArity ``Kernel.Expr.load 4 then
      match (e.getArg! 2).nat? with
      | some n => acc.push n
      | none => acc
    else acc
  match e with
  | .app f a => loadOperands a (loadOperands f acc)
  | .lam _ t b _ | .forallE _ t b _ => loadOperands b (loadOperands t acc)
  | .letE _ t v b _ => loadOperands b (loadOperands v (loadOperands t acc))
  | .mdata _ b | .proj _ _ b => loadOperands b acc
  | _ => acc

/-- The number of index slots and of scalar slots that a field takes. -/
private def shapeSlots : Derive.Shape → Nat × Nat
  | .scalar => (0, 1)
  | .index | .child _ => (1, 0)
  | .list _ => (2, 0)

/-- Prove `Fold.Case` for constructor number `k`, with shape `ctor`, case `e`, the equation that
unfolds the source on it, the input numbers `reads` that the case reads and the bridge theorems
of `reify`. The proof is a tactic script that the kernel checks. -/
private def proveCase (src : FoldSource) (arith inputs reference : Lean.Expr) (n m k : Nat)
    (ctor : Derive.CtorShape) (e : Lean.Expr) (equation : Name) (reads : Array Nat)
    (bridges : Array Lean.Expr) (h : Lean.Expr) : TermElabM Lean.Expr := do
  let α := src.scalar
  let length ← mkAppM ``List.length #[mkApp (mkConst ``Sig.ctors) src.sig]
  let c := mkApp3 (mkConst ``Fin.mk) length (mkNatLit k) h
  let caseType ← mkAppOptM ``Fold.Case
    #[some src.sig, some α, some src.family, some src.inst, some arith, some inputs,
      some reference, some c, some e]
  let goal ← mkFreshExprSyntheticOpaqueMVar caseType
  let (xs, g) ← goal.mvarId!.introN 7 [`read, `R, `index, `value, `args, `hR, `hM]
  g.withContext do
    let #[read, R, index, value, args, hR, hM] := xs.map mkFVar | unreachable!
    -- The row environment as a literal chain of pushes, the context that `reify` addresses.
    let row : List Ty := .index :: (List.replicate n .index ++ List.replicate m .scalar)
    let tag ← mkAppM ``Nat.toUInt64 #[mkNatLit k]
    let values := tag :: ((List.range n).map fun i => mkApp index (mkNatLit i)) ++
      ((List.range m).map fun j => mkApp value (mkNatLit j))
    let mut env ← mkAppOptM ``Kernel.Env.empty #[some α]
    let mut tail : List Ty := []
    for (t, v) in (row.zip values).reverse do
      env ← mkAppOptM ``Kernel.Env.push
        #[some α, some (← tyList tail), some (tyConst t), some v, some env]
      tail := t :: tail
    let lhs := mkAppN (mkConst ``Kernel.Expr.eval)
      #[α, arith, R, ← tyList row, mkConst ``Ty.scalar, env, e]
    let ctorFn := mkAppN (mkConst ctor.name src.tree.getAppFn.constLevels!) src.tree.getAppArgs
    let mut fields := #[]
    let mut rest := args
    for _ in ctor.fields do
      fields := fields.push (← mkAppM ``Prod.fst #[rest])
      rest ← mkAppM ``Prod.snd #[rest]
    let rhs ← src.result read (mkAppN ctorFn fields)
    let some (eqType, _, _) := (← g.getType).eq? |
      throwError "Fold.of: unexpected obligation{indentExpr (← g.getType)}"
    let g ← g.change (mkApp3 (mkConst ``Eq [Level.one]) eqType lhs rhs)
    -- The facts of `Field.Args.Matches`, one per field, and the user reads that `R` agrees on.
    let mut facts := #[]
    let mut hRest := hM
    let mut i := 0
    let mut j := 0
    for shape in ctor.fields, x in fields do
      let fact ← mkAppM ``And.left #[hRest]
      hRest ← mkAppM ``And.right #[hRest]
      let type ← match shape with
        | .scalar => mkEq (mkApp value (mkNatLit j)) x
        | .index => mkEq (mkApp index (mkNatLit i)) x
        | _ => mkEq (mkApp2 R inputs (mkApp index (mkNatLit i))) (← src.result read x)
      if shape matches .scalar then j := j + 1 else i := i + 1
      unless ← isDefEq (← inferType fact) type do
        throwError "Fold.of: unexpected field fact{indentExpr (← inferType fact)}"
      facts := facts.push (← mkExpectedTypeHint fact type)
    let hRType ← inferType hR
    for op in reads.toList.eraseDups do
      let premise := (← instantiateForall hRType #[mkNatLit op]).bindingDomain!
      facts := facts.push (mkApp2 hR (mkNatLit op) (← mkDecideProof premise))
    let lemmas ← (facts ++ bridges).mapM fun p => do
      `(Parser.Tactic.simpLemma| $(← exprToSyntax p):term)
    let equation := mkIdent equation
    let failed (reason : MessageData) : MessageData :=
      m!"Fold.of: we could not certify the case {.ofConstName ctor.name}: {reason}"
    let proof ← tryCatchRuntimeEx
      (do
        let remaining ← Lean.Elab.Tactic.run g <| Lean.Elab.Tactic.withoutRecover do
          Lean.Elab.Tactic.evalTactic (← `(tactic|
            (simp only [Expr.eval, Env.get_push_zero, Env.get_push_succ, Arithmetic.ofOps,
              Compare.index, IndexOp.eval, Bind.bind, Pure.pure, Except.instMonad, Except.bind,
              Except.pure, BEq.beq, instBEqOfDecidableEq, Bool.cond_decide, Bool.cond_eq_ite,
              decide_eq_true_eq, dite_eq_ite, Fold.ite_ok, $equation:ident, $[$lemmas],*]
             first
             | rfl
             | (repeat' first
                 | rfl
                 | split
                 | split at *
                 | simp_all [UInt64.lt_iff_toNat_lt, UInt64.le_iff_toNat_le]
                 | omega))))
        unless remaining.isEmpty do
          throwError (failed m!"the step and the function differ after simplification. This \
            happens with a comparison or a match outside the accepted language.")
        instantiateMVars goal)
      fun ex => throwError (failed ex.toMessageData)
    if proof.hasSorry || proof.hasSyntheticSorry || proof.hasMVar then
      throwError (failed m!"the proof is incomplete")
    return proof

/-- Elaborate the source function of `Fold.of`, recognize it and build its certified `Fold`. -/
def checkedFold (stx : Syntax) (expectedType? : Option Lean.Expr) : TermElabM Lean.Expr := do
  let outer ← getLCtx
  let f ← elabTerm stx none
  synthesizeSyntheticMVarsNoPostponing
  let f ← instantiateMVars f
  if f.hasMVar then
    throwError "Fold.of: we could not elaborate the source function{indentExpr f}"
  let src ← recognizeFold f
  let α := src.scalar
  let scalar ← try instantiateMVars (← synthInstance (← mkAppM ``Scalar #[α])) catch _ =>
    throwError "Fold.of: the scalar type {α} of {src.tree} has no `Scalar` instance"
  unless [``instScalarFloat32, ``instScalarFloat, ``instScalarConfigured].contains
      (scalar.getAppFn.constName?.getD Name.anonymous) do
    throwError "Fold.of: GPU execution requires TorchLean's named scalar instance, and \
      custom or variable scalar implementations must use CPU execution"
  let arithmeticType ← mkAppM ``Arithmetic #[α]
  let arith ← elabTermEnsuringType (← `(Arithmetic.ofOps)) (some arithmeticType)
  synthesizeSyntheticMVarsNoPostponing
  let arith ← instantiateMVars arith
  let device ← mkAppOptM ``Scalar.arithmetic #[some α, some scalar]
  unless ← isDefEq arith device do
    throwError "Fold.of: we could not show by `rfl` that the device arithmetic of {α} is \
      `Arithmetic.ofOps`, so we cannot certify the fold for the GPU"
  let native ← mkExpectedTypeHint (← mkEqRefl arith) (← mkEq arith device)
  let slots := src.ctors.map fun c => c.fields.foldl
    (fun (a, b) s => (a + (shapeSlots s).1, b + (shapeSlots s).2)) (0, 0)
  let n := slots.foldl (fun a s => max a s.1) 0
  let m := slots.foldl (fun a s => max a s.2) 0
  unless (← isDefEq (mkApp (mkConst ``Sig.indexSlots) src.sig) (mkNatLit n)) &&
      (← isDefEq (mkApp (mkConst ``Sig.scalarSlots) src.sig) (mkNatLit m)) do
    throwError "Fold.of: the slots of {src.sig} differ from its constructors"
  let readerType ← mkAppM ``Reader #[α]
  -- The cases are closed terms, so they leave the scope of the reader `R`.
  let (inputCount, cases) ← withLocalDeclD `read readerType fun R => do
    -- The reads with a literal input number give the number of inputs.
    let mut top := 0
    for ctor in src.ctors do
      let reads ← withEquation src ctor R fun _ body _ => return readNumbers R body
      for op in reads do
        if op < maxInputs then top := max top (op + 1)
    let inputCount := top
    let compile (ctor : Derive.CtorShape) :
        TermElabM (Lean.Expr × Array Lean.Expr × Name × Array Nat) :=
      withCase src ctor n m inputCount R fun ctx term equation reads => do
        let (e, bridges) ← try (reify α R ctx .scalar 4096 term).run #[] catch
          | .error ref msg =>
            throwErrorAt ref "Fold.of: in the case {.ofConstName ctor.name}: \
              {← stripPrefix msg}"
          | ex => throw ex
        let e ← instantiateMVars e
        -- Variables of the context of `Fold.of` are captured constants, variables of the case
        -- are not in the expression language.
        let foreign (id : FVarId) := !outer.contains id
        if let some bad := (e :: bridges.toList).findSome? fun p =>
            p.find? fun v => v.isFVar && foreign v.fvarId! then
          throwError "Fold.of: in the case {.ofConstName ctor.name}, the step depends on \
            {bad}, a variable of the case outside the expression language"
        for op in loadOperands e do
          if op != inputCount && !reads.contains op then
            throwError "Fold.of: in the case {.ofConstName ctor.name}, the function reads \
              input {op} through another definition. Write every read as `read n i` in the \
              function."
        return (e, bridges, equation, reads)
    return (inputCount, ← src.ctors.mapM compile)
  let inputs := mkNatLit inputCount
  let reference ← withLocalDeclD `read readerType fun read =>
    withLocalDeclD `s (mkApp (mkConst ``Fin) (mkApp (mkConst ``Sig.sorts) src.sig)) fun s =>
    withLocalDeclD `t src.tree fun t => do
      let call := src.source.beta (if src.form == .reader then #[read, t] else #[t])
      let body ← if src.form == .pure then
          mkAppOptM ``Except.ok #[some (mkConst ``Kernel.Error), some α, some call]
        else pure call
      mkLambdaFVars #[read, s, t] body
  let stepType ← mkAppM ``Kernel.Expr #[α, ← tyList (.index :: (List.replicate n .index ++
    List.replicate m .scalar)), mkConst ``Ty.scalar]
  let es := cases.toList.map (·.1)
  let length ← mkAppM ``List.length #[mkApp (mkConst ``Sig.ctors) src.sig]
  let mut proof := mkConst ``True.intro
  for (k, ctor, e, bridges, equation, reads) in
      ((List.range cases.size).zip (src.ctors.toList.zip cases.toList)).reverse do
    let caseProof ← withLocalDeclD `h (← mkLt (mkNatLit k) length) fun h => do
      mkLambdaFVars #[h] (← proveCase src arith inputs reference n m k ctor e equation reads
        bridges h)
    proof ← mkAppM ``And.intro #[caseProof, proof]
  let casesType ← mkAppOptM ``Fold.Cases #[some src.sig, some α, some src.family,
    some src.inst, some arith, some inputs, some reference, some (mkNatLit 0),
    some (← mkListLit stepType es)]
  unless ← isDefEq (← inferType proof) casesType do
    throwError "Fold.of: the case proofs do not prove{indentExpr casesType}"
  let hcases ← mkExpectedTypeHint proof casesType
  let tailList ← mkListLit stepType es.tail
  let correct ← mkAppOptM ``Fold.correct_select #[some src.sig, some α, some src.family,
    some src.inst, some arith, some inputs, some reference, some es.head!, some tailList,
    some hcases, some (← mkEqRefl (mkNatLit cases.size))]
  let step ← mkAppM ``Fold.select #[mkNatLit 0, es.head!, tailList]
  let toStep ← mkAppOptM ``Step.mk #[some α, some (mkApp (mkConst ``Sig.indexSlots) src.sig),
    some (mkApp (mkConst ``Sig.scalarSlots) src.sig), some scalar, some arith, some inputs,
    some step, some native]
  let result ← mkAppOptM ``Fold.mk #[some src.sig, some α, some src.family, some src.inst,
    some toStep, some reference, some correct]
  let result ← instantiateMVars result
  if result.hasMVar then
    throwError "Fold.of: the fold has unresolved metavariables"
  if let some expectedType := expectedType? then
    unless ← isDefEq (← inferType result) expectedType do
      throwError "Fold.of: expected {expectedType}, produced {← inferType result}"
  return result

/-- The term elaborator of `Fold.of`, which builds the certified fold of its source function. -/
@[term_elab Kernel.foldStx]
def foldOf : TermElab := fun stx expectedType? => withRef stx do
  let `(Fold.of $source:term) := stx | throwUnsupportedSyntax
  checkedFold source expectedType?

end TorchLean.Kernel.Elab
