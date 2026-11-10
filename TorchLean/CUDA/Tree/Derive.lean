/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Sig
public meta import Lean.Elab.Deriving.Basic

/-!
# Deriving `Sig.Inductive`

We provide a deriving handler for `Sig.Inductive`. With `TorchLean.Kernel` open, writing
`deriving Sig.Inductive` after an inductive type `Foo`, or `deriving instance Sig.Inductive for Foo`
later on, adds three declarations.

* `Foo.sig : Sig` is an abbreviation with one sort. Constructor `i` of `Foo` becomes constructor `i`
  of the signature and keeps its fields in order. A field of the scalar type becomes `.scalar`, a
  field of type `UInt64` becomes `.index`, a field of type `Foo ..` becomes `.child 0` and a field
  of type `List (Foo ..)` becomes `.list 0`.
* `Foo.sigSize` counts the constructors in a tree. It is the size measure of the instance.
* `Foo.instSigInductive` is an instance of `Sig.Inductive Foo.sig α (fun _ => Foo ..)`, where `α`
  is the scalar type.

The scalar type comes from the fields whose type is not `UInt64`, `Foo ..` or `List (Foo ..)`. If
one of them has a parameter of `Foo` as its type, the scalar type is that parameter. Otherwise it is
the type of the first of them, and it is `Unit` when there are none. Every other such field must
have the scalar type, and the handler reports the first field that does not. Recursive fields must
apply `Foo` to its own parameters in order. Other shapes, such as `Option Foo`, `Array Foo` or a
field whose type depends on earlier fields, are rejected with an error that names the field.

The handler also accepts a mutual block `A`, `B`, ... of such types. It then makes one signature
`A.sig`, named after the first type of the block, with one sort per type in the order of the block.
The constructors of all types are numbered in that order, and a field of type `B ..` or
`List (B ..)` refers to the sort of `B`. The family of types is the abbreviation `A.SigFamily`, so
the instance `A.instSigInductive` has type `Sig.Inductive A.sig α A.SigFamily`. The family looks up
the type of a sort in the list `[A .., B .., ...]`. Each type `B` of the block gets its own size
function `B.sigSize`.

Unification cannot find the signature from a tree or a sort, so we name it in both directions. For
one type we write `Arena.encode (σ := Foo.sig) (T := fun _ => Foo ..) (s := 0) t` to encode and
`a.decode (σ := Foo.sig) (fun _ => Foo ..) 0` to decode. For a mutual block we write
`Arena.encode (σ := A.sig) (T := A.SigFamily) (s := j) t` and
`a.decode (σ := A.sig) A.SigFamily j`.

A universe polymorphic type gets its instance at `Type`, which is where `Sig.Inductive` lives. When
`Foo` is private, `Foo.sig` is public and the other declarations are private. The compiler
checks in a module that the generated code has the same types in the public view, and these types
unfold `Foo.sig`. In a module we may also write `deriving @[expose] Sig.Inductive`.

Indexed families and types without constructors are not supported. Setting the option
`trace.Elab.Deriving` shows the commands that the handler elaborates.
-/

public meta section

namespace TorchLean.Kernel.Derive

open Lean Meta Elab Term Command

/-- The shape of a constructor field, which fixes the `Field` that stores it. -/
inductive Shape where
  /-- A field of the scalar type, stored as `Field.scalar`. -/
  | scalar
  /-- A field of type `UInt64`, stored as `Field.index`. -/
  | index
  /-- A field holding one tree of the given sort, stored as `Field.child`. -/
  | child (sort : Nat)
  /-- A field holding a list of trees of the given sort, stored as `Field.list`. -/
  | list (sort : Nat)

/-- A constructor of a block, with the sort that it builds and the shapes of its fields. -/
structure CtorShape where
  /-- The full name of the constructor. -/
  name : Name
  /-- The position of the type of the constructor in its block. -/
  sort : Nat
  /-- The shapes of the fields, in order. -/
  fields : Array Shape

/-- A block of inductive types whose fields have all been checked. -/
structure Block where
  /-- The types of the block, in order. -/
  types : Array Name
  /-- The names of the parameters, as used in the generated commands. -/
  params : Array Name
  /-- The scalar type, as a term over the parameters. -/
  scalar : Term
  /-- The constructors of all types of the block, in order. -/
  ctors : Array CtorShape

/-- The sort of `e` if `e` is a type of the block applied to the parameters `ps`. -/
def sortOf? (types : Array Name) (ps : Array Expr) (e : Expr) : Option Nat :=
  match e.getAppFn with
  | .const n _ => if e.getAppArgs == ps then types.findIdx? (· == n) else none
  | _ => none

/-- The shape of a field of type `ty`, or `none` when the field has to hold a scalar. -/
def shapeOf? (types : Array Name) (ps : Array Expr) (ty : Expr) : Option Shape :=
  if let some s := sortOf? types ps ty then some (.child s)
  else if ty.isAppOfArity ``List 1 then (sortOf? types ps ty.appArg!).map .list
  else if ty.isConstOf ``UInt64 then some .index
  else none

/-- Check the block of `declName`, compute the shapes of all fields and find the scalar type. -/
def analyze (declName : Name) : TermElabM Block := do
  let indVal ← getConstInfoInduct declName
  let pre := m!"cannot derive `Sig.Inductive` for `{.ofConstName declName}`"
  let types := indVal.all.toArray
  let allowed := if let #[t] := types then
      m!"Every field must have the scalar type, `UInt64`, `{.ofConstName t} ..` or \
        `List ({.ofConstName t} ..)`, where `..` are the parameters of `{.ofConstName t}` in order."
    else
      m!"Every field must have the scalar type, `UInt64`, `T ..` or `List (T ..)` for a type `T` \
        of the block {.andList (types.toList.map fun t => m!"`{.ofConstName t}`")}, where `..` are \
        the parameters in order."
  let us := indVal.levelParams.map fun _ => Level.zero
  for t in types do
    let v ← getConstInfoInduct t
    unless v.numIndices == 0 do
      throwError "{pre}: `{.ofConstName t}` is an indexed family"
    if v.ctors.isEmpty then
      throwError "{pre}: `{.ofConstName t}` has no constructors"
  let indType := indVal.type.instantiateLevelParams indVal.levelParams us
  forallBoundedTelescope indType indVal.numParams fun ps _ => do
    let mut ctors := #[]
    let mut scalars : Array (Expr × MessageData) := #[]
    let mut sort := 0
    for t in types do
      for c in (← getConstInfoInduct t).ctors do
        let cv ← getConstInfoCtor c
        let cty ← instantiateForall (cv.type.instantiateLevelParams cv.levelParams us) ps
        let (fields, found) ← forallBoundedTelescope cty cv.numFields fun xs _ => do
          let mut fields := #[]
          let mut found := #[]
          for x in xs do
            let ty := (← inferType x).cleanupAnnotations
            let what := m!"field `{← x.fvarId!.getUserName}` of constructor `{.ofConstName c}`"
            if ty.hasAnyFVar fun v => xs.contains (.fvar v) then
              throwError "{pre}: the type of {what} depends on earlier fields"
            if let some sh := shapeOf? types ps ty then
              fields := fields.push sh
            else if (ty.find? fun e => e.isConst && types.contains e.constName!).isSome then
              throwError "{pre}: {what} has type{indentExpr ty}\nwhich is not supported. {allowed}"
            else if !(← isDefEq (← inferType ty) (mkSort Level.one)) then
              throwError "{pre}: {what} has type{indentExpr ty}\nwhich is not in `Type`"
            else
              fields := fields.push .scalar
              found := found.push (ty, what)
          return (fields, found)
        ctors := ctors.push { name := c, sort, fields }
        scalars := scalars ++ found
      sort := sort + 1
    let pick := (scalars.find? fun (ty, _) => ps.contains ty) <|> scalars[0]?
    let scalar ← match pick with
      | none => pure (mkConst ``Unit)
      | some (α, src) => do
        for (ty, what) in scalars do
          unless (← isDefEq ty α) do
            let hint :=
              if ty.isConstOf ``Nat then m!" Index fields must have type `UInt64`." else m!""
            throwError "{pre}: {what} has type{indentExpr ty}\nbut the scalar type is\
              {indentExpr α}\nas set by {src}. {allowed}{hint}"
        pure α
    let mut params := #[]
    for p in ps do
      params := params.push (← p.fvarId!.getUserName)
    let scalarStx : Term ← match ps.findIdx? (· == scalar), scalar with
      | some i, _ => pure (mkIdent params[i]!)
      | none, .const n _ => pure (mkCIdent n)
      | none, e => withOptions (·.setBool `pp.sanitizeNames false) (PrettyPrinter.delab e)
    return { types, params, scalar := scalarStx, ctors }

/-- A doc comment holding `text`. -/
def mkDoc (text : String) : TSyntax ``Parser.Command.docComment :=
  ⟨mkNode ``Parser.Command.docComment #[mkAtom "/--", mkAtom (text ++ " -/")]⟩

/-- An identifier that declares exactly the full name `n`, whatever the current namespace is. We
drop the prefix of a private name here, since the modifier from `visOf` adds it back. -/
def rootIdent (n : Name) : Ident :=
  mkIdent (`_root_ ++ privateToUserName n)

/-- The modifier `private` if the name `n` is private, so that the generated declaration `n` is
private exactly when the type that it is named after is private. -/
def visOf (n : Name) : Option (TSyntax ``Parser.Command.visibility) :=
  if isPrivateName n then some ⟨mkNode ``Parser.Command.private #[mkAtom "private"]⟩ else none

/-- Implicit binders `{p}` for the parameters of the block. Unification finds their types. -/
def mkBinders (b : Block) : TermElabM (Array (TSyntax ``Parser.Term.bracketedBinder)) :=
  b.params.mapM fun n => do
    let stx ← `(Lean.Parser.Term.bracketedBinderF| {$(mkIdent n)})
    return ⟨stx.raw⟩

/-- The sort `j` as a term. A block with one type uses a numeral, a mutual block uses `Fin.mk`. -/
def sortTerm (b : Block) (j : Nat) : TermElabM Term :=
  if b.types.size == 1 then `($(quote j)) else `(⟨$(quote j), by decide⟩)

/-- The `Field` that stores a field of shape `sh`. -/
def fieldTerm (b : Block) (sh : Shape) : TermElabM Term :=
  match sh with
  | .scalar => `(.scalar)
  | .index => `(.index)
  | .child s => do `(.child $(← sortTerm b s))
  | .list s => do `(.list $(← sortTerm b s))

/-- Fresh variables for the fields of constructor `c`. -/
def fieldVars (c : CtorShape) : TermElabM (Array Ident) :=
  c.fields.mapM fun _ => return mkIdent (← mkFreshUserName `a)

/-- The term `@C _ .. _ a₁ .. aₙ`, which serves both as a pattern and as an application. -/
def ctorApp (b : Block) (c : CtorShape) (vars : Array Ident) : TermElabM Term := do
  let holes ← b.params.mapM fun _ => `(_)
  `(@$(mkCIdent c.name) $holes* $vars*)

/-- The nested pair `(a₁, .., aₙ, ())` that holds the fields of a layer. -/
def tuple (vars : Array Ident) : TermElabM Term := do
  vars.foldrM (fun v acc => `(($v, $acc))) (← `(()))

/-- The match alternative `| p₁, .., pₙ => rhs`. -/
def mkAlt (pats : Array Term) (rhs : Term) : TermElabM (TSyntax ``Parser.Term.matchAlt) :=
  `(Lean.Parser.Term.matchAltExpr| | $pats,* => $rhs)

/-- The name of the size function of the type `t`. -/
def sizeName (t : Name) : Name := t ++ `sigSize

/-- The name of the signature of the block. It never has a private prefix, since the signature is
always public. -/
def sigName (b : Block) : Name := privateToUserName b.types[0]! ++ `sig

/-- The command that defines the signature of the block. The signature is public even when the
types are private, because the compiler checks in a module that other modules see the same types
for the generated code, and these types unfold the signature. -/
def mkSigCmd (b : Block) : TermElabM Command := do
  let base := b.types[0]!
  let ctors ← b.ctors.mapM fun c => do
    let fs ← c.fields.mapM (fieldTerm b)
    `(⟨$(← sortTerm b c.sort), [$fs,*]⟩)
  let doc := s!"The signature of `{privateToUserName base}`, made by `deriving Sig.Inductive`."
  `(command|
    $(mkDoc doc):docComment $(⟨mkNode ``Parser.Command.public #[mkAtom "public"]⟩):visibility
    abbrev $(rootIdent (sigName b)) : $(mkCIdent ``Sig) :=
      { sorts := $(quote b.types.size), ctors := [$ctors,*] })

/-- The commands that define the size functions of the types of the block. -/
def mkSizeCmds (b : Block) : TermElabM (Array Command) := do
  let binders ← mkBinders b
  let ps := b.params.map mkIdent
  -- A recursive call refers to the function that is being defined by its name without the
  -- private prefix, since that is the name under which the elaborator finds it.
  let recIdent (s : Nat) : Ident := mkIdent (privateToUserName (sizeName b.types[s]!))
  let mut defs := #[]
  let mut sort := 0
  for ty in b.types do
    let mut alts := #[]
    for c in b.ctors do
      if c.sort == sort then
        let vars ← fieldVars c
        let mut body ← `(1)
        for v in vars, sh in c.fields do
          match sh with
          | .child s => body ← `($body + $(recIdent s) $v)
          | .list s => body ← `($body + (List.map (fun x => $(recIdent s) x) $v).sum)
          | _ => pure ()
        alts := alts.push (← mkAlt #[← ctorApp b c vars] body)
    let doc := s!"The number of constructors in a tree of type `{privateToUserName ty}`, made by \
      `deriving Sig.Inductive`."
    defs := defs.push (← `(command|
      $(mkDoc doc):docComment $[$(visOf (sizeName ty)):visibility]?
      def $(rootIdent (sizeName ty)) $binders:bracketedBinder* (t : @$(mkCIdent ty) $ps*) : Nat :=
        match t with $alts:matchAlt*))
    sort := sort + 1
  if b.types.size == 1 then return defs
  return #[← `(mutual $defs* end)]

/-- The tactic that rewrites the children of a layer and the size of a tree into sums. -/
def mkSimpTac (b : Block) : TermElabM (TSyntax `tactic) := do
  let names := #[``Layer.children, ``Field.Args.children, sigName b, ``Fin.getElem_fin,
    ``List.getElem_cons_succ, ``List.getElem_cons_zero, ``List.map_cons, ``List.map_nil,
    ``List.map_append, ``List.map_map, ``List.sum_cons, ``List.sum_nil, ``List.sum_append,
    ``Function.comp_def] ++ b.types.map sizeName ++
    (if b.types.size == 1 then #[] else #[b.types[0]! ++ `SigFamily])
  let lemmas ← names.mapM fun n => `(Parser.Tactic.simpLemma| $(mkCIdent n):term)
  `(tactic| simp only [$lemmas,*])

/-- The instance command for a block with a single type. -/
def mkInstCmd (b : Block) : TermElabM Command := do
  let ty := b.types[0]!
  let binders ← mkBinders b
  let ps := b.params.map mkIdent
  let mut viewAlts := #[]
  let mut mkAlts := #[]
  let mut i := 0
  for c in b.ctors do
    let vars ← fieldVars c
    let app ← ctorApp b c vars
    let tup ← tuple vars
    viewAlts := viewAlts.push
      (← mkAlt #[app] (← `(⟨⟨$(quote i), by decide⟩, Subsingleton.elim _ _, $tup⟩)))
    mkAlts := mkAlts.push (← mkAlt #[← `(⟨⟨$(quote i), _⟩, _, $tup⟩)] app)
    i := i + 1
  let simpTac ← mkSimpTac b
  let doc := s!"The `Sig.Inductive` instance of `{privateToUserName ty}`, made by \
    `deriving Sig.Inductive`."
  `(command|
    $(mkDoc doc):docComment $[$(visOf (ty ++ `instSigInductive)):visibility]?
    instance $(rootIdent (ty ++ `instSigInductive)):ident $binders:bracketedBinder* :
        $(mkCIdent ``Sig.Inductive) $(mkCIdent (sigName b)) $(b.scalar)
          (fun _ => @$(mkCIdent ty) $ps*) where
      view t := match t with $viewAlts:matchAlt*
      mk l := match l with $mkAlts:matchAlt*
      mk_view t := by cases t <;> rfl
      size t := $(mkCIdent (sizeName ty)) t
      size_lt t := by cases t <;> $simpTac <;> omega)

/-- The commands for the family of types and the instance of a mutual block. -/
def mkMutualCmds (b : Block) : TermElabM (Array Command) := do
  let base := b.types[0]!
  let binders ← mkBinders b
  let ps := b.params.map mkIdent
  let fam := base ++ `SigFamily
  let mut mvAlts := #[]
  let mut sizeAlts := #[]
  let mut ltAlts := #[]
  let simpTac ← mkSimpTac b
  let mut j := 0
  for ty in b.types do
    let pats := #[← `(⟨$(quote j), _⟩), ← `(t)]
    mvAlts := mvAlts.push (← mkAlt pats (← `(by cases t <;> rfl)))
    sizeAlts := sizeAlts.push (← mkAlt pats (← `($(mkCIdent (sizeName ty)) t)))
    ltAlts := ltAlts.push (← mkAlt pats (← `(by cases t <;> $simpTac <;> omega)))
    j := j + 1
  let mut viewAlts := #[]
  let mut mkAlts := #[]
  let mut i := 0
  for c in b.ctors do
    let vars ← fieldVars c
    let app ← ctorApp b c vars
    let tup ← tuple vars
    let rhs ← `(⟨⟨$(quote i), by decide⟩, rfl, $tup⟩)
    viewAlts := viewAlts.push (← mkAlt #[← `(⟨$(quote c.sort), _⟩), app] rhs)
    mkAlts := mkAlts.push (← mkAlt #[← `(_), ← `(⟨⟨$(quote i), _⟩, rfl, $tup⟩)] app)
    i := i + 1
  let tys ← b.types.mapM fun ty => `(@$(mkCIdent ty) $ps*)
  let user := privateToUserName base
  let famDoc := s!"The types of the block of `{user}`, indexed by the sorts of `{user}.sig`."
  let famCmd ← `(command|
    $(mkDoc famDoc):docComment $[$(visOf fam):visibility]?
    abbrev $(rootIdent fam) $binders:bracketedBinder* :=
      fun (s : Fin $(quote b.types.size)) => ([$tys,*][s.val]'s.isLt : Type))
  let instDoc := s!"The `Sig.Inductive` instance of the block of `{user}`, made by \
    `deriving Sig.Inductive`."
  let instCmd ← `(command|
    $(mkDoc instDoc):docComment $[$(visOf (base ++ `instSigInductive)):visibility]?
    instance $(rootIdent (base ++ `instSigInductive)):ident $binders:bracketedBinder* :
        $(mkCIdent ``Sig.Inductive) $(mkCIdent (sigName b)) $(b.scalar)
          (@$(mkCIdent fam) $ps*) where
      view {s} t := match s, t with $viewAlts:matchAlt*
      mk {s} l := match s, l with $mkAlts:matchAlt*
      mk_view {s} t := match s, t with $mvAlts:matchAlt*
      size {s} t := match s, t with $sizeAlts:matchAlt*
      size_lt {s} t := match s, t with $ltAlts:matchAlt*)
  return #[famCmd, instCmd]

/-- The commands that `deriving Sig.Inductive` elaborates for the block of `declName`. -/
def mkCmds (declName : Name) : TermElabM (Array Command) := do
  let b ← analyze declName
  let sig ← mkSigCmd b
  let sizes ← mkSizeCmds b
  if b.types.size == 1 then
    return #[sig] ++ sizes ++ #[← mkInstCmd b]
  let rest ← mkMutualCmds b
  return #[sig, rest[0]!] ++ sizes ++ #[rest[1]!]

/-- The deriving handler for `Sig.Inductive`. It treats each mutual block among `declNames` once. -/
def mkSigInductiveInstance (declNames : Array Name) : CommandElabM Bool := do
  let mut seen : Array Name := #[]
  for declName in declNames do
    unless seen.contains declName do
      let indVal ← getConstInfoInduct declName
      seen := seen ++ indVal.all.toArray
      let cmds ← liftTermElabM (mkCmds declName)
      trace[Elab.Deriving] "{cmds}"
      withEnableInfoTree false do
        elabCommand (mkNullNode cmds)
  return true

initialize registerDerivingHandler ``Sig.Inductive mkSigInductiveInstance

end TorchLean.Kernel.Derive

end
