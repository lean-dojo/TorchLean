/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.NN.Tactic.Autograd.Scalar

/-!
# Operator derivatives by composition

`op_deriv` proves that a backward map is the adjoint of the Fréchet derivative of a forward map,
in particular that an operator's stored backward is the adjoint of the derivative of its
forward. It works in three steps.

1. It rewrites the goal with the arm equation `Op.rule?_foo` of each operator constructor
   `Op.foo` under `Op.rule?`, with `Option.get_some`, `Function.comp_def`, the real-semantics
   equations registered with `@[op_deriv_simps]` and the lemmas given in brackets. It also
   unfolds the local definitions of the context, such as a `let` introduced with `intro`, and
   finds the operator constructors in their values as well.
2. It reduces the goal to `HasPullbackAt f ?D ?b x` and lets `fun_prop` build the derivative `?D`
   and the pullback `?b` from the `@[fun_prop]` rules of `HasPullbackAt` in
   `TorchLean.NN.FDeriv.Pullback`, `TorchLean.NN.FDeriv.PullbackTensor` and the operator files.
   The scalar derivative inside an entrywise map comes from the hypotheses or from the `OpDeriv`
   rules below: mathlib's `HasDerivAt` lemmas and the derivatives of the activations and pointwise
   operators of the library, tried at instance transparency, so that a rule for another function
   fails at once. If they do not apply, the `AutogradDeriv` rules of `autograd`, which unfold
   definitions, are tried as well. Side conditions come from the hypotheses, also with the two
   sides of a `≠` exchanged, `ne_of_gt`, `ne_of_lt` and `positivity`, first as they are and then
   with the entries of tensor arithmetic in them computed. Since `positivity` gives no sign to a
   product of two unsigned factors, it is tried again with each `a * a` written `a ^ 2`, and then
   also with each sum of a tensor written as the sum of its entries, so that a sum of squares such
   as `Tensor.sum (x * x)` is nonnegative. A
   disjunction is proved by one of its sides, and an inequality of numerals, such as the
   condition `1 ≤ p` of a real power `a ^ p`, by `norm_num`.
   `fun_prop` unifies the rules with the goal at reducible and instance transparency, so that the
   rule of another operator with the same head, such as `Prod.fst`, fails at once instead of
   unfolding both operators. A definition without a rule is unfolded only when it is given in
   brackets, and the error names the definitions applied to the variable that have neither a rule
   nor a scalar derivative lemma.
3. It proves that the backward map in the goal equals the composed pullback. Where both sides
   apply the same function, such as `Linear.inputDeriv W` at two cotangents, it compares the
   arguments. Otherwise it compares components of `WithLp` products, entries of tensors and
   values of functions, rewritten with the equations above, among them `Numeric.tanhDeriv_eq`
   and `Numeric.sigmoidDeriv_eq` for the slopes of `Activation.hasDerivAt_tanh` and
   `Context.hasDerivAt_sigmoid`. An equation of reals left after that is closed
   by `ring` once each pair of atoms with the same head, such as two entries of
   `Linear.inputDeriv W`, is proved equal in the same way. Sums, products and the other
   arithmetic operations are left to `ring` and never compared argument by argument, and an
   equation is tried once: one that is being proved further up or has failed before fails at once.
   An equation of reals still open at the end is tried with `ring_nf`, which also brings the
   arguments of atoms to normal form, such as the exponent `3 / 2 - 1` of a real power.

The goal may be `DifferentiableAt ℝ f x`, `DifferentiableAt ℝ f x ∧ v = (fderiv ℝ f x).adjoint δ`,
the certificate `HasFDerivAt f D x ∧ v = (fderiv ℝ f x).adjoint δ`, the adjoint equation alone,
`HasFDerivAt f D x` or `HasPullbackAt f D b x`. Nothing in the tactic depends on the operator
table, so it proves the same statements for any map composed of functions with
`HasPullbackAt` rules. A user adds the derivative of any other scalar function with
`@[aesop safe apply (rule_sets := [OpDeriv])]` or passes it as a hypothesis.

Mathlib's `fun_prop` does the composition and the `OpDeriv` rules the scalar calculus. SciLean's
`fun_trans` and `data_synth` (T. Skřivan) synthesize reverse derivatives by composition in Lean 4
in a similar way. Here the synthesized pullback is checked against a backward map that is already
written, so the comparison of step 3 is what lets a hand-written backward differ from the
composed one in the order of its products and in the grouping of its sums.
-/

public meta section

namespace TorchLean

open Lean Meta Elab Tactic

attribute [aesop safe 0 apply (rule_sets := [OpDeriv]) (index := [unindexed])
    (transparency := instances)]
  hasDerivAt_id' hasDerivAt_const HasDerivAt.fun_add HasDerivAt.fun_sub HasDerivAt.fun_neg
  HasDerivAt.fun_mul HasDerivAt.fun_div HasDerivAt.fun_pow HasDerivAt.inv HasDerivAt.exp
  HasDerivAt.log HasDerivAt.sqrt HasDerivAt.sin HasDerivAt.cos HasDerivAt.sinh HasDerivAt.cosh
  HasDerivAt.tanh HasDerivAt.arctan HasDerivAt.rpow_const
  Activation.hasDerivAt_relu Activation.hasDerivAt_leakyReLU Activation.hasDerivAt_elu
  Context.hasDerivAt_sigmoid Activation.hasDerivAt_softplus Activation.hasDerivAt_silu
  Activation.hasDerivAt_geluTanh Op.hasDerivAt_geluErf Activation.hasDerivAt_safeLog
  Activation.hasDerivAt_smoothAbs Op.hasDerivAt_clamp Op.hasDerivAt_const_min_of_ne
  Op.hasDerivAt_const_max_of_ne Op.hasDerivAt_min_const_of_ne Op.hasDerivAt_max_const_of_ne
  hasDerivAt_abs

attribute [aesop safe 0 apply (rule_sets := [OpDeriv]) (transparency := reducible)]
  Activation.hasDerivAt_tanh

attribute [op_deriv_simps] Numeric.tanhDeriv_eq Numeric.sigmoidDeriv_eq

attribute [aesop unsafe 40% apply (rule_sets := [OpDeriv]) (index := [unindexed])
    (transparency := instances)]
  HasDerivAt.rpow

attribute [aesop unsafe 50% apply (rule_sets := [OpDeriv])] ne_of_gt ne_of_lt Or.inl Or.inr

add_aesop_rules unsafe 20% tactic (rule_sets := [OpDeriv]) (index := [target (_ : ℝ) ≤ _])
  (by norm_num1 <;> fail)

add_aesop_rules unsafe 30% tactic (rule_sets := [OpDeriv])
  (by first
      | positivity
      | (simp only [← pow_two]; positivity)
      | (simp only [Tensor.mul_apply, Tensor.sum_eq_coord_sum]; (try simp only [← pow_two]);
         positivity))

add_aesop_rules unsafe 20% tactic (rule_sets := [OpDeriv]) (by apply Ne.symm; solve_by_elim)

/-- The arm equations `Op.rule?_foo` of the operator constructors `Op.foo` under `Op.rule?` in
`e`, for the constructors that have one. -/
def opRuleLemmas (e : Expr) : MetaM (Array Name) := do
  let env ← getEnv
  let found ← IO.mkRef (#[] : Array Name)
  forEachExpr e fun sub => do
    if sub.isAppOf ``Op.rule? then
      for arg in sub.getAppArgs do
        if let .const ctor _ := arg.getAppFn then
          if let some (.ctorInfo info) := env.find? ctor then
            if info.induct == ``Op then
              let name := Name.str ctor.getPrefix ("rule?_" ++ ctor.getString!)
              if env.contains name && !(← found.get).contains name then
                found.modify (·.push name)
  found.get

/-- Projections and arithmetic of `WithLp` products and continuous linear maps, which `op_deriv`
uses to compare a backward map with a composed pullback as a whole. -/
def opDerivWholeLemmas : Array Name :=
  #[``WithLp.toLp_fst, ``WithLp.toLp_snd, ``WithLp.add_fst, ``WithLp.add_snd,
    ``WithLp.sub_fst, ``WithLp.sub_snd, ``WithLp.neg_fst, ``WithLp.neg_snd,
    ``WithLp.zero_fst, ``WithLp.zero_snd, ``WithLp.smul_fst, ``WithLp.smul_snd,
    ``add_zero, ``zero_add, ``sub_zero, ``neg_zero, ``smul_zero, ``map_zero, ``map_add,
    ``map_sub, ``map_neg, ``map_smul, ``Tensor.sumCLM_apply, ``Tensor.fullCLM_apply,
    ``ContinuousLinearMap.comp_apply, ``ContinuousLinearMap.coe_id', ``id_eq,
    ``_root_.zero_apply, ``_root_.add_apply, ``_root_.sub_apply, ``neg_apply,
    ``_root_.smul_apply]

/-- Entries of tensor arithmetic, entrywise maps, generated tensors and slope maps, and sums over
all coordinates of a term that is zero but at one coordinate, which `op_deriv` uses to compare a
backward map with a composed pullback entry by entry. -/
def opDerivEntryLemmas : Array Name :=
  #[``Tensor.add_apply, ``Tensor.Rep.hSub_apply, ``Tensor.Rep.neg_apply,
    ``Tensor.Rep.zero_apply, ``Tensor.Rep.smul_apply, ``Tensor.mul_apply, ``Tensor.div_apply,
    ``Tensor.map_apply, ``Tensor.zipWith_apply, ``Tensor.zipWith3_apply,
    ``Tensor.full_apply, ``Tensor.Rep.get_ofFn, ``Op.slopeCLM_apply, ``Finset.sum_ite_eq,
    ``Finset.sum_ite_eq', ``Finset.mem_univ, ``ite_true, ``smul_eq_mul, ``mul_one, ``one_mul,
    ``mul_zero, ``zero_mul]

/-- Close a side condition of a `HasPullbackAt` rule: a hypothesis, also after the components of
`WithLp` pairs in it are reduced, a scalar derivative found by the `OpDeriv` rules and then by the
`AutogradDeriv` rules of `autograd`, or a sign condition, also after the entries of tensor
arithmetic in it are computed and each `a * a` in it is written `a ^ 2`, and then also after each
sum of a tensor in it is written as the sum of its entries. -/
macro "op_deriv_disch" : tactic => `(tactic| first
  | assumption
  | (intros
     try simp only [WithLp.toLp_fst, WithLp.toLp_snd]
     first
     | assumption
     | (apply HasDerivAt.congr_deriv
        · first
          | assumption
          | aesop (config := { terminal := true, enableSimp := false })
              (rule_sets := [$(mkIdent `OpDeriv):ident, -default])
              (erase Aesop.BuiltinRules.not_intro)
          | (simp only [Tensor.map_apply, Tensor.add_apply,
                Tensor.Rep.hSub_apply, Tensor.Rep.neg_apply, Tensor.Rep.smul_apply,
                Tensor.mul_apply, Tensor.div_apply, Tensor.full_apply, smul_eq_mul]
             aesop (config := { terminal := true, enableSimp := false })
              (rule_sets := [$(mkIdent `OpDeriv):ident, -default])
              (erase Aesop.BuiltinRules.not_intro))
          | aesop (config := { terminal := true, enableSimp := false })
              (rule_sets := [$(mkIdent `AutogradDeriv):ident, $(mkIdent `OpDeriv):ident,
                -default])
              (erase Aesop.BuiltinRules.not_intro)
        · first | rfl | ring1_nf)
     | solve_by_elim [ne_of_gt, ne_of_lt, Ne.symm]
     | positivity
     | (simp only [← pow_two]; positivity)
     | (simp only [Tensor.map_apply, Tensor.add_apply,
          Tensor.Rep.hSub_apply, Tensor.Rep.neg_apply, Tensor.Rep.smul_apply, Tensor.mul_apply,
          Tensor.div_apply, Tensor.full_apply, smul_eq_mul]
        first
        | assumption
        | solve_by_elim [ne_of_gt, ne_of_lt, Ne.symm]
        | positivity
        | (simp only [← pow_two]; positivity))
     | (simp only [Tensor.map_apply, Tensor.add_apply,
          Tensor.Rep.hSub_apply, Tensor.Rep.neg_apply, Tensor.Rep.smul_apply, Tensor.mul_apply,
          Tensor.div_apply, Tensor.full_apply, smul_eq_mul, Tensor.sum_eq_coord_sum]
        try simp only [← pow_two]
        positivity)))

open TSyntax.Compat in
/-- The syntax of a `simp only` call with the given lemma names and further arguments, which also
unfolds the local definitions of the context when `zetaDelta` is set. -/
def opDerivSimp (names : Array Name) (extra : Array Syntax) (zetaDelta := false) :
    TacticM Syntax := do
  let lemmas ← names.mapM fun n => do
    pure (← `(Parser.Tactic.simpLemma| $(mkCIdent n):term)).raw
  let args := lemmas ++ extra
  if zetaDelta then `(tactic| simp +zetaDelta only [$args,*]) else `(tactic| simp only [$args,*])

/-- The arithmetic operations, which `op_deriv` compares through entries and `ring` rather than
argument by argument. -/
def opDerivArith : Array Name :=
  #[``HAdd.hAdd, ``HSub.hSub, ``HMul.hMul, ``HDiv.hDiv, ``HSMul.hSMul, ``HPow.hPow, ``Neg.neg,
    ``Inv.inv]

/-- Whether `e` is an equation whose two sides apply the same constant, other than an arithmetic
operation, to the same number of arguments. -/
def sameHeadEq (e : Expr) : MetaM Bool := do
  let some (_, l, r) := (← whnfR (← instantiateMVars e)).eq? | return false
  match l.getAppFn, r.getAppFn with
  | .const a _, .const b _ =>
    return a == b && l.getAppNumArgs == r.getAppNumArgs && !opDerivArith.contains a
  | _, _ => return false

/-- The subterms of the real expression `e` that `ring` takes as atoms, added to `atoms`: those
that are not sums, differences, products, quotients, negations, inverses or natural powers of
other terms, and not numerals. -/
partial def ringAtoms (e : Expr) (atoms : Array Expr := #[]) : Array Expr :=
  if e.isAppOfArity ``HAdd.hAdd 6 || e.isAppOfArity ``HSub.hSub 6 ||
      e.isAppOfArity ``HMul.hMul 6 || e.isAppOfArity ``HDiv.hDiv 6 then
    ringAtoms e.appArg! (ringAtoms e.appFn!.appArg! atoms)
  else if e.isAppOfArity ``Neg.neg 3 || e.isAppOfArity ``Inv.inv 3 then
    ringAtoms e.appArg! atoms
  else if e.isAppOfArity ``HPow.hPow 6 && (e.getArg! 1).isConstOf ``Nat then
    ringAtoms e.appFn!.appArg! atoms
  else if e.isAppOfArity ``OfNat.ofNat 3 || atoms.contains e then atoms
  else atoms.push e

/-- The equations that the comparison of `op_deriv` is proving, so that it does not go round in
a cycle, and those that it has failed to prove, so that it tries each once. -/
structure OpDerivMemo where
  /-- The equations being proved. -/
  active : Std.HashSet Expr := {}
  /-- The equations that could not be proved. -/
  failed : Std.HashSet Expr := {}

/-- Split the main goals, equations between backward maps and composed pullbacks, into
components of `WithLp` products, entries of tensors, values of continuous linear maps and of
functions, then rewrite them with the entry lemmas and the given arguments and close what `rfl` or
`ring` closes. -/
def opDerivEntries (extra : Array Syntax) : TacticM Unit := do
  let simps ← `(Parser.Tactic.simpLemma| $(mkIdent `op_deriv_simps):term)
  let entry ← opDerivSimp (opDerivWholeLemmas ++ opDerivEntryLemmas) (#[simps.raw] ++ extra)
  evalTactic (← `(tactic| repeat' (first
    | refine withLp_prod_ext ?_ ?_
    | refine Tensor.Rep.ext fun _ => ?_
    | refine ContinuousLinearMap.ext fun _ => ?_
    | refine funext fun _ => ?_)))
  evalTactic (← `(tactic| all_goals try $(⟨entry⟩):tactic))
  evalTactic (← `(tactic| all_goals try first | with_reducible rfl | ring1))

mutual

/-- Prove the equation `g` between parts of a backward map and of a composed pullback, or throw
with the state restored: by `opDerivCongr`, otherwise by `opDerivEntries` and `opDerivAtoms` on
each equation left. It fails at once on an equation that is being proved further up or has failed
before, and after `fuel` nested congruences. -/
partial def opDerivEq (extra : Array Syntax) (memo : IO.Ref OpDerivMemo) (fuel : Nat)
    (g : MVarId) : TacticM Unit := g.withContext do
  let type ← instantiateMVars (← g.getType)
  let state ← memo.get
  if fuel == 0 || state.active.contains type || state.failed.contains type then
    throwError "op_deriv: no proof of{indentExpr type}"
  memo.modify fun m => { m with active := m.active.insert type }
  try
    opDerivCongr extra memo fuel g <|> do
      setGoals [g]
      opDerivEntries extra
      for g' in ← getUnsolvedGoals do
        opDerivAtoms extra memo fuel g'
  catch e =>
    memo.modify fun m => { m with failed := m.failed.insert type }
    throw e
  finally
    memo.modify fun m => { m with active := m.active.erase type }

/-- Prove the goal `g`, whose two sides apply the same function to the same number of arguments,
by one congruence step and `opDerivEq` on the equations of the arguments that differ, or throw.
Arguments count as equal when they unify at instance transparency, so that two large terms are
never unfolded to be compared. -/
partial def opDerivCongr (extra : Array Syntax) (memo : IO.Ref OpDerivMemo) (fuel : Nat)
    (g : MVarId) : TacticM Unit := g.withContext do
  unless ← sameHeadEq (← g.getType) do
    throwError "op_deriv: the two sides apply different functions"
  let goals ← g.congrN 1 (closePost := false)
  if goals == [g] then throwError "op_deriv: no congruence applies"
  for g' in goals do
    unless ← (do withReducibleAndInstances g'.refl; pure true) <|> pure false do
      opDerivEq extra memo (fuel - 1) g'

/-- Prove the equation of reals `g` by `opDerivCongr`, or else by `ring` after replacing each atom
of the right side that has an atom of the left side with the same head by that atom, where
`opDerivCongr` proves the two equal. Throw an error when neither closes it. -/
partial def opDerivAtoms (extra : Array Syntax) (memo : IO.Ref OpDerivMemo) (fuel : Nat)
    (g : MVarId) : TacticM Unit := g.withContext do
  if ← (do opDerivCongr extra memo fuel g; pure true) <|> pure false then return
  let some (_, l, r) := (← instantiateMVars (← g.getType)).eq? |
    throwError "op_deriv: the goal is not an equation"
  let lefts := ringAtoms l
  let rights := ringAtoms r
  let mut g := g
  for b in rights do
    if lefts.contains b || !b.getAppFn.isConst then continue
    for a in lefts do
      if rights.contains a || a.getAppFn != b.getAppFn || a.getAppNumArgs != b.getAppNumArgs then
        continue
      let h ← mkFreshExprMVar (← mkEq a b)
      try
        opDerivCongr extra memo fuel h.mvarId!
        let result ← g.rewrite (← g.getType) (← mkEqSymm h)
        g ← g.replaceTargetEq result.eNew result.eqProof
        break
      catch _ => pure ()
  setGoals [g]
  evalTactic (← `(tactic| ring1))

end

/-- The definitions applied to the variable of the map `f` that no `@[fun_prop]` rule of
`HasPullbackAt` names and that, for a name `Pre.foo`, have no scalar derivative lemma
`HasDerivAt.foo` or `Pre.hasDerivAt_foo`, for the error message of `op_deriv`. -/
def opDerivUnruled (f : Expr) : MetaM (Array Name) := do
  let env ← getEnv
  lambdaTelescope (← instantiateMVars f) fun xs body => do
    let found ← IO.mkRef (#[] : Array Name)
    forEachExpr body fun sub => do
      let .const name _ := sub.getAppFn | return
      if sub.getAppNumArgs == 0 || !xs.any (sub.containsFVar ·.fvarId!) then return
      if (← found.get).contains name || (← isInstance name) then return
      if (← getProjectionFnInfo? name).isSome then return
      let some (.defnInfo _) := env.find? name | return
      if let .str pre last := name then
        if env.contains (.str `HasDerivAt last) || env.contains (.str pre ("hasDerivAt_" ++ last))
        then return
      if (← Mathlib.Meta.FunProp.getTheoremsForFunction name ``HasPullbackAt).isEmpty then
        found.modify (·.push name)
    found.get

/-- Build the derivative and the pullback of the goal `HasPullbackAt f ?D ?b x` with `fun_prop`
at reducible and instance transparency, then prove that the backward maps of the remaining goals
equal the pullback, with `ring_nf` as the last try on each equation of reals left. -/
def opDerivCompose (extra : Array Syntax) (target : Expr) : TacticM Unit := do
  try
    evalTactic (← `(tactic| with_reducible_and_instances fun_prop (disch := op_deriv_disch)))
  catch e =>
    let args := target.getAppArgs
    let unruled ← opDerivUnruled args[args.size - 4]!
    let hint := if unruled.isEmpty then m!"" else
      m!"\nNo `HasPullbackAt` rule names {unruled.toList}: pass definitions to unfold in \
        brackets, `op_deriv [{unruled[0]!}]`, or add a rule."
    throwError "op_deriv: no composition of pullback rules proves{indentExpr target}{hint}\n{
      e.toMessageData}"
  let simps ← `(Parser.Tactic.simpLemma| $(mkIdent `op_deriv_simps):term)
  let whole ← opDerivSimp opDerivWholeLemmas (#[simps.raw] ++ extra)
  let memo ← IO.mkRef ({} : OpDerivMemo)
  let mut left := #[]
  for g in ← getUnsolvedGoals do
    if ← g.isAssigned then continue
    setGoals [g]
    evalTactic (← `(tactic| intros))
    evalTactic (← `(tactic| try $(⟨whole⟩):tactic))
    for g' in ← getUnsolvedGoals do
      setGoals [g']
      if ← (do evalTactic (← `(tactic| with_reducible rfl)); pure true) <|> pure false then
        continue
      let fuel := (← instantiateMVars (← g'.getType)).approxDepth.toNat
      try opDerivEq extra memo fuel g'
      catch _ =>
        setGoals [g']
        opDerivEntries extra
        for g'' in ← getUnsolvedGoals do
          setGoals [g'']
          unless ← (do evalTactic (← `(tactic| ring_nf <;> fail)); pure true) <|> pure false do
            left := left.push g''
  setGoals left.toList
  unless left.isEmpty do
    throwError "op_deriv: no proof that the backward map equals the composed pullback; pass \
      lemmas that relate them in brackets. The equations left are{indentD (
      MessageData.joinSep (left.toList.map MessageData.ofGoal) m!"\n")}"

/-- `op_deriv` proves that a backward map is the adjoint of the derivative of a forward map
composed of functions with `HasPullbackAt` rules, such as an operator's stored backward. Lemmas in
brackets are added to the rewriting of the forward map and to the matching of the pullback. -/
syntax (name := opDeriv) "op_deriv" (Parser.Tactic.simpArgs)? : tactic

/-- Run `op_deriv`: rewrite the operator rules and unfold local definitions, build the derivative
and the pullback with `fun_prop`, then match the backward map in the goal with the pullback. -/
@[tactic opDeriv]
def evalOpDeriv : Tactic := fun stx => withMainContext do
  let extra ← match stx[1].getOptional? with
    | some args => Mathlib.Tactic.getSimpArgs args
    | none => pure #[]
  let rules ← opRuleLemmas (← zetaReduce (← instantiateMVars (← getMainTarget)))
  let simps ← `(Parser.Tactic.simpLemma| $(mkIdent `op_deriv_simps):term)
  let expose ← opDerivSimp (rules ++ #[``Option.get_some, ``Function.comp_def])
    (#[simps.raw] ++ extra) (zetaDelta := true)
  evalTactic (← `(tactic| try $(⟨expose⟩):tactic))
  if (← getUnsolvedGoals).isEmpty then return
  evalTactic (← `(tactic| first
    | apply HasPullbackAt.differentiableAt_and
    | apply HasPullbackAt.cert_of_eq
    | apply HasPullbackAt.eq_fderiv_adjoint
    | apply HasPullbackAt.hasFDerivAt
    | (apply HasFDerivAt.differentiableAt; apply HasPullbackAt.hasFDerivAt)
    | skip))
  let target ← whnfR (← instantiateMVars (← getMainTarget))
  unless target.isAppOf ``HasPullbackAt do
    throwError "op_deriv: the goal is not a derivative or pullback statement{indentExpr target}"
  let args := target.getAppArgs
  unless args[args.size - 3]!.isMVar && args[args.size - 2]!.isMVar do
    evalTactic (← `(tactic| apply HasPullbackAt.congr_deriv))
  opDerivCompose extra target

end TorchLean
