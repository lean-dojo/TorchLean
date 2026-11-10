/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.CUDA.Tree.Expand

/-!
# Hylomorphisms on arenas

A `Hylo` is a recursive function whose calls need not follow a value: a call with the slots of a
row makes the calls of its `Expansion`, and combines their results by the expression of its
`Step`. It runs in two phases. `Expansion.expand` unfolds the calls from the roots, one level per
depth of the recursion, and tags every row with the position of its first call in the next level.
The fold of `Step` then computes the levels from the deepest up, and a row reads the result of its
call `j` as input `inputs` at its tag plus `j`. This is a hylomorphism (Meijer, Fokkinga and
Paterson 1991), an unfold followed by a fold, with the call tree stored as the levels of an arena.

`Hylo.correct` is a certificate about one call: if input `inputs` holds the reference results of
the calls from position `first` on, the step gives the reference result of the call.
`Hylo.foldr_expand` lifts it to the roots: the fold over the expanded levels gives the reference
result of every root, failures included. The expansion gives `none` when the levels would hold more
than `capacity` rows or a number of calls or an argument fails, and `Hylo.results` then evaluates
the reference, so `Hylo.results_eq` holds for every capacity. `Hylo.schedule_eq` states that a
successful level-by-level schedule gives the reference results as successes. The certificate is
about partial correctness: when the expansion finishes, its results are the reference results.
It does not show that the expansion finishes. A hylo whose calls do not shrink, such as a call
that calls itself with its own slots, has a certificate and expands past every capacity, and its
`Hylo.results` are then those of the reference, computed directly.

We usually write a recursive function by cases: guards on the slots of a call, and for each case
the calls it makes and the way it combines their results. A `Hylo.Branch` is one case, a list of
`Hylo.Call`s and a combining expression. From guarded branches and a last branch,
`Hylo.callsOf`, `Hylo.argIndexOf`, `Hylo.argValueOf` and `Hylo.stepOf` build the expressions of a
`Hylo`: a chain of conditions on the guards, and in a branch the choice of a call by its position
`j` (`Fold.select`). `Hylo.correct_branches` reduces the certificate to `Hylo.Branches.Correct`,
one obligation per branch under its guard and the negations of the guards before it, in which the
results of the calls are those of the reference. A guard may only read inputs below `inputs`
(`Expr.readsBelow`), so the step and the expansion see the same value of it.

On the device, `Hylo.runDevice` builds the levels with `Expansion.expandBlocks` and folds them
with `Step.runBlocks`, without copying the levels back to the host, and packs the inputs once for
both. The device computes with the arithmetic of the precision of the `Scalar` instance, and that
its `arithmetic` field is this arithmetic is the assumption of every kernel of `Kernel.Launch`. It
holds for the instances of the library. `Hylo.run` uses the device when LibTorch and a usable
device are present, and `Hylo.results` on the host otherwise or when the device gives up.
-/

@[expose] public section

namespace TorchLean.Kernel

open CUDA

variable {α : Type}

/-- Whether every load of `e` reads an input below `n`. -/
def Expr.readsBelow {Γ : List Ty} {t : Ty} (n : Nat) : Expr α Γ t → Bool
  | .scalar _ | .index _ | .predicate _ | .var _ => true
  | .load operand i => decide (operand < n) && i.readsBelow n
  | .binary _ x y => x.readsBelow n && y.readsBelow n
  | .neg x => x.readsBelow n
  | .indexBinary _ x y => x.readsBelow n && y.readsBelow n
  | .compare _ x y => x.readsBelow n && y.readsBelow n
  | .indexCompare _ x y => x.readsBelow n && y.readsBelow n
  | .cond p yes no => p.readsBelow n && yes.readsBelow n && no.readsBelow n
  | .letIn value body => value.readsBelow n && body.readsBelow n
  | .fold count initial body => count.readsBelow n && initial.readsBelow n && body.readsBelow n

/-- An expression whose loads read inputs below `n` has the same value under readers that agree
below `n`. -/
theorem Expr.eval_eq_of_readsBelow {Γ : List Ty} {t : Ty} {n : Nat} (arithmetic : Arithmetic α)
    {read R : Reader α} (hR : ∀ op < n, R op = read op) (e : Expr α Γ t) :
    e.readsBelow n → ∀ env : Env α Γ, e.eval arithmetic R env = e.eval arithmetic read env := by
  induction e with
  | scalar | index | predicate | var => intro _ _; rfl
  | load operand i ih =>
    intro h env
    simp only [readsBelow, Bool.and_eq_true, decide_eq_true_eq] at h
    simp only [Expr.eval, ih h.2 env, hR operand h.1]
  | binary op x y hx hy =>
    intro h env
    simp only [readsBelow, Bool.and_eq_true] at h
    simp only [Expr.eval, hx h.1 env, hy h.2 env]
  | neg x hx =>
    intro h env
    simp only [readsBelow] at h
    simp only [Expr.eval, hx h env]
  | indexBinary op x y hx hy =>
    intro h env
    simp only [readsBelow, Bool.and_eq_true] at h
    simp only [Expr.eval, hx h.1 env, hy h.2 env]
  | compare op x y hx hy =>
    intro h env
    simp only [readsBelow, Bool.and_eq_true] at h
    simp only [Expr.eval, hx h.1 env, hy h.2 env]
  | indexCompare op x y hx hy =>
    intro h env
    simp only [readsBelow, Bool.and_eq_true] at h
    simp only [Expr.eval, hx h.1 env, hy h.2 env]
  | cond p yes no hp hy hn =>
    intro h env
    simp only [readsBelow, Bool.and_eq_true] at h
    simp only [Expr.eval, hp h.1.1 env, hy h.1.2 env, hn h.2 env]
  | letIn value body hv hb =>
    intro h env
    simp only [readsBelow, Bool.and_eq_true] at h
    simp only [Expr.eval, hv h.1 env, hb h.2]
  | fold count initial body hc hi hb =>
    intro h env
    simp only [readsBelow, Bool.and_eq_true] at h
    simp only [Expr.eval, hc h.1.1 env, hi h.1.2 env, hb h.2]

/-- A hylomorphism over rows with `m` index slots and `s` scalar slots: a recursive function of the
slots of a call. A call makes `calls` calls, call `j` with the slots `argIndex` and `argValue` at
`j`, and the step combines their results, reading the result of call `j` as input `inputs` at the
position of the first call, the tag of the row, plus `j`. -/
structure Hylo (α : Type) (m s : Nat) extends Expansion α m s, Step α m s where
  /-- The reference function: the result of a call with the given slots. -/
  reference : Reader α → Env α (Slots m s) → Except Error α
  /-- The step gives the reference result of a call when its inputs before `inputs` agree with
  the reader and input `inputs` holds the reference results of its calls from position `first`
  on. -/
  correct : ∀ (read R : Reader α) (index : Nat → UInt64) (value : Nat → α) (first n : Nat),
    (∀ op < inputs, R op = read op) →
    calls.eval arithmetic read (Slots.env index value) = .ok n.toUInt64 →
    first + n < UInt64.size →
    (∀ j < n, ∃ (index' : Nat → UInt64) (value' : Nat → α),
      (∀ i : Fin m,
        (argIndex i).eval arithmetic read (Row.env j.toUInt64 index value) = .ok (index' i)) ∧
      (∀ i : Fin s,
        (argValue i).eval arithmetic read (Row.env j.toUInt64 index value) = .ok (value' i)) ∧
      R inputs (first + j).toUInt64 = reference read (Slots.env index' value')) →
    expr.eval arithmetic R (Row.env first.toUInt64 index value) =
      reference read (Slots.env index value)

/-- A call that a branch of a hylomorphism makes: its index and scalar slots, from the slots of the
caller. -/
structure Hylo.Call (α : Type) (m s : Nat) where
  /-- Index slot `i` of the call. -/
  index : Fin m → Expr α (Slots m s) .index
  /-- Scalar slot `i` of the call. -/
  value : Fin s → Expr α (Slots m s) .scalar

/-- A branch of a hylomorphism: the calls that a call makes, in order, and the step that combines
their results, reading the result of call `j` as input `inputs` at the tag plus `j`. -/
structure Hylo.Branch (α : Type) (m s : Nat) where
  /-- The calls of the branch. -/
  calls : List (Hylo.Call α m s)
  /-- The step on the row of the caller. -/
  combine : Expr α (Row m s) .scalar

namespace Hylo

variable {m s : Nat} [Inhabited α]

/-- The fold over the levels of an expansion from a frontier gives the reference result of every
row of the frontier. -/
theorem foldr_expandAux (H : Hylo α m s) (read : Reader α) :
    ∀ (fuel budget : Nat) (frontier : Arena.Level α) {levels : List (Arena.Level α)},
      H.expandAux read fuel budget frontier = some levels →
      levels.foldr (H.level read) #[] =
        Array.ofFn fun k : Fin frontier.tag.size => H.reference read (frontier.slots m s k)
  | 0, _, _, _, h => by simp [Expansion.expandAux] at h
  | fuel + 1, budget, frontier, levels, h => by
    unfold Expansion.expandAux at h
    by_cases h0 : frontier.tag.size = 0
    · simp only [h0, ite_true, Option.some.injEq] at h
      subst h
      apply Array.ext
      · simp [h0]
      · intro i hi _
        simp at hi
    simp only [h0, ite_false] at h
    cases hr : H.round read budget frontier with
    | none => rw [hr] at h; cases h
    | some p =>
      obtain ⟨level, next⟩ := p
      rw [hr] at h
      obtain ⟨rest, hrest, rfl⟩ := Option.map_eq_some_iff.1 h
      rw [List.foldr_cons, foldr_expandAux H read fuel _ next hrest]
      obtain ⟨c, htag, hindex, hscalar, hsize, htotal, hcalls, hargs⟩ := Expansion.round_spec hr
      apply Array.ext
      · simp [htag]
      · intro k hk1 hk2
        have hk : k < frontier.tag.size := by simpa using hk2
        rw [Step.getElem_level, Array.getElem_ofFn]
        have hle : ((List.range k).map c).sum + c k ≤
            ((List.range frontier.tag.size).map c).sum := by
          have := Expansion.sum_range_le c (show k + 1 ≤ frontier.tag.size by omega)
          simpa [List.range_succ] using this
        have henv : level.env m s k = Row.env ((List.range k).map c).sum.toUInt64
            (fun i => (frontier.index[i]?.bind (·[k]?)).getD 0)
            (fun j => (frontier.scalar[j]?.bind (·[k]?)).getD default) := by
          simp [Arena.Level.env, htag, hindex, hscalar, hk]
        rw [henv]
        apply H.correct read _ _ _ _ (c k)
        · intro op hop
          funext i
          simp [Step.reader, Nat.ne_of_lt hop]
        · exact hcalls k hk
        · omega
        · intro j hj
          obtain ⟨hi, hv⟩ := hargs k j hk hj
          refine ⟨fun i => (next.index[i]?.bind (·[((List.range k).map c).sum + j]?)).getD 0,
            fun i => (next.scalar[i]?.bind (·[((List.range k).map c).sum + j]?)).getD default,
            hi, hv, ?_⟩
          have hp : ((List.range k).map c).sum + j < next.tag.size := by omega
          have hmod : (((List.range k).map c).sum + j) % 18446744073709551616 =
              ((List.range k).map c).sum + j :=
            Nat.mod_eq_of_lt (show _ < UInt64.size by omega)
          simp [Step.reader, hmod, hp, Arena.Level.slots]

/-- The fold over the levels of an expansion gives the reference result of every root. -/
theorem foldr_expand (H : Hylo α m s) (read : Reader α) {capacity : Nat}
    {roots : Arena.Level α} {levels : List (Arena.Level α)}
    (h : H.expand read capacity roots = some levels) :
    levels.foldr (H.level read) #[] =
      Array.ofFn fun k : Fin roots.tag.size => H.reference read (roots.slots m s k) := by
  unfold Expansion.expand at h
  split at h
  · exact foldr_expandAux H read _ _ _ h
  · cases h

/-- The results of the roots computed level by level: the expansion, then the schedule of the
step, or `none` when either fails. -/
def schedule (H : Hylo α m s) (read : Reader α) (capacity : Nat) (roots : Arena.Level α) :
    Option (Array α) :=
  (H.expand read capacity roots).bind (H.toStep.schedule read)

/-- A successful schedule gives the reference result of every root, as a success. -/
theorem schedule_eq (H : Hylo α m s) (read : Reader α) {capacity : Nat} {roots : Arena.Level α}
    {vs : Array α} (h : H.schedule read capacity roots = some vs) :
    vs.map .ok =
      Array.ofFn fun k : Fin roots.tag.size => H.reference read (roots.slots m s k) := by
  obtain ⟨levels, hl, hs⟩ := Option.bind_eq_some_iff.1 h
  rw [Step.schedule_eq _ read levels hs, foldr_expand H read hl]

/-- The results of the roots: the fold over the expanded levels, or the reference when the
expansion fails. -/
def results (H : Hylo α m s) (read : Reader α) (capacity : Nat) (roots : Arena.Level α) :
    Array (Except Error α) :=
  match H.expand read capacity roots with
  | some levels => levels.foldr (H.level read) #[]
  | none => Array.ofFn fun k : Fin roots.tag.size => H.reference read (roots.slots m s k)

/-- The results of the roots are their reference results, for every capacity. -/
theorem results_eq (H : Hylo α m s) (read : Reader α) (capacity : Nat) (roots : Arena.Level α) :
    H.results read capacity roots =
      Array.ofFn fun k : Fin roots.tag.size => H.reference read (roots.slots m s k) := by
  unfold results
  cases h : H.expand read capacity roots with
  | none => rfl
  | some levels => exact foldr_expand H read h

end Hylo

/-! ### Branches -/

namespace Hylo

variable {m s : Nat}

/-- The expression of `f` at call number `j`, the tag of the row, among the calls `calls`, or `d`
when there are none. -/
def pick {t : Ty} (f : Call α m s → Expr α (Slots m s) t) (d : Expr α (Row m s) t) :
    List (Call α m s) → Expr α (Row m s) t
  | [] => d
  | c :: cs => Fold.select 0 ((f c).rename Var.succ) (cs.map fun c => (f c).rename Var.succ)

/-- The expression that runs `e b` for the first branch `b` of `bs` whose guard holds, and
`e last` when none does, with the guards brought into the variables of `e` by `lift`. -/
def guarded {Γ : List Ty} {t : Ty} (lift : Expr α (Slots m s) .predicate → Expr α Γ .predicate)
    (e : Branch α m s → Expr α Γ t) :
    List (Expr α (Slots m s) .predicate × Branch α m s) → Branch α m s → Expr α Γ t
  | [], last => e last
  | (g, b) :: bs, last => .cond (lift g) (e b) (guarded lift e bs last)

/-- The number of calls of a call: that of the first branch of `bs` whose guard holds on its slots,
or of `last`. -/
def callsOf (bs : List (Expr α (Slots m s) .predicate × Branch α m s)) (last : Branch α m s) :
    Expr α (Slots m s) .index :=
  guarded id (fun b => .index b.calls.length.toUInt64) bs last

/-- Index slot `i` of call `j` of a call, in the branch whose guard holds. A branch without calls
keeps the slot of the caller, which no call reads. -/
def argIndexOf (bs : List (Expr α (Slots m s) .predicate × Branch α m s)) (last : Branch α m s)
    (i : Fin m) : Expr α (Row m s) .index :=
  guarded (Expr.rename Var.succ) (fun b => pick (·.index i) (.var (Row.indexVar i)) b.calls) bs
    last

/-- Scalar slot `i` of call `j` of a call, in the branch whose guard holds. A branch without calls
keeps the slot of the caller, which no call reads. -/
def argValueOf (bs : List (Expr α (Slots m s) .predicate × Branch α m s)) (last : Branch α m s)
    (i : Fin s) : Expr α (Row m s) .scalar :=
  guarded (Expr.rename Var.succ) (fun b => pick (·.value i) (.var (Row.scalarVar i)) b.calls) bs
    last

/-- The step of a call: the step of the branch whose guard holds. -/
def stepOf (bs : List (Expr α (Slots m s) .predicate × Branch α m s)) (last : Branch α m s) :
    Expr α (Row m s) .scalar :=
  guarded (Expr.rename Var.succ) Branch.combine bs last

/-- An expression on the slots of a call, brought to the row of the call, has the same value. -/
theorem eval_rename_succ {t : Ty} (arithmetic : Arithmetic α) (read : Reader α)
    (e : Expr α (Slots m s) t) (tag : UInt64) (index : Nat → UInt64) (value : Nat → α) :
    (e.rename Var.succ).eval arithmetic read (Row.env tag index value) =
      e.eval arithmetic read (Slots.env index value) :=
  Expr.eval_rename e arithmetic read Var.succ _

/-- On the row of call `j`, `pick f d calls` is `f` of call `j`. -/
theorem eval_pick {t : Ty} (arithmetic : Arithmetic α) (read : Reader α)
    (f : Call α m s → Expr α (Slots m s) t) (d : Expr α (Row m s) t) {calls : List (Call α m s)}
    {j : Nat} (hj : j < calls.length) (hsize : j < UInt64.size) (index : Nat → UInt64)
    (value : Nat → α) :
    (pick f d calls).eval arithmetic read (Row.env j.toUInt64 index value) =
      (f calls[j]).eval arithmetic read (Slots.env index value) := by
  cases calls with
  | nil => exact absurd hj (Nat.not_lt_zero _)
  | cons c cs =>
    have h := Fold.eval_select arithmetic read (Slots.env (m := m) (s := s) index value)
      (es := cs.map fun c => (f c).rename Var.succ) (n := 0) (c := j)
      (e := (f c).rename Var.succ) (by simpa using hj) (by omega)
    rw [Nat.zero_add] at h
    refine h.trans ?_
    cases j with
    | zero => exact eval_rename_succ arithmetic read (f c) _ index value
    | succ j =>
      have hj' : j < cs.length := by simpa using hj
      simp only [List.getElem_cons_succ, List.getElem_map]
      exact eval_rename_succ arithmetic read (f cs[j]) _ index value

/-- Two numbers below `UInt64.size` with the same `UInt64` are equal. -/
private theorem eq_of_toUInt64_eq {a b : Nat} (ha : a < UInt64.size) (hb : b < UInt64.size)
    (h : a.toUInt64 = b.toUInt64) : a = b := by
  have := congrArg UInt64.toNat h
  rwa [Nat.toUInt64_eq, Nat.toUInt64_eq, UInt64.toNat_ofNat_of_lt' ha,
    UInt64.toNat_ofNat_of_lt' hb] at this

variable (m s) in
/-- The obligation of branch `b` on the calls whose reader and slots satisfy `P`: it has fewer than
`UInt64.size` calls, and if `R` agrees with `read` below `inputs` and input `inputs` of `R` holds
the reference results of its calls from position `first` on, its step gives the reference result
of the call. -/
def Branch.Correct (arithmetic : Arithmetic α) (inputs : Nat)
    (reference : Reader α → Env α (Slots m s) → Except Error α)
    (P : Reader α → Env α (Slots m s) → Prop) (b : Branch α m s) : Prop :=
  b.calls.length < UInt64.size ∧
  ∀ (read R : Reader α) (index : Nat → UInt64) (value : Nat → α) (first : Nat),
    (∀ op < inputs, R op = read op) → P read (Slots.env index value) →
    first + b.calls.length < UInt64.size →
    (∀ j (hj : j < b.calls.length), ∃ (index' : Nat → UInt64) (value' : Nat → α),
      (∀ i : Fin m, (b.calls[j].index i).eval arithmetic read (Slots.env index value) =
        .ok (index' i)) ∧
      (∀ i : Fin s, (b.calls[j].value i).eval arithmetic read (Slots.env index value) =
        .ok (value' i)) ∧
      R inputs (first + j).toUInt64 = reference read (Slots.env index' value')) →
    b.combine.eval arithmetic R (Row.env first.toUInt64 index value) =
      reference read (Slots.env index value)

variable (m s) in
/-- The obligations of the branches `bs`, then `last`, on the calls that satisfy `P`: every guard
reads inputs below `inputs`, and each branch has its obligation on the calls that satisfy `P`,
fail the guards before it and pass its own. -/
def Branches.Correct (arithmetic : Arithmetic α) (inputs : Nat)
    (reference : Reader α → Env α (Slots m s) → Except Error α) :
    (Reader α → Env α (Slots m s) → Prop) →
      List (Expr α (Slots m s) .predicate × Branch α m s) → Branch α m s → Prop
  | P, [], last => Branch.Correct m s arithmetic inputs reference P last
  | P, (g, b) :: bs, last => g.readsBelow inputs ∧
    Branch.Correct m s arithmetic inputs reference
      (fun read env => P read env ∧ g.eval arithmetic read env = .ok true) b ∧
    Branches.Correct arithmetic inputs reference
      (fun read env => P read env ∧ g.eval arithmetic read env = .ok false) bs last

/-- A branch whose obligation holds gives the reference result of a call when its number of calls
is that of the call and its calls are those of the call. -/
theorem Branch.Correct.eval {arithmetic : Arithmetic α} {inputs : Nat}
    {reference : Reader α → Env α (Slots m s) → Except Error α}
    {P : Reader α → Env α (Slots m s) → Prop} {b : Branch α m s}
    (h : Branch.Correct m s arithmetic inputs reference P b) (read R : Reader α)
    (index : Nat → UInt64) (value : Nat → α) (first n : Nat) (hR : ∀ op < inputs, R op = read op)
    (hP : P read (Slots.env index value)) (hn : b.calls.length.toUInt64 = n.toUInt64)
    (hfirst : first + n < UInt64.size)
    (hcall : ∀ j < n, ∃ (index' : Nat → UInt64) (value' : Nat → α),
      (∀ i : Fin m, (pick (·.index i) (.var (Row.indexVar i)) b.calls).eval arithmetic read
        (Row.env j.toUInt64 index value) = .ok (index' i)) ∧
      (∀ i : Fin s, (pick (·.value i) (.var (Row.scalarVar i)) b.calls).eval arithmetic read
        (Row.env j.toUInt64 index value) = .ok (value' i)) ∧
      R inputs (first + j).toUInt64 = reference read (Slots.env index' value')) :
    b.combine.eval arithmetic R (Row.env first.toUInt64 index value) =
      reference read (Slots.env index value) := by
  obtain ⟨hlength, hb⟩ := h
  have hn' := eq_of_toUInt64_eq hlength (by omega) hn
  subst hn'
  refine hb read R index value first hR hP hfirst fun j hj => ?_
  obtain ⟨index', value', hi, hv, hRj⟩ := hcall j hj
  refine ⟨index', value', fun i => ?_, fun i => ?_, hRj⟩
  · rw [← hi i, eval_pick arithmetic read _ _ hj (by omega)]
  · rw [← hv i, eval_pick arithmetic read _ _ hj (by omega)]

/-- The hylomorphism of branches meets the certificate of `Hylo.correct` on the calls that satisfy
`P` once the obligations of its branches hold. -/
theorem correct_guarded {arithmetic : Arithmetic α} {inputs : Nat}
    {reference : Reader α → Env α (Slots m s) → Except Error α} :
    ∀ {P : Reader α → Env α (Slots m s) → Prop}
      {bs : List (Expr α (Slots m s) .predicate × Branch α m s)} {last : Branch α m s},
      Branches.Correct m s arithmetic inputs reference P bs last →
      ∀ (read R : Reader α) (index : Nat → UInt64) (value : Nat → α) (first n : Nat),
        (∀ op < inputs, R op = read op) → P read (Slots.env index value) →
        (callsOf bs last).eval arithmetic read (Slots.env index value) = .ok n.toUInt64 →
        first + n < UInt64.size →
        (∀ j < n, ∃ (index' : Nat → UInt64) (value' : Nat → α),
          (∀ i : Fin m, (argIndexOf bs last i).eval arithmetic read
            (Row.env j.toUInt64 index value) = .ok (index' i)) ∧
          (∀ i : Fin s, (argValueOf bs last i).eval arithmetic read
            (Row.env j.toUInt64 index value) = .ok (value' i)) ∧
          R inputs (first + j).toUInt64 = reference read (Slots.env index' value')) →
        (stepOf bs last).eval arithmetic R (Row.env first.toUInt64 index value) =
          reference read (Slots.env index value)
  | P, [], last, h, read, R, index, value, first, n, hR, hP, hcalls, hfirst, hcall => by
    simp only [callsOf, guarded, Expr.eval, Except.ok.injEq] at hcalls
    exact h.eval read R index value first n hR hP hcalls hfirst hcall
  | P, (g, b) :: bs, last, ⟨hg, hb, hrest⟩, read, R, index, value, first, n, hR, hP, hcalls,
      hfirst, hcall => by
    have hgR := Expr.eval_eq_of_readsBelow arithmetic hR g hg (Slots.env index value)
    have hlift : ∀ (read : Reader α) (tag : UInt64),
        (Expr.rename (Δ := Row m s) Var.succ g).eval arithmetic read (Row.env tag index value) =
          g.eval arithmetic read (Slots.env index value) := fun read tag =>
      eval_rename_succ arithmetic read g tag index value
    cases hge : g.eval arithmetic read (Slots.env index value) with
    | error e =>
      simp [callsOf, guarded, Expr.eval, hge, Bind.bind, Except.bind] at hcalls
    | ok p =>
      cases p with
      | true =>
        simp only [callsOf, guarded, Expr.eval, id, hge, Bind.bind, Except.bind, Bool.cond_true,
          Except.ok.injEq] at hcalls
        simp only [stepOf, guarded, Expr.eval, hlift, hgR, hge, Bind.bind, Except.bind,
          Bool.cond_true]
        refine hb.eval read R index value first n hR ⟨hP, hge⟩ hcalls hfirst fun j hj => ?_
        obtain ⟨index', value', hi, hv, hRj⟩ := hcall j hj
        refine ⟨index', value', fun i => ?_, fun i => ?_, hRj⟩
        · have := hi i
          simp only [argIndexOf, guarded, Expr.eval, hlift, hge, Bind.bind, Except.bind,
            Bool.cond_true] at this
          exact this
        · have := hv i
          simp only [argValueOf, guarded, Expr.eval, hlift, hge, Bind.bind, Except.bind,
            Bool.cond_true] at this
          exact this
      | false =>
        simp only [callsOf, guarded, Expr.eval, id, hge, Bind.bind, Except.bind,
          Bool.cond_false] at hcalls
        simp only [stepOf, guarded, Expr.eval, hlift, hgR, hge, Bind.bind, Except.bind,
          Bool.cond_false]
        refine correct_guarded hrest read R index value first n hR ⟨hP, hge⟩ hcalls hfirst
          fun j hj => ?_
        obtain ⟨index', value', hi, hv, hRj⟩ := hcall j hj
        refine ⟨index', value', fun i => ?_, fun i => ?_, hRj⟩
        · have := hi i
          simp only [argIndexOf, guarded, Expr.eval, hlift, hge, Bind.bind, Except.bind,
            Bool.cond_false] at this
          exact this
        · have := hv i
          simp only [argValueOf, guarded, Expr.eval, hlift, hge, Bind.bind, Except.bind,
            Bool.cond_false] at this
          exact this

/-- The hylomorphism of branches meets the certificate of `Hylo.correct` once the obligations of its
branches hold. -/
theorem correct_branches {arithmetic : Arithmetic α} {inputs : Nat}
    {reference : Reader α → Env α (Slots m s) → Except Error α}
    {bs : List (Expr α (Slots m s) .predicate × Branch α m s)} {last : Branch α m s}
    (h : Branches.Correct m s arithmetic inputs reference (fun _ _ => True) bs last)
    (read R : Reader α) (index : Nat → UInt64) (value : Nat → α) (first n : Nat)
    (hR : ∀ op < inputs, R op = read op)
    (hcalls : (callsOf bs last).eval arithmetic read (Slots.env index value) = .ok n.toUInt64)
    (hfirst : first + n < UInt64.size)
    (hcall : ∀ j < n, ∃ (index' : Nat → UInt64) (value' : Nat → α),
      (∀ i : Fin m, (argIndexOf bs last i).eval arithmetic read
        (Row.env j.toUInt64 index value) = .ok (index' i)) ∧
      (∀ i : Fin s, (argValueOf bs last i).eval arithmetic read
        (Row.env j.toUInt64 index value) = .ok (value' i)) ∧
      R inputs (first + j).toUInt64 = reference read (Slots.env index' value')) :
    (stepOf bs last).eval arithmetic R (Row.env first.toUInt64 index value) =
      reference read (Slots.env index value) :=
  correct_guarded h read R index value first n hR trivial hcalls hfirst hcall

end Hylo

namespace Hylo

variable {m s : Nat} [Inhabited α]

/-- The results of the roots `roots` under a hylomorphism, computed on the device: the levels of
`Expansion.expandBlocks`, then one launch of the step per level from the deepest up
(`Step.runBlocks`), with the arrays `inputs` as the inputs, packed once for both. Returns `none`
when a read failed, or when the levels would hold more than `capacity` rows or more than
`Expansion.deviceCapacity`. -/
@[no_expose] def runDevice (H : Hylo α m s) (capacity : Nat) (roots : Arena.Level α)
    (inputs : Array (Array α)) : IO (Option (Array α)) := do
  let blocks := Step.inputBlocks H.scalar H.inputs inputs
  let some (handle, sizes) ← H.expandBlocks capacity roots blocks | return none
  H.toStep.runBlocks handle sizes blocks

/-- The results of the roots `roots` under a hylomorphism, with the arrays `inputs` as its inputs.
With LibTorch and a usable device they come from the device (`Hylo.runDevice`). Otherwise, or
when the device gives `none`, they are `Hylo.results` on the host at the capacity
`min capacity Expansion.deviceCapacity`, which are the reference results by `Hylo.results_eq`.
On the device it fails as `Hylo.runDevice` does: when a source does not render or compile, as
for an expression that reads an input at or past `inputs`, or when the roots do not have the
slots of the expansion. -/
@[no_expose] def run (H : Hylo α m s) (capacity : Nat) (roots : Arena.Level α)
    (inputs : Array (Array α)) : IO (Array (Except Error α)) := do
  if (← LibTorch.Buffer.RuntimeStatus.query) = .nativeAvailable then
    if let some vs ← H.runDevice capacity roots inputs then
      return vs.map .ok
  return H.results (Reader.ofArrays inputs) (min capacity Expansion.deviceCapacity) roots

end Hylo

end TorchLean.Kernel
