/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Arena
public import TorchLean.CUDA.Kernel.Scalar

/-!
# Folds over arenas

A `Fold` computes one result for every node of a value from the node's own row and the results
of its children, so the result at the root is a recursive function of the value. Its `Step` is a
kernel `Expr` over the variables of a row (`Row`: its tag, then its index slots, then its scalar
slots, named by `Row.indexVar` and `Row.scalarVar`), and it reads the result of the child on row
`p` of the next level as input `inputs` at index `p`. Every level is then one data-parallel pass
over its rows, and the passes run from the deepest level up. Compiled code computes the rows of a
level in runs, each run after the first in a task of its own (`Step.level_eq_levelTasks`), so the
rows of a large level are spread over the cores. A step knows only the numbers of index and scalar
slots of its rows, not the signature, so levels that other recursion schemes build run through
the same passes.

`Fold.correct` is a certificate about one layer: if input `inputs` holds the reference results of
the children, the step gives the reference result of the layer. `Fold.eval_of_decode` lifts it to
whole values: the level-by-level schedule returns the reference result for every arena that
decodes, read failures included, and `Fold.eval_encode` states this for the arena of a value.
A frontend proves the certificate one constructor at a time (`Fold.Cases`) for a step that
selects the constructor's case by the tag (`Fold.select`), and `Fold.correct_select` combines
the constructors.
-/

@[expose] public section

namespace TorchLean.Kernel

/-- The environment with no variables. -/
def Env.empty {α : Type} : Env α [] := ⟨fun v => nomatch v⟩

/-- Bind `n` variables of type `t` in front of `env`, the newest to `value 0` and the oldest to
`value (n - 1)`. -/
def Env.slots {α : Type} {Γ : List Ty} (t : Ty) (value : Nat → t.Value α) :
    (n : Nat) → Env α Γ → Env α (List.replicate n t ++ Γ)
  | 0, env => env
  | n + 1, env => Env.push (value 0) (Env.slots t (fun i => value (i + 1)) n env)

/-- Bind `n` variables of type `t`, the newest to `value 0` and the oldest to `value (n - 1)`. -/
def Env.replicate {α : Type} (t : Ty) (value : Nat → t.Value α) :
    (n : Nat) → Env α (List.replicate n t)
  | 0 => Env.empty
  | n + 1 => Env.push (value 0) (Env.replicate t (fun i => value (i + 1)) n)

/-- Variable `i` of `n` variables of type `t` bound in front of `Γ`, as `Env.slots` binds them. -/
def Var.slot {t : Ty} {Γ : List Ty} : (n : Nat) → Fin n → Var (List.replicate n t ++ Γ) t
  | _ + 1, ⟨0, _⟩ => .zero
  | n + 1, ⟨i + 1, h⟩ => .succ (Var.slot n ⟨i, Nat.lt_of_succ_lt_succ h⟩)

/-- Variable `i` of `n` variables of type `t`, as `Env.replicate` binds them. -/
def Var.replicate {t : Ty} : (n : Nat) → Fin n → Var (List.replicate n t) t
  | _ + 1, ⟨0, _⟩ => .zero
  | n + 1, ⟨i + 1, h⟩ => .succ (Var.replicate n ⟨i, Nat.lt_of_succ_lt_succ h⟩)

/-- The variable `v` of `Γ` behind `n` variables of type `s`. -/
def Var.skip {s t : Ty} {Γ : List Ty} : (n : Nat) → Var Γ t → Var (List.replicate n s ++ Γ) t
  | 0, v => v
  | n + 1, v => .succ (Var.skip n v)

/-- Variable `i` of the variables that `Env.slots` binds holds `value i`. -/
@[simp] theorem Env.get_slots_slot {α : Type} {Γ : List Ty} {t : Ty} (env : Env α Γ) :
    ∀ (n : Nat) (value : Nat → t.Value α) (i : Fin n),
      (Env.slots t value n env).get (Var.slot n i) = value i
  | _ + 1, _, ⟨0, _⟩ => rfl
  | n + 1, value, ⟨i + 1, _⟩ => Env.get_slots_slot env n (fun i => value (i + 1)) ⟨i, _⟩

/-- Variable `i` of the variables that `Env.replicate` binds holds `value i`. -/
@[simp] theorem Env.get_replicate_replicate {α : Type} {t : Ty} :
    ∀ (n : Nat) (value : Nat → t.Value α) (i : Fin n),
      (Env.replicate t value n).get (Var.replicate n i) = value i
  | _ + 1, _, ⟨0, _⟩ => rfl
  | n + 1, value, ⟨i + 1, _⟩ => Env.get_replicate_replicate n (fun i => value (i + 1)) ⟨i, _⟩

/-- The variables that `Env.slots` binds hide none of the variables behind them. -/
@[simp] theorem Env.get_slots_skip {α : Type} {Γ : List Ty} {s t : Ty} (env : Env α Γ)
    (v : Var Γ t) : ∀ (n : Nat) (value : Nat → s.Value α),
      (Env.slots s value n env).get (Var.skip n v) = env.get v
  | 0, _ => rfl
  | n + 1, value => Env.get_slots_skip env v n (fun i => value (i + 1))

/-- The variables of `m` index slots and then `s` scalar slots, newest first. -/
def Slots (m s : Nat) : List Ty :=
  List.replicate m .index ++ List.replicate s .scalar

/-- The variables of a row with `m` index slots and `s` scalar slots, newest first: the tag, then
the slots. -/
def Row (m s : Nat) : List Ty :=
  .index :: Slots m s

/-- The environment of slots with index slot `i` holding `index i` and scalar slot `j` holding
`value j`. -/
def Slots.env {α : Type} {m s : Nat} (index : Nat → UInt64) (value : Nat → α) :
    Env α (Slots m s) :=
  Env.slots .index index m (Env.replicate .scalar value s)

/-- The environment of a row with tag `tag`, index slot `i` holding `index i` and scalar slot `j`
holding `value j`. -/
def Row.env {α : Type} {m s : Nat} (tag : UInt64) (index : Nat → UInt64) (value : Nat → α) :
    Env α (Row m s) :=
  Env.push tag (Slots.env index value)

/-- The variable of index slot `i`. -/
def Slots.indexVar {m s : Nat} (i : Fin m) : Var (Slots m s) .index :=
  Var.slot m i

/-- The variable of scalar slot `j`. -/
def Slots.scalarVar {m s : Nat} (j : Fin s) : Var (Slots m s) .scalar :=
  Var.skip m (Var.replicate s j)

/-- The variable of index slot `i` of a row. -/
def Row.indexVar {m s : Nat} (i : Fin m) : Var (Row m s) .index :=
  .succ (Slots.indexVar i)

/-- The variable of scalar slot `j` of a row. -/
def Row.scalarVar {m s : Nat} (j : Fin s) : Var (Row m s) .scalar :=
  .succ (Slots.scalarVar j)

/-- Index slot `i` holds `index i`. -/
@[simp] theorem Slots.get_env_indexVar {α : Type} {m s : Nat} (index : Nat → UInt64)
    (value : Nat → α) (i : Fin m) :
    (Slots.env index value : Env α (Slots m s)).get (Slots.indexVar i) = index i :=
  Env.get_slots_slot _ _ _ i

/-- Scalar slot `j` holds `value j`. -/
@[simp] theorem Slots.get_env_scalarVar {α : Type} {m s : Nat} (index : Nat → UInt64)
    (value : Nat → α) (j : Fin s) :
    (Slots.env index value : Env α (Slots m s)).get (Slots.scalarVar j) = value j := by
  simp only [Slots.env, Slots.scalarVar, Env.get_slots_skip, Env.get_replicate_replicate]

/-- The first variable of a row holds its tag. -/
@[simp] theorem Row.get_env_zero {α : Type} {m s : Nat} (tag : UInt64) (index : Nat → UInt64)
    (value : Nat → α) : (Row.env tag index value : Env α (Row m s)).get .zero = tag :=
  rfl

/-- Index slot `i` of a row holds `index i`. -/
@[simp] theorem Row.get_env_indexVar {α : Type} {m s : Nat} (tag : UInt64)
    (index : Nat → UInt64) (value : Nat → α) (i : Fin m) :
    (Row.env tag index value : Env α (Row m s)).get (Row.indexVar i) = index i :=
  Slots.get_env_indexVar index value i

/-- Scalar slot `j` of a row holds `value j`. -/
@[simp] theorem Row.get_env_scalarVar {α : Type} {m s : Nat} (tag : UInt64)
    (index : Nat → UInt64) (value : Nat → α) (j : Fin s) :
    (Row.env tag index value : Env α (Row m s)).get (Row.scalarVar j) = value j :=
  Slots.get_env_scalarVar index value j

/-- The variables of a row of `σ`: the tag, the index slots and the scalar slots. -/
abbrev Sig.row (σ : Sig) : List Ty :=
  Row σ.indexSlots σ.scalarSlots

/-- The variables of row `k` of a level with `m` index slots and `s` scalar slots: its tag, its
index slots and its scalar slots, with defaults past the end of a column. -/
def Arena.Level.env {α : Type} [Inhabited α] (m s : Nat) (level : Arena.Level α) (k : Nat) :
    Env α (Row m s) :=
  Row.env (level.tag[k]?.getD 0) (fun i => (level.index[i]?.bind (·[k]?)).getD 0)
    (fun j => (level.scalar[j]?.bind (·[k]?)).getD default)

/-- A step over the rows of levels with `m` index slots and `s` scalar slots: an expression over
the variables of a row, which reads the results of the next level as input `inputs`, with the
scalar type and the arithmetic it runs in. -/
structure Step (α : Type) (m s : Nat) where
  /-- The scalar type with its device precision. -/
  scalar : Scalar α
  /-- The arithmetic of the expression. -/
  arithmetic : Arithmetic α
  /-- The number of inputs that the expression reads besides the results of the next level. -/
  inputs : Nat
  /-- The expression that computes the result of a row. -/
  expr : Expr α (Row m s) .scalar
  /-- The arithmetic is the device arithmetic of the scalar type, which the device kernels are
  assumed to compute, as every kernel of `Kernel.Launch` is. -/
  native : arithmetic = scalar.arithmetic

namespace Step

variable {α : Type} {m s : Nat}

/-- The reader of a level: input `inputs` reads the results `below` of the next level and fails
past their end, and the other inputs come from `read`. -/
def reader (read : Reader α) (inputs : Nat) (below : Array (Except Error α)) : Reader α :=
  fun op i => if op = inputs then below[i.toNat]?.getD (.error (.bounds inputs i below.size))
    else read op i

variable [Inhabited α]

/-- The results of the rows of `level`, given the results `below` of the next level. -/
def level (S : Step α m s) (read : Reader α) (level : Arena.Level α)
    (below : Array (Except Error α)) : Array (Except Error α) :=
  Array.ofFn fun k : Fin level.tag.size =>
    S.expr.eval S.arithmetic (reader read S.inputs below)
      (Row.env level.tag[k] (fun i => (level.index[i]?.bind (·[k]?)).getD 0)
        (fun j => (level.scalar[j]?.bind (·[k]?)).getD default))

/-- The rows of `level` cut into runs of `grain` rows, each run after the first computed in a task
of its own, and the runs appended in order. -/
def levelRuns (grain : Nat) (S : Step α m s) (read : Reader α) (level : Arena.Level α)
    (below : Array (Except Error α)) : Array (Except Error α) :=
  let n := level.tag.size
  let run (r : Nat) := (Array.range (min grain (n - r * grain))).map fun k =>
    S.expr.eval S.arithmetic (reader read S.inputs below) (level.env m s (r * grain + k))
  let tasks := (List.range ((n - 1) / grain)).map fun r => Task.spawn fun _ => run (r + 1)
  tasks.foldl (fun out t => out ++ t.get) (run 0)

private theorem foldl_runs {β : Type} (g n : Nat) (f : Nat → β) : ∀ m : Nat,
    (List.range m).foldl (fun out r =>
        out ++ (Array.range (min g (n - r * g))).map fun k => f (r * g + k)) #[] =
      (Array.range (min (m * g) n)).map f
  | 0 => by simp
  | m + 1 => by
    rw [List.range_succ, List.foldl_append, foldl_runs g n f m, List.foldl_cons, List.foldl_nil]
    by_cases h : n ≤ m * g
    · have h1 : min (m * g) n = n := Nat.min_eq_right h
      have h2 : min ((m + 1) * g) n = n := Nat.min_eq_right (by rw [Nat.succ_mul]; omega)
      have h3 : n - m * g = 0 := by omega
      simp [h1, h2, h3]
    · have h1 : min (m * g) n = m * g := Nat.min_eq_left (by omega)
      have h2 : min ((m + 1) * g) n = m * g + min g (n - m * g) := by
        rw [Nat.succ_mul]; omega
      rw [h1, h2, Array.range_add, Array.map_append, Array.map_map]
      rfl

private theorem foldl_tasks {β : Type} (g n : Nat) (hg : 0 < g) (f : Nat → β) :
    ((List.range ((n - 1) / g)).map fun r => Task.spawn fun _ =>
        (Array.range (min g (n - (r + 1) * g))).map fun k => f ((r + 1) * g + k)).foldl
      (fun out t => out ++ t.get) ((Array.range (min g (n - 0 * g))).map fun k => f (0 * g + k)) =
      (Array.range n).map f := by
  have key := foldl_runs g n f ((n - 1) / g + 1)
  rw [List.range_succ_eq_map, List.foldl_cons, List.foldl_map] at key
  have hmin : min (((n - 1) / g + 1) * g) n = n := by
    apply Nat.min_eq_right
    have := Nat.lt_div_mul_add (a := n - 1) hg
    rw [Nat.succ_mul]; omega
  rw [hmin] at key
  rw [List.foldl_map]
  simpa [Task.spawn] using key

/-- Cutting the rows of a level into runs of `grain` rows, `grain` positive, gives the same
results. -/
theorem levelRuns_eq (S : Step α m s) (read : Reader α) (level : Arena.Level α)
    (below : Array (Except Error α)) {grain : Nat} (hg : 0 < grain) :
    S.levelRuns grain read level below = S.level read level below := by
  rw [levelRuns, foldl_tasks grain _ hg fun k =>
    S.expr.eval S.arithmetic (reader read S.inputs below) (level.env m s k)]
  refine Array.ext (by simp [Step.level]) fun k hk _ => ?_
  have hk' : k < level.tag.size := by simpa using hk
  simp [Step.level, Arena.Level.env, hk']

/-- `Step.levelRuns` with runs of 256 rows. -/
def levelTasks (S : Step α m s) (read : Reader α) (level : Arena.Level α)
    (below : Array (Except Error α)) : Array (Except Error α) :=
  S.levelRuns 256 read level below

/-- Compiled code computes the rows of a level in runs of 256 rows, in parallel. -/
@[csimp] theorem level_eq_levelTasks : @level = @levelTasks := by
  funext α m s _ S read level below
  exact (levelRuns_eq S read level below (by decide)).symm

/-- The results of the rows of level `r` of an arena, computed one level at a time from the
deepest level up. -/
def results (S : Step α m s) (read : Reader α) (a : Arena α) (r : Nat) :
    Array (Except Error α) :=
  (a.levels.toList.drop r).foldr (S.level read) #[]

/-- The results of level `r` come from the level and the results of level `r + 1`. -/
theorem results_eq (S : Step α m s) (read : Reader α) {a : Arena α} {r : Nat}
    {level : Arena.Level α} (h : a.levels[r]? = some level) :
    S.results read a r = S.level read level (S.results read a (r + 1)) := by
  have hr : r < a.levels.toList.length := by
    have := (Array.getElem?_eq_some_iff.1 h).1
    simpa using this
  have hl : a.levels.toList[r] = level := by
    have := (Array.getElem?_eq_some_iff.1 h).2
    simpa using this
  rw [results, List.drop_eq_getElem_cons hr, List.foldr_cons, hl]
  rfl

end Step

namespace Fold

variable {σ : Sig} {α : Type} {T : Fin σ.sorts → Type} [σ.Inductive α T]

/-- The expression that runs `e` on rows with tag `n`, the first of `es` on rows with tag
`n + 1` and so on, the last case also running on every other row. -/
def select {Γ : List Ty} {t : Ty} :
    Nat → Expr α (.index :: Γ) t → List (Expr α (.index :: Γ) t) → Expr α (.index :: Γ) t
  | _, e, [] => e
  | n, e, e' :: es =>
    .cond (.indexCompare .eq (.var .zero) (.index n.toUInt64)) e (select (n + 1) e' es)

/-- On a row with tag `n + c`, the selecting expression runs case `c`. -/
theorem eval_select {Γ : List Ty} {t : Ty} (arithmetic : Arithmetic α) (read : Reader α)
    (env : Env α Γ) :
    ∀ {es : List (Expr α (.index :: Γ) t)} {n c : Nat} {e : Expr α (.index :: Γ) t}
      (hc : c < es.length + 1), n + c < UInt64.size →
      (select n e es).eval arithmetic read (env.push (n + c).toUInt64) =
        (e :: es)[c].eval arithmetic read (env.push (n + c).toUInt64)
  | [], _, 0, _, _, _ => rfl
  | e' :: es, n, 0, e, _, _ => by simp [select, Expr.eval, Compare.index, Bind.bind, Except.bind,
      Pure.pure, Except.pure]
  | e' :: es, n, c + 1, e, hc, hn => by
    have hne : ((n + (c + 1)).toUInt64 == n.toUInt64) = false := by
      rw [beq_eq_false_iff_ne]
      intro h
      have := congrArg UInt64.toNat h
      rw [Nat.toUInt64_eq, Nat.toUInt64_eq, UInt64.toNat_ofNat_of_lt' hn,
        UInt64.toNat_ofNat_of_lt' (by omega)] at this
      omega
    have ih := eval_select arithmetic read env (es := es) (n := n + 1) (c := c) (e := e')
      (by simpa using hc) (by omega)
    rw [show n + 1 + c = n + (c + 1) by omega] at ih
    simp only [select, Expr.eval, Compare.index, Env.get_push_zero, Bind.bind, Except.bind,
      Pure.pure, Except.pure, hne, Bool.cond_false, List.getElem_cons_succ]
    exact ih

variable (σ) in
/-- The obligation of constructor `c` with case `e`: if input `inputs` of `R` holds the
reference results of the children of a layer of `c`, and `R` agrees with `read` below `inputs`,
then `e` gives the reference result of the layer. -/
def Case (arithmetic : Arithmetic α) (inputs : Nat)
    (reference : Reader α → (s : Fin σ.sorts) → T s → Except Error α) (c : Fin σ.ctors.length)
    (e : Expr α σ.row .scalar) : Prop :=
  ∀ (read R : Reader α) (index : Nat → UInt64) (value : Nat → α)
    (args : Field.Args α T σ.ctors[c].fields),
    (∀ op < inputs, R op = read op) →
    Field.Args.Matches index value (fun s' p y => R inputs p = reference read s' y)
      σ.ctors[c].fields args 0 0 →
    e.eval arithmetic R (Row.env c.val.toUInt64 index value) =
      reference read σ.ctors[c].sort (Sig.Inductive.mk ⟨c, rfl, args⟩)

variable (σ) in
/-- The obligations of the constructors from number `k` on, the first with case `e` and the
next ones with the cases `es`. -/
def Cases (arithmetic : Arithmetic α) (inputs : Nat)
    (reference : Reader α → (s : Fin σ.sorts) → T s → Except Error α) :
    Nat → List (Expr α σ.row .scalar) → Prop
  | _, [] => True
  | k, e :: es => (∀ h : k < σ.ctors.length, Case σ arithmetic inputs reference ⟨k, h⟩ e) ∧
    Cases arithmetic inputs reference (k + 1) es

/-- The obligations of the constructors from number `k` on give the obligation of each one. -/
theorem Cases.case {arithmetic : Arithmetic α} {inputs : Nat}
    {reference : Reader α → (s : Fin σ.sorts) → T s → Except Error α} :
    ∀ {k : Nat} {es : List (Expr α σ.row .scalar)}, Cases σ arithmetic inputs reference k es →
      ∀ (i : Nat) (hi : i < es.length) (h : k + i < σ.ctors.length),
        Case σ arithmetic inputs reference ⟨k + i, h⟩ es[i]
  | _, [], _, _, hi, _ => absurd hi (Nat.not_lt_zero _)
  | _, _ :: _, hes, 0, _, h => hes.1 h
  | k, _ :: es, hes, i + 1, hi, h => by
    have := hes.2.case i (by simpa using hi) (by omega)
    simp only [show k + 1 + i = k + (i + 1) by omega] at this
    exact this

/-- The step that selects the case of each constructor by the tag computes the reference result
of every layer whose children's results it reads, once each constructor's obligation holds. -/
theorem correct_select {arithmetic : Arithmetic α} {inputs : Nat}
    {reference : Reader α → (s : Fin σ.sorts) → T s → Except Error α}
    {e : Expr α σ.row .scalar} {es : List (Expr α σ.row .scalar)}
    (hcases : Cases σ arithmetic inputs reference 0 (e :: es))
    (hlength : es.length + 1 = σ.ctors.length)
    (read R : Reader α) (index : Nat → UInt64) (value : Nat → α) {s : Fin σ.sorts}
    (l : Layer σ α T s) (hR : ∀ op < inputs, R op = read op)
    (hM : Field.Args.Matches index value (fun s' p y => R inputs p = reference read s' y)
      σ.ctors[l.ctor].fields l.args 0 0) :
    (select 0 e es).eval arithmetic R (Row.env l.ctor.val.toUInt64 index value) =
      reference read s (Sig.Inductive.mk l) := by
  obtain ⟨c, rfl, args⟩ := l
  have hc : c.val < es.length + 1 := by omega
  have hsize : 0 + c.val < UInt64.size := by
    have := σ.ctors_lt
    omega
  have h := eval_select arithmetic R
    (Slots.env (m := σ.indexSlots) (s := σ.scalarSlots) index value) (e := e) hc hsize
  rw [Nat.zero_add] at h
  have hcase := hcases.case c.val hc (by omega)
  simp only [Nat.zero_add] at hcase
  exact h.trans (hcase read R index value args hR hM)

end Fold

/-- A fold over values of `σ` with scalars in `α`: a step over the variables of a row, which reads
the results of the children as input `inputs`, certified against a reference function on every
layer. -/
structure Fold (σ : Sig) (α : Type) (T : Fin σ.sorts → Type) [σ.Inductive α T] extends
    Step α σ.indexSlots σ.scalarSlots where
  /-- The function that the fold computes, for every reader of the inputs. -/
  reference : Reader α → (s : Fin σ.sorts) → T s → Except Error α
  /-- If input `inputs` of `R` holds the reference results of the children of a layer, and `R`
  agrees with `read` below `inputs`, the step gives the reference result of the layer. -/
  correct : ∀ (read R : Reader α) (index : Nat → UInt64) (value : Nat → α) {s : Fin σ.sorts}
    (l : Layer σ α T s), (∀ op < inputs, R op = read op) →
    Field.Args.Matches index value (fun s' p y => R inputs p = reference read s' y)
      σ.ctors[l.ctor].fields l.args 0 0 →
    expr.eval arithmetic R (Row.env l.ctor.val.toUInt64 index value) =
      reference read s (Sig.Inductive.mk l)

namespace Fold

variable {σ : Sig} {α : Type} {T : Fin σ.sorts → Type} [σ.Inductive α T] [Inhabited α]

/-- The result of the root of an arena, row zero of level zero. An arena without levels has no
root, and its result is a failed read of row zero. -/
def eval (F : Fold σ α T) (read : Reader α) (a : Arena α) : Except Error α :=
  (F.results read a 0)[0]?.getD (.error (.bounds F.inputs 0 0))

/-- Every row that decodes to a value has the reference result of that value, for any fuel. -/
theorem results_of_decodeAt (F : Fold σ α T) (read : Reader α) (a : Arena α) :
    ∀ {fuel r k : Nat} {s : Fin σ.sorts} {y : T s}, a.decodeAt T fuel r k s = some y →
      (F.results read a r)[k]? = some (F.reference read s y)
  | 0, _, _, _, _, h => by simp [Arena.decodeAt] at h
  | fuel + 1, r, k, s, y, h => by
    obtain ⟨level, l, hl, htag, rfl, hM⟩ := Arena.decodeAt_eq_some h
    have hk : k < level.tag.size := (Array.getElem?_eq_some_iff.1 htag).1
    have hk' : level.tag[k] = l.ctor.val.toUInt64 := (Array.getElem?_eq_some_iff.1 htag).2
    rw [F.results_eq read hl, Step.level, Array.getElem?_ofFn]
    simp only [hk, dite_true, Option.some.injEq]
    rw [show level.tag[(⟨k, hk⟩ : Fin level.tag.size)] = l.ctor.val.toUInt64 from hk']
    refine F.correct read _ _ _ l (fun op hop => ?_) (Field.Args.Matches.imp ?_ hM)
    · funext i
      simp [Step.reader, Nat.ne_of_lt hop]
    · intro s' p c hc
      simp [Step.reader, results_of_decodeAt F read a hc]

/-- The schedule returns the reference result of the value that an arena decodes to. -/
theorem eval_of_decode (F : Fold σ α T) (read : Reader α) {a : Arena α} {s : Fin σ.sorts}
    {y : T s} (h : a.decode T s = some y) : F.eval read a = F.reference read s y := by
  simp [eval, results_of_decodeAt F read a h]

/-- The schedule on the arena of a value returns the reference result of the value, if it has
fewer than `2 ^ 64` nodes. -/
theorem eval_encode (F : Fold σ α T) (read : Reader α) {s : Fin σ.sorts} (t : T s)
    (ht : Sig.Inductive.count t < UInt64.size) :
    F.eval read (Arena.encode t) = F.reference read s t :=
  F.eval_of_decode read (Arena.decode_encode t ht)

end Fold

end TorchLean.Kernel
