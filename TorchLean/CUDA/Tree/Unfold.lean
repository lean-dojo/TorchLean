/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.CUDA.Tree.Expand

/-!
# Unfolds into arenas

An `Unfold` builds values of an inductive family from seeds. A seed is a row of slots, and the
seeds expand as the calls of an `Expansion` do: a seed makes one seed per child of its value, and
`Expansion.expand` lays them out level by level, with the children of row `k` on consecutive rows
of the next level from the sum of the numbers of children of the rows before it. On every level the
node expressions `tag`, `nodeIndex` and `nodeScalar` give the node of each row from that position
and the slots of its seed. The prefix sum of the numbers of children (Blelloch 1990) is then the
allocator of the nodes: the position of a child is its row, so a node refers to its children by
the position of the first, as the levels of `Arena.ofValues` do. The reference function gives the
value of a seed, so an unfold is an anamorphism (Meijer, Fokkinga and Paterson 1991) whose values
are stored as an arena instead of as linked cells.

The certificate `Unfold.Correct` is about one seed: if the calls of the seed have the reference
values `ys` and the node expressions succeed on its row, the reference value of the seed has the
children `ys` and the node of the row is its outermost layer. `Unfold.build_eq` lifts it to the
roots: the arena that `Unfold.build` gives is `Arena.ofValues` of the reference values of the
roots. `Unfold.results_eq` states that the results are the reference values for every capacity,
failures included: the values decoded from the built arena when the build succeeds, and the
reference values computed directly when it fails. `Unfold.results_fold` states that a `Fold` over
a built arena gives the results of the fold on those values, so an unfold that builds followed by
a fold computes their composition. Like the certificate of a `Hylo`, `Unfold.Correct` is about
partial correctness: it does not show that the build succeeds at any capacity.

On the device, `Unfold.buildDevice` builds the arena with `Expansion.buildHandle` and keeps it in
device memory as an `Arena.Device`, which `Fold.run` folds without copying it to the host. The
native build is trusted like `Expansion.expandHandle`, with the same assumption on the arithmetic of
the `Scalar` instance, and the tests compare the arenas it builds with those of `Unfold.build` bit
for bit. `Unfold.run` uses the device when LibTorch and a usable device are present, and
`Unfold.results` on the host otherwise or when the device gives up.
-/

@[expose] public section

namespace TorchLean.Kernel

open CUDA Sig.Inductive

variable {σ : Sig} {α : Type} {T : Fin σ.sorts → Type} [σ.Inductive α T]

/-- The certificate of an unfold of values of `σ` with the reference function `reference`, the
calls `calls`, `argIndex` and `argValue`, and the node expressions `tag`, `nodeIndex` and
`nodeScalar`: if the calls of a seed have the reference values `ys`, and the node expressions
succeed on the row of the seed with its first child at `first`, the reference value of the seed
has the children `ys`, and the node expressions give its constructor number and its slots, with
`0` and `default` past the slots of its constructor. -/
def Unfold.Correct (σ : Sig) {α : Type} (T : Fin σ.sorts → Type) [σ.Inductive α T] [Inhabited α]
    {m s : Nat} (arithmetic : Arithmetic α)
    (reference : Reader α → Env α (Slots m s) → Except Error (Σ i, T i))
    (calls : Expr α (Slots m s) .index) (argIndex : Fin m → Expr α (Row m s) .index)
    (argValue : Fin s → Expr α (Row m s) .scalar) (tag : Expr α (Row m s) .index)
    (nodeIndex : Fin σ.indexSlots → Expr α (Row m s) .index)
    (nodeScalar : Fin σ.scalarSlots → Expr α (Row m s) .scalar) : Prop :=
  ∀ (read : Reader α) (index : Nat → UInt64) (value : Nat → α) (first : Nat)
      (ys : List (Σ i, T i)),
    calls.eval arithmetic read (Slots.env index value) = .ok ys.length.toUInt64 →
    first + ys.length < UInt64.size →
    (∀ j (hj : j < ys.length), ∃ (index' : Nat → UInt64) (value' : Nat → α),
      (∀ i : Fin m,
        (argIndex i).eval arithmetic read (Row.env j.toUInt64 index value) = .ok (index' i)) ∧
      (∀ i : Fin s,
        (argValue i).eval arithmetic read (Row.env j.toUInt64 index value) = .ok (value' i)) ∧
      reference read (Slots.env index' value') = .ok ys[j]) →
    ∀ (tag' : UInt64) (index'' : Fin σ.indexSlots → UInt64) (value'' : Fin σ.scalarSlots → α),
      tag.eval arithmetic read (Row.env first.toUInt64 index value) = .ok tag' →
      (∀ i, (nodeIndex i).eval arithmetic read (Row.env first.toUInt64 index value) =
        .ok (index'' i)) →
      (∀ i, (nodeScalar i).eval arithmetic read (Row.env first.toUInt64 index value) =
        .ok (value'' i)) →
      ∃ x, reference read (Slots.env index value) = .ok x ∧ children x = ys ∧
        tag' = (view x.2).ctor.val.toUInt64 ∧
        (∀ i, index'' i = ((view x.2).args.indexSlots first).getD i 0) ∧
        ∀ i, value'' i = (view x.2).args.scalarSlots.getD i default

/-- An unfold of values of `σ` from seeds with `m` index slots and `s` scalar slots: a seed makes
the calls of its `Expansion`, one per child of its value, and the node expressions give the node
of the row of a seed from the position of its first child and its slots. -/
structure Unfold (σ : Sig) (α : Type) (T : Fin σ.sorts → Type) [σ.Inductive α T] [Inhabited α]
    (m s : Nat) extends Expansion α m s where
  /-- The reference function: the value of a seed with the given slots, with its sort. -/
  reference : Reader α → Env α (Slots m s) → Except Error (Σ i, T i)
  /-- The constructor number of the node of a row, from the position of its first child and its
  slots. -/
  tag : Expr α (Row m s) .index
  /-- Index column `i` of the node of a row. -/
  nodeIndex : Fin σ.indexSlots → Expr α (Row m s) .index
  /-- Scalar column `i` of the node of a row. -/
  nodeScalar : Fin σ.scalarSlots → Expr α (Row m s) .scalar
  /-- The certificate of the unfold. -/
  correct : Unfold.Correct σ T arithmetic reference calls argIndex argValue tag nodeIndex nodeScalar

namespace Unfold

/-- The blocks of `c 0, ..., c (n - 1)` entries of a list, each from the sum of the numbers
before it, make up the entries of the list before the sum of all `n` numbers. -/
theorem flatten_ofFn_take_drop {β : Type} (ys : List β) (c : Nat → Nat) : ∀ n : Nat,
    (List.ofFn fun k : Fin n => (ys.drop ((List.range k).map c).sum).take (c k)).flatten =
      ys.take ((List.range n).map c).sum
  | 0 => by simp
  | n + 1 => by
    rw [List.ofFn_succ_last, List.flatten_append]
    simp only [Fin.val_castSucc, Fin.val_last, List.flatten_cons, List.flatten_nil,
      List.append_nil]
    rw [flatten_ofFn_take_drop ys c n, List.range_succ, List.map_append, List.sum_append,
      List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, Nat.add_zero, List.take_add]

/-- The first child of row `k` of some values whose rows have `c 0, c 1, ...` children is at the
sum of the first `k` numbers. -/
theorem firstChild_ofFn {n : Nat} {x : Fin n → Σ i, T i} {c : Nat → Nat}
    (hc : ∀ k : Fin n, (children (x k)).length = c k) {k : Nat} (hk : k ≤ n) :
    firstChild (List.ofFn x) k = ((List.range k).map c).sum := by
  have h : (List.ofFn x).map (fun y => (children y).length) = (List.range n).map c := by
    apply List.ext_getElem (by simp)
    intro i h₁ h₂
    simp [hc]
  rw [firstChild, List.map_take, h, ← List.map_take, List.take_range, Nat.min_eq_left hk]

/-- Two arrays of columns are equal if they have as many columns, of `n` entries each, with the
same entries. -/
private theorem columns_eq {β : Type} {A B : Array (Array β)} {n : Nat} (hsize : A.size = B.size)
    (hA : ∀ i (hi : i < A.size), A[i].size = n) (hB : ∀ i (hi : i < B.size), B[i].size = n)
    (h : ∀ i k, i < A.size → k < n → A[i]?.bind (·[k]?) = B[i]?.bind (·[k]?)) : A = B := by
  refine Array.ext hsize fun i hi hi' => Array.ext (by rw [hA i hi, hB i hi']) fun k hk hk' => ?_
  have := h i k hi (by rw [← hA i hi]; exact hk)
  simpa [hi, hi', hk, hk'] using this

variable [Inhabited α] {m s : Nat}

/-- The values of `e` on the rows of a level, or `none` when one fails. -/
def column (U : Unfold σ α T m s) (read : Reader α) (level : Arena.Level α) {t : Ty}
    (e : Expr α (Row m s) t) : Option (Array (t.Value α)) :=
  (Array.ofFn fun k : Fin level.tag.size => e.eval U.arithmetic read (level.env m s k)).mapM
    Except.toOption

/-- The nodes of the rows of a level of the expansion, as a level of an arena: the values of
`tag`, `nodeIndex` and `nodeScalar` on every row, or `none` when one fails. -/
def nodeLevel (U : Unfold σ α T m s) (read : Reader α) (level : Arena.Level α) :
    Option (Arena.Level α) :=
  (U.column read level U.tag).bind fun tag =>
    ((Array.finRange σ.indexSlots).mapM fun i => U.column read level (U.nodeIndex i)).bind
      fun index =>
    ((Array.finRange σ.scalarSlots).mapM fun j => U.column read level (U.nodeScalar j)).map
      fun scalar => ⟨tag, index, scalar⟩

/-- The arena of the values of the roots, built on the host: the nodes of the levels of the
expansion, or `none` when the expansion or a node expression fails. -/
def build (U : Unfold σ α T m s) (read : Reader α) (capacity : Nat) (roots : Arena.Level α) :
    Option (Arena α) :=
  (U.expand read capacity roots).bind fun levels =>
    (levels.mapM (U.nodeLevel read)).map fun out => ⟨out.toArray⟩

/-- A column has one entry per row of its level. -/
theorem size_column {U : Unfold σ α T m s} {read : Reader α} {level : Arena.Level α} {t : Ty}
    {e : Expr α (Row m s) t} {col : Array (t.Value α)} (h : U.column read level e = some col) :
    col.size = level.tag.size := by
  simpa using Array.size_of_mapM_some h

/-- Row `k` of a column holds the value of its expression on row `k` of its level. -/
theorem eval_of_column {U : Unfold σ α T m s} {read : Reader α} {level : Arena.Level α}
    {t : Ty} {e : Expr α (Row m s) t} {col : Array (t.Value α)}
    (h : U.column read level e = some col) {k : Nat} (hk : k < level.tag.size) :
    ∃ v, col[k]? = some v ∧ e.eval U.arithmetic read (level.env m s k) = .ok v := by
  have hv := Array.getElem?_of_mapM_some h k (by simpa using hk)
  have hk' : k < col.size := by rw [size_column h]; exact hk
  rw [Array.getElem?_eq_getElem hk'] at hv
  simp only [Array.getElem_ofFn] at hv
  exact ⟨col[k], Array.getElem?_eq_getElem hk', Except.toOption_eq_some_iff.1 hv.symm⟩

/-- The columns of a family of expressions on a level: one column per expression, with one entry
per row, which holds the value of the expression on that row. -/
theorem columns_spec {U : Unfold σ α T m s} {read : Reader α} {level : Arena.Level α} {t : Ty}
    {a : Nat} {e : Fin a → Expr α (Row m s) t} {cols : Array (Array (t.Value α))}
    (h : (Array.finRange a).mapM (fun i => U.column read level (e i)) = some cols) :
    cols.size = a ∧ (∀ i (hi : i < cols.size), cols[i].size = level.tag.size) ∧
      ∀ (i : Fin a) {k : Nat}, k < level.tag.size → ∃ v, cols[i.1]?.bind (·[k]?) = some v ∧
        (e i).eval U.arithmetic read (level.env m s k) = .ok v := by
  have hsize : cols.size = a := by simpa using Array.size_of_mapM_some h
  have hcol : ∀ i (hi : i < cols.size), U.column read level (e ⟨i, hsize ▸ hi⟩) = some cols[i] := by
    intro i hi
    have := Array.getElem?_of_mapM_some h i (by simpa [hsize] using hi)
    rw [Array.getElem?_eq_getElem hi] at this
    simpa using this.symm
  refine ⟨hsize, fun i hi => size_column (hcol i hi), fun i k hk => ?_⟩
  have hi : i.1 < cols.size := by omega
  obtain ⟨v, hv, he⟩ := eval_of_column (hcol i hi) hk
  exact ⟨v, by simpa [Array.getElem?_eq_getElem hi] using hv, he⟩

/-- The level of the nodes of a level has one entry per row in each column, one index column per
index slot and one scalar column per scalar slot of the signature, and holds, in row `k`, the
values of `tag`, `nodeIndex` and `nodeScalar` on row `k`. -/
theorem nodeLevel_spec {U : Unfold σ α T m s} {read : Reader α} {level out : Arena.Level α}
    (h : U.nodeLevel read level = some out) :
    out.tag.size = level.tag.size ∧ out.index.size = σ.indexSlots ∧
      (∀ i (hi : i < out.index.size), out.index[i].size = level.tag.size) ∧
      out.scalar.size = σ.scalarSlots ∧
      (∀ i (hi : i < out.scalar.size), out.scalar[i].size = level.tag.size) ∧
      ∀ k, k < level.tag.size →
        (∃ v, out.tag[k]? = some v ∧ U.tag.eval U.arithmetic read (level.env m s k) = .ok v) ∧
        (∀ i : Fin σ.indexSlots, ∃ v, out.index[i.1]?.bind (·[k]?) = some v ∧
          (U.nodeIndex i).eval U.arithmetic read (level.env m s k) = .ok v) ∧
        ∀ i : Fin σ.scalarSlots, ∃ v, out.scalar[i.1]?.bind (·[k]?) = some v ∧
          (U.nodeScalar i).eval U.arithmetic read (level.env m s k) = .ok v := by
  simp only [nodeLevel, Option.bind_eq_some_iff, Option.map_eq_some_iff] at h
  obtain ⟨tag, htag, index, hindex, scalar, hscalar, rfl⟩ := h
  obtain ⟨his, hic, hie⟩ := columns_spec hindex
  obtain ⟨hss, hsc, hse⟩ := columns_spec hscalar
  exact ⟨size_column htag, his, hic, hss, hsc, fun k hk =>
    ⟨eval_of_column htag hk, fun i => hie i hk, fun i => hse i hk⟩⟩

/-- The level of the nodes of a round of the expansion is the level of the reference values of
the rows of its frontier, if the rows of the next level have the reference values `ys`, which are
then the children of those values. -/
theorem nodeLevel_round (U : Unfold σ α T m s) {read : Reader α} {budget : Nat}
    {frontier level next out : Arena.Level α}
    (hr : U.round read budget frontier = some (level, next))
    (hn : U.nodeLevel read level = some out) {ys : List (Σ i, T i)}
    (hys : ys.length = next.tag.size)
    (href : ∀ k (hk : k < ys.length), U.reference read (next.slots m s k) = .ok ys[k]) :
    ∃ xs : List (Σ i, T i), xs.length = frontier.tag.size ∧
      (∀ k (hk : k < xs.length), U.reference read (frontier.slots m s k) = .ok xs[k]) ∧
      xs.flatMap children = ys ∧ out = Arena.Level.encode xs := by
  obtain ⟨c, htag, hindex, hscalar, hsize, htotal, hcalls, hargs⟩ := Expansion.round_spec hr
  obtain ⟨hts, his, hic, hss, hsc, hrows⟩ := nodeLevel_spec hn
  have hlevel : level.tag.size = frontier.tag.size := by simp [htag]
  have hle : ∀ k, k < frontier.tag.size →
      ((List.range k).map c).sum + c k ≤ ((List.range frontier.tag.size).map c).sum := by
    intro k hk
    have := Expansion.sum_range_le c (show k + 1 ≤ frontier.tag.size by omega)
    simpa [List.range_succ] using this
  have hlen : ∀ k, k < frontier.tag.size →
      ((ys.drop ((List.range k).map c).sum).take (c k)).length = c k := by
    intro k hk
    have := hle k hk
    simp only [List.length_take, List.length_drop]
    omega
  have hrow : ∀ k (hk : k < frontier.tag.size), ∃ x : Σ i, T i,
      U.reference read (frontier.slots m s k) = .ok x ∧
      children x = (ys.drop ((List.range k).map c).sum).take (c k) ∧
      out.tag[k]? = some (view x.2).ctor.val.toUInt64 ∧
      (∀ i (hi : i < σ.indexSlots), out.index[i]?.bind (·[k]?) =
        some (((view x.2).args.indexSlots ((List.range k).map c).sum).getD i 0)) ∧
      ∀ i (hi : i < σ.scalarSlots), out.scalar[i]?.bind (·[k]?) =
        some ((view x.2).args.scalarSlots.getD i default) := by
    intro k hk
    have henv : level.env m s k = Row.env ((List.range k).map c).sum.toUInt64
        (fun i => (frontier.index[i]?.bind (·[k]?)).getD 0)
        (fun j => (frontier.scalar[j]?.bind (·[k]?)).getD default) := by
      simp [Arena.Level.env, htag, hindex, hscalar, hk]
    obtain ⟨⟨tag', htag', etag⟩, hidx, hsca⟩ := hrows k (by omega)
    choose vi hvi evi using hidx
    choose vs hvs evs using hsca
    rw [henv] at etag evi evs
    obtain ⟨x, hx, hch, htg, hix, hsx⟩ := U.correct read
      (fun i => (frontier.index[i]?.bind (·[k]?)).getD 0)
      (fun j => (frontier.scalar[j]?.bind (·[k]?)).getD default)
      ((List.range k).map c).sum ((ys.drop ((List.range k).map c).sum).take (c k))
      (by rw [hlen k hk]; exact hcalls k hk)
      (by have := hle k hk; rw [hlen k hk]; omega)
      (fun j hj => by
        rw [hlen k hk] at hj
        obtain ⟨hi, hv⟩ := hargs k j hk hj
        refine ⟨fun i => (next.index[i]?.bind (·[((List.range k).map c).sum + j]?)).getD 0,
          fun i => (next.scalar[i]?.bind (·[((List.range k).map c).sum + j]?)).getD default,
          hi, hv, ?_⟩
        have hp : ((List.range k).map c).sum + j < ys.length := by
          have := hle k hk
          omega
        simp only [List.getElem_take, List.getElem_drop]
        exact href _ hp)
      tag' vi vs etag evi evs
    exact ⟨x, hx, hch, by rw [htag', htg], fun i hi => by rw [hvi ⟨i, hi⟩, hix],
      fun i hi => by rw [hvs ⟨i, hi⟩, hsx]⟩
  choose x hx using hrow
  have hget : ∀ k (hk : k < frontier.tag.size),
      (List.ofFn fun k : Fin frontier.tag.size => x k.1 k.2)[k]? = some (x k hk) := by
    intro k hk
    simp [hk]
  have hfirst : ∀ k, k ≤ frontier.tag.size →
      firstChild (List.ofFn fun k : Fin frontier.tag.size => x k.1 k.2) k =
        ((List.range k).map c).sum :=
    fun k hk => firstChild_ofFn (fun k => by rw [(hx k.1 k.2).2.1, hlen k.1 k.2]) hk
  refine ⟨List.ofFn fun k : Fin frontier.tag.size => x k.1 k.2, by simp, fun k hk => ?_, ?_, ?_⟩
  · simp only [List.getElem_ofFn]
    exact (hx k _).1
  · have hmap : (List.ofFn fun k : Fin frontier.tag.size => x k.1 k.2).map children =
        List.ofFn fun k : Fin frontier.tag.size =>
          (ys.drop ((List.range k).map c).sum).take (c k) := by
      rw [List.map_ofFn]
      exact congrArg List.ofFn (funext fun k => (hx k.1 k.2).2.1)
    rw [List.flatMap_def, hmap, flatten_ofFn_take_drop, List.take_of_length_le (by omega)]
  · have etag : out.tag =
        (Arena.Level.encode (List.ofFn fun k : Fin frontier.tag.size => x k.1 k.2)).tag := by
      apply Array.ext_getElem?
      intro k
      rw [Arena.getElem?_tag]
      by_cases hk : k < frontier.tag.size
      · rw [(hx k hk).2.2.1, hget k hk]
        rfl
      · rw [Array.getElem?_eq_none (by omega), List.getElem?_eq_none (by simp; omega)]
        rfl
    have eindex : out.index =
        (Arena.Level.encode (List.ofFn fun k : Fin frontier.tag.size => x k.1 k.2)).index := by
      refine columns_eq (n := frontier.tag.size) (by simp [his]) (fun i hi => by
        rw [hic i hi, hlevel]) (fun i hi => by simp [Arena.size_getElem_index]) fun i k hi hk => ?_
      have hi' : i < σ.indexSlots := by omega
      rw [(hx k hk).2.2.2.1 i hi', Arena.index_encode_of_lt (hget k hk) hi', hfirst k (by omega)]
    have escalar : out.scalar =
        (Arena.Level.encode (List.ofFn fun k : Fin frontier.tag.size => x k.1 k.2)).scalar := by
      refine columns_eq (n := frontier.tag.size) (by simp [hss]) (fun i hi => by
        rw [hsc i hi, hlevel]) (fun i hi => by simp [Arena.size_getElem_scalar]) fun i k hi hk => ?_
      have hi' : i < σ.scalarSlots := by omega
      rw [(hx k hk).2.2.2.2 i hi', Arena.scalar_encode_of_lt (hget k hk) hi']
    calc out = ⟨out.tag, out.index, out.scalar⟩ := rfl
      _ = _ := by rw [etag, eindex, escalar]

/-- The nodes of the levels of an expansion from a frontier are the levels of the arena of the
reference values of its rows, with fewer than `2 ^ 64` nodes on every level below them. -/
theorem build_expandAux (U : Unfold σ α T m s) (read : Reader α) :
    ∀ (fuel budget : Nat) (frontier : Arena.Level α) {levels out : List (Arena.Level α)},
      U.expandAux read fuel budget frontier = some levels →
      levels.mapM (U.nodeLevel read) = some out →
      ∃ xs : List (Σ i, T i), xs.length = frontier.tag.size ∧
        (∀ k (hk : k < xs.length), U.reference read (frontier.slots m s k) = .ok xs[k]) ∧
        (∀ r, out[r]? = if nodes xs r = [] then none else some (Arena.Level.encode (nodes xs r))) ∧
        ∀ r, (nodes xs (r + 1)).length < UInt64.size
  | 0, _, _, _, _, h, _ => by simp [Expansion.expandAux] at h
  | fuel + 1, budget, frontier, levels, out, h, hout => by
    unfold Expansion.expandAux at h
    by_cases h0 : frontier.tag.size = 0
    · simp only [h0, ite_true, Option.some.injEq] at h
      subst h
      simp only [List.mapM_nil, pure, Option.some.injEq] at hout
      subst hout
      exact ⟨[], by simp [h0], fun k hk => absurd hk (by simp), fun r => by simp,
        fun r => by simp [UInt64.size]⟩
    simp only [h0, ite_false] at h
    cases hr : U.round read budget frontier with
    | none => rw [hr] at h; cases h
    | some p =>
      obtain ⟨level, next⟩ := p
      rw [hr] at h
      obtain ⟨rest, hrest, rfl⟩ := Option.map_eq_some_iff.1 h
      simp only [List.mapM_cons, Option.bind_eq_bind, Option.bind_eq_some_iff, pure,
        Option.some.injEq] at hout
      obtain ⟨nl, hnl, out', hout', rfl⟩ := hout
      obtain ⟨ys, hys, href, hlev, hbound⟩ := build_expandAux U read fuel _ next hrest hout'
      obtain ⟨c, -, -, -, hsize, htotal, -, -⟩ := Expansion.round_spec hr
      obtain ⟨xs, hxs, hxref, hch, hnl'⟩ := nodeLevel_round U hr hnl hys href
      refine ⟨xs, hxs, hxref, fun r => ?_, fun r => ?_⟩
      · cases r with
        | zero =>
          have hne : nodes xs 0 ≠ [] := by
            intro h
            simp only [nodes] at h
            simp [h] at hxs
            omega
          rw [List.getElem?_cons_zero, ite_eq_right hne, hnl']
          rfl
        | succ r => rw [List.getElem?_cons_succ, hlev r, nodes_succ', hch]
      · cases r with
        | zero =>
          simp only [nodes, hch]
          omega
        | succ r =>
          rw [nodes_succ', hch]
          exact hbound r

/-- A built arena is the arena of the reference values of the roots, which have fewer than
`2 ^ 64` nodes on every level below them. -/
theorem build_eq (U : Unfold σ α T m s) (read : Reader α) {capacity : Nat}
    {roots : Arena.Level α} {a : Arena α} (h : U.build read capacity roots = some a) :
    ∃ xs : List (Σ i, T i), xs.length = roots.tag.size ∧
      (∀ k (hk : k < xs.length), U.reference read (roots.slots m s k) = .ok xs[k]) ∧
      (∀ r, (nodes xs (r + 1)).length < UInt64.size) ∧ a = Arena.ofValues xs := by
  obtain ⟨levels, hl, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨out, hout, rfl⟩ := Option.map_eq_some_iff.1 h
  unfold Expansion.expand at hl
  split at hl
  · obtain ⟨xs, hxs, href, hlev, hbound⟩ := build_expandAux U read _ _ _ hl hout
    refine ⟨xs, hxs, href, hbound, ?_⟩
    have hlevels : out.toArray = (Arena.ofValues xs).levels := by
      apply Array.ext_getElem?
      intro r
      rw [List.getElem?_toArray, hlev r, Arena.getElem?_ofValues]
    rw [hlevels]
  · cases hl

/-- The values of the rows of level zero of an arena, decoded, with the reference value of root
`k` in place of row `k` when it does not decode. -/
def decodeRoots (U : Unfold σ α T m s) (read : Reader α) (roots : Arena.Level α) (a : Arena α) :
    Array (Except Error (Σ i, T i)) :=
  Array.ofFn fun k : Fin roots.tag.size =>
    match a.decodeRow T a.levels.size 0 k with
    | some x => .ok x
    | none => U.reference read (roots.slots m s k)

/-- The values of the roots: decoded from the built arena, or the reference values when the build
fails. -/
def results (U : Unfold σ α T m s) (read : Reader α) (capacity : Nat) (roots : Arena.Level α) :
    Array (Except Error (Σ i, T i)) :=
  match U.build read capacity roots with
  | some a => U.decodeRoots read roots a
  | none => Array.ofFn fun k : Fin roots.tag.size => U.reference read (roots.slots m s k)

/-- The roots of a built arena decode to the reference values of the roots. -/
theorem decodeRoots_build (U : Unfold σ α T m s) (read : Reader α) {capacity : Nat}
    {roots : Arena.Level α} {a : Arena α} (h : U.build read capacity roots = some a) :
    U.decodeRoots read roots a =
      Array.ofFn fun k : Fin roots.tag.size => U.reference read (roots.slots m s k) := by
  obtain ⟨xs, hxs, href, hbound, rfl⟩ := build_eq U read h
  unfold decodeRoots
  congr 1
  funext k
  have hk : k.1 < xs.length := by omega
  rw [Arena.decodeRow_ofValues hbound (List.getElem?_eq_getElem hk), href k hk]

/-- The values of the roots are their reference values, for every capacity. -/
theorem results_eq (U : Unfold σ α T m s) (read : Reader α) (capacity : Nat)
    (roots : Arena.Level α) :
    U.results read capacity roots =
      Array.ofFn fun k : Fin roots.tag.size => U.reference read (roots.slots m s k) := by
  unfold results
  cases h : U.build read capacity roots with
  | none => rfl
  | some a => exact decodeRoots_build U read h

/-- A fold over a built arena gives, on root `k`, the result of the fold on the reference value of
root `k`. -/
theorem results_fold (U : Unfold σ α T m s) (F : Fold σ α T) (read read' : Reader α)
    {capacity : Nat} {roots : Arena.Level α} {a : Arena α}
    (h : U.build read capacity roots = some a) {k : Nat} (hk : k < roots.tag.size) :
    (F.results read' a 0)[k]? =
      some ((U.reference read (roots.slots m s k)).bind fun x => F.reference read' x.1 x.2) := by
  obtain ⟨xs, hxs, href, hbound, rfl⟩ := build_eq U read h
  have hk' : k < xs.length := by omega
  rw [href k hk']
  exact F.results_of_decodeAt read' _
    (Arena.decodeAt_ofValues hbound (Arena.ofValues xs).levels.size (r := 0)
      (List.getElem?_eq_getElem hk') (by omega))

/-- The arena of the values of the roots `roots`, built on the device by
`Expansion.buildHandle` with the row kernels of the node expressions and the arrays `inputs` as
the inputs, and kept in device memory. Returns `none` when a read failed, or when the levels would
hold more than `capacity` rows or more than `Expansion.deviceCapacity`. -/
@[no_expose] def buildDevice (U : Unfold σ α T m s) (capacity : Nat) (roots : Arena.Level α)
    (inputs : Array (Array α)) : IO (Option (@Arena.Device α U.scalar)) :=
  letI := U.scalar
  do
    let some (handle, sizes) ← U.buildHandle capacity roots inputs U.tag U.nodeIndex U.nodeScalar
      | return none
    return some (Arena.Device.ofHandle handle sizes σ.indexSlots σ.scalarSlots)

/-- The values of the roots `roots`, with the arrays `inputs` as the inputs. With LibTorch and a
usable device they are decoded from the arena of `Unfold.buildDevice` copied to the host.
Otherwise, or when the device gives `none`, they are `Unfold.results` on the host at the capacity
`min capacity Expansion.deviceCapacity`, which are the reference values by `Unfold.results_eq`.
On the device it fails as `Unfold.buildDevice` does, and when a root of the device arena does not
decode. -/
@[no_expose] def run (U : Unfold σ α T m s) (capacity : Nat) (roots : Arena.Level α)
    (inputs : Array (Array α)) : IO (Array (Except Error (Σ i, T i))) := do
  if (← LibTorch.Buffer.RuntimeStatus.query) = .nativeAvailable then
    if let some d ← U.buildDevice capacity roots inputs then
      let a ← @Arena.Device.download α U.scalar d
      let some xs := (Array.range roots.tag.size).mapM fun k => a.decodeRow T a.levels.size 0 k
        | throw <| IO.userError "tree unfold: a root of the device arena does not decode"
      return xs.map .ok
  return U.results (Reader.ofArrays inputs) (min capacity Expansion.deviceCapacity) roots

end Unfold

end TorchLean.Kernel
