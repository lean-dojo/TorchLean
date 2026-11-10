/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Tree.Run

/-!
# Expansions of calls

An `Expansion` describes the calls of a recursive function over rows of slots: a call with the
slots of a row makes `calls` calls, and call `j` has the slots `argIndex` and `argValue` at `j`.
`Expansion.expand` unfolds the calls from the roots, one level per depth of the recursion: the
calls of row `k` are consecutive rows of the next level from the sum of the numbers of calls of the
rows before it, and that position is the tag of row `k`. Every level is built by one data-parallel
pass and a prefix sum of the numbers of calls (Blelloch 1990), as frontiers of graph searches are
expanded on GPUs (Merrill, Garland and Grimshaw 2012). The expansion gives `none` when the levels
would hold more than `capacity` rows or a number of calls or a slot of a call fails.
`Expansion.round_spec` states what one round computes, which is all that the hylomorphisms of
`Tree.Hylo` and the unfolds of `Tree.Unfold` use.

On the device, `Expansion.expandHandle` builds the levels through the private `@[extern]` entry
point `Arena.Handle.expand` of `csrc/libtorch/tree.cpp`: for each level, one launch of the row
kernel of `calls` over the rows, a prefix sum of the numbers of calls, and one launch of the row
kernel of each slot over the calls. Each kernel is compiled at its first launch and held for the
later levels. `Expansion.buildHandle` launches row kernels of node columns on every level as well
and keeps their columns in place of the slots, which is how an unfold builds the arena of its
values. The native expansion is trusted like the fold launches of `Step.runHandle`: the kernels are
assumed to compute with the arithmetic of the `Scalar` instance (`Expansion.native`), and the
tests compare the levels with those of `Expansion.expand` bit for bit.
-/

@[expose] public section

namespace TorchLean.Kernel

open CUDA

variable {α : Type}

/-- The slots of row `k` of a level with `m` index slots and `s` scalar slots, with defaults past
the end of a column. -/
def Arena.Level.slots [Inhabited α] (m s : Nat) (level : Arena.Level α) (k : Nat) :
    Env α (Slots m s) :=
  Slots.env (fun i => (level.index[i]?.bind (·[k]?)).getD 0)
    (fun j => (level.scalar[j]?.bind (·[k]?)).getD default)

/-- The calls of a recursive function over rows with `m` index slots and `s` scalar slots: a call
with the slots of a row makes `calls` calls, and call `j` has the slots `argIndex` and `argValue`
at `j`. -/
structure Expansion (α : Type) (m s : Nat) where
  /-- The scalar type with its device precision. -/
  scalar : Scalar α
  /-- The arithmetic of the expressions. -/
  arithmetic : Arithmetic α
  /-- The number of inputs that the expressions read. -/
  inputs : Nat
  /-- The number of calls that a call makes, from its slots. -/
  calls : Expr α (Slots m s) .index
  /-- Index slot `i` of call `j`, from `j` and the slots of the caller. -/
  argIndex : Fin m → Expr α (Row m s) .index
  /-- Scalar slot `i` of call `j`, from `j` and the slots of the caller. -/
  argValue : Fin s → Expr α (Row m s) .scalar
  /-- The arithmetic is the device arithmetic of the scalar type, which the device kernels are
  assumed to compute, as every kernel of `Kernel.Launch` is. -/
  native : arithmetic = scalar.arithmetic

namespace Expansion

/-- The exclusive prefix sums of the numbers `c 0, ..., c (n - 1)`, with their total. -/
def offsets (c : Nat → Nat) (n : Nat) : Array Nat × Nat :=
  Nat.fold n (fun k _ acc => (acc.1.push acc.2, acc.2 + c k)) (#[], 0)

/-- Entry `k` of the prefix sums is the sum of the numbers before `k`. -/
theorem offsets_eq (c : Nat → Nat) : ∀ n : Nat,
    offsets c n = ((Array.range n).map fun k => ((List.range k).map c).sum,
      ((List.range n).map c).sum)
  | 0 => by simp [offsets]
  | n + 1 => by
    have h : offsets c (n + 1) =
        ((offsets c n).1.push (offsets c n).2, (offsets c n).2 + c n) := by
      simp only [offsets, Nat.fold_succ]
    rw [h, offsets_eq c n, Array.range_succ, Array.map_append, List.range_succ,
      List.map_append, List.sum_append]
    simp

/-- The entries `f k j` for `k < n` and `j < c k`, in this order. -/
def spread {β : Type} (c : Nat → Nat) (f : Nat → Nat → β) (n : Nat) : Array β :=
  (Array.range n).flatMap fun k => (Array.range (c k)).map (f k)

/-- A spread over `n + 1` rows ends with the block of row `n`. -/
theorem spread_succ {β : Type} (c : Nat → Nat) (f : Nat → Nat → β) (n : Nat) :
    spread c f (n + 1) = spread c f n ++ (Array.range (c n)).map (f n) := by
  simp [spread, Array.range_succ]

/-- A spread has as many entries as the numbers add up to. -/
theorem size_spread {β : Type} (c : Nat → Nat) (f : Nat → Nat → β) :
    ∀ n, (spread c f n).size = ((List.range n).map c).sum
  | 0 => by simp [spread, (Array.range_eq_empty_iff (n := 0)).2 rfl]
  | n + 1 => by
    rw [spread_succ, Array.size_append, size_spread c f n, List.range_succ,
      List.map_append, List.sum_append]
    simp

/-- The sum of the first `k` numbers is at most the sum of the first `n` when `k ≤ n`. -/
theorem sum_range_le (c : Nat → Nat) {k n : Nat} (h : k ≤ n) :
    ((List.range k).map c).sum ≤ ((List.range n).map c).sum :=
  ((List.range_sublist.2 h).map c).sum_le_sum fun _ _ => Nat.zero_le _

/-- Entry `j` of the block `k` of a spread is at the sum of the sizes of the blocks before. -/
theorem getElem?_spread {β : Type} (c : Nat → Nat) (f : Nat → Nat → β) :
    ∀ {n k j : Nat}, k < n → j < c k →
      (spread c f n)[((List.range k).map c).sum + j]? = some (f k j)
  | 0, _, _, hk, _ => absurd hk (Nat.not_lt_zero _)
  | n + 1, k, j, hk, hj => by
    rw [spread_succ]
    by_cases hkn : k < n
    · have hle : ((List.range (k + 1)).map c).sum ≤ ((List.range n).map c).sum :=
        sum_range_le c hkn
      rw [List.range_succ, List.map_append, List.sum_append] at hle
      rw [Array.getElem?_append_left (by rw [size_spread]; simp at hle; omega)]
      exact getElem?_spread c f hkn hj
    · have hk' : k = n := by omega
      subst hk'
      rw [Array.getElem?_append_right (by rw [size_spread]; omega), size_spread]
      simp [hj]

variable {m s : Nat} [Inhabited α]

/-- The numbers of calls of the rows of a frontier, or `none` when one fails. -/
def counts (E : Expansion α m s) (read : Reader α) (frontier : Arena.Level α) :
    Option (Array UInt64) :=
  (Array.ofFn fun k : Fin frontier.tag.size =>
    E.calls.eval E.arithmetic read (frontier.slots m s k)).mapM Except.toOption

/-- The values of `e` on the calls of a frontier whose row `k` makes `c k` calls, or `none` when
one fails: entry `j` of the block of row `k` evaluates `e` with `j` and the slots of row `k`. -/
def children (E : Expansion α m s) (read : Reader α) (frontier : Arena.Level α) (c : Nat → Nat)
    {t : Ty} (e : Expr α (Row m s) t) : Option (Array (t.Value α)) :=
  (spread c (fun k j => e.eval E.arithmetic read
    (Env.push (t := .index) j.toUInt64 (frontier.slots m s k))) frontier.tag.size).mapM
    Except.toOption

/-- One round of the expansion: the frontier with the positions of the first calls of its rows as
tags, and the level of all calls, or `none` when a number of calls or a slot of a call fails, or
when the calls are more than `budget`. -/
def round (E : Expansion α m s) (read : Reader α) (budget : Nat) (frontier : Arena.Level α) :
    Option (Arena.Level α × Arena.Level α) :=
  (E.counts read frontier).bind fun counts =>
    let c := fun k => (counts[k]?.getD 0).toNat
    let o := offsets c frontier.tag.size
    if o.2 ≤ budget ∧ o.2 < UInt64.size then
      ((Array.finRange m).mapM fun i => E.children read frontier c (E.argIndex i)).bind
        fun index =>
      ((Array.finRange s).mapM fun j => E.children read frontier c (E.argValue j)).bind
        fun scalar =>
      some (⟨o.1.map (·.toUInt64), frontier.index, frontier.scalar⟩,
        ⟨Array.replicate o.2 0, index, scalar⟩)
    else none

/-- The levels of the calls from a frontier on, with at most `budget` rows after the frontier and
at most `fuel` rounds. -/
def expandAux (E : Expansion α m s) (read : Reader α) :
    Nat → Nat → Arena.Level α → Option (List (Arena.Level α))
  | 0, _, _ => none
  | fuel + 1, budget, frontier =>
    if frontier.tag.size = 0 then some []
    else match E.round read budget frontier with
      | none => none
      | some (level, next) =>
        (E.expandAux read fuel (budget - next.tag.size) next).map (level :: ·)

/-- The levels of the calls from the roots on, or `none` when they would hold more than `capacity`
rows or a number of calls or a slot of a call fails. Every round adds a row, so `capacity + 1`
rounds are enough. -/
def expand (E : Expansion α m s) (read : Reader α) (capacity : Nat) (roots : Arena.Level α) :
    Option (List (Arena.Level α)) :=
  if roots.tag.size ≤ capacity then
    E.expandAux read (capacity + 1) (capacity - roots.tag.size) roots
  else none

/-- The column `e` of the calls of a frontier: entry `first + j` is the value of `e` on call `j`
of the row whose calls start at `first`. -/
theorem children_spec {E : Expansion α m s} {read : Reader α} {frontier : Arena.Level α}
    {c : Nat → Nat} {t : Ty} {e : Expr α (Row m s) t} {column : Array (t.Value α)}
    (h : E.children read frontier c e = some column) {k j : Nat} (hk : k < frontier.tag.size)
    (hj : j < c k) :
    ∃ hp : ((List.range k).map c).sum + j < column.size,
      e.eval E.arithmetic read (Env.push (t := .index) j.toUInt64 (frontier.slots m s k)) =
        .ok column[((List.range k).map c).sum + j] := by
  have hsize := Array.size_of_mapM_some h
  have hget := getElem?_spread c (fun k j => e.eval E.arithmetic read
    (Env.push (t := .index) j.toUInt64 (frontier.slots m s k))) hk hj
  obtain ⟨hp, hval⟩ := Array.getElem?_eq_some_iff.1 hget
  have hp' : ((List.range k).map c).sum + j < column.size := by omega
  refine ⟨hp', ?_⟩
  have hv := Array.getElem?_of_mapM_some h _ hp
  rw [hval, Array.getElem?_eq_getElem hp'] at hv
  exact Except.toOption_eq_some_iff.1 hv.symm

/-- A round tags the frontier with the positions of the first calls of its rows, and the next
level holds the slots of the calls of row `k` from that position on. -/
theorem round_spec {E : Expansion α m s} {read : Reader α} {budget : Nat}
    {frontier level next : Arena.Level α} (h : E.round read budget frontier = some (level, next)) :
    ∃ c : Nat → Nat,
      level.tag =
        (Array.range frontier.tag.size).map (fun k => ((List.range k).map c).sum.toUInt64) ∧
      level.index = frontier.index ∧ level.scalar = frontier.scalar ∧
      next.tag.size = ((List.range frontier.tag.size).map c).sum ∧
      ((List.range frontier.tag.size).map c).sum < UInt64.size ∧
      (∀ k, k < frontier.tag.size →
        E.calls.eval E.arithmetic read (frontier.slots m s k) = .ok (c k).toUInt64) ∧
      (∀ k j, k < frontier.tag.size → j < c k →
        (∀ i : Fin m, (E.argIndex i).eval E.arithmetic read
          (Env.push (t := .index) j.toUInt64 (frontier.slots m s k)) =
            .ok ((next.index[i]?.bind (·[((List.range k).map c).sum + j]?)).getD 0)) ∧
        (∀ i : Fin s, (E.argValue i).eval E.arithmetic read
          (Env.push (t := .index) j.toUInt64 (frontier.slots m s k)) =
            .ok ((next.scalar[i]?.bind (·[((List.range k).map c).sum + j]?)).getD default))) := by
  simp only [round] at h
  obtain ⟨counts, hcounts, h⟩ := Option.bind_eq_some_iff.1 h
  split at h
  case isFalse => cases h
  case isTrue hbound =>
  obtain ⟨index, hindex, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨scalar, hscalar, h⟩ := Option.bind_eq_some_iff.1 h
  cases h
  refine ⟨fun k => (counts[k]?.getD 0).toNat, ?_, rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simp [offsets_eq]
  · simp [offsets_eq]
  · simpa [offsets_eq] using hbound.2
  · intro k hk
    have hs := Array.size_of_mapM_some hcounts
    have hk' : k < counts.size := by simp at hs; omega
    have hv := Array.getElem?_of_mapM_some hcounts k (by simpa using hk)
    rw [Array.getElem?_eq_getElem hk'] at hv
    have := Except.toOption_eq_some_iff.1 hv.symm
    simp only [Array.getElem_ofFn] at this
    rw [this]
    simp [hk']
  · intro k j hk hj
    constructor
    · intro i
      have hs := Array.size_of_mapM_some hindex
      have hi : i.1 < index.size := by simp at hs; omega
      have hv := Array.getElem?_of_mapM_some hindex i.1 (by simp)
      rw [Array.getElem?_eq_getElem hi] at hv
      simp only [Array.getElem_finRange, Fin.cast_mk, Fin.eta] at hv
      obtain ⟨hp, hval⟩ := children_spec hv.symm hk hj
      rw [hval]
      simp [hi, hp]
    · intro i
      have hs := Array.size_of_mapM_some hscalar
      have hi : i.1 < scalar.size := by simp at hs; omega
      have hv := Array.getElem?_of_mapM_some hscalar i.1 (by simp)
      rw [Array.getElem?_eq_getElem hi] at hv
      simp only [Array.getElem_finRange, Fin.cast_mk, Fin.eta] at hv
      obtain ⟨hp, hval⟩ := children_spec hv.symm hk hj
      rw [hval]
      simp [hi, hp]

end Expansion

/-- The levels of the calls from `count` roots, unfolded on the device. The roots are the blocks
`words` and `scalars` of a native handle, whose tags are not read. For each level, the row kernel
`calls` gives the number of calls of every row from its slots, a prefix sum of the numbers gives
the position of the first call of every row, which becomes its tag, and the row kernel
`arguments[p]` gives slot `p` of every call from its number within its row and the slots of the
row: the index slots first, then the scalar slots. The blocks `inputs` are the first inputs of
every kernel. Without node kernels, returns the handle of all levels, root level first, and their
sizes. With node kernels, the row kernel `nodes[p]` gives column `p` of the node of every row from
the position of its first call and its slots: the tag, then `nodeIndexSlots` index columns and
`nodeScalarSlots` scalar columns, and the handle holds these columns instead. Returns `none` when
a read failed, or when the levels would hold more than `capacity` rows. The runtime refuses a
capacity of `2 ^ 31` or more, and blocks or kernels that do not match the slots. -/
@[never_extract, extern "torchlean_tree_expand"]
private opaque Arena.Handle.expand (format : UInt32) (descriptor : @& String)
    (width indexSlots scalarSlots count : UInt64) (words scalars : @& ByteArray)
    (calls : @& String) (arguments nodes : @& Array String)
    (nodeIndexSlots nodeScalarSlots : UInt64) (inputs : @& Array ByteArray)
    (capacity : UInt64) : IO (Option (Arena.Handle × Array UInt64))

namespace Expansion

/-- The largest number of rows of the levels of a device expansion, whose positions the runtime
keeps in 64-bit signed integers and launches over in grids of at most `2 ^ 31` threads. -/
def deviceCapacity : Nat := 2 ^ 31 - 1

variable {m s : Nat} [Inhabited α]

/-- The device expansion from the roots `roots` with the input blocks `blocks` and the rendered
node kernels `nodes` of `nodeIndexSlots` index and `nodeScalarSlots` scalar columns, none for the
levels of the calls. -/
private def launch (E : Expansion α m s) (capacity : Nat) (roots : Arena.Level α)
    (blocks : Array ByteArray) (nodes : Array String) (nodeIndexSlots nodeScalarSlots : Nat) :
    IO (Option (Arena.Handle × Array Nat)) :=
  letI := E.scalar
  do
    let count := roots.tag.size
    unless roots.index.size == m && roots.scalar.size == s &&
        roots.index.all (·.size == count) && roots.scalar.all (·.size == count) do
      throw <| IO.userError "tree expand: the roots do not have the slots of the expansion"
    let precision := E.scalar.precision
    unless precision.supportsGPU do
      throw <| IO.userError "tree expand: the device has no arithmetic for the scalar format"
    let width := precision.bytes
    unless Numeric.Word.width α ≤ 8 * width do
      throw <| IO.userError "tree expand: the scalar words are wider than their storage"
    let calls ← IO.ofExcept (Step.rowSource E.scalar E.inputs E.calls)
    let index ← IO.ofExcept ((List.finRange m).mapM fun i =>
      Step.rowSource E.scalar E.inputs (E.argIndex i))
    let scalar ← IO.ofExcept ((List.finRange s).mapM fun j =>
      Step.rowSource E.scalar E.inputs (E.argValue j))
    let c : Arena.Columns α := ⟨#[count], roots.tag, roots.index, roots.scalar⟩
    let some (handle, sizes) ← Arena.Handle.expand precision.tag precision.descriptor
        width.toUInt64 m.toUInt64 s.toUInt64 count.toUInt64 c.words (c.scalars width) calls
        (index ++ scalar).toArray nodes nodeIndexSlots.toUInt64 nodeScalarSlots.toUInt64 blocks
        (min capacity deviceCapacity).toUInt64
      | return none
    return some (handle, sizes.map (·.toNat))

/-- The levels of the calls from the roots `roots`, unfolded on the device by `Arena.Handle.expand`
with the row kernels of `calls` and of the slots and the blocks `blocks` of `Step.inputBlocks` as
the inputs, as a native handle and the level sizes. Returns `none` when a read failed, or when the
levels would hold more than `capacity` rows, or more than `deviceCapacity`. Fails when a source
does not render or compile, when the device has no arithmetic for the scalars, or when the roots
do not have `m` index and `s` scalar columns of one entry per row. -/
@[no_expose] def expandBlocks (E : Expansion α m s) (capacity : Nat) (roots : Arena.Level α)
    (blocks : Array ByteArray) : IO (Option (Arena.Handle × Array Nat)) :=
  E.launch capacity roots blocks #[] 0 0

/-- The levels of the calls from the roots `roots`, unfolded on the device by `Arena.Handle.expand`
with the row kernels of `calls` and of the slots and the arrays `inputs` as the inputs
(`Expansion.expandBlocks`), as a native handle and the level sizes. Returns `none` when a read
failed, or when the levels would hold more than `capacity` rows, or more than `deviceCapacity`.
Fails as `Expansion.expandBlocks` does. -/
@[no_expose] def expandHandle (E : Expansion α m s) (capacity : Nat) (roots : Arena.Level α)
    (inputs : Array (Array α)) : IO (Option (Arena.Handle × Array Nat)) :=
  E.expandBlocks capacity roots (Step.inputBlocks E.scalar E.inputs inputs)

/-- The levels of `Expansion.expandHandle` copied back to the host, which the tests compare with
`Expansion.expand` at the capacity `min capacity deviceCapacity`. -/
@[no_expose] def expandDevice (E : Expansion α m s) (capacity : Nat) (roots : Arena.Level α)
    (inputs : Array (Array α)) : IO (Option (List (Arena.Level α))) :=
  letI := E.scalar
  do
    let some (handle, sizes) ← E.expandHandle capacity roots inputs | return none
    let precision := E.scalar.precision
    let width := precision.bytes
    let words ← handle.words precision.tag precision.descriptor width.toUInt64
    let scalars ← handle.scalars precision.tag precision.descriptor width.toUInt64
    let count := sizes.foldl (· + ·) 0
    unless words.size = 8 * (m + 1) * count ∧ scalars.size = width * s * count do
      throw <| IO.userError "tree expand: the device returned blocks of the wrong size"
    let a := Arena.ofColumns (Arena.Columns.ofBlocks sizes count m s width words scalars)
    return some a.levels.toList

/-- The nodes of the levels of the calls from the roots `roots`, built on the device by
`Arena.Handle.expand`: on every level, the row kernels of `tag`, `nodeIndex` and `nodeScalar` give
the columns of the node of every row from the position of its first call and its slots. Returns
the native handle of the node columns of all levels and the level sizes, or `none` as
`Expansion.expandHandle` does, also when a read of a node kernel failed. -/
@[no_expose] def buildHandle (E : Expansion α m s) {a b : Nat} (capacity : Nat)
    (roots : Arena.Level α) (inputs : Array (Array α)) (tag : Expr α (Row m s) .index)
    (nodeIndex : Fin a → Expr α (Row m s) .index) (nodeScalar : Fin b → Expr α (Row m s) .scalar) :
    IO (Option (Arena.Handle × Array Nat)) := do
  let tag ← IO.ofExcept (Step.rowSource E.scalar E.inputs tag)
  let index ← IO.ofExcept ((List.finRange a).mapM fun i =>
    Step.rowSource E.scalar E.inputs (nodeIndex i))
  let scalar ← IO.ofExcept ((List.finRange b).mapM fun j =>
    Step.rowSource E.scalar E.inputs (nodeScalar j))
  E.launch capacity roots (Step.inputBlocks E.scalar E.inputs inputs)
    (#[tag] ++ (index ++ scalar).toArray) a b

end Expansion

end TorchLean.Kernel
