/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Algebra.Order.Group.Nat
public import TorchLean.CUDA.Tree.Sig

/-!
# Leveled arenas

An `Arena` stores a value of an inductive family by levels: the root alone on level zero, and on
level `r + 1` the children of the rows of level `r`, in row order. A row holds its constructor tag,
its index slots and its scalar slots. A child field holds the row of the child on the next level,
and a list field the row of its first child and its length. Each level stores its slots by column,
which is the layout the device kernels read.

`Arena.encode` builds the arena of a value and `Arena.decode` reads a value back. The main theorem,
`Arena.decode_encode`, says that decoding the arena of a value gives the value back whenever the
value has fewer than `2 ^ 64` nodes. `Arena.map` changes the type of the scalars, so that the
arena of a type without scalar fields, whose scalars are `Unit`, can live in any scalar storage.
-/

@[expose] public section

namespace TorchLean.Kernel

/-- One level of an arena: the constructor tag of each row, and its index and scalar slots stored
by column, so that column `j` holds slot `j` of every row. -/
structure Arena.Level (α : Type) where
  /-- The constructor number of each row. -/
  tag : Array UInt64
  /-- The index slot columns. -/
  index : Array (Array UInt64)
  /-- The scalar slot columns. -/
  scalar : Array (Array α)
  deriving Inhabited

/-- An arena: an array of levels. In the arena of a value (`Arena.encode`) the value is alone on
level zero and the children of the rows of each level are on the next level, in row order. -/
structure Arena (α : Type) where
  /-- The levels, from the root down. -/
  levels : Array (Arena.Level α)
  deriving Inhabited

namespace Sig.Inductive

variable {σ : Sig} {α : Type} {T : Fin σ.sorts → Type} [σ.Inductive α T]

/-- The children of a value, each with its sort. -/
def children (x : Σ s, T s) : List (Σ s, T s) := (view x.2).children

/-- The total size of some values. -/
def weight (xs : List (Σ s, T s)) : Nat := (xs.map fun x => size x.2).sum

/-- Going down one level lowers the total size by at least the number of values. -/
theorem weight_flatMap_add_length_le (xs : List (Σ s, T s)) :
    weight (xs.flatMap children) + xs.length ≤ weight xs := by
  induction xs with
  | nil => simp [weight]
  | cons x xs ih =>
    have h := size_lt x.2
    simp only [weight, children, Layer.children, List.flatMap_cons, List.map_append,
      List.sum_append, List.map_cons, List.sum_cons, List.length_cons] at ih h ⊢
    omega

/-- The number of nodes of a value: the value itself and the nodes of its children. -/
def count {s : Fin σ.sorts} (t : T s) : Nat :=
  1 + ((view t).children.attach.map fun c => count c.1.2).sum
termination_by size t
decreasing_by
  have h₁ : size c.1.2 ≤ ((view t).children.map fun c => size c.2).sum :=
    List.le_sum_of_mem (List.mem_map.2 ⟨c.1, c.2, rfl⟩)
  have h₂ := size_lt t
  omega

/-- A value has one node more than its children have in total. -/
theorem count_eq (x : Σ s, T s) : count x.2 = ((children x).map fun c => count c.2).sum + 1 := by
  rw [count, List.attach_map_val (f := fun c : Σ s, T s => count c.2), children]
  omega

/-- The values on level `r` below some values: the values themselves on level zero, and the
children of the values on level `r`, in order, on level `r + 1`. -/
def nodes (xs : List (Σ s, T s)) : Nat → List (Σ s, T s)
  | 0 => xs
  | r + 1 => (nodes xs r).flatMap children

/-- Level `r + 1` below some values is level `r` below their children. -/
theorem nodes_succ' (xs : List (Σ s, T s)) (r : Nat) :
    nodes xs (r + 1) = nodes (xs.flatMap children) r := by
  induction r with
  | zero => rfl
  | succ r ih => rw [nodes, ih, nodes]

/-- There are no levels below no values. -/
@[simp] theorem nodes_nil (r : Nat) : nodes ([] : List (Σ s, T s)) r = [] := by
  induction r with
  | zero => rfl
  | succ r ih => simp [nodes, ih]

/-- Going down one level lowers the total node count by the number of values. -/
theorem sum_count_flatMap (xs : List (Σ s, T s)) :
    ((xs.flatMap children).map fun x => count x.2).sum + xs.length =
      (xs.map fun x => count x.2).sum := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
    simp only [List.flatMap_cons, List.map_append, List.sum_append, List.map_cons, List.sum_cons,
      List.length_cons]
    rw [count_eq x]
    omega

/-- Every level below some values has at most as many values as they have nodes. -/
theorem length_nodes_le (xs : List (Σ s, T s)) (r : Nat) :
    (nodes xs r).length ≤ (xs.map fun x => count x.2).sum := by
  induction r generalizing xs with
  | zero =>
    have := sum_count_flatMap xs
    simp only [nodes]
    omega
  | succ r ih =>
    rw [nodes_succ']
    have := ih (xs.flatMap children)
    have := sum_count_flatMap xs
    omega

/-- The row of the next level where the children of row `k` start. -/
def firstChild (xs : List (Σ s, T s)) (k : Nat) : Nat :=
  ((xs.take k).map fun x => (children x).length).sum

/-- The next level splits at the children of row `k`. -/
theorem flatMap_children_eq {xs : List (Σ s, T s)} {k : Nat} {x : Σ s, T s}
    (h : xs[k]? = some x) :
    xs.flatMap children =
      (xs.take k).flatMap children ++ (children x ++ (xs.drop (k + 1)).flatMap children) := by
  obtain ⟨hk, hx⟩ := List.getElem?_eq_some_iff.1 h
  conv_lhs => rw [← List.take_append_drop k xs, List.drop_eq_getElem_cons hk]
  rw [List.flatMap_append, List.flatMap_cons, hx]

/-- The children of the rows before row `k` fill the next level up to `firstChild xs k`. -/
theorem length_flatMap_take (xs : List (Σ s, T s)) (k : Nat) :
    ((xs.take k).flatMap children).length = firstChild xs k := by
  simp [firstChild, List.length_flatMap]

/-- Child `j` of row `k` is row `firstChild xs k + j` of the next level. -/
theorem getElem?_flatMap_children {xs : List (Σ s, T s)} {k : Nat} {x : Σ s, T s}
    (h : xs[k]? = some x) {j : Nat} (hj : j < (children x).length) :
    (xs.flatMap children)[firstChild xs k + j]? = (children x)[j]? := by
  rw [flatMap_children_eq h, List.getElem?_append_right (by rw [length_flatMap_take]; omega),
    length_flatMap_take, Nat.add_sub_cancel_left, List.getElem?_append_left hj]

/-- The children of row `k` end within the next level. -/
theorem firstChild_add_le {xs : List (Σ s, T s)} {k : Nat} {x : Σ s, T s}
    (h : xs[k]? = some x) :
    firstChild xs k + (children x).length ≤ (xs.flatMap children).length := by
  rw [flatMap_children_eq h, List.length_append, List.length_append, length_flatMap_take]
  omega

end Sig.Inductive

namespace Field.Args

variable {n : Nat} {α : Type} {T : Fin n → Type}

/-- Reads field values from the slots of a row, the index slots from `i` and the scalar slots
from `j` on, where `child s p` reads the child of sort `s` on row `p` of the next level and that
level has `bound` rows. The children of a list are on the rows `p + e` for `e` below its length.
The whole segment must lie within the next level, and that level must have at most `2 ^ 64` rows,
so the row numbers of the segment never wrap around. -/
def decode (index : Nat → Option UInt64) (scalar : Nat → Option α)
    (child : (s : Fin n) → UInt64 → Option (T s)) (bound : Nat) :
    (fs : List (Field n)) → Nat → Nat → Option (Field.Args α T fs)
  | [], _, _ => some ()
  | .scalar :: fs, i, j => do
    let x ← scalar j
    let rest ← decode index scalar child bound fs i (j + 1)
    pure (x, rest)
  | .index :: fs, i, j => do
    let x ← index i
    let rest ← decode index scalar child bound fs (i + 1) j
    pure (x, rest)
  | .child s :: fs, i, j => do
    let p ← index i
    let c ← child s p
    let rest ← decode index scalar child bound fs (i + 1) j
    pure (c, rest)
  | .list s :: fs, i, j => do
    let p ← index i
    let m ← index (i + 1)
    if p.toNat + m.toNat ≤ bound ∧ bound ≤ UInt64.size then
      let cs ← (List.range m.toNat).mapM fun e => child s (p + e.toUInt64)
      let rest ← decode index scalar child bound fs (i + 2) j
      pure (cs, rest)
    else none

/-- Reading the rows `p` to `p + l - 1` gives `cs` exactly when `cs` has `l` entries and row
`p + m` reads as `cs[m]`. -/
theorem mapM_range'_eq_some_iff {β : Type} {f : Nat → Option β} :
    ∀ {l p : Nat} {cs : List β}, (List.range' p l).mapM f = some cs ↔
      cs.length = l ∧ ∀ m (h : m < cs.length), f (p + m) = some cs[m]
  | 0, _, cs => by cases cs <;> simp
  | l + 1, p, [] => by simp [List.range'_succ, Option.bind_eq_some_iff]
  | l + 1, p, y :: cs => by
    simp only [List.range'_succ, List.mapM_cons, Option.bind_eq_bind, Option.pure_def,
      Option.bind_eq_some_iff, Option.some.injEq, List.cons.injEq, List.length_cons]
    constructor
    · rintro ⟨y', hy, cs', hcs, rfl, rfl⟩
      obtain ⟨hl, hm⟩ := mapM_range'_eq_some_iff.1 hcs
      refine ⟨by omega, fun m h => ?_⟩
      cases m with
      | zero => simpa using hy
      | succ m =>
        rw [show p + (m + 1) = p + 1 + m by omega]
        simpa using hm m (by simpa using h)
    · rintro ⟨hl, hm⟩
      refine ⟨y, by simpa using hm 0 (by simp), cs,
        mapM_range'_eq_some_iff.2 ⟨by omega, fun m h => ?_⟩, rfl, rfl⟩
      rw [show p + 1 + m = p + (m + 1) by omega]
      exact hm (m + 1) (by simpa using h)

/-- Field values are read back from their own slots: index slots from `i` on holding
`args.indexSlots first`, scalar slots from `j` on, and children on the rows from `first` on. -/
theorem decode_slots {index : Nat → Option UInt64} {scalar : Nat → Option α}
    {child : (s : Fin n) → UInt64 → Option (T s)} {bound : Nat} (hB : bound < UInt64.size) :
    {fs : List (Field n)} → (args : Field.Args α T fs) → (i j first : Nat) →
      (∀ i' v, (args.indexSlots first)[i']? = some v → index (i + i') = some v) →
      (∀ j' v, args.scalarSlots[j']? = some v → scalar (j + j') = some v) →
      (∀ m s y, args.children[m]? = some ⟨s, y⟩ → child s (first + m).toUInt64 = some y) →
      first + args.children.length ≤ bound →
      decode index scalar child bound fs i j = some args
  | [], (), _, _, _, _, _, _, _ => rfl
  | .scalar :: _, (x, rest), i, j, first, hi, hs, hc, hb => by
    have h₀ : scalar j = some x := by simpa using hs 0 x (by simp [scalarSlots])
    have h₁ := decode_slots hB rest i (j + 1) first
      (fun i' v h => hi i' v (by simpa [indexSlots, Array.getElem?_append] using h))
      (fun j' v h => by
        rw [show j + 1 + j' = j + (j' + 1) by omega]
        exact hs (j' + 1) v (by simpa [scalarSlots] using h))
      (fun m s y h => hc m s y (by simpa [children] using h)) (by simpa [children] using hb)
    simp [decode, h₀, h₁]
  | .index :: _, (x, rest), i, j, first, hi, hs, hc, hb => by
    have h₀ : index i = some x := by simpa using hi 0 x (by simp [indexSlots])
    have h₁ := decode_slots hB rest (i + 1) j first
      (fun i' v h => by
        rw [show i + 1 + i' = i + (i' + 1) by omega]
        exact hi (i' + 1) v (by simpa [indexSlots, Array.getElem?_append] using h))
      (fun j' v h => hs j' v (by simpa [scalarSlots] using h))
      (fun m s y h => hc m s y (by simpa [children] using h)) (by simpa [children] using hb)
    simp [decode, h₀, h₁]
  | .child s :: _, (c, rest), i, j, first, hi, hs, hc, hb => by
    have hb' : first + 1 + rest.children.length ≤ bound := by
      simp only [children, List.length_cons] at hb
      omega
    have h₀ : index i = some first.toUInt64 := by simpa using hi 0 _ (by simp [indexSlots])
    have h₁ : child s first.toUInt64 = some c := by simpa using hc 0 s c (by simp [children])
    have h₂ := decode_slots hB rest (i + 1) j (first + 1)
      (fun i' v h => by
        rw [show i + 1 + i' = i + (i' + 1) by omega]
        exact hi (i' + 1) v (by simpa [indexSlots, Array.getElem?_append] using h))
      (fun j' v h => hs j' v (by simpa [scalarSlots] using h))
      (fun m s' y h => by
        rw [show first + 1 + m = first + (m + 1) by omega]
        exact hc (m + 1) s' y (by simpa [children] using h))
      hb'
    simp only [Nat.toUInt64_eq] at h₀ h₁
    simp [decode, h₀, h₁, h₂]
  | .list s :: _, (cs, rest), i, j, first, hi, hs, hc, hb => by
    have hb' : first + cs.length + rest.children.length ≤ bound := by
      simp only [children, List.length_append, List.length_map] at hb
      omega
    have h₀ : index i = some first.toUInt64 := by simpa using hi 0 _ (by simp [indexSlots])
    have h₁ : index (i + 1) = some cs.length.toUInt64 :=
      hi 1 _ (by simp [indexSlots, Array.getElem?_append])
    have h₂ : (List.range cs.length).mapM (fun e => child s (first.toUInt64 + e.toUInt64)) =
        some cs := by
      rw [List.range_eq_range']
      refine mapM_range'_eq_some_iff.2 ⟨rfl, fun m hm => ?_⟩
      have h := hc m s cs[m] (by
        rw [children, List.getElem?_append_left (by simpa using hm)]
        simp [hm])
      simpa using h
    have h₃ := decode_slots hB rest (i + 2) j (first + cs.length)
      (fun i' v h => by
        rw [show i + 2 + i' = i + (i' + 2) by omega]
        exact hi (i' + 2) v (by simpa [indexSlots, Array.getElem?_append] using h))
      (fun j' v h => hs j' v (by simpa [scalarSlots] using h))
      (fun m s' y h => by
        rw [Nat.add_assoc]
        exact hc (cs.length + m) s' y (by
          rw [children, List.getElem?_append_right (by simp), List.length_map,
            Nat.add_sub_cancel_left]
          exact h))
      hb'
    simp only [Nat.toUInt64_eq] at h₀ h₁ h₂
    simp [decode, h₀, h₁, UInt64.toNat_ofNat_of_lt' (show first < UInt64.size by omega),
      UInt64.toNat_ofNat_of_lt' (show cs.length < UInt64.size by omega),
      show first + cs.length ≤ bound by omega, Nat.le_of_lt hB, h₂, h₃]

/-- Field values sit in the slots of a row, the index slots from `i` on and the scalar slots
from `j` on. A child `c` of sort `s` in slot `p` needs `child s p c`, and a list `cs` takes two
slots, its first row `p` and its length, with `child s (p + m) cs[m]` for each `m`. -/
def Matches (index : Nat → UInt64) (scalar : Nat → α)
    (child : (s : Fin n) → UInt64 → T s → Prop) :
    (fs : List (Field n)) → Field.Args α T fs → Nat → Nat → Prop
  | [], _, _, _ => True
  | .scalar :: fs, (x, rest), i, j => scalar j = x ∧ Matches index scalar child fs rest i (j + 1)
  | .index :: fs, (x, rest), i, j => index i = x ∧ Matches index scalar child fs rest (i + 1) j
  | .child s :: fs, (c, rest), i, j =>
    child s (index i) c ∧ Matches index scalar child fs rest (i + 1) j
  | .list s :: fs, (cs, rest), i, j =>
    (index (i + 1)).toNat = cs.length ∧
      (∀ m (h : m < cs.length), child s (index i + m.toUInt64) cs[m]) ∧
      Matches index scalar child fs rest (i + 2) j

/-- `Matches` holds for every child relation that follows from the given one. -/
theorem Matches.imp {index : Nat → UInt64} {scalar : Nat → α}
    {child child' : (s : Fin n) → UInt64 → T s → Prop}
    (hc : ∀ s p c, child s p c → child' s p c) :
    ∀ {fs : List (Field n)} {args : Field.Args α T fs} {i j : Nat},
      Matches index scalar child fs args i j → Matches index scalar child' fs args i j
  | [], _, _, _, h => h
  | .scalar :: _, (_, _), _, _, h => ⟨h.1, Matches.imp hc h.2⟩
  | .index :: _, (_, _), _, _, h => ⟨h.1, Matches.imp hc h.2⟩
  | .child _ :: _, (_, _), _, _, h => ⟨hc _ _ _ h.1, Matches.imp hc h.2⟩
  | .list _ :: _, (_, _), _, _, h =>
    ⟨h.1, fun m hm => hc _ _ _ (h.2.1 m hm), Matches.imp hc h.2.2⟩

/-- Field values that decode sit in the slots they were read from, and each child decodes at
the row its slot names. -/
theorem Matches.of_decode [Inhabited α] {index : Nat → Option UInt64}
    {scalar : Nat → Option α} {child : (s : Fin n) → UInt64 → Option (T s)} {bound : Nat} :
    ∀ {fs : List (Field n)} {args : Field.Args α T fs} {i j : Nat},
      decode index scalar child bound fs i j = some args →
      Matches (fun i => (index i).getD 0) (fun j => (scalar j).getD default)
        (fun s p c => child s p = some c) fs args i j
  | [], _, _, _, _ => trivial
  | .scalar :: _, (_, _), _, _, h => by
    simp only [decode, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
      Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨x, hx, rest, hrest, rfl, rfl⟩ := h
    exact ⟨by simp [hx], Matches.of_decode hrest⟩
  | .index :: _, (_, _), _, _, h => by
    simp only [decode, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
      Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨x, hx, rest, hrest, rfl, rfl⟩ := h
    exact ⟨by simp [hx], Matches.of_decode hrest⟩
  | .child _ :: _, (_, _), _, _, h => by
    simp only [decode, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
      Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨p, hp, c, hc, rest, hrest, rfl, rfl⟩ := h
    exact ⟨by simpa [hp] using hc, Matches.of_decode hrest⟩
  | .list _ :: _, (_, _), _, _, h => by
    simp only [decode, Option.bind_eq_bind, Option.pure_def, Option.bind_eq_some_iff,
      Option.ite_none_right_eq_some, Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨p, hp, m, hm, -, cs, hcs, rest, hrest, rfl, rfl⟩ := h
    rw [List.range_eq_range', mapM_range'_eq_some_iff] at hcs
    obtain ⟨hl, he⟩ := hcs
    refine ⟨by simp [hm, hl], fun e h => ?_, Matches.of_decode hrest⟩
    simpa [hp] using he e h

end Field.Args

namespace Arena

open Sig.Inductive

variable {σ : Sig} {α : Type} {T : Fin σ.sorts → Type} [σ.Inductive α T]

/-- The slots of the rows of a level: the index slots of each row, with its children numbered
from the row where they start on the next level, and its scalar slots. -/
def rowSlots (xs : List (Σ s, T s)) : List (Array UInt64 × List α) :=
  List.zipWith (fun x first => ((view x.2).args.indexSlots first, (view x.2).args.scalarSlots))
    xs (xs.scanl (fun first x => first + (children x).length) 0)

/-- The level of some rows. Each column has one entry per row, which is `0` or `default` past
the slots of the row's constructor. -/
def Level.encode [Inhabited α] (xs : List (Σ s, T s)) : Arena.Level α :=
  let rows := rowSlots xs
  { tag := (xs.map fun x => (view x.2).ctor.val.toUInt64).toArray
    index := Array.ofFn fun j : Fin σ.indexSlots => (rows.map fun row => row.1.getD j 0).toArray
    scalar := Array.ofFn fun j : Fin σ.scalarSlots =>
      (rows.map fun row => row.2.getD j default).toArray }

/-- Appends to `acc` the levels below some values, one level per round, until a level is
empty. -/
def encodeAux [Inhabited α] (xs : List (Σ s, T s)) (acc : Array (Arena.Level α)) :
    Array (Arena.Level α) :=
  if xs = [] then acc else encodeAux (nodes xs 1) (acc.push (Level.encode xs))
termination_by weight xs
decreasing_by
  show weight (xs.flatMap children) < weight xs
  have := weight_flatMap_add_length_le xs
  have := List.length_pos_iff.2 ‹_›
  omega

/-- The arena of some values: level `r` holds their nodes at depth `r`, with the values
themselves, in order, on level zero. -/
def ofValues [Inhabited α] (xs : List (Σ s, T s)) : Arena α := ⟨encodeAux xs #[]⟩

/-- The arena of a value: level `r` holds the nodes at depth `r`, with the value alone on level
zero. -/
def encode [Inhabited α] {s : Fin σ.sorts} (t : T s) : Arena α := ofValues [⟨s, t⟩]

variable (T) in
/-- Decodes row `k` of level `r` as a value of sort `s`, reading its children from the next
level on, with `fuel` bounding the number of levels it reads. -/
def decodeAt (a : Arena α) : (fuel r k : Nat) → (s : Fin σ.sorts) → Option (T s)
  | 0, _, _, _ => none
  | fuel + 1, r, k, s => do
    let level ← a.levels[r]?
    let tag ← level.tag[k]?
    let c ← σ.ctorOf? tag
    if hs : σ.ctors[c].sort = s then
      let args ← Field.Args.decode (fun i => level.index[i]?.bind (·[k]?))
        (fun j => level.scalar[j]?.bind (·[k]?)) (fun s' p => decodeAt a fuel (r + 1) p.toNat s')
        ((a.levels[r + 1]?.map (·.tag.size)).getD 0) σ.ctors[c].fields 0 0
      pure (Sig.Inductive.mk ⟨c, hs, args⟩)
    else none

variable (T) in
/-- Decodes the root of an arena as a value of sort `s`. -/
def decode (a : Arena α) (s : Fin σ.sorts) : Option (T s) := a.decodeAt T a.levels.size 0 0 s

variable (T) in
/-- Decodes row `k` of level `r` as a value of the sort of its constructor, reading its children
from the next level on, with `fuel` bounding the number of levels it reads. -/
def decodeRow (a : Arena α) (fuel r k : Nat) : Option (Σ s, T s) := do
  let level ← a.levels[r]?
  let tag ← level.tag[k]?
  let c ← σ.ctorOf? tag
  let y ← a.decodeAt T fuel r k σ.ctors[c].sort
  pure ⟨σ.ctors[c].sort, y⟩

/-- A row that decodes stores one layer of the value: the number of its constructor as its tag,
and slots that match its fields (`Field.Args.Matches`), with each child on a row of the next level
that decodes to that child. The slots are read with `0` and `default` past the stored columns. -/
theorem decodeAt_eq_some [Inhabited α] {a : Arena α} {fuel r k : Nat} {s : Fin σ.sorts}
    {y : T s} (h : a.decodeAt T (fuel + 1) r k s = some y) :
    ∃ (level : Arena.Level α) (l : Layer σ α T s), a.levels[r]? = some level ∧
      level.tag[k]? = some l.ctor.val.toUInt64 ∧ y = Sig.Inductive.mk l ∧
      Field.Args.Matches (fun i => (level.index[i]?.bind (·[k]?)).getD 0)
        (fun j => (level.scalar[j]?.bind (·[k]?)).getD default)
        (fun s' p c => a.decodeAt T fuel (r + 1) p.toNat s' = some c)
        σ.ctors[l.ctor].fields l.args 0 0 := by
  simp only [decodeAt, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨level, hl, tag, ht, c, hc, h⟩ := h
  split at h
  · rename_i hs
    simp only [Option.pure_def, Option.bind_eq_some_iff, Option.some.injEq] at h
    obtain ⟨args, hargs, rfl⟩ := h
    exact ⟨level, ⟨c, hs, args⟩, hl, by rw [ht, Sig.ctorOf?_eq_some.1 hc], rfl,
      Field.Args.Matches.of_decode hargs⟩
  · cases h

/-- Row `k` of a level has the slots of value `k`, with its children from `firstChild xs k`
on. -/
theorem getElem?_rowSlots {xs : List (Σ s, T s)} {k : Nat} {x : Σ s, T s}
    (h : xs[k]? = some x) :
    (rowSlots xs)[k]? =
      some ((view x.2).args.indexSlots (firstChild xs k), (view x.2).args.scalarSlots) := by
  have hk := (List.getElem?_eq_some_iff.1 h).1
  have hsum : ∀ (l : List (Σ s, T s)) b,
      l.foldl (fun first x => first + (children x).length) b =
        b + (l.map fun x => (children x).length).sum := by
    intro l
    induction l with
    | nil => simp
    | cons y l ih =>
      intro b
      simp only [List.foldl_cons, ih, List.map_cons, List.sum_cons]
      omega
  rw [rowSlots, List.getElem?_zipWith, h, List.getElem?_scanl, ite_eq_left (Nat.le_of_lt hk), hsum]
  simp [firstChild]

section Encode

variable [Inhabited α]

/-- Level `r` of `encodeAux xs acc` is level `r` of `acc` if `acc` has it, and otherwise the
level of the values `r - acc.size` levels below `xs`, if there are any. -/
theorem getElem?_encodeAux (xs : List (Σ s, T s)) (acc : Array (Arena.Level α)) (r : Nat) :
    (encodeAux xs acc)[r]? = if r < acc.size then acc[r]? else
      if nodes xs (r - acc.size) = [] then none
      else some (Level.encode (nodes xs (r - acc.size))) := by
  fun_induction encodeAux xs acc with
  | case1 acc =>
    simp only [nodes_nil, ite_true]
    split
    · rfl
    · exact Array.getElem?_eq_none (by omega)
  | case2 xs acc h ih =>
    rw [ih, Array.size_push]
    rcases Nat.lt_trichotomy r acc.size with hr | rfl | hr
    · simp [hr, Nat.lt_succ_of_lt hr, Array.getElem_push_lt]
    · simp [nodes, h]
    · have h₁ : ¬r < acc.size + 1 := by omega
      have h₂ : ¬r < acc.size := by omega
      rw [show r - acc.size = r - (acc.size + 1) + 1 by omega, nodes_succ' xs (r - (acc.size + 1))]
      simp only [h₁, h₂, ↓reduceIte]
      rfl

/-- Level `r` of the arena of some values is the level of their nodes at depth `r`, if there
are any. -/
theorem getElem?_ofValues (xs : List (Σ s, T s)) (r : Nat) :
    (ofValues xs).levels[r]? = if nodes xs r = [] then none
      else some (Level.encode (nodes xs r)) := by
  simp [ofValues, getElem?_encodeAux]

/-- The tag column of a level has one entry per row. -/
@[simp] theorem size_tag (xs : List (Σ s, T s)) : (Level.encode xs).tag.size = xs.length := by
  simp [Level.encode]

/-- Row `k` of the tag column of a level is the constructor number of value `k`. -/
theorem getElem?_tag (xs : List (Σ s, T s)) (k : Nat) :
    (Level.encode xs).tag[k]? = xs[k]?.map fun x => (view x.2).ctor.val.toUInt64 := by
  simp [Level.encode]

/-- Index slot `i` of row `k` of a level is index slot `i` of value `k`. -/
theorem index_encode {xs : List (Σ s, T s)} {k : Nat} {x : Σ s, T s} (h : xs[k]? = some x)
    {i : Nat} {v : UInt64}
    (hv : ((view x.2).args.indexSlots (firstChild xs k))[i]? = some v) :
    ((Level.encode xs).index[i]?).bind (·[k]?) = some v := by
  have hi : i < σ.indexSlots := by
    have h₁ := (Array.getElem?_eq_some_iff.1 hv).1
    rw [Field.Args.size_indexSlots] at h₁
    exact Nat.lt_of_lt_of_le h₁ (σ.indexSlots_le _)
  simp [Level.encode, hi, getElem?_rowSlots h, List.getD_eq_getElem?_getD, hv]

/-- Scalar slot `j` of row `k` of a level is scalar slot `j` of value `k`. -/
theorem scalar_encode {xs : List (Σ s, T s)} {k : Nat} {x : Σ s, T s} (h : xs[k]? = some x)
    {j : Nat} {v : α} (hv : (view x.2).args.scalarSlots[j]? = some v) :
    ((Level.encode xs).scalar[j]?).bind (·[k]?) = some v := by
  have hj : j < σ.scalarSlots := by
    have h₁ := (List.getElem?_eq_some_iff.1 hv).1
    rw [Field.Args.length_scalarSlots] at h₁
    exact Nat.lt_of_lt_of_le h₁ (σ.scalarSlots_le _)
  simp [Level.encode, hj, getElem?_rowSlots h, List.getD_eq_getElem?_getD, hv]

/-- Index column `i` of a level holds, in row `k`, index slot `i` of value `k`, or `0` past the
slots of its constructor. -/
theorem index_encode_of_lt {xs : List (Σ s, T s)} {k : Nat} {x : Σ s, T s} (h : xs[k]? = some x)
    {i : Nat} (hi : i < σ.indexSlots) :
    ((Level.encode xs).index[i]?).bind (·[k]?) =
      some (((view x.2).args.indexSlots (firstChild xs k)).getD i 0) := by
  simp [Level.encode, hi, getElem?_rowSlots h]

/-- Scalar column `j` of a level holds, in row `k`, scalar slot `j` of value `k`, or `default`
past the slots of its constructor. -/
theorem scalar_encode_of_lt {xs : List (Σ s, T s)} {k : Nat} {x : Σ s, T s}
    (h : xs[k]? = some x) {j : Nat} (hj : j < σ.scalarSlots) :
    ((Level.encode xs).scalar[j]?).bind (·[k]?) =
      some ((view x.2).args.scalarSlots.getD j default) := by
  simp [Level.encode, hj, getElem?_rowSlots h]

/-- A level has one index column per index slot of the signature. -/
@[simp] theorem size_index (xs : List (Σ s, T s)) :
    (Level.encode xs).index.size = σ.indexSlots := by
  simp [Level.encode]

/-- A level has one scalar column per scalar slot of the signature. -/
@[simp] theorem size_scalar (xs : List (Σ s, T s)) :
    (Level.encode xs).scalar.size = σ.scalarSlots := by
  simp [Level.encode]

/-- Every index column of a level has one entry per row. -/
theorem size_getElem_index (xs : List (Σ s, T s)) {i : Nat}
    (hi : i < (Level.encode xs).index.size) :
    (Level.encode xs).index[i].size = xs.length := by
  simp [Level.encode, rowSlots]

/-- Every scalar column of a level has one entry per row. -/
theorem size_getElem_scalar (xs : List (Σ s, T s)) {j : Nat}
    (hj : j < (Level.encode xs).scalar.size) :
    (Level.encode xs).scalar[j].size = xs.length := by
  simp [Level.encode, rowSlots]

/-- Row `k` of level `r` of the arena of some values decodes to their node `k` at depth `r`,
given fuel for the levels from `r` on, if every level below the values has fewer than `2 ^ 64`
nodes. -/
theorem decodeAt_ofValues {xs : List (Σ s, T s)}
    (hxs : ∀ r, (nodes xs (r + 1)).length < UInt64.size) (fuel : Nat) :
    ∀ {r k : Nat} {s : Fin σ.sorts} {y : T s},
      (nodes xs r)[k]? = some ⟨s, y⟩ → (ofValues xs).levels.size ≤ r + fuel →
        (ofValues xs).decodeAt T fuel r k s = some y := by
  induction fuel with
  | zero =>
    intro r k s y hy hsize
    have hne : nodes xs r ≠ [] := by
      rintro h
      simp [h] at hy
    have hr := getElem?_ofValues xs r
    rw [ite_eq_right hne] at hr
    have := (Array.getElem?_eq_some_iff.1 hr).1
    omega
  | succ fuel ih =>
    intro r k s y hy hsize
    have hne : nodes xs r ≠ [] := by
      rintro h
      simp [h] at hy
    have hr := getElem?_ofValues xs r
    rw [ite_eq_right hne] at hr
    have hbound : (((ofValues xs).levels[r + 1]?).map (·.tag.size)).getD 0 =
        (nodes xs (r + 1)).length := by
      rw [getElem?_ofValues]
      split <;> simp_all
    have hB : (nodes xs (r + 1)).length < UInt64.size := hxs r
    rw [decodeAt, hr, hbound]
    simp only [Option.bind_eq_bind, Option.bind_some, getElem?_tag, hy, Option.map_some,
      Sig.ctorOf?_toUInt64, dite_eq_left (view y).sort_eq]
    rw [Field.Args.decode_slots hB (view y).args 0 0 (firstChild (nodes xs r) k)]
    · exact congrArg some (mk_view y)
    · intro i v hv
      rw [Nat.zero_add]
      exact index_encode hy hv
    · intro j v hv
      rw [Nat.zero_add]
      exact scalar_encode hy hv
    · intro m s' y' hm
      have hm' : m < (children ⟨s, y⟩).length := (List.getElem?_eq_some_iff.1 hm).1
      have h₁ : (nodes xs (r + 1))[firstChild (nodes xs r) k + m]? =
          some ⟨s', y'⟩ := by
        rw [nodes, getElem?_flatMap_children hy hm']
        exact hm
      have hlt : firstChild (nodes xs r) k + m < UInt64.size := by
        have := (List.getElem?_eq_some_iff.1 h₁).1
        omega
      rw [Nat.toUInt64_eq, UInt64.toNat_ofNat_of_lt' hlt]
      exact ih h₁ (by omega)
    · exact firstChild_add_le hy

/-- Row `k` of level zero of the arena of some values decodes to value `k`, if the values have
fewer than `2 ^ 64` nodes in all. -/
theorem decodeAt_ofValues_zero {xs : List (Σ s, T s)}
    (hxs : (xs.map fun x => count x.2).sum < UInt64.size) {k : Nat} {s : Fin σ.sorts} {y : T s}
    (h : xs[k]? = some ⟨s, y⟩) :
    (ofValues xs).decodeAt T (ofValues xs).levels.size 0 k s = some y :=
  decodeAt_ofValues (fun r => Nat.lt_of_le_of_lt (length_nodes_le xs (r + 1)) hxs) _ h (by omega)

/-- Row `k` of level zero of the arena of some values decodes to value `k` with its sort, if every
level below the values has fewer than `2 ^ 64` nodes. -/
theorem decodeRow_ofValues {xs : List (Σ s, T s)}
    (hxs : ∀ r, (nodes xs (r + 1)).length < UInt64.size) {k : Nat} {x : Σ s, T s}
    (h : xs[k]? = some x) :
    (ofValues xs).decodeRow T (ofValues xs).levels.size 0 k = some x := by
  obtain ⟨s, y⟩ := x
  have hne : nodes xs 0 ≠ [] := by
    rintro h0
    simp only [nodes] at h0
    simp [h0] at h
  have hr := getElem?_ofValues xs 0
  rw [ite_eq_right hne] at hr
  have hy := decodeAt_ofValues hxs (ofValues xs).levels.size (r := 0) h (by omega)
  simp only [decodeRow, Option.bind_eq_bind, hr, Option.bind_some, getElem?_tag, nodes, h,
    Option.map_some, Sig.ctorOf?_toUInt64]
  rw [(view y).sort_eq, hy]
  rfl

/-- Decoding the arena of a value gives the value back, if it has fewer than `2 ^ 64` nodes. -/
theorem decode_encode {s : Fin σ.sorts} (t : T s) (ht : count t < UInt64.size) :
    (encode t).decode T s = some t :=
  decodeAt_ofValues_zero (by simpa using ht) rfl

end Encode

section Map

variable {β γ : Type}

/-- The arena with `f` applied to each scalar. The arena of a signature without scalar slots has
no scalars, so mapping it gives the same rows with scalars of any type, for example to upload it. -/
def map (f : α → β) (a : Arena α) : Arena β :=
  ⟨a.levels.map fun l => ⟨l.tag, l.index, l.scalar.map (·.map f)⟩⟩

/-- Mapping by `f` and then by `g` is mapping by their composition. -/
@[simp] theorem map_map (f : α → β) (g : β → γ) (a : Arena α) :
    (a.map f).map g = a.map (g ∘ f) := by
  simp [map, Function.comp_def]

/-- Mapping by the identity keeps an arena. -/
@[simp] theorem map_id (a : Arena α) : a.map id = a := by
  simp [map]

/-- An arena of `Unit` scalars, like the arenas of types without scalar fields, comes back from
mapping its scalars to any type and back. -/
theorem map_map_unit (f : Unit → β) (g : β → Unit) (a : Arena Unit) : (a.map f).map g = a := by
  rw [map_map, show g ∘ f = id from rfl, map_id]

end Map

section Columns

variable (σ) in
/-- An arena is wide for `σ` when every level has `σ.indexSlots` index columns and
`σ.scalarSlots` scalar columns, each with one entry per row. -/
def Wide (a : Arena α) : Prop :=
  ∀ l ∈ a.levels, l.index.size = σ.indexSlots ∧ l.scalar.size = σ.scalarSlots ∧
    (∀ c ∈ l.index, c.size = l.tag.size) ∧ ∀ c ∈ l.scalar, c.size = l.tag.size

instance (a : Arena α) : Decidable (a.Wide σ) := by
  unfold Wide
  infer_instance

/-- An arena as device columns, with all levels concatenated, root level first: the tags, each
index slot column and each scalar slot column, and the number of rows of each level. -/
structure Columns (α : Type) where
  /-- The number of rows of each level. -/
  sizes : Array Nat
  /-- The constructor numbers of all rows. -/
  tag : Array UInt64
  /-- The index slot columns. -/
  index : Array (Array UInt64)
  /-- The scalar slot columns. -/
  scalar : Array (Array α)
  deriving Inhabited

variable (σ) in
/-- The columns of an arena: `σ.indexSlots` index and `σ.scalarSlots` scalar columns, each the
concatenation of that column of every level. -/
def columns (a : Arena α) : Columns α where
  sizes := a.levels.map (·.tag.size)
  tag := (a.levels.map (·.tag)).flatten
  index := Array.ofFn fun i : Fin σ.indexSlots => (a.levels.map (·.index.getD i #[])).flatten
  scalar := Array.ofFn fun j : Fin σ.scalarSlots => (a.levels.map (·.scalar.getD j #[])).flatten

/-- Every index column of a wide arena has one entry per row. -/
theorem Wide.size_of_mem_index {a : Arena α} (h : a.Wide σ) :
    ∀ col ∈ (a.columns σ).index, col.size = (a.columns σ).tag.size := by
  intro col hcol
  simp only [columns] at hcol ⊢
  obtain ⟨i, rfl⟩ := Array.mem_ofFn.1 hcol
  simp only [Array.size_flatten, Array.map_map]
  congr 1
  refine Array.map_congr_left fun l hl => ?_
  obtain ⟨hi, -, hic, -⟩ := h l hl
  simp only [Function.comp_def]
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
  exact hic _ (Array.getElem_mem _)

/-- Every scalar column of a wide arena has one entry per row. -/
theorem Wide.size_of_mem_scalar {a : Arena α} (h : a.Wide σ) :
    ∀ col ∈ (a.columns σ).scalar, col.size = (a.columns σ).tag.size := by
  intro col hcol
  simp only [columns] at hcol ⊢
  obtain ⟨j, rfl⟩ := Array.mem_ofFn.1 hcol
  simp only [Array.size_flatten, Array.map_map]
  congr 1
  refine Array.map_congr_left fun l hl => ?_
  obtain ⟨-, hs, -, hsc⟩ := h l hl
  simp only [Function.comp_def]
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
  exact hsc _ (Array.getElem_mem _)

/-- The consecutive pieces of an array with the given sizes, from `start` on. -/
def pieces {β : Type} (xs : Array β) : List Nat → Nat → List (Array β)
  | [], _ => []
  | n :: ns, start => xs.extract start (start + n) :: pieces xs ns (start + n)

/-- The arena of some columns: level `r` holds the piece number `r` of every column, cut at the
sizes `sizes`. A piece that runs past the end of its column is cut short, and the rows past the
sum of the sizes are dropped. `Arena.ofColumns_columns` gives back the arena of wide columns. -/
def ofColumns (c : Columns α) : Arena α :=
  let sizes := c.sizes.toList
  let tag := (pieces c.tag sizes 0).toArray
  let index := c.index.map fun col => (pieces col sizes 0).toArray
  let scalar := c.scalar.map fun col => (pieces col sizes 0).toArray
  ⟨Array.ofFn fun r : Fin c.sizes.size =>
    ⟨tag.getD r #[], index.map (·.getD r #[]), scalar.map (·.getD r #[])⟩⟩

/-- Cutting an array at the sizes of some arrays, where it holds their concatenation from
`start` on, gives them back. -/
theorem pieces_flatten {β : Type} {xs : Array β} :
    ∀ (L : List (Array β)) (start : Nat), xs.toList.drop start = (L.map Array.toList).flatten →
      pieces xs (L.map Array.size) start = L
  | [], _, _ => rfl
  | l :: L, start, h => by
    have h₁ : xs.extract start (start + l.size) = l := by
      apply Array.ext'
      rw [Array.toList_extract, List.extract_eq_take_drop, Nat.add_sub_cancel_left, h,
        List.map_cons, List.flatten_cons, List.take_left' (by simp)]
    have h₂ : xs.toList.drop (start + l.size) = (L.map Array.toList).flatten := by
      rw [← List.drop_drop, h, List.map_cons, List.flatten_cons, List.drop_left' (by simp)]
    rw [List.map_cons, pieces, h₁, pieces_flatten L _ h₂]

/-- The columns of a wide arena give the arena back. -/
theorem ofColumns_columns {a : Arena α} (h : a.Wide σ) : ofColumns (a.columns σ) = a := by
  obtain ⟨levels⟩ := a
  have hpieces : ∀ {β : Type} (f : Arena.Level α → Array β),
      (∀ l ∈ levels, (f l).size = l.tag.size) →
      pieces (levels.map f).flatten (levels.map (·.tag.size)).toList 0 =
        (levels.map f).toList := by
    intro β f hf
    have hs : (levels.map (·.tag.size)).toList = (levels.map f).toList.map Array.size := by
      simp only [Array.toList_map, List.map_map]
      exact List.map_congr_left fun l hl => (hf l (Array.mem_def.2 hl)).symm
    rw [hs]
    exact pieces_flatten _ 0 (by simp)
  simp only [ofColumns, columns, Arena.mk.injEq]
  apply Array.ext
  · simp
  · intro r h₁ h₂
    obtain ⟨hi, hs, hic, hsc⟩ := h levels[r] (Array.getElem_mem h₂)
    have htag := hpieces (·.tag) fun _ _ => rfl
    have hindex : ∀ i : Fin σ.indexSlots,
        pieces (levels.map (·.index.getD i #[])).flatten (levels.map (·.tag.size)).toList 0 =
          (levels.map (·.index.getD i #[])).toList := by
      intro i
      refine hpieces _ fun l hl => ?_
      obtain ⟨hi', -, hic', -⟩ := h l hl
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
      exact hic' _ (Array.getElem_mem _)
    have hscalar : ∀ j : Fin σ.scalarSlots,
        pieces (levels.map (·.scalar.getD j #[])).flatten (levels.map (·.tag.size)).toList 0 =
          (levels.map (·.scalar.getD j #[])).toList := by
      intro j
      refine hpieces _ fun l hl => ?_
      obtain ⟨-, hs', -, hsc'⟩ := h l hl
      rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
      exact hsc' _ (Array.getElem_mem _)
    rw [Array.getElem_ofFn]
    have e₁ : (pieces (levels.map (·.tag)).flatten (levels.map (·.tag.size)).toList 0).toArray.getD
        r #[] = levels[r].tag := by
      rw [htag]
      simp [h₂]
    have e₂ : ((Array.ofFn fun i : Fin σ.indexSlots =>
        (levels.map (·.index.getD i #[])).flatten).map fun col =>
          (pieces col (levels.map (·.tag.size)).toList 0).toArray).map (·.getD r #[]) =
        levels[r].index := by
      apply Array.ext
      · simp [hi]
      · intro i hi₁ hi₂
        simp only [Array.getElem_map, Array.getElem_ofFn]
        rw [hindex ⟨i, by simpa using hi₁⟩]
        simp [h₂, hi₂]
    have e₃ : ((Array.ofFn fun j : Fin σ.scalarSlots =>
        (levels.map (·.scalar.getD j #[])).flatten).map fun col =>
          (pieces col (levels.map (·.tag.size)).toList 0).toArray).map (·.getD r #[]) =
        levels[r].scalar := by
      apply Array.ext
      · simp [hs]
      · intro j hj₁ hj₂
        simp only [Array.getElem_map, Array.getElem_ofFn]
        rw [hscalar ⟨j, by simpa using hj₁⟩]
        simp [h₂, hj₂]
    simp only [e₁, e₂, e₃]

/-- Every level of the arena of some values is wide. -/
theorem wide_ofValues [Inhabited α] (xs : List (Σ s, T s)) : (ofValues xs : Arena α).Wide σ := by
  intro l hl
  obtain ⟨r, hr, rfl⟩ := Array.getElem_of_mem hl
  have h := getElem?_ofValues xs r
  rw [Array.getElem?_eq_getElem hr] at h
  split at h
  · cases h
  · rw [Option.some.inj h]
    have hrows : (rowSlots (nodes xs r)).length = (nodes xs r).length := by
      simp [rowSlots]
    refine ⟨by simp [Level.encode], by simp [Level.encode], ?_, ?_⟩
    · intro c hc
      obtain ⟨i, rfl⟩ := Array.mem_ofFn.1 hc
      simp [hrows]
    · intro c hc
      obtain ⟨j, rfl⟩ := Array.mem_ofFn.1 hc
      simp [hrows]

/-- Mapping the scalars of a wide arena keeps it wide. -/
theorem Wide.map {β : Type} {a : Arena α} (h : a.Wide σ) (f : α → β) : (a.map f).Wide σ := by
  intro _ hl
  obtain ⟨l, hl', rfl⟩ := Array.mem_map.1 hl
  obtain ⟨h₁, h₂, h₃, h₄⟩ := h l hl'
  refine ⟨h₁, by simpa using h₂, h₃, fun _ hc => ?_⟩
  obtain ⟨c, hc', rfl⟩ := Array.mem_map.1 hc
  simpa using h₄ c hc'

end Columns

end Arena

end TorchLean.Kernel
