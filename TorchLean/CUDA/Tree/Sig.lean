/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Algebra.Order.BigOperators.Group.List
public import Mathlib.Data.List.MinMax
public import Mathlib.Order.Nat

/-!
# Signatures of inductive types

A `Sig` describes a family of inductive types by its constructors. Each constructor builds one
sort, and each of its fields is a scalar, a `UInt64` index, one child of some sort or a list of
children of some sort. Binary trees, syntax trees, rose trees, tries and mutual families fit,
as long as every field that isn't a child is a scalar or an index.

`Sig.Inductive σ α T` says that the Lean family `T` is described by `σ`: every value is one
`Layer`, a constructor with its field values, over values that are smaller in total. The device
encoding and the folds only use this interface, so their theorems hold for every signature.
-/

@[expose] public section

namespace TorchLean.Kernel

/-- The kind of a constructor field over `n` sorts: a scalar, an index, one child of a sort or a
list of children of a sort. -/
inductive Field (n : Nat) where
  | scalar
  | index
  | child (sort : Fin n)
  | list (sort : Fin n)
  deriving DecidableEq, Repr

/-- A constructor: the sort it builds and the kinds of its fields, in declaration order. -/
structure Ctor (n : Nat) where
  /-- The sort of the values the constructor builds. -/
  sort : Fin n
  /-- The kinds of the constructor's fields, in declaration order. -/
  fields : List (Field n)
  deriving DecidableEq, Repr

/-- A signature: the number of sorts and the constructors. Constructors are stored by number in
`UInt64` tags, so there are fewer than `2 ^ 64` of them. -/
structure Sig where
  /-- The number of sorts. -/
  sorts : Nat
  /-- The constructors, numbered by their position. -/
  ctors : List (Ctor sorts)
  /-- Every constructor number fits in a `UInt64` tag. -/
  ctors_lt : ctors.length < UInt64.size := by decide

/-- The value of a field: a scalar, an index, one child or a list of children. -/
abbrev Field.Value {n : Nat} (α : Type) (T : Fin n → Type) : Field n → Type
  | .scalar => α
  | .index => UInt64
  | .child s => T s
  | .list s => List (T s)

/-- The values of a list of fields, as a tuple nested to the right and ending in `Unit`. -/
abbrev Field.Args {n : Nat} (α : Type) (T : Fin n → Type) : List (Field n) → Type
  | [] => Unit
  | f :: fs => f.Value α T × Field.Args α T fs

/-- Index slots of a field: none for a scalar, one for an index or a child, and two for a list
(its first row and its length). -/
def Field.indexSlots {n : Nat} : Field n → Nat
  | .scalar => 0
  | .index => 1
  | .child _ => 1
  | .list _ => 2

/-- Scalar slots of a field: one for a scalar and none otherwise. -/
def Field.scalarSlots {n : Nat} : Field n → Nat
  | .scalar => 1
  | _ => 0

/-- Index slots of a constructor, one or two per field as `Field.indexSlots` says. -/
def Ctor.indexSlots {n : Nat} (c : Ctor n) : Nat := (c.fields.map Field.indexSlots).sum

/-- Scalar slots of a constructor, one per scalar field. -/
def Ctor.scalarSlots {n : Nat} (c : Ctor n) : Nat := (c.fields.map Field.scalarSlots).sum

/-- The number of index columns of a signature: the most index slots of any constructor. -/
def Sig.indexSlots (σ : Sig) : Nat := (σ.ctors.map Ctor.indexSlots).foldr max 0

/-- The number of scalar columns of a signature: the most scalar slots of any constructor. -/
def Sig.scalarSlots (σ : Sig) : Nat := (σ.ctors.map Ctor.scalarSlots).foldr max 0

/-- The constructor stored under a tag, if the tag names one. -/
def Sig.ctorOf? (σ : Sig) (tag : UInt64) : Option (Fin σ.ctors.length) :=
  if h : tag.toNat < σ.ctors.length then some ⟨tag.toNat, h⟩ else none

/-- Every constructor fits in the index columns. -/
theorem Sig.indexSlots_le (σ : Sig) (c : Fin σ.ctors.length) :
    σ.ctors[c].indexSlots ≤ σ.indexSlots :=
  List.le_max_of_le (List.mem_map.2 ⟨_, List.getElem_mem c.2, rfl⟩) le_rfl

/-- Every constructor fits in the scalar columns. -/
theorem Sig.scalarSlots_le (σ : Sig) (c : Fin σ.ctors.length) :
    σ.ctors[c].scalarSlots ≤ σ.scalarSlots :=
  List.le_max_of_le (List.mem_map.2 ⟨_, List.getElem_mem c.2, rfl⟩) le_rfl

/-- The tag of a constructor names that constructor. -/
@[simp] theorem Sig.ctorOf?_toUInt64 (σ : Sig) (c : Fin σ.ctors.length) :
    σ.ctorOf? c.val.toUInt64 = some c := by
  have h : c.val < UInt64.size := Nat.lt_trans c.2 σ.ctors_lt
  have hv : c.val.toUInt64.toNat = c.val := UInt64.toNat_ofNat_of_lt' h
  simp only [Sig.ctorOf?, hv, c.2, dite_true]

/-- A tag names a constructor exactly when it is that constructor's number. -/
theorem Sig.ctorOf?_eq_some {σ : Sig} {tag : UInt64} {c : Fin σ.ctors.length} :
    σ.ctorOf? tag = some c ↔ tag = c.val.toUInt64 := by
  constructor
  · intro h
    unfold Sig.ctorOf? at h
    split at h
    · cases h
      simp
    · cases h
  · rintro rfl
    exact σ.ctorOf?_toUInt64 c

namespace Field.Args

variable {n : Nat} {α : Type} {T : Fin n → Type}

/-- The children among field values, in field order, each with its sort. -/
def children : {fs : List (Field n)} → Field.Args α T fs → List (Σ s, T s)
  | [], _ => []
  | .scalar :: _, (_, rest) => children rest
  | .index :: _, (_, rest) => children rest
  | .child s :: _, (c, rest) => ⟨s, c⟩ :: children rest
  | .list s :: _, (cs, rest) => cs.map (⟨s, ·⟩) ++ children rest

/-- The index slots of field values when their children start at row `first` of the next level:
an index field stores itself, a child its row, and a list its first row and its length. -/
def indexSlots : {fs : List (Field n)} → Field.Args α T fs → Nat → Array UInt64
  | [], _, _ => #[]
  | .scalar :: _, (_, rest), first => indexSlots rest first
  | .index :: _, (x, rest), first => #[x] ++ indexSlots rest first
  | .child _ :: _, (_, rest), first => #[first.toUInt64] ++ indexSlots rest (first + 1)
  | .list _ :: _, (cs, rest), first =>
    #[first.toUInt64, cs.length.toUInt64] ++ indexSlots rest (first + cs.length)

/-- The scalar fields among field values, in field order. -/
def scalarSlots : {fs : List (Field n)} → Field.Args α T fs → List α
  | [], _ => []
  | .scalar :: _, (x, rest) => x :: scalarSlots rest
  | .index :: _, (_, rest) => scalarSlots rest
  | .child _ :: _, (_, rest) => scalarSlots rest
  | .list _ :: _, (_, rest) => scalarSlots rest

/-- The index slots of field values number `Field.indexSlots` per field. -/
theorem size_indexSlots : {fs : List (Field n)} → (args : Field.Args α T fs) → (first : Nat) →
    (indexSlots args first).size = (fs.map Field.indexSlots).sum
  | [], _, _ => rfl
  | .scalar :: _, (_, rest), first => by
    simp [indexSlots, Field.indexSlots, size_indexSlots rest first]
  | .index :: _, (_, rest), first => by
    simp [indexSlots, Field.indexSlots, size_indexSlots rest first]
  | .child _ :: _, (_, rest), first => by
    simp [indexSlots, Field.indexSlots, size_indexSlots rest]
  | .list _ :: _, (_, rest), first => by
    simp [indexSlots, Field.indexSlots, size_indexSlots rest]

/-- The scalar slots of field values number `Field.scalarSlots` per field. -/
theorem length_scalarSlots : {fs : List (Field n)} → (args : Field.Args α T fs) →
    (scalarSlots args).length = (fs.map Field.scalarSlots).sum
  | [], _ => rfl
  | .scalar :: _, (_, rest) => by
    simp [scalarSlots, Field.scalarSlots, length_scalarSlots rest]; omega
  | .index :: _, (_, rest) => by simp [scalarSlots, Field.scalarSlots, length_scalarSlots rest]
  | .child _ :: _, (_, rest) => by simp [scalarSlots, Field.scalarSlots, length_scalarSlots rest]
  | .list _ :: _, (_, rest) => by simp [scalarSlots, Field.scalarSlots, length_scalarSlots rest]

end Field.Args

/-- One layer of a value of sort `s`: a constructor of that sort and the values of its fields,
whose children are values of the family `T`. -/
structure Layer (σ : Sig) (α : Type) (T : Fin σ.sorts → Type) (s : Fin σ.sorts) where
  /-- The number of the constructor. -/
  ctor : Fin σ.ctors.length
  /-- The constructor builds values of sort `s`. -/
  sort_eq : σ.ctors[ctor].sort = s
  /-- The values of the constructor's fields. -/
  args : Field.Args α T σ.ctors[ctor].fields

/-- The children of a layer, in field order, each with its sort. -/
def Layer.children {σ : Sig} {α : Type} {T : Fin σ.sorts → Type} {s : Fin σ.sorts}
    (l : Layer σ α T s) : List (Σ s, T s) :=
  l.args.children

/-- `T` is the family of inductive types that `σ` describes, with scalar fields in `α`: every
value is one layer whose children are smaller in total `size`. -/
class Sig.Inductive (σ : Sig) (α : outParam Type) (T : Fin σ.sorts → Type) where
  intro ::
  /-- The outermost constructor of a value with its field values. -/
  view {s : Fin σ.sorts} : T s → Layer σ α T s
  /-- The value made of one layer. -/
  mk {s : Fin σ.sorts} : Layer σ α T s → T s
  /-- Every value is made of its outermost layer. -/
  mk_view {s : Fin σ.sorts} (t : T s) : mk (view t) = t
  /-- A measure on values that the children of a value decrease in total. -/
  size {s : Fin σ.sorts} : T s → Nat
  /-- The children of a value are smaller in total than the value. -/
  size_lt {s : Fin σ.sorts} (t : T s) : ((view t).children.map fun c => size c.2).sum < size t

end TorchLean.Kernel
