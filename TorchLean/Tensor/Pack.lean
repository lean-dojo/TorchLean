/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Arithmetic

/-!
# Heterogeneous Packs

`TorchLean.Pack F ss` holds one `F s` for every index `s` in the list `ss`. Tensor packs, runtime
references, graph variables, error bounds, optimizer states, mutable parameters and initialization
plans are packs over their own families, so they share `append`, `split`, `snoc`, `unsnoc`, `cast`,
`map`, `get`, `set` and the lemmas about them.

`TorchLean.TensorPack α shapes` is `Pack (fun s => Tensor α s) shapes`: one tensor for every
shape in `shapes`. Unlike a tensor, whose entries all have one element type and one rectangular
shape, a tensor pack may contain tensors of different ranks and extents, each with a statically
known shape. The `TensorPack` namespace holds the operations that need tensors: `fill`, `zero`,
`add`, `stackLeading` and printing. Lean prints a `Pack` operation on a tensor pack in prefix form,
as in `Pack.append xs ys`.

Tensor packs are the common representation for model parameters, gradients and typed graph
contexts. Ordinary supervised data uses the named `Sample.Supervised` record and becomes a pack only
at the graph-runtime boundary. This module holds the datatypes and their representation-independent
operations, so those layers do not depend on autograd proofs or a particular runtime.
-/

@[expose] public section

open TorchLean.Shape (size_append)

namespace TorchLean

/-- A heterogeneous list holding one `F s` for every index `s` in its index list. -/
inductive Pack {ι : Type} (F : ι → Type) : List ι → Type where
  /-- The empty pack. -/
  | nil : Pack F []
  /-- Add an entry whose index becomes the head of the pack's index list. -/
  | cons {s : ι} {ss : List ι} : F s → Pack F ss → Pack F (s :: ss)

namespace Pack

variable {ι : Type} {F G H : ι → Type}

/-- The pack holding one entry. -/
def singleton {s : ι} (x : F s) : Pack F [s] :=
  .cons x .nil

/-- The pack holding two entries, in order. -/
def pair {s t : ι} (x : F s) (y : F t) : Pack F [s, t] :=
  .cons x (.cons y .nil)

/-- The first entry of a nonempty pack. -/
def head {s : ι} {ss : List ι} (xs : Pack F (s :: ss)) : F s :=
  match xs with
  | .cons x _ => x

/-- Every entry after the first in a nonempty pack. -/
def tail {s : ι} {ss : List ι} (xs : Pack F (s :: ss)) : Pack F ss :=
  match xs with
  | .cons _ rest => rest

/-- A family of nonempty packs is determined by its heads and tails. -/
theorem eta_cons {κ : Type*} {s : ι} {ss : List ι} (f : κ → Pack F (s :: ss)) :
    f = fun y => .cons (f y).head (f y).tail := by
  funext y
  cases f y
  rfl

/-- The entry at position `i`, whose type is determined by the pack's index list. -/
def get : {ss : List ι} → Pack F ss → (i : Fin ss.length) → F (ss.get i)
  | [], .nil, i => nomatch i
  | _ :: _, .cons x _, ⟨0, _⟩ => x
  | _ :: _, .cons _ xs, ⟨Nat.succ i, hi⟩ => get xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩

/-- Reading position zero of a `cons` gives its head. -/
@[simp] theorem get_cons_zero {s : ι} {ss : List ι} (x : F s) (xs : Pack F ss)
    (h : 0 < (s :: ss).length) :
    get (.cons x xs) ⟨0, h⟩ = x := by
  rfl

/-- Reading a later position of a `cons` reads its tail one position earlier. -/
@[simp] theorem get_cons_succ {s : ι} {ss : List ι} (x : F s) (xs : Pack F ss) (i : Nat)
    (h : Nat.succ i < (s :: ss).length) :
    get (.cons x xs) ⟨Nat.succ i, h⟩ = get xs ⟨i, Nat.lt_of_succ_lt_succ h⟩ := by
  rfl

/-- Replace the entry at position `i` with `value`, whose index is the one the pack's index list
gives at that position. -/
def set : {ss : List ι} → Pack F ss → (i : Fin ss.length) → F (ss.get i) → Pack F ss
  | [], .nil, i, _ => nomatch i
  | _ :: _, .cons _ xs, ⟨0, _⟩, value => .cons value xs
  | _ :: _, .cons x xs, ⟨Nat.succ i, hi⟩, value =>
      .cons x (set xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩ value)

/-- Reading position `i` of `xs.set i value` gives `value`. -/
@[simp] theorem get_set_self : {ss : List ι} →
    (xs : Pack F ss) → (i : Fin ss.length) → (value : F (ss.get i)) →
    (xs.set i value).get i = value
  | [], .nil, i, _ => nomatch i
  | _ :: _, .cons _ _, ⟨0, _⟩, _ => rfl
  | _ :: _, .cons _ xs, ⟨Nat.succ i, hi⟩, value =>
      get_set_self xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩ value

/-- Replacing one position leaves every other position unchanged. -/
theorem get_set_of_ne : {ss : List ι} →
    (xs : Pack F ss) → (i j : Fin ss.length) → (value : F (ss.get i)) →
    i ≠ j → (xs.set i value).get j = xs.get j
  | [], .nil, i, _, _, _ => nomatch i
  | _ :: _, .cons _ _, ⟨0, _⟩, ⟨0, _⟩, _, different => absurd rfl different
  | _ :: _, .cons _ _, ⟨0, _⟩, ⟨Nat.succ _, _⟩, _, _ => rfl
  | _ :: _, .cons _ _, ⟨Nat.succ _, _⟩, ⟨0, _⟩, _, _ => rfl
  | _ :: _, .cons _ xs, ⟨Nat.succ i, hi⟩, ⟨Nat.succ j, hj⟩, value, different =>
      get_set_of_ne xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩ ⟨j, Nat.lt_of_succ_lt_succ hj⟩ value
        fun same => different (Fin.ext (congrArg Nat.succ (congrArg Fin.val same)))

/-- Apply an index-preserving function to every entry of a pack. -/
@[simp] def map (f : ∀ {s : ι}, F s → G s) : {ss : List ι} → Pack F ss → Pack G ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs => .cons (f x) (map (f := f) (ss := ss) xs)

/-- Apply an index-preserving effectful function to every entry, from left to right. -/
def mapM {m : Type → Type} [Monad m] (f : ∀ {s : ι}, F s → m (G s)) :
    {ss : List ι} → Pack F ss → m (Pack G ss)
  | [], .nil => pure .nil
  | _ :: _, .cons x xs => do
      let y ← f x
      let ys ← mapM f xs
      pure (.cons y ys)

/-- Combine two packs entrywise with an index-preserving binary function. -/
@[simp] def zipWith (f : ∀ {s : ι}, F s → G s → H s) :
    {ss : List ι} → Pack F ss → Pack G ss → Pack H ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys => .cons (f x y) (zipWith (f := f) (ss := ss) xs ys)

/-- Build the pack whose entry at every index `s` is `f s`. -/
@[simp] def fill (f : (s : ι) → F s) : {ss : List ι} → Pack F ss
  | [] => .nil
  | s :: ss => .cons (f s) (fill f (ss := ss))

/-- Concatenate two packs. -/
def append : {ss₁ ss₂ : List ι} → Pack F ss₁ → Pack F ss₂ → Pack F (ss₁ ++ ss₂)
  | [], _, .nil, ys => ys
  | _ :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a pack at a statically known boundary of its index list. -/
def split : {ss₁ ss₂ : List ι} → Pack F (ss₁ ++ ss₂) → Pack F ss₁ × Pack F ss₂
  | [], _, xs => (.nil, xs)
  | _ :: ss₁, ss₂, .cons x xs =>
      let (xsLeft, xsRight) := split (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xsLeft, xsRight)

/-- Splitting a concatenated pair of packs recovers both inputs. -/
@[simp] theorem split_append {ss₁ ss₂ : List ι} (xs : Pack F ss₁) (ys : Pack F ss₂) :
    split (append xs ys) = (xs, ys) := by
  induction ss₁ with
  | nil => cases xs; rfl
  | cons _ ss₁ ih =>
      cases xs with
      | cons x xs =>
          simp only [append, split]
          rw [ih xs]

/-- Splitting commutes with an index-preserving map on each side of the boundary. -/
@[simp] theorem split_map {ss₁ ss₂ : List ι} (f : ∀ {s : ι}, F s → G s)
    (xs : Pack F (ss₁ ++ ss₂)) :
    split (xs.map f) = ((split xs).1.map f, (split xs).2.map f) := by
  induction ss₁ with
  | nil => rfl
  | cons s ss ih =>
    cases xs with
    | cons x xs => simp only [map, split, ih xs]

/-- Reading the first entry commutes with an index-preserving map. -/
@[simp] theorem head_map {s : ι} {ss : List ι} (f : ∀ {t : ι}, F t → G t)
    (xs : Pack F (s :: ss)) :
    (xs.map f).head = f xs.head := by
  cases xs
  rfl

/-- Append one entry at the end of a pack. -/
def snoc {τ : ι} : {ss : List ι} → Pack F ss → F τ → Pack F (ss ++ [τ])
  | [], .nil, x => .cons x .nil
  | _ :: ss, .cons x xs, last => .cons x (snoc (ss := ss) xs last)

/-- Separate a nonempty pack into its prefix and final entry. -/
def unsnoc {τ : ι} : {ss : List ι} → Pack F (ss ++ [τ]) → Pack F ss × F τ
  | [], .cons x .nil => (.nil, x)
  | _ :: ss, .cons x xs =>
      let (init, last) := unsnoc (ss := ss) xs
      (.cons x init, last)

/-- Appending one entry with `snoc` is appending the one-entry pack. -/
theorem snoc_eq_append {τ : ι} {ss : List ι} (xs : Pack F ss) (x : F τ) :
    xs.snoc x = xs.append (.cons x .nil) := by
  induction xs with
  | nil => rfl
  | cons y ys ih => simp only [snoc, append, ih]

/-- Transport a pack along an equality of index lists. -/
def cast {ss₁ ss₂ : List ι} (h : ss₁ = ss₂) (xs : Pack F ss₁) : Pack F ss₂ :=
  Eq.mp (congrArg (Pack F) h) xs

/-- Casting along `rfl` does nothing. -/
@[simp] theorem cast_rfl {ss : List ι} (xs : Pack F ss) :
    cast rfl xs = xs := by
  rfl

/-- Two successive casts collapse into one along the composed equality. -/
@[simp] theorem cast_cast {ss₁ ss₂ ss₃ : List ι} (h₁ : ss₁ = ss₂) (h₂ : ss₂ = ss₃)
    (xs : Pack F ss₁) :
    cast h₂ (cast h₁ xs) = cast (h₁.trans h₂) xs := by
  cases h₁
  cases h₂
  rfl

/-- A cast of a `cons` keeps its head and casts its tail. -/
theorem cast_cons {s : ι} {ss₁ ss₂ : List ι} (h : s :: ss₁ = s :: ss₂) (h' : ss₁ = ss₂)
    (x : F s) (xs : Pack F ss₁) :
    cast h (.cons x xs) = .cons x (cast h' xs) := by
  cases h'
  rfl

/-- A cast followed by its inverse is the identity. -/
@[simp] theorem cast_symm {ss₁ ss₂ : List ι} (h : ss₁ = ss₂) (xs : Pack F ss₁) :
    cast h.symm (cast h xs) = xs := by
  cases h
  rfl

/-- Appending packs is associative, up to the cast along `List.append_assoc`. -/
theorem append_assoc {ss ts us : List ι} (xs : Pack F ss) (ys : Pack F ts) (zs : Pack F us) :
    cast (List.append_assoc ss ts us) ((xs.append ys).append zs) = xs.append (ys.append zs) := by
  induction ss with
  | nil => cases xs; rfl
  | cons s ss ih =>
    cases xs with
    | cons x xs =>
      change cast (congrArg (List.cons s) (List.append_assoc ss ts us))
        (.cons x ((xs.append ys).append zs)) = .cons x (xs.append (ys.append zs))
      rw [cast_cons _ (List.append_assoc ss ts us), ih xs]

/-- Appending the two halves of a split pack gives the pack back. -/
@[simp] theorem append_split {ps qs : List ι} (xs : Pack F (ps ++ qs)) :
    (split xs).1.append (split xs).2 = xs := by
  induction ps with
  | nil => rfl
  | cons s ps ih =>
    cases xs with
    | cons x xs =>
      simpa only [split, append] using congrArg (Pack.cons x) (ih xs)

/-- A pack cast along an equality of index lists equals any pack heterogeneously equal to it. -/
theorem cast_eq_of_heq {ss ts : List ι} (h : ss = ts) {xs : Pack F ss} {ys : Pack F ts}
    (hxy : HEq xs ys) : cast h xs = ys := by
  cases h
  exact eq_of_heq hxy

/-- Appending heterogeneously equal packs gives heterogeneously equal packs. -/
theorem append_heq {ss ts us vs : List ι} (h : ss = us) (h' : ts = vs) {xs : Pack F ss}
    {ys : Pack F ts} {xs' : Pack F us} {ys' : Pack F vs} (hx : HEq xs xs') (hy : HEq ys ys') :
    HEq (xs.append ys) (xs'.append ys') := by
  cases h
  cases h'
  cases eq_of_heq hx
  cases eq_of_heq hy
  rfl

/-- Appending packs is associative up to heterogeneous equality. -/
theorem append_assoc_heq {ss ts us : List ι} (xs : Pack F ss) (ys : Pack F ts)
    (zs : Pack F us) :
    HEq ((xs.append ys).append zs) (xs.append (ys.append zs)) :=
  (cast_heq _ _).symm.trans (heq_of_eq (append_assoc xs ys zs))

/-- Appending the empty pack gives a pack heterogeneously equal to the original. -/
theorem append_nil_heq {ps : List ι} (xs : Pack F ps) :
    HEq (xs.append (.nil : Pack F [])) xs := by
  induction ps with
  | nil => cases xs; rfl
  | cons s ps ih =>
    cases xs with
    | cons x xs =>
      exact (append_heq (ss := [s]) (us := [s]) rfl (List.append_nil ps)
        (HEq.rfl : HEq (Pack.cons x .nil : Pack F [s]) (.cons x .nil))
        (ih xs)).trans HEq.rfl

/-- `unsnoc` recovers the two arguments supplied to `snoc`. -/
@[simp] theorem unsnoc_snoc {ss : List ι} {τ : ι} (xs : Pack F ss) (x : F τ) :
    unsnoc (snoc xs x) = (xs, x) := by
  induction ss with
  | nil =>
      cases xs
      simp [snoc, unsnoc]
  | cons _ ss ih =>
      cases xs with
      | cons head tail => simp [snoc, unsnoc, ih]

/-- Appending the final entry returned by `unsnoc` to its prefix gives the pack back. -/
@[simp] theorem snoc_unsnoc {ss : List ι} {τ : ι} (xs : Pack F (ss ++ [τ])) :
    snoc (unsnoc xs).1 (unsnoc xs).2 = xs := by
  induction ss with
  | nil =>
      cases xs with
      | cons x xs => cases xs; simp [snoc, unsnoc]
  | cons _ ss ih =>
      cases xs with
      | cons head tail => simp [snoc, unsnoc, ih]

/-- The entries of a pack over a constant family, in order. -/
def toArray {β : Type} : {ss : List ι} → Pack (fun _ => β) ss → Array β
  | [], .nil => #[]
  | _ :: _, .cons x xs => #[x] ++ toArray xs

/-- Listing the entries of a pack gives one array element per index. -/
@[simp] theorem size_toArray {β : Type} {ss : List ι} (xs : Pack (fun _ => β) ss) :
    xs.toArray.size = ss.length := by
  induction xs with
  | nil => rfl
  | cons _ _ ih => simp [toArray, ih, Nat.add_comm]

/--
Build the extended index list together with the pack. Each constructor shares the index-list
suffix returned by the recursive call instead of appending to the original suffix again.
-/
private def snocWithIndices {τ : ι} : {ss : List ι} → Pack F ss → F τ →
    (indices : List ι) × Pack F indices
  | [], .nil, last => ⟨[τ], .cons last .nil⟩
  | s :: ss, .cons first rest, last =>
      let result := snocWithIndices (ss := ss) rest last
      ⟨s :: result.1, .cons (ss := result.1) first result.2⟩

private theorem snocWithIndices_eq {ss : List ι} {τ : ι} (xs : Pack F ss) (last : F τ) :
    snocWithIndices xs last = ⟨ss ++ [τ], snoc xs last⟩ := by
  induction ss with
  | nil =>
      cases xs
      rfl
  | cons s ss ih =>
      cases xs with
      | cons first rest =>
          exact congrArg
            (fun result : (indices : List ι) × Pack F indices =>
              (⟨s :: result.1, .cons (ss := result.1) first result.2⟩ :
                (indices : List ι) × Pack F indices)) (ih rest)

private theorem transport_snd_of_eq {ss : List ι} (result : (indices : List ι) × Pack F indices)
    (xs : Pack F ss) (h : result = ⟨ss, xs⟩) :
    Eq.mp (congrArg (Pack F) (congrArg Sigma.fst h)) result.2 = xs := by
  subst result
  rfl

/-- An implementation of `snoc` that shares index-list suffixes, with the same result type. -/
@[no_expose] def snocLinear {τ : ι} {ss : List ι} (xs : Pack F ss) (last : F τ) :
    Pack F (ss ++ [τ]) :=
  let result := snocWithIndices xs last
  have indices_eq : result.1 = ss ++ [τ] :=
    congrArg Sigma.fst (snocWithIndices_eq xs last)
  Eq.mp (congrArg (Pack F) indices_eq) result.2

private theorem snocLinear_eq_snoc {ss : List ι} {τ : ι} (xs : Pack F ss) (last : F τ) :
    snocLinear xs last = snoc xs last := by
  unfold snocLinear
  exact transport_snd_of_eq _ _ (snocWithIndices_eq xs last)

/-- Compile `snoc` with shared index-list suffixes while keeping its logical definition. -/
@[csimp] theorem snocLinear_eq : @snoc = @snocLinear := by
  funext ι F τ ss xs last
  exact (snocLinear_eq_snoc xs last).symm

end Pack

/-- A pack holding one tensor for every shape in its shape list. -/
abbrev TensorPack (α : Type) [TorchLean.Storage α] : List Shape → Type :=
  Pack (fun s => TorchLean.Tensor α s)

namespace TensorPack

variable {α : Type} [TorchLean.Storage α]

/-- Stack a family of packs, giving every parameter tensor the same leading axes. -/
def stackLeading (leading : Shape) :
    {shapes : List Shape} → (Fin leading.size → TensorPack α shapes) →
      TensorPack α (shapes.map (leading ++ ·))
  | [], _ => .nil
  | shape :: shapes, rows =>
      let first : Tensor α (Shape.dim leading.size shape) :=
        Tensor.dim fun index => (rows index).head
      .cons (Tensor.Rep.reshape
        (s := Shape.dim leading.size shape) (t := leading ++ shape)
        (Shape.size_append leading shape).symm first)
        (stackLeading leading (shapes := shapes) fun index => (rows index).tail)

/-- Evaluate a family of packs in row-major order and stack every parameter tensor. -/
def stackLeadingM {m : Type → Type} [Monad m] (leading : Shape) {shapes : List Shape}
    (f : Fin leading.size → m (TensorPack α shapes)) :
    m (TensorPack α (shapes.map (leading ++ ·))) := do
  let rows ← Tensor.sequenceFinM f
  pure (stackLeading leading rows)

/-- Construct a tensor pack whose every entry contains `value`. -/
@[simp] def fill (value : α) {ss : List Shape} : TensorPack α ss :=
  Pack.fill fun s => TorchLean.Tensor.full s value

/-- Construct the all-zero tensor pack. -/
def zero [Zero α] {ss : List Shape} : TensorPack α ss :=
  fill 0

/-- Print every tensor in a pack in state order, with its statically known shape. -/
instance [Repr α] {shapes : List Shape} : Repr (TensorPack α shapes) where
  reprPrec tensors _ :=
    let rec entries : {ss : List Shape} → TensorPack α ss → List Std.Format
      | [], .nil => []
      | shape :: _, .cons tensor rest =>
          (Std.Format.text (toString shape) ++ ":" ++ Std.Format.line ++ repr tensor) ::
            entries rest
    Std.Format.bracket "["
      (Std.Format.joinSep (entries tensors) ("," ++ Std.Format.line))
      "]"

/-- Add two tensor packs pointwise. -/
def add [Add α] {ss : List Shape}
    (xs ys : TensorPack α ss) : TensorPack α ss :=
  Pack.zipWith (· + ·) xs ys

end TensorPack
end TorchLean
