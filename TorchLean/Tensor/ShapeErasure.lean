/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.SomeTensor
public import TorchLean.Tensor.Pack

/-!
# Tensor Shape-Erasure Boundary

Conversions between statically shape-indexed `TensorPack` values and arrays of existentially
packaged `TorchLean.SomeTensor` values. Runtime tapes and heterogeneous external data use the erased
representation, and typed computation should recover a `TensorPack` as soon as it crosses that
boundary.
-/

@[expose] public section

open TorchLean.Tensor

namespace TorchLean.TensorPack

variable {α : Type} [Storage α]

/-- Erase the individual tensor shapes in a typed pack into a runtime array. -/
def erase : {ss : List Shape} →
    TensorPack α ss → Array (SomeTensor α)
  | [], .nil => #[]
  | _ :: ss, .cons tensor tensors =>
      #[SomeTensor.ofTensor tensor] ++ erase (ss := ss) tensors

/-- Tag each shape-erased tensor with its runtime index, beginning at `start`. -/
def eraseIndexed : {ss : List Shape} →
    TensorPack α ss → Nat → Array (Nat × SomeTensor α)
  | [], .nil, _ => #[]
  | _ :: ss, .cons tensor tensors, index =>
      #[(index, SomeTensor.ofTensor tensor)] ++
        eraseIndexed (ss := ss) tensors (index + 1)

/-- Push the shape-erased tensors of a pack onto `acc`, from the head of the pack onward. -/
def eraseAux : {ss : List Shape} →
    TensorPack α ss → Array (SomeTensor α) → Array (SomeTensor α)
  | [], .nil, acc => acc
  | _ :: ss, .cons tensor tensors, acc =>
      eraseAux (ss := ss) tensors (acc.push (SomeTensor.ofTensor tensor))

/-- Pushing onto `acc` appends the ordinary shape erasure to it. -/
theorem eraseAux_eq : {ss : List Shape} →
    (tensors : TensorPack α ss) → (acc : Array (SomeTensor α)) →
      eraseAux tensors acc = acc ++ erase tensors
  | [], .nil, acc => by simp [eraseAux, erase]
  | _ :: _, .cons tensor tensors, acc => by
      rw [eraseAux, eraseAux_eq tensors, erase]
      simp

/-- `erase` in one pass of pushes, which replaces it in compiled code. -/
def eraseFast {ss : List Shape} (tensors : TensorPack α ss) :
    Array (SomeTensor α) :=
  eraseAux tensors #[]

/-- Compiled code runs `eraseFast` in place of `erase`. -/
@[csimp] theorem eraseFast_eq :
    @erase = @eraseFast := by
  funext α storage ss tensors
  simp [eraseFast, eraseAux_eq]

/-- Push the indexed shape-erased tensors of a pack onto `acc`, numbering from `index`. -/
def eraseIndexedAux : {ss : List Shape} →
    TensorPack α ss → Nat → Array (Nat × SomeTensor α) → Array (Nat × SomeTensor α)
  | [], .nil, _, acc => acc
  | _ :: ss, .cons tensor tensors, index, acc =>
      eraseIndexedAux (ss := ss) tensors (index + 1)
        (acc.push (index, SomeTensor.ofTensor tensor))

/-- Pushing onto `acc` appends the ordinary indexed shape erasure to it. -/
theorem eraseIndexedAux_eq : {ss : List Shape} →
    (tensors : TensorPack α ss) → (index : Nat) → (acc : Array (Nat × SomeTensor α)) →
      eraseIndexedAux tensors index acc =
        acc ++ eraseIndexed tensors index
  | [], .nil, _, acc => by simp [eraseIndexedAux, eraseIndexed]
  | _ :: _, .cons tensor tensors, index, acc => by
      rw [eraseIndexedAux, eraseIndexedAux_eq tensors,
        eraseIndexed]
      simp

/-- `eraseIndexed` in one pass of pushes, which replaces it in compiled code. -/
def eraseIndexedFast {ss : List Shape} (tensors : TensorPack α ss)
    (index : Nat) : Array (Nat × SomeTensor α) :=
  eraseIndexedAux tensors index #[]

/-- Compiled code runs `eraseIndexedFast` in place of the indexed erasure. -/
@[csimp] theorem eraseIndexedFast_eq :
    @eraseIndexed = @eraseIndexedFast := by
  funext α storage ss tensors index
  simp [eraseIndexedFast, eraseIndexedAux_eq]

/--
Recover a statically shape-indexed pack from a prefix of a shape-erased runtime array.

`start` selects the first array entry to consume. The conversion checks every stored shape and
fails if the array is too short or an entry has the wrong shape. Entries after the requested pack
are ignored, so a typed prefix of a larger runtime context can be recovered.
-/
def ofErased (values : Array (SomeTensor α))
    (start : Nat := 0) : {shapes : List Shape} → Except String (TensorPack α shapes)
  | [] => pure .nil
  | shape :: shapes => do
      let value ← match values[start]? with
        | some value => pure value
        | none => throw s!"tensor pack: shape-erased array is missing entry {start}"
      if h : value.shape = shape then
        pure (.cons (value.cast h) (← ofErased values (start + 1) (shapes := shapes)))
      else
        throw <|
          s!"tensor pack: shape mismatch at entry {start} (expected {shape}, got " ++
            s!"{value.shape})"

/-!
## Order and shapes of the erased arrays

Lemmas relating a typed context `TensorPack α ss` to the runtime tape's array of
`SomeTensor α` values. Erasure keeps the order of the pack and stores each tensor's shape in
its `SomeTensor`, so `ofErased` recovers the pack from any array that holds the
erased segment at a known offset.
-/

/--
Every pair `(pid, pg)` in `eraseIndexed ts start` has `pid < start + ss.length`.

We use it to show that the runtime backward pass of a node sends gradient contributions only to
earlier nodes.
-/
theorem mem_eraseIndexed_lt :
    {ss : List Shape} → (ts : TensorPack α ss) → (start : Nat) →
      ∀ {pid : Nat} {pg : SomeTensor α},
        (pid, pg) ∈ eraseIndexed (α := α) (ss := ss) ts start → pid < start + ss.length
  | [], .nil, start, pid, pg, hmem => by
      simp [eraseIndexed] at hmem
  | _ :: ss, .cons _t ts, start, pid, pg, hmem => by
      simp [eraseIndexed] at hmem
      cases hmem with
      | inl h =>
          -- head element: `pid = start`
          have hpid : pid = start := h.1
          cases hpid
          -- After rewriting `(_ :: ss).length`, this is `start < start + Nat.succ ss.length`.
          simp
      | inr h =>
          have := mem_eraseIndexed_lt (ss := ss) ts (start + 1) (pid := pid)
            (pg := pg) h
          -- `start + 1 + ss.length = start + (ss.length + 1)`
          simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this

/--
`erase` ignores type-level casts of the shape list.
-/
@[simp] theorem erase_cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    (xs : TensorPack α ss₁) :
    erase (α := α) (ss := ss₂) (TorchLean.Pack.cast h xs) =
      erase (α := α) (ss := ss₁) xs := by
  cases h
  simp

/-- `erase` has the same size as the underlying shape list. -/
@[simp] theorem size_erase :
    {ss : List Shape} → (xs : TensorPack α ss) →
      (erase (α := α) (ss := ss) xs).size = ss.length
  | [], .nil => by simp [erase]
  | _ :: ss, .cons _t ts => by
      simp [erase, size_erase (ss := ss) ts, Nat.add_comm]

/-- Recovering a typed pack ignores array entries before and after its encoded segment. -/
theorem ofErased_surround (suffix : Array (SomeTensor α))
    : (pref : Array (SomeTensor α)) → {ss : List Shape} → (xs : TensorPack α ss) →
      ofErased (α := α) (pref ++ erase (α := α) xs ++ suffix) pref.size =
        .ok xs
  | _, [], .nil => by
      rfl
  | pref, _ :: ss, .cons x xs => by
      have hhead :
          ((pref ++ #[SomeTensor.ofTensor x]) ++
              (erase (α := α) xs ++ suffix))[pref.size]? =
            some (SomeTensor.ofTensor x) := by
        rw [Array.getElem?_append_left (by simp)]
        simp
      have ih := ofErased_surround suffix
        (pref ++ #[SomeTensor.ofTensor x]) (ss := ss) xs
      simp only [erase, Array.append_assoc]
      rw [← Array.append_assoc (xs := pref) (ys := #[SomeTensor.ofTensor x])]
      simp only [Array.append_assoc] at ih
      simp [Array.append_singleton] at ih
      rw [ofErased, hhead]
      simp [ih]

/-- Decoding at offset `pref.size` recovers `xs` from its erased array appended to any `pref`. -/
@[simp] theorem ofErased_append_erase
    (pref : Array (SomeTensor α)) {ss : List Shape} (xs : TensorPack α ss) :
    ofErased (α := α) (pref ++ erase (α := α) xs) pref.size =
      .ok xs := by
  simpa using ofErased_surround (α := α) #[] pref xs

/-- Decoding the erased array of `xs` returns `.ok xs`. -/
@[simp] theorem ofErased_erase
    {ss : List Shape} (xs : TensorPack α ss) :
    ofErased (α := α) (erase (α := α) xs) = .ok xs := by
  simpa using
    (ofErased_append_erase (α := α) (pref := #[]) (ss := ss) xs)

/-- Splitting a typed context splits its erased array at the same boundary. -/
theorem erase_split {left right : List Shape}
    (xs : TensorPack α (left ++ right)) :
    erase xs =
      erase (Pack.split (ss₁ := left) xs).1 ++
        erase (Pack.split (ss₁ := left) xs).2 := by
  induction left with
  | nil => simp [Pack.split, erase]
  | cons shape left ih =>
      cases xs with
      | cons x xs => simp [Pack.split, erase, ih xs, Array.append_assoc]

/-- Recovering the shapes `left` from the erased array of a pack over `left ++ right` gives the
first component of its `Pack.split`. -/
theorem ofErased_erase_prefix {left right : List Shape}
    (xs : TensorPack α (left ++ right)) :
    ofErased (erase xs) (shapes := left) =
      .ok (Pack.split (ss₁ := left) xs).1 := by
  rw [erase_split (left := left)]
  simpa using ofErased_surround (α := α)
    (erase (Pack.split (ss₁ := left) xs).2) #[] (Pack.split (ss₁ := left) xs).1

/-!
### Shape-erasing conversions

Erasure turns `TorchLean.Pack.snoc` into `Array.push` and `TorchLean.Pack.cons` into a
one-element prefix, and indexing the erased array agrees with `TorchLean.Pack.get`. We use
these facts to match runtime node ids with positions in the typed proof context `Γ ++ ss`.
-/

/--
Erasing `snoc xs t` pushes the erased `t` onto the erased array of `xs`.
-/
@[simp] theorem erase_snoc {ss : List Shape} {τ : Shape} (xs : TensorPack α ss)
    (t : Tensor α τ) :
    erase (α := α) (ss := ss ++ [τ])
        (TorchLean.Pack.snoc (ss := ss) (τ := τ) xs t) =
      (erase (α := α) (ss := ss) xs).push (SomeTensor.ofTensor t) := by
  induction ss with
  | nil =>
      cases xs
      simp [TorchLean.Pack.snoc, erase]
  | cons s ss ih =>
      cases xs with
      | cons x xs =>
          simp [TorchLean.Pack.snoc, erase, ih (xs := xs)]

/-- Erasing `Pack.cons x xs` puts the erased `x` in front of the erased array of `xs`. -/
theorem erase_cons {s : Shape} {ss : List Shape}
    (x : Tensor α s) (xs : TensorPack α ss) :
    erase (α := α) (ss := s :: ss) (Pack.cons x xs) =
      #[SomeTensor.ofTensor x] ++ erase (α := α) (ss := ss) xs := by
  rfl

/--
Entry `i` of the erased array of `xs` is `TorchLean.Pack.get xs i` with its shape erased.

It turns a runtime lookup `arr[i]` into the typed lookup `get xs i`.
-/
theorem get_erase :
    {ss : List Shape} → (xs : TensorPack α ss) → (i : Fin ss.length) →
      let arr := erase (α := α) (ss := ss) xs
      arr[i.1]'(by
        dsimp [arr]
        exact Nat.lt_of_lt_of_eq i.2 (size_erase (α := α) (ss := ss) xs).symm) =
        SomeTensor.ofTensor (TorchLean.Pack.get (ss := ss) xs i)
  | [], .nil, i => by
      cases i with
      | mk val isLt =>
        exact False.elim ((Nat.not_lt_zero val) isLt)
  | _ :: ss, .cons x xs, ⟨0, hi⟩ => by
      -- Identify the literal `0 : Fin (Nat.succ ss.length)` with the matched index `⟨0, hi⟩`, so
      -- that `TorchLean.Pack.get` reduces by `rfl`.
      have h0 : (0 : Fin (Nat.succ ss.length)) = ⟨0, hi⟩ := by
        ext
        rfl
      cases h0
      simp [erase]
      rfl
  | _ :: ss, .cons x xs, ⟨Nat.succ i, hi⟩ => by
      have : i < ss.length := Nat.lt_of_succ_lt_succ hi
      simpa [erase, TorchLean.Pack.get] using
        (get_erase xs ⟨i, this⟩)

end TorchLean.TensorPack
