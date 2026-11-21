/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Numerics.Reduction.Tree
public import TorchLean.Tensor.Contract
public import TorchLean.Tensor.Cumulative
public import TorchLean.Tensor.Elementwise.Real
public import TorchLean.Tensor.Window

/-!
# Reduction orders

A `ReduceOrder` is finite data that gives, for every count `n` of reduced entries, a FloatLib
`ReductionTree` over `Option (Fin n)` (`ReduceOrder.tree`): the leaf `none` is the start value of
the reduction and the leaf `some i` is entry `i`. `ReduceOrder.fold` evaluates that tree with a
step function. It is the one ordered reduction under every ordered operation:

- `reduceIn` for `reduce` (sum, product, maximum, minimum), with `sumIn`, `dotIn` and `meanIn`;
- `contractIn` for `contract`, with `matmulIn` and `batchMatmulIn` (broadcast batch prefixes);
- `poolIn` for the sum and average pools of `pool`; a maximum or minimum pool selects an entry,
  and `poolIn` gives `pool` there;
- `scanIn` for the prefixes of `cumsum` and `cumprod`: the prefix of `k` entries is folded with
  the order's tree over `k` entries;
- `selectIn` for the running selections `cummaxIn` and `cumminIn` of `cummax` and `cummin`: the
  index at a prefix of `k` entries folds its `k` indices with the order's tree over them and
  `ReduceOrder.pick`, which keeps one of two indices;
- `Einops.Lowering.einsumIn`, in `TorchLean/Tensor/Einops/Lowering/Einsum/Order.lean`, for
  `einsum`.

The orders:

- `leftFold`, the chain from the start value through the entries in index order, is the order of
  the existing definitions. `reduceIn_leftFold`, `sumIn_leftFold`, `dotIn_leftFold`,
  `meanIn_leftFold`, `contractIn_leftFold`, `matmulIn_leftFold`, `batchMatmulIn_leftFold`,
  `poolIn_leftFold`, `scanIn_add_leftFold`, `scanIn_mul_leftFold`, `cummaxIn_leftFold` and
  `cumminIn_leftFold` state that it gives the existing function, at every scalar.
- `pairwise` halves the leaves recursively. Its depth is at most `⌈log₂ (n + 1)⌉`
  (`depth_pairwise_le`), where `leftFold` has depth `n`.
- `blocked w` is the addition order of FloatLib's `BlockedDot.kernelEntry`: the start value, then
  one partial per block of `w` entries, left to right. A block chains its own entries from its
  first one, where the kernel chains fused multiply-adds from zero.
- `ofTree t` is a given tree over `Option ℕ`. Over `n` entries its leaves `some i` with `n ≤ i`
  are dropped (`prune`); a tree left with no leaf is the start value alone. Every order is, at
  each count, the `ofTree` of its own tree there (`tree_ofTree`).

An order `Covers` `n` entries when its tree over them has the start value and each entry as leaves,
once each, in any order. Coverage is decidable; `leftFold`, `pairwise` and `blocked w` cover every
count, and an order that covers `k` entries covers every smaller count (`Covers.mono`). Under an
associative and commutative step an order gives the left fold at every count it covers
(`fold_eq_leftFold`), so `reduceIn_eq_reduce`, `meanIn_eq_mean`, `contractIn_eq_contract`,
`batchMatmulIn_eq_batchMatmul`, `poolIn_eq_pool` and `scanIn_eq_leftFold` hold for every order
that covers the count each reduces. `pick` is commutative (`pick_comm`), and associative when its
test compares the images of the entries in a linear order (`pick_assoc`), so `selectIn_eq_leftFold`
holds then, and `cummaxIn_eq_leftFold` and `cumminIn_eq_leftFold` hold at every scalar whose tests
of `cummax` and `cummin` are such comparisons.
`reduceIn_sum_real` and its companions instantiate them over `ℝ` for sums, products, maxima and
minima, and `cummaxIn_real` and `cumminIn_real` for running maxima and minima with their indices.
`TorchLean/Tensor/ConfiguredBinary/Order.lean` instantiates the last two at every configured binary
format, NaN included.
The floating-point error of an ordered sum is bounded by the depth of its order in
`TorchLean/Verify/Floats/IEEE754/Order.lean`.
-/

@[expose] public section

open FloatLib.Numerics (ReductionTree)

namespace TorchLean.Tensor

/-- The order of a reduction: finite data that gives, for each count of entries, the tree that
combines them (`ReduceOrder.tree`). -/
inductive ReduceOrder where
  /-- The chain from the start value through the entries in index order. -/
  | leftFold
  /-- The balanced tree over the start value and the entries. -/
  | pairwise
  /-- The start value, then a chain over the blocks of `width` entries, each block the chain of
  its entries. -/
  | blocked (width : Nat)
  /-- A given tree: the start value at the leaf `none`, entry `i` at the leaf `some i`. -/
  | ofTree (tree : ReductionTree (Option Nat))
  deriving DecidableEq, Repr, Inhabited

namespace ReduceOrder

/-! ## Trees -/

/-- The balanced tree over `g lo, …, g (lo + len)`: the first `len / 2 + 1` leaves on the left,
the others on the right. -/
def span {γ : Type*} (g : Nat → γ) : Nat → Nat → ReductionTree γ
  | lo, 0 => .leaf (g lo)
  | lo, len + 1 => .node (span g lo (len / 2)) (span g (lo + len / 2 + 1) (len - len / 2))
termination_by _ len => len

theorem leaves_span {γ : Type*} (g : Nat → γ) :
    ∀ lo len, (span g lo len).leaves = (List.range' lo (len + 1)).map g
  | lo, 0 => by simp [span, ReductionTree.leaves]
  | lo, len + 1 => by
    rw [span, ReductionTree.leaves, leaves_span g lo (len / 2),
      leaves_span g (lo + len / 2 + 1) (len - len / 2), ← List.map_append,
      show lo + len / 2 + 1 = lo + (len / 2 + 1) by omega, List.range'_append_1]
    congr 2
    omega
termination_by _ len => len

/-- A balanced tree over at most `2 ^ k` leaves has depth at most `k`. -/
theorem depth_span_le {γ : Type*} (g : Nat → γ) :
    ∀ lo len k, len + 1 ≤ 2 ^ k → (span g lo len).depth ≤ k
  | lo, 0, k, _ => by simp [span, ReductionTree.depth]
  | lo, len + 1, k, h => by
    obtain ⟨k, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by cases k <;> simp_all⟩
    rw [pow_succ] at h
    have hl := depth_span_le g lo (len / 2) k (by omega)
    have hr := depth_span_le g (lo + len / 2 + 1) (len - len / 2) k (by omega)
    rw [span, ReductionTree.depth]
    omega
termination_by _ len => len

/-- `acc` followed, left to right, by one node per block of `w` leaves of the list, each block
the chain of its own leaves. -/
def blocks {γ : Type*} (w : Nat) (acc : ReductionTree γ) : List γ → ReductionTree γ
  | [] => acc
  | x :: xs => blocks w (.node acc (.leftChain x (xs.take (w - 1)))) (xs.drop (w - 1))
termination_by l => l.length
decreasing_by simp only [List.length_drop, List.length_cons]; omega

theorem leaves_blocks {γ : Type*} (w : Nat) :
    ∀ (acc : ReductionTree γ) (l : List γ), (blocks w acc l).leaves = acc.leaves ++ l
  | acc, [] => by simp [blocks]
  | acc, x :: xs => by
    rw [blocks, leaves_blocks, ReductionTree.leaves, ReductionTree.leaves_leftChain,
      List.append_assoc, List.cons_append, List.take_append_drop]
termination_by _ l => l.length
decreasing_by simp only [List.length_drop, List.length_cons]; omega

/-- The tree with each leaf `x` replaced by the leaf `y` where `keep x = some y` and dropped where
it is `none`: a node left with one child is that child, a tree left with no leaf is `none`. -/
def prune {γ δ : Type*} (keep : γ → Option δ) : ReductionTree γ → Option (ReductionTree δ)
  | .leaf x => (keep x).map .leaf
  | .node a b =>
    match prune keep a, prune keep b with
    | some a, some b => some (.node a b)
    | a, none => a
    | none, b => b

/-- The leaves of a pruned tree are the kept leaves, in order. -/
theorem leaves_prune {γ δ : Type*} (keep : γ → Option δ) (t : ReductionTree γ) :
    ((prune keep t).map ReductionTree.leaves).getD [] = t.leaves.filterMap keep := by
  induction t with
  | leaf x => cases h : keep x <;> simp [prune, ReductionTree.leaves, h]
  | node a b ha hb =>
    simp only [prune, ReductionTree.leaves, List.filterMap_append, ← ha, ← hb]
    generalize prune keep a = pa
    generalize prune keep b = pb
    cases pa <;> cases pb <;> simp [ReductionTree.leaves]

/-- Pruning that keeps every leaf relabels them. -/
theorem prune_of_isSome {γ δ : Type*} {keep : γ → Option δ} (d : δ) {t : ReductionTree γ}
    (h : ∀ x ∈ t.leaves, (keep x).isSome) :
    prune keep t = some (t.map fun x => (keep x).getD d) := by
  induction t with
  | leaf x =>
    obtain ⟨y, hy⟩ := Option.isSome_iff_exists.1 (h x (by simp [ReductionTree.leaves]))
    simp [prune, ReductionTree.map, hy]
  | node a b ha hb =>
    simp only [ReductionTree.leaves, List.mem_append] at h
    simp [prune, ReductionTree.map, ha fun x hx => h x (.inl hx), hb fun x hx => h x (.inr hx)]

/-- The leaf of `n` entries that a leaf of a given tree names: the start value, or entry `i` when
`i < n`. -/
def entry (n : Nat) : Option Nat → Option (Option (Fin n))
  | none => some none
  | some i => if h : i < n then some (some ⟨i, h⟩) else none

/-- The tree of the order over `n` entries. -/
def tree : ReduceOrder → (n : Nat) → ReductionTree (Option (Fin n))
  | leftFold, n => .leftChain none ((List.finRange n).map some)
  | pairwise, n =>
    span (fun | 0 => none | j + 1 => if h : j < n then some ⟨j, h⟩ else none) 0 n
  | blocked w, n => blocks w (.leaf none) ((List.finRange n).map some)
  | ofTree t, n => (prune (entry n) t).getD (.leaf none)

/-- At each count an order is the given tree of its own tree there. -/
theorem tree_ofTree (order : ReduceOrder) (n : Nat) :
    (ofTree ((order.tree n).map (Option.map Fin.val))).tree n = order.tree n := by
  have key : ∀ t : ReductionTree (Option (Fin n)),
      prune (entry n) (t.map (Option.map Fin.val)) = some t := by
    intro t
    induction t with
    | leaf x => cases x <;> simp [ReductionTree.map, prune, entry]
    | node a b ha hb => simp [ReductionTree.map, prune, ha, hb]
  simp only [tree, key, Option.getD_some]

/-- The pairwise tree over `n` entries has depth at most `⌈log₂ (n + 1)⌉`. -/
theorem depth_pairwise_le (n : Nat) : (pairwise.tree n).depth ≤ Nat.clog 2 (n + 1) :=
  depth_span_le _ 0 n _ (Nat.le_pow_clog (by norm_num) _)

/-- The tree over `n` entries evaluated with `step`, reading `start` at `none` and `value i` at
`some i`. -/
def fold {β : Type*} (order : ReduceOrder) (n : Nat) (step : β → β → β) (start : β)
    (value : Fin n → β) : β :=
  (order.tree n).eval step (·.elim start value)

/-- The left fold steps from the start value through the entries in index order. -/
theorem fold_leftFold {β : Type*} (n : Nat) (step : β → β → β) (start : β) (value : Fin n → β) :
    leftFold.fold n step start value =
      (List.finRange n).foldl (fun acc i => step acc (value i)) start := by
  simp [fold, tree, ReductionTree.eval_leftChain, List.foldl_map]

/-! ## Coverage -/

/-- The tree over `n` entries has the start value and each entry as leaves, once each. -/
def Covers (order : ReduceOrder) (n : Nat) : Prop :=
  (order.tree n).leaves.Perm (none :: (List.finRange n).map some)

instance (order : ReduceOrder) (n : Nat) : Decidable (order.Covers n) :=
  inferInstanceAs (Decidable (List.Perm _ _))

/-- An order covers `n` entries exactly when its tree over them repeats no leaf and has every
leaf. -/
theorem covers_iff {order : ReduceOrder} {n : Nat} :
    order.Covers n ↔ (order.tree n).leaves.Nodup ∧ ∀ x, x ∈ (order.tree n).leaves := by
  have hnodup : (none :: (List.finRange n).map some).Nodup :=
    List.nodup_cons.2 ⟨by simp, (List.nodup_finRange n).map (Option.some_injective _)⟩
  have hmem : ∀ x : Option (Fin n), x ∈ none :: (List.finRange n).map some := by
    rintro (_ | i) <;> simp
  refine ⟨fun h => ⟨h.nodup_iff.2 hnodup, fun x => h.mem_iff.2 (hmem x)⟩, fun ⟨hn, hm⟩ => ?_⟩
  exact (List.perm_ext_iff_of_nodup hn hnodup).2 fun x => iff_of_true (hm x) (hmem x)

theorem covers_leftFold (n : Nat) : leftFold.Covers n := by
  simp [Covers, tree, ReductionTree.leaves_leftChain]

theorem covers_pairwise (n : Nat) : pairwise.Covers n := by
  unfold Covers
  refine List.Perm.of_eq (List.ext_getElem (by simp [tree, leaves_span]) fun i h₁ h₂ => ?_)
  simp only [tree, leaves_span, List.getElem_map, List.getElem_range']
  cases i with
  | zero => simp
  | succ i =>
    have hi : i < n := by simp at h₂; omega
    simp [hi]

theorem covers_blocked (w n : Nat) : (blocked w).Covers n := by
  simp [Covers, tree, leaves_blocks, ReductionTree.leaves]

/-- Over `n` entries a given tree has its leaves kept by `entry n`, or the start value alone when
it keeps none. -/
theorem leaves_ofTree (t : ReductionTree (Option Nat)) (n : Nat) :
    ((ofTree t).tree n).leaves =
      if t.leaves.filterMap (entry n) = [] then [none] else t.leaves.filterMap (entry n) := by
  have h := leaves_prune (entry n) t
  simp only [tree]
  cases hp : prune (entry n) t with
  | none => simp_all [ReductionTree.leaves]
  | some u =>
    simp only [hp, Option.map_some, Option.getD_some] at h ⊢
    rw [← h, ite_eq_right fun hu => by simpa [hu] using u.length_leaves]

/-- An order that covers `k` entries covers every smaller count. -/
theorem Covers.mono {order : ReduceOrder} {k m : Nat} (h : order.Covers k) (hm : m ≤ k) :
    order.Covers m := by
  cases order with
  | leftFold => exact covers_leftFold m
  | pairwise => exact covers_pairwise m
  | blocked w => exact covers_blocked w m
  | ofTree t =>
    -- The leaves kept over `m` entries are those kept over `k`, less the entries from `m` on.
    have hkm : t.leaves.filterMap (entry m) =
        (t.leaves.filterMap (entry k)).filterMap (entry m ∘ Option.map Fin.val) := by
      rw [List.filterMap_filterMap]
      refine List.filterMap_congr fun x _ => ?_
      rcases x with _ | i
      · rfl
      · by_cases hik : i < k
        · simp [entry, hik]
        · simp [entry, hik, show ¬i < m by omega]
    rw [covers_iff, leaves_ofTree] at h ⊢
    by_cases hk : t.leaves.filterMap (entry k) = []
    · rw [ite_eq_left hk] at h
      rw [hkm, hk, List.filterMap_nil, ite_eq_left rfl]
      refine ⟨List.nodup_singleton _, fun x => ?_⟩
      rcases x with _ | i
      · exact List.mem_singleton_self _
      · simpa using h.2 (some (Fin.castLE hm i))
    · rw [ite_eq_right hk] at h
      have hnone : none ∈ t.leaves.filterMap (entry m) := by
        rw [hkm, List.mem_filterMap]
        exact ⟨none, h.2 none, rfl⟩
      rw [ite_eq_right (List.ne_nil_of_mem hnone), hkm]
      refine ⟨h.1.filterMap fun a a' b ha ha' => ?_, fun x => ?_⟩
      · rcases a with _ | a <;> rcases a' with _ | a' <;> rcases b with _ | b <;>
          simp only [Function.comp_apply, Option.map_none, Option.map_some, entry,
            Option.mem_def, Option.some.injEq, reduceCtorEq] at ha ha' <;>
          (try split at ha) <;> (try split at ha') <;> simp_all [Fin.ext_iff]
      · rw [List.mem_filterMap]
        rcases x with _ | i
        · exact ⟨none, h.2 none, rfl⟩
        · exact ⟨some (Fin.castLE hm i), h.2 _, by simp [entry, i.isLt]⟩

/-- Trees whose leaves are permutations of each other evaluate alike under an associative and
commutative step. -/
theorem eval_eq_of_perm {γ β : Type*} {a b : ReductionTree γ} (h : a.leaves.Perm b.leaves)
    {step : β → β → β} (hassoc : ∀ x y z, step (step x y) z = step x (step y z))
    (hcomm : ∀ x y, step x y = step y x) (value : γ → β) :
    a.eval step value = b.eval step value := by
  have : Std.Associative step := ⟨hassoc⟩
  have : Std.Commutative step := ⟨hcomm⟩
  have key : ∀ (t : ReductionTree γ) (init : Option β),
      (t.leaves.map (some ∘ value)).foldl (Option.merge step) init =
        Option.merge step init (some (t.eval step value)) := by
    intro t
    induction t with
    | leaf x => intro init; rfl
    | node l r hl hr =>
      intro init
      simp only [ReductionTree.leaves, ReductionTree.eval, List.map_append, List.foldl_append,
        hl, hr]
      rw [Std.Associative.assoc (op := Option.merge step), Option.merge_some_some]
  have hswap : ∀ x y z : Option β, Option.merge step (Option.merge step z x) y =
      Option.merge step (Option.merge step z y) x := by
    intro x y z
    rw [Std.Associative.assoc (op := Option.merge step),
      Std.Commutative.comm (op := Option.merge step) x y,
      ← Std.Associative.assoc (op := Option.merge step)]
  have hfold := (h.map (some ∘ value)).foldl_eq' (fun x _ y _ z => hswap x y z) none
  rw [key, key] at hfold
  simpa using hfold

/-- Under an associative and commutative step an order gives the left fold at every count it
covers. -/
theorem fold_eq_leftFold {β : Type*} {order : ReduceOrder} {n : Nat} (hcov : order.Covers n)
    {step : β → β → β} (hassoc : ∀ x y z, step (step x y) z = step x (step y z))
    (hcomm : ∀ x y, step x y = step y x) (start : β) (value : Fin n → β) :
    order.fold n step start value = leftFold.fold n step start value :=
  eval_eq_of_perm (List.Perm.trans hcov (List.Perm.symm (covers_leftFold n))) hassoc hcomm _

/-! ## Selections -/

/-- Of two indices, the later one when `move` holds of the entries at the earlier one and the later
one, and the earlier one otherwise. -/
def pick {β : Type*} {n : Nat} (move : β → β → Bool) (value : Fin n → β) (a b : Fin n) : Fin n :=
  if a ≤ b then (if move (value a) (value b) then b else a)
  else if move (value b) (value a) then a else b

theorem pick_comm {β : Type*} {n : Nat} (move : β → β → Bool) (value : Fin n → β) (a b : Fin n) :
    pick move value a b = pick move value b a := by
  unfold pick
  rcases lt_trichotomy a b with h | rfl | h
  · simp [h.le, not_le.2 h]
  · rfl
  · simp [h.le, not_le.2 h]

/-- When `move` compares the images of the entries under a map into a linear order, `pick` keeps
the index with the larger image, the later one at a tie, and is associative. -/
theorem pick_assoc {β γ : Type*} [LinearOrder γ] {n : Nat} {move : β → β → Bool} (f : β → γ)
    (hmove : ∀ p q, move p q = decide (f p ≤ f q)) (value : Fin n → β) (a b c : Fin n) :
    pick move value (pick move value a b) c = pick move value a (pick move value b c) := by
  let key : Fin n → γ ×ₗ Fin n := fun i => toLex (f (value i), i)
  have hpick : ∀ a b, pick move value a b = if key a ≤ key b then b else a := by
    intro a b
    simp only [pick, hmove, decide_eq_true_eq, key, Prod.Lex.toLex_le_toLex]
    by_cases hab : a ≤ b
    · simp only [hab, ite_true, and_true, ← le_iff_lt_or_eq]
    · simp only [hab, ite_false, and_false, or_false, ← not_le, ite_not]
  have hkey : ∀ a b, key (pick move value a b) = max (key a) (key b) := fun a b => by
    rw [hpick, max_def]
    exact apply_ite key _ _ _
  have hinj : ∀ a b, key a = key b → a = b := fun a b h => (Prod.ext_iff.1 (toLex.injective h)).2
  exact hinj _ _ (by simp only [hkey, max_assoc])

/-- Over indices in increasing order, from a start at or below each, the fold of `pick` moves to
the next index exactly when `move` holds of the entries at the current one and the next. -/
theorem foldl_pick {β : Type*} {n : Nat} (move : β → β → Bool) (value : Fin n → β) :
    ∀ (l : List (Fin n)) (start : Fin n), l.Pairwise (· ≤ ·) → (∀ j ∈ l, start ≤ j) →
      l.foldl (pick move value) start =
        l.foldl (fun best j => if move (value best) (value j) then j else best) start
  | [], _, _, _ => rfl
  | j :: l, start, hl, hstart => by
    have hj : pick move value start j = if move (value start) (value j) then j else start :=
      ite_eq_left (hstart j List.mem_cons_self)
    rw [List.foldl_cons, List.foldl_cons, hj]
    refine foldl_pick move value l _ (List.pairwise_cons.1 hl).2 fun k hk => ?_
    split
    · exact (List.pairwise_cons.1 hl).1 k hk
    · exact hstart k (List.mem_cons_of_mem _ hk)

/-- The left fold of `pick` over the first `m` indices from index `0` is the fold that moves to
the next index when `move` holds. -/
theorem fold_leftFold_pick {β : Type*} {m n : Nat} (move : β → β → Bool) (value : Fin n → β)
    (h : m ≤ n) (hn : 0 < n) :
    leftFold.fold m (pick move value) ⟨0, hn⟩ (Fin.castLE h) =
      ((List.finRange m).map (Fin.castLE h)).foldl
        (fun best j => if move (value best) (value j) then j else best) ⟨0, hn⟩ := by
  rw [fold_leftFold, ← foldl_pick move value _ _
    ((List.pairwise_lt_finRange m).map _ fun a b hab => Nat.le_of_lt hab)
    (fun j _ => Nat.zero_le j.val), List.foldl_map]

end ReduceOrder

/-! ## Ordered reductions -/

section

variable {α : Type} [Storage α] {s t : Shape}

/-- `reduce` in the order `order`: each output entry folds its fiber with `order.fold`, from the
seed of `op`, or from the first entry of the fiber when `op` has none. -/
def reduceIn (order : ReduceOrder) (op : ReduceOp) [ReduceOp.Scalar op α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := op) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d := by
      simp) :
    Tensor α (s.reduce dims keepdim) :=
  Rep.ofFn fun o =>
    let value := fun i : Fin (Shape.size (s.selectAxes dims)) =>
      x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))
    match hs : ReduceOp.Scalar.seed (op := op) (α := α) with
    | some seed => order.fold _ (ReduceOp.Scalar.step (op := op)) seed value
    | none =>
        order.fold _ (ReduceOp.Scalar.step (op := op))
          (value ⟨0, Shape.size_selectAxes_pos (h (by simp [hs]))⟩) value

theorem reduceIn_leftFold (op : ReduceOp) [ReduceOp.Scalar op α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := op) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d) :
    reduceIn .leftFold op x dims keepdim h = reduce op x dims keepdim h := by
  simp only [reduceIn, reduce, ReduceOrder.fold_leftFold]
  rfl

/-- Under an associative and commutative step, `reduceIn` in an order that covers the fiber is
`reduce`. -/
theorem reduceIn_eq_reduce {order : ReduceOrder} (op : ReduceOp) [ReduceOp.Scalar op α]
    (hassoc : ∀ x y z : α, ReduceOp.Scalar.step (op := op) (ReduceOp.Scalar.step (op := op) x y) z =
      ReduceOp.Scalar.step (op := op) x (ReduceOp.Scalar.step (op := op) y z))
    (hcomm : ∀ x y : α, ReduceOp.Scalar.step (op := op) x y = ReduceOp.Scalar.step (op := op) y x)
    (x : Tensor α s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := op) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (hcov : order.Covers (Shape.size (s.selectAxes dims))) :
    reduceIn order op x dims keepdim h = reduce op x dims keepdim h := by
  rw [← reduceIn_leftFold]
  simp only [reduceIn, ReduceOrder.fold_eq_leftFold hcov hassoc hcomm]

/-- `sum` in the order `order`. -/
def sumIn [Add α] [Zero α] (order : ReduceOrder) (x : Tensor α s) : α :=
  item (castShape (reduceIn order .sum x (List.finRange s.rank) false) (Shape.reduce_finRange s))

theorem sumIn_leftFold [Add α] [Zero α] (x : Tensor α s) : sumIn .leftFold x = sum x := by
  rw [sumIn, reduceIn_leftFold, sum]

/-- `dot` in the order `order`. -/
def dotIn [Add α] [Mul α] [Zero α] (order : ReduceOrder) (a b : Tensor α s) : α :=
  sumIn order (a * b)

theorem dotIn_leftFold [Add α] [Mul α] [Zero α] (a b : Tensor α s) :
    dotIn .leftFold a b = dot a b :=
  sumIn_leftFold _

/-- `mean` in the order `order`: the ordered sum divided by the count. -/
def meanIn [Context α] (order : ReduceOrder) (x : Tensor α s) (dims : List (Fin s.rank))
    (keepdim : Bool) : Tensor α (s.reduce dims keepdim) :=
  let count := Shape.size (s.selectAxes dims)
  map (Context.divNat · (if count = 0 then 1 else count)) (reduceIn order .sum x dims keepdim)

theorem meanIn_leftFold [Context α] (x : Tensor α s) (dims : List (Fin s.rank)) (keepdim : Bool) :
    meanIn .leftFold x dims keepdim = mean x dims keepdim := by
  simp only [meanIn, mean, reduceIn_leftFold]

/-- Under associative and commutative addition, `meanIn` in an order that covers the fiber is
`mean`. -/
theorem meanIn_eq_mean [Context α] {order : ReduceOrder}
    (hassoc : ∀ x y z : α, x + y + z = x + (y + z)) (hcomm : ∀ x y : α, x + y = y + x)
    (x : Tensor α s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (hcov : order.Covers (Shape.size (s.selectAxes dims))) :
    meanIn order x dims keepdim = mean x dims keepdim := by
  rw [← meanIn_leftFold]
  simp only [meanIn, reduceIn_eq_reduce .sum hassoc hcomm x dims keepdim _ hcov,
    reduceIn_leftFold]

/-- `contract` in the order `order`: each output entry adds the products of its contracted
entries with `order.fold`, from zero. -/
def contractIn [Add α] [Mul α] [Zero α] (order : ReduceOrder) (x : Tensor α s) (y : Tensor α t)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (out : Shape)
    (h : Shape.ContractValid s t dims batchDims)
    (hout : out = s.contract t dims batchDims := by tensor_shape) : Tensor α out :=
  Rep.ofFn fun o =>
    order.fold _ (· + ·) 0 fun i =>
      x (s.contractLeft t dims batchDims (Shape.Coord.reindexEq hout o)
          (Shape.Coord.unlinearize i)) *
        y (h.contractRight (Shape.Coord.reindexEq hout o) (Shape.Coord.unlinearize i))

theorem contractIn_leftFold [Add α] [Mul α] [Zero α] (x : Tensor α s) (y : Tensor α t)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (out : Shape)
    (h : Shape.ContractValid s t dims batchDims) (hout : out = s.contract t dims batchDims) :
    contractIn .leftFold x y dims batchDims out h hout =
      contract x y dims batchDims out h hout := by
  simp only [contractIn, contract, ReduceOrder.fold_leftFold]

/-- Under associative and commutative addition, `contractIn` in an order that covers the
contracted coordinates is `contract`. -/
theorem contractIn_eq_contract [AddCommSemigroup α] [Mul α] [Zero α] {order : ReduceOrder}
    (x : Tensor α s) (y : Tensor α t) (dims batchDims : List (Fin s.rank × Fin t.rank))
    (out : Shape) (h : Shape.ContractValid s t dims batchDims)
    (hout : out = s.contract t dims batchDims)
    (hcov : order.Covers (Shape.size ((dims.map Prod.fst).map s.extent))) :
    contractIn order x y dims batchDims out h hout = contract x y dims batchDims out h hout := by
  rw [← contractIn_leftFold]
  simp only [contractIn, ReduceOrder.fold_eq_leftFold (β := α) hcov add_assoc add_comm]

/-- `matmul` in the order `order`. -/
def matmulIn [Add α] [Mul α] [Zero α] {m n p : Nat} (order : ReduceOrder) (A : Tensor α [m, n])
    (B : Tensor α [n, p]) : Tensor α [m, p] :=
  contractIn order A B _ _ [m, p] .matMul

theorem matmulIn_leftFold [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α [m, n])
    (B : Tensor α [n, p]) : matmulIn .leftFold A B = matmul A B :=
  contractIn_leftFold ..

/-- `matmulCommonBatch` in the order `order`: `matmulIn` at each coordinate of the shared batch
prefix. -/
def matmulCommonBatchIn [Add α] [Mul α] [Zero α] {batch : Shape} {m n p : Nat}
    (order : ReduceOrder) (A : Tensor α (batch ++ [m, n])) (B : Tensor α (batch ++ [n, p])) :
    Tensor α (batch ++ [m, p]) :=
  match batch with
  | .scalar => matmulIn order A B
  | .dim _ _ =>
      Tensor.dim fun index =>
        matmulCommonBatchIn order (Tensor.unstack A index) (Tensor.unstack B index)

theorem matmulCommonBatchIn_leftFold [Add α] [Mul α] [Zero α] {batch : Shape} {m n p : Nat}
    (A : Tensor α (batch ++ [m, n])) (B : Tensor α (batch ++ [n, p])) :
    matmulCommonBatchIn .leftFold A B = matmulCommonBatch A B := by
  induction batch with
  | scalar => exact matmulIn_leftFold A B
  | dim k rest ih => simp only [matmulCommonBatchIn, matmulCommonBatch, ih]

/-- `batchMatmul` in the order `order`: both batch prefixes broadcast to `batch`, then
`matmulCommonBatchIn`. -/
def batchMatmulIn [Add α] [Mul α] [Zero α] {batchA batchB batch : Shape} {m n p : Nat}
    (order : ReduceOrder) (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (A : Tensor α (batchA ++ [m, n]))
    (B : Tensor α (batchB ++ [n, p])) : Tensor α (batch ++ [m, p]) :=
  have hA := Shape.CanBroadcastTo.concat_right [m, n] broadcastA
  have hB := Shape.CanBroadcastTo.concat_right [n, p] broadcastB
  matmulCommonBatchIn order (Tensor.broadcast A _ hA.dims hA.broadcastDims)
    (Tensor.broadcast B _ hB.dims hB.broadcastDims)

theorem batchMatmulIn_leftFold [Add α] [Mul α] [Zero α] {batchA batchB batch : Shape}
    {m n p : Nat} (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (A : Tensor α (batchA ++ [m, n]))
    (B : Tensor α (batchB ++ [n, p])) :
    batchMatmulIn .leftFold broadcastA broadcastB A B = batchMatmul broadcastA broadcastB A B :=
  matmulCommonBatchIn_leftFold _ _

/-- Under associative and commutative addition, `matmulCommonBatchIn` in an order that covers the
inner dimension is `matmulCommonBatch`. -/
theorem matmulCommonBatchIn_eq_matmulCommonBatch [AddCommSemigroup α] [Mul α] [Zero α]
    {order : ReduceOrder} {batch : Shape} {m n p : Nat} (A : Tensor α (batch ++ [m, n]))
    (B : Tensor α (batch ++ [n, p])) (hcov : order.Covers n) :
    matmulCommonBatchIn order A B = matmulCommonBatch A B := by
  induction batch with
  | scalar =>
      refine contractIn_eq_contract A B _ _ _ _ _ (hcov.mono ?_)
      exact Nat.le_of_eq (Shape.size_vector n)
  | dim k rest ih => simp only [matmulCommonBatchIn, matmulCommonBatch, ih]

/-- Under associative and commutative addition, `batchMatmulIn` in an order that covers the inner
dimension is `batchMatmul`. -/
theorem batchMatmulIn_eq_batchMatmul [AddCommSemigroup α] [Mul α] [Zero α] {order : ReduceOrder}
    {batchA batchB batch : Shape} {m n p : Nat} (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) (A : Tensor α (batchA ++ [m, n]))
    (B : Tensor α (batchB ++ [n, p])) (hcov : order.Covers n) :
    batchMatmulIn order broadcastA broadcastB A B = batchMatmul broadcastA broadcastB A B :=
  matmulCommonBatchIn_eq_matmulCommonBatch _ _ hcov

/-- A cumulative scan in the order `order`: the entry at `c` folds the prefix of its fiber up to
`c` with `order.fold`, from `init`. -/
def scanIn (order : ReduceOrder) (step : α → α → α) (init : α) (x : Tensor α s)
    (dim : Fin s.rank) : Tensor α s :=
  Rep.ofFn fun c => order.fold _ step init fun i : Fin ((Shape.Coord.get c dim).val + 1) =>
    x (Shape.Coord.update c dim (Fin.castLE (Shape.Coord.get c dim).isLt i))

theorem scanIn_add_leftFold [Add α] [Zero α] (x : Tensor α s) (dim : Fin s.rank) :
    scanIn .leftFold (· + ·) 0 x dim = cumsum x dim := by
  simp only [scanIn, cumsum, ReduceOrder.fold_leftFold, Tensor.foldl_ofFn]

theorem scanIn_mul_leftFold [Mul α] [One α] (x : Tensor α s) (dim : Fin s.rank) :
    scanIn .leftFold (· * ·) 1 x dim = cumprod x dim := by
  simp only [scanIn, cumprod, ReduceOrder.fold_leftFold, Tensor.foldl_ofFn]

/-- Under an associative and commutative step, a scan in an order that covers the length of the
axis is the left-fold scan. -/
theorem scanIn_eq_leftFold {order : ReduceOrder} {step : α → α → α}
    (hassoc : ∀ x y z, step (step x y) z = step x (step y z)) (hcomm : ∀ x y, step x y = step y x)
    (init : α) (x : Tensor α s) (dim : Fin s.rank) (hcov : order.Covers (s.extent dim)) :
    scanIn order step init x dim = scanIn .leftFold step init x dim := by
  simp only [scanIn,
    fun c : s.Coord => ReduceOrder.fold_eq_leftFold (hcov.mono (Shape.Coord.get c dim).isLt)
      hassoc hcomm (n := (Shape.Coord.get c dim).val + 1)]

/-- A running selection along `dim` in the order `order`, with its indices: the index at `c` folds
the indices of the prefix of its fiber up to `c` with `order.fold` and `ReduceOrder.pick`, from
index `0`, and the value is the input at that index. -/
def selectIn (order : ReduceOrder) (move : α → α → Bool) (x : Tensor α s) (dim : Fin s.rank) :
    Tensor α s × Tensor (Fin (s.extent dim)) s :=
  let indices : Tensor (Fin (s.extent dim)) s := Rep.ofFn fun c =>
    order.fold _ (ReduceOrder.pick move fun j => x (Shape.Coord.update c dim j))
      ⟨0, Nat.zero_lt_of_lt (Shape.Coord.get c dim).isLt⟩
      (Fin.castLE (Shape.Coord.get c dim).isLt)
  (Rep.ofFn fun c => x (Shape.Coord.update c dim (indices c)), indices)

/-- When `ReduceOrder.pick` is associative on every fiber, a selection in an order that covers the
length of the axis is the left-fold selection. -/
theorem selectIn_eq_leftFold {order : ReduceOrder} {move : α → α → Bool} {dim : Fin s.rank}
    (hassoc : ∀ (value : Fin (s.extent dim) → α) a b c, ReduceOrder.pick move value
      (ReduceOrder.pick move value a b) c =
        ReduceOrder.pick move value a (ReduceOrder.pick move value b c))
    (x : Tensor α s) (hcov : order.Covers (s.extent dim)) :
    selectIn order move x dim = selectIn .leftFold move x dim := by
  simp only [selectIn, fun (c : s.Coord) (value : Fin (s.extent dim) → α) =>
    ReduceOrder.fold_eq_leftFold (step := ReduceOrder.pick move value)
      (hcov.mono (Shape.Coord.get c dim).isLt) (hassoc value) (ReduceOrder.pick_comm move value)
      (n := (Shape.Coord.get c dim).val + 1)]

section Extrema

variable [BEq α] [LT α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- `cummax` in the order `order`: the selection that moves to an entry that is not equal to
itself (NaN), or, from an entry equal to itself, to one that it does not strictly exceed. -/
def cummaxIn (order : ReduceOrder) (x : Tensor α s) (dim : Fin s.rank) :
    Tensor α s × Tensor (Fin (s.extent dim)) s :=
  selectIn order (fun best a => !(a == a) || ((best == best) && !decide (best > a))) x dim

/-- `cummin` in the order `order`: the selection that moves to an entry that is not equal to
itself (NaN), or, from an entry equal to itself, to one that does not strictly exceed it. -/
def cumminIn (order : ReduceOrder) (x : Tensor α s) (dim : Fin s.rank) :
    Tensor α s × Tensor (Fin (s.extent dim)) s :=
  selectIn order (fun best a => !(a == a) || ((best == best) && !decide (a > best))) x dim

theorem cummaxIn_leftFold (x : Tensor α s) (dim : Fin s.rank) :
    cummaxIn .leftFold x dim = cummax x dim := by
  simp only [cummaxIn, selectIn, cummax, ReduceOrder.fold_leftFold_pick]

theorem cumminIn_leftFold (x : Tensor α s) (dim : Fin s.rank) :
    cumminIn .leftFold x dim = cummin x dim := by
  simp only [cumminIn, selectIn, cummin, ReduceOrder.fold_leftFold_pick]

/-- When the test of `cummaxIn` moves from `p` to `q` exactly when `f p ≤ f q`, for a map `f` into a
linear order, running maxima in an order that covers the length of the axis are the left fold's,
indices included. -/
theorem cummaxIn_eq_leftFold {γ : Type*} [LinearOrder γ] (f : α → γ)
    (hf : ∀ p q, (!(q == q) || ((p == p) && !decide (p > q))) = decide (f p ≤ f q))
    {order : ReduceOrder} (x : Tensor α s) (dim : Fin s.rank)
    (hcov : order.Covers (s.extent dim)) :
    cummaxIn order x dim = cummaxIn .leftFold x dim := by
  rw [cummaxIn_leftFold, cummaxIn,
    selectIn_eq_leftFold (fun v a b c => ReduceOrder.pick_assoc f hf v a b c) x hcov]
  exact cummaxIn_leftFold x dim

/-- When the test of `cumminIn` moves from `p` to `q` exactly when `f p ≤ f q`, for a map `f` into a
linear order, running minima in an order that covers the length of the axis are the left fold's,
indices included. -/
theorem cumminIn_eq_leftFold {γ : Type*} [LinearOrder γ] (f : α → γ)
    (hf : ∀ p q, (!(q == q) || ((p == p) && !decide (q > p))) = decide (f p ≤ f q))
    {order : ReduceOrder} (x : Tensor α s) (dim : Fin s.rank)
    (hcov : order.Covers (s.extent dim)) :
    cumminIn order x dim = cumminIn .leftFold x dim := by
  rw [cumminIn_leftFold, cumminIn,
    selectIn_eq_leftFold (fun v a b c => ReduceOrder.pick_assoc f hf v a b c) x hcov]
  exact cumminIn_leftFold x dim

/-- A value of `cummaxIn` is the input at its index. -/
theorem cummaxIn_fst_apply (order : ReduceOrder) (x : Tensor α s) (dim : Fin s.rank)
    (c : s.Coord) :
    (cummaxIn order x dim).1 c = x (Shape.Coord.update c dim ((cummaxIn order x dim).2 c)) :=
  Rep.get_ofFn _ c

/-- A value of `cumminIn` is the input at its index. -/
theorem cumminIn_fst_apply (order : ReduceOrder) (x : Tensor α s) (dim : Fin s.rank)
    (c : s.Coord) :
    (cumminIn order x dim).1 c = x (Shape.Coord.update c dim ((cumminIn order x dim).2 c)) :=
  Rep.get_ofFn _ c

end Extrema

section Pool

variable [Context α] {dims : List (Fin s.rank)} {kernel stride padLow padHigh dilation : List Nat}
  {ceilMode : Bool}

/-- `poolSum` in the order `order`. -/
def poolSumIn (order : ReduceOrder) (x : Tensor α s)
    (o : (s.pool dims kernel stride padLow padHigh dilation ceilMode).Coord) : α :=
  order.fold _ (· + ·) 0 fun i => poolInput x o (Shape.Coord.unlinearize i)

/-- `pool` in the order `order`: a sum or average pool adds its window with `poolSumIn`; a
maximum or minimum pool selects an entry and is `pool`. -/
def poolIn (order : ReduceOrder) (op : PoolOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool := false) :
    Tensor α (s.pool dims kernel stride padLow padHigh dilation ceilMode) :=
  match op with
  | .sum => Rep.ofFn fun o => poolSumIn order x o
  | .avg countIncludePad => Rep.ofFn fun o =>
      Context.divNat (poolSumIn order x o) (poolCount countIncludePad o)
  | op => pool op x dims kernel stride padLow padHigh dilation ceilMode

theorem poolIn_leftFold (op : PoolOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) :
    poolIn .leftFold op x dims kernel stride padLow padHigh dilation ceilMode =
      pool op x dims kernel stride padLow padHigh dilation ceilMode := by
  cases op <;> simp only [poolIn, pool, poolSumIn, poolSum, ReduceOrder.fold_leftFold]

/-- Under associative and commutative addition, `poolIn` in an order that covers the window is
`pool`. -/
theorem poolIn_eq_pool {order : ReduceOrder}
    (hassoc : ∀ x y z : α, x + y + z = x + (y + z)) (hcomm : ∀ x y : α, x + y = y + x)
    (op : PoolOp) (x : Tensor α s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (hcov : order.Covers (Shape.size (s.poolWindow dims kernel))) :
    poolIn order op x dims kernel stride padLow padHigh dilation ceilMode =
      pool op x dims kernel stride padLow padHigh dilation ceilMode := by
  rw [← poolIn_leftFold]
  cases op <;> simp only [poolIn, poolSumIn, ReduceOrder.fold_eq_leftFold hcov hassoc hcomm]

end Pool

end

/-! ## Over the reals -/

section Real

variable {s t : Shape} {order : ReduceOrder}

/-- Over `ℝ`, a sum in an order that covers the fiber is `reduce`. -/
theorem reduceIn_sum_real (x : Tensor ℝ s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (hcov : order.Covers (Shape.size (s.selectAxes dims))) :
    reduceIn order .sum x dims keepdim = reduce .sum x dims keepdim :=
  reduceIn_eq_reduce (α := ℝ) .sum (fun a b c => add_assoc a b c) (fun a b => add_comm a b)
    x dims keepdim _ hcov

/-- Over `ℝ`, a product in an order that covers the fiber is `reduce`. -/
theorem reduceIn_prod_real (x : Tensor ℝ s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (hcov : order.Covers (Shape.size (s.selectAxes dims))) :
    reduceIn order .prod x dims keepdim = reduce .prod x dims keepdim :=
  reduceIn_eq_reduce (α := ℝ) .prod (fun a b c => mul_assoc a b c) (fun a b => mul_comm a b)
    x dims keepdim _ hcov

/-- Over `ℝ`, a maximum in an order that covers the fiber is `reduce`. -/
theorem reduceIn_max_real (x : Tensor ℝ s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : ∀ d ∈ dims, 0 < s.extent d) (hcov : order.Covers (Shape.size (s.selectAxes dims))) :
    reduceIn order .max x dims keepdim (fun _ => h) = reduce .max x dims keepdim (fun _ => h) :=
  reduceIn_eq_reduce (α := ℝ) .max
    (fun a b c => by
      change Context.maximum (Context.maximum a b) c = Context.maximum a (Context.maximum b c)
      simp only [Context.maximum_real, max_assoc])
    (fun a b => by
      change Context.maximum a b = Context.maximum b a
      simp only [Context.maximum_real, max_comm])
    x dims keepdim _ hcov

/-- Over `ℝ`, a minimum in an order that covers the fiber is `reduce`. -/
theorem reduceIn_min_real (x : Tensor ℝ s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : ∀ d ∈ dims, 0 < s.extent d) (hcov : order.Covers (Shape.size (s.selectAxes dims))) :
    reduceIn order .min x dims keepdim (fun _ => h) = reduce .min x dims keepdim (fun _ => h) :=
  reduceIn_eq_reduce (α := ℝ) .min
    (fun a b c => by
      change Context.minimum (Context.minimum a b) c = Context.minimum a (Context.minimum b c)
      simp only [Context.minimum_real, min_assoc])
    (fun a b => by
      change Context.minimum a b = Context.minimum b a
      simp only [Context.minimum_real, min_comm])
    x dims keepdim _ hcov

/-- Over `ℝ`, a mean in an order that covers the fiber is `mean`. -/
theorem meanIn_real (x : Tensor ℝ s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (hcov : order.Covers (Shape.size (s.selectAxes dims))) :
    meanIn order x dims keepdim = mean x dims keepdim :=
  meanIn_eq_mean (fun a b c => add_assoc a b c) (fun a b => add_comm a b) x dims keepdim hcov

/-- Over `ℝ`, a contraction in an order that covers the contracted coordinates is `contract`. -/
theorem contractIn_real (x : Tensor ℝ s) (y : Tensor ℝ t)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (out : Shape)
    (h : Shape.ContractValid s t dims batchDims) (hout : out = s.contract t dims batchDims)
    (hcov : order.Covers (Shape.size ((dims.map Prod.fst).map s.extent))) :
    contractIn order x y dims batchDims out h hout = contract x y dims batchDims out h hout :=
  contractIn_eq_contract x y dims batchDims out h hout hcov

/-- Over `ℝ`, a pool in an order that covers the window is `pool`. -/
theorem poolIn_real (op : PoolOp) (x : Tensor ℝ s) (dims : List (Fin s.rank))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool)
    (hcov : order.Covers (Shape.size (s.poolWindow dims kernel))) :
    poolIn order op x dims kernel stride padLow padHigh dilation ceilMode =
      pool op x dims kernel stride padLow padHigh dilation ceilMode :=
  poolIn_eq_pool (fun a b c => add_assoc a b c) (fun a b => add_comm a b) op x dims kernel
    stride padLow padHigh dilation ceilMode hcov

/-- Over `ℝ`, a cumulative sum in an order that covers the length of the axis is `cumsum`. -/
theorem scanIn_add_real (x : Tensor ℝ s) (dim : Fin s.rank) (hcov : order.Covers (s.extent dim)) :
    scanIn order (· + ·) 0 x dim = cumsum x dim := by
  rw [scanIn_eq_leftFold add_assoc add_comm _ x dim hcov, scanIn_add_leftFold]

/-- Over `ℝ`, a cumulative product in an order that covers the length of the axis is
`cumprod`. -/
theorem scanIn_mul_real (x : Tensor ℝ s) (dim : Fin s.rank) (hcov : order.Covers (s.extent dim)) :
    scanIn order (· * ·) 1 x dim = cumprod x dim := by
  rw [scanIn_eq_leftFold mul_assoc mul_comm _ x dim hcov, scanIn_mul_leftFold]

/-- Over `ℝ`, running maxima in an order that covers the length of the axis are `cummax`, indices
included. -/
theorem cummaxIn_real (x : Tensor ℝ s) (dim : Fin s.rank) (hcov : order.Covers (s.extent dim)) :
    cummaxIn order x dim = cummax x dim :=
  (cummaxIn_eq_leftFold id (fun _ _ => Bool.eq_iff_iff.2 (by simp [not_lt])) x dim hcov).trans
    (cummaxIn_leftFold x dim)

/-- Over `ℝ`, running minima in an order that covers the length of the axis are `cummin`, indices
included. -/
theorem cumminIn_real (x : Tensor ℝ s) (dim : Fin s.rank) (hcov : order.Covers (s.extent dim)) :
    cumminIn order x dim = cummin x dim :=
  (cumminIn_eq_leftFold OrderDual.toDual (fun _ _ => Bool.eq_iff_iff.2 (by simp [not_lt])) x dim
    hcov).trans (cumminIn_leftFold x dim)

end Real

end TorchLean.Tensor
