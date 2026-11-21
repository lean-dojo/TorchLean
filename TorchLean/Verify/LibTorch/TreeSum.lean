/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.LibTorch.Contract
public import TorchLean.Verify.Floats.IEEE754.Order

/-!
# The deterministic CUDA sum

The summation schedule of the native flat sum when deterministic algorithms are enabled, as a
FloatLib `ReductionTree`, and as a `Tensor.ReduceOrder`. The LibTorch CUDA backend is an FFI
boundary: this module states the schedule and proves what agreement with it gives, and it does not
prove that a native sum agrees.

- `treeSum`: the schedule over sum trees with leaves of any type and a padding leaf.
  `eval_treeSum` proves that it adds every input exactly once, and `map_treeSum` that relabelling
  the leaves commutes with it.
- `sumOrder n`: the schedule over `n` entries as the reduction order `ReduceOrder.ofTree`, with
  the start value at the padding leaves. `treeSum_eq_tree` proves that the schedule over any `n`
  leaves is the tree of `sumOrder n` read at them.
- `reduceSumTree`: the executable evaluation of the schedule over the values of any format of
  FloatLib's configured binary family. `reduceSumTree_eq_fold` proves that it is
  `ReduceOrder.fold` in `sumOrder` from `+0`, for every input list.
- `abs_reduceSumTree_sub_le`: in an IEEE format a finite result is within the unit-roundoff
  enclosure of the exact sum in the depth of the schedule.
- `abs_toReal_sub_sum_le`: the same bound for a finite native sum that agrees with
  `reduceSumTree` up to NaN encodings.

For `n ≥ 1`, `sumOrder n` does not satisfy `ReduceOrder.Covers`: each of its at least 256 lanes
starts from a padding leaf. So `IEEE754.abs_fold_add_sub_le` does not apply to it, and
`abs_reduceSumTree_sub_le` gives the same depth bound for it. It is not `ReduceOrder.blocked`, whose
blocks are runs of consecutive entries: a lane adds the entries one stride apart.

`TorchLeanTest/CUDA/ReduceTree.lean` compares the native binary32 sum with
`reduceSumTree` on fixtures, counting any two NaN results as equal. LibTorch, its CUDA
dependencies, compiler flags, GPU hardware and driver behavior are assumptions, not proved by Lean.

External references for the assumptions named here:

- IEEE 754-2019 defines binary32 and binary64 arithmetic and special values:
  https://standards.ieee.org/ieee/754/6210/
- NVIDIA CUDA C Programming Guide documents the CUDA execution and memory model:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite toReal)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Numerics (ReductionTree)

namespace TorchLean.LibTorch

open FloatLib.Floats.Formats.BinaryInterchange
open IEEE754 (evalSum exactSum sumAbs)
open Contract (EqUpToNaN)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-! ## The deterministic CUDA sum

`treeSum` is the summation schedule of the native flat sum when deterministic algorithms are
enabled (`deterministic_sum` in `csrc/libtorch/torchlean.cpp`). Otherwise the native sum is
`aten::sum`, whose order is implementation-defined and which nothing here describes. One pass over
`m` values uses `min (⌈m / 256⌉) 65535` blocks of 256 lanes and the stride `256 · blocks`. Every
lane starts at `+0` and adds the inputs at its index plus each multiple of the stride, in
increasing order, and a lane past the end adds nothing. Each block then halves its lanes eight
times, lane `i` becoming lane `i` plus lane `i + width`, and the block results are the next pass's
values. Passes repeat while more than one value remains, a single value also takes one pass, and no
values sum to `+0`. The zero additions are part of the schedule: they decide signed zeros (a lone
`-0` sums to `+0`). The schedule differs from the left fold: on binary32 `[2^26, 1, -2^26]` the
fold gives `0` and the tree `1`.
-/

section Schedule

variable {γ : Type*}

/-- Halving: each tree of the first half of the list joined with the tree as far into the second
half. -/
def halve (v : List (ReductionTree γ)) : List (ReductionTree γ) :=
  List.zipWith .node (v.take (v.length / 2)) (v.drop (v.length / 2))

/-- One pass of the tree sum, with `pad` at the leaves the lanes start from: the schedules of the
`min (⌈m / 256⌉) 65535` block results. -/
def treePass (pad : γ) (ts : List (ReductionTree γ)) : List (ReductionTree γ) :=
  let blocks := min ((ts.length + 255) / 256) 65535
  let stride := 256 * blocks
  let lane (l : Nat) : ReductionTree γ :=
    (List.range ((ts.length + stride - 1) / stride)).foldl
      (fun acc k => (ts[k * stride + l]?).elim acc acc.node) (.leaf pad)
  (List.range blocks).map fun b =>
    ((halve^[8]) ((List.range 256).map fun j => lane (b * 256 + j))).headD (.leaf pad)

/-- One `treePass` leaves `min ((ts.length + 255) / 256) 65535` values. -/
theorem treePass_length (pad : γ) (ts : List (ReductionTree γ)) :
    (treePass pad ts).length = min ((ts.length + 255) / 256) 65535 := by
  simp [treePass]

/-- The tree sum schedule: passes of `treePass` while more than one value remains. -/
def treeSum (pad : γ) (ts : List (ReductionTree γ)) : ReductionTree γ :=
  let us := treePass pad ts
  if 1 < us.length then treeSum pad us else us.headD (.leaf pad)
termination_by ts.length
decreasing_by
  simp +zetaDelta only [treePass_length] at *
  omega

private theorem map_map {δ ε : Type*} (f : γ → δ) (g : δ → ε) (t : ReductionTree γ) :
    (t.map f).map g = t.map (g ∘ f) := by
  induction t with
  | leaf x => rfl
  | node a b ha hb => simp only [ReductionTree.map, ha, hb]

private theorem map_iterate_halve {δ : Type*} (g : γ → δ) :
    ∀ (m : Nat) (v : List (ReductionTree γ)),
      ((halve^[m]) v).map (ReductionTree.map g) = (halve^[m]) (v.map (ReductionTree.map g))
  | 0, _ => rfl
  | m + 1, v => by
    rw [Function.iterate_succ_apply, Function.iterate_succ_apply, map_iterate_halve g m]
    congr 1
    simp only [halve, List.map_zipWith, List.length_map, ← List.map_take, ← List.map_drop,
      List.zipWith_map]
    rfl

private theorem map_headD {δ : Type*} (g : γ → δ) (d : ReductionTree γ) :
    ∀ v : List (ReductionTree γ), (v.headD d).map g = (v.map (ReductionTree.map g)).headD (d.map g)
  | [] => rfl
  | _ :: _ => rfl

/-- Relabelling the leaves commutes with a pass. -/
theorem map_treePass {δ : Type*} (g : γ → δ) (pad : γ) (ts : List (ReductionTree γ)) :
    (treePass pad ts).map (ReductionTree.map g) =
      treePass (g pad) (ts.map (ReductionTree.map g)) := by
  have hlane : ∀ (L : List Nat) (S l : Nat) (acc : ReductionTree γ),
      (L.foldl (fun acc k => (ts[k * S + l]?).elim acc acc.node) acc).map g =
        L.foldl (fun acc k => ((ts.map (ReductionTree.map g))[k * S + l]?).elim acc acc.node)
          (acc.map g) := by
    intro L S l
    induction L with
    | nil => intro acc; rfl
    | cons k L ih =>
      intro acc
      rw [List.foldl_cons, List.foldl_cons, ih, List.getElem?_map]
      cases ts[k * S + l]? <;> rfl
  simp only [treePass, List.length_map, List.map_map]
  refine List.map_congr_left fun b _ => ?_
  rw [Function.comp_apply, map_headD, map_iterate_halve, List.map_map]
  congr 2
  exact List.map_congr_left fun j _ => hlane _ _ _ _

/-- Relabelling the leaves commutes with the schedule. -/
theorem map_treeSum {δ : Type*} (g : γ → δ) (pad : γ) (ts : List (ReductionTree γ)) :
    (treeSum pad ts).map g = treeSum (g pad) (ts.map (ReductionTree.map g)) := by
  rw [treeSum.eq_def pad ts, treeSum.eq_def (g pad)]
  simp only [← map_treePass g pad ts, List.length_map]
  by_cases h : 1 < (treePass pad ts).length
  · rw [ite_eq_left h, ite_eq_left h, map_treeSum g pad (treePass pad ts)]
  · rw [ite_eq_right h, ite_eq_right h, map_headD]
    rfl
termination_by ts.length
decreasing_by
  rw [treePass_length] at h ⊢
  omega

end Schedule

/-- The deterministic CUDA sum of `xs`: the executable evaluation of its `treeSum` schedule. -/
def reduceSumTree [ExecFloat.Add (Configured.Family format code plan)] (xs : List Value) :
    Value :=
  evalSum (treeSum ExecFloat.Binary.zero (xs.map .leaf))

section Sum

variable {γ M : Type*} [AddCommMonoid M]

private theorem sum_map_range (g : Nat → M) (n : Nat) :
    ((List.range n).map g).sum = ∑ i ∈ Finset.range n, g i := rfl

private theorem sum_range_mul (g : Nat → M) (B W : Nat) :
    ∑ i ∈ Finset.range (B * W), g i =
      ∑ b ∈ Finset.range B, ∑ j ∈ Finset.range W, g (b * W + j) := by
  induction B with
  | zero => simp
  | succ B ih => rw [Nat.succ_mul, Finset.sum_range_add, ih, Finset.sum_range_succ]

private theorem sum_map_eval_eq {α : Type*} (f : α → M) (ts : List (ReductionTree α)) :
    (ts.map (ReductionTree.eval (· + ·) f)).sum =
      ∑ i ∈ Finset.range ts.length, ((ts[i]?).map (ReductionTree.eval (· + ·) f)).getD 0 := by
  induction ts with
  | nil => simp
  | cons t ts ih =>
    rw [List.map_cons, List.sum_cons, ih, List.length_cons, Finset.sum_range_succ']
    simp [add_comm]

private theorem eval_foldl {α β : Type*} (f : α → M)
    (F : ReductionTree α → β → ReductionTree α) (c : β → M)
    (hF : ∀ acc k, (F acc k).eval (· + ·) f = acc.eval (· + ·) f + c k) :
    ∀ (L : List β) (acc : ReductionTree α),
      (L.foldl F acc).eval (· + ·) f = acc.eval (· + ·) f + (L.map c).sum
  | [], acc => by simp
  | k :: L, acc => by
    rw [List.foldl_cons, eval_foldl f F c hF L, hF, List.map_cons, List.sum_cons, add_assoc]

private theorem sum_zipWith_node {α : Type*} (f : α → M) :
    ∀ (a b : List (ReductionTree α)), a.length = b.length →
    ((List.zipWith ReductionTree.node a b).map (ReductionTree.eval (· + ·) f)).sum =
      (a.map (ReductionTree.eval (· + ·) f)).sum + (b.map (ReductionTree.eval (· + ·) f)).sum
  | [], [], _ => by simp
  | x :: a, y :: b, h => by
    simp only [List.zipWith_cons_cons, List.map_cons, List.sum_cons, ReductionTree.eval]
    rw [sum_zipWith_node f a b (by simpa using h)]
    abel
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

private theorem eval_iterate_halve {α : Type*} (f : α → M) (m : Nat) :
    ∀ (v : List (ReductionTree α)), v.length = 2 ^ m → ∀ d : ReductionTree α,
    (((halve^[m]) v).headD d).eval (· + ·) f = (v.map (ReductionTree.eval (· + ·) f)).sum := by
  induction m with
  | zero =>
    intro v hv d
    match v, hv with
    | [t], _ => simp
  | succ m ih =>
    intro v hv d
    rw [Function.iterate_succ_apply, ih _ (by simp [halve]; rw [hv, pow_succ]; omega) d, halve,
      sum_zipWith_node f _ _ (by simp; omega), ← List.sum_append, ← List.map_append,
      List.take_append_drop]

private theorem sum_treePass (pad : γ) (f : γ → M) (hf : f pad = 0)
    (ts : List (ReductionTree γ)) :
    ((treePass pad ts).map (ReductionTree.eval (· + ·) f)).sum =
      (ts.map (ReductionTree.eval (· + ·) f)).sum := by
  unfold treePass
  simp only []
  generalize hB : min ((ts.length + 255) / 256) 65535 = B
  have hn : ts.length ≤ (ts.length + 256 * B - 1) / (256 * B) * (256 * B) := by
    rcases Nat.eq_zero_or_pos B with h0 | hpos
    · subst h0; omega
    · have := Nat.lt_div_mul_add (a := ts.length + 256 * B - 1) (b := 256 * B) (by omega)
      omega
  generalize (ts.length + 256 * B - 1) / (256 * B) = K at hn
  set c : Nat → M := fun i => ((ts[i]?).map (ReductionTree.eval (· + ·) f)).getD 0 with hc
  have hlane : ∀ l, ((List.range K).foldl
      (fun acc k => (ts[k * (256 * B) + l]?).elim acc acc.node)
      (ReductionTree.leaf pad)).eval (· + ·) f =
        ∑ k ∈ Finset.range K, c (k * (256 * B) + l) := by
    intro l
    rw [eval_foldl f _ (fun k => c (k * (256 * B) + l)), sum_map_range]
    · simp [ReductionTree.eval, hf]
    · intro acc k
      simp only [hc]
      cases ts[k * (256 * B) + l]? <;> simp [ReductionTree.eval]
  rw [List.map_map, sum_map_range]
  calc _ = ∑ b ∈ Finset.range B, ∑ j ∈ Finset.range 256,
        ∑ k ∈ Finset.range K, c (k * (256 * B) + (b * 256 + j)) := by
        refine Finset.sum_congr rfl fun b _ => ?_
        rw [Function.comp_apply, eval_iterate_halve f 8, List.map_map, sum_map_range]
        · exact Finset.sum_congr rfl fun j _ => by rw [Function.comp_apply, hlane]
        · simp
    _ = ∑ i ∈ Finset.range (B * 256), ∑ k ∈ Finset.range K, c (k * (256 * B) + i) :=
        (sum_range_mul (fun i => ∑ k ∈ Finset.range K, c (k * (256 * B) + i)) B 256).symm
    _ = ∑ k ∈ Finset.range K, ∑ i ∈ Finset.range (256 * B), c (k * (256 * B) + i) := by
        rw [Finset.sum_comm, Nat.mul_comm B 256]
    _ = ∑ i ∈ Finset.range (K * (256 * B)), c i := (sum_range_mul c K (256 * B)).symm
    _ = ∑ i ∈ Finset.range ts.length, c i := by
        obtain ⟨r, hr⟩ := Nat.exists_eq_add_of_le hn
        have hzero : ∑ i ∈ Finset.range r, c (ts.length + i) = 0 :=
          Finset.sum_eq_zero fun i _ => by simp [hc]
        rw [hr, Finset.sum_range_add, hzero, add_zero]
    _ = _ := (sum_map_eval_eq f ts).symm

/-- The tree sum schedule adds every input exactly once: evaluating it with any addition of a
commutative monoid sums the inputs' evaluations, given that the padding leaf evaluates to zero. -/
theorem eval_treeSum (pad : γ) (f : γ → M) (hf : f pad = 0) (ts : List (ReductionTree γ)) :
    (treeSum pad ts).eval (· + ·) f = (ts.map (ReductionTree.eval (· + ·) f)).sum := by
  rw [treeSum]
  by_cases h : 1 < (treePass pad ts).length
  · rw [ite_eq_left h, eval_treeSum pad f hf (treePass pad ts), sum_treePass pad f hf]
  · rw [ite_eq_right h, ← sum_treePass pad f hf ts]
    generalize treePass pad ts = us at h ⊢
    match us, h with
    | [], _ => simp [ReductionTree.eval, hf]
    | [u], _ => simp
    | _ :: _ :: _, h => simp at h
termination_by ts.length
decreasing_by
  rw [treePass_length] at h ⊢
  omega

end Sum

/-! ## The order -/

open Tensor (ReduceOrder)

/-- The tree sum schedule over `n` entries as a reduction order: the schedule over the leaves
`some 0, …, some (n - 1)`, with the start value `none` at the padding leaves. -/
def sumOrder (n : Nat) : ReduceOrder :=
  .ofTree (treeSum none ((List.range n).map (.leaf ∘ some)))

/-- The schedule over any leaves is the tree of `sumOrder` over their count, read at them and at
the padding. -/
theorem treeSum_eq_tree {γ : Type*} (pad : γ) (xs : List γ) :
    treeSum pad (xs.map .leaf) =
      ((sumOrder xs.length).tree xs.length).map (·.elim pad (xs[·])) := by
  -- Every leaf of the schedule over the indices is the padding or an index below the count.
  have hkept : ∀ x ∈ (treeSum none ((List.range xs.length).map (.leaf ∘ some))).leaves,
      (ReduceOrder.entry xs.length x).isSome := by
    let c : Option Nat → Nat := fun x => if (ReduceOrder.entry xs.length x).isSome then 0 else 1
    have hR : (((List.range xs.length).map (.leaf ∘ some)).map
        (ReductionTree.eval (· + ·) c)).sum = 0 := by
      refine List.sum_eq_zero fun a ha => ?_
      simp only [List.map_map, List.mem_map, List.mem_range, Function.comp_apply] at ha
      obtain ⟨i, hi, rfl⟩ := ha
      simp [c, ReductionTree.eval, ReduceOrder.entry, hi]
    have h := eval_treeSum none c (by simp [c, ReduceOrder.entry])
      ((List.range xs.length).map (.leaf ∘ some))
    rw [hR, ReductionTree.eval_add_eq_sum, List.sum_eq_zero_iff] at h
    intro x hx
    exact Option.isSome_iff_ne_none.2 (by simpa [c] using h _ (List.mem_map_of_mem hx))
  have hxs : xs.map .leaf = ((List.range xs.length).map (.leaf ∘ some)).map
      (ReductionTree.map fun x : Option Nat => x.elim pad fun i => xs[i]?.getD pad) := by
    refine List.ext_getElem (by simp) fun i h₁ _ => ?_
    have hi : i < xs.length := by simpa using h₁
    simp [ReductionTree.map, hi]
  simp only [sumOrder, ReduceOrder.tree]
  rw [ReduceOrder.prune_of_isSome none hkept, Option.getD_some, map_map, hxs]
  refine (map_treeSum (fun x : Option Nat => x.elim pad fun i => xs[i]?.getD pad) none
    ((List.range xs.length).map (.leaf ∘ some))).symm.trans ?_
  congr 1
  funext x
  rcases x with _ | i
  · rfl
  · by_cases hi : i < xs.length <;> simp [ReduceOrder.entry, hi]

/-- The deterministic CUDA sum is the sum in the order `sumOrder` over the count of its inputs,
from `+0`. -/
theorem reduceSumTree_eq_fold [ExecFloat.Add (Configured.Family format code plan)]
    (xs : List Value) :
    reduceSumTree xs =
      (sumOrder xs.length).fold xs.length (· + ·) ExecFloat.Binary.zero (xs[·]) := by
  rw [IEEE754.fold_add_eq_evalSum, reduceSumTree, treeSum_eq_tree]

section Enclosure

variable [cap : ExecFloat.Add (Configured.Family format code plan)]
  [ExecFloat.Capability.HasSpec cap Configured.Spec.add]

/-- In an IEEE format a finite deterministic CUDA sum is within the unit-roundoff enclosure of the
exact sum of its inputs in the depth of its schedule. -/
theorem abs_reduceSumTree_sub_le (xs : List Value)
    (hresult : isFinite (reduceSumTree xs) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    |toReal (reduceSumTree xs) - (xs.map toReal).sum| ≤
      ((1 + Model.unitRoundoffAt format) ^ (treeSum ExecFloat.Binary.zero (xs.map .leaf)).depth -
        1) * (xs.map fun x => |toReal x|).sum := by
  have bound := IEEE754.evalSum_enclosure_depth _
    (IEEE754.finiteEvalSumTree_of_isFinite _ hresult hformat) hformat
  have hsum : exactSum (treeSum ExecFloat.Binary.zero (xs.map .leaf)) = (xs.map toReal).sum := by
    rw [exactSum, eval_treeSum _ _ (by simp [ExecFloat.Binary.zero]), List.map_map]
    rfl
  have habs : sumAbs (treeSum ExecFloat.Binary.zero (xs.map .leaf)) =
      (xs.map fun x => |toReal x|).sum := by
    rw [sumAbs, ReductionTree.sumAbs, eval_treeSum _ _ (by simp [ExecFloat.Binary.zero]),
      List.map_map]
    rfl
  rw [hsum, habs] at bound
  exact bound

/-- A finite native sum that agrees with the deterministic CUDA sum up to NaN encodings has the
same enclosure. -/
theorem abs_toReal_sub_sum_le {native : Value} {xs : List Value}
    (h : EqUpToNaN native (reduceSumTree xs)) (hresult : isFinite native = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    |toReal native - (xs.map toReal).sum| ≤
      ((1 + Model.unitRoundoffAt format) ^ (treeSum ExecFloat.Binary.zero (xs.map .leaf)).depth -
        1) * (xs.map fun x => |toReal x|).sum := by
  have heq := h.eq_of_isFinite hresult
  rw [heq] at hresult ⊢
  exact abs_reduceSumTree_sub_le xs hresult hformat

end Enclosure

end TorchLean.LibTorch
