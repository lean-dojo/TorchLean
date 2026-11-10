/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.LibTorch.Contract
public import TorchLean.Verify.Floats.IEEE754.Reductions

/-!
# The deterministic CUDA sum

The summation schedule of the native flat sum when deterministic algorithms are enabled, as a
FloatLib `ReductionTree` over the values of any format of FloatLib's configured binary family. The
LibTorch CUDA backend is an FFI boundary: this module states the schedule and proves what agreement
with it gives, and it does not prove that a native sum agrees.

- `treeSum`: the schedule over sum trees. `eval_treeSum` proves that it adds every input exactly
  once.
- `reduceSumTree`: the executable evaluation of the schedule.
- `abs_reduceSumTree_sub_le`: in an IEEE format a finite result is within the unit-roundoff
  enclosure of the exact sum, with one rounding per addition of the schedule, the additions of
  padding zeros included.
- `abs_toReal_sub_sum_le`: the same bound for a finite native sum that agrees with
  `reduceSumTree` up to NaN encodings.

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
open IEEE754 (evalSum exactSum sumAbs evalSum_enclosure finiteEvalSumTree_of_isFinite)
open Contract (EqUpToNaN)

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)
local notation "Tree" => ReductionTree (ExecFloat (Configured.Family format code plan))

/-! ## The deterministic CUDA sum

`treeSum` is the summation schedule of the native flat sum when deterministic algorithms are
enabled (`deterministic_sum` in `csrc/libtorch/torchlean.cpp`). Otherwise the native sum is
`at::sum`, whose order is implementation-defined and which nothing here describes. One pass over
`m` values uses `min (⌈m / 256⌉) 65535` blocks of 256 lanes and the stride `256 · blocks`. Every
lane starts at `+0` and adds the inputs at its index plus each multiple of the stride, in
increasing order, and a lane past the end adds nothing. Each block then halves its lanes eight
times, lane `i` becoming lane `i` plus lane `i + width`, and the block results are the next pass's
values. Passes repeat while more than one value remains, a single value also takes one pass, and no
values sum to `+0`. The zero additions are part of the schedule: they decide signed zeros (a lone
`-0` sums to `+0`). The schedule differs from the left fold: on binary32 `[2^26, 1, -2^26]` the
fold gives `0` and the tree `1`.
-/

/-- One pass of the tree sum: the schedules of the `min (⌈m / 256⌉) 65535` block results. -/
def treePass (ts : List Tree) : List Tree :=
  let blocks := min ((ts.length + 255) / 256) 65535
  let stride := 256 * blocks
  let lane (l : Nat) : Tree :=
    (List.range ((ts.length + stride - 1) / stride)).foldl
      (fun acc k => (ts[k * stride + l]?).elim acc acc.node) (.leaf ExecFloat.Binary.zero)
  let halve (v : List Tree) : List Tree :=
    List.zipWith .node (v.take (v.length / 2)) (v.drop (v.length / 2))
  (List.range blocks).map fun b =>
    ((halve^[8]) ((List.range 256).map fun j => lane (b * 256 + j))).headD
      (.leaf ExecFloat.Binary.zero)

/-- One `treePass` leaves `min ((ts.length + 255) / 256) 65535` values. -/
theorem treePass_length (ts : List Tree) :
    (treePass ts).length = min ((ts.length + 255) / 256) 65535 := by
  simp [treePass]

/-- The tree sum schedule: passes of `treePass` while more than one value remains. -/
def treeSum (ts : List Tree) : Tree :=
  let us := treePass ts
  if 1 < us.length then treeSum us else us.headD (.leaf ExecFloat.Binary.zero)
termination_by ts.length
decreasing_by
  simp +zetaDelta only [treePass_length] at *
  omega

/-- The deterministic CUDA sum of `xs`: the executable evaluation of its `treeSum` schedule. -/
def reduceSumTree [ExecFloat.Add (Configured.Family format code plan)] (xs : List Value) :
    Value :=
  evalSum (treeSum (xs.map .leaf))

section Sum

variable {M : Type*} [AddCommMonoid M]

private theorem sum_map_range (g : Nat → M) (n : Nat) :
    ((List.range n).map g).sum = ∑ i ∈ Finset.range n, g i := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.map_append, List.sum_append, ih, Finset.sum_range_succ]
    simp

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
    (((fun v : List (ReductionTree α) =>
        List.zipWith ReductionTree.node (v.take (v.length / 2)) (v.drop (v.length / 2)))^[m]
          v).headD d).eval (· + ·) f = (v.map (ReductionTree.eval (· + ·) f)).sum := by
  induction m with
  | zero =>
    intro v hv d
    match v, hv with
    | [t], _ => simp
  | succ m ih =>
    intro v hv d
    rw [Function.iterate_succ_apply, ih _ (by simp; rw [hv, pow_succ]; omega) d,
      sum_zipWith_node f _ _ (by simp; omega), ← List.sum_append, ← List.map_append,
      List.take_append_drop]

private theorem sum_treePass (f : Value → M) (hf : f ExecFloat.Binary.zero = 0) (ts : List Tree) :
    ((treePass ts).map (ReductionTree.eval (· + ·) f)).sum =
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
      (ReductionTree.leaf ExecFloat.Binary.zero)).eval (· + ·) f =
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
theorem eval_treeSum (f : Value → M) (hf : f ExecFloat.Binary.zero = 0) (ts : List Tree) :
    (treeSum ts).eval (· + ·) f = (ts.map (ReductionTree.eval (· + ·) f)).sum := by
  rw [treeSum]
  by_cases h : 1 < (treePass ts).length
  · rw [ite_eq_left h, eval_treeSum f hf (treePass ts), sum_treePass f hf]
  · rw [ite_eq_right h, ← sum_treePass f hf ts]
    generalize treePass ts = us at h ⊢
    match us, h with
    | [], _ => simp [ReductionTree.eval, hf]
    | [u], _ => simp
    | _ :: _ :: _, h => simp at h
termination_by ts.length
decreasing_by
  rw [treePass_length] at h ⊢
  omega

end Sum

section Enclosure

variable [cap : ExecFloat.Add (Configured.Family format code plan)]
  [ExecFloat.Capability.HasSpec cap Configured.Spec.add]

/-- In an IEEE format a finite deterministic CUDA sum is within the unit-roundoff enclosure of the
exact sum of its inputs, with one rounding per addition of its schedule. -/
theorem abs_reduceSumTree_sub_le (xs : List Value)
    (hresult : isFinite (reduceSumTree xs) = true)
    (hformat : format.isIEEE = true := by ieee_format) :
    |toReal (reduceSumTree xs) - (xs.map toReal).sum| ≤
      ((1 + Model.unitRoundoffAt format) ^ (treeSum (xs.map .leaf)).nodeCount - 1) *
        (xs.map fun x => |toReal x|).sum := by
  have bound := evalSum_enclosure (treeSum (xs.map .leaf))
    (finiteEvalSumTree_of_isFinite _ hresult hformat) hformat
  have hsum : exactSum (treeSum (xs.map .leaf)) = (xs.map toReal).sum := by
    rw [exactSum, eval_treeSum _ (by simp [ExecFloat.Binary.zero]), List.map_map]
    rfl
  have habs : sumAbs (treeSum (xs.map .leaf)) = (xs.map fun x => |toReal x|).sum := by
    rw [sumAbs, ReductionTree.sumAbs, eval_treeSum _ (by simp [ExecFloat.Binary.zero]),
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
      ((1 + Model.unitRoundoffAt format) ^ (treeSum (xs.map .leaf)).nodeCount - 1) *
        (xs.map fun x => |toReal x|).sum := by
  have heq := h.eq_of_isFinite hresult
  rw [heq] at hresult ⊢
  exact abs_reduceSumTree_sub_le xs hresult hformat

end Enclosure

end TorchLean.LibTorch
