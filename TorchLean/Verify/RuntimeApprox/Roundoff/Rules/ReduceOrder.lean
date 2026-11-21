/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.List
public import TorchLean.Verify.Floats.IEEE754.Order
public import TorchLean.Verify.RuntimeApprox.Roundoff.Rules.Elementwise
public import FloatLib.Floats.Formats.IEEE754.Native.Roundoff
public import FloatLib.Numerics.MixedError.Basic
public meta import TorchLean.Verify.RuntimeApprox.Roundoff.Rules
public import FloatLib.Tactic.Roundoff.Extension

/-!
# Reductions in an order for `roundoff`

An operator in an order other than the left fold (`Tensor.ReduceOrder`) computes each output entry
as `order.fold n step start value`, the order's tree over the entries. `reduceIn_sum_apply`,
`reduceIn_prod_apply`, `reduceIn_max_apply` and `reduceIn_min_apply` read an entry of
`Tensor.reduceIn order` as that fold, and `matmulCommonBatchIn_apply` an entry of
`Tensor.matmulCommonBatchIn order` at every batch, through `matmulIn_apply` and the fold lemmas
`ReduceOrder.fold_cast`, `ReduceOrder.fold_size_singleton` and `ReduceOrder.fold_unlinearize_congr`.
The contractions, scans and windows already read as folds through their definitions. A value of
a running maximum or minimum in an order is the input at the index the order selects
(`Tensor.cummaxIn_fst_apply`, `Tensor.cumminIn_fst_apply`), so it carries the input's error there
and adds none.

A map that commutes with the step on the values a fold combines commutes with the fold:
`ReduceOrder.map_fold_of_forall` when a property of the entries that the step keeps gives the
commutation, `ReduceOrder.map_fold_of_operands` when a property of the result of the mapped step
passes to its operands. With an associative and commutative mapped step, the image of a fold in an
order that covers its entries is the image of the left fold (`ReduceOrder.map_fold_eq_leftFold`).

The rules bound a sum in every format of the configured binary family on an IEEE descriptor, with
the depth `d` of the order's tree at `n` entries in place of the `n` steps of the left fold: a
finite fold is within `((1 + u) ^ d - 1) · (|start| + ∑ |value i|)` of the exact left fold of the
entries' values from the start's value, stated absolutely and as a mixed error.

- `abs_pairwise_sum_sub_le`, `mixedError_pairwise_sum`: `pairwise`, at depth `⌈log₂ (n + 1)⌉`
  (`TorchLean.IEEE754.abs_fold_pairwise_sub_le`).
- `abs_blocked_sum_sub_le`, `mixedError_blocked_sum`: `blocked w`, which covers every count,
  at the depth of its tree.
- `abs_ordered_sum_sub_le`, `mixedError_ordered_sum`: every order under the premise that it
  covers the `n` entries (`ReduceOrder.Covers`), at the depth of its tree
  (`TorchLean.IEEE754.abs_fold_add_sub_le`).

The rules are tried in this order, so `pairwise` and `blocked w` carry no coverage premise. The left
fold never reaches them: its programs are rewritten to the operators' own left folds.

The IEEE maximum and minimum of finite values decode to the real `max` and `min`, which are
associative and commutative, so on finite entries a maximum or minimum in an order that covers
them has the real value of the left fold (`toReal_ordered_maximum`, `toReal_ordered_minimum`, with
the `pairwise` and `blocked w` cases tried first). The stored values can differ, in the sign of a
zero. The rules rewrite such a fold to the left fold, whose analysis applies.

The import of a native binary32 sum in an order is the binary32 sum of the imports in that order
when that sum is finite (`toReal_ofFloat32_fold_add`), since a finite configured sum has finite
operands; the sum rules then bound it.

A product in an order that covers its `n` entries rounds `n` times, as the left fold does, so its
relative error is the left fold's `(1 + u) ^ n - 1` in every order; only the absolute error from
underflow depends on the tree. `productBudget` gives the budget of one rounded product from those
of its operands, and `mixedError_ordered_prod` folds it over the order's tree: a finite product is
the left fold of the values with relative error `(1 + u) ^ n - 1` (`productBudget_relative`,
`ReduceOrder.nodeCount_tree`) and the absolute error of that fold, with `abs_ordered_prod_sub_le`
as an absolute bound and the `pairwise` and `blocked w` cases tried first. The import of a native
binary32 product in an order is the binary32 product of the imports (`toReal_ofFloat32_fold_mul`).
-/

public section

open FloatLib.Numerics (ReductionTree)

namespace TorchLean.Tensor

namespace ReduceOrder

/-- A fold over `a` entries is the fold over `b = a` entries of the entries cast back. -/
theorem fold_cast {β : Type*} (order : ReduceOrder) {a b : Nat} (h : a = b) (step : β → β → β)
    (start : β) (value : Fin a → β) :
    order.fold a step start value = order.fold b step start fun i => value (Fin.cast h.symm i) := by
  subst h
  rfl

/-- A fold over the indices of a vector is the fold over its coordinates. -/
theorem fold_size_singleton {β : Type*} (order : ReduceOrder) (n : Nat) (step : β → β → β)
    (start : β) (value : Fin n → β) :
    order.fold n step start value =
      order.fold (Shape.size [n]) step start
        fun i => value (Shape.Coord.unlinearize i : Shape.Coord [n]).1 := by
  have size : n = Shape.size [n] := by simp [Shape.size]
  rw [fold_cast order size]
  congr 1
  funext i
  congr 1
  apply Fin.ext
  simp [Shape.Coord.unlinearizeFast_eq, Shape.Coord.unlinearizeFast]
  rfl

/-- Folds over the coordinates of equal shapes agree when the entries agree at coordinates with
equal indices. -/
theorem fold_unlinearize_congr {β : Type*} (order : ReduceOrder) {t u : Shape} (h : t = u)
    (step : β → β → β) (start : β) (f : t.Coord → β) (g : u.Coord → β)
    (hfg : ∀ (c : t.Coord) (c' : u.Coord),
      (∀ k, Shape.Coord.index c k = Shape.Coord.index c' k) → f c = g c') :
    order.fold (Shape.size t) step start (fun i => f (Shape.Coord.unlinearize i)) =
      order.fold (Shape.size u) step start (fun i => g (Shape.Coord.unlinearize i)) := by
  subst h
  rw [funext fun c => hfg c c fun _ => rfl]

/-- A map `f` that takes `step` to `step'` on values where `p` holds, a property that `step` keeps,
takes a fold of such values to the fold of their images, and `p` holds of the fold. -/
theorem map_fold_of_forall {β γ : Type*} {p : β → Prop} (order : ReduceOrder) (n : Nat)
    (f : β → γ) {step : β → β → β} {step' : γ → γ → γ} (hp : ∀ x y, p x → p y → p (step x y))
    (hstep : ∀ x y, p x → p y → f (step x y) = step' (f x) (f y)) (start : β)
    (value : Fin n → β) (hstart : p start) (hvalue : ∀ i, p (value i)) :
    p (order.fold n step start value) ∧
      f (order.fold n step start value) = order.fold n step' (f start) fun i => f (value i) := by
  have key : ∀ t : ReductionTree (Option (Fin n)),
      p (t.eval step (·.elim start value)) ∧ f (t.eval step (·.elim start value)) =
        t.eval step' (·.elim (f start) fun i => f (value i)) := by
    intro t
    induction t with
    | leaf o =>
      cases o with
      | none => exact ⟨hstart, rfl⟩
      | some i => exact ⟨hvalue i, rfl⟩
    | node a b ha hb =>
      refine ⟨hp _ _ ha.1 hb.1, ?_⟩
      simp only [ReductionTree.eval]
      rw [hstep _ _ ha.1 hb.1, ha.2, hb.2]
  exact key _

/-- A map `f` that takes `step` to `step'` on values whose images satisfy `p`, where `p` of a result
of `step'` gives `p` of both operands, takes a fold to the fold of the images when `p` holds of
that fold. -/
theorem map_fold_of_operands {β γ : Type*} {p : γ → Prop} (order : ReduceOrder) (n : Nat)
    (f : β → γ) {step : β → β → β} {step' : γ → γ → γ}
    (hstep : ∀ x y, p (f x) → p (f y) → f (step x y) = step' (f x) (f y))
    (hoperands : ∀ x y, p (step' x y) → p x ∧ p y) (start : β) (value : Fin n → β)
    (h : p (order.fold n step' (f start) fun i => f (value i))) :
    f (order.fold n step start value) = order.fold n step' (f start) fun i => f (value i) := by
  have key : ∀ t : ReductionTree (Option (Fin n)),
      p (t.eval step' (·.elim (f start) fun i => f (value i))) →
        f (t.eval step (·.elim start value)) =
          t.eval step' (·.elim (f start) fun i => f (value i)) := by
    intro t
    induction t with
    | leaf o => intro _; cases o <;> rfl
    | node a b ha hb =>
      intro h
      simp only [ReductionTree.eval] at h ⊢
      obtain ⟨hpa, hpb⟩ := hoperands _ _ h
      rw [hstep _ _ (by rw [ha hpa]; exact hpa) (by rw [hb hpb]; exact hpb), ha hpa, hb hpb]
  exact key _ h

/-- Under the premises of `map_fold_of_forall`, with `step'` associative and commutative, the image
of a fold in an order that covers its entries is the image of the left fold. -/
theorem map_fold_eq_leftFold {β γ : Type*} {p : β → Prop} {order : ReduceOrder} {n : Nat}
    (hcov : order.Covers n) (f : β → γ) {step : β → β → β} {step' : γ → γ → γ}
    (hp : ∀ x y, p x → p y → p (step x y))
    (hstep : ∀ x y, p x → p y → f (step x y) = step' (f x) (f y))
    (hassoc : ∀ x y z, step' (step' x y) z = step' x (step' y z))
    (hcomm : ∀ x y, step' x y = step' y x) (start : β) (value : Fin n → β) (hstart : p start)
    (hvalue : ∀ i, p (value i)) :
    f (order.fold n step start value) =
      f ((List.finRange n).foldl (fun acc i => step acc (value i)) start) := by
  rw [← fold_leftFold, (map_fold_of_forall order n f hp hstep start value hstart hvalue).2,
    (map_fold_of_forall leftFold n f hp hstep start value hstart hvalue).2]
  exact fold_eq_leftFold hcov hassoc hcomm _ _

/-- The tree of an order that covers `n` entries has `n` nodes, one per entry after the start. -/
theorem nodeCount_tree {order : ReduceOrder} {n : Nat} (hcov : order.Covers n) :
    (order.tree n).nodeCount = n := by
  have h := List.Perm.length_eq hcov
  rw [ReductionTree.length_leaves] at h
  simpa using h

end ReduceOrder

variable {α : Type} [Storage α] {s : Shape}

/-- An entry of a sum in the order `order` is the order's fold, from zero, of the entries the
output coordinate reduces. -/
@[roundoff_rule]
theorem reduceIn_sum_apply [Add α] [Zero α] (order : ReduceOrder) (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .sum) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) :
    reduceIn order .sum x dims keepdim h o =
      order.fold _ (· + ·) 0 fun i =>
        x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) :=
  Rep.get_ofFn _ o

/-- An entry of a product in the order `order` is the order's fold, from one, of the entries the
output coordinate reduces. -/
@[roundoff_rule]
theorem reduceIn_prod_apply [Mul α] [One α] (order : ReduceOrder) (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .prod) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) :
    reduceIn order .prod x dims keepdim h o =
      order.fold _ (· * ·) 1 fun i =>
        x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) :=
  Rep.get_ofFn _ o

/-- An entry of a maximum in the order `order` is the order's fold of `Context.maximum`, from the
first entry the output coordinate reduces, of those entries. -/
@[roundoff_rule]
theorem reduceIn_max_apply [Context α] (order : ReduceOrder) (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .max) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) :
    reduceIn order .max x dims keepdim h o =
      order.fold _ Context.maximum
        (x (s.reduceSource dims keepdim o
          (Shape.Coord.unlinearize ⟨0, Shape.size_selectAxes_pos (h rfl)⟩)))
        fun i => x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) :=
  Rep.get_ofFn _ o

/-- An entry of a minimum in the order `order` is the order's fold of `Context.minimum`, from the
first entry the output coordinate reduces, of those entries. -/
@[roundoff_rule]
theorem reduceIn_min_apply [Context α] (order : ReduceOrder) (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : (ReduceOp.Scalar.seed (op := .min) (α := α)).isNone → ∀ d ∈ dims, 0 < s.extent d)
    (o : (s.reduce dims keepdim).Coord) :
    reduceIn order .min x dims keepdim h o =
      order.fold _ Context.minimum
        (x (s.reduceSource dims keepdim o
          (Shape.Coord.unlinearize ⟨0, Shape.size_selectAxes_pos (h rfl)⟩)))
        fun i => x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)) :=
  Rep.get_ofFn _ o

/-- An entry of a matrix product in the order `order` is the order's fold, from zero, of
`A i k * B k j`. -/
@[roundoff_rule]
theorem matmulIn_apply [Add α] [Mul α] [Zero α] {m n p : Nat} (order : ReduceOrder)
    (A : Tensor α [m, n]) (B : Tensor α [n, p]) (c : Shape.Coord [m, p]) :
    matmulIn order A B c =
      order.fold n (· + ·) 0 fun k => A (c.1, k, PUnit.unit) * B (k, c.2.1, PUnit.unit) := by
  rw [matmulIn, contractIn, Rep.get_ofFn]
  generalize hc' : Shape.Coord.reindexEq _ c = c'
  have hc : ∀ k, Shape.Coord.index c' k = Shape.Coord.index c k := fun k => by
    rw [← hc']
    exact Shape.Coord.index_reindexEq _ c k
  refine ((ReduceOrder.fold_size_singleton order n _ 0 _).trans
    (ReduceOrder.fold_unlinearize_congr order rfl _ 0
      (fun (r : Shape.Coord [n]) => A (c.1, r.1, PUnit.unit) * B (r.1, c.2.1, PUnit.unit))
      (fun r => A (Shape.contractLeft [m, n] [n, p] _ _ c' r) *
        B ((Shape.ContractValid.matMul (m := m) (n := n) (p := p)).contractRight c' r)) ?_)).symm
  intro r r' hrr
  have hA : ((c.1, r.1, PUnit.unit) : Shape.Coord [m, n]) =
      Shape.contractLeft [m, n] [n, p] _ _ c' r' := by
    refine Shape.Coord.ext_index (s := [m, n]) ?_
    intro k hk
    match k, hk with
    | 0, _ => exact (hc 0).symm
    | 1, _ => exact hrr 0
    | k + 2, hk =>
      exact absurd hk (by
        simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
        omega)
  have hB : ((r.1, c.2.1, PUnit.unit) : Shape.Coord [n, p]) =
      (Shape.ContractValid.matMul (m := m) (n := n) (p := p)).contractRight c' r' := by
    refine Shape.Coord.ext_index (s := [n, p]) ?_
    intro k hk
    match k, hk with
    | 0, _ => exact hrr 0
    | 1, _ => exact (hc 1).symm
    | k + 2, hk =>
      exact absurd hk (by
        simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
        omega)
  show A _ * B _ = A _ * B _
  rw [hA, hB]

/-- An entry of a batched matrix product in the order `order`, at batch coordinate `b` and matrix
coordinate `(i, j)`, is the order's fold, from zero, of `left b i k * right b k j`. -/
@[roundoff_rule]
theorem matmulCommonBatchIn_apply [Add α] [Mul α] [Zero α] {batch : Shape} {m n p : Nat}
    (order : ReduceOrder) (left : Tensor α (batch ++ [m, n])) (right : Tensor α (batch ++ [n, p]))
    (coordinate : (batch ++ [m, p]).Coord) :
    matmulCommonBatchIn order left right coordinate =
      let split := Shape.Coord.appendEquiv batch [m, p] coordinate
      order.fold n (· + ·) 0 fun k =>
        left ((Shape.Coord.appendEquiv batch [m, n]).symm (split.1, split.2.1, k, PUnit.unit)) *
          right ((Shape.Coord.appendEquiv batch [n, p]).symm
            (split.1, k, split.2.2.1, PUnit.unit)) := by
  induction batch with
  | scalar =>
      change matmulIn order left right coordinate = _
      rw [matmulIn_apply]
      rfl
  | dim count batch ih =>
      change (Tensor.dim fun outer =>
        matmulCommonBatchIn order (Tensor.unstack left outer)
          (Tensor.unstack right outer)) coordinate = _
      rw [Tensor.dim_apply, ih]
      simp only [Tensor.unstack_apply]
      rfl

end TorchLean.Tensor

namespace TorchLean.Roundoff

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange FloatLib.Numerics
open ExecFloat.Binary (toReal isFinite ofFloat32)
open Tensor (ReduceOrder)

section Sums

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]
  [cap : ExecFloat.Add (Configured.Family format code plan)]
  [ExecFloat.Capability.HasSpec cap Configured.Spec.add]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- A finite sum in an order that covers its `n` entries is within `((1 + u) ^ d - 1)` times the
absolute sum of the left fold of the values, `d` the depth of the order's tree. -/
theorem abs_ordered_sum_sub_le {order : ReduceOrder} {n : Nat} (hcov : order.Covers n)
    (start : Value) (value : Fin n → Value)
    (hresult : isFinite (order.fold n (· + ·) start value) = true)
    (hformat : format.isIEEE = true) :
    |toReal (order.fold n (· + ·) start value) -
        (List.finRange n).foldl (fun total i => total + toReal (value i)) (toReal start)| ≤
      ((1 + Model.unitRoundoffAt format) ^ (order.tree n).depth - 1) *
        (|toReal start| + ∑ i, |toReal (value i)|) := by
  rw [List.finRange_foldl_add_eq_add_sum]
  exact IEEE754.abs_fold_add_sub_le hcov start value hresult hformat

/-- A finite pairwise sum of `n` entries is within `((1 + u) ^ ⌈log₂ (n + 1)⌉ - 1)` times the
absolute sum of the left fold of the values. -/
theorem abs_pairwise_sum_sub_le (n : Nat) (start : Value) (value : Fin n → Value)
    (hresult : isFinite (ReduceOrder.pairwise.fold n (· + ·) start value) = true)
    (hformat : format.isIEEE = true) :
    |toReal (ReduceOrder.pairwise.fold n (· + ·) start value) -
        (List.finRange n).foldl (fun total i => total + toReal (value i)) (toReal start)| ≤
      ((1 + Model.unitRoundoffAt format) ^ Nat.clog 2 (n + 1) - 1) *
        (|toReal start| + ∑ i, |toReal (value i)|) := by
  rw [List.finRange_foldl_add_eq_add_sum]
  exact IEEE754.abs_fold_pairwise_sub_le n start value hresult hformat

/-- `abs_ordered_sum_sub_le` in `blocked w`, which covers every count. -/
theorem abs_blocked_sum_sub_le (w n : Nat) (start : Value) (value : Fin n → Value)
    (hresult : isFinite ((ReduceOrder.blocked w).fold n (· + ·) start value) = true)
    (hformat : format.isIEEE = true) :
    |toReal ((ReduceOrder.blocked w).fold n (· + ·) start value) -
        (List.finRange n).foldl (fun total i => total + toReal (value i)) (toReal start)| ≤
      ((1 + Model.unitRoundoffAt format) ^ ((ReduceOrder.blocked w).tree n).depth - 1) *
        (|toReal start| + ∑ i, |toReal (value i)|) :=
  abs_ordered_sum_sub_le (ReduceOrder.covers_blocked w n) start value hresult hformat

/-- `abs_ordered_sum_sub_le` as a mixed error with no input perturbation. -/
theorem mixedError_ordered_sum {order : ReduceOrder} {n : Nat} (hcov : order.Covers n)
    (start : Value) (value : Fin n → Value)
    (hresult : isFinite (order.fold n (· + ·) start value) = true)
    (hformat : format.isIEEE = true) :
    MixedError (toReal (order.fold n (· + ·) start value))
      ((List.finRange n).foldl (fun total i => total + toReal (value i)) (toReal start)) 0
      (((1 + Model.unitRoundoffAt format) ^ (order.tree n).depth - 1) *
        (|toReal start| + ∑ i, |toReal (value i)|)) :=
  MixedError.of_abs_sub_le (abs_ordered_sum_sub_le hcov start value hresult hformat)

/-- `abs_pairwise_sum_sub_le` as a mixed error with no input perturbation. -/
theorem mixedError_pairwise_sum (n : Nat) (start : Value) (value : Fin n → Value)
    (hresult : isFinite (ReduceOrder.pairwise.fold n (· + ·) start value) = true)
    (hformat : format.isIEEE = true) :
    MixedError (toReal (ReduceOrder.pairwise.fold n (· + ·) start value))
      ((List.finRange n).foldl (fun total i => total + toReal (value i)) (toReal start)) 0
      (((1 + Model.unitRoundoffAt format) ^ Nat.clog 2 (n + 1) - 1) *
        (|toReal start| + ∑ i, |toReal (value i)|)) :=
  MixedError.of_abs_sub_le (abs_pairwise_sum_sub_le n start value hresult hformat)

/-- `abs_blocked_sum_sub_le` as a mixed error with no input perturbation. -/
theorem mixedError_blocked_sum (w n : Nat) (start : Value) (value : Fin n → Value)
    (hresult : isFinite ((ReduceOrder.blocked w).fold n (· + ·) start value) = true)
    (hformat : format.isIEEE = true) :
    MixedError (toReal ((ReduceOrder.blocked w).fold n (· + ·) start value))
      ((List.finRange n).foldl (fun total i => total + toReal (value i)) (toReal start)) 0
      (((1 + Model.unitRoundoffAt format) ^ ((ReduceOrder.blocked w).tree n).depth - 1) *
        (|toReal start| + ∑ i, |toReal (value i)|)) :=
  mixedError_ordered_sum (ReduceOrder.covers_blocked w n) start value hresult hformat

attribute [roundoff_rule] abs_pairwise_sum_sub_le abs_blocked_sum_sub_le abs_ordered_sum_sub_le
  mixedError_pairwise_sum mixedError_blocked_sum mixedError_ordered_sum

end Sums

/-- The budget of a rounded product from those of its operands `x` and `y`, each the absolute
value of an exact operand, a relative error and an absolute error: the mixed error of the exact
product (`MixedError.mul`) after one rounding of relative error `u` and absolute error `η`
(`MixedError.trans`). -/
def productBudget (u η : ℝ) (x y : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (x.1 * y.1, (1 + u) * (x.2.1 + y.2.1 + x.2.1 * y.2.1) + u,
    (1 + u) * ((1 + y.2.1) * x.2.2 * y.1 + (1 + x.2.1) * y.2.2 * x.1 + x.2.2 * y.2.2) + η)

/-- Over a tree whose leaves have no relative error, the relative error of `productBudget` is
`(1 + u) ^ k - 1`, `k` the tree's nodes. -/
theorem productBudget_relative (u η : ℝ) {ι : Type} (t : ReductionTree ι)
    (leaf : ι → ℝ × ℝ × ℝ) (hleaf : ∀ i, (leaf i).2.1 = 0) :
    (t.eval (productBudget u η) leaf).2.1 = (1 + u) ^ t.nodeCount - 1 := by
  induction t with
  | leaf i => simp [ReductionTree.eval, ReductionTree.nodeCount, hleaf]
  | node a b ha hb =>
    simp only [ReductionTree.eval, ReductionTree.nodeCount, productBudget, ha, hb]
    ring

section Products

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]
  [cap : ExecFloat.Mul (Configured.Family format code plan)]
  [ExecFloat.Capability.HasSpec cap Configured.Spec.mul]

local notation "Value" => ExecFloat (Configured.Family format code plan)

local notation "Budget" =>
  productBudget (Model.unitRoundoffAt format) (Model.underflowErrorAt format)

/-- A finite product in an order that covers its `n` entries is the left fold of the values with
the left fold's relative error `(1 + u) ^ n - 1` and the absolute error that `productBudget`
accumulates over the order's tree. -/
theorem mixedError_ordered_prod {order : ReduceOrder} {n : Nat} (hcov : order.Covers n)
    (start : Value) (value : Fin n → Value)
    (hresult : isFinite (order.fold n (· * ·) start value) = true)
    (hformat : format.isIEEE = true) :
    MixedError (toReal (order.fold n (· * ·) start value))
      ((List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start))
      ((1 + Model.unitRoundoffAt format) ^ n - 1)
      (order.fold n Budget (|toReal start|, 0, 0) fun i => (|toReal (value i)|, 0, 0)).2.2 := by
  have key : ∀ t : ReductionTree (Option (Fin n)),
      isFinite (t.eval (· * ·) (·.elim start value)) = true →
      (t.eval Budget (·.elim (|toReal start|, 0, 0) fun i => (|toReal (value i)|, 0, 0))).1 =
        |t.eval (· * ·) (·.elim (toReal start) fun i => toReal (value i))| ∧
      MixedError (toReal (t.eval (· * ·) (·.elim start value)))
        (t.eval (· * ·) (·.elim (toReal start) fun i => toReal (value i)))
        (t.eval Budget
          (·.elim (|toReal start|, 0, 0) fun i => (|toReal (value i)|, 0, 0))).2.1
        (t.eval Budget
          (·.elim (|toReal start|, 0, 0) fun i => (|toReal (value i)|, 0, 0))).2.2 := by
    intro t
    induction t with
    | leaf o => intro _; cases o <;> exact ⟨rfl, .refl _⟩
    | node a b ha hb =>
      intro h
      simp only [ReductionTree.eval] at h ⊢
      obtain ⟨hfa, hfb⟩ := ExecFloat.Binary.isFinite_of_isFinite_mul h hformat
      obtain ⟨hma, hea⟩ := ha hfa
      obtain ⟨hmb, heb⟩ := hb hfb
      refine ⟨by rw [productBudget, hma, hmb, abs_mul], ?_⟩
      have step := (ExecFloat.Binary.mul_rounding h hformat).trans (hea.mul heb)
      rw [← hma, ← hmb] at step
      exact step
  have hrel := productBudget_relative (Model.unitRoundoffAt format) (Model.underflowErrorAt format)
    (order.tree n) (·.elim (|toReal start|, 0, 0) fun i => (|toReal (value i)|, 0, 0))
    fun o => by cases o <;> rfl
  rw [ReduceOrder.nodeCount_tree hcov] at hrel
  have herr := (key (order.tree n) hresult).2
  rw [hrel] at herr
  have h := ReduceOrder.fold_eq_leftFold hcov (step := fun x y : ℝ => x * y) mul_assoc mul_comm
    (toReal start) fun i => toReal (value i)
  rw [ReduceOrder.fold_leftFold] at h
  rw [← h]
  exact herr

/-- `mixedError_ordered_prod` as an absolute error. -/
theorem abs_ordered_prod_sub_le {order : ReduceOrder} {n : Nat} (hcov : order.Covers n)
    (start : Value) (value : Fin n → Value)
    (hresult : isFinite (order.fold n (· * ·) start value) = true)
    (hformat : format.isIEEE = true) :
    |toReal (order.fold n (· * ·) start value) -
        (List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start)| ≤
      ((1 + Model.unitRoundoffAt format) ^ n - 1) *
          |(List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start)| +
        (order.fold n Budget (|toReal start|, 0, 0) fun i => (|toReal (value i)|, 0, 0)).2.2 :=
  (mixedError_ordered_prod hcov start value hresult hformat).bound

/-- `mixedError_ordered_prod` in `pairwise`. -/
theorem mixedError_pairwise_prod (n : Nat) (start : Value) (value : Fin n → Value)
    (hresult : isFinite (ReduceOrder.pairwise.fold n (· * ·) start value) = true)
    (hformat : format.isIEEE = true) :
    MixedError (toReal (ReduceOrder.pairwise.fold n (· * ·) start value))
      ((List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start))
      ((1 + Model.unitRoundoffAt format) ^ n - 1)
      (ReduceOrder.pairwise.fold n Budget (|toReal start|, 0, 0)
        fun i => (|toReal (value i)|, 0, 0)).2.2 :=
  mixedError_ordered_prod (ReduceOrder.covers_pairwise n) start value hresult hformat

/-- `mixedError_ordered_prod` in `blocked w`. -/
theorem mixedError_blocked_prod (w n : Nat) (start : Value) (value : Fin n → Value)
    (hresult : isFinite ((ReduceOrder.blocked w).fold n (· * ·) start value) = true)
    (hformat : format.isIEEE = true) :
    MixedError (toReal ((ReduceOrder.blocked w).fold n (· * ·) start value))
      ((List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start))
      ((1 + Model.unitRoundoffAt format) ^ n - 1)
      ((ReduceOrder.blocked w).fold n Budget (|toReal start|, 0, 0)
        fun i => (|toReal (value i)|, 0, 0)).2.2 :=
  mixedError_ordered_prod (ReduceOrder.covers_blocked w n) start value hresult hformat

/-- `abs_ordered_prod_sub_le` in `pairwise`. -/
theorem abs_pairwise_prod_sub_le (n : Nat) (start : Value) (value : Fin n → Value)
    (hresult : isFinite (ReduceOrder.pairwise.fold n (· * ·) start value) = true)
    (hformat : format.isIEEE = true) :
    |toReal (ReduceOrder.pairwise.fold n (· * ·) start value) -
        (List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start)| ≤
      ((1 + Model.unitRoundoffAt format) ^ n - 1) *
          |(List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start)| +
        (ReduceOrder.pairwise.fold n Budget (|toReal start|, 0, 0)
          fun i => (|toReal (value i)|, 0, 0)).2.2 :=
  abs_ordered_prod_sub_le (ReduceOrder.covers_pairwise n) start value hresult hformat

/-- `abs_ordered_prod_sub_le` in `blocked w`. -/
theorem abs_blocked_prod_sub_le (w n : Nat) (start : Value) (value : Fin n → Value)
    (hresult : isFinite ((ReduceOrder.blocked w).fold n (· * ·) start value) = true)
    (hformat : format.isIEEE = true) :
    |toReal ((ReduceOrder.blocked w).fold n (· * ·) start value) -
        (List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start)| ≤
      ((1 + Model.unitRoundoffAt format) ^ n - 1) *
          |(List.finRange n).foldl (fun product i => product * toReal (value i)) (toReal start)| +
        ((ReduceOrder.blocked w).fold n Budget (|toReal start|, 0, 0)
          fun i => (|toReal (value i)|, 0, 0)).2.2 :=
  abs_ordered_prod_sub_le (ReduceOrder.covers_blocked w n) start value hresult hformat

attribute [roundoff_rule] abs_pairwise_prod_sub_le abs_blocked_prod_sub_le abs_ordered_prod_sub_le
  mixedError_pairwise_prod mixedError_blocked_prod mixedError_ordered_prod

end Products

section Extrema

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
  [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" =>
  ExecFloat.Binary format.expWidth format.fracWidth format.encoding format.exponentBias
    format.expWidth_ge_two format.fracWidth_pos format.exponentBias_pos
    format.exponentBias_le_maxFinite plan code

/-- On finite entries the IEEE maximum in an order that covers the `n` entries has the real value
of the left fold. -/
theorem toReal_ordered_maximum {order : ReduceOrder} {n : Nat} (hcov : order.Covers n)
    (start : Value) (value : Fin n → Value) (hstart : isFinite start = true)
    (hvalue : ∀ i, isFinite (value i) = true) :
    toReal (order.fold n Context.maximum start value) =
      toReal ((List.finRange n).foldl (fun acc i => Context.maximum acc (value i)) start) :=
  ReduceOrder.map_fold_eq_leftFold (p := fun v : Value => isFinite v = true)
    (step := Context.maximum) (step' := max) hcov toReal (fun _ _ => isFinite_maximum)
    (fun _ _ => toReal_maximum) max_assoc max_comm start value hstart hvalue

/-- On finite entries the IEEE minimum in an order that covers the `n` entries has the real value
of the left fold. -/
theorem toReal_ordered_minimum {order : ReduceOrder} {n : Nat} (hcov : order.Covers n)
    (start : Value) (value : Fin n → Value) (hstart : isFinite start = true)
    (hvalue : ∀ i, isFinite (value i) = true) :
    toReal (order.fold n Context.minimum start value) =
      toReal ((List.finRange n).foldl (fun acc i => Context.minimum acc (value i)) start) :=
  ReduceOrder.map_fold_eq_leftFold (p := fun v : Value => isFinite v = true)
    (step := Context.minimum) (step' := min) hcov toReal (fun _ _ => isFinite_minimum)
    (fun _ _ => toReal_minimum) min_assoc min_comm start value hstart hvalue

/-- `toReal_ordered_maximum` in `pairwise`. -/
theorem toReal_pairwise_maximum (n : Nat) (start : Value) (value : Fin n → Value)
    (hstart : isFinite start = true) (hvalue : ∀ i, isFinite (value i) = true) :
    toReal (ReduceOrder.pairwise.fold n Context.maximum start value) =
      toReal ((List.finRange n).foldl (fun acc i => Context.maximum acc (value i)) start) :=
  toReal_ordered_maximum (ReduceOrder.covers_pairwise n) start value hstart hvalue

/-- `toReal_ordered_maximum` in `blocked w`. -/
theorem toReal_blocked_maximum (w n : Nat) (start : Value) (value : Fin n → Value)
    (hstart : isFinite start = true) (hvalue : ∀ i, isFinite (value i) = true) :
    toReal ((ReduceOrder.blocked w).fold n Context.maximum start value) =
      toReal ((List.finRange n).foldl (fun acc i => Context.maximum acc (value i)) start) :=
  toReal_ordered_maximum (ReduceOrder.covers_blocked w n) start value hstart hvalue

/-- `toReal_ordered_minimum` in `pairwise`. -/
theorem toReal_pairwise_minimum (n : Nat) (start : Value) (value : Fin n → Value)
    (hstart : isFinite start = true) (hvalue : ∀ i, isFinite (value i) = true) :
    toReal (ReduceOrder.pairwise.fold n Context.minimum start value) =
      toReal ((List.finRange n).foldl (fun acc i => Context.minimum acc (value i)) start) :=
  toReal_ordered_minimum (ReduceOrder.covers_pairwise n) start value hstart hvalue

/-- `toReal_ordered_minimum` in `blocked w`. -/
theorem toReal_blocked_minimum (w n : Nat) (start : Value) (value : Fin n → Value)
    (hstart : isFinite start = true) (hvalue : ∀ i, isFinite (value i) = true) :
    toReal ((ReduceOrder.blocked w).fold n Context.minimum start value) =
      toReal ((List.finRange n).foldl (fun acc i => Context.minimum acc (value i)) start) :=
  toReal_ordered_minimum (ReduceOrder.covers_blocked w n) start value hstart hvalue

attribute [roundoff_rule] toReal_pairwise_maximum toReal_blocked_maximum toReal_ordered_maximum
  toReal_pairwise_minimum toReal_blocked_minimum toReal_ordered_minimum

end Extrema

/-- The import of a native binary32 sum in an order is the binary32 sum of the imports in that
order, when that sum is finite. -/
@[roundoff_rule]
theorem toReal_ofFloat32_fold_add (order : ReduceOrder) (n : Nat) (start : Float32)
    (value : Fin n → Float32)
    (hresult : isFinite (order.fold n (· + ·) (ofFloat32 start : Binary32)
      fun i => ofFloat32 (value i)) = true) :
    toReal (ofFloat32 (order.fold n (· + ·) start value) : Binary32) =
      toReal (order.fold n (· + ·) (ofFloat32 start : Binary32) fun i => ofFloat32 (value i)) :=
  congrArg toReal (ReduceOrder.map_fold_of_operands order n (p := fun v => isFinite v = true)
    (fun x : Float32 => (ofFloat32 x : Binary32))
    (fun x y hx hy => ExecFloat.Binary.ofFloat32_add_of_isFinite x y
      ((ExecFloat.Binary.isFinite_ofFloat32 x).symm.trans hx)
      ((ExecFloat.Binary.isFinite_ofFloat32 y).symm.trans hy))
    (fun _ _ h => ExecFloat.Binary.isFinite_of_isFinite_add h) start value hresult)

/-- The import of a native binary32 product in an order is the binary32 product of the imports
in that order, when that product is finite. -/
@[roundoff_rule]
theorem toReal_ofFloat32_fold_mul (order : ReduceOrder) (n : Nat) (start : Float32)
    (value : Fin n → Float32)
    (hresult : isFinite (order.fold n (· * ·) (ofFloat32 start : Binary32)
      fun i => ofFloat32 (value i)) = true) :
    toReal (ofFloat32 (order.fold n (· * ·) start value) : Binary32) =
      toReal (order.fold n (· * ·) (ofFloat32 start : Binary32) fun i => ofFloat32 (value i)) :=
  congrArg toReal (ReduceOrder.map_fold_of_operands order n (p := fun v => isFinite v = true)
    (fun x : Float32 => (ofFloat32 x : Binary32))
    (fun x y hx hy => ExecFloat.Binary.ofFloat32_mul_of_isFinite x y
      ((ExecFloat.Binary.isFinite_ofFloat32 x).symm.trans hx)
      ((ExecFloat.Binary.isFinite_ofFloat32 y).symm.trans hy))
    (fun _ _ h => ExecFloat.Binary.isFinite_of_isFinite_mul h) start value hresult)

attribute [roundoff_rule] Tensor.cummaxIn_fst_apply Tensor.cumminIn_fst_apply

end TorchLean.Roundoff
