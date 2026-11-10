/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.NN.FDeriv.TensorCoordinates

/-!
# Derivative certificates of the scans

We prove the `derivative` certificates of `Op.cumprod` and `Op.selectiveScan` at every input.
Over the reals an entry of `Tensor.cumprod` is a finite product of entries of its fiber, so it has
the derivative `cumprodCLM`, and swapping the double sum over each fiber gives the adjoint identity
with `cumprodBackward` (`inner_cumprodJVP`). The rule `HasPullbackAt.cumprod` lets `op_deriv`
prove the cumulative product certificate.

Each fiber of `Tensor.selectiveScan` along `dim` is the recursion `StateSpace.stateAt`, whose
derivative `StateSpace.stateDeriv` follows by induction on the steps (`hasFDerivAt_stateAt`).
Summation by parts against the adjoint states of `scanAdjoints` (`sum_stateDeriv_mul_eq`) gives
the adjoint identity with `selectiveScanBackward` (`hasPullbackAt_selectiveScan`). The rule
`HasPullbackAt.selectiveScan` lets `op_deriv` prove the certificate jointly in the two
coefficients, the input and the initial state.
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

namespace Tensor

/-! ## Folds over prefixes -/

/-- A left fold over `Tensor.ofFn f` is the fold of `f` over `List.finRange n`. -/
private theorem foldl_ofFn {β γ : Type} [Storage β] {n : Nat} (step : γ → β → γ) (init : γ)
    (f : Fin n → β) :
    Tensor.foldl step init (Tensor.ofFn f) =
      (List.finRange n).foldl (fun acc i => step acc (f i)) init := by
  rw [Tensor.ofFn, Rep.ofFn_eq_map_coordinates, Rep.foldl_map,
    ← Fin.foldl_eq_foldl_finRange, fin_foldl_eq_coordinates]

/-- A product over `Fin (k + 1)` read through `Fin.castLE` is the product over the indices of
`Fin n` at most `k`. -/
@[to_additive /-- A sum over `Fin (k + 1)` read through `Fin.castLE` is the sum over the indices
of `Fin n` at most `k`. -/]
theorem prod_castLE_eq_prod_Iic {M : Type*} [CommMonoid M] {n : Nat} (k : Fin n)
    (f : Fin n → M) :
    ∏ i : Fin (k.val + 1), f (Fin.castLE k.isLt i) = ∏ i ∈ Finset.Iic k, f i := by
  refine Finset.prod_nbij' (fun i => Fin.castLE k.isLt i)
    (fun j => ⟨min j.val k.val, by omega⟩) ?_ ?_ ?_ ?_ ?_
  · intro i _
    have hi := i.isLt
    simp only [Finset.mem_Iic, Fin.le_iff_val_le_val, Fin.val_castLE]
    omega
  · intro j _
    exact Finset.mem_univ _
  · intro i _
    have hi := i.isLt
    exact Fin.ext (by simp only [Fin.val_castLE]; omega)
  · intro j hj
    simp only [Finset.mem_Iic, Fin.le_iff_val_le_val] at hj
    exact Fin.ext (by simp only [Fin.val_castLE]; omega)
  · intro i _
    rfl

/-- A left fold of sums over the prefix through `k` of `Fin n` is the sum over the indices at most
`k`. -/
private theorem foldl_add_prefix {n : Nat} (k : Fin n) (f : Fin n → ℝ) :
    ((List.finRange (k.val + 1)).map (Fin.castLE k.isLt)).foldl (fun acc i => acc + f i) 0 =
      ∑ i ∈ Finset.Iic k, f i := by
  rw [List.foldl_map, ← sum_castLE_eq_sum_Iic, Fin.sum_univ_def, List.sum_eq_foldl,
    List.foldl_map]

/-- A left fold of products over the prefix through `k` of `Fin n` that skips the index `i` is the
product over the indices at most `k` other than `i`. -/
private theorem foldl_mul_prefix_erase {n : Nat} (k i : Fin n) (f : Fin n → ℝ) :
    ((List.finRange (k.val + 1)).map (Fin.castLE k.isLt)).foldl
        (fun product j => if j = i then product else product * f j) 1 =
      ∏ j ∈ (Finset.Iic k).erase i, f j := by
  have hstep : (fun (product : ℝ) (j : Fin n) => if j = i then product else product * f j) =
      fun product j => product * if j = i then 1 else f j := by
    funext product j
    split_ifs <;> simp
  rw [hstep, List.foldl_map, ← Finset.filter_ne', Finset.prod_filter, ← prod_castLE_eq_prod_Iic,
    Fin.prod_univ_def, List.prod_eq_foldl, List.foldl_map]
  congr 1
  funext product j
  congr 1
  split_ifs <;> simp_all

/-- A left fold over `List.finRange n` adding the values of `f` at the indices satisfying `p` is
the sum of `f` over those indices. -/
private theorem foldl_ite_add {n : Nat} (p : Fin n → Prop) [DecidablePred p] (f : Fin n → ℝ) :
    (List.finRange n).foldl (fun acc j => if p j then acc + f j else acc) 0 =
      ∑ j ∈ Finset.univ.filter p, f j := by
  have hstep : (fun (acc : ℝ) (j : Fin n) => if p j then acc + f j else acc) =
      fun acc j => acc + if p j then f j else 0 := by
    funext acc j
    split_ifs <;> simp
  rw [hstep, Finset.sum_filter, Fin.sum_univ_def, List.sum_eq_foldl, List.foldl_map]

/-- A sum over `Finset.Iic k` is the sum over `Fin n` of the values at the indices at most `k`. -/
private theorem sum_Iic_eq_sum_ite {n : Nat} (k : Fin n) (f : Fin n → ℝ) :
    ∑ i ∈ Finset.Iic k, f i = ∑ i, if i ≤ k then f i else 0 := by
  rw [← Finset.filter_ge_eq_Iic, Finset.sum_filter]

/-- A sum over `Finset.Ici k` is the sum over `Fin n` of the values at the indices at least `k`. -/
private theorem sum_Ici_eq_sum_ite {n : Nat} (k : Fin n) (f : Fin n → ℝ) :
    ∑ i ∈ Finset.Ici k, f i = ∑ i, if k ≤ i then f i else 0 := by
  rw [← Finset.filter_le_eq_Ici, Finset.sum_filter]

/-! ## Real forms of the cumulative product -/

variable {s : Shape}

/-- Over the reals, an entry of `cumprod x dim` is the product of the entries of its fiber along
`dim` with index at most its own. -/
theorem cumprod_apply_eq_prod (x : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord) :
    cumprod x dim c =
      ∏ i ∈ Finset.Iic (Shape.Coord.get c dim), x (Shape.Coord.update c dim i) := by
  rw [cumprod_apply, foldl_ofFn, ← prod_castLE_eq_prod_Iic, Fin.prod_univ_def,
    List.prod_eq_foldl, List.foldl_map]

/-- Over the reals, an entry of `cumprodJVP x dx dim` adds, over the indices `i` of its prefix
along `dim`, the tangent at `i` times the product of the other entries of the prefix. -/
theorem cumprodJVP_apply_eq_sum (x dx : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord) :
    cumprodJVP x dx dim c =
      ∑ i ∈ Finset.Iic (Shape.Coord.get c dim), dx (Shape.Coord.update c dim i) *
        ∏ j ∈ (Finset.Iic (Shape.Coord.get c dim)).erase i, x (Shape.Coord.update c dim j) := by
  rw [cumprodJVP, Rep.get_ofFn]
  simp only [foldl_mul_prefix_erase]
  rw [foldl_add_prefix]

/-- Over the reals, an entry of `cumprodBackward x g dim` adds, over the indices `j` along `dim`
at least its own, the gradient at `j` times the product of the entries with index at most `j` other
than its own. -/
theorem cumprodBackward_apply_eq_sum (x g : Tensor ℝ s) (dim : Fin s.rank) (c : s.Coord) :
    cumprodBackward x g dim c =
      ∑ j ∈ Finset.Ici (Shape.Coord.get c dim), g (Shape.Coord.update c dim j) *
        ∏ i ∈ (Finset.Iic j).erase (Shape.Coord.get c dim), x (Shape.Coord.update c dim i) := by
  rw [cumprodBackward_apply]
  simp only [foldl_mul_prefix_erase]
  rw [foldl_ite_add, Finset.filter_le_eq_Ici]

/-- The derivative of the real cumulative product along `dim` at `x`, as a continuous linear map:
it sends a tangent `u` to `cumprodJVP x u dim`. -/
noncomputable def cumprodCLM (x : Tensor ℝ s) (dim : Fin s.rank) : Tensor ℝ s →L[ℝ] Tensor ℝ s :=
  LinearMap.toContinuousLinearMap
    { toFun := fun u => cumprodJVP x u dim
      map_add' := fun u v => Rep.ext fun c => by
        simp only [cumprodJVP_apply_eq_sum, Tensor.add_apply, add_mul, Finset.sum_add_distrib]
      map_smul' := fun a u => Rep.ext fun c => by
        simp only [cumprodJVP_apply_eq_sum, Rep.smul_apply, RingHom.id_apply, smul_eq_mul,
          Finset.mul_sum, mul_assoc] }

/-- `cumprodCLM x dim` applied to `u` is `cumprodJVP x u dim`. -/
@[simp] theorem cumprodCLM_apply (x u : Tensor ℝ s) (dim : Fin s.rank) :
    cumprodCLM x dim u = cumprodJVP x u dim :=
  rfl

/-- Over the reals, the adjoint of `cumprodJVP x · dim` is `cumprodBackward x · dim`. -/
theorem inner_cumprodJVP (x u δ : Tensor ℝ s) (dim : Fin s.rank) :
    ⟪cumprodJVP x u dim, δ⟫ = ⟪u, cumprodBackward x δ dim⟫ := by
  rw [inner_eq_sum, inner_eq_sum]
  simp only [cumprodJVP_apply_eq_sum, cumprodBackward_apply_eq_sum, sum_Iic_eq_sum_ite,
    sum_Ici_eq_sum_ite,
    Finset.sum_mul, Finset.mul_sum, ite_mul, mul_ite, zero_mul, mul_zero]
  rw [← Fintype.sum_prod_type', ← Fintype.sum_prod_type']
  have hinv : Function.Involutive fun p : s.Coord × Fin (s.extent dim) =>
      (Shape.Coord.update p.1 dim p.2, Shape.Coord.get p.1 dim) := fun p => by
    simp only [Shape.Coord.update_update, Shape.Coord.update_get, Shape.Coord.get_update_self]
  refine Fintype.sum_equiv (hinv.toPerm _) _ _ fun p => ?_
  simp only [Function.Involutive.coe_toPerm, Shape.Coord.update_update, Shape.Coord.update_get,
    Shape.Coord.get_update_self]
  split_ifs <;> ring

/-- Over the reals, the cumulative product along `dim` has the derivative `cumprodCLM x dim` at
every `x`. -/
theorem hasFDerivAt_cumprod (x : Tensor ℝ s) (dim : Fin s.rank) :
    HasFDerivAt (fun y => cumprod y dim) (cumprodCLM x dim) x := by
  have h : ∀ c : s.Coord, HasFDerivAt (fun y : Tensor ℝ s => cumprod y dim c)
      ((coordinateCLM c).comp (cumprodCLM x dim)) x := fun c => by
    have hp := HasFDerivAt.finsetProd
      (u := Finset.Iic (Shape.Coord.get c dim))
      (g := fun i (y : Tensor ℝ s) => y (Shape.Coord.update c dim i))
      (g' := fun i => coordinateCLM (Shape.Coord.update c dim i)) (x := x)
      fun i _ => (coordinateCLM (Shape.Coord.update c dim i)).hasFDerivAt
    have hf : (fun y : Tensor ℝ s => cumprod y dim c) =
        fun y => ∏ i ∈ Finset.Iic (Shape.Coord.get c dim), y (Shape.Coord.update c dim i) :=
      funext fun y => cumprod_apply_eq_prod y dim c
    rw [hf]
    refine hp.congr_fderiv (ContinuousLinearMap.ext fun u => ?_)
    simp only [_root_.sum_apply, _root_.smul_apply, coordinateCLM_apply, smul_eq_mul,
      ContinuousLinearMap.comp_apply, cumprodCLM_apply, cumprodJVP_apply_eq_sum]
    exact Finset.sum_congr rfl fun i _ => mul_comm _ _
  exact (hasFDerivAt_of_coordinates (fun y : Tensor ℝ s => cumprod y dim) _ x h).congr_fderiv
    (ContinuousLinearMap.ext fun u => Rep.ext fun c => by rw [assemble_apply]; rfl)

/-- Over the reals, the cumulative product along `dim` has the derivative `cumprodCLM x dim` at
every `x` and the pullback `cumprodBackward x · dim`. -/
theorem hasPullbackAt_cumprod (x : Tensor ℝ s) (dim : Fin s.rank) :
    HasPullbackAt (fun y => cumprod y dim) (cumprodCLM x dim) (fun δ => cumprodBackward x δ dim)
      x :=
  ⟨hasFDerivAt_cumprod x dim, fun u δ => by rw [cumprodCLM_apply, inner_cumprodJVP]⟩

end Tensor

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}
  {f : E → Tensor ℝ s} {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E}

/-- A cumulative product along `dim`, pulled back by `cumprodBackward` at the inner value. -/
@[fun_prop]
theorem cumprod (dim : Fin s.rank) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.cumprod (f y) dim) ((Tensor.cumprodCLM (f x) dim).comp Df)
      (fun δ => bf (Tensor.cumprodBackward (f x) δ dim)) x :=
  (Tensor.hasPullbackAt_cumprod (f x) dim).comp hf

end HasPullbackAt

/-- The backward of `Op.cumprod dim` is the adjoint of the derivative of its forward at every
input, zero entries included. -/
@[op_cert TorchLean.Op.cumprod derivative]
theorem Op.cumprod_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.cumprod dim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.cumprod dim).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.cumprod dim).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  op_deriv

namespace StateSpace

variable {n : Nat}

/-! ## The states and adjoint states of one fiber -/

/-- The state of a diagonal affine scan of length `n` after its first `k` steps:
`h (k + 1) = a k * h k + b k * x k` from `h 0 = initial`, unchanged past step `n`. -/
noncomputable def stateAt (a b x : Fin n → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | k + 1 =>
    if h : k < n then a ⟨k, h⟩ * stateAt a b x initial k + b ⟨k, h⟩ * x ⟨k, h⟩
    else stateAt a b x initial k

/-- The fold of `scanStates` over its first `k` steps holds the state after them and the states
so far, newest first. -/
private theorem foldl_take_finRange (a b x : Fin n → ℝ) (initial : ℝ) {k : ℕ} (hk : k ≤ n) :
    ((List.finRange n).take k).foldl (fun (acc : ℝ × List ℝ) t =>
        (a t * acc.1 + b t * x t, (a t * acc.1 + b t * x t) :: acc.2)) (initial, []) =
      (stateAt a b x initial k,
        ((List.range k).map fun i => stateAt a b x initial (i + 1)).reverse) := by
  induction k with
  | zero => simp [stateAt]
  | succ k ih =>
    have hk' : k < n := hk
    rw [List.take_add_one, List.foldl_append, ih hk'.le,
      List.getElem?_eq_getElem (by simpa using hk')]
    simp [stateAt, hk', List.range_succ]

/-- Over the reals, the entry of `scanStates a b x initial` at a step `t` is the state after the
first `t + 1` steps. -/
theorem scanStates_getD (a b x : Fin n → ℝ) (initial : ℝ) {t : ℕ} (ht : t < n) :
    (scanStates a b x initial).getD t 0 = stateAt a b x initial (t + 1) := by
  have h := foldl_take_finRange a b x initial le_rfl
  rw [List.take_of_length_le (by simp)] at h
  rw [scanStates, h]
  simp [List.getD_eq_getElem?_getD, List.getElem?_range ht]

/-- Over the reals, the state before a step `t` of `scanStates a b x initial`, read as `initial`
before the first step, is the state after the first `t` steps. -/
theorem scanStates_getD_pred (a b x : Fin n → ℝ) (initial : ℝ) (t : Fin n) :
    (if t.val = 0 then initial else (scanStates a b x initial).getD (t.val - 1) 0) =
      stateAt a b x initial t.val := by
  split_ifs with h
  · rw [h]
    rfl
  · rw [scanStates_getD a b x initial (by omega), Nat.sub_add_cancel (Nat.pos_of_ne_zero h)]

/-- One step of the adjoint scan of `scanAdjoints a g` at the step `t`. -/
private noncomputable def adjointStep (a g : Fin n → ℝ) (t : Fin n)
    (acc : Option (Fin n) × ℝ × List ℝ) : Option (Fin n) × ℝ × List ℝ :=
  let adjoint := match acc.1 with
    | none => g t
    | some later => g t + acc.2.1 * a later
  (some t, adjoint, adjoint :: acc.2.2)

/-- The adjoint scan of `scanAdjoints a g` over the steps from `k` on. -/
private noncomputable def adjointsFrom (a g : Fin n → ℝ) (k : ℕ) :
    Option (Fin n) × ℝ × List ℝ :=
  ((List.finRange n).drop k).foldr (adjointStep a g) (none, 0, [])

/-- The adjoint states of `scanAdjoints a g` are those of the adjoint scan over every step. -/
private theorem scanAdjoints_eq (a g : Fin n → ℝ) :
    scanAdjoints a g = (adjointsFrom a g 0).2.2 := by
  rw [scanAdjoints, List.foldl_reverse]
  rfl

/-- Before the last step, the adjoint scan from `k` is the adjoint step at `k` after the adjoint
scan from `k + 1`. -/
private theorem adjointsFrom_of_lt (a g : Fin n → ℝ) {k : ℕ} (hk : k < n) :
    adjointsFrom a g k = adjointStep a g ⟨k, hk⟩ (adjointsFrom a g (k + 1)) := by
  rw [adjointsFrom, List.drop_eq_getElem_cons (by simpa using hk), List.foldr_cons,
    List.getElem_finRange]
  rfl

/-- The adjoint scan over the steps from `n` on is the empty accumulator. -/
private theorem adjointsFrom_self (a g : Fin n → ℝ) : adjointsFrom a g n = (none, 0, []) := by
  rw [adjointsFrom, List.drop_eq_nil_of_le (by simp)]
  rfl

/-- The adjoint scan from `k` records the step `k` when `k < n` and no step otherwise. -/
private theorem adjointsFrom_fst (a g : Fin n → ℝ) {k : ℕ} (hk : k ≤ n) :
    (adjointsFrom a g k).1 = if h : k < n then some ⟨k, h⟩ else none := by
  rcases Nat.lt_or_eq_of_le hk with h | rfl
  · rw [adjointsFrom_of_lt a g h, dite_eq_left h]
    rfl
  · rw [adjointsFrom_self, dite_eq_right (lt_irrefl _)]

/-- The entry `t` of the adjoint states of the scan from `k` is the first adjoint state of the
scan from `k + t`. -/
private theorem getD_adjointsFrom (a g : Fin n → ℝ) (k t : ℕ) (h : k + t ≤ n) :
    (adjointsFrom a g k).2.2.getD t 0 = (adjointsFrom a g (k + t)).2.2.getD 0 0 := by
  induction t generalizing k with
  | zero => rfl
  | succ t ih =>
    rw [adjointsFrom_of_lt a g (by omega : k < n), show k + (t + 1) = k + 1 + t by omega,
      ← ih (k + 1) (by omega)]
    simp only [adjointStep, List.getD_cons_succ]

/-- The first of the adjoint states of the scan from `k` is its current adjoint state. -/
private theorem getD_zero_adjointsFrom (a g : Fin n → ℝ) {k : ℕ} (hk : k ≤ n) :
    (adjointsFrom a g k).2.2.getD 0 0 = (adjointsFrom a g k).2.1 := by
  rcases Nat.lt_or_eq_of_le hk with h | rfl
  · rw [adjointsFrom_of_lt a g h]
    simp only [adjointStep, List.getD_cons_zero]
  · rw [adjointsFrom_self]
    rfl

/-- The entry `t` of `scanAdjoints a g` is the current adjoint state of the adjoint scan from
`t`. -/
private theorem scanAdjoints_getD_eq (a g : Fin n → ℝ) {t : ℕ} (ht : t ≤ n) :
    (scanAdjoints a g).getD t 0 = (adjointsFrom a g t).2.1 := by
  rw [scanAdjoints_eq, getD_adjointsFrom a g 0 t (by omega), Nat.zero_add,
    getD_zero_adjointsFrom a g ht]

/-- Over the reals, before the last step the adjoint states satisfy
`λ t = g t + λ (t + 1) * a (t + 1)`, reading `a` as zero past the last step. -/
theorem scanAdjoints_getD (a g : Fin n → ℝ) {t : ℕ} (ht : t < n) :
    (scanAdjoints a g).getD t 0 =
      g ⟨t, ht⟩ + (scanAdjoints a g).getD (t + 1) 0 *
        (if h : t + 1 < n then a ⟨t + 1, h⟩ else 0) := by
  rw [scanAdjoints_getD_eq a g ht.le, scanAdjoints_getD_eq a g (by omega : t + 1 ≤ n),
    adjointsFrom_of_lt a g ht]
  simp only [adjointStep]
  rw [adjointsFrom_fst a g (by omega : t + 1 ≤ n)]
  split_ifs with h
  · rfl
  · rw [mul_zero, add_zero]

/-- Summation by parts for a forward recursion `Y (t + 1) = A t * Y t + R t` against a backward
recursion `Λ t = G t + Λ (t + 1) * A (t + 1)` over the first `n` steps, with `A n = 0`. -/
theorem sum_mul_succ_eq_of_rec (A G R Y Λ : ℕ → ℝ)
    (hY : ∀ t < n, Y (t + 1) = A t * Y t + R t)
    (hΛ : ∀ t < n, Λ t = G t + Λ (t + 1) * A (t + 1)) (hA : A n = 0) :
    ∑ t ∈ Finset.range n, G t * Y (t + 1) =
      ∑ t ∈ Finset.range n, Λ t * R t + Λ 0 * A 0 * Y 0 := by
  have h : ∀ t ∈ Finset.range n, G t * Y (t + 1) =
      Λ t * R t + (Λ t * A t * Y t - Λ (t + 1) * A (t + 1) * Y (t + 1)) := fun t ht => by
    have ht' := Finset.mem_range.mp ht
    linear_combination (-(Y (t + 1))) * hΛ t ht' + Λ t * hY t ht'
  rw [Finset.sum_congr rfl h, Finset.sum_add_distrib,
    Finset.sum_range_sub' (fun t => Λ t * A t * Y t), hA]
  ring

/-- Over the reals, for a tangent `Y` of the states with `Y (t + 1) = a t * Y t + R t`, the
tangents after each step against the cotangent `g` are the adjoint states of `scanAdjoints a g`
against `R`, plus the first adjoint state times `a 0` against `Y 0`. -/
theorem sum_mul_tangent_eq (a g R : Fin n → ℝ) (Y : ℕ → ℝ)
    (hY : ∀ t : Fin n, Y (t.val + 1) = a t * Y t.val + R t) :
    ∑ t : Fin n, Y (t.val + 1) * g t =
      ∑ t : Fin n, (scanAdjoints a g).getD t.val 0 * R t +
        (if h : 0 < n then (scanAdjoints a g).getD 0 0 * a ⟨0, h⟩ else 0) * Y 0 := by
  have hsum := sum_mul_succ_eq_of_rec (n := n) (fun t => if h : t < n then a ⟨t, h⟩ else 0)
    (fun t => if h : t < n then g ⟨t, h⟩ else 0) (fun t => if h : t < n then R ⟨t, h⟩ else 0) Y
    (fun t => (scanAdjoints a g).getD t 0)
    (fun t ht => by simp only [dite_eq_left ht]; exact hY ⟨t, ht⟩)
    (fun t ht => by simp only [dite_eq_left ht]; exact scanAdjoints_getD a g ht)
    (by simp)
  have hL : ∑ t : Fin n, Y (t.val + 1) * g t =
      ∑ t ∈ Finset.range n, (if h : t < n then g ⟨t, h⟩ else 0) * Y (t + 1) := by
    rw [← Fin.sum_univ_eq_sum_range]
    refine Finset.sum_congr rfl fun t _ => ?_
    simp only [dite_eq_left t.isLt, Fin.eta]
    exact mul_comm _ _
  have hR : ∑ t : Fin n, (scanAdjoints a g).getD t.val 0 * R t =
      ∑ t ∈ Finset.range n, (scanAdjoints a g).getD t 0 * if h : t < n then R ⟨t, h⟩ else 0 := by
    rw [← Fin.sum_univ_eq_sum_range]
    refine Finset.sum_congr rfl fun t _ => ?_
    simp only [dite_eq_left t.isLt, Fin.eta]
  rw [hL, hsum, hR]
  split_ifs <;> ring

/-! ## The derivative of the states -/

variable {P : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P]

/-- The derivative at `p` of the state after `k` steps of a scan whose coefficients, inputs and
initial state are the continuous linear functions `aL`, `bL`, `xL` and `hL` of a point. -/
noncomputable def stateDeriv (aL bL xL : Fin n → P →L[ℝ] ℝ) (hL : P →L[ℝ] ℝ) (p : P) :
    ℕ → P →L[ℝ] ℝ
  | 0 => hL
  | k + 1 =>
    if h : k < n then
      stateAt (aL · p) (bL · p) (xL · p) (hL p) k • aL ⟨k, h⟩ +
        aL ⟨k, h⟩ p • stateDeriv aL bL xL hL p k + xL ⟨k, h⟩ p • bL ⟨k, h⟩ +
        bL ⟨k, h⟩ p • xL ⟨k, h⟩
    else stateDeriv aL bL xL hL p k

/-- The derivative of the initial state is `hL`. -/
@[simp] theorem stateDeriv_zero (aL bL xL : Fin n → P →L[ℝ] ℝ) (hL : P →L[ℝ] ℝ) (p : P) :
    stateDeriv aL bL xL hL p 0 = hL :=
  rfl

/-- The state after `k` steps of a scan whose coefficients, inputs and initial state are
continuous linear functions of a point has the derivative `stateDeriv` at every point. -/
theorem hasFDerivAt_stateAt (aL bL xL : Fin n → P →L[ℝ] ℝ) (hL : P →L[ℝ] ℝ) (p : P) (k : ℕ) :
    HasFDerivAt (fun q => stateAt (aL · q) (bL · q) (xL · q) (hL q) k)
      (stateDeriv aL bL xL hL p k) p := by
  induction k with
  | zero => exact hL.hasFDerivAt
  | succ k ih =>
    by_cases h : k < n
    · have hfun : (fun q => stateAt (aL · q) (bL · q) (xL · q) (hL q) (k + 1)) = fun q =>
          aL ⟨k, h⟩ q * stateAt (aL · q) (bL · q) (xL · q) (hL q) k +
            bL ⟨k, h⟩ q * xL ⟨k, h⟩ q :=
        funext fun q => by simp only [stateAt, dite_eq_left h]
      rw [hfun]
      refine (((aL ⟨k, h⟩).hasFDerivAt.mul ih).add
        ((bL ⟨k, h⟩).hasFDerivAt.mul (xL ⟨k, h⟩).hasFDerivAt)).congr_fderiv ?_
      simp only [stateDeriv, dite_eq_left h]
      abel
    · have hfun : (fun q => stateAt (aL · q) (bL · q) (xL · q) (hL q) (k + 1)) =
          fun q => stateAt (aL · q) (bL · q) (xL · q) (hL q) k :=
        funext fun q => by simp only [stateAt, dite_eq_right h]
      rw [hfun]
      simpa only [stateDeriv, dite_eq_right h] using ih

/-- Before the last step, the tangent of the states satisfies
`dh (t + 1) = a t * dh t + h t * da t + x t * db t + b t * dx t`. -/
theorem stateDeriv_succ (aL bL xL : Fin n → P →L[ℝ] ℝ) (hL : P →L[ℝ] ℝ) (p u : P) {t : ℕ}
    (ht : t < n) :
    stateDeriv aL bL xL hL p (t + 1) u =
      aL ⟨t, ht⟩ p * stateDeriv aL bL xL hL p t u +
        (stateAt (aL · p) (bL · p) (xL · p) (hL p) t * aL ⟨t, ht⟩ u +
          xL ⟨t, ht⟩ p * bL ⟨t, ht⟩ u + bL ⟨t, ht⟩ p * xL ⟨t, ht⟩ u) := by
  simp only [stateDeriv, dite_eq_left ht, _root_.add_apply, _root_.smul_apply, smul_eq_mul]
  ring

/-- Over the reals, the derivatives along `u` of the states after each step against a cotangent
`g` are the derivatives along `u` of the coefficients, the inputs and the initial state against
`λ t * h t`, `λ t * x t`, `λ t * b t` and `λ 0 * a 0`, with `λ` the adjoint states of
`scanAdjoints` and `h t` the state before step `t`. -/
theorem sum_stateDeriv_mul_eq (aL bL xL : Fin n → P →L[ℝ] ℝ) (hL : P →L[ℝ] ℝ) (p u : P)
    (g : Fin n → ℝ) :
    ∑ t : Fin n, stateDeriv aL bL xL hL p (t.val + 1) u * g t =
      ∑ t : Fin n, aL t u * ((scanAdjoints (aL · p) g).getD t.val 0 *
          stateAt (aL · p) (bL · p) (xL · p) (hL p) t.val) +
        (∑ t : Fin n, bL t u * ((scanAdjoints (aL · p) g).getD t.val 0 * xL t p) +
          (∑ t : Fin n, xL t u * ((scanAdjoints (aL · p) g).getD t.val 0 * bL t p) +
            hL u * if h : 0 < n then (scanAdjoints (aL · p) g).getD 0 0 * aL ⟨0, h⟩ p
              else 0)) := by
  have key := sum_mul_tangent_eq (aL · p) g
    (fun t => stateAt (aL · p) (bL · p) (xL · p) (hL p) t.val * aL t u + xL t p * bL t u +
      bL t p * xL t u)
    (fun k => stateDeriv aL bL xL hL p k u) fun t => stateDeriv_succ aL bL xL hL p u t.isLt
  simp only [stateDeriv_zero] at key
  rw [key, ← add_assoc, ← add_assoc, ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  congr 1
  · exact Finset.sum_congr rfl fun t _ => by ring
  · exact mul_comm _ _

end StateSpace

namespace Tensor

variable {s : Shape}

/-! ## The selective scan over the reals -/

/-- The coefficient of a real scan along `dim` at the step `t` and the other indices `rest`, as a
continuous linear functional of the coefficient tensor. -/
noncomputable def scanCoefficientCLM (dim : Fin s.rank) (shared : Bool)
    (t : Fin (s.axisSize dim.val)) (rest : Shape.Coord (s.eraseIdx dim.val)) :
    Tensor ℝ (scanCoefficients s dim shared) →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := fun a => scanCoefficient dim shared a t rest
      map_add' := fun a a' => by
        cases shared
        · exact Tensor.add_apply a a' ((Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, rest))
        · exact Tensor.add_apply a a' rest
      map_smul' := fun r a => by
        cases shared
        · exact Rep.smul_apply r a ((Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, rest))
        · exact Rep.smul_apply r a rest }

/-- `scanCoefficientCLM dim shared t rest` reads the coefficient at the step `t` and the indices
`rest`. -/
@[simp] theorem scanCoefficientCLM_apply (dim : Fin s.rank) (shared : Bool)
    (t : Fin (s.axisSize dim.val)) (rest : Shape.Coord (s.eraseIdx dim.val))
    (a : Tensor ℝ (scanCoefficients s dim shared)) :
    scanCoefficientCLM dim shared t rest a = scanCoefficient dim shared a t rest :=
  rfl

/-- Over the reals, a shared coefficient gradient is at the indices `rest` the sum of `f` over the
steps. -/
theorem scanCoefficientGrad_true_apply (dim : Fin s.rank)
    (f : Fin (s.axisSize dim.val) → Shape.Coord (s.eraseIdx dim.val) → ℝ)
    (rest : Shape.Coord (s.eraseIdx dim.val)) :
    scanCoefficientGrad dim true f rest = ∑ t, f t rest := by
  rw [scanCoefficientGrad, Rep.get_ofFn, List.finRange_foldl_add_eq_add_sum, zero_add]

/-- A per-step coefficient gradient is `f` at each step and each choice of the other indices. -/
theorem scanCoefficientGrad_false_apply {α : Type} [Storage α] [Context α] (dim : Fin s.rank)
    (f : Fin (s.axisSize dim.val) → Shape.Coord (s.eraseIdx dim.val) → α)
    (t : Fin (s.axisSize dim.val)) (rest : Shape.Coord (s.eraseIdx dim.val)) :
    scanCoefficientGrad dim false f
        ((Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, rest)) = f t rest := by
  simp only [scanCoefficientGrad, Rep.get_ofFn, Equiv.apply_symm_apply]

/-- The inner product of real tensors is the sum over the fibers along `dim` of the sums of the
products along each fiber. -/
private theorem inner_eq_sum_axisEquiv (dim : Fin s.rank) (u v : Tensor ℝ s) :
    ⟪u, v⟫ = ∑ rest, ∑ t, u ((Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, rest)) *
      v ((Shape.Coord.axisEquiv dim.val (shape := s)).symm (t, rest)) := by
  rw [inner_eq_sum, ← (Shape.Coord.axisEquiv dim.val (shape := s)).symm.sum_comp,
    Fintype.sum_prod_type_right]

/-- Over the reals, the inner product of a coefficient tangent with the coefficient gradient of
`f` is the sum over the steps and the other indices of the coefficient times `f`. -/
theorem inner_scanCoefficientGrad (dim : Fin s.rank) (shared : Bool)
    (da : Tensor ℝ (scanCoefficients s dim shared))
    (f : Fin (s.axisSize dim.val) → Shape.Coord (s.eraseIdx dim.val) → ℝ) :
    ⟪da, scanCoefficientGrad dim shared f⟫ =
      ∑ rest, ∑ t, scanCoefficient dim shared da t rest * f t rest := by
  cases shared
  · refine (inner_eq_sum_axisEquiv dim da (scanCoefficientGrad dim false f)).trans
      (Finset.sum_congr rfl fun rest _ => Finset.sum_congr rfl fun t _ => ?_)
    exact congrArg (HMul.hMul _) (scanCoefficientGrad_false_apply dim f t rest)
  · rw [inner_eq_sum]
    refine Finset.sum_congr rfl fun rest _ => ?_
    rw [scanCoefficientGrad_true_apply, Finset.mul_sum]
    rfl

/-- Over the reals, an entry of `selectiveScan` is the state of its fiber along `dim` after the
steps up to its own. -/
theorem selectiveScan_apply_eq_stateAt (dim : Fin s.rank) (sharedA sharedB : Bool)
    (a : Tensor ℝ (scanCoefficients s dim sharedA))
    (b : Tensor ℝ (scanCoefficients s dim sharedB)) (x : Tensor ℝ s)
    (initial : Tensor ℝ (s.eraseIdx dim.val)) (c : s.Coord) :
    selectiveScan dim sharedA sharedB a b x initial c =
      StateSpace.stateAt
        (scanCoefficient dim sharedA a · (Shape.Coord.axisEquiv dim.val (shape := s) c).2)
        (scanCoefficient dim sharedB b · (Shape.Coord.axisEquiv dim.val (shape := s) c).2)
        (fun t => x ((Shape.Coord.axisEquiv dim.val (shape := s)).symm
          (t, (Shape.Coord.axisEquiv dim.val (shape := s) c).2)))
        (initial (Shape.Coord.axisEquiv dim.val (shape := s) c).2)
        ((Shape.Coord.axisEquiv dim.val (shape := s) c).1.val + 1) := by
  rw [selectiveScan, Rep.get_ofFn]
  exact StateSpace.scanStates_getD _ _ _ _ (Shape.Coord.axisEquiv dim.val (shape := s) c).1.isLt

/-- The inputs of a real selective scan along `dim` on their nested L2 product: the two
coefficients, the input and the initial state. -/
abbrev ScanInputs (s : Shape) (dim : Fin s.rank) (sharedA sharedB : Bool) : Type :=
  WithLp 2 (Tensor ℝ (scanCoefficients s dim sharedA) ×
    WithLp 2 (Tensor ℝ (scanCoefficients s dim sharedB) ×
      WithLp 2 (Tensor ℝ s × Tensor ℝ (s.eraseIdx dim.val))))

/-- The derivative at `p` of the real selective scan along `dim` as a map of its inputs: its
entry at `c` is the derivative `StateSpace.stateDeriv` of the state of the fiber of `c`. -/
noncomputable def selectiveScanCLM (dim : Fin s.rank) (sharedA sharedB : Bool)
    (p : ScanInputs s dim sharedA sharedB) : ScanInputs s dim sharedA sharedB →L[ℝ] Tensor ℝ s :=
  assemble fun c =>
    StateSpace.stateDeriv
      (fun t => (scanCoefficientCLM dim sharedA t
        (Shape.Coord.axisEquiv dim.val (shape := s) c).2).comp (WithLp.fstL 2 ℝ _ _))
      (fun t => (scanCoefficientCLM dim sharedB t
        (Shape.Coord.axisEquiv dim.val (shape := s) c).2).comp
          ((WithLp.fstL 2 ℝ _ _).comp (WithLp.sndL 2 ℝ _ _)))
      (fun t => (coordinateCLM ((Shape.Coord.axisEquiv dim.val (shape := s)).symm
          (t, (Shape.Coord.axisEquiv dim.val (shape := s) c).2))).comp
        ((WithLp.fstL 2 ℝ _ _).comp ((WithLp.sndL 2 ℝ _ _).comp (WithLp.sndL 2 ℝ _ _))))
      ((coordinateCLM (Shape.Coord.axisEquiv dim.val (shape := s) c).2).comp
        ((WithLp.sndL 2 ℝ _ _).comp ((WithLp.sndL 2 ℝ _ _).comp (WithLp.sndL 2 ℝ _ _))))
      p ((Shape.Coord.axisEquiv dim.val (shape := s) c).1.val + 1)

/-- Over the reals, the selective scan along `dim` as a map of its four inputs on their nested L2
product has the derivative `selectiveScanCLM` and the pullback given by the four gradients of
`selectiveScanBackward` at every input. -/
theorem hasPullbackAt_selectiveScan (dim : Fin s.rank) (sharedA sharedB : Bool)
    (a : Tensor ℝ (scanCoefficients s dim sharedA))
    (b : Tensor ℝ (scanCoefficients s dim sharedB)) (x : Tensor ℝ s)
    (initial : Tensor ℝ (s.eraseIdx dim.val)) :
    HasPullbackAt (fun p : ScanInputs s dim sharedA sharedB =>
        selectiveScan dim sharedA sharedB p.fst p.snd.fst p.snd.snd.fst p.snd.snd.snd)
      (selectiveScanCLM dim sharedA sharedB
        (WithLp.toLp 2 (a, WithLp.toLp 2 (b, WithLp.toLp 2 (x, initial)))))
      (fun δ => WithLp.toLp 2 ((selectiveScanBackward dim sharedA sharedB a b x initial δ).1,
        WithLp.toLp 2 ((selectiveScanBackward dim sharedA sharedB a b x initial δ).2.1,
          WithLp.toLp 2 ((selectiveScanBackward dim sharedA sharedB a b x initial δ).2.2.1,
            (selectiveScanBackward dim sharedA sharedB a b x initial δ).2.2.2))))
      (WithLp.toLp 2 (a, WithLp.toLp 2 (b, WithLp.toLp 2 (x, initial)))) := by
  refine ⟨hasFDerivAt_of_coordinates _ _ _ fun c => ?_, fun u δ => ?_⟩
  · refine (StateSpace.hasFDerivAt_stateAt _ _ _ _ _ _).congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun q => ?_)
    exact selectiveScan_apply_eq_stateAt dim sharedA sharedB _ _ _ _ c
  · simp only [selectiveScanCLM, selectiveScanBackward, WithLp.prod_inner_apply]
    rw [inner_eq_sum_axisEquiv dim, inner_eq_sum_axisEquiv dim, inner_scanCoefficientGrad,
      inner_scanCoefficientGrad, inner_eq_sum]
    simp only [assemble_apply, Rep.get_ofFn, Equiv.apply_symm_apply,
      StateSpace.scanStates_getD_pred, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun rest _ => ?_
    exact StateSpace.sum_stateDeriv_mul_eq _ _ _ _ _ u _

end Tensor

namespace HasPullbackAt

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}

/-- A selective scan along `dim` of four maps into its two coefficients, its input and its initial
state, pulled back by the four gradients of `selectiveScanBackward` at their values. -/
@[fun_prop]
theorem selectiveScan (dim : Fin s.rank) (sharedA sharedB : Bool)
    {fa : E → Tensor ℝ (Tensor.scanCoefficients s dim sharedA)}
    {fb : E → Tensor ℝ (Tensor.scanCoefficients s dim sharedB)} {fx : E → Tensor ℝ s}
    {fi : E → Tensor ℝ (s.eraseIdx dim.val)}
    {Da : E →L[ℝ] Tensor ℝ (Tensor.scanCoefficients s dim sharedA)}
    {Db : E →L[ℝ] Tensor ℝ (Tensor.scanCoefficients s dim sharedB)} {Dx : E →L[ℝ] Tensor ℝ s}
    {Di : E →L[ℝ] Tensor ℝ (s.eraseIdx dim.val)}
    {ba : Tensor ℝ (Tensor.scanCoefficients s dim sharedA) → E}
    {bb : Tensor ℝ (Tensor.scanCoefficients s dim sharedB) → E} {bx : Tensor ℝ s → E}
    {bi : Tensor ℝ (s.eraseIdx dim.val) → E}
    (ha : HasPullbackAt fa Da ba x) (hb : HasPullbackAt fb Db bb x)
    (hx : HasPullbackAt fx Dx bx x) (hi : HasPullbackAt fi Di bi x) :
    HasPullbackAt
      (fun y => Tensor.selectiveScan dim sharedA sharedB (fa y) (fb y) (fx y) (fi y))
      ((Tensor.selectiveScanCLM dim sharedA sharedB
          (WithLp.toLp 2 (fa x, WithLp.toLp 2 (fb x, WithLp.toLp 2 (fx x, fi x))))).comp
        ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
          (Da.prod ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
            (Db.prod ((WithLp.prodContinuousLinearEquiv 2 ℝ _ _).symm.toContinuousLinearMap.comp
              (Dx.prod Di)))))))
      (fun δ =>
        ba (Tensor.selectiveScanBackward dim sharedA sharedB (fa x) (fb x) (fx x) (fi x) δ).1 +
          (bb (Tensor.selectiveScanBackward dim sharedA sharedB (fa x) (fb x) (fx x) (fi x)
              δ).2.1 +
            (bx (Tensor.selectiveScanBackward dim sharedA sharedB (fa x) (fb x) (fx x) (fi x)
                δ).2.2.1 +
              bi (Tensor.selectiveScanBackward dim sharedA sharedB (fa x) (fb x) (fx x) (fi x)
                δ).2.2.2)))
      x := by
  have h := HasPullbackAt.comp (x := x)
    (f := fun y => WithLp.toLp 2 (fa y, WithLp.toLp 2 (fb y, WithLp.toLp 2 (fx y, fi y))))
    (Tensor.hasPullbackAt_selectiveScan dim sharedA sharedB (fa x) (fb x) (fx x) (fi x))
    (toLp_prod ha (toLp_prod hb (toLp_prod hx hi)))
  exact h

end HasPullbackAt

/-- Over the reals, the forward of `Op.selectiveScan` is differentiable in its two coefficients,
its input and its initial state on their nested L2 product at every input, and its backward is
the adjoint of that derivative. -/
@[op_cert TorchLean.Op.selectiveScan derivative 0 1 2 3]
theorem Op.selectiveScan_prod_backward_eq_adjoint {s : Shape} (dim : Fin s.rank)
    (sharedA sharedB : Bool) (a : Tensor ℝ (Tensor.scanCoefficients s dim sharedA))
    (b : Tensor ℝ (Tensor.scanCoefficients s dim sharedB)) (x : Tensor ℝ s)
    (initial : Tensor ℝ (s.eraseIdx dim.val)) (δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ (Tensor.scanCoefficients s dim sharedA) ×
          WithLp 2 (Tensor ℝ (Tensor.scanCoefficients s dim sharedB) ×
            WithLp 2 (Tensor ℝ s × Tensor ℝ (s.eraseIdx dim.val)))) =>
        (((Op.selectiveScan dim sharedA sharedB).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd.fst, p.snd.snd.snd, ())).1)
      (WithLp.toLp 2 (a, WithLp.toLp 2 (b, WithLp.toLp 2 (x, initial)))) ∧
    WithLp.toLp 2
        ((((Op.selectiveScan dim sharedA sharedB).rule? (α := ℝ) |>.get rfl).backward
          (a, b, x, initial, ()) (δ, ())).1,
        WithLp.toLp 2
          ((((Op.selectiveScan dim sharedA sharedB).rule? (α := ℝ) |>.get rfl).backward
            (a, b, x, initial, ()) (δ, ())).2.1,
          WithLp.toLp 2
            ((((Op.selectiveScan dim sharedA sharedB).rule? (α := ℝ) |>.get rfl).backward
              (a, b, x, initial, ()) (δ, ())).2.2.1,
            (((Op.selectiveScan dim sharedA sharedB).rule? (α := ℝ) |>.get rfl).backward
              (a, b, x, initial, ()) (δ, ())).2.2.2.1))) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ (Tensor.scanCoefficients s dim sharedA) ×
          WithLp 2 (Tensor ℝ (Tensor.scanCoefficients s dim sharedB) ×
            WithLp 2 (Tensor ℝ s × Tensor ℝ (s.eraseIdx dim.val)))) =>
        (((Op.selectiveScan dim sharedA sharedB).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd.fst, p.snd.snd.fst, p.snd.snd.snd, ())).1)
        (WithLp.toLp 2 (a, WithLp.toLp 2 (b, WithLp.toLp 2 (x, initial))))).adjoint δ := by
  op_deriv

end TorchLean
