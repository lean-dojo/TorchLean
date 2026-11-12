/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Backward
public import TorchLean.Tensor.Rescale

/-!
# Variance over any axes

`twoPassVar x dims keepdim correction` is the variance of each fiber of `x` over `dims`, in two
passes (`fiberVar`): the mean is the row-major sum divided once by the number of entries, and the
sum of squared deviations is divided once by that number less `correction`
(`TorchLean.Context.divNat`). An empty fiber divides by one. With `correction = 0` this is the
population variance.

`var` is `twoPassVar` wherever that is finite (`var_eq_twoPassVar`), so it is finite wherever
`twoPassVar` is (`var_finite`). Where `twoPassVar` is not finite, `var` takes `fiberVar` of the
fiber's entries divided by `2 ^ k` and divides it twice by `1 / 2 ^ k`, each quotient rounded once,
for the least `k` from `1` to `Nat.log2 n + 2` that gives a finite value
(`TorchLean.Context.rescale`), and keeps the two-pass value when none does. In binary16 the
deviations of `[181, -1447/8]` from its mean round to `±181` and their squares to `32768`, whose sum
`65536` overflows, so `twoPassVar` is `+∞`. The variance is about `32738.4`, and `var` is `32768`,
from the fiber halved.

Written by hand, the two passes take the divisor of `torch.var` (PyTorch 2.8 docs),
`max(0, N - correction)`. The default differs: ours is `0`, the biased estimator that the
`BatchNorm1d` docs name for the batch statistics (the batch norm of `TorchLean.NN.Layer.Norm`
calls `var` with it), and `torch.var`'s is `1`. An empty fiber gives `0` here and NaN in torch.
On `Float` the small counts agree with torch: `[5]` at `correction = 1` is `0 / 0`, a NaN, and
`[1, 3]` at `correction = 2` is `+∞`. No test compares `var` with torch, since the importer
lowers `torch.var` through other operators.

`twoPassVar_apply` reads an entry through the two folds, and `var_apply` reads an entry of `var`
as `Context.rescale` of the entry of `twoPassVar`. `castShape_twoPassVar_singleton_apply` reads
the two-pass variance over one axis, cast to the shape without that axis, as the two-pass variance
of the vector along it.
-/

@[expose] public section

namespace TorchLean

/-- Under `NonemptyAxis axis s`, the extent of `s` along `axis` is positive. -/
theorem Shape.NonemptyAxis.extent_pos {axis : Nat} {s : Shape} (h : NonemptyAxis axis s)
    (hb : axis < s.rank) : 0 < s.extent ⟨axis, hb⟩ := by
  induction h with
  | zero => exact Nat.succ_pos _
  | succ _ ih => exact ih (Nat.lt_of_succ_lt_succ hb)

namespace Tensor

variable {α : Type} [Storage α] {s : Shape}

omit [Storage α] in
/-- The two-pass variance of `value`: the sum of squared deviations from the mean divided once by
`n - correction` (in `Nat`, so zero when `n ≤ correction`), or by one when `n = 0`. The mean
divides the sum once by `n`, or by one when `n = 0`. Both sums fold in index order. -/
def fiberVar [Context α] {n : Nat} (value : Fin n → α) (correction : Nat) : α :=
  let mean := Context.divNat ((List.finRange n).foldl (fun acc i => acc + value i) 0)
    (if n = 0 then 1 else n)
  Context.divNat ((List.finRange n).foldl (fun acc i =>
      let difference := value i - mean
      acc + difference * difference) 0)
    (if n = 0 then 1 else n - correction)

/-- The two-pass variance (`fiberVar`) of each fiber of `x` over the axes `dims`, the fiber's
entries in row-major order. -/
def twoPassVar [Context α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) (correction : Nat := 0) :
    Tensor α (s.reduce dims keepdim) :=
  Rep.ofFn fun o => fiberVar (fun i : Fin (Shape.size (s.selectAxes dims)) =>
    x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))) correction

/-- An entry of `twoPassVar x dims keepdim correction` is the sum of squared deviations of its
fiber from the fiber's mean, divided by the corrected count. -/
theorem twoPassVar_apply [Context α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) (correction : Nat)
    (o : (s.reduce dims keepdim).Coord) :
    twoPassVar x dims keepdim correction o =
      let n := Shape.size (s.selectAxes dims)
      let value := fun i : Fin n =>
        x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))
      let mean := Context.divNat ((List.finRange n).foldl (fun acc i => acc + value i) 0)
        (if n = 0 then 1 else n)
      Context.divNat ((List.finRange n).foldl (fun acc i =>
          let difference := value i - mean
          acc + difference * difference) 0)
        (if n = 0 then 1 else n - correction) :=
  Rep.get_ofFn _ o

/-- The variance of `x` over the axes `dims`: per fiber of `n` entries, the two-pass variance
`fiberVar` where it is finite. Where it is not, the first finite `fiberVar y correction / c / c`
for `k = 1, ..., Nat.log2 n + 2`, where `y` is the entries divided by `2 ^ k` and `c = 1 / 2 ^ k`
(`Context.rescale`), or the two-pass value when none is finite. -/
def var [Context α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) (correction : Nat := 0) :
    Tensor α (s.reduce dims keepdim) :=
  Rep.ofFn fun o =>
    let value := fun i : Fin (Shape.size (s.selectAxes dims)) =>
      x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))
    Context.rescale (fiberVar value correction) value fun y c => fiberVar y correction / c / c

/-- An entry of `var x dims keepdim correction` is `Context.rescale` of the entry of `twoPassVar`,
with the entries of its fiber and the two-pass variance of the scaled entries. -/
theorem var_apply [Context α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) (correction : Nat)
    (o : (s.reduce dims keepdim).Coord) :
    var x dims keepdim correction o =
      Context.rescale (twoPassVar x dims keepdim correction o)
        (fun i : Fin (Shape.size (s.selectAxes dims)) =>
          x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)))
        fun y c => fiberVar y correction / c / c := by
  simp only [var, twoPassVar, Rep.get_ofFn]

/-- An entry of `var` is the entry of `twoPassVar` when that is finite. -/
theorem var_eq_twoPassVar [Context α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) (correction : Nat)
    (o : (s.reduce dims keepdim).Coord)
    (h : Context.isFinite (twoPassVar x dims keepdim correction o) = true) :
    var x dims keepdim correction o = twoPassVar x dims keepdim correction o := by
  rw [var_apply]
  exact Context.rescale_eq_self _ _ h

/-- An entry of `var` is finite where the entry of `twoPassVar` is. -/
theorem var_finite [Context α] (x : Tensor α s)
    (dims : List (Fin s.rank)) (keepdim : Bool) (correction : Nat)
    (o : (s.reduce dims keepdim).Coord)
    (h : Context.isFinite (twoPassVar x dims keepdim correction o) = true) :
    Context.isFinite (var x dims keepdim correction o) = true := by
  rw [var_eq_twoPassVar x dims keepdim correction o h]
  exact h

/-- An entry of the two-pass population variance over one axis, read through a cast to the shape
without that axis, is the two-pass variance of the vector along the axis. The sum of the vector and
the sum of its squared deviations from the mean are each divided by the length, or by one when the
axis is empty. -/
theorem castShape_twoPassVar_singleton_apply [Context α]
    (x : Tensor α s) (dim : Fin s.rank)
    (q : s.reduce [dim] false = s.eraseIdx dim) (c : Shape.Coord (s.eraseIdx dim)) :
    castShape (twoPassVar x [dim] false) q c =
      let value := fun j : Fin (s.axisSize dim) =>
        x ((Shape.Coord.axisEquiv dim (shape := s)).symm (j, c))
      let count := if s.axisSize dim = 0 then 1 else s.axisSize dim
      let mean := Context.divNat (sum (ofFn value)) count
      Context.divNat (sum (ofFn fun j => (value j - mean) * (value j - mean))) count := by
  have hshape : ([s.axisSize dim] : Shape) = s.selectAxes [dim] := by
    rw [Shape.selectAxes_singleton, ← Shape.axisSize_eq_extent]
  have hsize : Shape.size (s.selectAxes [dim]) = s.axisSize dim := by
    rw [← hshape, Shape.size_eq_prod, List.prod_singleton]
  have hindex : ∀ {t u : Shape} (e : t = u) (c : u.Coord) (k : Nat),
      Shape.Coord.index (_root_.cast (congrArg Shape.Coord e.symm) c) k =
        Shape.Coord.index c k := by
    intro t u e c k
    subst e
    rfl
  have hsrc (r : Shape.Coord [s.axisSize dim]) (r' : (s.selectAxes [dim]).Coord)
      (hrr : ∀ k, Shape.Coord.index r k = Shape.Coord.index r' k) :
      (Shape.Coord.axisEquiv dim (shape := s)).symm (r.1, c) =
        s.reduceSource [dim] false (_root_.cast (congrArg Shape.Coord q.symm) c)
          r' := by
    apply Shape.Coord.ext_index
    intro k hk
    rw [Shape.index_reduceSource, dite_eq_left hk, Shape.Coord.index_axisEquiv_symm]
    by_cases hka : k = dim
    · have hd : (⟨k, hk⟩ : Fin s.rank) = dim := Fin.ext hka
      rw [ite_eq_left hka, hd, ite_eq_left (List.mem_singleton_self _), Shape.sortedAxes_singleton,
        List.idxOf_cons_self]
      exact hrr 0
    · have hd : (⟨k, hk⟩ : Fin s.rank) ≠ dim := fun e => hka (congrArg Fin.val e)
      rw [ite_eq_right hka, ite_eq_right (mt List.mem_singleton.mp hd), Shape.keptIndex,
        Bool.cond_false, Shape.idxOf_otherAxes_singleton s hd, hindex q]
      simp only [Fin.lt_def]
      split_ifs <;> rfl
  have hsum (f : α → α) :
      sum (ofFn fun j : Fin (s.axisSize dim) =>
          f (x ((Shape.Coord.axisEquiv dim (shape := s)).symm (j, c)))) =
        (List.finRange (Shape.size (s.selectAxes [dim]))).foldl
          (fun acc i => acc + f (x (s.reduceSource [dim] false
            (_root_.cast (congrArg Shape.Coord q.symm) c) (Shape.Coord.unlinearize i)))) 0 := by
    rw [sum_eq_foldl, Rep.foldl_eq_fin_foldl, Fin.foldl_eq_finRange_foldl]
    simp only [Rep.getFlat_eq_get, ofFn, Rep.get_ofFn]
    refine foldl_finRange_congr hshape
      (fun a (r : Shape.Coord [s.axisSize dim]) =>
        a + f (x ((Shape.Coord.axisEquiv dim (shape := s)).symm (r.1, c))))
      (fun a r' => a + f (x (s.reduceSource [dim] false
        (_root_.cast (congrArg Shape.Coord q.symm) c) r'))) ?_ 0
    intro a r r' hrr
    show a + f (x _) = a + f (x _)
    rw [hsrc r r' hrr]
  have hden : (if Shape.size (s.selectAxes [dim]) = 0 then 1
      else Shape.size (s.selectAxes [dim])) =
      (if s.axisSize dim = 0 then 1 else s.axisSize dim) := by
    rw [hsize]
  rw [castShape_apply, twoPassVar_apply]
  dsimp only
  rw [Nat.sub_zero, hden, ← hsum (fun v => v)]
  congr 1
  exact (hsum fun v => (v - Context.divNat (sum (ofFn fun j : Fin (s.axisSize dim) =>
      x ((Shape.Coord.axisEquiv dim (shape := s)).symm (j, c))))
        (if s.axisSize dim = 0 then 1 else s.axisSize dim)) *
    (v - Context.divNat (sum (ofFn fun j : Fin (s.axisSize dim) =>
      x ((Shape.Coord.axisEquiv dim (shape := s)).symm (j, c))))
        (if s.axisSize dim = 0 then 1 else s.axisSize dim))).symm

end Tensor

end TorchLean
