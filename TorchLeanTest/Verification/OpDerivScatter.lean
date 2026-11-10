/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Op.Scatter
public import TorchLean.NN.Tactic.OpDeriv

/-!
# `op_deriv` through the scatters, and the certificates of `Op.scatter`

The first theorems are composites that `op_deriv` proves through the `fun_prop` rules of the sum
scatter and of the scatter maximum, with the backward pass written as the composed pullback. The
examples after them instantiate the derivative certificate of `Op.scatter` for every rule and its
`convention` certificate for the maximum and the minimum, on `[3]` along its axis at index tensors
of shape `[2]`, and apply the derivative certificate of the maximum and the minimum wherever the
forward is differentiable.
-/

@[expose] public section

namespace TorchLean.Test.OpDerivScatter

open TorchLean Tensor

variable {s t : Shape}

/-- The sum of a sum scatter of `x * x` with fixed updates has the backward pass `F * x + F * x`,
for `F` the filled cotangent. -/
theorem sumScatterSum_backward (dims : List (Fin s.rank)) (h : dims.Nodup)
    (index : Tensor (Port.IndexTuple (dims.map s.extent)) t)
    (u : Tensor ℝ (t ++ s.eraseAxes dims)) (x : Tensor ℝ s) (δ : ℝ) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.sum (Tensor.scatter .sum (y * y) dims
        (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) u)) x ∧
      Tensor.full s δ * x + Tensor.full s δ * x =
        (fderiv ℝ (fun y : Tensor ℝ s => Tensor.sum (Tensor.scatter .sum (y * y) dims
          (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) u)) x).adjoint δ := by
  op_deriv

/-- The sum of a sum scatter of `x * x` with the updates `u`, jointly in `x` and `u`: the backward
pass is `F * x + F * x` on the base and the read of `F` where each update lands on the updates,
for `F` the filled cotangent. -/
theorem sumScatterSum_prod_backward (dims : List (Fin s.rank)) (h : dims.Nodup)
    (index : Tensor (Port.IndexTuple (dims.map s.extent)) t) (x : Tensor ℝ s)
    (u : Tensor ℝ (t ++ s.eraseAxes dims)) (δ : ℝ) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) =>
        Tensor.sum (Tensor.scatter .sum (p.fst * p.fst) dims
          (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) p.snd))
        (WithLp.toLp 2 (x, u)) ∧
      WithLp.toLp 2 (Tensor.full s δ * x + Tensor.full s δ * x,
          scatterUpdatesBackward .sum dims h (Tensor.map (Op.axisTuples _ dims) index)
            (Tensor.full s δ)) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ (t ++ s.eraseAxes dims)) =>
          Tensor.sum (Tensor.scatter .sum (p.fst * p.fst) dims
            (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) p.snd))
          (WithLp.toLp 2 (x, u))).adjoint δ := by
  op_deriv

/-- A scatter maximum of `x * x` with fixed updates is differentiable in `x` where every entry has
a unique largest participant. -/
theorem scatterMax_differentiableAt (dims : List (Fin s.rank)) (h : dims.Nodup)
    (index : Tensor (Port.IndexTuple (dims.map s.extent)) t)
    (u : Tensor ℝ (t ++ s.eraseAxes dims)) (x : Tensor ℝ s)
    (hx : ∀ c,
      (∀ v, scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v = c →
        u v < (x * x) c) ∨
      ∃ v, scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v = c ∧
        (x * x) c < u v ∧
        ∀ v', scatterTarget dims h (Tensor.map (Op.axisTuples _ dims) index) v' = c → v' ≠ v →
          u v' < u v) :
    DifferentiableAt ℝ (fun y : Tensor ℝ s => Tensor.scatter .max (y * y) dims
      (Tensor.map (fun i => some (Op.axisTuples _ dims i)) index) u) x := by
  op_deriv

section Concrete

variable (base : Tensor ℝ [3])
  (index : Tensor (Port.IndexTuple ([(0 : Fin 1)].map (Shape.extent [3]))) [2])
  (updates : Tensor ℝ ([2] ++ Shape.eraseAxes [3] [(0 : Fin 1)])) (δ : Tensor ℝ [3])

/-- The joint scatter certificate for a sum on `[3]` along its axis at index tensors of shape
`[2]`. -/
example := Op.scatter_prod_backward_eq_adjoint (s := [3]) (t := [2]) .sum [(0 : Fin 1)]
  (List.nodup_singleton _) base index updates nofun nofun δ

/-- The joint scatter certificate for a replacement on `[3]` along its axis at index tensors of
shape `[2]`. -/
example := Op.scatter_prod_backward_eq_adjoint (s := [3]) (t := [2]) .replace [(0 : Fin 1)]
  (List.nodup_singleton _) base index updates nofun nofun δ

/-- The joint scatter certificate for a product on `[3]` along its axis at index tensors of shape
`[2]`, at the zero base and zero updates. -/
example := Op.scatter_prod_backward_eq_adjoint (s := [3]) (t := [2]) .prod [(0 : Fin 1)]
  (List.nodup_singleton _) 0 index 0 nofun nofun δ

/-- The joint scatter certificate for a maximum on `[3]` along its axis at index tensors of shape
`[2]`, at updates below every base entry. -/
example (hu : ∀ v c, updates v < base c) :=
  Op.scatter_prod_backward_eq_adjoint (s := [3]) (t := [2]) .max [(0 : Fin 1)]
    (List.nodup_singleton _) base index updates (fun _ c => Or.inl fun v _ => hu v c) nofun δ

/-- The joint scatter certificate for a minimum on `[3]` along its axis at index tensors of shape
`[2]`, at updates above every base entry. -/
example (hu : ∀ v c, base c < updates v) :=
  Op.scatter_prod_backward_eq_adjoint (s := [3]) (t := [2]) .min [(0 : Fin 1)]
    (List.nodup_singleton _) base index updates nofun (fun _ c => Or.inl fun v _ => hu v c) δ

/-- The convention certificate for a maximum on `[3]` along its axis at index tensors of shape
`[2]`. -/
example := (Op.scatter_backward_eq (s := [3]) (t := [2]) .max [(0 : Fin 1)]
  (List.nodup_singleton _) base index updates δ).1 rfl

/-- The convention certificate for a minimum on `[3]` along its axis at index tensors of shape
`[2]`. -/
example := (Op.scatter_backward_eq (s := [3]) (t := [2]) .min [(0 : Fin 1)]
  (List.nodup_singleton _) base index updates δ).2 rfl

/-- We apply the joint scatter certificate for a maximum on `[3]` along its axis at index tensors
of shape `[2]` at every input where the forward is differentiable, so that its hypothesis asks
for no more than differentiability. -/
example (hd : DifferentiableAt ℝ
    (fun p : WithLp 2 (Tensor ℝ [3] × Tensor ℝ ([2] ++ Shape.eraseAxes [3] [(0 : Fin 1)])) =>
      (((Op.scatter (s := [3]) (t := [2]) .max [(0 : Fin 1)] (List.nodup_singleton _)).rule?
        (α := ℝ) |>.get rfl).forward (p.fst, index, p.snd, ())).1)
    (WithLp.toLp 2 (base, updates))) :=
  have hP (j : Shape.Coord [2]) :
      Tensor.map (fun i => some (Op.axisTuples _ [(0 : Fin 1)] i)) index j =
        some (Tensor.map (Op.axisTuples _ [(0 : Fin 1)]) index j) :=
    (Tensor.map_apply _ index j).trans (congrArg some (Tensor.map_apply _ index j).symm)
  Op.scatter_prod_backward_eq_adjoint (s := [3]) (t := [2]) .max [(0 : Fin 1)]
    (List.nodup_singleton _) base index updates
    (fun _ => (differentiableAt_scatter_max_iff _ _ _ hP _).mp hd) nofun δ

/-- We apply the joint scatter certificate for a minimum on `[3]` along its axis at index tensors
of shape `[2]` at every input where the forward is differentiable, so that its hypothesis asks
for no more than differentiability. -/
example (hd : DifferentiableAt ℝ
    (fun p : WithLp 2 (Tensor ℝ [3] × Tensor ℝ ([2] ++ Shape.eraseAxes [3] [(0 : Fin 1)])) =>
      (((Op.scatter (s := [3]) (t := [2]) .min [(0 : Fin 1)] (List.nodup_singleton _)).rule?
        (α := ℝ) |>.get rfl).forward (p.fst, index, p.snd, ())).1)
    (WithLp.toLp 2 (base, updates))) :=
  have hP (j : Shape.Coord [2]) :
      Tensor.map (fun i => some (Op.axisTuples _ [(0 : Fin 1)] i)) index j =
        some (Tensor.map (Op.axisTuples _ [(0 : Fin 1)]) index j) :=
    (Tensor.map_apply _ index j).trans (congrArg some (Tensor.map_apply _ index j).symm)
  Op.scatter_prod_backward_eq_adjoint (s := [3]) (t := [2]) .min [(0 : Fin 1)]
    (List.nodup_singleton _) base index updates nofun
    (fun _ => (differentiableAt_scatter_min_iff _ _ _ hP _).mp hd) δ

end Concrete

end TorchLean.Test.OpDerivScatter
