/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.FDeriv.Fold
public import NN.Proofs.Autograd.FDeriv.Reindex
public import NN.Spec.Core.Context.Real

/-!
# Coordinate derivatives on real tensors

The tensor representation already carries its Euclidean norm. Reading its coordinates is a
continuous linear equivalence with a finite function space, so coordinatewise derivative proofs
give Fréchet derivatives on the tensor itself. In particular, a matrix proof can keep its row and
column indices instead of introducing a second flattened layout.

The coordinate function space is only used to assemble the derivative. The result has the native
tensor type and the Euclidean topology used by the tensor inner-product and adjoint theorems.
-/

@[expose] public section

namespace Proofs.Autograd.TensorCoordinates

open Spec TorchLean
open TorchLean.Tensor
open scoped ContDiff

noncomputable section

/-- The native coordinate view, with continuity supplied by finite dimensionality. -/
def coordinateEquiv (shape : Shape) : Tensor ℝ shape ≃L[ℝ] (shape.Coord → ℝ) :=
  (Tensor.toEuclidean shape.toList).toContinuousLinearEquiv.trans
    (EuclideanSpace.equiv (𝕜 := ℝ) (ι := shape.Coord))

/-- Reading the coordinate view performs the same lookup as the tensor representation. -/
@[simp] theorem coordinateEquiv_apply {shape : Shape}
    (x : Tensor ℝ shape) (i : shape.Coord) :
    coordinateEquiv shape x i = x i := rfl

/-- Rebuilding a tensor preserves each supplied coordinate. -/
@[simp] theorem coordinateEquiv_symm_apply {shape : Shape}
    (x : shape.Coord → ℝ) (i : shape.Coord) :
    (coordinateEquiv shape).symm x i = x i := by
  exact Internal.Rep.get_ofFn _ i

/-- A single tensor coordinate is a continuous linear functional. -/
def coordinateCLM {shape : Shape} (i : shape.Coord) : Tensor ℝ shape →L[ℝ] ℝ :=
  (ContinuousLinearMap.proj i).comp (coordinateEquiv shape).toContinuousLinearMap

/-- The coordinate functional uses the native lookup. -/
@[simp] theorem coordinateCLM_apply {shape : Shape}
    (i : shape.Coord) (x : Tensor ℝ shape) :
    coordinateCLM i x = x i := rfl

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Local smoothness of a tensor is equivalent to local smoothness of every coordinate. -/
theorem contDiffAt_iff_coordinates {shape : Shape} {n : ℕ∞ω}
    (f : E → Tensor ℝ shape) (x : E) :
    ContDiffAt ℝ n f x ↔ ∀ i, ContDiffAt ℝ n (fun y => f y i) x := by
  constructor
  · intro hf i
    simpa only [Function.comp_def, coordinateCLM_apply] using
      (coordinateCLM i).contDiff.contDiffAt.comp x hf
  · intro h
    have hpi := (contDiffAt_pi (𝕜 := ℝ)).2 h
    have hcomp := (coordinateEquiv shape).symm.contDiff.contDiffAt.comp x hpi
    have hfun : (coordinateEquiv shape).symm ∘ (fun y i => f y i) = f := by
      funext y
      apply Internal.Rep.ext
      intro i
      exact coordinateEquiv_symm_apply _ i
    rwa [hfun] at hcomp

/-- Reading a fixed coordinate preserves local smoothness. -/
@[fun_prop] theorem contDiffAt_coordinate {shape : Shape} {n : ℕ∞ω}
    {f : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) (i : shape.Coord) :
    ContDiffAt ℝ n (fun y => f y i) x :=
  (contDiffAt_iff_coordinates f x).1 hf i

/-- An elementwise map needs smoothness only at the current coordinate values. -/
@[fun_prop] theorem contDiffAt_map {shape : Shape} {n : ℕ∞ω}
    {g : ℝ → ℝ} {f : E → Tensor ℝ shape} {x : E}
    (hg : ∀ i, ContDiffAt ℝ n g (f x i)) (hf : ContDiffAt ℝ n f x) :
    ContDiffAt ℝ n (fun y => Tensor.map g (f y)) x := by
  apply (contDiffAt_iff_coordinates _ x).2
  intro i
  simpa only [Tensor.map, Internal.Rep.map_apply, Function.comp_def] using
    (hg i).comp x (contDiffAt_coordinate hf i)

/-- Locally smooth coordinate families assemble into a locally smooth native tensor. -/
@[fun_prop] theorem contDiffAt_ofFn {shape : Shape} {n : ℕ∞ω}
    {f : E → shape.Coord → ℝ} {x : E} (hf : ∀ i, ContDiffAt ℝ n (fun y => f y i) x) :
    ContDiffAt ℝ n (fun y => Internal.Rep.ofFn (f y)) x := by
  apply (contDiffAt_iff_coordinates _ x).2
  intro i
  simpa only [Internal.Rep.get_ofFn] using hf i

/-- A binary elementwise map needs smoothness only at each current pair of coordinates. -/
@[fun_prop] theorem contDiffAt_map2 {shape : Shape} {n : ℕ∞ω}
    {op : ℝ → ℝ → ℝ} {f g : E → Tensor ℝ shape} {x : E}
    (hop : ∀ i, ContDiffAt ℝ n (fun p : ℝ × ℝ => op p.1 p.2) (f x i, g x i))
    (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    ContDiffAt ℝ n (fun y => Tensor.map2Spec op (f y) (g y)) x := by
  apply (contDiffAt_iff_coordinates _ x).2
  intro i
  simpa only [Tensor.map2Spec_apply, Function.comp_def] using
    (hop i).comp x ((contDiffAt_coordinate hf i).prodMk (contDiffAt_coordinate hg i))

/-- Tensor smoothness is equivalent to smoothness of its finitely many coordinates. -/
theorem contDiff_iff_coordinates {shape : Shape} {n : ℕ∞ω} (f : E → Tensor ℝ shape) :
    ContDiff ℝ n f ↔ ∀ i, ContDiff ℝ n (fun x => f x i) := by
  simp only [contDiff_iff_contDiffAt, contDiffAt_iff_coordinates]
  exact forall_comm

/-- Reading a fixed tensor coordinate preserves the differentiability order. -/
@[fun_prop] theorem contDiff_coordinate {shape : Shape} {n : ℕ∞ω}
    {f : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) (i : shape.Coord) :
    ContDiff ℝ n (fun x => f x i) :=
  (contDiff_iff_coordinates f).1 hf i

/-- Mapping a smooth scalar function over a smooth tensor preserves its differentiability order. -/
@[fun_prop] theorem contDiff_map {shape : Shape} {n : ℕ∞ω}
    {g : ℝ → ℝ} {f : E → Tensor ℝ shape} (hg : ContDiff ℝ n g) (hf : ContDiff ℝ n f) :
    ContDiff ℝ n (fun x => Tensor.map g (f x)) :=
  contDiff_iff_contDiffAt.2 fun _ => contDiffAt_map (fun _ => hg.contDiffAt) hf.contDiffAt

/-- Smooth coordinate families assemble into a smooth native tensor. -/
@[fun_prop] theorem contDiff_ofFn {shape : Shape} {n : ℕ∞ω}
    {f : E → shape.Coord → ℝ} (hf : ∀ i, ContDiff ℝ n (fun x => f x i)) :
    ContDiff ℝ n (fun x => Internal.Rep.ofFn (f x)) :=
  contDiff_iff_contDiffAt.2 fun _ => contDiffAt_ofFn (fun i => (hf i).contDiffAt)

/-- Reading a packed storage index preserves local smoothness. -/
@[fun_prop] theorem contDiffAt_getFlat {shape : Shape} {n : ℕ∞ω}
    {f : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x)
    (i : Fin (Internal.Shape.size shape)) :
    ContDiffAt ℝ n (fun y => (f y).getFlat i) x := by
  simpa only [Internal.Rep.get, Internal.Coord.linearize_unlinearize] using
    contDiffAt_coordinate hf (Internal.Coord.unlinearize i)

/-- Packed row-major lookups are the same continuous coordinate functionals. -/
@[fun_prop] theorem contDiff_getFlat {shape : Shape} {n : ℕ∞ω}
    {f : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) (i : Fin (Internal.Shape.size shape)) :
    ContDiff ℝ n (fun x => (f x).getFlat i) :=
  contDiff_iff_contDiffAt.2 fun _ => contDiffAt_getFlat hf.contDiffAt i

/-- Summation preserves local smoothness, including for empty tensors. -/
@[fun_prop] theorem contDiffAt_sumSpec {shape : Shape} {n : ℕ∞ω}
    {f : E → Tensor ℝ shape} {x : E} (hf : ContDiffAt ℝ n f x) :
    ContDiffAt ℝ n (fun y => sumSpec (f y)) x := by
  simp only [sumSpec, foldlSpec, Internal.Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  exact List.contDiffAt_foldl _ (by fun_prop) contDiffAt_const
    (fun i _ => contDiffAt_getFlat hf i)

/-- Summing the packed tensor preserves smoothness, including when it has no entries. -/
@[fun_prop] theorem contDiff_sumSpec {shape : Shape} {n : ℕ∞ω}
    {f : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) :
    ContDiff ℝ n (fun x => sumSpec (f x)) :=
  contDiff_iff_contDiffAt.2 fun _ => contDiffAt_sumSpec hf.contDiffAt

/-- Applying a smooth binary scalar function to two tensors preserves their smoothness. -/
@[fun_prop] theorem contDiff_map2 {shape : Shape} {n : ℕ∞ω}
    {op : ℝ → ℝ → ℝ} {f g : E → Tensor ℝ shape}
    (hop : ContDiff ℝ n (fun p : ℝ × ℝ => op p.1 p.2))
    (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) :
    ContDiff ℝ n (fun x => Tensor.map2Spec op (f x) (g x)) :=
  contDiff_iff_contDiffAt.2 fun _ =>
    contDiffAt_map2 (fun _ => hop.contDiffAt) hf.contDiffAt hg.contDiffAt

/-- Dot products preserve local smoothness in both tensor arguments. -/
@[fun_prop] theorem contDiffAt_dotSpec {shape : Shape} {n : ℕ∞ω}
    {f g : E → Tensor ℝ shape} {x : E}
    (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    ContDiffAt ℝ n (fun y => dotSpec (f y) (g y)) x := by
  apply contDiffAt_sumSpec
  exact contDiffAt_map2 (fun _ => by fun_prop) hf hg

/-- Dot products of two smooth tensor families are smooth. -/
@[fun_prop] theorem contDiff_dotSpec {shape : Shape} {n : ℕ∞ω}
    {f g : E → Tensor ℝ shape} (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) :
    ContDiff ℝ n (fun x => dotSpec (f x) (g x)) :=
  contDiff_iff_contDiffAt.2 fun _ => contDiffAt_dotSpec hf.contDiffAt hg.contDiffAt

/-- Matrix multiplication needs smoothness of its inputs only near the evaluation point. -/
@[fun_prop] theorem contDiffAt_matMulSpec {n : ℕ∞ω} {rows inner cols : Nat}
    {f : E → Tensor ℝ [rows, inner]} {g : E → Tensor ℝ [inner, cols]} {x : E}
    (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    ContDiffAt ℝ n (fun y => matMulSpec (f y) (g y)) x := by
  unfold matMulSpec
  apply contDiffAt_ofFn
  intro i
  simp only [get2_eq_apply]
  exact List.contDiffAt_foldl _ (by fun_prop) contDiffAt_const
    (fun k _ => (contDiffAt_coordinate hf _).mul (contDiffAt_coordinate hg _))

/-- Matrix multiplication preserves smoothness when both operands vary. -/
@[fun_prop] theorem contDiff_matMulSpec {n : ℕ∞ω} {rows inner cols : Nat}
    {f : E → Tensor ℝ [rows, inner]} {g : E → Tensor ℝ [inner, cols]}
    (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) :
    ContDiff ℝ n (fun x => matMulSpec (f x) (g x)) :=
  contDiff_iff_contDiffAt.2 fun _ => contDiffAt_matMulSpec hf.contDiffAt hg.contDiffAt

/-- Fixed coordinate reads preserve local smoothness, including repeated reads and empty outputs. -/
@[fun_prop] theorem contDiffAt_pull {source target : Shape} {n : ℕ∞ω}
    (index : target.Coord → source.Coord) {f : E → Tensor ℝ source} {x : E}
    (hf : ContDiffAt ℝ n f x) :
    ContDiffAt ℝ n (fun y => Internal.Rep.pull index (f y)) x := by
  simpa only [Function.comp_def, Reindex.pullCLM_apply] using
    (Reindex.pullCLM index).contDiff.contDiffAt.comp x hf

/-- Fixed coordinate reads preserve smoothness, including repeated reads and empty outputs. -/
@[fun_prop] theorem contDiff_pull {source target : Shape} {n : ℕ∞ω}
    (index : target.Coord → source.Coord) {f : E → Tensor ℝ source}
    (hf : ContDiff ℝ n f) : ContDiff ℝ n (fun x => Internal.Rep.pull index (f x)) :=
  contDiff_iff_contDiffAt.2 fun _ => contDiffAt_pull index hf.contDiffAt

/-- Assemble one scalar derivative for each output coordinate into a tensor derivative. -/
def assemble {shape : Shape} (derivatives : shape.Coord → E →L[ℝ] ℝ) :
    E →L[ℝ] Tensor ℝ shape :=
  (coordinateEquiv shape).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi derivatives)

/-- Each coordinate of the assembled tangent is given by its supplied scalar derivative. -/
@[simp] theorem assemble_apply {shape : Shape}
    (derivatives : shape.Coord → E →L[ℝ] ℝ) (dx : E) (i : shape.Coord) :
    assemble derivatives dx i = derivatives i dx := by
  exact coordinateEquiv_symm_apply _ i

/-- Scalar coordinate derivatives determine the full tensor Fréchet derivative.

Finiteness of the shape matters here: assembling the coordinates is a continuous linear map.
The hypotheses therefore imply a Fréchet derivative in the tensor norm, not just separate
directional derivatives of its entries. -/
theorem hasFDerivAt_of_coordinates {shape : Shape} (f : E → Tensor ℝ shape)
    (derivatives : shape.Coord → E →L[ℝ] ℝ) (x : E)
    (h : ∀ i, HasFDerivAt (fun y => f y i) (derivatives i) x) :
    HasFDerivAt f (assemble derivatives) x := by
  have hpi := (hasFDerivAt_pi (𝕜 := ℝ)
    (φ := fun i y => f y i) (φ' := derivatives) (x := x)).2 h
  have hcomp :=
    (coordinateEquiv shape).symm.toContinuousLinearMap.hasFDerivAt.comp x hpi
  have hfun :
      f = (coordinateEquiv shape).symm ∘ (fun y i => f y i) := by
    funext y
    apply Internal.Rep.ext
    intro i
    exact (coordinateEquiv_symm_apply _ i).symm
  exact hcomp.congr_of_eventuallyEq hfun.eventuallyEq

end

end Proofs.Autograd.TensorCoordinates
