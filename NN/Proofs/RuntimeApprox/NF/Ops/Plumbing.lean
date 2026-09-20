/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp


/-!
# NF Tensor Approximation Plumbing

Shape-generic lemmas for `approxTensor` and `linfNorm`. The later NF proof files use these lemmas
to turn scalar absolute-error facts into tensor-level forward-error bounds.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

/-! ## Tensor Approximation Plumbing -/

/-- Enlarge an existing tensor approximation budget.

Composed operators often prove a row- or coordinate-level estimate and then embed it into a global
infinity-norm budget. This lemma records that monotonicity once, without unfolding the tensor
distance predicate at every use site.
-/
theorem approxTensor_mono {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    {xS : SpecTensor s} {xR : Tensor α s} {eps eps' : SpecScalar}
    (h : approxTensor (α := α) (toSpec := toSpec) xS xR eps) (hle : eps ≤ eps') :
    approxTensor (α := α) (toSpec := toSpec) xS xR eps' :=
  le_trans h hle

/-- `linfNorm` is always nonnegative. -/
theorem linf_norm_nonneg : ∀ {s : Shape} (t : SpecTensor s), 0 ≤ linfNorm t := by
  intro s t
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item t]
      dsimp [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm]
      exact abs_nonneg t.item
  | dim n s ih =>
      rw [← Tensor.dim_unstack t]
      have : (0 : ℝ) ≤
          (List.finRange n).foldl
            (fun acc i => max acc (tensorLinfNorm (α := ℝ) (t.unstack i))) 0 := by
        simpa using
          (List.le_foldl_max_init (List.finRange n)
            (fun i => tensorLinfNorm (α := ℝ) (t.unstack i)) 0)
      simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using this

/--
Componentwise bound for `linfNorm` on a dimensioned tensor.

The norm of any component `t[i]` is bounded by the norm of the whole tensor.
-/
theorem linf_norm_le_get_dim {n : Nat} {s : Shape} (t : SpecTensor (.dim n s)) (i : Fin n) :
    linfNorm (t.unstack i) ≤ linfNorm t := by
  rw [show t = Tensor.dim (Tensor.unstack t) from (Tensor.dim_unstack t).symm]
  have hi : i ∈ List.finRange n := List.mem_finRange i
  have hle :=
    List.le_foldl_max_of_mem (List.finRange n)
      (fun j => linfNorm (t.unstack j)) (acc := (0 : ℝ)) hi
  simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using hle

/--
Scalar characterization of `approxTensor` on scalar tensors.

This rewrites `approxTensor (Tensor.scalar x) (Tensor.scalar xR) eps` into the usual absolute-error
inequality `|toSpec xR - x| ≤ eps`.
-/
theorem approxTensor_scalar_iff {α : Type} {toSpec : α → SpecScalar} {x : SpecScalar} {xR : α}
    {eps : SpecScalar} :
    approxTensor (α := α) (toSpec := toSpec) (Tensor.scalar x) (Tensor.scalar xR) eps ↔
      abs (toSpec xR - x) ≤ eps := by
  -- `tensor_distance linf_norm` is `|x - toSpec xR|` on scalar tensors.
  constructor
  · intro h
    have h' : abs (x - toSpec xR) ≤ eps := by
      simpa [approxTensor, approxWith, tensorToSpec, linfNorm, RuntimeApprox.linfNorm,
        tensorDistance, NN.MLTheory.Robustness.Spec.tensor_distance_tensor_sub_eq_sub_spec,
        tensorLinfNorm, TorchLean.Tensor.map, TorchLean.Tensor.subSpec,
        TorchLean.Tensor.map2Spec_scalar, Tensor.item, MathFunctions.abs] using h
    simpa [abs_sub_comm] using h'
  · intro h
    have h' : abs (x - toSpec xR) ≤ eps := by
      simpa [abs_sub_comm] using h
    simpa [approxTensor, approxWith, tensorToSpec, linfNorm, RuntimeApprox.linfNorm,
      tensorDistance, NN.MLTheory.Robustness.Spec.tensor_distance_tensor_sub_eq_sub_spec,
      tensorLinfNorm, TorchLean.Tensor.map, TorchLean.Tensor.subSpec,
      TorchLean.Tensor.map2Spec_scalar, Tensor.item, MathFunctions.abs] using h'

/-- Scalar characterization of `approxTensor` stated for arbitrary rank-zero tensors. -/
theorem approxTensor_scalar_item_iff {α : Type} {toSpec : α → SpecScalar}
    {x : SpecTensor .scalar} {xR : Tensor α .scalar} {eps : SpecScalar} :
    approxTensor (α := α) (toSpec := toSpec) x xR eps ↔
      abs (toSpec xR.item - x.item) ≤ eps := by
  rw [← Tensor.scalar_item x, ← Tensor.scalar_item xR]
  exact approxTensor_scalar_iff

/--
Projection lemma for `approxTensor` on dimensioned tensors.

If `xS` approximates `xR` within `eps`, then each component `xS[i]` approximates `xR[i]` within
  `eps`.
-/
theorem approxTensor_dim_get {α : Type} {toSpec : α → SpecScalar} {n : Nat} {s : Shape}
    {xS : SpecTensor (.dim n s)} {xR : Tensor α (.dim n s)} {eps : SpecScalar}
    (h : approxTensor (α := α) (toSpec := toSpec) xS xR eps) (i : Fin n) :
    approxTensor (α := α) (toSpec := toSpec)
      (xS.unstack i)
      (xR.unstack i)
      eps := by
  have hComp :=
    linf_norm_le_get_dim
      (t := tensorDistance.tensorSub (α := SpecScalar) xS
        (tensorToSpec (α := α) (toSpec := toSpec) xR)) i
  have hResult := le_trans hComp h
  simpa [approxTensor, approxWith, tensorDistance, tensorToSpec,
    NN.MLTheory.Robustness.Spec.tensor_distance_tensor_sub_eq_sub_spec,
    TorchLean.Tensor.subSpec, TorchLean.Tensor.map2Spec, TorchLean.Tensor.map,
    TorchLean.Tensor.unstack,
    TorchLean.Tensor.Internal.Rep.map_unstack,
    TorchLean.Tensor.Internal.Rep.zipWith_unstack] using hResult

/-- Every valid tensor approximation has a nonnegative error budget.

This fact is independent of the executable scalar type.  Keeping it at the plumbing layer avoids
repeating the same norm argument in each backend-specific development.
-/
theorem approxTensor_eps_nonneg {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    {xS : SpecTensor s} {xR : Tensor α s} {eps : SpecScalar}
    (hx : approxTensor (α := α) (toSpec := toSpec) xS xR eps) : 0 ≤ eps := by
  have h0 :
      0 ≤ tensorDistance (α := SpecScalar) linfNorm xS
        (tensorToSpec (α := α) (toSpec := toSpec) xR) := by
    simpa [tensorDistance, linfNorm] using
      (linf_norm_nonneg
        (t := tensorDistance.tensorSub (α := SpecScalar) (s := s) xS
          (tensorToSpec (α := α) (toSpec := toSpec) xR)))
  exact le_trans h0 hx

/-- Assemble a dimensioned tensor approximation from uniform componentwise approximations.

The premise uses one common infinity-norm budget for every component.  This is the converse of
`approxTensor_dim_get` and is deliberately polymorphic in the runtime element representation, so NF,
IEEE execution, interval, and future quantized backends can share the same structural proof.
-/
theorem approxTensor_dim_of_forall {α : Type} {toSpec : α → SpecScalar} {n : Nat} {s : Shape}
    {xS : SpecTensor (.dim n s)} {xR : Tensor α (.dim n s)} {eps : SpecScalar}
    (hε : 0 ≤ eps)
    (h : ∀ i : Fin n,
      approxTensor (α := α) (toSpec := toSpec)
        (xS.unstack i) (xR.unstack i) eps) :
    approxTensor (α := α) (toSpec := toSpec) xS xR eps := by
  classical
  let xSf := Tensor.unstack xS
  let xRf := Tensor.unstack xR
  rw [show xS = Tensor.dim xSf from (Tensor.dim_unstack xS).symm]
  rw [show xR = Tensor.dim xRf from (Tensor.dim_unstack xR).symm]
  change ∀ i, approxTensor (α := α) (toSpec := toSpec) (xSf i) (xRf i) eps at h
  have hf :
      ∀ i ∈ List.finRange n,
        tensorDistance (α := SpecScalar) linfNorm (xSf i)
            (tensorToSpec (α := α) (toSpec := toSpec) (xRf i)) ≤ eps := by
    intro i _hi
    simpa [approxTensor, approxWith] using h i
  have hfold :=
    List.foldl_max_le_of_le (List.finRange n)
      (fun i =>
        tensorDistance (α := SpecScalar) linfNorm (xSf i)
          (tensorToSpec (α := α) (toSpec := toSpec) (xRf i)))
      (acc := (0 : SpecScalar)) (eps := eps) hε hf
  simpa [approxTensor, approxWith, tensorDistance, linfNorm, RuntimeApprox.linfNorm,
    tensorToSpec, tensorLinfNorm, MathFunctions.abs,
    NN.MLTheory.Robustness.Spec.tensor_distance_tensor_sub_eq_sub_spec,
    TorchLean.Tensor.subSpec, TorchLean.Tensor.map2Spec, TorchLean.Tensor.map,
    TorchLean.Tensor.dim, TorchLean.Tensor.unstack,
    TorchLean.Tensor.Internal.Rep.map_stack,
    TorchLean.Tensor.Internal.Rep.zipWith_stack,
    TorchLean.Tensor.Internal.Rep.unstack_stack,
    TorchLean.Tensor.Internal.Rep.map_unstack,
    TorchLean.Tensor.Internal.Rep.zipWith_unstack] using hfold

-- ---------------------------------------------------------------------------
-- Generic lifting lemmas for elementwise ops (`map_spec`, `map2_spec`)
-- ---------------------------------------------------------------------------

/--
Lift a scalar approximation bound to an elementwise `mapSpec`.

Given a scalar bound of the form
`|toSpec (fR xR) - fS x| ≤ bnd (toSpec xR) eps`
and an input approximation `approxTensor xS xR eps`, this produces an approximation bound for
`map_spec fS xS` vs `map_spec fR xR`, with an output epsilon computed by taking the `linfNorm` of
the pointwise bound.
-/
theorem approxTensor_map_spec_of_scalar_bound {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    (fS : SpecScalar → SpecScalar) (fR : α → α) (bnd : SpecScalar → SpecScalar → SpecScalar) :
    ∀ {xS : SpecTensor s} {xR : Tensor α s} {eps : SpecScalar},
      approxTensor (α := α) (toSpec := toSpec) xS xR eps →
        (∀ {x : SpecScalar} {xR : α},
          abs (toSpec xR - x) ≤ eps →
            abs (toSpec (fR xR) - fS x) ≤ bnd (toSpec xR) eps) →
        approxTensor (α := α) (toSpec := toSpec)
          (mapSpec (s := s) fS xS)
          (mapSpec (s := s) fR xR)
          (linfNorm
            (mapSpec (s := s) (fun a => bnd a eps) (tensorToSpec (α := α) (toSpec := toSpec)
              xR))) := by
  intro xS xR eps hx hscalar
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx ⊢
      have hx' :=
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := xS.item) (xR := xR.item) (eps := eps)).1 hx
      have herr :
          abs (toSpec (fR xR.item) - fS xS.item) ≤ bnd (toSpec xR.item) eps :=
        hscalar hx'
      have herr' :
          abs (toSpec (fR xR.item) - fS xS.item) ≤ abs (bnd (toSpec xR.item) eps) :=
        le_trans herr (le_abs_self _)
      exact
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := fS xS.item) (xR := fR xR.item)
          (eps := linfNorm
            (mapSpec (s := Shape.scalar) (fun a => bnd a eps)
              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar xR.item))))).2 (by
                simpa [tensorToSpec, linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
                  mapSpec, Tensor.map, Tensor.item, MathFunctions.abs] using herr')
  | dim n s ih =>
      let boundTensor :=
        mapSpec (s := Shape.dim n s) (fun a => bnd a eps)
          (tensorToSpec (α := α) (toSpec := toSpec) xR)
      let B := linfNorm boundTensor
      have hB_nonneg : 0 ≤ B := linf_norm_nonneg boundTensor
      apply approxTensor_dim_of_forall hB_nonneg
      intro i
      have hx_i :=
        approxTensor_dim_get (α := α) (toSpec := toSpec)
          (xS := xS) (xR := xR) (eps := eps) hx i
      have hih := ih (xS := xS.unstack i) (xR := xR.unstack i) hx_i
      have hB_ge := linf_norm_le_get_dim boundTensor i
      have hB_ge' :
          linfNorm
              (mapSpec (s := s) (fun a => bnd a eps)
                (tensorToSpec (α := α) (toSpec := toSpec) (xR.unstack i)))
            ≤ B := by
        simpa [B, boundTensor, tensorToSpec, mapSpec,
          TorchLean.Tensor.map, TorchLean.Tensor.unstack,
          TorchLean.Tensor.Internal.Rep.map_unstack] using hB_ge
      have hmono := approxTensor_mono hih hB_ge'
      simpa [B, boundTensor, tensorToSpec, mapSpec,
        TorchLean.Tensor.map, TorchLean.Tensor.unstack,
        TorchLean.Tensor.Internal.Rep.map_unstack] using hmono

/--
Lift a domain-restricted scalar approximation bound to an elementwise `mapSpec`.

The predicate is carried by `Tensor.Forall`, so callers can certify operations such as square root
once at the scalar level without rebuilding an elementwise tensor proof for every operation.
-/
theorem approxTensor_map_spec_of_scalar_bound_of_forall
    {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    (predicate : SpecScalar → Prop)
    (fS : SpecScalar → SpecScalar) (fR : α → α)
    (bnd : SpecScalar → SpecScalar → SpecScalar) :
    ∀ {xS : SpecTensor s} {xR : Tensor α s} {eps : SpecScalar},
      approxTensor (α := α) (toSpec := toSpec) xS xR eps →
      Tensor.Forall predicate xS →
        (∀ {x : SpecScalar} {xR : α},
          predicate x →
          abs (toSpec xR - x) ≤ eps →
            abs (toSpec (fR xR) - fS x) ≤ bnd (toSpec xR) eps) →
        approxTensor (α := α) (toSpec := toSpec)
          (mapSpec (s := s) fS xS)
          (mapSpec (s := s) fR xR)
          (linfNorm
            (mapSpec (s := s) (fun a => bnd a eps)
              (tensorToSpec (α := α) (toSpec := toSpec) xR))) := by
  intro xS xR eps hx hdom hscalar
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx ⊢
      have hx' :=
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := xS.item) (xR := xR.item) (eps := eps)).1 hx
      have hdom' : predicate xS.item := by
        simpa [Tensor.Forall] using hdom
      have herr :
          abs (toSpec (fR xR.item) - fS xS.item) ≤ bnd (toSpec xR.item) eps :=
        hscalar hdom' hx'
      exact
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := fS xS.item) (xR := fR xR.item)
          (eps := linfNorm
            (mapSpec (s := Shape.scalar) (fun a => bnd a eps)
              (tensorToSpec (α := α) (toSpec := toSpec)
                (Tensor.scalar xR.item))))).2 (by
                  simpa [tensorToSpec, linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
                    mapSpec, Tensor.map, Tensor.item, MathFunctions.abs] using
                    (le_trans herr (le_abs_self (bnd (toSpec xR.item) eps))))
  | dim n inner ih =>
      let boundTensor :=
        mapSpec (s := Shape.dim n inner) (fun a => bnd a eps)
          (tensorToSpec (α := α) (toSpec := toSpec) xR)
      let bound := linfNorm boundTensor
      have hbound : 0 ≤ bound := linf_norm_nonneg boundTensor
      apply approxTensor_dim_of_forall hbound
      intro i
      have hxI :=
        approxTensor_dim_get (α := α) (toSpec := toSpec)
          (xS := xS) (xR := xR) (eps := eps) hx i
      have hdomI : Tensor.Forall predicate (xS.unstack i) := hdom i
      have hlocal := ih hxI hdomI
      have hle :
          linfNorm
              (mapSpec (s := inner) (fun a => bnd a eps)
                (tensorToSpec (α := α) (toSpec := toSpec) (xR.unstack i))) ≤
            bound := by
        have h := linf_norm_le_get_dim boundTensor i
        simpa [bound, boundTensor, tensorToSpec, mapSpec,
          TorchLean.Tensor.map, TorchLean.Tensor.unstack,
          TorchLean.Tensor.Internal.Rep.map_unstack] using h
      have hmono := approxTensor_mono hlocal hle
      simpa [bound, boundTensor, tensorToSpec, mapSpec,
        TorchLean.Tensor.map, TorchLean.Tensor.unstack,
        TorchLean.Tensor.Internal.Rep.map_unstack] using hmono

/--
Lift a scalar approximation bound that depends on the original runtime scalar.

This variant is useful when the certified budget observes representation-specific runtime
intermediates that cannot be reconstructed from `toSpec xR`.
-/
theorem approxTensor_map_spec_of_runtime_scalar_bound
    {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    (fS : SpecScalar → SpecScalar) (fR : α → α)
    (bnd : α → SpecScalar → SpecScalar) :
    ∀ {xS : SpecTensor s} {xR : Tensor α s} {eps : SpecScalar},
      approxTensor (α := α) (toSpec := toSpec) xS xR eps →
        (∀ {x : SpecScalar} {xR : α},
          abs (toSpec xR - x) ≤ eps →
            abs (toSpec (fR xR) - fS x) ≤ bnd xR eps) →
        approxTensor (α := α) (toSpec := toSpec)
          (mapSpec (s := s) fS xS)
          (mapSpec (s := s) fR xR)
          (linfNorm (Tensor.map (fun a => bnd a eps) xR)) := by
  intro xS xR eps hx hscalar
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx ⊢
      have hx' :=
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := xS.item) (xR := xR.item) (eps := eps)).1 hx
      have herr := hscalar hx'
      exact
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := fS xS.item) (xR := fR xR.item)
          (eps := linfNorm
            (Tensor.map (fun a => bnd a eps)
              (Tensor.scalar xR.item)))).2 (by
                simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
                  mapSpec, Tensor.map, Tensor.item, MathFunctions.abs] using
                  (le_trans herr (le_abs_self (bnd xR.item eps))))
  | dim n inner ih =>
      let boundTensor :=
        Tensor.map (fun a => bnd a eps) xR
      let bound := linfNorm boundTensor
      have hbound : 0 ≤ bound := linf_norm_nonneg boundTensor
      apply approxTensor_dim_of_forall hbound
      intro i
      have hxI :=
        approxTensor_dim_get (α := α) (toSpec := toSpec)
          (xS := xS) (xR := xR) (eps := eps) hx i
      have hlocal := ih hxI
      have hle :
          linfNorm
              (Tensor.map (fun a => bnd a eps) (xR.unstack i)) ≤
            bound := by
        have h := linf_norm_le_get_dim boundTensor i
        simpa [bound, boundTensor, mapSpec, TorchLean.Tensor.map,
          TorchLean.Tensor.unstack,
          TorchLean.Tensor.Internal.Rep.map_unstack] using h
      have hmono := approxTensor_mono hlocal hle
      simpa [bound, boundTensor, mapSpec, TorchLean.Tensor.map,
        TorchLean.Tensor.unstack,
        TorchLean.Tensor.Internal.Rep.map_unstack] using hmono

/--
Lift a scalar approximation bound to an elementwise `map2Spec`.

This is the binary analogue of `approxTensor_map_spec_of_scalar_bound`, used for elementwise
arithmetic (`add`, `sub`, `mulElem`, etc.).
-/
theorem approxTensor_map2_spec_of_scalar_bound {α : Type} {toSpec : α → SpecScalar} {s : Shape}
    (fS : SpecScalar → SpecScalar → SpecScalar) (fR : α → α → α)
    (bnd : SpecScalar → SpecScalar → SpecScalar → SpecScalar → SpecScalar) :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor α s} {epsx epsy : SpecScalar},
      approxTensor (α := α) (toSpec := toSpec) xS xR epsx →
      approxTensor (α := α) (toSpec := toSpec) yS yR epsy →
        (∀ {x y : SpecScalar} {xR yR : α},
          abs (toSpec xR - x) ≤ epsx →
          abs (toSpec yR - y) ≤ epsy →
            abs (toSpec (fR xR yR) - fS x y) ≤ bnd (toSpec xR) (toSpec yR) epsx epsy) →
        approxTensor (α := α) (toSpec := toSpec)
          (map2Spec fS xS yS)
          (map2Spec fR xR yR)
          (linfNorm
            (map2Spec (fun a b => bnd a b epsx epsy)
              (tensorToSpec (α := α) (toSpec := toSpec) xR)
              (tensorToSpec (α := α) (toSpec := toSpec) yR))) := by
  intro xS yS xR yR epsx epsy hx hy hscalar
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx
      rw [← Tensor.scalar_item yS, ← Tensor.scalar_item yR] at hy
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item yS,
        ← Tensor.scalar_item xR, ← Tensor.scalar_item yR]
      have hx' :=
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := xS.item) (xR := xR.item) (eps := epsx)).1 hx
      have hy' :=
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := yS.item) (xR := yR.item) (eps := epsy)).1 hy
      have herr :
          abs (toSpec (fR xR.item yR.item) - fS xS.item yS.item) ≤
            bnd (toSpec xR.item) (toSpec yR.item) epsx epsy :=
        hscalar hx' hy'
      have herr' :
          abs (toSpec (fR xR.item yR.item) - fS xS.item yS.item) ≤
            abs (bnd (toSpec xR.item) (toSpec yR.item) epsx epsy) :=
        le_trans herr (le_abs_self _)
      exact
        (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
          (x := fS xS.item yS.item) (xR := fR xR.item yR.item)
          (eps := linfNorm
            (map2Spec (fun a b => bnd a b epsx epsy)
              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar xR.item))
              (tensorToSpec (α := α) (toSpec := toSpec) (Tensor.scalar yR.item))))).2 (by
                simpa [tensorToSpec, linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
                  mapSpec, map2Spec, Tensor.map, Tensor.item, MathFunctions.abs] using herr')
  | dim n s ih =>
      let boundTensor :=
        map2Spec (fun a b => bnd a b epsx epsy)
          (tensorToSpec (α := α) (toSpec := toSpec) xR)
          (tensorToSpec (α := α) (toSpec := toSpec) yR)
      let B := linfNorm boundTensor
      have hB_nonneg : 0 ≤ B := linf_norm_nonneg boundTensor
      apply approxTensor_dim_of_forall hB_nonneg
      intro i
      have hx_i :=
        approxTensor_dim_get (α := α) (toSpec := toSpec)
          (xS := xS) (xR := xR) (eps := epsx) hx i
      have hy_i :=
        approxTensor_dim_get (α := α) (toSpec := toSpec)
          (xS := yS) (xR := yR) (eps := epsy) hy i
      have hih :=
        ih (xS := xS.unstack i) (yS := yS.unstack i)
          (xR := xR.unstack i) (yR := yR.unstack i) hx_i hy_i
      have hB_ge := linf_norm_le_get_dim boundTensor i
      have hB_ge' :
          linfNorm
              (map2Spec (fun a b => bnd a b epsx epsy)
                (tensorToSpec (α := α) (toSpec := toSpec) (xR.unstack i))
                (tensorToSpec (α := α) (toSpec := toSpec) (yR.unstack i)))
            ≤ B := by
        simpa [B, boundTensor, tensorToSpec, map2Spec,
          TorchLean.Tensor.map, TorchLean.Tensor.unstack,
          TorchLean.Tensor.Internal.Rep.map_unstack,
          TorchLean.Tensor.Internal.Rep.zipWith_unstack] using hB_ge
      have hmono := approxTensor_mono hih hB_ge'
      simpa [B, boundTensor, tensorToSpec, map2Spec,
        TorchLean.Tensor.map, TorchLean.Tensor.unstack,
        TorchLean.Tensor.Internal.Rep.map_unstack,
        TorchLean.Tensor.Internal.Rep.zipWith_unstack] using hmono

end

end RuntimeApprox
end Proofs
