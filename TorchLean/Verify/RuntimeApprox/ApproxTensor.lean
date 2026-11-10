/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Graph.Forward

/-!
# NF Tensor Approximation Plumbing

Shape-generic lemmas about `ApproxTensor` and `linfNorm`. An `ApproxTensor` budget is an
`L∞` bound, so it caps the error of every entry. The NF operator proofs use these lemmas to turn
scalar absolute-error bounds into forward-error bounds on whole tensors.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean

open TorchLean.Tensor
open Robustness

noncomputable section

/-! ## Tensor Approximation Plumbing -/

/-- An approximation within `eps` is also an approximation within any larger budget `eps'`.

Composed operators often prove a row- or coordinate-level estimate and then widen it to a global
`L∞` budget. The lemma takes that step once, so no use site has to unfold the distance predicate.
-/
theorem ApproxTensor.mono {α : Type} [Storage α] {toSpec : α → ℝ} {s : Shape}
    {xS : Tensor ℝ s} {xR : Tensor α s} {eps eps' : ℝ}
    (h : ApproxTensor (α := α) (toSpec := toSpec) xS xR eps) (hle : eps ≤ eps') :
    ApproxTensor (α := α) (toSpec := toSpec) xS xR eps' :=
  le_trans h hle

/-- The L∞ norm `linfNorm` of a real tensor is nonnegative. -/
theorem Tensor.linfNorm_nonneg : ∀ {s : Shape} (t : Tensor ℝ s), 0 ≤ linfNorm t := by
  intro s t
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item t]
      dsimp [linfNorm]
      exact abs_nonneg t.item
  | dim n s ih =>
      rw [← Tensor.dim_unstack t]
      have : (0 : ℝ) ≤
          (List.finRange n).foldl
            (fun acc i => max acc (linfNorm (α := ℝ) (t.unstack i))) 0 := by
        simpa using
          (List.le_foldl_max_init (List.finRange n)
            (fun i => linfNorm (α := ℝ) (t.unstack i)) 0)
      simpa [linfNorm] using this

/--
Each slice `t.unstack i` of a real tensor has `linfNorm` at most that of the whole tensor.
-/
theorem Tensor.linfNorm_unstack_le {n : Nat} {s : Shape} (t : Tensor ℝ (.dim n s)) (i : Fin n) :
    linfNorm (t.unstack i) ≤ linfNorm t := by
  rw [show t = Tensor.dim (Tensor.unstack t) from (Tensor.dim_unstack t).symm]
  have hi : i ∈ List.finRange n := List.mem_finRange i
  have hle :=
    List.le_foldl_max_of_mem (List.finRange n)
      (fun j => linfNorm (t.unstack j)) (acc := (0 : ℝ)) hi
  simpa [linfNorm] using hle

/--
On scalar tensors, `ApproxTensor (Tensor.scalar x) (Tensor.scalar xR) eps` is equivalent to the
absolute-error inequality `|toSpec xR - x| ≤ eps`.
-/
theorem approxTensor_scalar_iff {α : Type} [Storage α] {toSpec : α → ℝ}
    {x : ℝ} {xR : α} {eps : ℝ} :
    ApproxTensor (α := α) (toSpec := toSpec) (Tensor.scalar x) (Tensor.scalar xR) eps ↔
      abs (toSpec xR - x) ≤ eps := by
  -- On scalar tensors `distance linfNorm` unfolds to `|x - toSpec xR|`, and `abs_sub_comm` turns
  -- it into the stated `|toSpec xR - x|`.
  constructor
  · intro h
    have h' : abs (x - toSpec xR) ≤ eps := by
      simpa [ApproxTensor, ApproxWith, toReal,
        distance, linfNorm, Tensor.sub_def,
        Tensor.zipWith_scalar, Tensor.item, MathFunctions.abs] using h
    simpa [abs_sub_comm] using h'
  · intro h
    have h' : abs (x - toSpec xR) ≤ eps := by
      simpa [abs_sub_comm] using h
    simpa [ApproxTensor, ApproxWith, toReal,
      distance, linfNorm, Tensor.sub_def,
      Tensor.zipWith_scalar, Tensor.item, MathFunctions.abs] using h'

/-- For rank-zero tensors, `ApproxTensor x xR eps` is equivalent to
`|toSpec xR.item - x.item| ≤ eps`. -/
theorem approxTensor_scalar_item_iff {α : Type} [Storage α] {toSpec : α → ℝ}
    {x : Tensor ℝ .scalar} {xR : Tensor α .scalar} {eps : ℝ} :
    ApproxTensor (α := α) (toSpec := toSpec) x xR eps ↔
      abs (toSpec xR.item - x.item) ≤ eps := by
  rw [← Tensor.scalar_item x, ← Tensor.scalar_item xR]
  simpa only [Tensor.item_scalar] using
    (approxTensor_scalar_iff (α := α) (toSpec := toSpec)
      (x := x.item) (xR := xR.item) (eps := eps))

/--
If `xS` approximates `xR` within `eps`, then each slice `xS.unstack i` approximates `xR.unstack i`
within the same `eps`.
-/
theorem ApproxTensor.unstack {α : Type} [Storage α] {toSpec : α → ℝ} {n : Nat} {s : Shape}
    {xS : Tensor ℝ (.dim n s)} {xR : Tensor α (.dim n s)} {eps : ℝ}
    (h : ApproxTensor (α := α) (toSpec := toSpec) xS xR eps) (i : Fin n) :
    ApproxTensor (α := α) (toSpec := toSpec)
      (xS.unstack i)
      (xR.unstack i)
      eps := by
  have hComp :=
    linfNorm_unstack_le
      (t := xS -
        toReal (α := α) (toSpec := toSpec) xR) i
  have hResult := le_trans hComp h
  simpa [ApproxTensor, ApproxWith, distance, toReal,
    Tensor.sub_def,
    ← Tensor.unstack_map,
    ← Tensor.unstack_zipWith] using hResult

/-- A tensor approximation within `eps` bounds the error of every entry, at any shape:
`|toSpec (xR index) - xS index| ≤ eps` for each coordinate `index`. -/
theorem ApproxTensor.abs_sub_le {α : Type} [Storage α] {toSpec : α → ℝ} {s : Shape}
    {xS : Tensor ℝ s} {xR : Tensor α s} {eps : ℝ}
    (h : ApproxTensor (toSpec := toSpec) xS xR eps) (index : s.Coord) :
    |toSpec (xR index) - xS index| ≤ eps := by
  induction s with
  | scalar =>
      cases index
      exact (approxTensor_scalar_item_iff (toSpec := toSpec)).mp h
  | dim n s ih =>
      simpa only [Tensor.unstack_apply] using
        ih (unstack h index.1) index.2

/-- The budget `eps` of any tensor approximation is nonnegative, because it bounds a norm.

The argument uses nothing about the runtime scalar type, so it lives here once for every backend.
-/
theorem ApproxTensor.eps_nonneg {α : Type} [Storage α] {toSpec : α → ℝ} {s : Shape}
    {xS : Tensor ℝ s} {xR : Tensor α s} {eps : ℝ}
    (hx : ApproxTensor (α := α) (toSpec := toSpec) xS xR eps) : 0 ≤ eps := by
  have h0 :
      0 ≤ distance (α := ℝ) linfNorm xS
        (toReal (α := α) (toSpec := toSpec) xR) := by
    simpa [distance] using
      (linfNorm_nonneg
        (t := xS -
          toReal (α := α) (toSpec := toSpec) xR))
  exact le_trans h0 hx

/-- If each slice `xS.unstack i` approximates `xR.unstack i` within one common budget `eps`, then
`xS` approximates `xR` within `eps`. The side condition `0 ≤ eps` covers the case `n = 0`.

The lemma is the converse of `ApproxTensor.unstack`. It is polymorphic in the runtime element
type, so every backend with a `Storage` instance can use the same structural proof.
-/
theorem approxTensor_dim_of_forall {α : Type} [Storage α] {toSpec : α → ℝ}
    {n : Nat} {s : Shape}
    {xS : Tensor ℝ (.dim n s)} {xR : Tensor α (.dim n s)} {eps : ℝ}
    (hε : 0 ≤ eps)
    (h : ∀ i : Fin n,
      ApproxTensor (α := α) (toSpec := toSpec)
        (xS.unstack i) (xR.unstack i) eps) :
    ApproxTensor (α := α) (toSpec := toSpec) xS xR eps := by
  classical
  let xSf := Tensor.unstack xS
  let xRf := Tensor.unstack xR
  rw [show xS = Tensor.dim xSf from (Tensor.dim_unstack xS).symm]
  rw [show xR = Tensor.dim xRf from (Tensor.dim_unstack xR).symm]
  change ∀ i, ApproxTensor (α := α) (toSpec := toSpec) (xSf i) (xRf i) eps at h
  have hf :
      ∀ i ∈ List.finRange n,
        distance (α := ℝ) linfNorm (xSf i)
            (toReal (α := α) (toSpec := toSpec) (xRf i)) ≤ eps := by
    intro i _hi
    simpa [ApproxTensor, ApproxWith] using h i
  have hfold :=
    List.foldl_max_le_of_le (List.finRange n)
      (fun i =>
        distance (α := ℝ) linfNorm (xSf i)
          (toReal (α := α) (toSpec := toSpec) (xRf i)))
      (acc := (0 : ℝ)) (eps := eps) hε hf
  simpa [ApproxTensor, ApproxWith, distance,
    toReal, linfNorm, MathFunctions.abs,
    Tensor.sub_def,
    Tensor.map_dim,
    Tensor.zipWith_dim,
    Tensor.unstack_dim,
    ← Tensor.unstack_map,
    Tensor.Rep.zipWith_unstack] using hfold

-- ---------------------------------------------------------------------------
-- Generic lifting lemmas for elementwise ops (`Tensor.map`, `zipWith`)
-- ---------------------------------------------------------------------------

/-- A uniform entrywise bound is a tensor approximation, the converse of `ApproxTensor.abs_sub_le`.
-/
theorem approxTensor_of_forall {α : Type} [Storage α] {toSpec : α → ℝ} {s : Shape}
    {xS : Tensor ℝ s} {xR : Tensor α s} {eps : ℝ} (hε : 0 ≤ eps)
    (h : ∀ index : s.Coord, |toSpec (xR index) - xS index| ≤ eps) :
    ApproxTensor (toSpec := toSpec) xS xR eps := by
  induction s with
  | scalar =>
      exact (approxTensor_scalar_item_iff (toSpec := toSpec)).mpr (h PUnit.unit)
  | dim n s ih =>
      refine approxTensor_dim_of_forall hε fun i => ih fun index => ?_
      simpa only [Tensor.unstack_apply] using h (i, index)

/--
Lift a domain-restricted scalar approximation bound to an elementwise `Tensor.map`.

The predicate is carried by `Tensor.Forall`, so callers can certify operations such as square root
once at the scalar level without rebuilding an elementwise tensor proof for every operation.
-/
theorem ApproxTensor.map_of_forall
    {α : Type} [Storage α] {toSpec : α → ℝ} {s : Shape}
    (predicate : ℝ → Prop)
    (fS : ℝ → ℝ) (fR : α → α)
    (bnd : ℝ → ℝ → ℝ) :
    ∀ {xS : Tensor ℝ s} {xR : Tensor α s} {eps : ℝ},
      ApproxTensor (α := α) (toSpec := toSpec) xS xR eps →
      Tensor.Forall predicate xS →
        (∀ {x : ℝ} {xR : α},
          predicate x →
          abs (toSpec xR - x) ≤ eps →
            abs (toSpec (fR xR) - fS x) ≤ bnd (toSpec xR) eps) →
        ApproxTensor (α := α) (toSpec := toSpec)
          (Tensor.map (shape := s) fS xS)
          (Tensor.map (shape := s) fR xR)
          (linfNorm
            (Tensor.map (shape := s) (fun a => bnd a eps)
              (toReal (α := α) (toSpec := toSpec) xR))) := by
  intro xS xR eps hx hdom hscalar
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx ⊢
      rw [Tensor.map_scalar, Tensor.map_scalar]
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
            (Tensor.map (shape := Shape.scalar) (fun a => bnd a eps)
              (toReal (α := α) (toSpec := toSpec)
                (Tensor.scalar xR.item))))).2 (by
                  simpa [toReal, linfNorm,
                    Tensor.item, MathFunctions.abs] using
                    (le_trans herr (le_abs_self (bnd (toSpec xR.item) eps))))
  | dim n inner ih =>
      let boundTensor :=
        Tensor.map (shape := Shape.dim n inner) (fun a => bnd a eps)
          (toReal (α := α) (toSpec := toSpec) xR)
      let bound := linfNorm boundTensor
      have hbound : 0 ≤ bound := linfNorm_nonneg boundTensor
      apply approxTensor_dim_of_forall hbound
      intro i
      have hxI :=
        unstack (α := α) (toSpec := toSpec)
          (xS := xS) (xR := xR) (eps := eps) hx i
      have hdomI : Tensor.Forall predicate (xS.unstack i) := hdom i
      have hlocal := ih hxI hdomI
      have hle :
          linfNorm
              (Tensor.map (shape := inner) (fun a => bnd a eps)
                (toReal (α := α) (toSpec := toSpec) (xR.unstack i))) ≤
            bound := by
        have h := linfNorm_unstack_le boundTensor i
        simpa [bound, boundTensor, toReal,
          ← Tensor.unstack_map] using h
      have hmono := mono hlocal hle
      simpa [bound, boundTensor, toReal,
        ← Tensor.unstack_map] using hmono

/--
Lift a scalar approximation bound to an elementwise `Tensor.map`.

Given `ApproxTensor xS xR eps` and a scalar bound `|toSpec (fR xR) - fS x| ≤ bnd (toSpec xR) eps`
for every input within `eps`, the tensors `Tensor.map fS xS` and `Tensor.map fR xR` agree within
the `linfNorm` of the pointwise bounds. The lemma is
`ApproxTensor.map_of_forall` at the trivial predicate.
-/
theorem ApproxTensor.map {α : Type} [Storage α] {toSpec : α → ℝ}
    {s : Shape}
    (fS : ℝ → ℝ) (fR : α → α) (bnd : ℝ → ℝ → ℝ) :
    ∀ {xS : Tensor ℝ s} {xR : Tensor α s} {eps : ℝ},
      ApproxTensor (α := α) (toSpec := toSpec) xS xR eps →
        (∀ {x : ℝ} {xR : α},
          abs (toSpec xR - x) ≤ eps →
            abs (toSpec (fR xR) - fS x) ≤ bnd (toSpec xR) eps) →
        ApproxTensor (α := α) (toSpec := toSpec)
          (Tensor.map (shape := s) fS xS)
          (Tensor.map (shape := s) fR xR)
          (linfNorm
            (Tensor.map (shape := s) (fun a => bnd a eps) (toReal (α := α) (toSpec := toSpec)
              xR))) := by
  intro xS xR eps hx hscalar
  exact map_of_forall (α := α) (toSpec := toSpec)
    (fun _ => True) fS fR bnd hx (Tensor.forall_true xS) (fun _ h => hscalar h)

/--
Lift a scalar approximation bound that reads the runtime scalar itself to an elementwise
`Tensor.map`.

Some certified budgets are computed from runtime intermediates, such as the rounded denominator of
the sigmoid, so `bnd` takes the runtime value `xR`. The output budget is the `linfNorm`
of `bnd xR eps` over the entries.
-/
theorem ApproxTensor.map_of_runtime_bound
    {α : Type} [Storage α] {toSpec : α → ℝ} {s : Shape}
    (fS : ℝ → ℝ) (fR : α → α)
    (bnd : α → ℝ → ℝ) :
    ∀ {xS : Tensor ℝ s} {xR : Tensor α s} {eps : ℝ},
      ApproxTensor (α := α) (toSpec := toSpec) xS xR eps →
        (∀ {x : ℝ} {xR : α},
          abs (toSpec xR - x) ≤ eps →
            abs (toSpec (fR xR) - fS x) ≤ bnd xR eps) →
        ApproxTensor (α := α) (toSpec := toSpec)
          (Tensor.map (shape := s) fS xS)
          (Tensor.map (shape := s) fR xR)
          (linfNorm (Tensor.map (fun a => bnd a eps) xR)) := by
  intro xS xR eps hx hscalar
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx ⊢
      rw [Tensor.map_scalar, Tensor.map_scalar]
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
                simpa [linfNorm,
                  Tensor.item, MathFunctions.abs] using
                  (le_trans herr (le_abs_self (bnd xR.item eps))))
  | dim n inner ih =>
      let boundTensor :=
        Tensor.map (fun a => bnd a eps) xR
      let bound := linfNorm boundTensor
      have hbound : 0 ≤ bound := linfNorm_nonneg boundTensor
      apply approxTensor_dim_of_forall hbound
      intro i
      have hxI :=
        unstack (α := α) (toSpec := toSpec)
          (xS := xS) (xR := xR) (eps := eps) hx i
      have hlocal := ih hxI
      have hle :
          linfNorm
              (Tensor.map (fun a => bnd a eps) (xR.unstack i)) ≤
            bound := by
        have h := linfNorm_unstack_le boundTensor i
        simpa [bound, boundTensor,
          ← Tensor.unstack_map] using h
      have hmono := mono hlocal hle
      simpa [bound, boundTensor,
        ← Tensor.unstack_map] using hmono

/--
Lift a scalar approximation bound to an elementwise `zipWith`.

It is the binary analogue of `map`, and the tensor bounds for elementwise addition, subtraction,
multiplication and safe division (`ApproxTensor.safeDiv`) go through it.
-/
theorem ApproxTensor.zipWith {α : Type} [Storage α] {toSpec : α → ℝ}
    {s : Shape}
    (fS : ℝ → ℝ → ℝ) (fR : α → α → α)
    (bnd : ℝ → ℝ → ℝ → ℝ → ℝ) :
    ∀ {xS yS : Tensor ℝ s} {xR yR : Tensor α s} {epsx epsy : ℝ},
      ApproxTensor (α := α) (toSpec := toSpec) xS xR epsx →
      ApproxTensor (α := α) (toSpec := toSpec) yS yR epsy →
        (∀ {x y : ℝ} {xR yR : α},
          abs (toSpec xR - x) ≤ epsx →
          abs (toSpec yR - y) ≤ epsy →
            abs (toSpec (fR xR yR) - fS x y) ≤ bnd (toSpec xR) (toSpec yR) epsx epsy) →
        ApproxTensor (α := α) (toSpec := toSpec)
          (Tensor.zipWith fS xS yS)
          (Tensor.zipWith fR xR yR)
          (linfNorm
            (Tensor.zipWith (fun a b => bnd a b epsx epsy)
              (toReal (α := α) (toSpec := toSpec) xR)
              (toReal (α := α) (toSpec := toSpec) yR))) := by
  intro xS yS xR yR epsx epsy hx hy hscalar
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx
      rw [← Tensor.scalar_item yS, ← Tensor.scalar_item yR] at hy
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item yS,
        ← Tensor.scalar_item xR, ← Tensor.scalar_item yR]
      rw [Tensor.zipWith_scalar, Tensor.zipWith_scalar]
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
            (Tensor.zipWith (fun a b => bnd a b epsx epsy)
              (toReal (α := α) (toSpec := toSpec) (Tensor.scalar xR.item))
              (toReal (α := α) (toSpec := toSpec) (Tensor.scalar yR.item))))).2 (by
                simpa [toReal, linfNorm,
                  Tensor.item, MathFunctions.abs] using herr')
  | dim n s ih =>
      let boundTensor :=
        Tensor.zipWith (fun a b => bnd a b epsx epsy)
          (toReal (α := α) (toSpec := toSpec) xR)
          (toReal (α := α) (toSpec := toSpec) yR)
      let B := linfNorm boundTensor
      have hB_nonneg : 0 ≤ B := linfNorm_nonneg boundTensor
      apply approxTensor_dim_of_forall hB_nonneg
      intro i
      have hx_i :=
        unstack (α := α) (toSpec := toSpec)
          (xS := xS) (xR := xR) (eps := epsx) hx i
      have hy_i :=
        unstack (α := α) (toSpec := toSpec)
          (xS := yS) (xR := yR) (eps := epsy) hy i
      have hih :=
        ih (xS := xS.unstack i) (yS := yS.unstack i)
          (xR := xR.unstack i) (yR := yR.unstack i) hx_i hy_i
      have hB_ge := linfNorm_unstack_le boundTensor i
      have hB_ge' :
          linfNorm
              (Tensor.zipWith (fun a b => bnd a b epsx epsy)
                (toReal (α := α) (toSpec := toSpec) (xR.unstack i))
                (toReal (α := α) (toSpec := toSpec) (yR.unstack i)))
            ≤ B := by
        simpa [B, boundTensor, toReal,
          ← Tensor.unstack_map,
          ← Tensor.unstack_zipWith] using hB_ge
      have hmono := mono hih hB_ge'
      simpa [B, boundTensor, toReal,
        ← Tensor.unstack_map,
        ← Tensor.unstack_zipWith] using hmono

end

end TorchLean
