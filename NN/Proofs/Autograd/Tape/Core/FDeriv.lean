/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Core.Vectorization
public import NN.Proofs.Autograd.Tape.Core.Soundness
public import Mathlib.Analysis.Calculus.FDeriv.Prod
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# FDeriv

Analytic (`HasFDerivAt`/`fderiv`) correctness for **tape-style SSA/DAG graphs**.

`NN/Proofs/Autograd/Tape/Core/Soundness.lean` proves the global JVP/VJP adjointness law for DAG
  graphs
against the tensor dot product.

This file adds the analytic upgrade (spec-level over `ℝ`):

* vectorize heterogeneous contexts into Euclidean space;
* assume each node's JVP is the Fréchet derivative of its forward map;
* derive `jvp = fderiv` and therefore `backprop = (fderiv eval)†`.

## PyTorch correspondence / citations
- `backpropVec` is the proof-level analogue of a VJP accumulation pass over a dynamic tape.
  The main theorem `backpropVec_eq_adjoint_fderiv` corresponds to the slogan
  “reverse-mode = adjoint of the derivative of the forward map”.
  https://pytorch.org/docs/stable/autograd.html
- For PyTorch’s functional API perspective (Jacobian/VJP/JVP): see the “functional higher level”
  autograd docs.
  https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec TorchLean
open TorchLean TorchLean.Tensor

open scoped BigOperators

noncomputable section

-- ---------------------------------------------------------------------------
-- Tensor vectorization (`Tensor ℝ s` ↔ `Vec (Spec.Shape.size s)`)
-- ---------------------------------------------------------------------------

/-- Vectorize a tensor by flattening it (spec flattening order) and then using the Euclidean
  equivalence. -/
def tensorToVec {s : Shape} (t : Tensor ℝ s) : Vec (Spec.Shape.size s) :=
  getScalarE (n := Spec.Shape.size s) (flattenSpec (α := ℝ) t)

/-- The sole coordinate obtained by vectorizing a scalar tensor is the scalar itself. -/
@[simp] theorem tensorToVec_scalar (x : ℝ) (i : Fin (Spec.Shape.size Shape.scalar)) :
    tensorToVec (t := (Tensor.scalar x : Tensor ℝ .scalar)) i = x := by
  change (getScalarE (flattenSpec (Tensor.scalar x))).ofLp i = x
  rw [getScalarE_ofLp]
  cases i with
  | mk value hvalue =>
      have hzero : value = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hvalue)
      subst value
      rfl

/-- Inverse of `tensorToVec`: interpret a vector as a tensor of shape `s`. -/
def vecToTensor {s : Shape} (v : Vec (Spec.Shape.size s)) : Tensor ℝ s :=
  unflattenSpec (α := ℝ) s (ofFnE (n := Spec.Shape.size s) v)

/-- Vectorizing a tensor built from a vector gives the vector back. -/
@[simp] theorem tensorToVec_vecToTensor {s : Shape} (v : Vec (Spec.Shape.size s)) :
    tensorToVec (t := vecToTensor (s := s) v) = v := by
  have hunf :
      flattenSpec (α := ℝ)
          (unflattenSpec (α := ℝ) s (ofFnE (n := Spec.Shape.size s) v))
        =
      ofFnE (n := Spec.Shape.size s) v :=
    flattenSpec_unflattenSpec (shape := s) (tensor := ofFnE (n := Spec.Shape.size s) v)
  have := congrArg (getScalarE (n := Spec.Shape.size s)) hunf
  exact this.trans (getScalarE_ofFnE (n := Spec.Shape.size s) v)

/-- The other round trip, so `tensorToVec` and `vecToTensor` are mutually inverse.

Having both directions as `simp` lemmas is what lets the calculus below work entirely in
`EuclideanSpace` and still state its conclusions about tensors. -/
@[simp] theorem vecToTensor_tensorToVec {s : Shape} (t : Tensor ℝ s) :
    vecToTensor (s := s) (tensorToVec (t := t)) = t := by
  have hround :
      ofFnE (n := Spec.Shape.size s) (getScalarE (n := Spec.Shape.size s) (flattenSpec (α := ℝ) t))
        =
      flattenSpec (α := ℝ) t := by
    simp
  change unflattenSpec (α := ℝ) s
      (ofFnE (n := Spec.Shape.size s)
        (getScalarE (n := Spec.Shape.size s) (flattenSpec (α := ℝ) t))) = t
  rw [hround]
  exact unflattenSpec_flattenSpec (shape := s) (tensor := t)

-- ---------------------------------------------------------------------------
-- Context vectorization (`TorchLean.TensorPack ℝ Γ` ↔ `Vec (ctxSize Γ)`)
-- ---------------------------------------------------------------------------

/-- Total number of scalar coordinates in a heterogeneous context shape list. -/
def ctxSize : List Shape → Nat
  | [] => 0
  | s :: ss => Spec.Shape.size s + ctxSize ss

/-- A vectorized context containing every entry of a `TorchLean.TensorPack ℝ Γ`. -/
abbrev CtxVec (Γ : List Shape) := Vec (ctxSize Γ)

/--
Build a Euclidean vector from its coordinate function.

This is the small helper used throughout the file when reindexing vectors across context
concatenation and shape casts.
-/
def vecOfFun {n : Nat} (f : Fin n → ℝ) : Vec n :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm f

/-- Coordinates of `vecOfFun f` are the values of `f`. -/
@[simp] theorem vecOfFun_apply {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    vecOfFun (n := n) f i = f i := by
  simp [vecOfFun, EuclideanSpace.equiv]

/-- The same statement through the `WithLp` wrapper, which is the form `simp` meets in practice. -/
@[simp] theorem vecOfFun_ofLp {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    (vecOfFun (n := n) f).ofLp i = f i := by
  simp [vecOfFun, EuclideanSpace.equiv]

/-- Removing the `WithLp` wrapper from `vecOfFun` recovers its coordinate function. -/
theorem vecOfFun_ofLp_eq {n : Nat} (f : Fin n → ℝ) :
    (vecOfFun (n := n) f).ofLp = f := by
  funext i
  exact vecOfFun_ofLp f i

/-- Rebuilding a vector from its own coordinates changes nothing. -/
@[simp] theorem vecOfFun_eta {n : Nat} (v : Vec n) :
    vecOfFun (n := n) (fun i => v i) = v := by
  classical
  -- `vecOfFun` is `EuclideanSpace.equiv.symm`, and the forward map is definitionally `fun i => v
  -- i`.
  simp [vecOfFun, EuclideanSpace.equiv]

/-- The `ofLp` variant of eta. Both directions are needed because the reindexing lemmas below
sometimes produce a bare coordinate function and sometimes an unwrapped one. -/
@[simp] theorem vecOfFun_eta_ofLp {n : Nat} (v : Vec n) :
    vecOfFun (n := n) (fun i => v.ofLp i) = v := by
  classical
  simp [vecOfFun, EuclideanSpace.equiv]

/--
Flatten a typed context `TorchLean.TensorPack ℝ Γ` into one big Euclidean vector.

Unlike PyTorch's dynamically typed saved-tensor array, this is an actual typed isomorphism: shapes
are tracked in `Γ`, so the split points are definitional from `ctxSize`.
-/
def flattenCtx : {Γ : List Shape} → TorchLean.TensorPack ℝ Γ → CtxVec Γ
  | [], .nil => 0
  | s :: ss, .cons x xs =>
      vecOfFun (n := Spec.Shape.size s + ctxSize ss)
        (Fin.append (tensorToVec x) (flattenCtx (Γ := ss) xs))

/-- Inverse of `flattenCtx`: split a `CtxVec Γ` back into a `TorchLean.TensorPack ℝ Γ`. -/
def unflattenCtx : {Γ : List Shape} → CtxVec Γ → TorchLean.TensorPack ℝ Γ
  | [], _ => .nil
  | s :: ss, v =>
      let head : Vec (Spec.Shape.size s) :=
        vecOfFun (n := Spec.Shape.size s) fun i => v (Fin.castAdd (ctxSize ss) i)
      let tail : Vec (ctxSize ss) :=
        vecOfFun (n := ctxSize ss) fun i => v (Fin.natAdd (Spec.Shape.size s) i)
      .cons (vecToTensor (s := s) head) (unflattenCtx (Γ := ss) tail)

/-- Unflattening a flattened context recovers the original tensor pack. -/
@[simp] theorem unflattenCtx_flattenCtx {Γ : List Shape}
    (xs : TorchLean.TensorPack ℝ Γ) :
    unflattenCtx (Γ := Γ) (flattenCtx (Γ := Γ) xs) = xs := by
  induction Γ with
  | nil =>
      cases xs
      rfl
  | cons s ss ih =>
      cases xs with
      | cons x xs =>
          have hhead :
              vecOfFun (n := Spec.Shape.size s) (fun i =>
                (flattenCtx (Γ := s :: ss) (.cons x xs)) (Fin.castAdd (ctxSize ss) i)) =
                tensorToVec x := by
            rw [show flattenCtx (Γ := s :: ss) (.cons x xs) =
              vecOfFun (Fin.append (tensorToVec x) (flattenCtx xs)) by rfl]
            ext i
            simp
          have htail :
              vecOfFun (n := ctxSize ss) (fun i =>
                (flattenCtx (Γ := s :: ss) (.cons x xs))
                  (Fin.natAdd (Spec.Shape.size s) i)) =
                flattenCtx xs := by
            rw [show flattenCtx (Γ := s :: ss) (.cons x xs) =
              vecOfFun (Fin.append (tensorToVec x) (flattenCtx xs)) by rfl]
            ext i
            simp
          simp only [unflattenCtx]
          rw [hhead, vecToTensor_tensorToVec, htail, ih]

/-- And the reverse round trip, so a saved-tensor context and one flat gradient vector are the same
data. This is what makes it legitimate to differentiate the whole graph as a single map
`Vec (ctxSize Γ) → Vec n` and then read the result back per parameter. -/
@[simp] theorem flattenCtx_unflattenCtx {Γ : List Shape} (v : CtxVec Γ) :
    flattenCtx (Γ := Γ) (unflattenCtx (Γ := Γ) v) = v := by
  induction Γ with
  | nil =>
      ext i
      exact i.elim0
  | cons s ss ih =>
      let head : Vec (Spec.Shape.size s) :=
        vecOfFun fun i => v (Fin.castAdd (ctxSize ss) i)
      let tail : Vec (ctxSize ss) :=
        vecOfFun fun i => v (Fin.natAdd (Spec.Shape.size s) i)
      rw [show unflattenCtx (Γ := s :: ss) v =
        .cons (vecToTensor head) (unflattenCtx tail) by rfl]
      rw [show flattenCtx (Γ := s :: ss) (.cons (vecToTensor head) (unflattenCtx tail)) =
        vecOfFun (Fin.append (tensorToVec (vecToTensor head)) (flattenCtx (unflattenCtx tail))) by
          rfl]
      rw [tensorToVec_vecToTensor, ih]
      ext i
      rw [vecOfFun_apply]
      simpa only [head, tail, vecOfFun_ofLp_eq] using
        congrArg (fun f : Fin (Spec.Shape.size s + ctxSize ss) → ℝ => f i)
          (Fin.append_castAdd_natAdd (f := v) (m := Spec.Shape.size s) (n := ctxSize ss))

-- ---------------------------------------------------------------------------
-- Dot/inner agreement (`dotList` ↔ Euclidean inner product)
-- ---------------------------------------------------------------------------

/-- Cast a `Vec n` to `Vec m` along an equality, by reindexing coordinates. -/
def castVec {n m : Nat} (h : n = m) : Vec n → Vec m :=
  fun v => vecOfFun (n := m) fun i => v (Fin.cast h.symm i)

/-- Coordinates of a cast vector are the coordinates of the original at the cast index. -/
@[simp] theorem castVec_apply {n m : Nat} (h : n = m) (v : Vec n) (i : Fin m) :
    castVec (n := n) (m := m) h v i = v (Fin.cast h.symm i) := by
  simp [castVec]

/-- `castVec` reindexes the coordinate function stored under the `WithLp` wrapper. -/
@[simp] theorem castVec_ofLp {n m : Nat} (h : n = m) (v : Vec n) (i : Fin m) :
    (castVec (n := n) (m := m) h v).ofLp i = v.ofLp (Fin.cast h.symm i) := by
  simp [castVec]

/-- Casting along `rfl` is the identity. -/
@[simp] theorem castVec_rfl {n : Nat} (v : Vec n) : castVec (n := n) (m := n) rfl v = v := by
  ext i
  simp [castVec]

/-- `castVec` is additive. -/
@[simp] theorem castVec_add {n m : Nat} (h : n = m) (u v : Vec n) :
    castVec (n := n) (m := m) h (u + v) = castVec (n := n) (m := m) h u + castVec (n := n) (m := m)
      h v := by
  ext i
  simp []

/-- `castVec` commutes with scalar multiplication; with additivity this makes it linear, which is
what allows a cast to be pushed through a derivative without a separate argument each time. -/
@[simp] theorem castVec_smul {n m : Nat} (h : n = m) (r : ℝ) (v : Vec n) :
    castVec (n := n) (m := m) h (r • v) = r • castVec (n := n) (m := m) h v := by
  ext i
  simp [smul_eq_mul]

/-- Two casts compose into one along the composed equality. -/
@[simp] theorem castVec_castVec {n m k : Nat} (h₁ : n = m) (h₂ : m = k) (v : Vec n) :
    castVec h₂ (castVec h₁ v) = castVec (h₁.trans h₂) v := by
  cases h₁
  cases h₂
  ext i
  simp [castVec]

/--
`castVec` preserves the Euclidean inner product.

This is the core “cast isometry” lemma used throughout the vectorized graph development.
-/
theorem inner_castVec_castVec {n m : Nat} (h : n = m) (x y : Vec n) :
    inner ℝ (castVec h x) (castVec h y) = inner ℝ x y := by
  cases h
  simp [castVec]

/--
`sumSpec` over an outer dimension is a sum over slices.

This tensor-level “Fubini rule” is used to relate `Spec.dot` to Euclidean inner products after
vectorization.
-/
theorem sum_spec_dim {n : Nat} {s : Shape} (values : Fin n → Tensor ℝ s) :
    sumSpec (Tensor.dim values) = ∑ i : Fin n, sumSpec (values i) := by
  simpa only [get_dim] using
    (Spec.sum_spec_dim (t := Tensor.dim values))

-- Decompose `tensorToVec` along `finProdFinEquiv` when the inner size is positive.
/--
Coordinate characterization of `tensorToVec` on a tensor `.dim n s`.

Informally, the vectorization order is the standard product order induced by `finProdFinEquiv`.
-/
theorem tensorToVec_dim_apply {n : Nat} {s : Shape} (hmpos : 0 < Spec.Shape.size s)
    (f : Fin n → Tensor ℝ s) (p : Fin n × Fin (Spec.Shape.size s)) :
    tensorToVec (t := Tensor.dim f) (finProdFinEquiv p) = tensorToVec (t := f p.1) p.2 := by
  classical
  let m : Nat := Spec.Shape.size s
  have hmpos' : 0 < m := by
    dsimp [m]
    exact hmpos
  have hp2lt : p.2.val < m := by
    dsimp [m]
    exact p.2.isLt
  have hdiv : (p.2.val + m * p.1.val) / m = p.1.val := by
    calc
      (p.2.val + m * p.1.val) / m = p.2.val / m + p.1.val := Nat.add_mul_div_left p.2.val p.1.val
        hmpos'
      _ = p.1.val := by
        have : p.2.val / m = 0 := Nat.div_eq_of_lt hp2lt
        simp [this]
  have hmod : (p.2.val + m * p.1.val) % m = p.2.val := by
    calc
      (p.2.val + m * p.1.val) % m = p.2.val % m := Nat.add_mul_mod_self_left p.2.val m p.1.val
      _ = p.2.val := by
        simp [Nat.mod_eq_of_lt hp2lt]
  have houter : (p.2.val + m * p.1.val) / m < n := by
    simp [hdiv]
  have hi : (⟨(p.2.val + m * p.1.val) / m, houter⟩ : Fin n) = p.1 := by
    apply Fin.ext
    simp [hdiv]
  calc
    tensorToVec (t := Tensor.dim f) (finProdFinEquiv p) =
        TorchLean.Tensor.getScalar (flattenSpec (Tensor.dim f)) (finProdFinEquiv p) :=
      getScalarE_ofLp _ _
    _ = TorchLean.Tensor.getScalar (flattenSpec (f p.1)) p.2 := by
      have hidx :
          p.1.val * Spec.Shape.size s + p.2.val < n * Spec.Shape.size s := by
        simpa [finProdFinEquiv, Nat.mul_comm, Nat.add_comm] using
          (finProdFinEquiv p).isLt
      have hflat :
          finProdFinEquiv p =
            (⟨p.1.val * Spec.Shape.size s + p.2.val, hidx⟩ :
              Fin (n * Spec.Shape.size s)) := by
        apply Fin.ext
        simp [finProdFinEquiv, Nat.mul_comm, Nat.add_comm]
      rw [hflat]
      exact
        TorchLean.Tensor.ShapeChange.Internal.flattenSpec_dim_apply
          (values := f) (outer := p.1) (inner := p.2) hidx
    _ = tensorToVec (t := f p.1) p.2 := (getScalarE_ofLp _ _).symm

-- Inner product decomposition across an outer dimension.
/-- `tensorToVec` turns dot products on `.dim n s` into sums of Euclidean inner products over
slices. -/
theorem inner_tensorToVec_dim {n : Nat} {s : Shape} (a b : Fin n → Tensor ℝ s) :
    inner ℝ (tensorToVec (t := Tensor.dim a)) (tensorToVec (t := Tensor.dim b))
      =
    ∑ i : Fin n, inner ℝ (tensorToVec (t := a i)) (tensorToVec (t := b i)) := by
  classical
  by_cases hm : Spec.Shape.size s = 0
  · have hmul : n * Spec.Shape.size s = 0 := by simp [hm]
    -- LHS: transport to `Fin 0`.
    have hL :
        inner ℝ (tensorToVec (t := Tensor.dim a)) (tensorToVec (t := Tensor.dim b)) = 0 := by
      let e : Fin 0 ≃ Fin (n * Spec.Shape.size s) := Equiv.cast (congrArg Fin hmul.symm)
      calc
        inner ℝ (tensorToVec (t := Tensor.dim a)) (tensorToVec (t := Tensor.dim b))
            =
          ∑ i : Fin (n * Spec.Shape.size s),
            tensorToVec (t := Tensor.dim a) i * tensorToVec (t := Tensor.dim b) i := by
              rw [inner_eq_sum_mul]
              rfl
        _ =
          ∑ i : Fin 0,
            tensorToVec (t := Tensor.dim a) (e i) * tensorToVec (t := Tensor.dim b) (e i) := by
              simpa using
                (Equiv.sum_comp (e := e)
                  (g := fun i : Fin (n * Spec.Shape.size s) =>
                    tensorToVec (t := Tensor.dim a) i * tensorToVec (t := Tensor.dim b) i)).symm
        _ = 0 := by simp
    have hterm : ∀ i : Fin n, inner ℝ (tensorToVec (t := a i)) (tensorToVec (t := b i)) = 0 := by
      intro i
      let e : Fin 0 ≃ Fin (Spec.Shape.size s) := Equiv.cast (congrArg Fin hm.symm)
      calc
        inner ℝ (tensorToVec (t := a i)) (tensorToVec (t := b i))
            =
          ∑ j : Fin (Spec.Shape.size s),
            tensorToVec (t := a i) j * tensorToVec (t := b i) j := by
              simpa using
                inner_eq_sum_mul (x := tensorToVec (t := a i)) (y := tensorToVec (t := b i))
        _ =
          ∑ j : Fin 0,
            tensorToVec (t := a i) (e j) * tensorToVec (t := b i) (e j) := by
              simpa using
                (Equiv.sum_comp (e := e)
                  (g := fun j : Fin (Spec.Shape.size s) =>
                    tensorToVec (t := a i) j * tensorToVec (t := b i) j)).symm
        _ = 0 := by simp
    have hR :
        (∑ i : Fin n, inner ℝ (tensorToVec (t := a i)) (tensorToVec (t := b i))) = 0 := by
      simp [hterm]
    simp [hL, hR]
  · have hmpos : 0 < Spec.Shape.size s := Nat.pos_of_ne_zero hm
    calc
      inner ℝ (tensorToVec (t := Tensor.dim a)) (tensorToVec (t := Tensor.dim b))
          =
        ∑ i : Fin (n * Spec.Shape.size s),
          tensorToVec (t := Tensor.dim a) i * tensorToVec (t := Tensor.dim b) i := by
            rw [inner_eq_sum_mul]
            rfl
      _ =
        ∑ p : Fin n × Fin (Spec.Shape.size s),
          tensorToVec (t := Tensor.dim a) (finProdFinEquiv p) *
            tensorToVec (t := Tensor.dim b) (finProdFinEquiv p) := by
          simpa using
            (Equiv.sum_comp (e := finProdFinEquiv)
              (g := fun i : Fin (n * Spec.Shape.size s) =>
                tensorToVec (t := Tensor.dim a) i * tensorToVec (t := Tensor.dim b) i)).symm
      _ =
        ∑ p : Fin n × Fin (Spec.Shape.size s),
          tensorToVec (t := a p.1) p.2 * tensorToVec (t := b p.1) p.2 := by
          refine Finset.sum_congr rfl ?_
          intro p _
          simp [tensorToVec_dim_apply (hmpos := hmpos)]
      _ =
        ∑ i : Fin n, ∑ j : Fin (Spec.Shape.size s),
          tensorToVec (t := a i) j * tensorToVec (t := b i) j := by
          simp [Fintype.sum_prod_type]
      _ =
        ∑ i : Fin n, inner ℝ (tensorToVec (t := a i)) (tensorToVec (t := b i)) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          simpa using
            (inner_eq_sum_mul (x := tensorToVec (t := a i)) (y := tensorToVec (t := b i))).symm

-- Dot product agrees with the Euclidean inner product after vectorization.
/--
Main agreement lemma: tensor dot equals Euclidean inner product of vectorizations.

This is the bridge between `soundness.lean` (stated using `Spec.dot`) and the analytic theorems
here (stated using Euclidean `inner`).
-/
theorem dot_eq_inner_tensorToVec {s : Shape} (a b : Tensor ℝ s) :
    dot a b = inner ℝ (tensorToVec (t := a)) (tensorToVec (t := b)) := by
  classical
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item a, ← Tensor.scalar_item b]
      let vx : Vec 1 := tensorToVec (t := Tensor.scalar a.item)
      let vy : Vec 1 := tensorToVec (t := Tensor.scalar b.item)
      have hinner : inner ℝ vx vy = ∑ i : Fin 1, vx i * vy i :=
        inner_eq_sum_mul (x := vx) (y := vy)
      have hvx0 : vx.ofLp 0 = a.item := by
        change ((PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin 1 => ℝ)).symm
          (fun _ : Fin 1 => a.item)).ofLp 0 = a.item
        rfl
      have hvy0 : vy.ofLp 0 = b.item := by
        change ((PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin 1 => ℝ)).symm
          (fun _ : Fin 1 => b.item)).ofLp 0 = b.item
        rfl
      calc
        dot (Tensor.scalar a.item) (Tensor.scalar b.item) = a.item * b.item := by
          rw [Spec.dot_eq_tensorAlgebra_dot]
          simp [Proofs.TensorAlgebra.dot]
        _ = vx.ofLp 0 * vy.ofLp 0 := by simp [hvx0, hvy0]
        _ = inner ℝ vx vy := by
          simpa using hinner.symm
  | dim n s ih =>
      rw [← Tensor.dim_unstack a, ← Tensor.dim_unstack b]
      have hdot :
          dot (Tensor.dim (Tensor.unstack a)) (Tensor.dim (Tensor.unstack b)) =
            ∑ i : Fin n, dot (Tensor.unstack a i) (Tensor.unstack b i) := by
        calc
          dot (Tensor.dim (Tensor.unstack a)) (Tensor.dim (Tensor.unstack b)) =
              Proofs.TensorAlgebra.dot
                (Tensor.dim (Tensor.unstack a))
                (Tensor.dim (Tensor.unstack b)) :=
            Spec.dot_eq_tensorAlgebra_dot _ _
          _ = ∑ i : Fin n,
              Proofs.TensorAlgebra.dot (Tensor.unstack a i) (Tensor.unstack b i) := by
            simp only [Proofs.TensorAlgebra.dot, Tensor.unstack_dim]
            exact
              List.finRange_foldl_add_eq_finset_sum
                (fun i : Fin n =>
                  Proofs.TensorAlgebra.dot (Tensor.unstack a i) (Tensor.unstack b i))
          _ = ∑ i : Fin n, dot (Tensor.unstack a i) (Tensor.unstack b i) := by
            apply Finset.sum_congr rfl
            intro i _
            exact (Spec.dot_eq_tensorAlgebra_dot _ _).symm
      have hinter :
          inner ℝ
              (tensorToVec (t := Tensor.dim (Tensor.unstack a)))
              (tensorToVec (t := Tensor.dim (Tensor.unstack b))) =
            ∑ i : Fin n,
              inner ℝ
                (tensorToVec (t := Tensor.unstack a i))
                (tensorToVec (t := Tensor.unstack b i)) :=
        inner_tensorToVec_dim (a := Tensor.unstack a) (b := Tensor.unstack b)
      calc
        dot (Tensor.dim (Tensor.unstack a)) (Tensor.dim (Tensor.unstack b)) =
            ∑ i : Fin n,
              inner ℝ
                (tensorToVec (t := Tensor.unstack a i))
                (tensorToVec (t := Tensor.unstack b i)) := by
          refine hdot.trans ?_
          refine Finset.sum_congr rfl ?_
          intro i _
          simpa using
            (ih (a := Tensor.unstack a i) (b := Tensor.unstack b i))
        _ = inner ℝ
            (tensorToVec (t := Tensor.dim (Tensor.unstack a)))
            (tensorToVec (t := Tensor.dim (Tensor.unstack b))) := hinter.symm

/-- Concatenate two Euclidean vectors using `Fin.append`. -/
def appendVec {m n : Nat} (a : Vec m) (b : Vec n) : Vec (m + n) :=
  vecOfFun (n := m + n) (Fin.append a b)

/-- The empty context flattens to the zero vector of the zero-dimensional space. -/
@[simp] theorem flattenCtx_nil :
    flattenCtx (TorchLean.TensorPack.nil : TorchLean.TensorPack ℝ []) = 0 := rfl

/-- Flattening a `cons` concatenates the head tensor's coordinates in front of the tail's.

This is the layout convention the whole file depends on: parameters appear in context order, so a
gradient vector can be split back apart by `Fin.castAdd` / `Fin.natAdd` alone. -/
@[simp] theorem flattenCtx_cons {s : Shape} {ss : List Shape}
    (x : Tensor ℝ s) (xs : TorchLean.TensorPack ℝ ss) :
    flattenCtx (TorchLean.TensorPack.cons x xs) =
      appendVec (tensorToVec x) (flattenCtx xs) := rfl

/-- Left half of a concatenation reads from the first vector. -/
@[simp] theorem appendVec_ofLp_castAdd {m n : Nat} (a : Vec m) (b : Vec n) (i : Fin m) :
    (appendVec a b).ofLp (Fin.castAdd n i) = a.ofLp i := by
  simp [appendVec]

/-- Right half of a concatenation reads from the second vector. -/
@[simp] theorem appendVec_ofLp_natAdd {m n : Nat} (a : Vec m) (b : Vec n) (i : Fin n) :
    (appendVec a b).ofLp (Fin.natAdd m i) = b.ofLp i := by
  simp [appendVec]

/-- Reassociating concatenated vectors only changes their finite-index representation. -/
theorem castVec_appendVec_assoc {m n p : Nat} (a : Vec m) (b : Vec n) (c : Vec p) :
    castVec (Nat.add_assoc m n p) (appendVec (appendVec a b) c) =
      appendVec a (appendVec b c) := by
  apply PiLp.ext
  intro i
  simp only [castVec_ofLp, appendVec, vecOfFun_ofLp_eq]
  rw [Fin.append_assoc]
  rfl

/-- Casting the right block of a concatenation is the same as casting the full vector. -/
theorem appendVec_cast_right {m n p : Nat} (h : n = p) (a : Vec m) (b : Vec n) :
    appendVec a (castVec h b) =
      castVec (congrArg (m + ·) h) (appendVec a b) := by
  subst p
  simp

/-- Inner product of concatenated vectors splits as a sum of inner products. -/
theorem inner_append {m n : Nat} (a c : Vec m) (b d : Vec n) :
    inner ℝ (appendVec (m := m) (n := n) a b) (appendVec (m := m) (n := n) c d)
      =
    inner ℝ a c + inner ℝ b d := by
  classical
  have h0 :=
    inner_eq_sum_mul (x := appendVec (m := m) (n := n) a b)
      (y := appendVec (m := m) (n := n) c d)
  have hsum :
      (∑ i : Fin (m + n), (Fin.append a b i) * (Fin.append c d i))
        =
      ∑ s : Fin m ⊕ Fin n, (Fin.append a b (finSumFinEquiv s)) * (Fin.append c d (finSumFinEquiv s))
        := by
    simpa using
      (Equiv.sum_comp (e := finSumFinEquiv)
        (g := fun i : Fin (m + n) => (Fin.append a b i) * (Fin.append c d i))).symm
  let fSum : Fin m ⊕ Fin n → ℝ := fun s =>
    (Fin.append a b (finSumFinEquiv s)) * (Fin.append c d (finSumFinEquiv s))
  have hsplit : (∑ s : Fin m ⊕ Fin n, fSum s)
      = (∑ i : Fin m, fSum (Sum.inl i)) + (∑ j : Fin n, fSum (Sum.inr j)) := by
    simp [fSum]
  have hleft : (∑ i : Fin m, fSum (Sum.inl i)) = ∑ i : Fin m, a i * c i := by
    refine Finset.sum_congr rfl ?_
    intro i _
    simp [fSum, finSumFinEquiv_apply_left]
  have hright : (∑ j : Fin n, fSum (Sum.inr j)) = ∑ j : Fin n, b j * d j := by
    refine Finset.sum_congr rfl ?_
    intro j _
    simp [fSum, finSumFinEquiv_apply_right]
  calc
    inner ℝ (appendVec (m := m) (n := n) a b) (appendVec (m := m) (n := n) c d)
        = ∑ i : Fin (m + n), (Fin.append a b i) * (Fin.append c d i) := h0
    _ = ∑ s : Fin m ⊕ Fin n, fSum s := by
          simpa [fSum] using hsum
    _ = (∑ i : Fin m, a i * c i) + (∑ j : Fin n, b j * d j) := by
          simp [hsplit, hleft, hright]
    _ = inner ℝ a c + inner ℝ b d := by
          simp [inner_eq_sum_mul]

/--
`TensorPack.dotList` equals Euclidean inner product of `flattenCtx`.

This shows that the “context inner product” used in tape soundness is exactly the Euclidean inner
product on the vectorized context representation.
-/
theorem dotList_eq_inner_flattenCtx {Γ : List Shape}
    (x y : TorchLean.TensorPack ℝ Γ) :
    TensorPack.dotList (ss := Γ) x y = inner ℝ (flattenCtx (Γ := Γ) x) (flattenCtx (Γ := Γ) y) := by
  classical
  induction Γ with
  | nil =>
      cases x
      cases y
      simp [TensorPack.dotList, flattenCtx]
  | cons s ss ih =>
      cases x with
      | cons xh xt =>
          cases y with
          | cons yh yt =>
              -- Split the inner product across the append, then use the IH and
              -- `dot_eq_inner_tensorToVec`.
              have hinter :
                  inner ℝ (flattenCtx (Γ := s :: ss) (TorchLean.TensorPack.cons xh xt))
                        (flattenCtx (Γ := s :: ss) (TorchLean.TensorPack.cons yh yt))
                    =
                  inner ℝ (tensorToVec (t := xh)) (tensorToVec (t := yh))
                    + inner ℝ (flattenCtx (Γ := ss) xt) (flattenCtx (Γ := ss) yt) := by
                change
                  inner ℝ
                    (appendVec (m := Spec.Shape.size s) (n := ctxSize ss) (tensorToVec (t := xh))
                      (flattenCtx (Γ := ss) xt))
                    (appendVec (m := Spec.Shape.size s) (n := ctxSize ss) (tensorToVec (t := yh))
                      (flattenCtx (Γ := ss) yt))
                    =
                  inner ℝ (tensorToVec (t := xh)) (tensorToVec (t := yh))
                    + inner ℝ (flattenCtx (Γ := ss) xt) (flattenCtx (Γ := ss) yt)
                exact
                  inner_append (m := Spec.Shape.size s) (n := ctxSize ss)
                    (a := tensorToVec (t := xh)) (c := tensorToVec (t := yh))
                    (b := flattenCtx (Γ := ss) xt) (d := flattenCtx (Γ := ss) yt)
              calc
                TensorPack.dotList (ss := s :: ss) (TorchLean.TensorPack.cons xh xt)
                    (TorchLean.TensorPack.cons yh yt)
                    = dot xh yh + TensorPack.dotList (ss := ss) xt yt := by
                        simp [TensorPack.dotList]
                _ = inner ℝ (tensorToVec (t := xh)) (tensorToVec (t := yh))
                      + inner ℝ (flattenCtx (Γ := ss) xt) (flattenCtx (Γ := ss) yt) := by
                        simp [dot_eq_inner_tensorToVec, ih]
                _ = inner ℝ (flattenCtx (Γ := s :: ss) (TorchLean.TensorPack.cons xh xt))
                        (flattenCtx (Γ := s :: ss) (TorchLean.TensorPack.cons yh yt)) := by
                        exact hinter.symm

-- ---------------------------------------------------------------------------
-- Vector graph semantics (for calculus)
-- ---------------------------------------------------------------------------

/-- Cast a vectorized context along an equality of shape lists (reindexing coordinates). -/
def castCtxVec {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) : CtxVec Γ₁ → CtxVec Γ₂ :=
  castVec (congrArg ctxSize h)

/-- Casting a vectorized context along `rfl` is the identity. -/
@[simp] theorem castCtxVec_rfl {Γ : List Shape} (v : CtxVec Γ) :
    castCtxVec (Γ₁ := Γ) (Γ₂ := Γ) rfl v = v := by
  simp [castCtxVec]

/-- Context casts compose. Together with `castCtxVec_rfl` this keeps the casts introduced by graph
composition from piling up in the goal. -/
@[simp] theorem castCtxVec_cast {Γ₁ Γ₂ Γ₃ : List Shape} (h₁ : Γ₁ = Γ₂) (h₂ : Γ₂ = Γ₃)
    (v : CtxVec Γ₁) :
    castCtxVec (Γ₁ := Γ₂) (Γ₂ := Γ₃) h₂ (castCtxVec (Γ₁ := Γ₁) (Γ₂ := Γ₂) h₁ v)
      =
    castCtxVec (Γ₁ := Γ₁) (Γ₂ := Γ₃) (h₁.trans h₂) v := by
  cases h₁
  cases h₂
  simp [castCtxVec]

/-!
The next few lemmas are bookkeeping for splitting/concatenating vectorized contexts.
They are “obvious” from the list structure of `Γ`, but it is useful to expose them as named facts
so that the calculus proofs later can use them without redoing shape arithmetic.
-/

/-- `castCtxVec` is inner-product preserving (up to flipping the cast on the other argument). -/
theorem inner_castCtxVec {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) (x : CtxVec Γ₁) (y : CtxVec Γ₂) :
    inner ℝ (castCtxVec (Γ₁ := Γ₁) (Γ₂ := Γ₂) h x) y
      =
    inner ℝ x (castCtxVec (Γ₁ := Γ₂) (Γ₂ := Γ₁) h.symm y) := by
  cases h
  simp [castCtxVec, castVec]

/-- `ctxSize` respects list append (sizes add). -/
theorem ctxSize_append (Γ ss : List Shape) : ctxSize (Γ ++ ss) = ctxSize Γ + ctxSize ss := by
  induction Γ with
  | nil => simp [ctxSize]
  | cons s Γ ih => simp [ctxSize, ih, Nat.add_assoc]

/-- Specialized `ctxSize_append` for snoc (`Γ ++ [τ]`). -/
theorem ctxSize_snoc (ss : List Shape) (τ : Shape) :
    ctxSize (ss ++ [τ]) = ctxSize ss + Spec.Shape.size τ := by
  -- `ctxSize [τ] = Spec.Shape.size τ`.
  simp [ctxSize, ctxSize_append]

/-- Append one tensor-vector block to a vectorized context. -/
def snocCtx {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ) (t : Vec (Spec.Shape.size τ)) :
    CtxVec (Γ ++ [τ]) :=
  castVec (ctxSize_snoc Γ τ).symm (appendVec (m := ctxSize Γ) (n := Spec.Shape.size τ) ctx t)

/-- Prefixing a context vector commutes with appending its final tensor block. -/
theorem appendVec_snocCtx {s : Shape} {Γ : List Shape} {τ : Shape}
    (a : Vec s.size) (ctx : CtxVec Γ) (t : Vec τ.size) :
    appendVec a (snocCtx ctx t) =
      snocCtx (Γ := s :: Γ) (appendVec a ctx) t := by
  unfold snocCtx
  rw [appendVec_cast_right]
  rw [← castVec_appendVec_assoc]
  rw [castVec_castVec]
  congr 1

/-- Inverse of `snocCtx`: split `CtxVec (Γ ++ [τ])` into its prefix and last block. -/
def unsnocCtx {Γ : List Shape} {τ : Shape} (ctx : CtxVec (Γ ++ [τ])) :
    CtxVec Γ × Vec (Spec.Shape.size τ) :=
  let ctx' : Vec (ctxSize Γ + Spec.Shape.size τ) := castVec (ctxSize_snoc Γ τ) ctx
  let head : CtxVec Γ := vecOfFun (n := ctxSize Γ) fun i => ctx' (Fin.castAdd (Spec.Shape.size τ) i)
  let last : Vec (Spec.Shape.size τ) :=
    vecOfFun (n := Spec.Shape.size τ) fun i => ctx' (Fin.natAdd (ctxSize Γ) i)
  (head, last)

/-- `unsnocCtx (snocCtx ctx t) = (ctx, t)`. -/
theorem unsnocCtx_snocCtx {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ)
    (t : Vec (Spec.Shape.size τ)) :
    unsnocCtx (Γ := Γ) (τ := τ) (snocCtx (Γ := Γ) (τ := τ) ctx t) = (ctx, t) := by
  classical
  simp [unsnocCtx, snocCtx, appendVec, Fin.append_left, Fin.append_right]

/-- `snocCtx (unsnocCtx ctx) = ctx`. -/
theorem snocCtx_unsnocCtx {Γ : List Shape} {τ : Shape} (ctx : CtxVec (Γ ++ [τ])) :
    snocCtx (Γ := Γ) (τ := τ) (unsnocCtx (Γ := Γ) (τ := τ) ctx).1 (unsnocCtx (Γ := Γ) (τ := τ)
      ctx).2 = ctx := by
  classical
  -- move to the `(ctxSize Γ + size τ)` representation
  have hcancel :
      castVec (ctxSize_snoc Γ τ).symm (castVec (ctxSize_snoc Γ τ) ctx) = ctx := by
    simp [castVec_castVec]
  -- reconstruct by `appendVec` on the `(ctxSize Γ + size τ)` representation.
  have happ :
      appendVec (m := ctxSize Γ) (n := Spec.Shape.size τ)
          (vecOfFun (n := ctxSize Γ) fun i => (castVec (ctxSize_snoc Γ τ) ctx) (Fin.castAdd
            (Spec.Shape.size τ) i))
          (vecOfFun (n := Spec.Shape.size τ) fun i => (castVec (ctxSize_snoc Γ τ) ctx) (Fin.natAdd
            (ctxSize Γ) i))
        =
      castVec (ctxSize_snoc Γ τ) ctx := by
    ext i
    simp [appendVec, Fin.append, Fin.addCases, vecOfFun]
  -- Apply the cast back to `(ctxSize Γ + size τ)` and then cancel the cast pair.
  simpa [unsnocCtx, snocCtx, hcancel, happ] using
    (congrArg (castVec (ctxSize_snoc Γ τ).symm) happ).trans hcancel

namespace Node

/-- Vectorized forward map of a tape `Node`: `CtxVec Γ → Vec (Spec.Shape.size τ)`. -/
def forwardVec {Γ : List Shape} {τ : Shape} (node : Node Γ τ) :
    CtxVec Γ → Vec (Spec.Shape.size τ) :=
  fun ctxV => tensorToVec (t := node.forward (unflattenCtx (Γ := Γ) ctxV))

/-- Vectorized JVP of a tape `Node`: the node-level forward-mode action on tangents. -/
def jvpVec {Γ : List Shape} {τ : Shape} (node : Node Γ τ) :
    CtxVec Γ → CtxVec Γ → Vec (Spec.Shape.size τ) :=
  fun ctxV dctxV =>
    tensorToVec (t := node.jvp (unflattenCtx (Γ := Γ) ctxV) (unflattenCtx (Γ := Γ) dctxV))

/-- Vectorized VJP of a tape `Node`: pushes a cotangent vector back to the input context. -/
def vjpVec {Γ : List Shape} {τ : Shape} (node : Node Γ τ) :
    CtxVec Γ → Vec (Spec.Shape.size τ) → CtxVec Γ :=
  fun ctxV δV =>
    flattenCtx (Γ := Γ) (node.vjp (unflattenCtx (Γ := Γ) ctxV) (vecToTensor (s := τ) δV))

/--
Vectorized form of `Node.correct` (adjointness law).

Statement: `⟪jvp(x,dx), δ⟫ = ⟪dx, vjp(x,δ)⟫`.
-/
theorem correct_inner {Γ : List Shape} {τ : Shape} (node : Node Γ τ) :
    ∀ (ctxV dctxV : CtxVec Γ) (δV : Vec (Spec.Shape.size τ)),
      inner ℝ (node.jvpVec ctxV dctxV) δV = inner ℝ dctxV (node.vjpVec ctxV δV) := by
  intro ctxV dctxV δV
  let ctx := unflattenCtx (Γ := Γ) ctxV
  let dctx := unflattenCtx (Γ := Γ) dctxV
  let δ := vecToTensor (s := τ) δV
  have hdot := node.correct ctx dctx δ
  -- Convert `dot`/`dotList` to inner products.
  have hleft : dot (node.jvp ctx dctx) δ = inner ℝ (node.jvpVec ctxV dctxV) δV := by
    simp [Node.jvpVec, ctx, dctx, δ, dot_eq_inner_tensorToVec]
  have hright :
      TensorPack.dotList (ss := Γ) dctx (node.vjp ctx δ) = inner ℝ dctxV (node.vjpVec ctxV δV) := by
    simpa [Node.vjpVec, ctx, dctx, δ] using
      (dotList_eq_inner_flattenCtx (Γ := Γ) (x := dctx) (y := node.vjp ctx δ))
  -- Finish.
  simpa [hleft, hright] using hdot

end Node

-- ---------------------------------------------------------------------------
-- Graph semantics on Euclidean contexts
-- ---------------------------------------------------------------------------

namespace Graph

variable {Γ : List Shape}

/--
Vectorized evaluation of a tape `Graph`.

Returns a `CtxVec (Γ ++ ss)` containing the original inputs and all intermediate node outputs.
-/
def evalVec {ss : List Shape} (g : Graph Γ ss) (xV : CtxVec Γ) : CtxVec (Γ ++ ss) :=
  match g with
  | .nil =>
      castCtxVec (Γ₁ := Γ) (Γ₂ := Γ ++ []) (List.append_nil Γ).symm xV
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctxV : CtxVec (Γ ++ ss) := evalVec (ss := ss) g xV
      let yV : Vec (Spec.Shape.size τ) := node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV
      castCtxVec (Γ₁ := (Γ ++ ss) ++ [τ]) (Γ₂ := Γ ++ (ss ++ [τ]))
        (List.append_assoc Γ ss [τ])
        (snocCtx (Γ := (Γ ++ ss)) (τ := τ) ctxV yV)

/-- Vectorized JVP for a whole graph: forward-mode derivative of `evalVec`. -/
def jvpVec {ss : List Shape} (g : Graph Γ ss) (xV dxV : CtxVec Γ) : CtxVec (Γ ++ ss) :=
  match g with
  | .nil =>
      castCtxVec (Γ₁ := Γ) (Γ₂ := Γ ++ []) (List.append_nil Γ).symm dxV
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctxV : CtxVec (Γ ++ ss) := evalVec (ss := ss) g xV
      let dctxV : CtxVec (Γ ++ ss) := jvpVec (ss := ss) g xV dxV
      let dyV : Vec (Spec.Shape.size τ) := node.jvpVec (Γ := Γ ++ ss) (τ := τ) ctxV dctxV
      castCtxVec (Γ₁ := (Γ ++ ss) ++ [τ]) (Γ₂ := Γ ++ (ss ++ [τ]))
        (List.append_assoc Γ ss [τ])
        (snocCtx (Γ := (Γ ++ ss)) (τ := τ) dctxV dyV)

/--
Vectorized reverse-mode accumulation (VJP) for a whole graph.

`seedV` is a cotangent for the entire `Γ ++ ss` context (inputs plus intermediates), matching the
global tape soundness theorem.
-/
def backpropVec {ss : List Shape} (g : Graph Γ ss) (xV : CtxVec Γ) (seedV : CtxVec (Γ ++ ss)) :
  CtxVec Γ :=
  match g with
  | .nil =>
      castCtxVec (Γ₁ := Γ ++ []) (Γ₂ := Γ) (List.append_nil Γ) seedV
  | .snoc (ss := ss) (τ := τ) g node =>
      let assoc := List.append_assoc Γ ss [τ]
      let seedV' : CtxVec ((Γ ++ ss) ++ [τ]) :=
        castCtxVec (Γ₁ := Γ ++ (ss ++ [τ])) (Γ₂ := (Γ ++ ss) ++ [τ]) assoc.symm seedV
      let seedPrevV : CtxVec (Γ ++ ss) := (unsnocCtx (Γ := (Γ ++ ss)) (τ := τ) seedV').1
      let seedOutV : Vec (Spec.Shape.size τ) := (unsnocCtx (Γ := (Γ ++ ss)) (τ := τ) seedV').2
      let ctxV : CtxVec (Γ ++ ss) := evalVec (ss := ss) g xV
      let contribV : CtxVec (Γ ++ ss) := node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV seedOutV
      backpropVec (ss := ss) g xV (seedPrevV + contribV)

end Graph

-- ---------------------------------------------------------------------------
-- Vectorization transport: `flattenCtx` commutes with the context operations
-- ---------------------------------------------------------------------------

/-- `flattenCtx` commutes with casting a context along a shape-list equality. -/
theorem flattenCtx_cast {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) (xs : TorchLean.TensorPack ℝ Γ₁) :
    flattenCtx (Γ := Γ₂) (TorchLean.TensorPack.cast h xs) =
      castCtxVec (Γ₁ := Γ₁) (Γ₂ := Γ₂) h (flattenCtx xs) := by
  cases h
  simp

/-- Vectorization is additive: `tensorToVec` maps `addSpec` to vector addition. -/
theorem tensorToVec_addSpec {s : Shape} (a b : Tensor ℝ s) :
    tensorToVec (t := addSpec a b) = tensorToVec (t := a) + tensorToVec (t := b) := by
  refine ext_inner_right ℝ ?_
  intro w
  have hw : w = tensorToVec (t := vecToTensor (s := s) w) :=
    (tensorToVec_vecToTensor (s := s) w).symm
  rw [hw, ← dot_eq_inner_tensorToVec, inner_add_left, ← dot_eq_inner_tensorToVec,
    ← dot_eq_inner_tensorToVec, dot_add_left]

/-- The context inner product is additive in its left argument. -/
private theorem dotList_add_left' {Γ : List Shape} (u v z : TorchLean.TensorPack ℝ Γ) :
    TensorPack.dotList (TorchLean.TensorPack.add u v) z =
      TensorPack.dotList u z + TensorPack.dotList v z := by
  induction Γ with
  | nil =>
    cases u
    cases v
    cases z
    simp [TensorPack.dotList, TorchLean.TensorPack.add]
  | cons s Γ ih =>
    cases u with
    | cons uh ut =>
      cases v with
      | cons vh vt =>
        cases z with
        | cons zh zt =>
          show TensorPack.dotList
              (TorchLean.TensorPack.cons (addSpec uh vh) (TorchLean.TensorPack.add ut vt))
              (TorchLean.TensorPack.cons zh zt)
              = _
          simp only [TensorPack.dotList, dot_add_left, ih]
          ring

/-- `flattenCtx` maps context addition to vector addition. -/
theorem flattenCtx_add {Γ : List Shape} (u v : TorchLean.TensorPack ℝ Γ) :
    flattenCtx (TorchLean.TensorPack.add u v) = flattenCtx u + flattenCtx v := by
  refine ext_inner_right ℝ ?_
  intro w
  have hw : w = flattenCtx (unflattenCtx (Γ := Γ) w) :=
    (flattenCtx_unflattenCtx (Γ := Γ) w).symm
  rw [hw, ← dotList_eq_inner_flattenCtx, inner_add_left, ← dotList_eq_inner_flattenCtx,
    ← dotList_eq_inner_flattenCtx]
  exact dotList_add_left' u v _

/-- `flattenCtx` maps `TorchLean.TensorPack.snoc` to `snocCtx`. -/
theorem flattenCtx_snoc {Γ : List Shape} {τ : Shape} (xs : TorchLean.TensorPack ℝ Γ)
    (y : Tensor ℝ τ) :
    flattenCtx (Γ := Γ ++ [τ]) (TorchLean.TensorPack.snoc xs y)
      = snocCtx (Γ := Γ) (τ := τ) (flattenCtx xs) (tensorToVec (t := y)) := by
  induction Γ with
  | nil =>
    cases xs
    change flattenCtx (TorchLean.TensorPack.cons y TorchLean.TensorPack.nil) = _
    rw [flattenCtx_cons, flattenCtx_nil]
    let h : ctxSize [] + τ.size = τ.size + ctxSize [] := by simp [ctxSize]
    change appendVec (tensorToVec y) 0 = castVec h (appendVec 0 (tensorToVec y))
    apply PiLp.ext
    intro i
    induction i using Fin.addCases with
    | left i =>
      rw [appendVec_ofLp_castAdd, castVec_ofLp]
      have hidx :
          Fin.cast h.symm (Fin.castAdd (ctxSize []) i) = Fin.natAdd (ctxSize []) i := by
        apply Fin.ext
        simpa only [ctxSize, Fin.val_natAdd, Nat.zero_add, Fin.val_castAdd] using
          Fin.val_cast h.symm (Fin.castAdd (ctxSize []) i)
      rw [hidx, appendVec_ofLp_natAdd]
    | right i => exact i.elim0
  | cons s Γ ih =>
    cases xs with
    | cons x xs =>
      change flattenCtx (TorchLean.TensorPack.cons x (TorchLean.TensorPack.snoc xs y)) = _
      rw [flattenCtx_cons, ih, flattenCtx_cons, appendVec_snocCtx]

/-- `flattenCtx` maps `TorchLean.TensorPack.unsnoc` to `unsnocCtx`. -/
theorem unsnocCtx_flattenCtx {Γ : List Shape} {τ : Shape} (w : TorchLean.TensorPack ℝ (Γ ++ [τ])) :
    unsnocCtx (Γ := Γ) (τ := τ) (flattenCtx w)
      = (flattenCtx (TorchLean.TensorPack.unsnoc w).1,
          tensorToVec (t := (TorchLean.TensorPack.unsnoc w).2)) := by
  conv_lhs => rw [← TorchLean.TensorPack.snoc_unsnoc (α := ℝ) (ss := Γ) (τ := τ) (xs := w)]
  rw [flattenCtx_snoc, unsnocCtx_snocCtx]

namespace Node

/-- The vectorized forward map, evaluated on a flattened context. -/
theorem forwardVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ)
    (x : TorchLean.TensorPack ℝ Γ) :
    node.forwardVec (Γ := Γ) (τ := τ) (flattenCtx x) = tensorToVec (t := node.forward x) := by
  simp [Node.forwardVec]

/-- The vectorized JVP, evaluated on flattened contexts. -/
theorem jvpVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ)
    (x dx : TorchLean.TensorPack ℝ Γ) :
    node.jvpVec (Γ := Γ) (τ := τ) (flattenCtx x) (flattenCtx dx)
      = tensorToVec (t := node.jvp x dx) := by
  simp [Node.jvpVec]

/-- The vectorized VJP, evaluated on a flattened context and a vectorized cotangent. -/
theorem vjpVec_flattenCtx {Γ : List Shape} {τ : Shape} (node : Node Γ τ)
    (x : TorchLean.TensorPack ℝ Γ) (δ : Tensor ℝ τ) :
    node.vjpVec (Γ := Γ) (τ := τ) (flattenCtx x) (tensorToVec (t := δ))
      = flattenCtx (node.vjp x δ) := by
  simp [Node.vjpVec]

end Node

namespace Graph

variable {Γ : List Shape}

/-- The Euclidean graph evaluation is the flattening of the `TensorPack` evaluation. -/
theorem evalVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ) :
    evalVec (Γ := Γ) (ss := ss) g (flattenCtx x) = flattenCtx (eval (Γ := Γ) (ss := ss) g x) := by
  induction g with
  | nil =>
    simp [evalVec, eval, flattenCtx_cast]
  | snoc g node ih =>
    simp [evalVec, eval, flattenCtx_cast, flattenCtx_snoc, ih, Node.forwardVec_flattenCtx]

/-- The Euclidean graph JVP is the flattening of the `TensorPack` JVP. -/
theorem jvpVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x dx : TorchLean.TensorPack ℝ Γ) :
    jvpVec (Γ := Γ) (ss := ss) g (flattenCtx x) (flattenCtx dx)
      = flattenCtx (jvpCtx (Γ := Γ) (ss := ss) g x dx) := by
  induction g with
  | nil =>
    simp [jvpVec, jvpCtx, flattenCtx_cast]
  | snoc g node ih =>
    simp [jvpVec, jvpCtx, flattenCtx_cast, flattenCtx_snoc, ih, evalVec_flattenCtx,
      Node.jvpVec_flattenCtx]

/-- The Euclidean reverse pass is the flattening of the `TensorPack` reverse pass. -/
theorem backpropVec_flattenCtx {ss : List Shape} (g : Graph Γ ss) (x : TorchLean.TensorPack ℝ Γ)
    (seed : TorchLean.TensorPack ℝ (Γ ++ ss)) :
    backpropVec (Γ := Γ) (ss := ss) g (flattenCtx x) (flattenCtx seed)
      = flattenCtx (backpropCtx (Γ := Γ) (ss := ss) g x seed) := by
  induction g with
  | nil =>
    simp [backpropVec, backpropCtx, flattenCtx_cast]
  | snoc g node ih =>
    rename_i ss τ
    simp only [backpropVec, backpropCtx]
    rw [← flattenCtx_cast, unsnocCtx_flattenCtx, evalVec_flattenCtx,
      Node.vjpVec_flattenCtx, ← flattenCtx_add, ih]

end Graph

namespace Graph

variable {Γ : List Shape}

/-- Vectorized tape soundness: `⟪jvp, seed⟫ = ⟪dx, backprop seed⟫`. It is `backprop_correct`
read through `flattenCtx`, a bijection carrying `dotList` to the Euclidean inner product. -/
theorem backprop_correct_inner {ss : List Shape} (g : Graph Γ ss) :
    ∀ xV dxV seedV,
      inner ℝ (jvpVec (Γ := Γ) (ss := ss) g xV dxV) seedV =
        inner ℝ dxV (backpropVec (Γ := Γ) (ss := ss) g xV seedV) := by
  intro xV dxV seedV
  obtain ⟨x, rfl⟩ : ∃ x, flattenCtx x = xV := ⟨_, flattenCtx_unflattenCtx xV⟩
  obtain ⟨dx, rfl⟩ : ∃ dx, flattenCtx dx = dxV := ⟨_, flattenCtx_unflattenCtx dxV⟩
  obtain ⟨seed, rfl⟩ : ∃ seed, flattenCtx seed = seedV := ⟨_, flattenCtx_unflattenCtx seedV⟩
  rw [jvpVec_flattenCtx, backpropVec_flattenCtx, ← dotList_eq_inner_flattenCtx,
    ← dotList_eq_inner_flattenCtx]
  exact backprop_correct g x dx seed

end Graph

-- ---------------------------------------------------------------------------
-- Analytic upgrade: `jvp = fderiv`, `backprop = (fderiv eval)†`
-- ---------------------------------------------------------------------------

/--
Per-node analytic correctness assumption: JVP is the Fréchet derivative.

This is the hypothesis that upgrades the dot-level soundness theorem into an `fderiv` statement.
-/
structure NodeFDerivCorrect {Γ : List Shape} {τ : Shape} (node : Node Γ τ) where
  /-- The derivative packaged as a continuous linear map. -/
  deriv : CtxVec Γ → (CtxVec Γ →L[ℝ] Vec (Spec.Shape.size τ))
  /-- The forward map has the above derivative everywhere. -/
  hasFDerivAt : ∀ xV, HasFDerivAt (node.forwardVec (Γ := Γ) (τ := τ)) (deriv xV) xV
  /-- The node's JVP function agrees with the packaged derivative. -/
  jvp_eq : ∀ xV dxV, node.jvpVec (Γ := Γ) (τ := τ) xV dxV = (deriv xV) dxV

/-- Graph predicate: every node satisfies `NodeFDerivCorrect`. -/
def GraphFDerivCorrect {Γ : List Shape} : ∀ {ss : List Shape}, Graph Γ ss → Type
  | _, .nil => PUnit
  | _, .snoc g node => GraphFDerivCorrect g × NodeFDerivCorrect node

-- ---------------------------------------------------------------------------
-- Pointwise analytic upgrade: allow per-node assumptions (ReLU kinks, log domain, …)
-- ---------------------------------------------------------------------------

/--
Pointwise per-node analytic correctness.

Used when a node is only differentiable under side conditions at a particular basepoint `xV`
(e.g. `inv`, `sqrt`, `log`, or piecewise ops).
-/
structure NodeFDerivCorrectAt {Γ : List Shape} {τ : Shape} (node : Node Γ τ) (xV : CtxVec Γ) where
  /-- The Fréchet derivative at `xV`, as a continuous linear map on the flattened context. -/
  deriv : CtxVec Γ →L[ℝ] Vec (Spec.Shape.size τ)
  /-- `deriv` really is the derivative of the node's forward pass at `xV`. -/
  hasFDerivAt : HasFDerivAt (node.forwardVec (Γ := Γ) (τ := τ)) deriv xV
  /-- The node's hand-written JVP agrees with the analytic derivative. This is the field that turns
  an analysis fact into a statement about the code that actually runs. -/
  jvp_eq : ∀ dxV, node.jvpVec (Γ := Γ) (τ := τ) xV dxV = deriv dxV

/--
Specialize a global `NodeFDerivCorrect` proof to a particular basepoint.

This is the common “turn an everywhere-differentiable node into a pointwise differentiable node”
adapter used when assembling `GraphFDerivCorrectAt` proofs.
-/
def NodeFDerivCorrect.at {Γ : List Shape} {τ : Shape} {node : Node Γ τ}
    (hn : NodeFDerivCorrect node) (xV : CtxVec Γ) : NodeFDerivCorrectAt node xV :=
  ⟨hn.deriv xV, hn.hasFDerivAt xV, fun dxV => hn.jvp_eq xV dxV⟩

/--
Pointwise graph predicate: every node is differentiable at the *actual* intermediate values.

Note the recursion uses `Graph.evalVec` to compute the basepoint for each successive node.
-/
def GraphFDerivCorrectAt {Γ : List Shape} : ∀ {ss : List Shape}, Graph Γ ss → CtxVec Γ → Type
  | _, .nil => fun _ => PUnit
  | _, .snoc g node => fun xV => GraphFDerivCorrectAt g xV × NodeFDerivCorrectAt node (Graph.evalVec
    (Γ := Γ) g xV)

namespace Graph

variable {Γ : List Shape}

-- A linear map wrapper for `Fin.append` on Euclidean vectors.
/-- `Fin.append` packaged as a continuous linear map on Euclidean vectors. -/
def appendCLM (m n : Nat) : (Vec m × Vec n) →L[ℝ] Vec (m + n) := by
  classical
  let fLin : (Vec m × Vec n) →ₗ[ℝ] Vec (m + n) :=
    { toFun := fun p => appendVec (m := m) (n := n) p.1 p.2
      map_add' := by
        intro p q
        ext i
        cases i using Fin.addCases <;> simp [appendVec, Fin.append, Fin.addCases]
      map_smul' := by
        intro r p
        ext i
        cases i using Fin.addCases <;> simp [appendVec, Fin.append, Fin.addCases, Prod.smul_fst,
          Prod.smul_snd] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

-- Reindexing a vector along an equality of dimensions.
/-- `castVec` packaged as a continuous linear map (finite-dimensional, hence continuous). -/
def castCLM {n m : Nat} (h : n = m) : Vec n →L[ℝ] Vec m := by
  classical
  let fLin : Vec n →ₗ[ℝ] Vec m :=
    { toFun := castVec h
      map_add' := by
        intro x y
        ext i
        simp [castVec]
      map_smul' := by
        intro r x
        ext i
        simp [castVec] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

-- The CLM for `snocCtx` viewed as a function on pairs.
/-- Continuous linear map version of `snocCtx` (concatenation + cast). -/
def snocCLM {Γ : List Shape} {τ : Shape} :
    (CtxVec Γ × Vec (Spec.Shape.size τ)) →L[ℝ] CtxVec (Γ ++ [τ]) :=
  (castCLM (h := (ctxSize_snoc Γ τ).symm)).comp
    (appendCLM (m := ctxSize Γ) (n := Spec.Shape.size τ))

-- Main analytic statement: `HasFDerivAt` for `evalVec` and identification of `jvpVec`.
/--
Main induction: `evalVec` is differentiable and its derivative agrees with `jvpVec`.

This is the technical heart of the `jvp = fderiv` theorem.
-/
theorem hasFDerivAt_evalVec_and_jvp
    {ss : List Shape} (g : Graph Γ ss)
    (hg : GraphFDerivCorrect (Γ := Γ) g) :
    ∀ xV : CtxVec Γ,
      ∃ D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ ss),
        HasFDerivAt (evalVec (Γ := Γ) (ss := ss) g) D xV
          ∧
        (∀ dxV : CtxVec Γ, jvpVec (Γ := Γ) (ss := ss) g xV dxV = D dxV) := by
  classical
  induction g with
  | nil =>
      intro xV
      -- `evalVec`/`jvpVec` are just casts along `append_nil`.
      let h : ctxSize Γ = ctxSize (Γ ++ []) := congrArg ctxSize (List.append_nil Γ).symm
      let D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ []) := castCLM (h := h)
      refine ⟨D, ?_, ?_⟩
      · -- `evalVec` is the CLM itself.
        change HasFDerivAt (castVec h) D xV
        exact D.hasFDerivAt
      · intro dxV
        simp [Graph.jvpVec, castCtxVec, castVec, castCLM, D]
  | snoc g node ih =>
      intro xV
      rename_i ss τ
      rcases hg with ⟨hg_g, hg_node⟩
      -- IH for the prefix graph.
      rcases ih (hg := hg_g) xV with ⟨Dg, hDg, hJg⟩
      let ctxV : CtxVec (Γ ++ ss) := evalVec (Γ := Γ) (ss := ss) g xV
      -- node derivative at the vectorized context.
      let Dn : CtxVec (Γ ++ ss) →L[ℝ] Vec (Spec.Shape.size τ) := hg_node.deriv ctxV
      have hnode : HasFDerivAt (node.forwardVec (Γ := Γ ++ ss) (τ := τ)) Dn ctxV :=
        hg_node.hasFDerivAt ctxV
      -- derivative for the output component `yV`.
      have hy :
          HasFDerivAt
            (fun xV : CtxVec Γ => node.forwardVec (Γ := Γ ++ ss) (τ := τ) (evalVec (Γ := Γ) (ss :=
              ss) g xV))
            (Dn.comp Dg) xV := by
        change HasFDerivAt
          ((node.forwardVec (Γ := Γ ++ ss) (τ := τ)) ∘
            (evalVec (Γ := Γ) (ss := ss) g))
          (Dn.comp Dg) xV
        simpa [ctxV, Function.comp] using (hnode.comp xV hDg)
      -- pair derivative: `(ctxV, yV)`.
      have hpair :
          HasFDerivAt
            (fun xV : CtxVec Γ =>
              (evalVec (Γ := Γ) (ss := ss) g xV,
                node.forwardVec (Γ := Γ ++ ss) (τ := τ) (evalVec (Γ := Γ) (ss := ss) g xV)))
            (Dg.prod (Dn.comp Dg)) xV :=
        hDg.prodMk hy
      -- the `snocCtx` CLM and the assoc cast.
      let assoc := List.append_assoc Γ ss [τ]
      let hAssoc : ctxSize ((Γ ++ ss) ++ [τ]) = ctxSize (Γ ++ (ss ++ [τ])) := congrArg ctxSize assoc
      let Dcast : CtxVec ((Γ ++ ss) ++ [τ]) →L[ℝ] CtxVec (Γ ++ (ss ++ [τ])) := castCLM (h := hAssoc)
      let D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ (ss ++ [τ])) := (Dcast.comp (snocCLM (Γ := Γ ++ ss) (τ :=
        τ))).comp (Dg.prod (Dn.comp Dg))
      refine ⟨D, ?_, ?_⟩
      · -- `HasFDerivAt` for the composed graph evaluation.
        have hsnoc :
            HasFDerivAt
              (fun p : CtxVec (Γ ++ ss) × Vec (Spec.Shape.size τ) =>
                snocCtx (Γ := Γ ++ ss) (τ := τ) p.1 p.2)
              (snocCLM (Γ := Γ ++ ss) (τ := τ)) (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV)
                := by
          change HasFDerivAt
            (snocCLM (Γ := Γ ++ ss) (τ := τ))
            (snocCLM (Γ := Γ ++ ss) (τ := τ))
            (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV)
          exact (snocCLM (Γ := Γ ++ ss) (τ := τ)).hasFDerivAt
        have hcomp1 :=
          (Dcast.hasFDerivAt (x := snocCtx (Γ := Γ ++ ss) (τ := τ) ctxV (node.forwardVec (Γ := Γ ++
            ss) (τ := τ) ctxV))).comp xV
            (hsnoc.comp xV hpair)
        -- `evalVec` is definitionally this composition.
        change HasFDerivAt
          ((castCLM hAssoc) ∘
            (fun p : CtxVec (Γ ++ ss) × Vec (Spec.Shape.size τ) =>
              snocCtx (Γ := Γ ++ ss) (τ := τ) p.1 p.2) ∘
            fun xV : CtxVec Γ =>
              (evalVec (Γ := Γ) (ss := ss) g xV,
                node.forwardVec (Γ := Γ ++ ss) (τ := τ) (evalVec (Γ := Γ) (ss := ss) g xV)))
          D xV
        simpa [ctxV, D, Dcast, snocCLM, Function.comp, ContinuousLinearMap.comp_assoc] using hcomp1
      · intro dxV
        -- identify the JVP with the derivative application
        have hx : jvpVec (Γ := Γ) (ss := ss) g xV dxV = Dg dxV := hJg dxV
        have hy' :
            node.jvpVec (Γ := Γ ++ ss) (τ := τ) ctxV (Dg dxV) = (Dn.comp Dg) dxV := by
          simpa [Dn, ContinuousLinearMap.comp_apply] using (hg_node.jvp_eq ctxV (Dg dxV))
        -- unfold the graph JVP and simplify pointwise through casts/append.
        ext i
        simp [Graph.jvpVec, ctxV, hx, hy', D, Dcast, snocCLM, snocCtx, castCtxVec, castVec, castCLM,
          appendCLM,
          ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply]

/-!
Convenience corollaries:

Once we have `HasFDerivAt evalVec = jvpVec`, the rest are immediate:
`jvpVec = fderiv`, then `backpropVec = (fderiv evalVec)†` by the inner-product characterization of
  adjoints.
-/

/-- Under `GraphFDerivCorrect`, the graph JVP equals the Fréchet derivative `fderiv` of `evalVec`.
  -/
theorem jvpVec_eq_fderiv
    {ss : List Shape} (g : Graph Γ ss) (hg : GraphFDerivCorrect (Γ := Γ) g) :
    ∀ xV dxV,
      jvpVec (Γ := Γ) (ss := ss) g xV dxV = (fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV) dxV := by
  intro xV dxV
  rcases hasFDerivAt_evalVec_and_jvp (Γ := Γ) (ss := ss) (g := g) hg xV with ⟨D, hD, hJ⟩
  have hfderiv : fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV = D := by
    simpa using hD.fderiv
  simpa [hfderiv] using hJ dxV

/--
Main analytic theorem: `backpropVec` equals the adjoint of the derivative of `evalVec`.

This is the proof-level formalization of “reverse-mode computes a VJP”, stated as an equality of
linear maps in a Euclidean space.
-/
theorem backpropVec_eq_adjoint_fderiv
    {ss : List Shape} (g : Graph Γ ss) (hg : GraphFDerivCorrect (Γ := Γ) g) :
    ∀ (xV : CtxVec Γ) (seedV : CtxVec (Γ ++ ss)),
      backpropVec (Γ := Γ) (ss := ss) g xV seedV
        =
      (fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV).adjoint seedV := by
  intro xV seedV
  classical
  rcases hasFDerivAt_evalVec_and_jvp (Γ := Γ) (ss := ss) (g := g) hg xV with ⟨D, hD, hJ⟩
  have hfderiv : fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV = D := by
    simpa using hD.fderiv

  -- Use the global inner-product adjointness law.
  have hdot :
      ∀ dxV : CtxVec Γ,
        inner ℝ (D dxV) seedV =
          inner ℝ dxV (backpropVec (Γ := Γ) (ss := ss) g xV seedV) := by
    intro dxV
    -- From tape soundness on vectors.
    have h := Graph.backprop_correct_inner (Γ := Γ) (ss := ss) g xV dxV seedV
    simpa [hJ dxV] using h

  -- Identify the unique vector satisfying the adjointness law.
  let u : CtxVec Γ := backpropVec (Γ := Γ) (ss := ss) g xV seedV
  let v : CtxVec Γ := D.adjoint seedV
  have hforall : ∀ dxV : CtxVec Γ, inner ℝ dxV u = inner ℝ dxV v := by
    intro dxV
    calc
      inner ℝ dxV u
          = inner ℝ (D dxV) seedV := by
              simpa [u] using (hdot dxV).symm
      _ = inner ℝ dxV (D.adjoint seedV) := by
            simpa using
              (ContinuousLinearMap.adjoint_inner_right (A := D) (x := dxV) (y := seedV)).symm
      _ = inner ℝ dxV v := by simp [v]

  have h0 : inner ℝ (u - v) (u - v) = 0 := by
    have hEq := hforall (dxV := (u - v))
    have : inner ℝ (u - v) u - inner ℝ (u - v) v = 0 := by
      simpa [sub_eq_zero] using congrArg (fun t => t - inner ℝ (u - v) v) hEq
    calc
      inner ℝ (u - v) (u - v) = inner ℝ (u - v) u - inner ℝ (u - v) v := by
        -- avoid simp rewriting `inner_self` to `‖·‖^2`
        exact inner_sub_right (x := (u - v)) (y := u) (z := v)
      _ = 0 := this
  have huv : u - v = 0 := (inner_self_eq_zero (𝕜 := ℝ) (x := (u - v))).1 h0
  have huv' : u = v := sub_eq_zero.mp huv

  -- Rewrite `v` using `fderiv` and finish.
  calc
    backpropVec (Γ := Γ) (ss := ss) g xV seedV = v := by simpa [u] using huv'
    _ = (fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV).adjoint seedV := by
          simp [v, hfderiv]

-- ---------------------------------------------------------------------------
-- Pointwise versions: `HasFDerivAt` only at the actual execution point.
-- ---------------------------------------------------------------------------

/--
Pointwise induction: `evalVec` is differentiable at `xV`, and its derivative agrees with `jvpVec`.

This is the version used for graphs involving non-smooth or partial primitives, where we only
assume differentiability at the values encountered during execution.
-/
theorem hasFDerivAt_evalVec_and_jvp_at
    {ss : List Shape} (g : Graph Γ ss) :
    ∀ xV : CtxVec Γ,
      GraphFDerivCorrectAt (Γ := Γ) (ss := ss) g xV →
        ∃ D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ ss),
          HasFDerivAt (evalVec (Γ := Γ) (ss := ss) g) D xV
            ∧
          (∀ dxV : CtxVec Γ, jvpVec (Γ := Γ) (ss := ss) g xV dxV = D dxV) := by
  classical
  induction g with
  | nil =>
      intro xV _hg
      let h : ctxSize Γ = ctxSize (Γ ++ []) := congrArg ctxSize (List.append_nil Γ).symm
      let D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ []) := castCLM (h := h)
      refine ⟨D, ?_, ?_⟩
      · change HasFDerivAt (castVec h) D xV
        exact D.hasFDerivAt
      · intro dxV
        simp [Graph.jvpVec, castCtxVec, castVec, castCLM, D]
  | snoc g node ih =>
      intro xV hg
      rename_i ss τ
      rcases hg with ⟨hg_g, hg_node⟩
      rcases ih (xV := xV) hg_g with ⟨Dg, hDg, hJg⟩
      let ctxV : CtxVec (Γ ++ ss) := evalVec (Γ := Γ) (ss := ss) g xV
      let Dn : CtxVec (Γ ++ ss) →L[ℝ] Vec (Spec.Shape.size τ) := hg_node.deriv
      have hy :
          HasFDerivAt
            (node.forwardVec (Γ := Γ ++ ss) (τ := τ))
            Dn ctxV := hg_node.hasFDerivAt
      have hpair :
          HasFDerivAt
            (fun x : CtxVec Γ =>
              (evalVec (Γ := Γ) (ss := ss) g x,
                node.forwardVec (Γ := Γ ++ ss) (τ := τ) (evalVec (Γ := Γ) (ss := ss) g x)))
            (Dg.prod (Dn.comp Dg)) xV :=
        hDg.prodMk (hy.comp xV hDg)
      let assoc := List.append_assoc Γ ss [τ]
      let hAssoc : ctxSize ((Γ ++ ss) ++ [τ]) = ctxSize (Γ ++ (ss ++ [τ])) := congrArg ctxSize assoc
      let Dcast : CtxVec ((Γ ++ ss) ++ [τ]) →L[ℝ] CtxVec (Γ ++ (ss ++ [τ])) := castCLM (h := hAssoc)
      let D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ (ss ++ [τ])) :=
        (Dcast.comp (snocCLM (Γ := Γ ++ ss) (τ := τ))).comp (Dg.prod (Dn.comp Dg))
      refine ⟨D, ?_, ?_⟩
      · have hsnoc :
            HasFDerivAt
              (fun p : CtxVec (Γ ++ ss) × Vec (Spec.Shape.size τ) =>
                snocCtx (Γ := Γ ++ ss) (τ := τ) p.1 p.2)
              (snocCLM (Γ := Γ ++ ss) (τ := τ))
              (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV) := by
          change HasFDerivAt
            (snocCLM (Γ := Γ ++ ss) (τ := τ))
            (snocCLM (Γ := Γ ++ ss) (τ := τ))
            (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV)
          exact (snocCLM (Γ := Γ ++ ss) (τ := τ)).hasFDerivAt
        have hcomp1 :=
          (Dcast.hasFDerivAt (x := snocCtx (Γ := Γ ++ ss) (τ := τ) ctxV
            (node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV))).comp xV
              (hsnoc.comp xV hpair)
        change HasFDerivAt
          ((castCLM hAssoc) ∘
            (fun p : CtxVec (Γ ++ ss) × Vec (Spec.Shape.size τ) =>
              snocCtx (Γ := Γ ++ ss) (τ := τ) p.1 p.2) ∘
            fun xV : CtxVec Γ =>
              (evalVec (Γ := Γ) (ss := ss) g xV,
                node.forwardVec (Γ := Γ ++ ss) (τ := τ) (evalVec (Γ := Γ) (ss := ss) g xV)))
          D xV
        simpa [ctxV, D, Dcast, snocCLM, Function.comp, ContinuousLinearMap.comp_assoc] using hcomp1
      · intro dxV
        have hx : jvpVec (Γ := Γ) (ss := ss) g xV dxV = Dg dxV := hJg dxV
        have hy' :
            node.jvpVec (Γ := Γ ++ ss) (τ := τ) ctxV (Dg dxV) = (Dn.comp Dg) dxV := by
          simpa [Dn, ContinuousLinearMap.comp_apply] using (hg_node.jvp_eq (dxV := Dg dxV))
        ext i
        simp [Graph.jvpVec, ctxV, hx, hy', D, Dcast, snocCLM, snocCtx, castCtxVec, castVec,
          castCLM, appendCLM, ContinuousLinearMap.comp_apply, ContinuousLinearMap.prod_apply]

/-!
Pointwise corollaries: these mirror `jvpVec_eq_fderiv` and `backpropVec_eq_adjoint_fderiv`, but
only require `GraphFDerivCorrectAt` at the specific execution point.
-/

/-- Pointwise version of `jvpVec_eq_fderiv`. -/
theorem jvpVec_eq_fderiv_at
    {ss : List Shape} (g : Graph Γ ss) :
    ∀ xV dxV,
      GraphFDerivCorrectAt (Γ := Γ) (ss := ss) g xV →
        jvpVec (Γ := Γ) (ss := ss) g xV dxV
          =
        (fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV) dxV := by
  intro xV dxV hg
  rcases hasFDerivAt_evalVec_and_jvp_at (Γ := Γ) (ss := ss) (g := g) (xV := xV) hg with
    ⟨D, hD, hJ⟩
  have hfderiv : fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV = D := by
    simpa using hD.fderiv
  simpa [hfderiv] using hJ dxV

/-- Pointwise version of `backpropVec_eq_adjoint_fderiv`. -/
theorem backpropVec_eq_adjoint_fderiv_at
    {ss : List Shape} (g : Graph Γ ss) :
    ∀ (xV : CtxVec Γ) (seedV : CtxVec (Γ ++ ss)),
      GraphFDerivCorrectAt (Γ := Γ) (ss := ss) g xV →
        backpropVec (Γ := Γ) (ss := ss) g xV seedV
          =
        (fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV).adjoint seedV := by
  intro xV seedV hg
  classical
  rcases hasFDerivAt_evalVec_and_jvp_at (Γ := Γ) (ss := ss) (g := g) (xV := xV) hg with
    ⟨D, hD, hJ⟩
  have hfderiv : fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV = D := by
    simpa using hD.fderiv

  have hdot :
      ∀ dxV : CtxVec Γ,
        inner ℝ (D dxV) seedV =
          inner ℝ dxV (backpropVec (Γ := Γ) (ss := ss) g xV seedV) := by
    intro dxV
    have h := Graph.backprop_correct_inner (Γ := Γ) (ss := ss) g xV dxV seedV
    simpa [hJ dxV] using h

  let u : CtxVec Γ := backpropVec (Γ := Γ) (ss := ss) g xV seedV
  let v : CtxVec Γ := D.adjoint seedV
  have hforall : ∀ dxV : CtxVec Γ, inner ℝ dxV u = inner ℝ dxV v := by
    intro dxV
    calc
      inner ℝ dxV u
          = inner ℝ (D dxV) seedV := by
              simpa [u] using (hdot dxV).symm
      _ = inner ℝ dxV (D.adjoint seedV) := by
            simpa using
              (ContinuousLinearMap.adjoint_inner_right (A := D) (x := dxV) (y := seedV)).symm
      _ = inner ℝ dxV v := by simp [v]

  have h0 : inner ℝ (u - v) (u - v) = 0 := by
    have hEq := hforall (dxV := (u - v))
    have : inner ℝ (u - v) u - inner ℝ (u - v) v = 0 := by
      simpa [sub_eq_zero] using congrArg (fun t => t - inner ℝ (u - v) v) hEq
    calc
      inner ℝ (u - v) (u - v) = inner ℝ (u - v) u - inner ℝ (u - v) v := by
        -- avoid simp rewriting `inner_self` to `‖·‖^2`
        exact inner_sub_right (x := (u - v)) (y := u) (z := v)
      _ = 0 := this
  have huv : u - v = 0 := (inner_self_eq_zero (𝕜 := ℝ) (x := (u - v))).1 h0
  have huv' : u = v := sub_eq_zero.mp huv

  calc
    backpropVec (Γ := Γ) (ss := ss) g xV seedV = v := by simpa [u] using huv'
    _ = (fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g) xV).adjoint seedV := by
          simp [v, hfderiv]

end Graph

end
end Autograd
end Proofs
