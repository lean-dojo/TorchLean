/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Vec
public import TorchLean.NN.Tape.Graph
public import Mathlib.Analysis.Calculus.FDeriv.Prod
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# FDeriv

Analytic correctness (`HasFDerivAt`, `fderiv`) for SSA tapes over `ℝ`.
`Tape.Graph.jvpCtx_backpropCtx_adjoint` proves the global JVP/VJP adjointness law for tape graphs
against the tensor dot product.

This file proves the analytic statements at a fixed environment `d : Δ`. The proofs flatten every
context into one Euclidean vector and assume that each node's JVP is the Fréchet derivative of its
forward map at the point where the node runs (`RealGraph.FDerivCorrectAt`). They derive
`jvp = fderiv` for an executable graph `Tape.GraphData ℝ Δ Γ ss`, and from it
`backprop = (fderiv eval)†` for a proof-carrying graph `Tape.Graph (α := ℝ) Δ Γ ss`.

## PyTorch correspondence / citations
- `backpropVec` corresponds to PyTorch's VJP accumulation pass over a dynamic tape, and
  `Tape.Graph.backpropVec_eq_fderiv` proves that this reverse pass is the adjoint of
  the derivative of the forward map.
  https://pytorch.org/docs/stable/autograd.html
- PyTorch's functional API offers the same Jacobian, VJP and JVP views of a function.
  https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean

open TorchLean.Tensor
open Rep (dot)

open scoped BigOperators

noncomputable section

open Vec (inner_eq_sum_mul)

-- ---------------------------------------------------------------------------
-- Tensor vectorization (`Tensor ℝ s` ↔ `Vec (Shape.size s)`)
-- ---------------------------------------------------------------------------

/-- Vectorize a tensor by flattening it (spec flattening order) and then using the Euclidean
  equivalence. -/
def Tensor.toVec {s : Shape} (t : Tensor ℝ s) : Vec (Shape.size s) :=
  getScalarE (n := Shape.size s) (Tensor.reshape (α := ℝ) t [_] (Shape.size_vector _).symm)

/-- The sole coordinate obtained by vectorizing a scalar tensor is the scalar itself. -/
@[simp] theorem Tensor.toVec_scalar (x : ℝ) (i : Fin (Shape.size Shape.scalar)) :
    toVec (t := (Tensor.scalar x : Tensor ℝ .scalar)) i = x := by
  change (getScalarE (Tensor.reshape (Tensor.scalar x) [_] (Shape.size_vector _).symm)).ofLp i = x
  rw [getScalarE_ofLp]
  cases i with
  | mk value hvalue =>
      have hzero : value = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ hvalue)
      subst value
      rfl

/-- Inverse of `toVec`: interpret a vector as a tensor of shape `s`. -/
def Tensor.ofVec {s : Shape} (v : Vec (Shape.size s)) : Tensor ℝ s :=
  Tensor.reshape (α := ℝ) (ofFnE (n := Shape.size s) v) s (Shape.size_vector _)

/-- Vectorizing a tensor built from a vector gives the vector back. -/
@[simp] theorem Tensor.toVec_ofVec {s : Shape} (v : Vec (Shape.size s)) :
    toVec (t := ofVec (s := s) v) = v := by
  have hunf :
      Tensor.reshape (α := ℝ) (Tensor.reshape (α := ℝ) (ofFnE (n := Shape.size s) v) s
          (Shape.size_vector _)) [_] (Shape.size_vector _).symm
        =
      ofFnE (n := Shape.size s) v :=
    reshape_reshape_self (ofFnE (n := Shape.size s) v) _ _
  have := congrArg (getScalarE (n := Shape.size s)) hunf
  exact this.trans (getScalarE_ofFnE (n := Shape.size s) v)

/-- `ofVec` undoes `toVec`, so the two maps are mutually inverse.

With both round trips as `simp` lemmas, the calculus below can work in `EuclideanSpace` and state
its conclusions about tensors. -/
@[simp] theorem Tensor.ofVec_toVec {s : Shape} (t : Tensor ℝ s) :
    ofVec (s := s) (toVec (t := t)) = t := by
  have hround :
      ofFnE (n := Shape.size s) (getScalarE (n := Shape.size s) (Tensor.reshape (α := ℝ) t
          [_] (Shape.size_vector _).symm))
        =
      Tensor.reshape (α := ℝ) t [_] (Shape.size_vector _).symm := by
    simp
  change Tensor.reshape (α := ℝ) (ofFnE (n := Shape.size s)
        (getScalarE (n := Shape.size s) (Tensor.reshape (α := ℝ) t [_]
            (Shape.size_vector _).symm))) s (Shape.size_vector _) = t
  rw [hround]
  exact reshape_reshape_self t _ _

-- ---------------------------------------------------------------------------
-- Context vectorization (`TorchLean.TensorPack ℝ Γ` ↔ `Vec (ctxSize Γ)`)
-- ---------------------------------------------------------------------------

/-- Total number of scalar coordinates in a heterogeneous context shape list. -/
def ctxSize : List Shape → Nat
  | [] => 0
  | s :: ss => Shape.size s + ctxSize ss

/-- A vectorized context containing every entry of a `TorchLean.TensorPack ℝ Γ`. -/
abbrev CtxVec (Γ : List Shape) := Vec (ctxSize Γ)

/--
Flatten a typed context `TorchLean.TensorPack ℝ Γ` into one Euclidean vector, entry by entry in
context order.

PyTorch keeps saved tensors in a dynamically typed array. Here the shapes live in `Γ`, so the split
points follow from `ctxSize` by definition and flattening is a typed isomorphism.
-/
def TensorPack.toVec : {Γ : List Shape} → TensorPack ℝ Γ → CtxVec Γ
  | [], .nil => 0
  | _ :: ss, .cons x xs =>
      (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (Tensor.toVec x, toVec (Γ := ss) xs)

/-- Inverse of `toVec`: split a `CtxVec Γ` back into a `TorchLean.TensorPack ℝ Γ`. -/
def TensorPack.ofVec : {Γ : List Shape} → CtxVec Γ → TensorPack ℝ Γ
  | [], _ => .nil
  | s :: ss, v =>
      let head : Vec (Shape.size s) :=
        WithLp.toLp 2 fun i => v (Fin.castAdd (ctxSize ss) i)
      let tail : Vec (ctxSize ss) :=
        WithLp.toLp 2 fun i => v (Fin.natAdd (Shape.size s) i)
      .cons (Tensor.ofVec (s := s) head) (ofVec (Γ := ss) tail)

/-- Left half of a concatenation reads from the first vector. -/
@[simp] theorem Vec.append_ofLp_castAdd {m n : Nat} (a : Vec m) (b : Vec n) (i : Fin m) :
    ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (a, b)).ofLp (Fin.castAdd n i) = a.ofLp i :=
  congrArg (fun p => p.1.ofLp i) ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).apply_symm_apply (a, b))

/-- Right half of a concatenation reads from the second vector. -/
@[simp] theorem Vec.append_ofLp_natAdd {m n : Nat} (a : Vec m) (b : Vec n) (i : Fin n) :
    ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (a, b)).ofLp (Fin.natAdd m i) = b.ofLp i :=
  congrArg (fun p => p.2.ofLp i) ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).apply_symm_apply (a, b))

/-- Unflattening a flattened context recovers the original tensor pack. -/
@[simp] theorem TensorPack.ofVec_toVec {Γ : List Shape}
    (xs : TensorPack ℝ Γ) :
    ofVec (Γ := Γ) (toVec (Γ := Γ) xs) = xs := by
  induction Γ with
  | nil =>
      cases xs
      rfl
  | cons s ss ih =>
      cases xs with
      | cons x xs =>
          have hhead :
              WithLp.toLp 2 (fun i =>
                (toVec (Γ := s :: ss) (.cons x xs)) (Fin.castAdd (ctxSize ss) i)) =
                Tensor.toVec x := by
            ext i
            exact (PiLp.toLp_apply _ _ _ i).trans (Vec.append_ofLp_castAdd _ _ i)
          have htail :
              WithLp.toLp 2 (fun i =>
                (toVec (Γ := s :: ss) (.cons x xs))
                  (Fin.natAdd (Shape.size s) i)) =
                toVec xs := by
            ext i
            exact (PiLp.toLp_apply _ _ _ i).trans (Vec.append_ofLp_natAdd _ _ i)
          simp only [ofVec]
          rw [hhead, Tensor.ofVec_toVec, htail, ih]

/-- Flattening an unflattened vector gives it back, so a context and one flat vector carry the same
data. The whole graph can therefore be differentiated as a single map `Vec (ctxSize Γ) → Vec n`,
with the result read back per parameter. -/
@[simp] theorem TensorPack.toVec_ofVec {Γ : List Shape} (v : CtxVec Γ) :
    toVec (Γ := Γ) (ofVec (Γ := Γ) v) = v := by
  induction Γ with
  | nil =>
      ext i
      exact i.elim0
  | cons s ss ih =>
      let head : Vec (Shape.size s) :=
        WithLp.toLp 2 fun i => v (Fin.castAdd (ctxSize ss) i)
      let tail : Vec (ctxSize ss) :=
        WithLp.toLp 2 fun i => v (Fin.natAdd (Shape.size s) i)
      rw [show ofVec (Γ := s :: ss) v =
        .cons (Tensor.ofVec head) (ofVec tail) by rfl]
      rw [show toVec (Γ := s :: ss) (.cons (Tensor.ofVec head) (ofVec tail)) =
        (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm
          (Tensor.toVec (Tensor.ofVec head), toVec (ofVec tail)) by rfl]
      rw [Tensor.toVec_ofVec, ih]
      ext i
      induction i using Fin.addCases with
      | left i => exact (Vec.append_ofLp_castAdd head tail i).trans (PiLp.toLp_apply _ _ _ i)
      | right i => exact (Vec.append_ofLp_natAdd head tail i).trans (PiLp.toLp_apply _ _ _ i)

-- ---------------------------------------------------------------------------
-- Dot/inner agreement (`dotList` ↔ Euclidean inner product)
-- ---------------------------------------------------------------------------

/--
Coordinates of `toVec` on a tensor `.dim n s`, in the product order of `finProdFinEquiv`.

No positivity hypothesis on `Shape.size s` is needed: the coordinate
`p.2 : Fin (Shape.size s)` already witnesses it.
-/
theorem Tensor.toVec_dim_apply {n : Nat} {s : Shape}
    (f : Fin n → Tensor ℝ s) (p : Fin n × Fin (Shape.size s)) :
    toVec (t := Tensor.dim f) (finProdFinEquiv p) = toVec (t := f p.1) p.2 := by
  calc
    toVec (t := Tensor.dim f) (finProdFinEquiv p) =
        Tensor.getScalar (Tensor.reshape (Tensor.dim f) [_] (Shape.size_vector _).symm)
            (finProdFinEquiv p) :=
      getScalarE_ofLp _ _
    _ = Tensor.getScalar (Tensor.reshape (f p.1) [_] (Shape.size_vector _).symm) p.2 := by
      have hidx :
          p.1.val * Shape.size s + p.2.val < n * Shape.size s := by
        simpa [finProdFinEquiv, Nat.mul_comm, Nat.add_comm] using
          (finProdFinEquiv p).isLt
      have hflat :
          finProdFinEquiv p =
            (⟨p.1.val * Shape.size s + p.2.val, hidx⟩ :
              Fin (n * Shape.size s)) := by
        apply Fin.ext
        simp [finProdFinEquiv, Nat.mul_comm, Nat.add_comm]
      exact (congrArg (Tensor.getScalar
          (Tensor.reshape (Tensor.dim f) [_] (Shape.size_vector _).symm)) hflat).trans
        (Tensor.getScalar_reshape_dim
          (values := f) (outer := p.1) (inner := p.2) hidx)
    _ = toVec (t := f p.1) p.2 := (getScalarE_ofLp _ _).symm

/-- `toVec` turns dot products on `.dim n s` into sums of Euclidean inner products over
slices. -/
theorem Tensor.inner_toVec_dim {n : Nat} {s : Shape} (a b : Fin n → Tensor ℝ s) :
    inner ℝ (toVec (t := Tensor.dim a)) (toVec (t := Tensor.dim b))
      =
    ∑ i : Fin n, inner ℝ (toVec (t := a i)) (toVec (t := b i)) := by
  classical
  -- Reindex the flat coordinate sum by `finProdFinEquiv`, then split it into outer and inner sums.
  calc
    inner ℝ (toVec (t := Tensor.dim a)) (toVec (t := Tensor.dim b))
        =
      ∑ i : Fin (n * Shape.size s),
        toVec (t := Tensor.dim a) i * toVec (t := Tensor.dim b) i := by
          rw [inner_eq_sum_mul]
          rfl
    _ =
      ∑ p : Fin n × Fin (Shape.size s),
        toVec (t := Tensor.dim a) (finProdFinEquiv p) *
          toVec (t := Tensor.dim b) (finProdFinEquiv p) := by
        exact
          (Equiv.sum_comp (e := finProdFinEquiv)
            (g := fun i : Fin (n * Shape.size s) =>
              toVec (t := Tensor.dim a) i * toVec (t := Tensor.dim b) i)).symm
    _ =
      ∑ p : Fin n × Fin (Shape.size s),
        toVec (t := a p.1) p.2 * toVec (t := b p.1) p.2 := by
        refine Finset.sum_congr rfl ?_
        intro p _
        simp [toVec_dim_apply]
    _ =
      ∑ i : Fin n, ∑ j : Fin (Shape.size s),
        toVec (t := a i) j * toVec (t := b i) j := by
        simp [Fintype.sum_prod_type]
    _ =
      ∑ i : Fin n, inner ℝ (toVec (t := a i)) (toVec (t := b i)) := by
        refine Finset.sum_congr rfl ?_
        intro i _
        simpa using
          (inner_eq_sum_mul (x := toVec (t := a i)) (y := toVec (t := b i))).symm

/--
The pairing `dot` of two real tensors is the Euclidean inner product of their vectorizations.
This carries the tape adjointness laws, stated with `dot`, to the `inner` statements here.
-/
theorem Tensor.dot_eq_inner_toVec {s : Shape} (a b : Tensor ℝ s) :
    Rep.dot a b = inner ℝ (toVec (t := a)) (toVec (t := b)) := by
  classical
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item a, ← Tensor.scalar_item b]
      let vx : Vec 1 := toVec (t := Tensor.scalar a.item)
      let vy : Vec 1 := toVec (t := Tensor.scalar b.item)
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
        Rep.dot (Tensor.scalar a.item) (Tensor.scalar b.item) = a.item * b.item := by
          simp only [dot_scalar, Tensor.item_scalar]
        _ = vx.ofLp 0 * vy.ofLp 0 := by simp [hvx0, hvy0]
        _ = inner ℝ vx vy := by
          simpa using hinner.symm
  | dim n s ih =>
      rw [← Tensor.dim_unstack a, ← Tensor.dim_unstack b]
      have hdot :
          Rep.dot (Tensor.dim (Tensor.unstack a)) (Tensor.dim (Tensor.unstack b)) =
            ∑ i : Fin n, Rep.dot (Tensor.unstack a i) (Tensor.unstack b i) := by
        rw [dot_eq_sum_unstack]
        simp only [Tensor.unstack_dim]
      have hinter :
          inner ℝ
              (toVec (t := Tensor.dim (Tensor.unstack a)))
              (toVec (t := Tensor.dim (Tensor.unstack b))) =
            ∑ i : Fin n,
              inner ℝ
                (toVec (t := Tensor.unstack a i))
                (toVec (t := Tensor.unstack b i)) :=
        inner_toVec_dim (a := Tensor.unstack a) (b := Tensor.unstack b)
      calc
        Rep.dot (Tensor.dim (Tensor.unstack a)) (Tensor.dim (Tensor.unstack b)) =
            ∑ i : Fin n,
              inner ℝ
                (toVec (t := Tensor.unstack a i))
                (toVec (t := Tensor.unstack b i)) := by
          refine hdot.trans ?_
          refine Finset.sum_congr rfl ?_
          intro i _
          simpa using
            (ih (a := Tensor.unstack a i) (b := Tensor.unstack b i))
        _ = inner ℝ
            (toVec (t := Tensor.dim (Tensor.unstack a)))
            (toVec (t := Tensor.dim (Tensor.unstack b))) := hinter.symm

/-- The empty context flattens to the zero vector of the zero-dimensional space. -/
@[simp] theorem TensorPack.toVec_nil :
    toVec (Pack.nil : TensorPack ℝ []) = 0 := rfl

/-- Flattening a `cons` concatenates the head tensor's coordinates in front of the tail's.

The whole file relies on this layout: entries appear in context order, so `Fin.castAdd` and
`Fin.natAdd` alone split a gradient vector back into its blocks. -/
@[simp] theorem TensorPack.toVec_cons {s : Shape} {ss : List Shape}
    (x : Tensor ℝ s) (xs : TensorPack ℝ ss) :
    toVec (Pack.cons x xs) =
      (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (Tensor.toVec x, toVec xs) := rfl

/-- Reassociating concatenated vectors only changes their finite-index representation. -/
theorem Vec.append_assoc {m n p : Nat} (a : Vec m) (b : Vec n) (c : Vec p) :
    LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (Nat.add_assoc m n p))
        ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm
          ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (a, b), c)) =
      (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm
        (a, (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (b, c)) := by
  apply PiLp.ext
  intro i
  rw [LinearIsometryEquiv.piLpCongrLeft_apply, Equiv.piCongrLeft'_apply, finCongr_symm_apply]
  induction i using Fin.addCases with
  | left i =>
    rw [show Fin.cast (Nat.add_assoc m n p).symm (Fin.castAdd (n + p) i) =
      Fin.castAdd p (Fin.castAdd n i) from Fin.ext rfl]
    simp only [append_ofLp_castAdd]
  | right j =>
    induction j using Fin.addCases with
    | left j =>
      rw [show Fin.cast (Nat.add_assoc m n p).symm (Fin.natAdd m (Fin.castAdd p j)) =
        Fin.castAdd p (Fin.natAdd m j) from Fin.ext rfl]
      simp only [append_ofLp_castAdd, append_ofLp_natAdd]
    | right k =>
      rw [show Fin.cast (Nat.add_assoc m n p).symm (Fin.natAdd m (Fin.natAdd n k)) =
        Fin.natAdd (m + n) k from Fin.ext (Nat.add_assoc m n k).symm]
      simp only [append_ofLp_natAdd]

/-- Casting the right block of a concatenation is the same as casting the full vector. -/
theorem Vec.append_cast_right {m n p : Nat} (h : n = p) (a : Vec m) (b : Vec n) :
    (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm
        (a, LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr h) b) =
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (congrArg (m + ·) h))
        ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (a, b)) := by
  subst p
  rfl

/-- Inner product of concatenated vectors splits as a sum of inner products. -/
theorem Vec.inner_append {m n : Nat} (a c : Vec m) (b d : Vec n) :
    inner ℝ ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (a, b) : Vec (m + n))
        ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (c, d))
      =
    inner ℝ a c + inner ℝ b d := by
  -- Split the coordinate sum over `Fin (m + n)` into its `castAdd` and `natAdd` halves.
  simp only [inner_eq_sum_mul, Fin.sum_univ_add, append_ofLp_castAdd, append_ofLp_natAdd]

/--
The context pairing `TensorPack.dotList` at `ℝ` is the Euclidean inner product of the
`toVec` vectorizations.
-/
theorem TensorPack.realDotList_eq_inner_toVec {Γ : List Shape}
    (x y : TensorPack ℝ Γ) :
    TensorPack.dotList (ss := Γ) x y =
      inner ℝ (toVec (Γ := Γ) x) (toVec (Γ := Γ) y) := by
  classical
  induction Γ with
  | nil =>
      cases x
      cases y
      simp [TensorPack.dotList, toVec]
  | cons s ss ih =>
      cases x with
      | cons xh xt =>
          cases y with
          | cons yh yt =>
              -- Split the inner product across the append, then use the IH and
              -- `dot_eq_inner_toVec`.
              have hinter :
                  inner ℝ (toVec (Γ := s :: ss) (Pack.cons xh xt))
                        (toVec (Γ := s :: ss) (Pack.cons yh yt))
                    =
                  inner ℝ (Tensor.toVec (t := xh)) (Tensor.toVec (t := yh))
                    + inner ℝ (toVec (Γ := ss) xt) (toVec (Γ := ss) yt) := by
                rw [toVec_cons, toVec_cons]
                exact
                  Vec.inner_append (m := Shape.size s) (n := ctxSize ss)
                    (a := Tensor.toVec (t := xh)) (c := Tensor.toVec (t := yh))
                    (b := toVec (Γ := ss) xt) (d := toVec (Γ := ss) yt)
              calc
                TensorPack.dotList (ss := s :: ss) (Pack.cons xh xt)
                    (Pack.cons yh yt)
                    = Rep.dot xh yh + TensorPack.dotList (ss := ss) xt yt := by
                        simp [TensorPack.dotList]
                _ = inner ℝ (Tensor.toVec (t := xh)) (Tensor.toVec (t := yh))
                      + inner ℝ (toVec (Γ := ss) xt) (toVec (Γ := ss) yt) := by
                        simp [dot_eq_inner_toVec, ih]
                _ = inner ℝ (toVec (Γ := s :: ss) (Pack.cons xh xt))
                        (toVec (Γ := s :: ss) (Pack.cons yh yt)) := by
                        exact hinter.symm

-- ---------------------------------------------------------------------------
-- Vector graph semantics (for calculus)
-- ---------------------------------------------------------------------------

/-- Cast a vectorized context along an equality of shape lists (reindexing coordinates). -/
def CtxVec.cast {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) : CtxVec Γ₁ → CtxVec Γ₂ :=
  LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (congrArg ctxSize h))

/-- Casting a vectorized context along `rfl` is the identity. -/
@[simp] theorem CtxVec.cast_rfl {Γ : List Shape} (v : CtxVec Γ) :
    cast (Γ₁ := Γ) (Γ₂ := Γ) rfl v = v :=
  rfl

/-- Context casts compose. Together with `cast_rfl` this keeps the casts introduced by graph
composition from piling up in the goal. -/
@[simp] theorem CtxVec.cast_cast {Γ₁ Γ₂ Γ₃ : List Shape} (h₁ : Γ₁ = Γ₂) (h₂ : Γ₂ = Γ₃)
    (v : CtxVec Γ₁) :
    cast (Γ₁ := Γ₂) (Γ₂ := Γ₃) h₂ (cast (Γ₁ := Γ₁) (Γ₂ := Γ₂) h₁ v)
      =
    cast (Γ₁ := Γ₁) (Γ₂ := Γ₃) (h₁.trans h₂) v := by
  rfl

/-!
Bookkeeping for splitting and concatenating vectorized contexts. Each fact follows from the list
structure of `Γ`, and naming it spares the calculus proofs below from redoing shape arithmetic.
-/

/-- `ctxSize` respects list append (sizes add). -/
theorem ctxSize_append (Γ ss : List Shape) : ctxSize (Γ ++ ss) = ctxSize Γ + ctxSize ss := by
  induction Γ with
  | nil => simp [ctxSize]
  | cons s Γ ih => simp [ctxSize, ih, Nat.add_assoc]

/-- Specialized `ctxSize_append` for snoc (`Γ ++ [τ]`). -/
theorem ctxSize_snoc (ss : List Shape) (τ : Shape) :
    ctxSize (ss ++ [τ]) = ctxSize ss + Shape.size τ := by
  -- `ctxSize [τ] = Shape.size τ`.
  simp [ctxSize, ctxSize_append]

/-- Append one tensor-vector block to a vectorized context. -/
def CtxVec.snoc {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ) (t : Vec (Shape.size τ)) :
    CtxVec (Γ ++ [τ]) :=
  LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (ctxSize_snoc Γ τ).symm)
    ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (ctx, t))

/-- Prefixing a context vector commutes with appending its final tensor block. -/
theorem Vec.append_snoc {s : Shape} {Γ : List Shape} {τ : Shape}
    (a : Vec s.size) (ctx : CtxVec Γ) (t : Vec τ.size) :
    (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (a, CtxVec.snoc ctx t) =
      CtxVec.snoc (Γ := s :: Γ) ((EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm (a, ctx)) t := by
  unfold CtxVec.snoc
  rw [append_cast_right, ← append_assoc]
  rfl

/-- Inverse of `snoc`: split `CtxVec (Γ ++ [τ])` into its prefix and last block. -/
def CtxVec.unsnoc {Γ : List Shape} {τ : Shape} (ctx : CtxVec (Γ ++ [τ])) :
    CtxVec Γ × Vec (Shape.size τ) :=
  EuclideanSpace.finAddEquivProd (𝕜 := ℝ)
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (ctxSize_snoc Γ τ)) ctx)

/-- `unsnoc (snoc ctx t) = (ctx, t)`. -/
theorem CtxVec.unsnoc_snoc {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ)
    (t : Vec (Shape.size τ)) :
    unsnoc (Γ := Γ) (τ := τ) (snoc (Γ := Γ) (τ := τ) ctx t) = (ctx, t) := by
  rw [unsnoc, snoc, ← finCongr_symm (ctxSize_snoc Γ τ),
    ← LinearIsometryEquiv.piLpCongrLeft_symm, LinearIsometryEquiv.apply_symm_apply,
    ContinuousLinearEquiv.apply_symm_apply]

namespace Tape

namespace NodeData

/-- Vectorized forward map of a tape node at the environment `d`:
`CtxVec Γ → Vec (Shape.size τ)`. -/
def forwardVec {Δ : Type} {Γ : List Shape} {τ : Shape} (node : NodeData ℝ Δ Γ τ) (d : Δ) :
    CtxVec Γ → Vec (Shape.size τ) :=
  fun ctxV => toVec (t := node.forward (TensorPack.ofVec (Γ := Γ) ctxV) d)

/-- Vectorized JVP of a tape node at the environment `d`: the node-level forward-mode action on
tangents. -/
def jvpVec {Δ : Type} {Γ : List Shape} {τ : Shape} (node : NodeData ℝ Δ Γ τ) (d : Δ) :
    CtxVec Γ → CtxVec Γ → Vec (Shape.size τ) :=
  fun ctxV dctxV =>
    toVec (t := node.jvp (TensorPack.ofVec (Γ := Γ) ctxV) (TensorPack.ofVec (Γ := Γ) dctxV) d)

/-- Vectorized VJP of a tape node at the environment `d`: pushes a cotangent vector back to the
input context. -/
def vjpVec {Δ : Type} {Γ : List Shape} {τ : Shape} (node : NodeData ℝ Δ Γ τ) (d : Δ) :
    CtxVec Γ → Vec (Shape.size τ) → CtxVec Γ :=
  fun ctxV δV =>
    TensorPack.toVec (Γ := Γ) (node.vjp (TensorPack.ofVec (Γ := Γ) ctxV) d (ofVec (s := τ) δV))

end NodeData

namespace LawfulNode

/--
Vectorized form of the adjointness law `LawfulNode.jvp_vjp_adjoint` at the environment `d`:
`⟪jvp(x,dx), δ⟫ = ⟪dx, vjp(x,δ)⟫`.
-/
theorem jvpVec_vjpVec_adjoint {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : LawfulNode (α := ℝ) Δ Γ τ) (d : Δ) :
    ∀ (ctxV dctxV : CtxVec Γ) (δV : Vec (Shape.size τ)),
      inner ℝ (node.jvpVec d ctxV dctxV) δV = inner ℝ dctxV (node.vjpVec d ctxV δV) := by
  intro ctxV dctxV δV
  let ctx := TensorPack.ofVec (Γ := Γ) ctxV
  let dctx := TensorPack.ofVec (Γ := Γ) dctxV
  let δ := ofVec (s := τ) δV
  have hdot := node.jvp_vjp_adjoint ctx dctx d δ
  -- Convert `dot`/`dotList` to inner products.
  have hleft : Rep.dot (node.jvp ctx dctx d) δ = inner ℝ (node.jvpVec d ctxV dctxV) δV := by
    simp [NodeData.jvpVec, ctx, dctx, δ, dot_eq_inner_toVec]
  have hright :
      TensorPack.dotList (ss := Γ) dctx (node.vjp ctx d δ) =
        inner ℝ dctxV (node.vjpVec d ctxV δV) := by
    simpa [NodeData.vjpVec, ctx, dctx, δ] using
      (TensorPack.realDotList_eq_inner_toVec (Γ := Γ) (x := dctx) (y := node.vjp ctx d δ))
  simpa [hleft, hright] using hdot

end LawfulNode

-- ---------------------------------------------------------------------------
-- Graph semantics on Euclidean contexts
-- ---------------------------------------------------------------------------

namespace GraphData

variable {Δ : Type} {Γ : List Shape}

/--
Vectorized evaluation of an executable tape graph `GraphData` at the environment `d`: the inputs
followed by every node output, as a `CtxVec (Γ ++ ss)`.
-/
def evalVec {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ) (xV : CtxVec Γ) :
    CtxVec (Γ ++ ss) :=
  match g with
  | .nil =>
      CtxVec.cast (Γ₁ := Γ) (Γ₂ := Γ ++ []) (List.append_nil Γ).symm xV
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctxV : CtxVec (Γ ++ ss) := evalVec (ss := ss) g d xV
      let yV : Vec (Shape.size τ) := node.forwardVec (Γ := Γ ++ ss) (τ := τ) d ctxV
      CtxVec.cast (Γ₁ := (Γ ++ ss) ++ [τ]) (Γ₂ := Γ ++ (ss ++ [τ]))
        (List.append_assoc Γ ss [τ])
        (CtxVec.snoc (Γ := (Γ ++ ss)) (τ := τ) ctxV yV)

/-- Vectorized JVP for a whole graph at the environment `d`: forward-mode derivative of
`evalVec`. -/
def jvpVec {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ) (xV dxV : CtxVec Γ) :
    CtxVec (Γ ++ ss) :=
  match g with
  | .nil =>
      CtxVec.cast (Γ₁ := Γ) (Γ₂ := Γ ++ []) (List.append_nil Γ).symm dxV
  | .snoc (ss := ss) (τ := τ) g node =>
      let ctxV : CtxVec (Γ ++ ss) := evalVec (ss := ss) g d xV
      let dctxV : CtxVec (Γ ++ ss) := jvpVec (ss := ss) g d xV dxV
      let dyV : Vec (Shape.size τ) := node.jvpVec (Γ := Γ ++ ss) (τ := τ) d ctxV dctxV
      CtxVec.cast (Γ₁ := (Γ ++ ss) ++ [τ]) (Γ₂ := Γ ++ (ss ++ [τ]))
        (List.append_assoc Γ ss [τ])
        (CtxVec.snoc (Γ := (Γ ++ ss)) (τ := τ) dctxV dyV)

/--
Vectorized reverse-mode accumulation (VJP) for a whole graph at the environment `d`.

`seedV` is a cotangent for the entire `Γ ++ ss` context (inputs plus intermediates), matching the
global tape soundness theorem.
-/
def backpropVec {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ) (xV : CtxVec Γ)
    (seedV : CtxVec (Γ ++ ss)) : CtxVec Γ :=
  match g with
  | .nil =>
      CtxVec.cast (Γ₁ := Γ ++ []) (Γ₂ := Γ) (List.append_nil Γ) seedV
  | .snoc (ss := ss) (τ := τ) g node =>
      let assoc := List.append_assoc Γ ss [τ]
      let seedV' : CtxVec ((Γ ++ ss) ++ [τ]) :=
        CtxVec.cast (Γ₁ := Γ ++ (ss ++ [τ])) (Γ₂ := (Γ ++ ss) ++ [τ]) assoc.symm seedV
      let seedPrevV : CtxVec (Γ ++ ss) := (CtxVec.unsnoc (Γ := (Γ ++ ss)) (τ := τ) seedV').1
      let seedOutV : Vec (Shape.size τ) := (CtxVec.unsnoc (Γ := (Γ ++ ss)) (τ := τ) seedV').2
      let ctxV : CtxVec (Γ ++ ss) := evalVec (ss := ss) g d xV
      let contribV : CtxVec (Γ ++ ss) := node.vjpVec (Γ := Γ ++ ss) (τ := τ) d ctxV seedOutV
      backpropVec (ss := ss) g d xV (seedPrevV + contribV)

end GraphData

end Tape

-- ---------------------------------------------------------------------------
-- Vectorization transport: `TensorPack.toVec` commutes with the context operations
-- ---------------------------------------------------------------------------

/-- `toVec` commutes with casting a context along a shape-list equality. -/
theorem TensorPack.toVec_cast {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) (xs : TensorPack ℝ Γ₁) :
    toVec (Γ := Γ₂) (Pack.cast h xs) =
      CtxVec.cast (Γ₁ := Γ₁) (Γ₂ := Γ₂) h (toVec xs) := by
  cases h
  simp

/-- Vectorization is additive: `toVec` maps `+` to vector addition. -/
theorem Tensor.toVec_add {s : Shape} (a b : Tensor ℝ s) :
    toVec (t := a + b) = toVec (t := a) + toVec (t := b) := by
  refine ext_inner_right ℝ ?_
  intro w
  have hw : w = toVec (t := ofVec (s := s) w) :=
    (toVec_ofVec (s := s) w).symm
  rw [hw, ← dot_eq_inner_toVec, inner_add_left, ← dot_eq_inner_toVec,
    ← dot_eq_inner_toVec, dot_add_left]

/-- The context inner product is additive in its left argument. -/
private theorem TensorPack.realDotList_add_left' {Γ : List Shape} (u v z : TensorPack ℝ Γ) :
    TensorPack.dotList (TensorPack.add u v) z =
      TensorPack.dotList u z + TensorPack.dotList v z := by
  induction Γ with
  | nil =>
    cases u
    cases v
    cases z
    simp [TensorPack.dotList, TensorPack.add]
  | cons s Γ ih =>
    cases u with
    | cons uh ut =>
      cases v with
      | cons vh vt =>
        cases z with
        | cons zh zt =>
          show TensorPack.dotList
              (Pack.cons (uh + vh) (TensorPack.add ut vt))
              (Pack.cons zh zt)
              = _
          simp only [TensorPack.dotList, Tensor.dot_add_left, ih]
          ring

/-- `toVec` maps context addition to vector addition. -/
theorem TensorPack.toVec_add {Γ : List Shape} (u v : TensorPack ℝ Γ) :
    toVec (TensorPack.add u v) = toVec u + toVec v := by
  refine ext_inner_right ℝ ?_
  intro w
  have hw : w = toVec (ofVec (Γ := Γ) w) :=
    (toVec_ofVec (Γ := Γ) w).symm
  rw [hw, ← realDotList_eq_inner_toVec, inner_add_left, ← realDotList_eq_inner_toVec,
    ← realDotList_eq_inner_toVec]
  exact realDotList_add_left' u v _

/-- `toVec` maps `TorchLean.Pack.snoc` to `CtxVec.snoc`. -/
theorem TensorPack.toVec_snoc {Γ : List Shape} {τ : Shape} (xs : TensorPack ℝ Γ)
    (y : Tensor ℝ τ) :
    toVec (Γ := Γ ++ [τ]) (Pack.snoc xs y)
      = CtxVec.snoc (Γ := Γ) (τ := τ) (toVec xs) (Tensor.toVec (t := y)) := by
  induction Γ with
  | nil =>
    cases xs
    change toVec (Pack.cons y Pack.nil) = _
    rw [toVec_cons, toVec_nil]
    have h : ctxSize [] + τ.size = τ.size + ctxSize [] := by simp [ctxSize]
    change _ = LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr h) _
    apply PiLp.ext
    intro i
    induction i using Fin.addCases with
    | left i =>
      rw [Vec.append_ofLp_castAdd, LinearIsometryEquiv.piLpCongrLeft_apply,
        Equiv.piCongrLeft'_apply, finCongr_symm_apply]
      have hidx :
          Fin.cast h.symm (Fin.castAdd (ctxSize []) i) = Fin.natAdd (ctxSize []) i := by
        apply Fin.ext
        simpa only [ctxSize, Fin.val_natAdd, Nat.zero_add, Fin.val_castAdd] using
          Fin.val_cast h.symm (Fin.castAdd (ctxSize []) i)
      rw [hidx, Vec.append_ofLp_natAdd]
    | right i => exact i.elim0
  | cons s Γ ih =>
    cases xs with
    | cons x xs =>
      change toVec (Pack.cons x (Pack.snoc xs y)) = _
      rw [toVec_cons, ih, toVec_cons, Vec.append_snoc]

/-- `TensorPack.toVec` maps `TorchLean.Pack.unsnoc` to `unsnoc`. -/
theorem CtxVec.unsnoc_toVec {Γ : List Shape} {τ : Shape} (w : TensorPack ℝ (Γ ++ [τ])) :
    unsnoc (Γ := Γ) (τ := τ) (TensorPack.toVec w)
      = (TensorPack.toVec (Pack.unsnoc w).1,
          toVec (t := (Pack.unsnoc w).2)) := by
  conv_lhs => rw [← Pack.snoc_unsnoc (ss := Γ) (τ := τ) (xs := w)]
  rw [TensorPack.toVec_snoc, unsnoc_snoc]

namespace Tape

namespace NodeData

/-- The vectorized forward map, evaluated on a flattened context. -/
theorem forwardVec_flattenCtx {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : NodeData ℝ Δ Γ τ) (d : Δ) (x : TensorPack ℝ Γ) :
    node.forwardVec (Γ := Γ) (τ := τ) d (TensorPack.toVec x) = toVec (t := node.forward x d) := by
  simp [NodeData.forwardVec]

/-- The vectorized JVP, evaluated on flattened contexts. -/
theorem jvpVec_flattenCtx {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : NodeData ℝ Δ Γ τ) (d : Δ) (x dx : TensorPack ℝ Γ) :
    node.jvpVec (Γ := Γ) (τ := τ) d (TensorPack.toVec x) (TensorPack.toVec dx)
      = toVec (t := node.jvp x dx d) := by
  simp [NodeData.jvpVec]

/-- The vectorized VJP, evaluated on a flattened context and a vectorized cotangent. -/
theorem vjpVec_flattenCtx {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : NodeData ℝ Δ Γ τ) (d : Δ) (x : TensorPack ℝ Γ) (δ : Tensor ℝ τ) :
    node.vjpVec (Γ := Γ) (τ := τ) d (TensorPack.toVec x) (toVec (t := δ))
      = TensorPack.toVec (node.vjp x d δ) := by
  simp [NodeData.vjpVec]

end NodeData

namespace GraphData

variable {Δ : Type} {Γ : List Shape}

/-- The Euclidean graph evaluation is the flattening of the `TensorPack` evaluation. -/
theorem evalVec_flattenCtx {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ)
    (x : TensorPack ℝ Γ) :
    evalVec (Γ := Γ) (ss := ss) g d (TensorPack.toVec x) =
      TensorPack.toVec (eval (Γ := Γ) (ss := ss) g x d) := by
  induction g with
  | nil =>
    simp [evalVec, eval, TensorPack.toVec_cast]
  | snoc g node ih =>
    simp [evalVec, eval, TensorPack.toVec_cast, TensorPack.toVec_snoc, ih,
      NodeData.forwardVec_flattenCtx]

/-- The Euclidean graph JVP is the flattening of the `TensorPack` JVP. -/
theorem jvpVec_flattenCtx {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ)
    (x dx : TensorPack ℝ Γ) :
    jvpVec (Γ := Γ) (ss := ss) g d (TensorPack.toVec x) (TensorPack.toVec dx)
      = TensorPack.toVec (jvpCtx (Γ := Γ) (ss := ss) g x dx d) := by
  induction g with
  | nil =>
    simp [jvpVec, jvpCtx, TensorPack.toVec_cast]
  | snoc g node ih =>
    simp [jvpVec, jvpCtx, TensorPack.toVec_cast, TensorPack.toVec_snoc, ih, evalVec_flattenCtx,
      NodeData.jvpVec_flattenCtx]

/-- The Euclidean reverse pass is the flattening of the `TensorPack` reverse pass. -/
theorem backpropVec_flattenCtx {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ)
    (x : TensorPack ℝ Γ) (seed : TensorPack ℝ (Γ ++ ss)) :
    backpropVec (Γ := Γ) (ss := ss) g d (TensorPack.toVec x) (TensorPack.toVec seed)
      = TensorPack.toVec (backpropCtx (Γ := Γ) (ss := ss) g x d seed) := by
  induction g with
  | nil =>
    simp [backpropVec, backpropCtx, TensorPack.toVec_cast]
  | snoc g node ih =>
    rename_i ss τ
    simp only [backpropVec]
    rw [← TensorPack.toVec_cast, CtxVec.unsnoc_toVec, evalVec_flattenCtx,
      NodeData.vjpVec_flattenCtx, ← TensorPack.toVec_add]
    exact ih _

end GraphData

namespace Graph

variable {Δ : Type} {Γ : List Shape}

/-- Vectorized tape soundness at the environment `d`: `⟪jvp, seed⟫ = ⟪dx, backprop seed⟫`. It is
`jvpCtx_backpropCtx_adjoint` read through `TensorPack.toVec`, a bijection carrying `dotList` to the
Euclidean inner product. -/
theorem jvpVec_backpropVec_adjoint {ss : List Shape} (g : Graph (α := ℝ) Δ Γ ss) (d : Δ) :
    ∀ xV dxV seedV,
      inner ℝ (g.toData.jvpVec d xV dxV) seedV =
        inner ℝ dxV (g.toData.backpropVec d xV seedV) := by
  intro xV dxV seedV
  obtain ⟨x, rfl⟩ : ∃ x, TensorPack.toVec x = xV := ⟨_, TensorPack.toVec_ofVec xV⟩
  obtain ⟨dx, rfl⟩ : ∃ dx, TensorPack.toVec dx = dxV := ⟨_, TensorPack.toVec_ofVec dxV⟩
  obtain ⟨seed, rfl⟩ : ∃ seed, TensorPack.toVec seed = seedV := ⟨_, TensorPack.toVec_ofVec seedV⟩
  rw [GraphData.jvpVec_flattenCtx, GraphData.backpropVec_flattenCtx,
    ← TensorPack.realDotList_eq_inner_toVec, ← TensorPack.realDotList_eq_inner_toVec]
  exact jvpCtx_backpropCtx_adjoint g x dx d seed

end Graph

end Tape

-- ---------------------------------------------------------------------------
-- Analytic correctness: `jvp = fderiv`, `backprop = (fderiv eval)†`
-- ---------------------------------------------------------------------------

/--
Per-node analytic correctness assumption at the environment `d`: the node's JVP is the Fréchet
derivative of its forward map at every point. With it, the dot-product soundness theorem becomes
an `fderiv` statement.
-/
structure Tape.RealNode.FDerivCorrect {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : NodeData ℝ Δ Γ τ) (d : Δ) where
  /-- The derivative packaged as a continuous linear map. -/
  deriv : CtxVec Γ → (CtxVec Γ →L[ℝ] Vec (Shape.size τ))
  /-- The forward map has the above derivative everywhere. -/
  hasFDerivAt : ∀ xV, HasFDerivAt (node.forwardVec (Γ := Γ) (τ := τ) d) (deriv xV) xV
  /-- The node's JVP function agrees with the packaged derivative. -/
  jvp_eq : ∀ xV dxV, node.jvpVec (Γ := Γ) (τ := τ) d xV dxV = (deriv xV) dxV

/-- A `RealNode.FDerivCorrect` for every node of the graph, at the environment `d`. -/
def Tape.RealGraph.FDerivCorrect {Δ : Type} {Γ : List Shape} :
    ∀ {ss : List Shape}, GraphData ℝ Δ Γ ss → Δ → Type
  | _, .nil => fun _ => PUnit
  | _, .snoc g node => fun d => FDerivCorrect g d × RealNode.FDerivCorrect node d

-- ---------------------------------------------------------------------------
-- Pointwise analytic correctness: allow per-node assumptions (ReLU kinks, log domain, …)
-- ---------------------------------------------------------------------------

/--
Per-node analytic correctness at one basepoint `xV`, for nodes that are differentiable only under
side conditions there (`inv`, `sqrt`, `log`, piecewise ops).
-/
structure Tape.RealNode.FDerivCorrectAt {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : NodeData ℝ Δ Γ τ) (d : Δ) (xV : CtxVec Γ) where
  /-- The Fréchet derivative at `xV`, as a continuous linear map on the flattened context. -/
  deriv : CtxVec Γ →L[ℝ] Vec (Shape.size τ)
  /-- `deriv` is the derivative of the node's forward map at `xV`. -/
  hasFDerivAt : HasFDerivAt (node.forwardVec (Γ := Γ) (τ := τ) d) deriv xV
  /-- The node's hand-written JVP agrees with `deriv`. This field ties the JVP that
  `GraphData.jvpVec` composes to the derivative in `hasFDerivAt`. -/
  jvp_eq : ∀ dxV, node.jvpVec (Γ := Γ) (τ := τ) d xV dxV = deriv dxV

/--
Specialize a global `FDerivCorrect` proof to the basepoint `xV`, for assembling
`RealGraph.FDerivCorrectAt` proofs from everywhere-differentiable nodes.
-/
def Tape.RealNode.FDerivCorrect.at {Δ : Type} {Γ : List Shape} {τ : Shape}
    {node : NodeData ℝ Δ Γ τ} {d : Δ}
    (hn : FDerivCorrect node d) (xV : CtxVec Γ) : FDerivCorrectAt node d xV :=
  ⟨hn.deriv xV, hn.hasFDerivAt xV, fun dxV => hn.jvp_eq xV dxV⟩

/--
A `Tape.RealNode.FDerivCorrectAt` for every node of the graph, at the intermediate value that
reaches it from `xV`. The recursion computes each node's basepoint with `Tape.GraphData.evalVec`.
-/
def Tape.RealGraph.FDerivCorrectAt {Δ : Type} {Γ : List Shape} :
    ∀ {ss : List Shape}, GraphData ℝ Δ Γ ss → Δ → CtxVec Γ → Type
  | _, .nil => fun _ _ => PUnit
  | _, .snoc g node => fun d xV =>
      FDerivCorrectAt g d xV ×
        RealNode.FDerivCorrectAt node d (GraphData.evalVec (Γ := Γ) g d xV)

/--
Specialize an everywhere-correct graph proof to a pointwise graph proof, the graph-level analogue
of `RealNode.FDerivCorrect.at`.

Each node certificate is read at the intermediate value that `Tape.GraphData.evalVec` produces
there. A globally smooth block can then feed a pointwise block such as LayerNorm, and a global
proof `hg` enters the pointwise `fderiv` theorems below as `hg.at xV`.
-/
def Tape.RealGraph.FDerivCorrect.at {Δ : Type} {Γ : List Shape} {ss : List Shape}
    {g : GraphData ℝ Δ Γ ss} {d : Δ}
    (hg : FDerivCorrect (Γ := Γ) g d) (xV : CtxVec Γ) :
    FDerivCorrectAt (Γ := Γ) (ss := ss) g d xV := by
  induction g with
  | nil =>
      exact PUnit.unit
  | @snoc ss τ g node ih =>
      rcases hg with ⟨hgPrefix, hn⟩
      exact ⟨ih hgPrefix,
        RealNode.FDerivCorrect.at hn (GraphData.evalVec (Γ := Γ) (ss := ss) g d xV)⟩

/-- Continuous linear map version of `CtxVec.snoc` (concatenation + cast). -/
def snocCLM {Γ : List Shape} {τ : Shape} :
    (CtxVec Γ × Vec (Shape.size τ)) →L[ℝ] CtxVec (Γ ++ [τ]) :=
  (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
      (finCongr (ctxSize_snoc Γ τ).symm)).toContinuousLinearEquiv.toContinuousLinearMap.comp
    (EuclideanSpace.finAddEquivProd (𝕜 := ℝ)).symm.toContinuousLinearMap

namespace Tape

namespace GraphData

variable {Δ : Type} {Γ : List Shape}

-- ---------------------------------------------------------------------------
-- Pointwise analytic statements: `HasFDerivAt` only at the actual execution point.
-- ---------------------------------------------------------------------------

/--
Main induction: under `RealGraph.FDerivCorrectAt`, `evalVec` has a derivative at `xV` that agrees
with `jvpVec`.

The `jvp = fderiv` theorem rests on it, and stating it pointwise means that graphs with non-smooth
or partial primitives need differentiability only at the values met during execution.
-/
theorem hasFDerivAt_evalVec_and_jvp
    {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ) :
    ∀ xV : CtxVec Γ,
      RealGraph.FDerivCorrectAt (Γ := Γ) (ss := ss) g d xV →
        ∃ D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ ss),
          HasFDerivAt (evalVec (Γ := Γ) (ss := ss) g d) D xV
            ∧
          (∀ dxV : CtxVec Γ, jvpVec (Γ := Γ) (ss := ss) g d xV dxV = D dxV) := by
  classical
  induction g with
  | nil =>
      intro xV _hg
      let h : ctxSize Γ = ctxSize (Γ ++ []) := congrArg ctxSize (List.append_nil Γ).symm
      let D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ []) :=
        (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
          (finCongr h)).toContinuousLinearEquiv.toContinuousLinearMap
      refine ⟨D, ?_, ?_⟩
      · change HasFDerivAt D D xV
        exact D.hasFDerivAt
      · intro dxV
        rfl
  | snoc g node ih =>
      intro xV hg
      rename_i ss τ
      rcases hg with ⟨hg_g, hg_node⟩
      rcases ih (xV := xV) hg_g with ⟨Dg, hDg, hJg⟩
      let ctxV : CtxVec (Γ ++ ss) := evalVec (Γ := Γ) (ss := ss) g d xV
      let Dn : CtxVec (Γ ++ ss) →L[ℝ] Vec (Shape.size τ) := hg_node.deriv
      have hy :
          HasFDerivAt
            (node.forwardVec (Γ := Γ ++ ss) (τ := τ) d)
            Dn ctxV := hg_node.hasFDerivAt
      have hpair :
          HasFDerivAt
            (fun x : CtxVec Γ =>
              (evalVec (Γ := Γ) (ss := ss) g d x,
                node.forwardVec (Γ := Γ ++ ss) (τ := τ) d (evalVec (Γ := Γ) (ss := ss) g d x)))
            (Dg.prod (Dn.comp Dg)) xV :=
        hDg.prodMk (hy.comp xV hDg)
      let assoc := List.append_assoc Γ ss [τ]
      let hAssoc : ctxSize ((Γ ++ ss) ++ [τ]) = ctxSize (Γ ++ (ss ++ [τ])) := congrArg ctxSize assoc
      let Dcast : CtxVec ((Γ ++ ss) ++ [τ]) →L[ℝ] CtxVec (Γ ++ (ss ++ [τ])) :=
        (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
          (finCongr hAssoc)).toContinuousLinearEquiv.toContinuousLinearMap
      let D : CtxVec Γ →L[ℝ] CtxVec (Γ ++ (ss ++ [τ])) :=
        (Dcast.comp (snocCLM (Γ := Γ ++ ss) (τ := τ))).comp (Dg.prod (Dn.comp Dg))
      refine ⟨D, ?_, ?_⟩
      · have hsnoc :
            HasFDerivAt
              (fun p : CtxVec (Γ ++ ss) × Vec (Shape.size τ) =>
                CtxVec.snoc (Γ := Γ ++ ss) (τ := τ) p.1 p.2)
              (snocCLM (Γ := Γ ++ ss) (τ := τ))
              (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) d ctxV) := by
          change HasFDerivAt
            (snocCLM (Γ := Γ ++ ss) (τ := τ))
            (snocCLM (Γ := Γ ++ ss) (τ := τ))
            (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) d ctxV)
          exact (snocCLM (Γ := Γ ++ ss) (τ := τ)).hasFDerivAt
        have hcomp1 := Dcast.hasFDerivAt.comp xV (hsnoc.comp xV hpair)
        change HasFDerivAt
          (Dcast ∘
            (fun p : CtxVec (Γ ++ ss) × Vec (Shape.size τ) =>
              CtxVec.snoc (Γ := Γ ++ ss) (τ := τ) p.1 p.2) ∘
            fun xV : CtxVec Γ =>
              (evalVec (Γ := Γ) (ss := ss) g d xV,
                node.forwardVec (Γ := Γ ++ ss) (τ := τ) d (evalVec (Γ := Γ) (ss := ss) g d xV)))
          D xV
        rw [show D = Dcast.comp ((snocCLM (Γ := Γ ++ ss) (τ := τ)).comp (Dg.prod (Dn.comp Dg))) from
          ContinuousLinearMap.comp_assoc _ _ _]
        exact hcomp1
      · intro dxV
        have hx : jvpVec (Γ := Γ) (ss := ss) g d xV dxV = Dg dxV := hJg dxV
        have hy' :
            node.jvpVec (Γ := Γ ++ ss) (τ := τ) d ctxV (Dg dxV) = (Dn.comp Dg) dxV := by
          simpa [Dn, ContinuousLinearMap.comp_apply] using (hg_node.jvp_eq (dxV := Dg dxV))
        change CtxVec.cast assoc (CtxVec.snoc (Γ := Γ ++ ss) (τ := τ) (jvpVec g d xV dxV)
            (node.jvpVec (Γ := Γ ++ ss) (τ := τ) d ctxV (jvpVec g d xV dxV))) = D dxV
        rw [hx, hy']
        rfl

/-!
Pointwise corollaries of `hasFDerivAt_evalVec_and_jvp`. The identity `jvpVec = fderiv` is immediate,
and the chain rule composes `evalVec` with any map that is differentiable at the point. For a
proof-carrying graph, `backpropVec = (fderiv evalVec)†` comes from the inner-product
characterization of the adjoint and the vectorized soundness law
`Graph.jvpVec_backpropVec_adjoint`.
-/

/-- Under `RealGraph.FDerivCorrectAt`, the graph JVP at `xV` equals the Fréchet derivative of
`evalVec` at `xV`. -/
theorem jvpVec_eq_fderiv
    {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ) :
    ∀ xV dxV,
      RealGraph.FDerivCorrectAt (Γ := Γ) (ss := ss) g d xV →
        jvpVec (Γ := Γ) (ss := ss) g d xV dxV
          =
        (fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g d) xV) dxV := by
  intro xV dxV hg
  obtain ⟨D, hD, hJ⟩ :=
    hasFDerivAt_evalVec_and_jvp (Γ := Γ) (ss := ss) (g := g) (d := d) (xV := xV) hg
  rw [hD.fderiv]
  exact hJ dxV

/-- Chain rule through a graph: if `f` has derivative `Df` at `x` and the graph's certificate holds
at `f x`, then `evalVec g d ∘ f` has derivative `(fderiv ℝ (evalVec g d) (f x)).comp Df` at `x`. -/
theorem hasFDerivAt_evalVec_comp {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {ss : List Shape} (g : GraphData ℝ Δ Γ ss) (d : Δ) {f : E → CtxVec Γ}
    {Df : E →L[ℝ] CtxVec Γ} (x : E)
    (hg : RealGraph.FDerivCorrectAt (Γ := Γ) (ss := ss) g d (f x)) (hf : HasFDerivAt f Df x) :
    HasFDerivAt (evalVec (Γ := Γ) (ss := ss) g d ∘ f)
      ((fderiv ℝ (evalVec (Γ := Γ) (ss := ss) g d) (f x)).comp Df) x := by
  obtain ⟨D, hD, -⟩ := hasFDerivAt_evalVec_and_jvp (Γ := Γ) (ss := ss) g d (f x) hg
  rw [hD.fderiv]
  exact hD.comp x hf

end GraphData

namespace Graph

variable {Δ : Type} {Γ : List Shape}

/-- Main analytic theorem: under `RealGraph.FDerivCorrectAt`, `backpropVec` at `xV` is the adjoint
of the derivative of `evalVec` at `xV`. -/
theorem backpropVec_eq_fderiv
    {ss : List Shape} (g : Graph (α := ℝ) Δ Γ ss) (d : Δ) :
    ∀ (xV : CtxVec Γ) (seedV : CtxVec (Γ ++ ss)),
      RealGraph.FDerivCorrectAt (Γ := Γ) (ss := ss) g.toData d xV →
        g.toData.backpropVec d xV seedV
          =
        (fderiv ℝ (g.toData.evalVec d) xV).adjoint seedV := by
  intro xV seedV hg
  obtain ⟨D, hD, hJ⟩ := GraphData.hasFDerivAt_evalVec_and_jvp g.toData d xV hg
  rw [hD.fderiv]
  -- Both sides have the same inner product against every tangent `dxV`: the adjoint by
  -- definition, `backpropVec` by tape soundness once `jvpVec` is identified with `D`.
  refine ext_inner_left ℝ fun dxV => ?_
  rw [ContinuousLinearMap.adjoint_inner_right, ← hJ dxV]
  exact (Graph.jvpVec_backpropVec_adjoint (Γ := Γ) (ss := ss) g d xV dxV seedV).symm

end Graph

end Tape

end
end TorchLean
