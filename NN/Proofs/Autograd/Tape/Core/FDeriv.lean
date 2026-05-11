/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Core.Vectorization
public import NN.Proofs.Autograd.Tape.Core.Soundness

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Analysis.Calculus.FDeriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Linear
public import Mathlib.Analysis.Calculus.FDeriv.Prod
public import Mathlib.Analysis.InnerProductSpace.Adjoint
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.Normed.Module.FiniteDimension
public import Mathlib.Data.Fin.Tuple.Basic
public import Mathlib.Data.Fintype.Sum
public import Mathlib.Logic.Equiv.Fin.Basic

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

open Spec
open Tensor

open scoped BigOperators

noncomputable section

-- ---------------------------------------------------------------------------
-- Tensor vectorization (`Tensor ℝ s` ↔ `Vec (Shape.size s)`)
-- ---------------------------------------------------------------------------

/-- Vectorize a tensor by flattening it (spec flattening order) and then using the Euclidean
  equivalence. -/
def toVecT {s : Shape} (t : Tensor ℝ s) : Vec (Shape.size s) :=
  toVecE (n := Shape.size s) (flattenSpec (α := ℝ) t)

/-- Inverse of `toVecT`: interpret a vector as a tensor of shape `s`. -/
def ofVecT {s : Shape} (v : Vec (Shape.size s)) : Tensor ℝ s :=
  unflattenSpec (α := ℝ) s (ofVecE (n := Shape.size s) v)

@[simp] lemma toVecT_ofVecT {s : Shape} (v : Vec (Shape.size s)) :
    toVecT (t := ofVecT (s := s) v) = v := by
  have hunf :
      flattenSpec (α := ℝ)
          (unflattenSpec (α := ℝ) s (ofVecE (n := Shape.size s) v))
        =
      ofVecE (n := Shape.size s) v :=
    unflatten_flatten_inverse (s := s) (v := ofVecE (n := Shape.size s) v)
  have := congrArg (toVecE (n := Shape.size s)) hunf
  simpa [toVecT, ofVecT] using this.trans (toVecE_ofVecE (n := Shape.size s) v)

@[simp] lemma ofVecT_toVecT {s : Shape} (t : Tensor ℝ s) :
    ofVecT (s := s) (toVecT (t := t)) = t := by
  have hround :
      ofVecE (n := Shape.size s) (toVecE (n := Shape.size s) (flattenSpec (α := ℝ) t))
        =
      flattenSpec (α := ℝ) t := by
    simp
  simpa [ofVecT, toVecT, hround] using (flatten_unflatten_inverse (s := s) (t := t))

-- ---------------------------------------------------------------------------
-- Context vectorization (`TList Γ` ↔ `Vec (ctxSize Γ)`)
-- ---------------------------------------------------------------------------

/-- Total number of scalar coordinates in a heterogeneous context shape list. -/
def ctxSize : List Shape → Nat
  | [] => 0
  | s :: ss => Shape.size s + ctxSize ss

/-- A vectorized context: one Euclidean vector containing all `TList Γ` entries concatenated. -/
abbrev CtxVec (Γ : List Shape) := Vec (ctxSize Γ)

/--
Build a Euclidean vector from its coordinate function.

This is the small helper used throughout the file when reindexing vectors across context
concatenation and shape casts.
-/
def vecOfFun {n : Nat} (f : Fin n → ℝ) : Vec n :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm f

@[simp] lemma vecOfFun_apply {n : Nat} (f : Fin n → ℝ) (i : Fin n) :
    vecOfFun (n := n) f i = f i := by
  simp [vecOfFun, EuclideanSpace.equiv]

@[simp] lemma vecOfFun_eta {n : Nat} (v : Vec n) :
    vecOfFun (n := n) (fun i => v i) = v := by
  classical
  -- `vecOfFun` is `EuclideanSpace.equiv.symm`, and the forward map is definitionally `fun i => v
  -- i`.
  simp [vecOfFun, EuclideanSpace.equiv]

/--
Flatten a typed context `TList Γ` into one big Euclidean vector.

Unlike PyTorch’s runtime “saved tensor list”, this is an actual typed isomorphism: shapes are
tracked in `Γ`, so the split points are definitional from `ctxSize`.
-/
def flattenCtx : {Γ : List Shape} → TList Γ → CtxVec Γ
  | [], .nil => 0
  | s :: ss, .cons x xs =>
      vecOfFun (n := Shape.size s + ctxSize ss) (Fin.append (toVecT x) (flattenCtx (Γ := ss) xs))

/-- Inverse of `flattenCtx`: split a `CtxVec Γ` back into a `TList Γ`. -/
def unflattenCtx : {Γ : List Shape} → CtxVec Γ → TList Γ
  | [], _ => .nil
  | s :: ss, v =>
      let head : Vec (Shape.size s) :=
        vecOfFun (n := Shape.size s) fun i => v (Fin.castAdd (ctxSize ss) i)
      let tail : Vec (ctxSize ss) :=
        vecOfFun (n := ctxSize ss) fun i => v (Fin.natAdd (Shape.size s) i)
      .cons (ofVecT (s := s) head) (unflattenCtx (Γ := ss) tail)

@[simp] theorem unflattenCtx_flattenCtx {Γ : List Shape} (xs : TList Γ) :
    unflattenCtx (Γ := Γ) (flattenCtx (Γ := Γ) xs) = xs := by
  induction Γ with
  | nil =>
      cases xs
      rfl
  | cons s ss ih =>
      cases xs with
      | cons x xs =>
          simp [flattenCtx, unflattenCtx, ih]

@[simp] theorem flattenCtx_unflattenCtx {Γ : List Shape} (v : CtxVec Γ) :
    flattenCtx (Γ := Γ) (unflattenCtx (Γ := Γ) v) = v := by
  induction Γ with
  | nil =>
      ext i
      exact i.elim0
  | cons s ss ih =>
      ext i
      -- reduce to the `Fin.append_castAdd_natAdd` lemma on the underlying functions
      simpa [flattenCtx, unflattenCtx, ih] using
        congrArg (fun f : Fin (Shape.size s + ctxSize ss) → ℝ => f i)
          (Fin.append_castAdd_natAdd (f := v) (m := Shape.size s) (n := ctxSize ss))

-- ---------------------------------------------------------------------------
-- Dot/inner compatibility (`dotList` ↔ Euclidean inner product)
-- ---------------------------------------------------------------------------

/-- Cast a `Vec n` to `Vec m` along an equality, by reindexing coordinates. -/
def castVec {n m : Nat} (h : n = m) : Vec n → Vec m :=
  fun v => vecOfFun (n := m) fun i => v (Fin.cast h.symm i)

@[simp] lemma castVec_apply {n m : Nat} (h : n = m) (v : Vec n) (i : Fin m) :
    castVec (n := n) (m := m) h v i = v (Fin.cast h.symm i) := by
  simp [castVec]

@[simp] lemma castVec_rfl {n : Nat} (v : Vec n) : castVec (n := n) (m := n) rfl v = v := by
  ext i
  simp [castVec]

@[simp] lemma castVec_add {n m : Nat} (h : n = m) (u v : Vec n) :
    castVec (n := n) (m := m) h (u + v) = castVec (n := n) (m := m) h u + castVec (n := n) (m := m)
      h v := by
  ext i
  simp []

@[simp] lemma castVec_smul {n m : Nat} (h : n = m) (r : ℝ) (v : Vec n) :
    castVec (n := n) (m := m) h (r • v) = r • castVec (n := n) (m := m) h v := by
  ext i
  simp [smul_eq_mul]

@[simp] lemma castVec_castVec {n m k : Nat} (h₁ : n = m) (h₂ : m = k) (v : Vec n) :
    castVec h₂ (castVec h₁ v) = castVec (h₁.trans h₂) v := by
  cases h₁
  cases h₂
  ext i
  simp [castVec]

/--
`castVec` preserves the Euclidean inner product.

This is the core “cast isometry” lemma used throughout the vectorized graph development.
-/
lemma inner_castVec_castVec {n m : Nat} (h : n = m) (x y : Vec n) :
    inner ℝ (castVec h x) (castVec h y) = inner ℝ x y := by
  cases h
  simp [castVec]

/--
`sum_spec` over an outer dimension is a sum over slices.

This tensor-level “Fubini rule” is used to relate `Spec.dot` to Euclidean inner products after
vectorization.
-/
lemma sum_spec_dim {n : Nat} {s : Shape} (values : Fin n → Tensor ℝ s) :
    sumSpec (Tensor.dim values) = ∑ i : Fin n, sumSpec (values i) := by
  classical
  let g : Fin n → ℝ := fun i => sumSpec (values i)
  have go_eq :
      ∀ k acc, k ≤ n →
        tensorFoldlSpec.go (· + ·) n s values k acc =
          acc + (Finset.univ.filter (fun i : Fin n => k ≤ i.val)).sum g := by
    intro k acc hk
    induction hn : n - k generalizing k acc with
    | zero =>
        have hk' : k = n := by
          have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
          exact Nat.le_antisymm hk this
        subst k
        simp [tensorFoldlSpec.go]
        have hfilter :
            (Finset.univ.filter (fun i : Fin n => n ≤ i.val)) = (∅ : Finset (Fin n)) := by
          ext i
          simp [Nat.not_le_of_lt i.isLt]
        simp [hfilter]
    | succ m ih =>
        have hlt : k < n := by
          have : 0 < n - k := by
            simp [hn]
          exact Nat.sub_pos_iff_lt.mp this
        have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
        -- Peel one `go` loop step at index `k`.
        rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := values) (k := k) (acc := acc) hlt]
        have hstep :
            tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩) = acc + g ⟨k, hlt⟩ := by
          -- folding with addition adds `sum_spec` to the accumulator
          simpa [g] using
            (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := values ⟨k, hlt⟩))
        have h_next : n - (k + 1) = m := by
          rw [Nat.sub_succ, hn]
          rfl
        have ih' := ih (k := k + 1) (acc := acc + g ⟨k, hlt⟩) hk1
        have ih'' :
            tensorFoldlSpec.go (· + ·) n s values (k + 1) (acc + g ⟨k, hlt⟩) =
              (acc + g ⟨k, hlt⟩) +
                (Finset.univ.filter (fun i : Fin n => k + 1 ≤ i.val)).sum g := by
          simpa [h_next] using ih'
        let Sk : Finset (Fin n) := Finset.univ.filter (fun i : Fin n => k ≤ i.val)
        let Sk1 : Finset (Fin n) := Finset.univ.filter (fun i : Fin n => k + 1 ≤ i.val)
        have hSk : Sk = insert (⟨k, hlt⟩ : Fin n) Sk1 := by
          ext i
          constructor
          · intro hiSk
            have hle : k ≤ i.val := by
              simpa [Sk] using hiSk
            have hcase : k = i.val ∨ k < i.val := Nat.eq_or_lt_of_le hle
            refine (Finset.mem_insert).2 ?_
            cases hcase with
            | inl hEq =>
                left
                apply Fin.ext
                exact hEq.symm
            | inr hLt =>
                right
                have hk1' : k + 1 ≤ i.val := Nat.succ_le_of_lt hLt
                simpa [Sk1] using hk1'
          · intro hiIns
            have hi' : i = (⟨k, hlt⟩ : Fin n) ∨ i ∈ Sk1 := (Finset.mem_insert).1 hiIns
            cases hi' with
            | inl hEq =>
                subst hEq
                simp [Sk]
            | inr hiSk1 =>
                have hk1' : k + 1 ≤ i.val := by
                  simpa [Sk1] using hiSk1
                have hle : k ≤ i.val := Nat.le_trans (Nat.le_succ k) hk1'
                simpa [Sk] using hle
        have hk_not_mem1 : (⟨k, hlt⟩ : Fin n) ∉ Sk1 := by
          simp [Sk1]
        have hSk_sum : Sk.sum g = g ⟨k, hlt⟩ + Sk1.sum g := by
          have :
              (insert (⟨k, hlt⟩ : Fin n) Sk1).sum g = g ⟨k, hlt⟩ + Sk1.sum g := by
            simpa using
              (Finset.sum_insert (s := Sk1) (a := (⟨k, hlt⟩ : Fin n)) (f := g) hk_not_mem1)
          simpa [hSk] using this
        calc
          tensorFoldlSpec.go (· + ·) n s values (k + 1)
              (tensorFoldlSpec (· + ·) acc (values ⟨k, hlt⟩))
              =
            tensorFoldlSpec.go (· + ·) n s values (k + 1) (acc + g ⟨k, hlt⟩) := by
              simp [hstep]
          _ = (acc + g ⟨k, hlt⟩) + Sk1.sum g := ih''
          _ = acc + (g ⟨k, hlt⟩ + Sk1.sum g) := by
              ring
          _ = acc + Sk.sum g := by
              simp [hSk_sum, Sk]
  have hfilter0 :
      (Finset.univ.filter (fun i : Fin n => (0 : Nat) ≤ i.val)) = (Finset.univ : Finset (Fin n)) :=
        by
    ext i
    simp
  have h0 := go_eq (k := 0) (acc := (0 : ℝ)) (Nat.zero_le n)
  simpa [sumSpec, tensorFoldlSpec, g, hfilter0] using h0

-- Decompose `toVecT` along `finProdFinEquiv` when the inner size is positive.
/--
Coordinate characterization of `toVecT` on a tensor `.dim n s`.

Informally, the vectorization order is the standard product order induced by `finProdFinEquiv`.
-/
lemma toVecT_dim_apply {n : Nat} {s : Shape} (hmpos : 0 < Shape.size s)
    (f : Fin n → Tensor ℝ s) (p : Fin n × Fin (Shape.size s)) :
    toVecT (t := Tensor.dim f) (finProdFinEquiv p) = toVecT (t := f p.1) p.2 := by
  classical
  let m : Nat := Shape.size s
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
  cases hfi : flattenSpec (α := ℝ) (f p.1) with
  | dim gfi =>
      simp [toVecT, toVecE, Spec.toVec, flattenSpec, m, hfi, hdiv, hmod]

-- Inner product decomposition across an outer dimension.
/-- `toVecT` turns dot products on `.dim n s` into sums of Euclidean inner products over slices. -/
lemma inner_toVecT_dim {n : Nat} {s : Shape} (a b : Fin n → Tensor ℝ s) :
    inner ℝ (toVecT (t := Tensor.dim a)) (toVecT (t := Tensor.dim b))
      =
    ∑ i : Fin n, inner ℝ (toVecT (t := a i)) (toVecT (t := b i)) := by
  classical
  by_cases hm : Shape.size s = 0
  · have hmul : n * Shape.size s = 0 := by simp [hm]
    -- LHS: transport to `Fin 0`.
    have hL :
        inner ℝ (toVecT (t := Tensor.dim a)) (toVecT (t := Tensor.dim b)) = 0 := by
      let e : Fin 0 ≃ Fin (n * Shape.size s) := Equiv.cast (congrArg Fin hmul.symm)
      calc
        inner ℝ (toVecT (t := Tensor.dim a)) (toVecT (t := Tensor.dim b))
            =
          ∑ i : Fin (n * Shape.size s),
            toVecT (t := Tensor.dim a) i * toVecT (t := Tensor.dim b) i := by
              simpa using
                inner_eq_sum_mul (x := toVecT (t := Tensor.dim a)) (y := toVecT (t := Tensor.dim b))
        _ =
          ∑ i : Fin 0,
            toVecT (t := Tensor.dim a) (e i) * toVecT (t := Tensor.dim b) (e i) := by
              simpa using
                (Equiv.sum_comp (e := e)
                  (g := fun i : Fin (n * Shape.size s) =>
                    toVecT (t := Tensor.dim a) i * toVecT (t := Tensor.dim b) i)).symm
        _ = 0 := by simp
    have hterm : ∀ i : Fin n, inner ℝ (toVecT (t := a i)) (toVecT (t := b i)) = 0 := by
      intro i
      let e : Fin 0 ≃ Fin (Shape.size s) := Equiv.cast (congrArg Fin hm.symm)
      calc
        inner ℝ (toVecT (t := a i)) (toVecT (t := b i))
            =
          ∑ j : Fin (Shape.size s),
            toVecT (t := a i) j * toVecT (t := b i) j := by
              simpa using inner_eq_sum_mul (x := toVecT (t := a i)) (y := toVecT (t := b i))
        _ =
          ∑ j : Fin 0,
            toVecT (t := a i) (e j) * toVecT (t := b i) (e j) := by
              simpa using
                (Equiv.sum_comp (e := e)
                  (g := fun j : Fin (Shape.size s) =>
                    toVecT (t := a i) j * toVecT (t := b i) j)).symm
        _ = 0 := by simp
    have hR :
        (∑ i : Fin n, inner ℝ (toVecT (t := a i)) (toVecT (t := b i))) = 0 := by
      simp [hterm]
    simp [hL, hR]
  · have hmpos : 0 < Shape.size s := Nat.pos_of_ne_zero hm
    calc
      inner ℝ (toVecT (t := Tensor.dim a)) (toVecT (t := Tensor.dim b))
          =
        ∑ i : Fin (n * Shape.size s),
          toVecT (t := Tensor.dim a) i * toVecT (t := Tensor.dim b) i := by
            simpa using
              inner_eq_sum_mul (x := toVecT (t := Tensor.dim a)) (y := toVecT (t := Tensor.dim b))
      _ =
        ∑ p : Fin n × Fin (Shape.size s),
          toVecT (t := Tensor.dim a) (finProdFinEquiv p) *
            toVecT (t := Tensor.dim b) (finProdFinEquiv p) := by
          simpa using
            (Equiv.sum_comp (e := finProdFinEquiv)
              (g := fun i : Fin (n * Shape.size s) =>
                toVecT (t := Tensor.dim a) i * toVecT (t := Tensor.dim b) i)).symm
      _ =
        ∑ p : Fin n × Fin (Shape.size s),
          toVecT (t := a p.1) p.2 * toVecT (t := b p.1) p.2 := by
          refine Finset.sum_congr rfl ?_
          intro p _
          simp [toVecT_dim_apply (hmpos := hmpos)]
      _ =
        ∑ i : Fin n, ∑ j : Fin (Shape.size s),
          toVecT (t := a i) j * toVecT (t := b i) j := by
          simp [Fintype.sum_prod_type]
      _ =
        ∑ i : Fin n, inner ℝ (toVecT (t := a i)) (toVecT (t := b i)) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          simpa using (inner_eq_sum_mul (x := toVecT (t := a i)) (y := toVecT (t := b i))).symm

-- Dot product agrees with the Euclidean inner product after vectorization.
/--
Main compatibility lemma: tensor dot equals Euclidean inner product of vectorizations.

This is the bridge between `soundness.lean` (stated using `Spec.dot`) and the analytic theorems
here (stated using Euclidean `inner`).
-/
theorem dot_eq_inner_toVecT {s : Shape} (a b : Tensor ℝ s) :
    dot a b = inner ℝ (toVecT (t := a)) (toVecT (t := b)) := by
  classical
  induction s with
  | scalar =>
      cases a with
      | scalar x =>
          cases b with
          | scalar y =>
              let vx : Vec 1 := toVecT (t := Tensor.scalar x)
              let vy : Vec 1 := toVecT (t := Tensor.scalar y)
              have hinner : inner ℝ vx vy = ∑ i : Fin 1, vx i * vy i :=
                inner_eq_sum_mul (x := vx) (y := vy)
              simpa [vx, vy, dot, sumSpec, tensorFoldlSpec, mulSpec, map2Spec, toVecT, toVecE,
                Spec.toVec, flattenSpec, Shape.size] using hinner.symm
  | dim n s ih =>
      cases a with
      | dim fa =>
          cases b with
          | dim fb =>
              have hdot :
                  dot (Tensor.dim fa) (Tensor.dim fb) = ∑ i : Fin n, dot (fa i) (fb i) := by
                simp [dot, mulSpec, map2Spec, sum_spec_dim]
              have hinter :
                  inner ℝ (toVecT (t := Tensor.dim fa)) (toVecT (t := Tensor.dim fb))
                    =
                  ∑ i : Fin n, inner ℝ (toVecT (t := fa i)) (toVecT (t := fb i)) :=
                inner_toVecT_dim (a := fa) (b := fb)
              calc
                dot (Tensor.dim fa) (Tensor.dim fb)
                    = ∑ i : Fin n, inner ℝ (toVecT (t := fa i)) (toVecT (t := fb i)) := by
                        refine (hdot.trans ?_)
                        refine Finset.sum_congr rfl ?_
                        intro i _
                        simpa using (ih (a := fa i) (b := fb i))
                _ = inner ℝ (toVecT (t := Tensor.dim fa)) (toVecT (t := Tensor.dim fb)) := by
                        simp [hinter]

/-- Concatenate two Euclidean vectors using `Fin.append`. -/
def appendVec {m n : Nat} (a : Vec m) (b : Vec n) : Vec (m + n) :=
  vecOfFun (n := m + n) (Fin.append a b)

/-- Inner product of concatenated vectors splits as a sum of inner products. -/
lemma inner_append {m n : Nat} (a c : Vec m) (b d : Vec n) :
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
`TList.dotList` equals Euclidean inner product of `flattenCtx`.

This shows that the “context inner product” used in tape soundness is exactly the Euclidean inner
product on the vectorized context representation.
-/
theorem dotList_eq_inner_flattenCtx {Γ : List Shape} (x y : TList Γ) :
    TList.dotList (ss := Γ) x y = inner ℝ (flattenCtx (Γ := Γ) x) (flattenCtx (Γ := Γ) y) := by
  classical
  induction Γ with
  | nil =>
      cases x
      cases y
      simp [TList.dotList, flattenCtx]
  | cons s ss ih =>
      cases x with
      | cons xh xt =>
          cases y with
          | cons yh yt =>
              -- Split the inner product across the append, then use IH and `dot_eq_inner_toVecT`.
              have hinter :
                  inner ℝ (flattenCtx (Γ := s :: ss) (TList.cons xh xt))
                        (flattenCtx (Γ := s :: ss) (TList.cons yh yt))
                    =
                  inner ℝ (toVecT (t := xh)) (toVecT (t := yh))
                    + inner ℝ (flattenCtx (Γ := ss) xt) (flattenCtx (Γ := ss) yt) := by
                simpa [flattenCtx] using
                  (inner_append (m := Shape.size s) (n := ctxSize ss)
                    (a := toVecT (t := xh)) (c := toVecT (t := yh))
                    (b := flattenCtx (Γ := ss) xt) (d := flattenCtx (Γ := ss) yt))
              calc
                TList.dotList (ss := s :: ss) (TList.cons xh xt) (TList.cons yh yt)
                    = dot xh yh + TList.dotList (ss := ss) xt yt := by
                        simp [TList.dotList]
                _ = inner ℝ (toVecT (t := xh)) (toVecT (t := yh))
                      + inner ℝ (flattenCtx (Γ := ss) xt) (flattenCtx (Γ := ss) yt) := by
                        simp [dot_eq_inner_toVecT, ih]
                _ = inner ℝ (flattenCtx (Γ := s :: ss) (TList.cons xh xt))
                        (flattenCtx (Γ := s :: ss) (TList.cons yh yt)) := by
                        simp [hinter]

-- ---------------------------------------------------------------------------
-- Vector graph semantics (for calculus)
-- ---------------------------------------------------------------------------

/-- Cast a vectorized context along an equality of shape lists (reindexing coordinates). -/
def castCtxVec {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) : CtxVec Γ₁ → CtxVec Γ₂ :=
  castVec (congrArg ctxSize h)

@[simp] lemma castCtxVec_rfl {Γ : List Shape} (v : CtxVec Γ) : castCtxVec (Γ₁ := Γ) (Γ₂ := Γ) rfl v
  = v := by
  simp [castCtxVec]

@[simp] lemma castCtxVec_cast {Γ₁ Γ₂ Γ₃ : List Shape} (h₁ : Γ₁ = Γ₂) (h₂ : Γ₂ = Γ₃) (v : CtxVec Γ₁)
  :
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
lemma inner_castCtxVec {Γ₁ Γ₂ : List Shape} (h : Γ₁ = Γ₂) (x : CtxVec Γ₁) (y : CtxVec Γ₂) :
    inner ℝ (castCtxVec (Γ₁ := Γ₁) (Γ₂ := Γ₂) h x) y
      =
    inner ℝ x (castCtxVec (Γ₁ := Γ₂) (Γ₂ := Γ₁) h.symm y) := by
  cases h
  simp [castCtxVec, castVec]

/-- `ctxSize` respects list append (sizes add). -/
lemma ctxSize_append (Γ ss : List Shape) : ctxSize (Γ ++ ss) = ctxSize Γ + ctxSize ss := by
  induction Γ with
  | nil => simp [ctxSize]
  | cons s Γ ih => simp [ctxSize, ih, Nat.add_assoc]

/-- Specialized `ctxSize_append` for snoc (`Γ ++ [τ]`). -/
lemma ctxSize_snoc (ss : List Shape) (τ : Shape) :
    ctxSize (ss ++ [τ]) = ctxSize ss + Shape.size τ := by
  -- `ctxSize [τ] = Shape.size τ`.
  simp [ctxSize, ctxSize_append]

/-- Append one tensor-vector block to a vectorized context. -/
def snocCtx {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ) (t : Vec (Shape.size τ)) : CtxVec (Γ ++
  [τ]) :=
  castVec (ctxSize_snoc Γ τ).symm (appendVec (m := ctxSize Γ) (n := Shape.size τ) ctx t)

/-- Inverse of `snocCtx`: split `CtxVec (Γ ++ [τ])` into its prefix and last block. -/
def unsnocCtx {Γ : List Shape} {τ : Shape} (ctx : CtxVec (Γ ++ [τ])) : CtxVec Γ × Vec (Shape.size τ)
  :=
  let ctx' : Vec (ctxSize Γ + Shape.size τ) := castVec (ctxSize_snoc Γ τ) ctx
  let head : CtxVec Γ := vecOfFun (n := ctxSize Γ) fun i => ctx' (Fin.castAdd (Shape.size τ) i)
  let last : Vec (Shape.size τ) := vecOfFun (n := Shape.size τ) fun i => ctx' (Fin.natAdd (ctxSize
    Γ) i)
  (head, last)

/-- `unsnocCtx (snocCtx ctx t) = (ctx, t)`. -/
theorem unsnocCtx_snocCtx {Γ : List Shape} {τ : Shape} (ctx : CtxVec Γ) (t : Vec (Shape.size τ)) :
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
      appendVec (m := ctxSize Γ) (n := Shape.size τ)
          (vecOfFun (n := ctxSize Γ) fun i => (castVec (ctxSize_snoc Γ τ) ctx) (Fin.castAdd
            (Shape.size τ) i))
          (vecOfFun (n := Shape.size τ) fun i => (castVec (ctxSize_snoc Γ τ) ctx) (Fin.natAdd
            (ctxSize Γ) i))
        =
      castVec (ctxSize_snoc Γ τ) ctx := by
    ext i
    simp [appendVec, Fin.append, Fin.addCases, vecOfFun]
  -- Apply the cast back to `(ctxSize Γ + size τ)` and then cancel the cast pair.
  simpa [unsnocCtx, snocCtx, hcancel, happ] using
    (congrArg (castVec (ctxSize_snoc Γ τ).symm) happ).trans hcancel

namespace Node

/-- Vectorized forward map of a tape `Node`: `CtxVec Γ → Vec (Shape.size τ)`. -/
def forwardVec {Γ : List Shape} {τ : Shape} (node : Node Γ τ) : CtxVec Γ → Vec (Shape.size τ) :=
  fun ctxV => toVecT (t := node.forward (unflattenCtx (Γ := Γ) ctxV))

/-- Vectorized JVP of a tape `Node`: the node-level forward-mode action on tangents. -/
def jvpVec {Γ : List Shape} {τ : Shape} (node : Node Γ τ) : CtxVec Γ → CtxVec Γ → Vec (Shape.size τ)
  :=
  fun ctxV dctxV =>
    toVecT (t := node.jvp (unflattenCtx (Γ := Γ) ctxV) (unflattenCtx (Γ := Γ) dctxV))

/-- Vectorized VJP of a tape `Node`: pushes a cotangent vector back to the input context. -/
def vjpVec {Γ : List Shape} {τ : Shape} (node : Node Γ τ) : CtxVec Γ → Vec (Shape.size τ) → CtxVec Γ
  :=
  fun ctxV δV =>
    flattenCtx (Γ := Γ) (node.vjp (unflattenCtx (Γ := Γ) ctxV) (ofVecT (s := τ) δV))

/--
Vectorized form of `Node.correct` (adjointness law).

Statement: `⟪jvp(x,dx), δ⟫ = ⟪dx, vjp(x,δ)⟫`.
-/
theorem correct_inner {Γ : List Shape} {τ : Shape} (node : Node Γ τ) :
    ∀ (ctxV dctxV : CtxVec Γ) (δV : Vec (Shape.size τ)),
      inner ℝ (node.jvpVec ctxV dctxV) δV = inner ℝ dctxV (node.vjpVec ctxV δV) := by
  intro ctxV dctxV δV
  let ctx := unflattenCtx (Γ := Γ) ctxV
  let dctx := unflattenCtx (Γ := Γ) dctxV
  let δ := ofVecT (s := τ) δV
  have hdot := node.correct ctx dctx δ
  -- Convert `dot`/`dotList` to inner products.
  have hleft : dot (node.jvp ctx dctx) δ = inner ℝ (node.jvpVec ctxV dctxV) δV := by
    simp [Node.jvpVec, ctx, dctx, δ, dot_eq_inner_toVecT]
  have hright :
      TList.dotList (ss := Γ) dctx (node.vjp ctx δ) = inner ℝ dctxV (node.vjpVec ctxV δV) := by
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
      let yV : Vec (Shape.size τ) := node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV
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
      let dyV : Vec (Shape.size τ) := node.jvpVec (Γ := Γ ++ ss) (τ := τ) ctxV dctxV
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
      let seedOutV : Vec (Shape.size τ) := (unsnocCtx (Γ := (Γ ++ ss)) (τ := τ) seedV').2
      let ctxV : CtxVec (Γ ++ ss) := evalVec (ss := ss) g xV
      let contribV : CtxVec (Γ ++ ss) := node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV seedOutV
      backpropVec (ss := ss) g xV (seedPrevV + contribV)

/-!
The next theorem is exactly `soundness.lean` rewritten into Euclidean vector form.
It is the key input to later “`backprop = (fderiv eval)†`” proofs.
-/

/-- Vectorized tape soundness: `⟪jvp, seed⟫ = ⟪dx, backprop seed⟫`. -/
theorem backprop_correct_inner {ss : List Shape} (g : Graph Γ ss) :
    ∀ xV dxV seedV,
      inner ℝ (jvpVec (Γ := Γ) (ss := ss) g xV dxV) seedV =
        inner ℝ dxV (backpropVec (Γ := Γ) (ss := ss) g xV seedV) := by
  classical
  induction g with
  | nil =>
      intro xV dxV seedV
      -- casts are inverse on inner products
      have hleft := inner_castCtxVec (Γ₁ := Γ) (Γ₂ := Γ ++ []) (h := (List.append_nil Γ).symm) dxV
        seedV
      -- `backpropVec nil` is the inverse cast
      simpa [Graph.jvpVec, Graph.backpropVec] using hleft
  | snoc g node ih =>
      intro xV dxV seedV
      rename_i ss τ
      let ctxV : CtxVec (Γ ++ ss) := evalVec (Γ := Γ) (ss := ss) g xV
      let dctxV : CtxVec (Γ ++ ss) := jvpVec (Γ := Γ) (ss := ss) g xV dxV
      let dyV : Vec (Shape.size τ) := node.jvpVec (Γ := Γ ++ ss) (τ := τ) ctxV dctxV
      let assoc := List.append_assoc Γ ss [τ]
      let seedV' : CtxVec ((Γ ++ ss) ++ [τ]) :=
        castCtxVec (Γ₁ := Γ ++ (ss ++ [τ])) (Γ₂ := (Γ ++ ss) ++ [τ]) assoc.symm seedV
      let seedPrevV : CtxVec (Γ ++ ss) := (unsnocCtx (Γ := (Γ ++ ss)) (τ := τ) seedV').1
      let seedOutV : Vec (Shape.size τ) := (unsnocCtx (Γ := (Γ ++ ss)) (τ := τ) seedV').2
      have hseed : snocCtx (Γ := (Γ ++ ss)) (τ := τ) seedPrevV seedOutV = seedV' := by
        simpa [seedPrevV, seedOutV] using
          (snocCtx_unsnocCtx (Γ := (Γ ++ ss)) (τ := τ) seedV')

      -- Move the outer `assoc` cast from the JVP output onto the seed.
      have hjvp_cast :
          inner ℝ (jvpVec (Γ := Γ) (ss := ss ++ [τ]) (Graph.snoc g node) xV dxV) seedV
            =
          inner ℝ (snocCtx (Γ := (Γ ++ ss)) (τ := τ) dctxV dyV) seedV' := by
        have hcast := inner_castCtxVec (Γ₁ := (Γ ++ ss) ++ [τ]) (Γ₂ := Γ ++ (ss ++ [τ])) (h :=
          assoc)
          (snocCtx (Γ := (Γ ++ ss)) (τ := τ) dctxV dyV) seedV
        simpa [Graph.jvpVec, ctxV, dctxV, dyV, seedV', assoc] using hcast

      -- Split the inner product across the `snocCtx` append.
      have hinter :
          inner ℝ (snocCtx (Γ := (Γ ++ ss)) (τ := τ) dctxV dyV) seedV'
            =
          inner ℝ dctxV seedPrevV + inner ℝ dyV seedOutV := by
        -- Cancel the `snocCtx` casts on both sides and use `inner_append`.
        have hsnoc :
            inner ℝ (snocCtx (Γ := (Γ ++ ss)) (τ := τ) dctxV dyV)
                  (snocCtx (Γ := (Γ ++ ss)) (τ := τ) seedPrevV seedOutV)
              =
            inner ℝ dctxV seedPrevV + inner ℝ dyV seedOutV := by
          -- transport to the `(ctxSize + size)` representation
          have hcast' :
              inner ℝ (snocCtx (Γ := (Γ ++ ss)) (τ := τ) dctxV dyV)
                    (snocCtx (Γ := (Γ ++ ss)) (τ := τ) seedPrevV seedOutV)
                =
              inner ℝ (appendVec (m := ctxSize (Γ ++ ss)) (n := Shape.size τ) dctxV dyV)
                    (appendVec (m := ctxSize (Γ ++ ss)) (n := Shape.size τ) seedPrevV seedOutV) :=
                      by
            simpa [snocCtx] using
              (inner_castVec_castVec (h := (ctxSize_snoc (Γ ++ ss) τ).symm)
                (x := appendVec (m := ctxSize (Γ ++ ss)) (n := Shape.size τ) dctxV dyV)
                (y := appendVec (m := ctxSize (Γ ++ ss)) (n := Shape.size τ) seedPrevV seedOutV))
          -- apply `inner_append` and simplify
          simp [hcast', inner_append]
        simpa [hseed] using hsnoc

      -- Use node-level adjointness to rewrite the output term.
      have hlocal :
          inner ℝ dyV seedOutV = inner ℝ dctxV (node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV seedOutV)
            := by
        simpa [dyV] using (Node.correct_inner (node := node) ctxV dctxV seedOutV)

      -- Combine and apply IH on the previous graph.
      have hadd :
          inner ℝ dctxV seedPrevV + inner ℝ dctxV (node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV
            seedOutV)
            =
          inner ℝ dctxV (seedPrevV + node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV seedOutV) := by
        simpa using
          (inner_add_right (x := dctxV) (y := seedPrevV)
            (z := node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV seedOutV)).symm

      calc
        inner ℝ (jvpVec (Γ := Γ) (ss := ss ++ [τ]) (Graph.snoc g node) xV dxV) seedV
            = inner ℝ (snocCtx (Γ := (Γ ++ ss)) (τ := τ) dctxV dyV) seedV' := hjvp_cast
        _ = inner ℝ dctxV seedPrevV + inner ℝ dyV seedOutV := hinter
        _ = inner ℝ dctxV seedPrevV + inner ℝ dctxV (node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV
          seedOutV) := by
              simp [hlocal]
        _ = inner ℝ dctxV (seedPrevV + node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV seedOutV) := hadd
        _ = inner ℝ dxV (backpropVec (Γ := Γ) (ss := ss) g xV (seedPrevV + node.vjpVec (Γ := Γ ++
          ss) (τ := τ) ctxV seedOutV)) := by
              simpa [dctxV] using (ih xV dxV (seedPrevV + node.vjpVec (Γ := Γ ++ ss) (τ := τ) ctxV
                seedOutV))
        _ = inner ℝ dxV (backpropVec (Γ := Γ) (ss := ss ++ [τ]) (Graph.snoc g node) xV seedV) := by
              simp [Graph.backpropVec, ctxV, seedV', seedPrevV, seedOutV]

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
  deriv : CtxVec Γ → (CtxVec Γ →L[ℝ] Vec (Shape.size τ))
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
  /-- deriv. -/
  deriv : CtxVec Γ →L[ℝ] Vec (Shape.size τ)
  /-- has FDeriv At. -/
  hasFDerivAt : HasFDerivAt (node.forwardVec (Γ := Γ) (τ := τ)) deriv xV
  /-- jvp eq. -/
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
    (CtxVec Γ × Vec (Shape.size τ)) →L[ℝ] CtxVec (Γ ++ [τ]) :=
  (castCLM (h := (ctxSize_snoc Γ τ).symm)).comp (appendCLM (m := ctxSize Γ) (n := Shape.size τ))

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
        simpa [Graph.evalVec, castCtxVec, castVec, castCLM, D, h] using (D.hasFDerivAt (x := xV))
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
      let Dn : CtxVec (Γ ++ ss) →L[ℝ] Vec (Shape.size τ) := hg_node.deriv ctxV
      have hnode : HasFDerivAt (node.forwardVec (Γ := Γ ++ ss) (τ := τ)) Dn ctxV :=
        hg_node.hasFDerivAt ctxV
      -- derivative for the output component `yV`.
      have hy :
          HasFDerivAt
            (fun xV : CtxVec Γ => node.forwardVec (Γ := Γ ++ ss) (τ := τ) (evalVec (Γ := Γ) (ss :=
              ss) g xV))
            (Dn.comp Dg) xV := by
        simpa [ctxV] using (hnode.comp xV hDg)
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
              (fun p : CtxVec (Γ ++ ss) × Vec (Shape.size τ) =>
                snocCtx (Γ := Γ ++ ss) (τ := τ) p.1 p.2)
              (snocCLM (Γ := Γ ++ ss) (τ := τ)) (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV)
                := by
          simpa [snocCtx, snocCLM, castVec, castCLM, appendCLM] using
            ((snocCLM (Γ := Γ ++ ss) (τ := τ)).hasFDerivAt (x := (ctxV, node.forwardVec (Γ := Γ ++
              ss) (τ := τ) ctxV)))
        have hcomp1 :=
          (Dcast.hasFDerivAt (x := snocCtx (Γ := Γ ++ ss) (τ := τ) ctxV (node.forwardVec (Γ := Γ ++
            ss) (τ := τ) ctxV))).comp xV
            (hsnoc.comp xV hpair)
        -- `evalVec` is definitionally this composition.
        simpa [Graph.evalVec, ctxV, castCtxVec, castVec, assoc, D, Dcast, snocCLM] using hcomp1
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
      · simpa [Graph.evalVec, castCtxVec, castVec, castCLM, D, h] using (D.hasFDerivAt (x := xV))
      · intro dxV
        simp [Graph.jvpVec, castCtxVec, castVec, castCLM, D]
  | snoc g node ih =>
      intro xV hg
      rename_i ss τ
      rcases hg with ⟨hg_g, hg_node⟩
      rcases ih (xV := xV) hg_g with ⟨Dg, hDg, hJg⟩
      let ctxV : CtxVec (Γ ++ ss) := evalVec (Γ := Γ) (ss := ss) g xV
      let Dn : CtxVec (Γ ++ ss) →L[ℝ] Vec (Shape.size τ) := hg_node.deriv
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
              (fun p : CtxVec (Γ ++ ss) × Vec (Shape.size τ) =>
                snocCtx (Γ := Γ ++ ss) (τ := τ) p.1 p.2)
              (snocCLM (Γ := Γ ++ ss) (τ := τ))
              (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV) := by
          simpa [snocCtx, snocCLM, castVec, castCLM, appendCLM] using
            ((snocCLM (Γ := Γ ++ ss) (τ := τ)).hasFDerivAt
              (x := (ctxV, node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV)))
        have hcomp1 :=
          (Dcast.hasFDerivAt (x := snocCtx (Γ := Γ ++ ss) (τ := τ) ctxV
            (node.forwardVec (Γ := Γ ++ ss) (τ := τ) ctxV))).comp xV
              (hsnoc.comp xV hpair)
        simpa [Graph.evalVec, ctxV, castCtxVec, castVec, assoc, D, Dcast, snocCLM] using hcomp1
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
