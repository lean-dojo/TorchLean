/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.OfFn

/-!
# Vectorization

The analytic autograd proofs share the `Vec` alias (`EuclideanSpace ℝ (Fin n)`) defined here, the
basic `Tensor ℝ [n]` ↔ `Vec n` conversions and the split of a flattened `m * n` vector into `m`
rows.

## PyTorch correspondence / citations
These conversions play the same role as treating a length-`n` tensor as an element of
$\mathbb R^n$ when using standard analysis results (mean value theorem, operator norms, etc.).
https://pytorch.org/docs/stable/linalg.html
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

open scoped BigOperators

noncomputable section

/-- Euclidean vectors over `ℝ`. -/
abbrev Vec (n : Nat) := EuclideanSpace ℝ (Fin n)

/--
Convert a rank-one tensor (`Tensor ℝ [n]`) into a Euclidean vector `Vec n`, the analysis-friendly
view of a length-`n` tensor as an element of $\mathbb R^n$.
-/
def Tensor.getScalarE {n : Nat} (t : Tensor ℝ [n]) : Vec n :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm (Tensor.getScalar t)

/-- Coordinate evaluation of the Euclidean view of a rank-one tensor. -/
@[simp] theorem Tensor.getScalarE_ofLp {n : Nat} (t : Tensor ℝ [n]) (i : Fin n) :
    (getScalarE t).ofLp i = Tensor.getScalar t i := by
  simp [getScalarE, EuclideanSpace.equiv]

/--
Convert a Euclidean vector `Vec n` back into a rank-one tensor (`Tensor ℝ [n]`), inverting
`getScalarE`.
-/
def Tensor.ofFnE {n : Nat} (v : Vec n) : Tensor ℝ [n] :=
  Tensor.ofFn ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)) v)

/-- `getScalarE` is a left inverse of `ofFnE`. -/
@[simp] theorem Tensor.getScalarE_ofFnE {n : Nat} (v : Vec n) : getScalarE (ofFnE v) = v := by
  ext i
  simp [getScalarE, ofFnE, EuclideanSpace.equiv]

/-- Coordinate evaluation commutes with conversion from a Euclidean vector. -/
@[simp] theorem Tensor.getScalar_ofFnE {n : Nat} (v : Vec n) (i : Fin n) :
    Tensor.getScalar (ofFnE v) i = v.ofLp i := by
  rw [← getScalarE_ofLp, getScalarE_ofFnE]

/-- `ofFnE` is a left inverse of `getScalarE`. -/
@[simp] theorem Tensor.ofFnE_getScalarE {n : Nat} (t : Tensor ℝ [n]) :
    ofFnE (getScalarE t) = t := by
  classical
  let e := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)
  change Tensor.ofFn (e (e.symm (fun i => t.getScalar i))) = t
  rw [e.apply_symm_apply]
  exact Tensor.ofFn_getScalar t

/--
The Euclidean inner product on `Vec n` is $\langle x,y\rangle=\sum_i x_i y_i$.
-/
theorem Vec.inner_eq_sum_mul {n : Nat} (x y : Vec n) :
    inner ℝ x y = ∑ i : Fin n, x i * y i := by
  classical
  simpa [Vec, dotProduct, mul_comm] using
    (EuclideanSpace.inner_eq_star_dotProduct (x := x) (y := y))

/-- Split a flattened `m*n` vector into `m` rows of length `n`. -/
def rows {m n : Nat} (x : Vec (m * n)) : Fin m → Vec n :=
  fun i => WithLp.toLp 2 fun j => x (finProdFinEquiv (i, j))

/-- Inverse of `rows`: assemble `m` rows back into a flattened `m*n` vector. -/
def unrows {m n : Nat} (r : Fin m → Vec n) : Vec (m * n) :=
  WithLp.toLp 2 fun ip =>
    let p : Fin m × Fin n := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)).symm ip
    r p.1 p.2

/-- Row recovered from a flat index built by `finProdFinEquiv`. -/
@[simp]
theorem divNat_finProdFinEquiv {m n : Nat} (p : Fin m × Fin n) :
    (finProdFinEquiv (m := m) (n := n) p).divNat = p.1 := by
  have h := congrArg Prod.fst
    ((finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)).left_inv p)
  -- `finProdFinEquiv.symm` is `(divNat, modNat)`.
  simpa [finProdFinEquiv] using h

/-- Column recovered from a flat index built by `finProdFinEquiv`. -/
@[simp]
theorem modNat_finProdFinEquiv {m n : Nat} (p : Fin m × Fin n) :
    (finProdFinEquiv (m := m) (n := n) p).modNat = p.2 := by
  have h := congrArg Prod.snd
    ((finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)).left_inv p)
  simpa [finProdFinEquiv] using h

/-- Continuous-linear-map version of `rows`. -/
def rowsCLM {m n : Nat} : Vec (m * n) →L[ℝ] (Fin m → Vec n) := by
  classical
  let fLin : Vec (m * n) →ₗ[ℝ] (Fin m → Vec n) :=
    { toFun := rows (m := m) (n := n)
      map_add' := by
        intro x y
        funext i
        ext j
        simp [rows, Pi.add_apply]
      map_smul' := by
        intro a x
        funext i
        ext j
        simp [rows, Pi.smul_apply, smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Continuous-linear-map version of `unrows`. -/
def unrowsCLM {m n : Nat} : (Fin m → Vec n) →L[ℝ] Vec (m * n) := by
  classical
  let fLin : (Fin m → Vec n) →ₗ[ℝ] Vec (m * n) :=
    { toFun := unrows (m := m) (n := n)
      map_add' := by
        intro r₁ r₂
        ext ip
        simp [unrows, Pi.add_apply]
      map_smul' := by
        intro a r
        ext ip
        simp [unrows, Pi.smul_apply, smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The bundled row splitting computes `rows`. -/
@[simp] theorem rowsCLM_apply {m n : Nat} (x : Vec (m * n)) :
    rowsCLM (m := m) (n := n) x = rows (m := m) (n := n) x := rfl

/-- The bundled row reassembly computes `unrows`. -/
@[simp] theorem unrowsCLM_apply {m n : Nat} (r : Fin m → Vec n) :
    unrowsCLM (m := m) (n := n) r = unrows (m := m) (n := n) r := rfl

/-- Splitting a flattened matrix into rows and reassembling is the identity. -/
@[simp] theorem unrows_rows {m n : Nat} (x : Vec (m * n)) :
    unrows (m := m) (n := n) (rows (m := m) (n := n) x) = x := by
  ext ip
  obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective ip
  simp [unrows, rows, divNat_finProdFinEquiv, modNat_finProdFinEquiv]

/-- The inner product of two row stacks is the sum of the rowwise inner products. -/
theorem inner_unrows_unrows {m n : Nat} (r s : Fin m → Vec n) :
    inner ℝ (unrows (m := m) (n := n) r) (unrows (m := m) (n := n) s) =
      ∑ i : Fin m, inner ℝ (r i) (s i) := by
  classical
  -- Reindex the flat sum by `finProdFinEquiv`, then split it into a double sum over `(i, j)`.
  have hsum :
      (∑ ip : Fin (m * n), unrows (m := m) (n := n) r ip * unrows (m := m) (n := n) s ip) =
        ∑ p : Fin m × Fin n,
          unrows (m := m) (n := n) r (finProdFinEquiv p) *
            unrows (m := m) (n := n) s (finProdFinEquiv p) :=
    (Equiv.sum_comp (e := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)))
      (g := fun ip => unrows (m := m) (n := n) r ip * unrows (m := m) (n := n) s ip)).symm
  rw [Vec.inner_eq_sum_mul, hsum, Fintype.sum_prod_type]
  simp [unrows, Vec.inner_eq_sum_mul, divNat_finProdFinEquiv, modNat_finProdFinEquiv]

end
end TorchLean
