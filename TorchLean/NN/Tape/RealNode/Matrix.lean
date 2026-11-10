/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealNode.Elementwise
public import TorchLean.NN.Tape.RealNode.Shape

/-!
# Matrix tape nodes

Tape nodes for matrix multiplication, transpose, row and column broadcasts and row means, each with
a global `RealNode.FDerivCorrect` certificate. Matrices live on the tape flattened in row-major
order, with entry `(i, j)` of an `m × n` matrix at index `j + n * i` (`val_idxMN`). The proofs run
on these flat vectors, and `toVec_idxMN` connects them to the entries `Tensor.get2 A i j`.
Transpose, the broadcasts and the row mean are linear, and `matmul` is bilinear, so its derivative
is the product rule.
-/

@[expose] public section

open TorchLean.Shape (size size_cons)

-- In this file simp leaves out `size_cons`. It would rewrite the sizes inside `Fin` types before
-- the `ofFn`, `WithLp.toLp` and cast lemmas match. The proofs that unfold a size name it.
attribute [-simp] TorchLean.Shape.size_cons

namespace TorchLean

open Tensor
open Rep (dot)

noncomputable section

open scoped BigOperators

section

open CtxVec

-- ---------------------------------------------------------------------------
-- Linear algebra: matrix-matrix multiplication
-- ---------------------------------------------------------------------------

namespace Matmul

open scoped BigOperators

-- `simp` reads the column count `Shape.size [n]` of a flattened `m × n` matrix as `n`.
attribute [simp] Shape.size_vector

/-- Convert `(i,j)` coordinates into a flattened index for an `m×n` matrix vectorization. -/
def idxMN {m n : Nat} (i : Fin m) (j : Fin n) : Fin (Shape.size [m, n]) :=
  let hn : Shape.size [n] = n := Shape.size_vector n
  -- `Shape.size [m, n]` unfolds to `m * Shape.size [n]`, where `finProdFinEquiv` lands.
  finProdFinEquiv (i, Fin.cast hn.symm j)

/-- The flattened index of entry `(i, j)` is `j + n * i`: row-major layout, as in PyTorch. -/
theorem val_idxMN {m n : Nat} (i : Fin m) (j : Fin n) :
    (idxMN (m := m) (n := n) i j).val = j.val + n * i.val := by
  change j.val + Shape.size (Shape.dim n Shape.scalar) * i.val = _
  simp

/-- Every flattened matrix index is `idxMN` of its row and column. -/
theorem idxMN_divNat_modNat {m n : Nat} (ip : Fin (Shape.size [m, n])) :
    idxMN (m := m) (n := n) (ip.divNat (m := m) (n := Shape.size [n]))
      (Fin.cast (Shape.size_vector n)
        (ip.modNat (m := m) (n := Shape.size [n]))) = ip := by
  apply Fin.ext
  change (ip.modNat (m := m) (n := Shape.size [n])).val +
      Shape.size [n] * (ip.divNat (m := m) (n := Shape.size [n])).val = ip.val
  exact Nat.mod_add_div _ _

/-- Coordinate `idxMN i j` of a flattened matrix is its entry `Tensor.get2 A i j`.

Matrix and attention nodes are proved on flat vectors, where Mathlib's calculus lives, and stated
about `Tensor ℝ [m, n]`. With this lemma a proof moves between the two without unfolding the
flattening. -/
theorem toVec_idxMN {m n : Nat} (A : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    toVec (t := A) (idxMN (m := m) (n := n) i j) = Tensor.get2 A i j := by
  have h1 : 0 < Shape.size Shape.scalar := by simp
  have hrow := toVec_dim_apply (Tensor.unstack A)
    (i, finProdFinEquiv (j, (⟨0, h1⟩ : Fin (Shape.size Shape.scalar))))
  have hcol := toVec_dim_apply (Tensor.unstack (Tensor.unstack A i))
    (j, (⟨0, h1⟩ : Fin (Shape.size Shape.scalar)))
  rw [Tensor.dim_unstack] at hrow hcol
  have hidx :
      (finProdFinEquiv (i, finProdFinEquiv (j, (⟨0, h1⟩ : Fin (Shape.size Shape.scalar)))) :
        Fin (Shape.size [m, n])) = idxMN (m := m) (n := n) i j := by
    apply Fin.ext
    rw [val_idxMN]
    change (0 + Shape.size Shape.scalar * j.val) +
      Shape.size (Shape.dim n Shape.scalar) * i.val = j.val + n * i.val
    simp
  rw [← hidx]
  refine hrow.trans (hcol.trans ?_)
  rw [← Tensor.scalar_item (Tensor.unstack (Tensor.unstack A i) j), toVec_scalar]
  rfl

/-- Entry `(i, j)` of `Tensor.ofVec v` is the flat coordinate `v (idxMN i j)`. -/
private theorem get2_ofVec {m n : Nat} (v : Vec (Shape.size [m, n])) (i : Fin m)
    (j : Fin n) :
    Tensor.get2 (ofVec (s := .dim m (.dim n .scalar)) v) i j =
      v (idxMN (m := m) (n := n) i j) := by
  have htv :
      toVec (t := ofVec (s := .dim m (.dim n .scalar)) v)
          (idxMN (m := m) (n := n) i j)
        = v (idxMN (m := m) (n := n) i j) := by
    simp
  exact (toVec_idxMN (A := ofVec (s := .dim m (.dim n .scalar)) v) i j).symm.trans htv

/-- Matrix multiplication on flattened matrices, `(m×n) × (n×p) → (m×p)`, as a sum over the
contracted index. -/
def vec {m n p : Nat} (a : Vec (Shape.size [m, n])) (b : Vec (Shape.size [n, p])) :
    Vec (Shape.size [m, p]) :=
  WithLp.toLp 2 fun ip =>
    let hp : Shape.size [p] = p := Shape.size_vector p
    let i : Fin m := ip.divNat (m := m) (n := Shape.size [p])
    let k' : Fin (Shape.size [p]) := ip.modNat (m := m) (n := Shape.size [p])
    let k : Fin p := Fin.cast hp k'
    ∑ j : Fin n, a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)

/-- Entry `(i, k)` of a matrix product is the usual sum over the contracted index.

The `let`s in the statement do the flat-index arithmetic, splitting one flat index `ip` into a
row by `divNat` and a column by `modNat`. With the split inside the statement, `simp` can use the
lemma on goals written in flat coordinates. -/
@[simp] theorem vec_apply {m n p : Nat} (a : Vec (Shape.size [m, n]))
    (b : Vec (Shape.size [n, p])) (ip : Fin (Shape.size [m, p])) :
    vec (m := m) (n := n) (p := p) a b ip =
      let hp : Shape.size [p] = p := Shape.size_vector p
      let i : Fin m := ip.divNat (m := m) (n := Shape.size [p])
      let k' : Fin (Shape.size [p]) := ip.modNat (m := m) (n := Shape.size [p])
      let k : Fin p := Fin.cast hp k'
      ∑ j : Fin n, a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k) := by
  simp [vec]

/-!
Matrix multiplication is defined on flattened matrices, the form that `CtxVec` and `HasFDerivAt`
work with.

PyTorch analogue: `torch.matmul` / `@` operator on 2D tensors.
https://pytorch.org/docs/stable/generated/torch.matmul.html
-/

/-- For fixed left operand `a`, `mulLeftCLM a` is the linear map `b ↦ a*b`. -/
def mulLeftCLM {m n p : Nat} (a : Vec (Shape.size [m, n])) :
    Vec (Shape.size [n, p]) →L[ℝ] Vec (Shape.size [m, p]) := by
  classical
  let fLin : Vec (Shape.size [n, p]) →ₗ[ℝ] Vec (Shape.size [m, p]) :=
    { toFun := fun b => vec (m := m) (n := n) (p := p) a b
      map_add' := by
        intro b1 b2
        ext ip
        simp [vec, Finset.sum_add_distrib, mul_add]
      map_smul' := by
        intro r b
        ext ip
        simp [vec, Finset.mul_sum, mul_left_comm] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- `mulLeftCLM a` computes `vec a`. -/
@[simp] theorem mulLeftCLM_apply {m n p : Nat} (a : Vec (Shape.size [m, n]))
    (b : Vec (Shape.size [n, p])) :
    mulLeftCLM (m := m) (n := n) (p := p) a b = vec (m := m) (n := n) (p := p) a b :=
  rfl

/-- Continuous bilinear map for matrix multiplication on flattened vectors.

It sends `a` to `mulLeftCLM a` and is linear in `a`. Continuity comes free in finite
dimension, so the proof needs no operator-norm bound. -/
def bilin {m n p : Nat} :
    Vec (Shape.size [m, n]) →L[ℝ] Vec (Shape.size [n, p]) →L[ℝ]
      Vec (Shape.size [m, p]) := by
  classical
  let fLin : Vec (Shape.size [m, n]) →ₗ[ℝ] Vec (Shape.size [n, p]) →L[ℝ]
      Vec (Shape.size [m, p]) :=
    { toFun := fun a => mulLeftCLM (m := m) (n := n) (p := p) a
      map_add' := by
        intro a1 a2
        ext b ip
        simp [vec, Finset.sum_add_distrib, add_mul]
      map_smul' := by
        intro r a
        ext b ip
        simp [vec, smul_eq_mul, Finset.mul_sum, mul_assoc] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- The bilinear packaging of matrix multiplication computes `vec`.

`bilin` exists so the differentiability proofs can use Mathlib's bilinear-map API, and this
simp lemma lets every other proof forget the packaging. -/
@[simp] theorem bilin_apply {m n p : Nat} (a : Vec (Shape.size [m, n]))
    (b : Vec (Shape.size [n, p])) :
    bilin (m := m) (n := n) (p := p) a b = vec (m := m) (n := n) (p := p) a b := rfl

/-- Flattening `Tensor.matmul` of `Tensor.ofVec aV` and `Tensor.ofVec bV` gives `vec aV bV`. -/
theorem toVec_matmul {m n p : Nat} (aV : Vec (Shape.size [m, n]))
    (bV : Vec (Shape.size [n, p])) :
    toVec (t := Tensor.matmul (ofVec (s := .dim m (.dim n .scalar)) aV)
        (ofVec (s := .dim n (.dim p .scalar)) bV))
      =
    vec (m := m) (n := n) (p := p) aV bV := by
  classical
  ext ip
  -- represent `ip` as a row/column pair using `Fin.divNat/modNat` for `m * Shape.size [p]`
  let i : Fin m := ip.divNat (m := m) (n := Shape.size [p])
  let k' : Fin (Shape.size [p]) := ip.modNat (m := m) (n := Shape.size [p])
  let hp : Shape.size [p] = p := Shape.size_vector p
  let k : Fin p := Fin.cast hp k'
  -- interpret LHS coordinate via `get2` and the matrix entry lemma
  have hL :
      toVec
          (t := Tensor.matmul (ofVec (s := .dim m (.dim n .scalar)) aV)
            (ofVec (s := .dim n (.dim p .scalar)) bV)) ip
        =
      Tensor.get2
          (Tensor.matmul (ofVec (s := .dim m (.dim n .scalar)) aV)
            (ofVec (s := .dim n (.dim p .scalar)) bV)) i k := by
    have hip : idxMN (m := m) (n := p) i k = ip := idxMN_divNat_modNat ip
    rw [←hip]
    exact toVec_idxMN
      (A := Tensor.matmul (ofVec (s := .dim m (.dim n .scalar)) aV)
        (ofVec (s := .dim n (.dim p .scalar)) bV))
      i k
  have hEntry :=
    get2_matmul_eq_sum
      (A := ofVec (s := .dim m (.dim n .scalar)) aV)
      (B := ofVec (s := .dim n (.dim p .scalar)) bV)
      (i := i) (j := k)
  have hA : ∀ j : Fin n,
      Tensor.get2 (ofVec (s := .dim m (.dim n .scalar)) aV) i j =
        aV (idxMN (m := m) (n := n) i j) :=
    fun j => get2_ofVec (v := aV) i j
  have hB : ∀ j : Fin n,
      Tensor.get2 (ofVec (s := .dim n (.dim p .scalar)) bV) j k =
        bV (idxMN (m := n) (n := p) j k) :=
    fun j => get2_ofVec (v := bV) j k
  calc
    toVec
        (t := Tensor.matmul (ofVec (s := .dim m (.dim n .scalar)) aV)
          (ofVec (s := .dim n (.dim p .scalar)) bV)) ip
        =
      Tensor.get2
          (Tensor.matmul (ofVec (s := .dim m (.dim n .scalar)) aV)
            (ofVec (s := .dim n (.dim p .scalar)) bV)) i k := hL
    _ = ∑ j : Fin n,
          Tensor.get2 (ofVec (s := .dim m (.dim n .scalar)) aV) i j *
            Tensor.get2 (ofVec (s := .dim n (.dim p .scalar)) bV) j k := by
          simpa using hEntry
    _ = ∑ j : Fin n, aV (idxMN (m := m) (n := n) i j) * bV (idxMN (m := n) (n := p) j k) := by
          simp [hA, hB]
    _ = vec (m := m) (n := n) (p := p) aV bV ip := by
          rw [vec_apply]

end Matmul

-- ---------------------------------------------------------------------------
-- Linear algebra: matrix transpose
-- ---------------------------------------------------------------------------

namespace Tape.RealNode.MatTranspose

/-- A matrix shape `[m, n]` has `m * n` entries. -/
theorem matSize_eq_mul (m n : Nat) : Shape.size [m, n] = m * n := by
  simp [Shape.size_cons]

/-- Equivalence implementing matrix transpose on flattened indices. -/
def equiv (m n : Nat) : Fin (m * n) ≃ Fin (n * m) :=
  (finProdFinEquiv.symm.trans (Equiv.prodComm (Fin m) (Fin n))).trans finProdFinEquiv

/-- The transpose index equivalence is symmetric up to swapping `m` and `n`. -/
private theorem equiv_symm (m n : Nat) :
    (equiv m n).symm = equiv n m := by
  ext k; simp [equiv, Equiv.prodComm_symm]

/-- Transpose on flattened matrices: `(m×n)` flattened row-major → `(n×m)` flattened row-major. -/
def vec {m n : Nat} (a : Vec (Shape.size [m, n])) : Vec (Shape.size [n, m]) :=
  LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (matSize_eq_mul n m).symm)
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (equiv m n)
      (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr (matSize_eq_mul m n)) a))

/-- Adjointness of `vec` with respect to the standard inner product on vectors. -/
private theorem inner_vec {m n : Nat} (x : Vec (Shape.size [m, n]))
    (y : Vec (Shape.size [n, m])) :
    inner ℝ (vec (m := m) (n := n) x) y =
      inner ℝ x (vec (m := n) (n := m) y) := by
  -- Move the size casts across the inner product. What remains is `inner_map_eq_flip` for the
  -- permutation by `equiv m n`, whose inverse is `equiv n m`.
  unfold vec
  rw [LinearIsometryEquiv.inner_map_eq_flip, LinearIsometryEquiv.piLpCongrLeft_symm, finCongr_symm,
    LinearIsometryEquiv.inner_map_eq_flip, LinearIsometryEquiv.piLpCongrLeft_symm,
    equiv_symm, LinearIsometryEquiv.inner_map_eq_flip,
    LinearIsometryEquiv.piLpCongrLeft_symm, finCongr_symm]

end Tape.RealNode.MatTranspose

/-!
Transpose is implemented as a coordinate permutation on flattened matrices.

PyTorch analogue: `A.transpose(0, 1)` for a 2D tensor.
https://pytorch.org/docs/stable/generated/torch.transpose.html
-/

/-- Tape node computing matrix transpose: `(m×n) ↦ (n×m)`. -/
def Tape.RealNode.matrixTranspose {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) :
    LawfulNode (α := ℝ) Unit Γ (.dim n (.dim m .scalar)) :=
  LawfulNode.ofFn (Γ := Γ) (τ := .dim n (.dim m .scalar))
    (f := fun xV => MatTranspose.vec (m := m) (n := n) (CtxVec.get (Γ := Γ) (s := .dim m
      (.dim n .scalar)) A xV))
    (jvp := fun _xV dxV => MatTranspose.vec (m := m) (n := n) (CtxVec.get (Γ := Γ) (s :=
      .dim m (.dim n .scalar)) A dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A
        (MatTranspose.vec (m := n) (n := m) δV))
    (correct_inner := by
      intro _xV dxV δV
      have hT :=
        MatTranspose.inner_vec (m := m) (n := n)
          (x := CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV) (y := δV)
      have hCtx :=
        CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV
          (MatTranspose.vec (m := n) (n := m) δV)
      calc
        inner ℝ
            (MatTranspose.vec (m := m) (n := n)
              (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)) δV
            =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
              (MatTranspose.vec (m := n) (n := m) δV) := hT
        _ =
            inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A
                (MatTranspose.vec (m := n) (n := m) δV)) := by
            simpa using hCtx.symm)

/-- `FDerivCorrect` for `matrixTranspose`, whose derivative at every point is the transpose
itself. -/
def Tape.RealNode.matrixTransposeFDeriv {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) :
    FDerivCorrect (matrixTranspose (Γ := Γ) (m := m) (n := n) A).toNodeData () := by
  classical
  -- `MatTranspose.vec` is a size cast, a coordinate permutation, and a size cast. Each is already a
  -- continuous linear map.
  let Tlin : Vec (Shape.size [m, n]) →L[ℝ] Vec (Shape.size [n, m]) :=
    (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr
        (MatTranspose.matSize_eq_mul n m).symm)).toContinuousLinearEquiv.toContinuousLinearMap.comp
      ((LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
          (MatTranspose.equiv m n)).toContinuousLinearEquiv.toContinuousLinearMap.comp
        (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr
          (MatTranspose.matSize_eq_mul m n))).toContinuousLinearEquiv.toContinuousLinearMap)
  refine
    { deriv := fun _xV => Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hCLM :=
      (Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)).hasFDerivAt (x := xV)
    have hfun :
        (NodeData.forwardVec (Γ := Γ) (τ := .dim n (.dim m .scalar))
            (matrixTranspose (Γ := Γ) (m := m) (n := n) A).toNodeData ()) =
          (fun x : CtxVec Γ =>
            (Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)) x) := by
      funext x
      simp [matrixTranspose, LawfulNode.forwardVec_ofFn, ContinuousLinearMap.comp_apply,
        CtxVec.getCLM_apply, Tlin, MatTranspose.vec]
    exact hCLM.congr_of_eventuallyEq hfun.eventuallyEq
  · intro _xV dxV
    simp [matrixTranspose, LawfulNode.jvpVec_ofFn, ContinuousLinearMap.comp_apply,
      CtxVec.getCLM_apply, Tlin, MatTranspose.vec]

/-- Matrix multiplication node `(A, B) ↦ A B` on 2D tensors. Its VJP sends the cotangent `δ` to
`δ Bᵀ` for `A` and to `Aᵀ δ` for `B`. -/
def Tape.RealNode.matmul {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    LawfulNode (α := ℝ) Unit Γ (.dim m (.dim p .scalar)) :=
  LawfulNode.ofFn (Γ := Γ) (τ := .dim m (.dim p .scalar))
    (f := fun xV =>
      let aT := ofVec (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := ofVec (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      toVec (t := Tensor.matmul aT bT))
    (jvp := fun xV dxV =>
      let aT := ofVec (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := ofVec (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let daT := ofVec (s := .dim m (.dim n .scalar))
        (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
      let dbT := ofVec (s := .dim n (.dim p .scalar))
        (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
      toVec (t := Tensor.matmul daT bT + Tensor.matmul aT dbT))
    (vjp := fun xV δV =>
      let aT := ofVec (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := ofVec (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let δT := ofVec (s := .dim m (.dim p .scalar)) δV
      let dA := Tensor.matmul δT (Tensor.permute bT (Shape.adjacentSwap [n, p] 0) [p, n])
      let dB := Tensor.matmul (Tensor.permute aT (Shape.adjacentSwap [m, n] 0) [n, m]) δT
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVec (t := dA)) +
        CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVec (t := dB)))
    (correct_inner := by
      intro xV dxV δV
      classical
      let aT := ofVec (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := ofVec (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let daT := ofVec (s := .dim m (.dim n .scalar))
        (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
      let dbT := ofVec (s := .dim n (.dim p .scalar))
        (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
      let δT := ofVec (s := .dim m (.dim p .scalar)) δV
      let dC := Tensor.matmul daT bT + Tensor.matmul aT dbT
      let dA := Tensor.matmul δT (Tensor.permute bT (Shape.adjacentSwap [n, p] 0) [p, n])
      let dB := Tensor.matmul (Tensor.permute aT (Shape.adjacentSwap [m, n] 0) [n, m]) δT

      -- LHS: rewrite `inner` into tensor `dot` using vectorization.
      have hL : inner ℝ (toVec (t := dC)) δV = Rep.dot dC δT := by
        simp [dot_eq_inner_toVec, δT, toVec_ofVec]

      -- RHS: split the context inner into the two single-slot contributions.
      have hR :
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVec (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVec (t := dB)))
            =
          Rep.dot daT dA + Rep.dot dbT dB := by
        have hA' :
            inner ℝ dxV
                (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVec (t := dA)))
              =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
              (toVec (t := dA)) := by
          simpa using
            (CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV
              (toVec (t := dA)))
        have hB' :
            inner ℝ dxV
                (CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVec (t := dB)))
              =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
              (toVec (t := dB)) := by
          simpa using
            (CtxVec.inner_get_single (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV
              (toVec (t := dB)))
        have hdotA :
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
                (toVec (t := dA)) =
              Rep.dot daT dA := by
          simp [dot_eq_inner_toVec, daT, toVec_ofVec]
        have hdotB :
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
                (toVec (t := dB)) =
              Rep.dot dbT dB := by
          simp [dot_eq_inner_toVec, dbT, toVec_ofVec]
        calc
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVec (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVec (t := dB)))
              =
              inner ℝ dxV
                  (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A
                    (toVec (t := dA))) +
                inner ℝ dxV
                  (CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B
                    (toVec (t := dB))) := by
                simp [inner_add_right]
          _ =
              inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
                  (toVec (t := dA))
                +
                inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV)
                  (toVec (t := dB)) := by
                simp [hA', hB']
          _ = Rep.dot daT dA + Rep.dot dbT dB := by
                simp [hdotA, hdotB]

      -- Algebra on dots: unfold `dC` and apply the two matmul adjointness lemmas.
      have hdotC : Rep.dot dC δT = Rep.dot daT dA + Rep.dot dbT dB := by
        have hadd :
            Rep.dot dC δT =
              Rep.dot (Tensor.matmul daT bT) δT + Rep.dot (Tensor.matmul aT dbT) δT := by
          simpa [dC] using
            (Tensor.dot_add_left (left := Tensor.matmul daT bT)
              (middle := Tensor.matmul aT dbT) (right := δT))
        have h1 : Rep.dot (Tensor.matmul daT bT) δT = Rep.dot daT dA := by
          simpa [dA] using (dot_matmul_right_adjoint (A := daT) (B := bT) (C := δT))
        have h2 : Rep.dot (Tensor.matmul aT dbT) δT = Rep.dot dbT dB := by
          simpa [dB] using (dot_matmul_left_adjoint (A := aT) (B := dbT) (C := δT))
        calc
          Rep.dot dC δT
              = Rep.dot (Tensor.matmul daT bT) δT + Rep.dot (Tensor.matmul aT dbT) δT := hadd
          _ = Rep.dot daT dA + Rep.dot dbT dB := by simp [h1, h2]

      calc
        inner ℝ (toVec (t := dC)) δV = Rep.dot dC δT := hL
        _ = Rep.dot daT dA + Rep.dot dbT dB := hdotC
        _ =
            inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVec (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVec (t := dB))) :=
                  hR.symm
      )

/--
`FDerivCorrect` for `matmul`.

The derivative at `(A, B)` is `dA B + A dB`, the product rule for the bilinear map
`Matmul.bilin`.
-/
def Tape.RealNode.matmulFDeriv {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    FDerivCorrect (matmul (Γ := Γ) (m := m) (n := n) (p := p) A B).toNodeData () := by
  classical
  let fA : CtxVec Γ → Vec (Shape.size [m, n]) :=
    fun x => CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A x
  let fB : CtxVec Γ → Vec (Shape.size [n, p]) :=
    fun x => CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B x
  let Bmul :
      Vec (Shape.size [m, n]) →L[ℝ] Vec (Shape.size [n, p]) →L[ℝ]
        Vec (Shape.size [m, p]) :=
    Matmul.bilin (m := m) (n := n) (p := p)

  refine
    { deriv := fun xV =>
        (Bmul.precompR (CtxVec Γ) (fA xV) (CtxVec.getCLM (Γ := Γ) (s := .dim n (.dim p .scalar)) B))
          +
          (Bmul.precompL (CtxVec Γ) (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A) (fB
            xV))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hA :
        HasFDerivAt fA (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A) xV := by
      have h := (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A).hasFDerivAt (x := xV)
      have hfun : fA = fun x => (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A) x := by
        funext x; simp [fA, CtxVec.getCLM_apply]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq
    have hB :
        HasFDerivAt fB (CtxVec.getCLM (Γ := Γ) (s := .dim n (.dim p .scalar)) B) xV := by
      have h := (CtxVec.getCLM (Γ := Γ) (s := .dim n (.dim p .scalar)) B).hasFDerivAt (x := xV)
      have hfun : fB = fun x => (CtxVec.getCLM (Γ := Γ) (s := .dim n (.dim p .scalar)) B) x := by
        funext x; simp [fB, CtxVec.getCLM_apply]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq

    have hbilin :=
      ContinuousLinearMap.hasFDerivAt_of_bilinear (B := Bmul) (hf := hA) (hg := hB)

    have hEq :
        (NodeData.forwardVec (Γ := Γ) (τ := .dim m (.dim p .scalar))
          (matmul (Γ := Γ) (m := m) (n := n) (p := p) A B).toNodeData ())
          =
        (fun xV : CtxVec Γ => (Bmul (fA xV)) (fB xV)) := by
      funext ctxV
      simp [matmul, LawfulNode.forwardVec_ofFn, fA, fB, Bmul, Matmul.toVec_matmul]
    exact hbilin.congr_of_eventuallyEq hEq.eventuallyEq

  · intro xV dxV
    -- Rewrite the node JVP into the bilinear derivative formula: `toVec` respects matrix
    -- addition, and `toVec (Tensor.matmul (ofVec a) (ofVec b))` is
    -- `Matmul.vec a b`.
    ext ip
    -- After expanding, the two bilinear terms may appear in the opposite order.
    simp [matmul, LawfulNode.jvpVec_ofFn, fA, fB, Bmul, toVec_add,
      Matmul.toVec_matmul, ContinuousLinearMap.comp_apply,
      CtxVec.getCLM_apply]
    ring

-- ---------------------------------------------------------------------------
-- Matrix broadcasts and row-wise reductions (linear)
-- ---------------------------------------------------------------------------

namespace Tape.RealNode.MatrixLinear

open scoped BigOperators

/-- Broadcast a vector `v : Vec m` across the last axis to a flattened `(m×n)` matrix. -/
def broadcastRowCLM {m n : Nat} : Vec m →L[ℝ] Vec (Shape.size [m, n]) := by
  classical
  let fLin : Vec m →ₗ[ℝ] Vec (Shape.size [m, n]) :=
    { toFun := fun v =>
        WithLp.toLp 2 fun ip => v (ip.divNat (m := m) (n := Shape.size [n]))
      map_add' := by
        intro v w
        ext ip
        simp
      map_smul' := by
        intro a v
        ext ip
        simp [smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Broadcast a vector `v : Vec n` across the first axis to a flattened `(m×n)` matrix. -/
def broadcastColCLM {m n : Nat} : Vec n →L[ℝ] Vec (Shape.size [m, n]) := by
  classical
  let hn : Shape.size [n] = n := Shape.size_vector n
  let fLin : Vec n →ₗ[ℝ] Vec (Shape.size [m, n]) :=
    { toFun := fun v =>
        let v' : Vec (Shape.size [n]) :=
          LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hn.symm) v
        WithLp.toLp 2 fun ip => v' (ip.modNat (m := m) (n := Shape.size [n]))
      map_add' := by
        intro v w
        ext ip
        simp
      map_smul' := by
        intro a v
        ext ip
        simp [smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Row-wise sum: flattened `(m×n)` matrix → vector `m`. -/
def rowSumCLM {m n : Nat} : Vec (Shape.size [m, n]) →L[ℝ] Vec m := by
  classical
  let fLin : Vec (Shape.size [m, n]) →ₗ[ℝ] Vec m :=
    { toFun := fun x => WithLp.toLp 2 fun i =>
        ∑ j : Fin (Shape.size [n]), x (finProdFinEquiv (i, j))
      map_add' := by
        intro x y
        apply PiLp.ext
        intro i
        simp only [WithLp.ofLp_add, Pi.add_apply]
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro j _
        rfl
      map_smul' := by
        intro a x
        apply PiLp.ext
        intro i
        simp only [WithLp.ofLp_smul, Pi.smul_apply, RingHom.id_apply,
          smul_eq_mul]
        rw [Finset.mul_sum]
        rfl }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Row-wise mean: flattened `(m×n)` matrix → vector `m`. -/
def rowMeanCLM {m n : Nat} : Vec (Shape.size [m, n]) →L[ℝ] Vec m :=
  ((1 : ℝ) / (n : ℝ)) • rowSumCLM (m := m) (n := n)

end Tape.RealNode.MatrixLinear

/-- Broadcast a vector `(.dim m .scalar)` across columns to `(.dim m (.dim n .scalar))`. -/
def Tape.RealNode.broadcastRow {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m .scalar)) : LawfulNode (α := ℝ) Unit Γ (.dim m (.dim n .scalar)) :=
  LawfulNode.ofFn (Γ := Γ) (τ := .dim m (.dim n .scalar))
    (f := fun xV => MatrixLinear.broadcastRowCLM (m := m) (n := n) (getVec (Γ := Γ) (n := m) idx
      xV))
    (jvp := fun _xV dxV => MatrixLinear.broadcastRowCLM (m := m) (n := n) (getVec (Γ := Γ) (n := m)
      idx dxV))
    (vjp := fun _xV δV =>
      singleVec (Γ := Γ) (n := m) idx ((MatrixLinear.broadcastRowCLM (m := m) (n := n)).adjoint δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        inner_getVec_singleVec (Γ := Γ) (n := m) idx dxV
          ((MatrixLinear.broadcastRowCLM (m := m) (n := n)).adjoint δV)
      have hadj :
          inner ℝ
              (MatrixLinear.broadcastRowCLM (m := m) (n := n) (getVec (Γ := Γ) (n := m) idx dxV))
              δV
            =
          inner ℝ (getVec (Γ := Γ) (n := m) idx dxV)
            ((MatrixLinear.broadcastRowCLM (m := m) (n := n)).adjoint δV) := by
        simpa using
          (ContinuousLinearMap.adjoint_inner_right (A := MatrixLinear.broadcastRowCLM (m := m) (n :=
            n))
            (x := getVec (Γ := Γ) (n := m) idx dxV) (y := δV)).symm
      exact hadj.trans hctx.symm)

/-- `FDerivCorrect` for `broadcastRow` (linear op). -/
def Tape.RealNode.broadcastRowFDeriv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m .scalar)) :
    FDerivCorrect (broadcastRow (Γ := Γ) (m := m) (n := n) idx).toNodeData () := by
  classical
  refine
    { deriv := fun _ =>
        (MatrixLinear.broadcastRowCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := m) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Shape.size [m, n]) :=
      (MatrixLinear.broadcastRowCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := m) idx)
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (NodeData.forwardVec (Γ := Γ) (τ := .dim m (.dim n .scalar))
            (broadcastRow (Γ := Γ) (m := m) (n := n) idx).toNodeData ())
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [broadcastRow, D, LawfulNode.forwardVec_ofFn, getVecCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    simp [broadcastRow, LawfulNode.jvpVec_ofFn, getVecCLM_apply, ContinuousLinearMap.comp_apply]

/-- Broadcast a vector `(.dim n .scalar)` across rows to `(.dim m (.dim n .scalar))`. -/
def Tape.RealNode.broadcastCol {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim n .scalar)) : LawfulNode (α := ℝ) Unit Γ (.dim m (.dim n .scalar)) :=
  LawfulNode.ofFn (Γ := Γ) (τ := .dim m (.dim n .scalar))
    (f := fun xV => MatrixLinear.broadcastColCLM (m := m) (n := n) (getVec (Γ := Γ) (n := n) idx
      xV))
    (jvp := fun _xV dxV => MatrixLinear.broadcastColCLM (m := m) (n := n) (getVec (Γ := Γ) (n := n)
      idx dxV))
    (vjp := fun _xV δV =>
      singleVec (Γ := Γ) (n := n) idx ((MatrixLinear.broadcastColCLM (m := m) (n := n)).adjoint δV))
    (correct_inner := by
      intro _xV dxV δV
      classical
      have hctx :=
        inner_getVec_singleVec (Γ := Γ) (n := n) idx dxV
          ((MatrixLinear.broadcastColCLM (m := m) (n := n)).adjoint δV)
      have hadj :
          inner ℝ
              (MatrixLinear.broadcastColCLM (m := m) (n := n) (getVec (Γ := Γ) (n := n) idx dxV))
              δV
            =
          inner ℝ (getVec (Γ := Γ) (n := n) idx dxV)
            ((MatrixLinear.broadcastColCLM (m := m) (n := n)).adjoint δV) := by
        simpa using
          (ContinuousLinearMap.adjoint_inner_right (A := MatrixLinear.broadcastColCLM (m := m) (n :=
            n))
            (x := getVec (Γ := Γ) (n := n) idx dxV) (y := δV)).symm
      exact hadj.trans hctx.symm)

/-- `FDerivCorrect` for `broadcastCol` (linear op). -/
def Tape.RealNode.broadcastColFDeriv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim n .scalar)) :
    FDerivCorrect (broadcastCol (Γ := Γ) (m := m) (n := n) idx).toNodeData () := by
  classical
  refine
    { deriv := fun _ =>
        (MatrixLinear.broadcastColCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := n) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Shape.size [m, n]) :=
      (MatrixLinear.broadcastColCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := n) idx)
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (NodeData.forwardVec (Γ := Γ) (τ := .dim m (.dim n .scalar))
            (broadcastCol (Γ := Γ) (m := m) (n := n) idx).toNodeData ())
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [broadcastCol, D, LawfulNode.forwardVec_ofFn, getVecCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    simp [broadcastCol, LawfulNode.jvpVec_ofFn, getVecCLM_apply, ContinuousLinearMap.comp_apply]

-- ---------------------------------------------------------------------------
-- Reshapes (defined in `Shape.lean`) and the row-wise mean
-- ---------------------------------------------------------------------------

/-!
Shape-only nodes (`reshape` and the reindexing adapters) live in
`TorchLean.NN.Tape.RealNode.Shape` (namespace `Tape.RealNode`).
-/

/-- Row-wise mean (reduce last axis): `(.dim m (.dim n .scalar)) → (.dim m .scalar)`. -/
def Tape.RealNode.rowMean {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) : LawfulNode (α := ℝ) Unit Γ (.dim m .scalar) :=
  let outShape : Shape := .dim m .scalar
  let hsz : Shape.size outShape = m := by simp [outShape]
  LawfulNode.ofFn (Γ := Γ) (τ := outShape)
    (f := fun xV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz.symm) <|
        MatrixLinear.rowMeanCLM (m := m) (n := n)
          (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx xV))
    (jvp := fun _xV dxV =>
      LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz.symm) <|
        MatrixLinear.rowMeanCLM (m := m) (n := n)
          (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV))
    (vjp := fun _xV δV =>
      let δm : Vec m := LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz) δV
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) idx
        ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm))
    (correct_inner := by
      intro _xV dxV δV
      classical
      let dxMat : Vec (Shape.size [m, n]) :=
        CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV
      let δm : Vec m := LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz) δV
      have hctx :=
        (CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV
          ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm))
      have hLcast :
          inner ℝ (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ (finCongr hsz.symm)
              (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat)) δV =
            inner ℝ (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) δm := by
        rw [LinearIsometryEquiv.inner_map_eq_flip, LinearIsometryEquiv.piLpCongrLeft_symm,
          finCongr_symm]
      have hadj :
          inner ℝ (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) δm =
            inner ℝ dxMat ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm) := by
        simpa [δm] using
          (ContinuousLinearMap.adjoint_inner_right (A := MatrixLinear.rowMeanCLM (m := m) (n := n))
            (x := dxMat) (y := δm)).symm
      exact (hLcast.trans hadj).trans hctx.symm)

/-- `FDerivCorrect` for `rowMean` (reduce-mean along the last axis). -/
def Tape.RealNode.rowMeanFDeriv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) :
    FDerivCorrect (rowMean (Γ := Γ) (m := m) (n := n) idx).toNodeData () := by
  classical
  let outShape : Shape := .dim m .scalar
  let hsz : Shape.size outShape = m := by simp [outShape]
  refine
    { deriv := fun _ =>
        (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
            (finCongr hsz.symm)).toContinuousLinearEquiv.toContinuousLinearMap.comp
          ((MatrixLinear.rowMeanCLM (m := m) (n := n)).comp
            (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) idx))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Shape.size outShape) :=
      (LinearIsometryEquiv.piLpCongrLeft 2 ℝ ℝ
          (finCongr hsz.symm)).toContinuousLinearEquiv.toContinuousLinearMap.comp
        ((MatrixLinear.rowMeanCLM (m := m) (n := n)).comp
          (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) idx))
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (NodeData.forwardVec (Γ := Γ) (τ := outShape)
            (rowMean (Γ := Γ) (m := m) (n := n) idx).toNodeData ())
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [rowMean, outShape, D, LawfulNode.forwardVec_ofFn, CtxVec.getCLM_apply,
        ContinuousLinearMap.comp_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext i
    simp [rowMean, outShape, LawfulNode.jvpVec_ofFn, ContinuousLinearMap.comp_apply,
      CtxVec.getCLM_apply]


end

end

end TorchLean
