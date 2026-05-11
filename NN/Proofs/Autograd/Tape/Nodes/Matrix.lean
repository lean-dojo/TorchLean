/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Arithmetic

/-!
# Matrix tape nodes

Matrix multiplication, transpose, row/column broadcasting, and row means, with VJP correctness facts
stated at the vectorized tape level.
-/

@[expose] public section

namespace Proofs
namespace Autograd

open Spec
open Tensor

noncomputable section

open scoped BigOperators

namespace TapeNodes

-- ---------------------------------------------------------------------------
-- Linear algebra: matrix-matrix multiplication
-- ---------------------------------------------------------------------------

namespace Matmul

open scoped BigOperators

/-- Flattened size of an `m×n` matrix shape: `Shape.size (.dim m (.dim n .scalar)) = m*n`. -/
abbrev matSize (m n : Nat) : Nat :=
  Shape.size (.dim m (.dim n .scalar))

/-- Flattened size of a length-`n` vector shape: `Shape.size (.dim n .scalar) = n`. -/
abbrev vecSize (n : Nat) : Nat :=
  Shape.size (.dim n .scalar)

  @[simp] lemma vecSize_eq (n : Nat) : vecSize n = n := by
    simp [vecSize, Shape.size]

  /-- Convert `(i,j)` coordinates into a flattened index for an `m×n` matrix vectorization. -/
  def idxMN {m n : Nat} (i : Fin m) (j : Fin n) : Fin (matSize m n) :=
    let hn : vecSize n = n := vecSize_eq n
    -- `matSize m n` is definitionally `m * vecSize n`, so `finProdFinEquiv` targets `Fin (matSize m
    -- n)`.
    finProdFinEquiv (i, Fin.cast hn.symm j)

  /-- Relate the tensor vectorization `toVecT` to `Spec.get2` at a matrix coordinate. -/
  private lemma toVecT_get2 {m n : Nat} (A : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin m) (j :
    Fin n) :
      toVecT (t := A) (idxMN (m := m) (n := n) i j) = Spec.get2 A i j := by
  cases n with
  | zero =>
      exact (Fin.elim0 j)
  | succ n =>
      cases A with
      | dim rows =>
          let hn : vecSize (Nat.succ n) = Nat.succ n := by simp [vecSize, Shape.size]
          let j' : Fin (vecSize (Nat.succ n)) := Fin.cast hn.symm j
          have hmpos : 0 < vecSize (Nat.succ n) := by simp [vecSize, Shape.size]
          have houter :=
            toVecT_dim_apply (n := m) (s := .dim (Nat.succ n) .scalar) (hmpos := hmpos) (f := rows)
              (p := (i, j'))
          cases hrow : rows i with
          | dim cols =>
              let k0 : Fin 1 := 0
              have hinnerPos : 0 < Shape.size Shape.scalar := by simp [Shape.size]
              have hinner :=
                toVecT_dim_apply (n := Nat.succ n) (s := Shape.scalar) (hmpos := hinnerPos) (f :=
                  cols)
                  (p := (j, k0))
              have hjidx : finProdFinEquiv (j, k0) = j' := by
                apply Fin.ext
                simp [j', k0, Shape.size, finProdFinEquiv]
              cases hx : cols j with
              | scalar x =>
                  have hscalar : toVecT (t := (Tensor.scalar x : Tensor ℝ Shape.scalar)) k0 = x :=
                    by
                    simpa [toVecT, toVecE, flattenSpec, Shape.size, Spec.toVec, k0] using
                      (euclideanEquiv_symm_ofLp
                        (n := Shape.size Shape.scalar)
                        (f := fun _ : Fin (Shape.size Shape.scalar) => x)
                        (i := k0))
                  have hidx : idxMN (m := m) (n := Nat.succ n) i j = finProdFinEquiv (i, j') := by
                    apply Fin.ext
                    simp [idxMN, j', hn]
                  have houter' :
                      toVecT (t := Tensor.dim rows) (idxMN (m := m) (n := Nat.succ n) i j) =
                        toVecT (t := rows i) j' := by
                    simpa [hidx] using houter
                  have hrowCoord : toVecT (t := rows i) j' = x := by
                    have hconv :
                        toVecT (t := Tensor.dim cols) (finProdFinEquiv (j, k0)) =
                          toVecT (t := Tensor.dim cols) j' :=
                      congrArg (fun z => toVecT (t := Tensor.dim cols) z) hjidx
                    have hinner' : toVecT (t := Tensor.dim cols) j' = toVecT (t := cols j) k0 :=
                      hconv.symm.trans hinner
                    have hinner'' :
                        toVecT (t := Tensor.dim cols) j' = toVecT (t := (Tensor.scalar x : Tensor ℝ
                          Shape.scalar)) k0 := by
                      simpa [hx] using hinner'
                    simpa [hrow] using (hinner''.trans hscalar)
                  -- `get2` picks out exactly this scalar entry.
                  simpa [Spec.get2, Spec.get, Spec.getAtSpec, hrow, hx, houter', hrowCoord]
                    using (houter'.trans hrowCoord)

  /-- `Spec.get2` of an `ofVecT`-constructed matrix reads back the corresponding flattened entry. -/
  private lemma get2_ofVecT {m n : Nat} (v : Vec (matSize m n)) (i : Fin m) (j : Fin n) :
      Spec.get2 (ofVecT (s := .dim m (.dim n .scalar)) v) i j = v (idxMN (m := m) (n := n) i j) :=
        by
  have htv :
      toVecT (t := ofVecT (s := .dim m (.dim n .scalar)) v) (idxMN (m := m) (n := n) i j) = v (idxMN
        (m := m) (n := n) i j) := by
    simp
  exact (toVecT_get2 (A := ofVecT (s := .dim m (.dim n .scalar)) v) i j).symm.trans htv

  /-- Entrywise formula for matrix addition: `(A + B)[i,j] = A[i,j] + B[i,j]`. -/
  private lemma get2_add_spec {m n : Nat} (A B : Tensor ℝ (.dim m (.dim n .scalar))) (i : Fin m) (j
    : Fin n) :
      Spec.get2 (addSpec A B) i j = Spec.get2 A i j + Spec.get2 B i j := by
  cases A with
  | dim rowsA =>
      cases B with
      | dim rowsB =>
          cases hrowA : rowsA i with
          | dim colsA =>
              cases hrowB : rowsB i with
              | dim colsB =>
                  cases hA : colsA j with
                  | scalar a =>
                      cases hB : colsB j with
                      | scalar b =>
                          simp [addSpec, Spec.Tensor.addSpec, Spec.Tensor.map2Spec, Spec.get2,
                            Spec.get, Spec.getAtSpec,
                            hrowA, hrowB, hA, hB]

  /-- Vectorization commutes with matrix addition: `toVecT (A + B) = toVecT A + toVecT B`. -/
  lemma toVecT_add_spec_mat {m n : Nat} (A B : Tensor ℝ (.dim m (.dim n .scalar))) :
      toVecT (t := addSpec A B) = toVecT (t := A) + toVecT (t := B) := by
  classical
  ext ip
  let hp : vecSize n = n := by simp [vecSize, Shape.size]
  let i : Fin m := (ip.divNat (m := m) (n := vecSize n))
  let j' : Fin (vecSize n) := (ip.modNat (m := m) (n := vecSize n))
  let j : Fin n := Fin.cast hp j'
  have hip : idxMN (m := m) (n := n) i j = ip := by
    have hbase : finProdFinEquiv (i, j') = ip := by
      simpa [i, j'] using
        (Equiv.apply_symm_apply (e := (finProdFinEquiv : Fin m × Fin (vecSize n) ≃ Fin (m * vecSize
          n))) ip)
    simpa [idxMN, j, hp] using hbase
  -- Convert the LHS via `get2`, use elementwise addition, then convert back.
  have hgetL : toVecT (t := addSpec A B) ip = Spec.get2 (addSpec A B) i j := by
    -- rewrite the index to match `toVecT_get2`
    rw [←hip]
    exact toVecT_get2 (A := addSpec A B) i j
  have hgetA : toVecT (t := A) ip = Spec.get2 A i j := by
    rw [←hip]
    exact toVecT_get2 (A := A) i j
  have hgetB : toVecT (t := B) ip = Spec.get2 B i j := by
    rw [←hip]
    exact toVecT_get2 (A := B) i j
  calc
    toVecT (t := addSpec A B) ip
        = Spec.get2 (addSpec A B) i j := hgetL
    _ = Spec.get2 A i j + Spec.get2 B i j := get2_add_spec (A := A) (B := B) i j
    _ = toVecT (t := A) ip + toVecT (t := B) ip := by simp [hgetA, hgetB]

/-- A bilinear map on flattened matrices: `(m×n) × (n×p) → (m×p)` on `Vec (Shape.size ...)`. -/
def matmulVec {m n p : Nat} (a : Vec (matSize m n)) (b : Vec (matSize n p)) : Vec (matSize m p) :=
  vecOfFun (n := matSize m p) fun ip =>
    let hp : vecSize p = p := vecSize_eq p
    let i : Fin m := (ip.divNat (m := m) (n := vecSize p))
    let k' : Fin (vecSize p) := (ip.modNat (m := m) (n := vecSize p))
    let k : Fin p := Fin.cast hp k'
    ∑ j : Fin n, a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)

@[simp] lemma matmulVec_apply {m n p : Nat} (a : Vec (matSize m n)) (b : Vec (matSize n p))
    (ip : Fin (matSize m p)) :
    matmulVec (m := m) (n := n) (p := p) a b ip =
      let hp : vecSize p = p := vecSize_eq p
      let i : Fin m := (ip.divNat (m := m) (n := vecSize p))
      let k' : Fin (vecSize p) := (ip.modNat (m := m) (n := vecSize p))
      let k : Fin p := Fin.cast hp k'
      ∑ j : Fin n, a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k) := by
  simp [matmulVec]

/-!
Matrix multiplication is developed at the vector level (flattened matrices) to integrate cleanly
with `CtxVec` and the `HasFDerivAt` machinery.

PyTorch analogue: `torch.matmul` / `@` operator on 2D tensors.
https://pytorch.org/docs/stable/generated/torch.matmul.html
-/

/-- For fixed left operand `a`, `matmulCLMRight a` is the linear map `b ↦ a*b`. -/
def matmulCLMRight {m n p : Nat} (a : Vec (matSize m n)) : Vec (matSize n p) →L[ℝ] Vec (matSize m p)
  := by
  classical
  let fLin : Vec (matSize n p) →ₗ[ℝ] Vec (matSize m p) :=
    { toFun := fun b => matmulVec (m := m) (n := n) (p := p) a b
      map_add' := by
        intro b1 b2
        ext ip
        simp [matmulVec, Finset.sum_add_distrib, mul_add]
      map_smul' := by
        intro r b
        ext ip
        simp [matmulVec, Finset.mul_sum, mul_left_comm] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

set_option maxHeartbeats 5000000 in
/-- Continuous bilinear map for matrix multiplication on flattened vectors. -/
def matmulBilin {m n p : Nat} :
    Vec (matSize m n) →L[ℝ] Vec (matSize n p) →L[ℝ] Vec (matSize m p) := by
  classical
  -- Define the underlying bilinear map on pairs.
  let f : Vec (matSize m n) × Vec (matSize n p) → Vec (matSize m p) :=
    fun x => matmulVec (m := m) (n := n) (p := p) x.1 x.2
  have hf : IsBoundedBilinearMap ℝ f := by
    refine
      { add_left := ?_
        smul_left := ?_
        add_right := ?_
        smul_right := ?_
        bound := ?_ }
    · intro a1 a2 b
      ext ip
      simp [f, matmulVec, Finset.sum_add_distrib, add_mul,
        ]
    · intro r a b
      ext ip
      simp [f, matmulVec, smul_eq_mul, Finset.mul_sum, mul_assoc]
    · intro a b1 b2
      ext ip
      simp [f, matmulVec, Finset.sum_add_distrib, mul_add,
        ]
    · intro r a b
      ext ip
      simp [f, matmulVec, smul_eq_mul, Finset.mul_sum, mul_left_comm]
    · -- A crude global bound for Euclidean (L2) norms, using `‖x i‖ ≤ ‖x‖` and `‖∑‖ ≤ ∑‖‖`.
      refine ⟨Real.sqrt (matSize m p) * (n : ℝ) + 1, ?_, ?_⟩
      · -- positivity
        have hnonneg : 0 ≤ Real.sqrt (matSize m p) * (n : ℝ) := by
          have hs : 0 ≤ Real.sqrt (matSize m p) := Real.sqrt_nonneg _
          have hn : 0 ≤ (n : ℝ) := by exact_mod_cast (Nat.zero_le n)
          exact mul_nonneg hs hn
        exact add_pos_of_nonneg_of_pos hnonneg zero_lt_one
      · intro a b
        -- Bound each coordinate by `n * ‖a‖ * ‖b‖`.
        let M : ℝ := (n : ℝ) * ‖a‖ * ‖b‖
        have hM : 0 ≤ M := by
          have hn : 0 ≤ (n : ℝ) := by exact_mod_cast (Nat.zero_le n)
          exact mul_nonneg (mul_nonneg hn (norm_nonneg a)) (norm_nonneg b)
        have hcoord : ∀ ip : Fin (matSize m p), ‖matmulVec (m := m) (n := n) (p := p) a b ip‖ ≤ M :=
          by
          intro ip
          let hp : vecSize p = p := vecSize_eq p
          let i : Fin m := (ip.divNat (m := m) (n := vecSize p))
          let k' : Fin (vecSize p) := (ip.modNat (m := m) (n := vecSize p))
          let k : Fin p := Fin.cast hp k'
          -- unfold the coordinate formula and apply triangle inequality
          have hsum :
              ‖∑ j : Fin n,
                  a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)‖
                ≤
              ∑ j : Fin n,
                  ‖a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)‖ := by
            -- `norm_sum_le` on `Finset.univ`
            simpa using
              (norm_sum_le (s := (Finset.univ : Finset (Fin n)))
                (f := fun j : Fin n => a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p)
                  j k)))
          have hterm :
              ∀ j : Fin n,
                ‖a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)‖
                  ≤ ‖a‖ * ‖b‖ := by
            intro j
            have ha : ‖a (idxMN (m := m) (n := n) i j)‖ ≤ ‖a‖ :=
              PiLp.norm_apply_le (x := a) (i := idxMN (m := m) (n := n) i j)
            have hb : ‖b (idxMN (m := n) (n := p) j k)‖ ≤ ‖b‖ :=
              PiLp.norm_apply_le (x := b) (i := idxMN (m := n) (n := p) j k)
            -- `‖x*y‖ = ‖x‖*‖y‖` and then bound each factor by the vector norms.
            calc
              ‖a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)‖
                  = ‖a (idxMN (m := m) (n := n) i j)‖ * ‖b (idxMN (m := n) (n := p) j k)‖ := by
                      exact norm_mul (a (idxMN (m := m) (n := n) i j)) (b (idxMN (m := n) (n := p) j
                        k))
              _ ≤ ‖a‖ * ‖b‖ := by
                    exact mul_le_mul ha hb (norm_nonneg _) (norm_nonneg _)
          have hsum' :
              ∑ j : Fin n,
                  ‖a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)‖
                ≤
              ∑ _j : Fin n, ‖a‖ * ‖b‖ := by
            refine Finset.sum_le_sum ?_
            intro j hj
            exact hterm j
          -- assemble
          have hmain :
              ‖matmulVec (m := m) (n := n) (p := p) a b ip‖ ≤ (n : ℝ) * (‖a‖ * ‖b‖) := by
            -- unfold `matmulVec` at coordinate `ip`
            have hdef :
                matmulVec (m := m) (n := n) (p := p) a b ip
                  =
                ∑ j : Fin n,
                  a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k) := by
              simp [i, k', k]
            -- use triangle inequality + term bounds
            have h0 :=
              (hdef ▸ hsum)
            have h1 : ‖∑ j : Fin n,
                  a (idxMN (m := m) (n := n) i j) * b (idxMN (m := n) (n := p) j k)‖
                ≤ (n : ℝ) * (‖a‖ * ‖b‖) := by
              -- bound the RHS sum of norms by `n * (‖a‖*‖b‖)`
              have hcard :
                  (∑ _j : Fin n, ‖a‖ * ‖b‖) = (n : ℝ) * (‖a‖ * ‖b‖) := by
                simp
              exact h0.trans (hsum'.trans_eq hcard)
            simpa [hdef] using h1
          -- final coordinate bound, rewriting `M`
          have : (n : ℝ) * (‖a‖ * ‖b‖) = M := by
            simp [M, mul_assoc]
          simpa [this] using hmain
        -- Now bound the full `L2` norm via coordinatewise square bounds.
        have hL2 :
            ‖matmulVec (m := m) (n := n) (p := p) a b‖ ≤ Real.sqrt (matSize m p) * M := by
          -- Use `EuclideanSpace.norm_eq` and compare sums under `sqrt`.
          have hsq_le :
              ∑ ip : Fin (matSize m p), ‖matmulVec (m := m) (n := n) (p := p) a b ip‖ ^ 2
                ≤ ∑ _ip : Fin (matSize m p), M ^ 2 := by
            refine Finset.sum_le_sum ?_
            intro ip hip
            have h0 : ‖matmulVec (m := m) (n := n) (p := p) a b ip‖ ≤ M := hcoord ip
            have hn0 : 0 ≤ ‖matmulVec (m := m) (n := n) (p := p) a b ip‖ := norm_nonneg _
            exact pow_le_pow_left₀ hn0 h0 2
          have hnorm :
              ‖matmulVec (m := m) (n := n) (p := p) a b‖
                = Real.sqrt (∑ ip : Fin (matSize m p), ‖matmulVec (m := m) (n := n) (p := p) a b ip‖
                  ^ 2) := by
            simp [EuclideanSpace.norm_eq]
          have hnorm' :
              ‖matmulVec (m := m) (n := n) (p := p) a b‖
                ≤ Real.sqrt (∑ _ip : Fin (matSize m p), M ^ 2) := by
            -- apply `sqrt_le_sqrt` to the sum inequality
            have hsum_nonneg :
                0 ≤ ∑ ip : Fin (matSize m p), ‖matmulVec (m := m) (n := n) (p := p) a b ip‖ ^ 2 :=
                  by
              exact Finset.sum_nonneg (fun _ _ => sq_nonneg _)
            have hsum_nonneg' :
                0 ≤ ∑ _ip : Fin (matSize m p), M ^ 2 := by
              exact Finset.sum_nonneg (fun _ _ => sq_nonneg _)
            -- rewrite using `hnorm` and compare
            have := Real.sqrt_le_sqrt hsq_le
            simpa [hnorm] using this
          -- compute the RHS sqrt
          have hsum_const :
              (∑ _ip : Fin (matSize m p), M ^ 2) = (matSize m p : ℝ) * (M ^ 2) := by
            simp []
          have hsqrt :
              Real.sqrt (∑ _ip : Fin (matSize m p), M ^ 2) = Real.sqrt (matSize m p) * M := by
            have hk : 0 ≤ (matSize m p : ℝ) := by exact_mod_cast (Nat.zero_le (matSize m p))
            have hM' : 0 ≤ M := hM
            calc
              Real.sqrt (∑ _ip : Fin (matSize m p), M ^ 2)
                  = Real.sqrt ((matSize m p : ℝ) * (M ^ 2)) := by simp [hsum_const]
              _ = Real.sqrt (matSize m p : ℝ) * Real.sqrt (M ^ 2) := by
                    simp
              _ = Real.sqrt (matSize m p) * M := by
                    simp [Real.sqrt_sq_eq_abs, abs_of_nonneg hM']
          exact (hnorm'.trans_eq hsqrt)
        -- Finish by absorbing the `+ 1` slack.
        have hA :
            Real.sqrt (matSize m p) * M ≤ (Real.sqrt (matSize m p) * (n : ℝ) + 1) * ‖a‖ * ‖b‖ := by
          have : Real.sqrt (matSize m p) * M = (Real.sqrt (matSize m p) * (n : ℝ)) * ‖a‖ * ‖b‖ := by
            simp [M, mul_assoc, mul_left_comm, mul_comm]
          -- use `(X ≤ X+1)` and multiply by nonneg `‖a‖*‖b‖`
          have hX : (Real.sqrt (matSize m p) * (n : ℝ)) ≤ (Real.sqrt (matSize m p) * (n : ℝ) + 1) :=
            by
            simp
          have hnn : 0 ≤ ‖a‖ * ‖b‖ := mul_nonneg (norm_nonneg a) (norm_nonneg b)
          calc
            Real.sqrt (matSize m p) * M
                = (Real.sqrt (matSize m p) * (n : ℝ)) * ‖a‖ * ‖b‖ := this
            _ ≤ (Real.sqrt (matSize m p) * (n : ℝ) + 1) * ‖a‖ * ‖b‖ := by
                  have h := mul_le_mul_of_nonneg_right hX hnn
                  simpa [mul_assoc] using h
        exact hL2.trans hA
  -- curry the bounded bilinear map into a `→L →L` map
  exact hf.toContinuousLinearMap

set_option maxHeartbeats 5000000 in
@[simp] lemma matmulBilin_apply {m n p : Nat} (a : Vec (matSize m n)) (b : Vec (matSize n p)) :
    matmulBilin (m := m) (n := n) (p := p) a b = matmulVec (m := m) (n := n) (p := p) a b := by
  classical
  -- Unfold to the bounded bilinear map and use `toContinuousLinearMap_apply`.
  simp [matmulBilin]

/-- `Spec.mat_mul_spec` agrees with `matmulVec` after flattening both inputs/outputs. -/
lemma forward_eq_matmulVec {m n p : Nat} (aV : Vec (matSize m n)) (bV : Vec (matSize n p)) :
    toVecT (t := Spec.matMulSpec (ofVecT (s := .dim m (.dim n .scalar)) aV) (ofVecT (s := .dim n
      (.dim p .scalar)) bV))
      =
    matmulVec (m := m) (n := n) (p := p) aV bV := by
  classical
  ext ip
  -- represent `ip` as a row/column pair using `Fin.divNat/modNat` for `m * vecSize p`
  let i : Fin m := (ip.divNat (m := m) (n := vecSize p))
  let k' : Fin (vecSize p) := (ip.modNat (m := m) (n := vecSize p))
  let hp : vecSize p = p := by simp [vecSize, Shape.size]
  let k : Fin p := Fin.cast hp k'
  -- interpret LHS coordinate via `get2` and the matrix entry lemma
  have hL :
      toVecT
          (t := Spec.matMulSpec (ofVecT (s := .dim m (.dim n .scalar)) aV)
            (ofVecT (s := .dim n (.dim p .scalar)) bV)) ip
        =
      Spec.get2
          (Spec.matMulSpec (ofVecT (s := .dim m (.dim n .scalar)) aV)
            (ofVecT (s := .dim n (.dim p .scalar)) bV)) i k := by
    -- rewrite `ip` as the flattened `(i,k)` index
    have hip : idxMN (m := m) (n := p) i k = ip := by
      have hbase : finProdFinEquiv (i, k') = ip := by
        simpa [i, k'] using
          (Equiv.apply_symm_apply (e := (finProdFinEquiv : Fin m × Fin (vecSize p) ≃ Fin (m *
            vecSize p))) ip)
      simpa [idxMN, k, hp] using hbase
    rw [←hip]
    exact toVecT_get2
      (A := Spec.matMulSpec (ofVecT (s := .dim m (.dim n .scalar)) aV)
        (ofVecT (s := .dim n (.dim p .scalar)) bV))
      i k
  have hEntry :=
    get2_mat_mul_spec
      (A := ofVecT (s := .dim m (.dim n .scalar)) aV)
      (B := ofVecT (s := .dim n (.dim p .scalar)) bV)
      (i := i) (j := k)
  have hA : ∀ j : Fin n, Spec.get2 (ofVecT (s := .dim m (.dim n .scalar)) aV) i j = aV (idxMN (m :=
    m) (n := n) i j) :=
    fun j => get2_ofVecT (v := aV) i j
  have hB : ∀ j : Fin n, Spec.get2 (ofVecT (s := .dim n (.dim p .scalar)) bV) j k = bV (idxMN (m :=
    n) (n := p) j k) :=
    fun j => get2_ofVecT (v := bV) j k
  calc
    toVecT
        (t := Spec.matMulSpec (ofVecT (s := .dim m (.dim n .scalar)) aV)
          (ofVecT (s := .dim n (.dim p .scalar)) bV)) ip
        =
      Spec.get2
          (Spec.matMulSpec (ofVecT (s := .dim m (.dim n .scalar)) aV)
            (ofVecT (s := .dim n (.dim p .scalar)) bV)) i k := hL
    _ = ∑ j : Fin n,
          Spec.get2 (ofVecT (s := .dim m (.dim n .scalar)) aV) i j *
            Spec.get2 (ofVecT (s := .dim n (.dim p .scalar)) bV) j k := by
          simpa using hEntry
    _ = ∑ j : Fin n, aV (idxMN (m := m) (n := n) i j) * bV (idxMN (m := n) (n := p) j k) := by
          simp [hA, hB]
    _ = matmulVec (m := m) (n := n) (p := p) aV bV ip := by
          simp [matmulVec, i, k, k']

end Matmul

-- ---------------------------------------------------------------------------
-- Linear algebra: matrix transpose
-- ---------------------------------------------------------------------------

namespace MatTranspose

open Matmul

/-- Helper: `matSize m n` is definitionally `m * n`. -/
lemma matSize_eq_mul (m n : Nat) : Matmul.matSize m n = m * n := by
  simp [Matmul.matSize, Shape.size]

/-- Equivalence implementing matrix transpose on flattened indices. -/
def transposeEquiv (m n : Nat) : Fin (m * n) ≃ Fin (n * m) :=
  (finProdFinEquiv.symm.trans (Equiv.prodComm (Fin m) (Fin n))).trans finProdFinEquiv

/-- The transpose index equivalence is symmetric up to swapping `m` and `n`. -/
private lemma transposeEquiv_symm (m n : Nat) :
    (transposeEquiv m n).symm = transposeEquiv n m := by
  ext k; simp [transposeEquiv, Equiv.prodComm_symm]

/-- Transpose on flattened matrices: `(m×n)` flattened row-major → `(n×m)` flattened row-major. -/
def transposeVec {m n : Nat} (a : Vec (Matmul.matSize m n)) : Vec (Matmul.matSize n m) :=
  castVec (matSize_eq_mul n m).symm <|
    vecOfFun (n := n * m) (fun k : Fin (n * m) =>
      (castVec (matSize_eq_mul m n) a) ((transposeEquiv m n).symm k))

/-- Adjointness of `transposeVec` with respect to the standard inner product on vectors. -/
private lemma inner_transposeVec {m n : Nat} (x : Vec (Matmul.matSize m n)) (y : Vec (Matmul.matSize
  n m)) :
    inner ℝ (transposeVec (m := m) (n := n) x) y =
      inner ℝ x (transposeVec (m := n) (n := m) y) := by
  classical
  let x' : Vec (m * n) := castVec (matSize_eq_mul m n) x
  let y' : Vec (n * m) := castVec (matSize_eq_mul n m) y
  let e : Fin (m * n) ≃ Fin (n * m) := transposeEquiv m n
  have hx :
      castVec (matSize_eq_mul n m) (transposeVec (m := m) (n := n) x) =
        vecOfFun (n := n * m) (fun k : Fin (n * m) => x' (e.symm k)) := by
    ext k
    simp [transposeVec, x', e, castVec_castVec]
  have hy :
      castVec (matSize_eq_mul m n) (transposeVec (m := n) (n := m) y) =
        vecOfFun (n := m * n) (fun k : Fin (m * n) => y' (e k)) := by
    have hswap : (transposeEquiv n m).symm = e := by
      simpa [e] using (transposeEquiv_symm (m := n) (n := m))
    ext k
    have hk : (transposeEquiv n m).symm k = e k := by
      simpa using congrArg (fun f => f k) hswap
    have hL :
        castVec (matSize_eq_mul m n) (transposeVec (m := n) (n := m) y) k =
          castVec (matSize_eq_mul n m) y ((transposeEquiv n m).symm k) := by
      simp [transposeVec]
    calc
      castVec (matSize_eq_mul m n) (transposeVec (m := n) (n := m) y) k
          = castVec (matSize_eq_mul n m) y ((transposeEquiv n m).symm k) := hL
      _ = castVec (matSize_eq_mul n m) y (e k) := by simp [hk]
      _ = y' (e k) := by rfl
  have hL :
      inner ℝ (transposeVec (m := m) (n := n) x) y =
        inner ℝ (castVec (matSize_eq_mul n m) (transposeVec (m := m) (n := n) x))
          (castVec (matSize_eq_mul n m) y) := by
    simpa using
      (inner_castVec_castVec (h := matSize_eq_mul n m) (x := transposeVec (m := m) (n := n) x) (y :=
        y)).symm
  calc
    inner ℝ (transposeVec (m := m) (n := n) x) y
        = inner ℝ (vecOfFun (n := n * m) (fun k : Fin (n * m) => x' (e.symm k))) y' := by
            simp [hL, hx, y']
    _ = ∑ k : Fin (n * m), x' (e.symm k) * y' k := by
          simpa using
            (inner_eq_sum_mul (x := vecOfFun (n := n * m) (fun k : Fin (n * m) => x' (e.symm k))) (y
              := y'))
    _ = ∑ i : Fin (m * n), x' i * y' (e i) := by
          -- change variables `k = e i`
          have hsum :
              (∑ k : Fin (n * m), x' (e.symm k) * y' k) =
                ∑ i : Fin (m * n), x' (e.symm (e i)) * y' (e i) := by
            simpa using (Equiv.sum_comp (e := e) (g := fun k : Fin (n * m) => x' (e.symm k) * y'
              k)).symm
          -- simplify `e.symm (e i)` pointwise under the sum
          refine hsum.trans ?_
          refine Finset.sum_congr rfl ?_
          intro i _
          have hxidx : x' (e.symm (e i)) = x' i := by
            simp
          -- rewrite the left factor, then close by reflexivity
          simp [hxidx]
    _ = inner ℝ x' (vecOfFun (n := m * n) (fun i : Fin (m * n) => y' (e i))) := by
          simpa using
            (inner_eq_sum_mul (x := x') (y := vecOfFun (n := m * n) (fun i : Fin (m * n) => y' (e
              i)))).symm
    _ = inner ℝ (castVec (matSize_eq_mul m n) x) (castVec (matSize_eq_mul m n) (transposeVec (m :=
      n) (n := m) y)) := by
          simp [x', hy]
    _ = inner ℝ x (transposeVec (m := n) (n := m) y) := by
          simpa using
            inner_castVec_castVec (h := matSize_eq_mul m n) (x := x) (y := transposeVec (m := n) (n
              := m) y)

end MatTranspose

/-!
Transpose is implemented as a coordinate permutation on flattened matrices.

PyTorch analogue: `A.transpose(0, 1)` for a 2D tensor.
https://pytorch.org/docs/stable/generated/torch.transpose.html
-/

/-- Tape node computing matrix transpose: `(m×n) ↦ (n×m)`. -/
def matrixTranspose {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) : Node Γ (.dim n (.dim m .scalar)) :=
  Node.ofVec (Γ := Γ) (τ := .dim n (.dim m .scalar))
    (f := fun xV => MatTranspose.transposeVec (m := m) (n := n) (CtxVec.get (Γ := Γ) (s := .dim m
      (.dim n .scalar)) A xV))
    (jvp := fun _xV dxV => MatTranspose.transposeVec (m := m) (n := n) (CtxVec.get (Γ := Γ) (s :=
      .dim m (.dim n .scalar)) A dxV))
    (vjp := fun _xV δV =>
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A
        (MatTranspose.transposeVec (m := n) (n := m) δV))
    (correct_inner := by
      intro _xV dxV δV
      have hT :=
        MatTranspose.inner_transposeVec (m := m) (n := n)
          (x := CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV) (y := δV)
      have hCtx :=
        CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV
          (MatTranspose.transposeVec (m := n) (n := m) δV)
      calc
        inner ℝ
            (MatTranspose.transposeVec (m := m) (n := n)
              (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)) δV
            =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV)
              (MatTranspose.transposeVec (m := n) (n := m) δV) := hT
        _ =
            inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A
                (MatTranspose.transposeVec (m := n) (n := m) δV)) := by
            simpa using hCtx.symm)

/-- `NodeFDerivCorrect` for `matrix_transpose` (it is linear/isometric). -/
def matrixTransposeFderiv {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (matrixTranspose (Γ := Γ) (m := m) (n := n) A) := by
  classical
  let Tlin : Vec (Matmul.matSize m n) →L[ℝ] Vec (Matmul.matSize n m) := by
    classical
    let fLin : Vec (Matmul.matSize m n) →ₗ[ℝ] Vec (Matmul.matSize n m) :=
      { toFun := fun v => MatTranspose.transposeVec (m := m) (n := n) v
        map_add' := by
          intro x y
          ext i
          simp [MatTranspose.transposeVec, vecOfFun]
        map_smul' := by
          intro r x
          ext i
          simp [MatTranspose.transposeVec, vecOfFun, smul_eq_mul] }
    refine ⟨fLin, ?_⟩
    exact LinearMap.continuous_of_finiteDimensional (f := fLin)
  refine
    { deriv := fun _xV => Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hCLM :=
      (Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)).hasFDerivAt (x := xV)
    have hfun :
        (Node.forwardVec (Γ := Γ) (τ := .dim n (.dim m .scalar))
            (matrixTranspose (Γ := Γ) (m := m) (n := n) A)) =
          (fun x : CtxVec Γ =>
            (Tlin.comp (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) A)) x) := by
      funext x
      simp [matrixTranspose, Node.forwardVec_ofVec, ContinuousLinearMap.comp_apply,
        CtxVec.getCLM_apply, Tlin]
    exact hCLM.congr_of_eventuallyEq hfun.eventuallyEq
  · intro _xV dxV
    simp [matrixTranspose, Node.jvpVec_ofVec, ContinuousLinearMap.comp_apply, CtxVec.getCLM_apply,
      Tlin]

/-- Matrix multiplication node on 2D tensors. -/
def matmul {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    Node Γ (.dim m (.dim p .scalar)) :=
  Node.ofVec (Γ := Γ) (τ := .dim m (.dim p .scalar))
    (f := fun xV =>
      let aT := ofVecT (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := ofVecT (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      toVecT (t := Spec.matMulSpec aT bT))
    (jvp := fun xV dxV =>
      let aT := ofVecT (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := ofVecT (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let daT := ofVecT (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A dxV)
      let dbT := ofVecT (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B dxV)
      toVecT (t := addSpec (Spec.matMulSpec daT bT) (Spec.matMulSpec aT dbT)))
    (vjp := fun xV δV =>
      let aT := ofVecT (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := ofVecT (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let δT := ofVecT (s := .dim m (.dim p .scalar)) δV
      let dA := Spec.matMulSpec δT (matrixTransposeSpec bT)
      let dB := Spec.matMulSpec (matrixTransposeSpec aT) δT
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVecT (t := dA)) +
        CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVecT (t := dB)))
    (correct_inner := by
      intro xV dxV δV
      classical
      -- abbreviate tensors
      let aT := ofVecT (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A xV)
      let bT := ofVecT (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B xV)
      let daT := ofVecT (s := .dim m (.dim n .scalar)) (CtxVec.get (Γ := Γ) (s := .dim m (.dim n
        .scalar)) A dxV)
      let dbT := ofVecT (s := .dim n (.dim p .scalar)) (CtxVec.get (Γ := Γ) (s := .dim n (.dim p
        .scalar)) B dxV)
      let δT := ofVecT (s := .dim m (.dim p .scalar)) δV
      let dC := addSpec (Spec.matMulSpec daT bT) (Spec.matMulSpec aT dbT)
      let dA := Spec.matMulSpec δT (matrixTransposeSpec bT)
      let dB := Spec.matMulSpec (matrixTransposeSpec aT) δT

      -- LHS: rewrite `inner` into tensor `dot` using vectorization.
      have hL : inner ℝ (toVecT (t := dC)) δV = dot dC δT := by
        simp [dot_eq_inner_toVecT, δT, toVecT_ofVecT]

      -- RHS: split the context inner into the two single-slot contributions.
      have hR :
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVecT (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVecT (t := dB)))
            =
          dot daT dA + dot dbT dB := by
        have hA' :
            inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVecT (t := dA)))
              =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV) (toVecT (t := dA)) :=
              by
          simpa using
            (CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV (toVecT (t :=
              dA)))
        have hB' :
            inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVecT (t := dB)))
              =
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV) (toVecT (t := dB)) :=
              by
          simpa using
            (CtxVec.inner_get_single (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV (toVecT (t :=
              dB)))
        have hdotA :
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV) (toVecT (t := dA)) =
              dot daT dA := by
          simp [dot_eq_inner_toVecT, daT, toVecT_ofVecT]
        have hdotB :
            inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV) (toVecT (t := dB)) =
              dot dbT dB := by
          simp [dot_eq_inner_toVecT, dbT, toVecT_ofVecT]
        calc
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVecT (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVecT (t := dB)))
              =
              inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVecT (t :=
                dA))) +
                inner ℝ dxV (CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVecT (t :=
                  dB))) := by
                simp [inner_add_right]
          _ =
              inner ℝ (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A dxV) (toVecT (t := dA))
                +
                inner ℝ (CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B dxV) (toVecT (t :=
                  dB)) := by
                simp [hA', hB']
          _ = dot daT dA + dot dbT dB := by
                simp [hdotA, hdotB]

      -- Algebra on dots: unfold `dC` and apply the two matmul adjointness lemmas.
      have hdotC : dot dC δT = dot daT dA + dot dbT dB := by
        have hadd :
            dot dC δT =
              dot (Spec.matMulSpec daT bT) δT + dot (Spec.matMulSpec aT dbT) δT := by
          simpa [dC] using
            (dot_add_left (a := Spec.matMulSpec daT bT) (b := Spec.matMulSpec aT dbT) (c := δT))
        have h1 : dot (Spec.matMulSpec daT bT) δT = dot daT dA := by
          simpa [dA] using (dot_mat_mul_right_adjoint (A := daT) (B := bT) (C := δT))
        have h2 : dot (Spec.matMulSpec aT dbT) δT = dot dbT dB := by
          simpa [dB] using (dot_mat_mul_left_adjoint (A := aT) (B := dbT) (C := δT))
        calc
          dot dC δT
              = dot (Spec.matMulSpec daT bT) δT + dot (Spec.matMulSpec aT dbT) δT := hadd
          _ = dot daT dA + dot dbT dB := by simp [h1, h2]

      -- combine
      calc
        inner ℝ (toVecT (t := dC)) δV = dot dC δT := hL
        _ = dot daT dA + dot dbT dB := hdotC
        _ =
            inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) A (toVecT (t := dA)) +
                CtxVec.single (Γ := Γ) (s := .dim n (.dim p .scalar)) B (toVecT (t := dB))) :=
                  hR.symm
      )

set_option maxHeartbeats 5000000 in
/--
`NodeFDerivCorrect` for the matrix-matrix multiplication node.

This packages the product rule and the dot/adjointness lemmas for `Spec.mat_mul_spec`.
-/
def matmulFderiv {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    NodeFDerivCorrect (matmul (Γ := Γ) (m := m) (n := n) (p := p) A B) :=
by
  classical
  let fA : CtxVec Γ → Vec (Matmul.matSize m n) :=
    fun x => CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) A x
  let fB : CtxVec Γ → Vec (Matmul.matSize n p) :=
    fun x => CtxVec.get (Γ := Γ) (s := .dim n (.dim p .scalar)) B x
  let Bmul : Vec (Matmul.matSize m n) →L[ℝ] Vec (Matmul.matSize n p) →L[ℝ] Vec (Matmul.matSize m p)
    :=
    Matmul.matmulBilin (m := m) (n := n) (p := p)

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
        (Node.forwardVec (Γ := Γ) (τ := .dim m (.dim p .scalar)) (matmul (Γ := Γ) (m := m) (n := n)
          (p := p) A B))
          =
        (fun xV : CtxVec Γ => (Bmul (fA xV)) (fB xV)) := by
      funext ctxV
      simp [matmul, Node.forwardVec_ofVec, fA, fB, Bmul, Matmul.forward_eq_matmulVec]
    exact hbilin.congr_of_eventuallyEq hEq.eventuallyEq

  · intro xV dxV
    -- Rewrite the node JVP into the bilinear derivative formula.
    -- We use that `toVecT` respects matrix addition and that `toVecT (mat_mul_spec (ofVecT a)
    -- (ofVecT b))`
    -- is exactly `Matmul.matmulVec a b`.
    ext ip
    -- After expanding, the two bilinear terms may appear in the opposite order.
    simp [matmul, Node.jvpVec_ofVec, fA, fB, Bmul, Matmul.toVecT_add_spec_mat,
      Matmul.forward_eq_matmulVec, ContinuousLinearMap.add_apply, ContinuousLinearMap.comp_apply,
      CtxVec.getCLM_apply, add_comm]

-- ---------------------------------------------------------------------------
-- Matrix broadcasts and row-wise reductions (linear)
-- ---------------------------------------------------------------------------

namespace MatrixLinear

open Matmul

open scoped BigOperators

/-- Broadcast a vector `v : Vec m` across the last axis to a flattened `(m×n)` matrix. -/
def broadcastRowCLM {m n : Nat} : Vec m →L[ℝ] Vec (matSize m n) := by
  classical
  let fLin : Vec m →ₗ[ℝ] Vec (matSize m n) :=
    { toFun := fun v => vecOfFun (n := matSize m n) fun ip => v (ip.divNat (m := m) (n := vecSize
      n))
      map_add' := by
        intro v w
        ext ip
        simp [vecOfFun]
      map_smul' := by
        intro a v
        ext ip
        simp [vecOfFun, smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Broadcast a vector `v : Vec n` across the first axis to a flattened `(m×n)` matrix. -/
def broadcastColCLM {m n : Nat} : Vec n →L[ℝ] Vec (matSize m n) := by
  classical
  let hn : vecSize n = n := by simp [vecSize, Shape.size]
  let fLin : Vec n →ₗ[ℝ] Vec (matSize m n) :=
    { toFun := fun v =>
        let v' : Vec (vecSize n) := castVec hn.symm v
        vecOfFun (n := matSize m n) fun ip => v' (ip.modNat (m := m) (n := vecSize n))
      map_add' := by
        intro v w
        ext ip
        simp [vecOfFun]
      map_smul' := by
        intro a v
        ext ip
        simp [vecOfFun, smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Row-wise sum: flattened `(m×n)` matrix → vector `m`. -/
def rowSumCLM {m n : Nat} : Vec (matSize m n) →L[ℝ] Vec m := by
  classical
  let fLin : Vec (matSize m n) →ₗ[ℝ] Vec m :=
    { toFun := fun x => vecOfFun (n := m) fun i => ∑ j : Fin (vecSize n), x (finProdFinEquiv (i, j))
      map_add' := by
        intro x y
        ext i
        simp [vecOfFun, Finset.sum_add_distrib]
      map_smul' := by
        intro a x
        ext i
        simp [vecOfFun, Finset.mul_sum, smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Row-wise mean: flattened `(m×n)` matrix → vector `m`. -/
def rowMeanCLM {m n : Nat} : Vec (matSize m n) →L[ℝ] Vec m :=
  ((1 : ℝ) / (n : ℝ)) • rowSumCLM (m := m) (n := n)

end MatrixLinear

/-- Broadcast a vector `(.dim m .scalar)` across columns to `(.dim m (.dim n .scalar))`. -/
def broadcastRow {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m .scalar)) : Node Γ (.dim m (.dim n .scalar)) :=
  Node.ofVec (Γ := Γ) (τ := .dim m (.dim n .scalar))
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

/-- `NodeFDerivCorrect` for `broadcast_row` (linear op). -/
def broadcastRowFderiv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m .scalar)) :
    NodeFDerivCorrect (broadcastRow (Γ := Γ) (m := m) (n := n) idx) :=
by
  classical
  refine
    { deriv := fun _ =>
        (MatrixLinear.broadcastRowCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := m) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Matmul.matSize m n) :=
      (MatrixLinear.broadcastRowCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := m) idx)
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := .dim m (.dim n .scalar))
            (broadcastRow (Γ := Γ) (m := m) (n := n) idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [broadcastRow, D, Node.forwardVec_ofVec, getVecCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    simp [broadcastRow, Node.jvpVec_ofVec, getVecCLM_apply, ContinuousLinearMap.comp_apply]

/-- Broadcast a vector `(.dim n .scalar)` across rows to `(.dim m (.dim n .scalar))`. -/
def broadcastCol {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim n .scalar)) : Node Γ (.dim m (.dim n .scalar)) :=
  Node.ofVec (Γ := Γ) (τ := .dim m (.dim n .scalar))
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

/-- `NodeFDerivCorrect` for `broadcast_col` (linear op). -/
def broadcastColFderiv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim n .scalar)) :
    NodeFDerivCorrect (broadcastCol (Γ := Γ) (m := m) (n := n) idx) :=
by
  classical
  refine
    { deriv := fun _ =>
        (MatrixLinear.broadcastColCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := n) idx)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Matmul.matSize m n) :=
      (MatrixLinear.broadcastColCLM (m := m) (n := n)).comp (getVecCLM (Γ := Γ) (n := n) idx)
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := .dim m (.dim n .scalar))
            (broadcastCol (Γ := Γ) (m := m) (n := n) idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [broadcastCol, D, Node.forwardVec_ofVec, getVecCLM_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext ip
    simp [broadcastCol, Node.jvpVec_ofVec, getVecCLM_apply, ContinuousLinearMap.comp_apply]

-- ---------------------------------------------------------------------------
-- Shape-preserving reshapes (vector reinterpretation)
-- ---------------------------------------------------------------------------

/-!
Shape-only nodes (`reshape`, `flatten`, and similar) live in `NN.Proofs.Autograd.Tape.Nodes.Shape`
(namespace `TapeNodes.ShapeOps`).
-/

/-- Row-wise mean (reduce last axis): `(.dim m (.dim n .scalar)) → (.dim m .scalar)`. -/
def rowMean {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) : Node Γ (.dim m .scalar) :=
  let outShape : Shape := .dim m .scalar
  let hsz : Shape.size outShape = m := by simp [outShape, Shape.size]
  Node.ofVec (Γ := Γ) (τ := outShape)
    (f := fun xV =>
      castVec hsz.symm <|
        MatrixLinear.rowMeanCLM (m := m) (n := n)
          (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx xV))
    (jvp := fun _xV dxV =>
      castVec hsz.symm <|
        MatrixLinear.rowMeanCLM (m := m) (n := n)
          (CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV))
    (vjp := fun _xV δV =>
      let δm : Vec m := castVec hsz δV
      CtxVec.single (Γ := Γ) (s := .dim m (.dim n .scalar)) idx
        ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm))
    (correct_inner := by
      intro _xV dxV δV
      classical
      let dxMat : Vec (Matmul.matSize m n) :=
        CtxVec.get (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV
      let δm : Vec m := castVec hsz δV
      have hctx :=
        (CtxVec.inner_get_single (Γ := Γ) (s := .dim m (.dim n .scalar)) idx dxV
          ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm))
      have hLcast :
          inner ℝ (castVec hsz.symm (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat)) δV =
            inner ℝ (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) δm := by
        have hδ : castVec hsz.symm (castVec hsz δV) = δV := by
          simp
        calc
          inner ℝ (castVec hsz.symm (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat)) δV
              =
              inner ℝ (castVec hsz.symm (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat))
                (castVec hsz.symm (castVec hsz δV)) := by
                  simp [hδ]
          _ = inner ℝ (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) (castVec hsz δV) := by
                simpa using
                  (inner_castVec_castVec (h := hsz.symm)
                    (x := MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) (y := castVec hsz δV))
      have hadj :
          inner ℝ (MatrixLinear.rowMeanCLM (m := m) (n := n) dxMat) δm =
            inner ℝ dxMat ((MatrixLinear.rowMeanCLM (m := m) (n := n)).adjoint δm) := by
        simpa [δm] using
          (ContinuousLinearMap.adjoint_inner_right (A := MatrixLinear.rowMeanCLM (m := m) (n := n))
            (x := dxMat) (y := δm)).symm
      -- combine
      exact (hLcast.trans hadj).trans hctx.symm)

/-- `NodeFDerivCorrect` for `row_mean` (reduce-mean along the last axis). -/
def rowMeanFderiv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (rowMean (Γ := Γ) (m := m) (n := n) idx) :=
by
  classical
  let outShape : Shape := .dim m .scalar
  let hsz : Shape.size outShape = m := by simp [outShape, Shape.size]
  refine
    { deriv := fun _ =>
        (Graph.castCLM (h := hsz.symm)).comp
          ((MatrixLinear.rowMeanCLM (m := m) (n := n)).comp
            (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) idx))
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    let D : CtxVec Γ →L[ℝ] Vec (Shape.size outShape) :=
      (Graph.castCLM (h := hsz.symm)).comp
        ((MatrixLinear.rowMeanCLM (m := m) (n := n)).comp
          (CtxVec.getCLM (Γ := Γ) (s := .dim m (.dim n .scalar)) idx))
    have hD : HasFDerivAt (fun x : CtxVec Γ => D x) D xV := D.hasFDerivAt (x := xV)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := outShape)
            (rowMean (Γ := Γ) (m := m) (n := n) idx))
          =
        fun x : CtxVec Γ => D x := by
      funext x
      simp [rowMean, outShape, D, Node.forwardVec_ofVec, CtxVec.getCLM_apply, Graph.castCLM,
        ContinuousLinearMap.comp_apply]
    exact hD.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    ext i
    simp [rowMean, outShape, Node.jvpVec_ofVec, Graph.castCLM, ContinuousLinearMap.comp_apply,
      CtxVec.getCLM_apply]


end TapeNodes

end

end Autograd
end Proofs
