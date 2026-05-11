/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.Matrix

/-!
# Last-axis softmax and log-softmax tape nodes

Softmax and log-softmax over matrix rows, together with their Jacobian/VJP lemmas and
`NodeFDerivCorrect` wrappers for graph-level autograd proofs.
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
-- Axis softmax on matrices (row-wise; last axis)
-- ---------------------------------------------------------------------------

namespace SoftmaxLastAxis

open scoped BigOperators

/-- Flattened size for an `m×n` matrix when viewed as a single vector (`m*n`). -/
abbrev MNSize (m n : Nat) : Nat := m * n

/-- Split a flattened `m*n` vector into `m` rows of length `n`. -/
def rows {m n : Nat} (x : Vec (MNSize m n)) : Fin m → Vec n :=
  fun i => vecOfFun (n := n) fun j => x (finProdFinEquiv (i, j))

/-- Inverse of `rows`: assemble `m` rows back into a flattened `m*n` vector. -/
def unrows {m n : Nat} (r : Fin m → Vec n) : Vec (MNSize m n) :=
  vecOfFun (n := MNSize m n) fun ip =>
    let p : Fin m × Fin n := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)).symm ip
    r p.1 p.2

@[simp] lemma divNat_finProdFinEquiv {m n : Nat} (p : Fin m × Fin n) :
    (finProdFinEquiv (m := m) (n := n) p).divNat = p.1 := by
  have h := congrArg Prod.fst
    ((finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)).left_inv p)
  -- `finProdFinEquiv.symm` is `(divNat, modNat)`.
  simpa [finProdFinEquiv] using h

@[simp] lemma modNat_finProdFinEquiv {m n : Nat} (p : Fin m × Fin n) :
    (finProdFinEquiv (m := m) (n := n) p).modNat = p.2 := by
  have h := congrArg Prod.snd
    ((finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)).left_inv p)
  simpa [finProdFinEquiv] using h

/-- Continuous-linear-map version of `rows`. -/
def rowsCLM {m n : Nat} : Vec (MNSize m n) →L[ℝ] (Fin m → Vec n) := by
  classical
  let fLin : Vec (MNSize m n) →ₗ[ℝ] (Fin m → Vec n) :=
    { toFun := rows (m := m) (n := n)
      map_add' := by
        intro x y
        funext i
        ext j
        simp [rows, vecOfFun, Pi.add_apply]
      map_smul' := by
        intro a x
        funext i
        ext j
        simp [rows, vecOfFun, Pi.smul_apply, smul_eq_mul] }
  refine { toLinearMap := fLin, cont := ?_ }
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

/-- Continuous-linear-map version of `unrows`. -/
def unrowsCLM {m n : Nat} : (Fin m → Vec n) →L[ℝ] Vec (MNSize m n) := by
  classical
  let fLin : (Fin m → Vec n) →ₗ[ℝ] Vec (MNSize m n) :=
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

/-- Apply `softmaxVec` independently to each row of an `m×n` matrix (flattened representation). -/
def forwardMN {m n : Nat} (x : Vec (MNSize m n)) : Vec (MNSize m n) :=
  unrows (m := m) (n := n) (fun i => softmaxVec (n := n) (rows (m := m) (n := n) x i))

/-- JVP of `forwardMN`, computed rowwise. -/
def jvpMN {m n : Nat} (x dx : Vec (MNSize m n)) : Vec (MNSize m n) :=
  unrows (m := m) (n := n) (fun i =>
    softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) dx i))

/-- Derivative of `forwardMN` as a continuous linear map. -/
def derivMN {m n : Nat} (x : Vec (MNSize m n)) : Vec (MNSize m n) →L[ℝ] Vec (MNSize m n) :=
  let r : Fin m → Vec n := rows (m := m) (n := n) x
  let DG : (Fin m → Vec n) →L[ℝ] (Fin m → Vec n) :=
    ContinuousLinearMap.pi (R := ℝ) (fun i : Fin m =>
      (softmaxDerivCLM (n := n) (r i)).comp (ContinuousLinearMap.proj (R := ℝ) i))
  (unrowsCLM (m := m) (n := n)).comp (DG.comp (rowsCLM (m := m) (n := n)))

/-- `HasFDerivAt` statement for `forwardMN`, using the rowwise softmax derivative. -/
theorem hasFDerivAt_forwardMN {m n : Nat} (x : Vec (MNSize m n)) :
    HasFDerivAt (forwardMN (m := m) (n := n)) (derivMN (m := m) (n := n) x) x := by
  classical
  -- middle map: apply `softmaxVec` independently to each row
  let G : (Fin m → Vec n) → (Fin m → Vec n) := fun r => fun i => softmaxVec (n := n) (r i)
  let r0 : Fin m → Vec n := rows (m := m) (n := n) x
  let DG : (Fin m → Vec n) →L[ℝ] (Fin m → Vec n) :=
    ContinuousLinearMap.pi (R := ℝ) (fun i : Fin m =>
      (softmaxDerivCLM (n := n) (r0 i)).comp (ContinuousLinearMap.proj (R := ℝ) i))

  have hG : HasFDerivAt G DG r0 := by
    refine (hasFDerivAt_pi (𝕜 := ℝ)
      (φ := fun i : Fin m => fun r : (Fin m → Vec n) => softmaxVec (n := n) (r i))
      (φ' := fun i : Fin m =>
        (softmaxDerivCLM (n := n) (r0 i)).comp (ContinuousLinearMap.proj (R := ℝ) i))
      (x := r0)).2 ?_
    intro i
    have hsoft : HasFDerivAt (softmaxVec (n := n)) (softmaxDerivCLM (n := n) (r0 i)) (r0 i) :=
      hasFDerivAt_softmaxVec (n := n) (r0 i)
    have happly :
        HasFDerivAt (fun r : (Fin m → Vec n) => r i) (ContinuousLinearMap.proj (R := ℝ) i) r0 := by
      simpa using ((ContinuousLinearMap.proj (R := ℝ) i).hasFDerivAt (x := r0))
    exact hsoft.comp r0 happly

  -- chain with the linear reshapes
  have hrows : HasFDerivAt (rowsCLM (m := m) (n := n)) (rowsCLM (m := m) (n := n)) x := by
    simpa using ((rowsCLM (m := m) (n := n)).hasFDerivAt (x := x))
  have hunrows :
      HasFDerivAt (unrowsCLM (m := m) (n := n)) (unrowsCLM (m := m) (n := n)) (G r0) := by
    simpa using ((unrowsCLM (m := m) (n := n)).hasFDerivAt (x := G r0))
  have hmid :
      HasFDerivAt (fun z : Vec (MNSize m n) => G ((rowsCLM (m := m) (n := n)) z))
        (DG.comp (rowsCLM (m := m) (n := n))) x := by
    exact hG.comp x hrows
  have hcomp := hunrows.comp x hmid
  -- rewrite to `forwardMN`/`derivMN`
  simpa [forwardMN, derivMN, G, r0, DG] using hcomp

/-- JVP computed by `jvpMN` agrees with applying the derivative `derivMN`. -/
theorem jvpMN_eq_derivMN {m n : Nat} (x dx : Vec (MNSize m n)) :
    jvpMN (m := m) (n := n) x dx = (derivMN (m := m) (n := n) x) dx := by
  classical
  ext ip
  -- decode `ip` into `(i,j)`
  let p : Fin m × Fin n := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)).symm ip
  -- unfold everything to rowwise application
  simp [jvpMN, derivMN, rowsCLM, rows, unrows, unrowsCLM, MNSize,
    ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply,
    softmaxJvp_eq_deriv]

/-- Symmetry property of the JVP under the inner product (rowwise `softmaxJvp` commutation). -/
theorem inner_jvpMN_comm {m n : Nat} (x dx δ : Vec (MNSize m n)) :
    inner ℝ (jvpMN (m := m) (n := n) x dx) δ =
      inner ℝ dx (jvpMN (m := m) (n := n) x δ) := by
  classical
  -- Expand to sums over `(i,j)` and apply the vector lemma rowwise.
  have hL :
      inner ℝ (jvpMN (m := m) (n := n) x dx) δ
        =
      ∑ i : Fin m,
        inner ℝ (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) dx i))
          (rows (m := m) (n := n) δ i) := by
    -- `inner` on `Vec (m*n)` is a sum over `(i,j)`.
    have :
        inner ℝ (jvpMN (m := m) (n := n) x dx) δ
          =
        ∑ p : Fin m × Fin n,
          (softmaxJvp (n := n) (rows (m := m) (n := n) x p.1) (rows (m := m) (n := n) dx p.1) p.2) *
            δ (finProdFinEquiv p) := by
      -- Reindex the `Fin (m*n)` sum by `finProdFinEquiv`, then unfold `unrows` at `finProdFinEquiv
      -- p`.
      let g : Fin (m * n) → ℝ := fun ip =>
        (jvpMN (m := m) (n := n) x dx ip) * δ ip
      have hsum :
          (∑ ip : Fin (m * n), g ip) = ∑ p : Fin m × Fin n, g (finProdFinEquiv p) := by
        simpa [g] using
          (Equiv.sum_comp (e := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n))) (g := g)).symm
      -- rewrite the inner product to `∑ ip, g ip`, apply the reindexing lemma, then unfold `g`.
      calc
        inner ℝ (jvpMN (m := m) (n := n) x dx) δ
            = ∑ ip : Fin (m * n), g ip := by
                simp [inner_eq_sum_mul, g]
        _ = ∑ p : Fin m × Fin n, g (finProdFinEquiv p) := hsum
        _ = ∑ p : Fin m × Fin n,
              (softmaxJvp (n := n)
                    (rows (m := m) (n := n) x p.1)
                    (rows (m := m) (n := n) dx p.1) p.2) * δ (finProdFinEquiv p) := by
                classical
                refine Fintype.sum_congr _ _ ?_
                intro p
                simp [g, jvpMN, unrows, rows, MNSize, vecOfFun, mul_comm]
    -- split the product sum into a double sum
    -- and recognize each inner product on `Vec n`
    calc
      inner ℝ (jvpMN (m := m) (n := n) x dx) δ
          = ∑ p : Fin m × Fin n,
              (softmaxJvp (n := n) (rows (m := m) (n := n) x p.1) (rows (m := m) (n := n) dx p.1)
                p.2) *
                δ (finProdFinEquiv p) := this
      _ = ∑ i : Fin m, ∑ j : Fin n,
              (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) dx i) j) *
                δ (finProdFinEquiv (i, j)) := by
            simp [Fintype.sum_prod_type]
      _ = ∑ i : Fin m,
            inner ℝ (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) dx i))
              (rows (m := m) (n := n) δ i) := by
            refine Finset.sum_congr rfl ?_
            intro i _hi
            -- `inner` expands to `∑ j, ...`
            simp [inner_eq_sum_mul, rows]
  have hR :
      inner ℝ dx (jvpMN (m := m) (n := n) x δ)
        =
      ∑ i : Fin m,
        inner ℝ (rows (m := m) (n := n) dx i)
          (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) δ i)) := by
    have :
        inner ℝ dx (jvpMN (m := m) (n := n) x δ)
          =
        ∑ p : Fin m × Fin n,
          dx (finProdFinEquiv p) *
            (softmaxJvp (n := n) (rows (m := m) (n := n) x p.1) (rows (m := m) (n := n) δ p.1) p.2)
              := by
      let g : Fin (m * n) → ℝ := fun ip =>
        dx ip * (jvpMN (m := m) (n := n) x δ ip)
      have hsum :
          (∑ ip : Fin (m * n), g ip) = ∑ p : Fin m × Fin n, g (finProdFinEquiv p) := by
        simpa [g] using
          (Equiv.sum_comp (e := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n))) (g := g)).symm
      calc
        inner ℝ dx (jvpMN (m := m) (n := n) x δ)
            = ∑ ip : Fin (m * n), g ip := by
                simp [inner_eq_sum_mul, g]
        _ = ∑ p : Fin m × Fin n, g (finProdFinEquiv p) := hsum
        _ = ∑ p : Fin m × Fin n,
              dx (finProdFinEquiv p) *
                (softmaxJvp (n := n)
                    (rows (m := m) (n := n) x p.1)
                    (rows (m := m) (n := n) δ p.1) p.2) := by
                classical
                refine Fintype.sum_congr _ _ ?_
                intro p
                simp [g, jvpMN, unrows, rows, MNSize, vecOfFun, mul_comm]
    calc
      inner ℝ dx (jvpMN (m := m) (n := n) x δ)
          = ∑ p : Fin m × Fin n,
              dx (finProdFinEquiv p) *
                (softmaxJvp (n := n) (rows (m := m) (n := n) x p.1) (rows (m := m) (n := n) δ p.1)
                  p.2) := this
      _ = ∑ i : Fin m, ∑ j : Fin n,
              dx (finProdFinEquiv (i, j)) *
                (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) δ i) j) :=
                  by
            simp [Fintype.sum_prod_type]
      _ = ∑ i : Fin m,
            inner ℝ (rows (m := m) (n := n) dx i)
              (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) δ i)) := by
            refine Finset.sum_congr rfl ?_
            intro i _hi
            simp [inner_eq_sum_mul, rows, mul_comm]
  -- finish by applying the vector lemma per row
  have hrow :
      ∀ i : Fin m,
        inner ℝ (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) dx i))
            (rows (m := m) (n := n) δ i)
          =
        inner ℝ (rows (m := m) (n := n) dx i)
            (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) δ i)) := by
    intro i
    simpa using
      inner_softmaxJvp_comm (n := n)
        (x := rows (m := m) (n := n) x i)
        (dx := rows (m := m) (n := n) dx i)
        (δ := rows (m := m) (n := n) δ i)
  calc
    inner ℝ (jvpMN (m := m) (n := n) x dx) δ
        = ∑ i : Fin m,
            inner ℝ (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) dx i))
              (rows (m := m) (n := n) δ i) := hL
    _ = ∑ i : Fin m,
            inner ℝ (rows (m := m) (n := n) dx i)
              (softmaxJvp (n := n) (rows (m := m) (n := n) x i) (rows (m := m) (n := n) δ i)) := by
          refine Finset.sum_congr rfl ?_
          intro i _hi
          exact hrow i
    _ = inner ℝ dx (jvpMN (m := m) (n := n) x δ) := hR.symm

end SoftmaxLastAxis

namespace LogSoftmaxLastAxis

open scoped BigOperators

/-- Reuse the `m*n` flattened size from `SoftmaxLastAxis`. -/
abbrev MNSize (m n : Nat) : Nat := SoftmaxLastAxis.MNSize m n

/-- Apply `logSoftmaxVec` independently to each row (flattened representation). -/
def forwardMN {m n : Nat} (x : Vec (MNSize m n)) : Vec (MNSize m n) :=
  SoftmaxLastAxis.unrows (m := m) (n := n) fun i =>
    logSoftmaxVec (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)

/-- JVP of `forwardMN`, computed rowwise via `logSoftmaxJvp`. -/
def jvpMN {m n : Nat} (x dx : Vec (MNSize m n)) : Vec (MNSize m n) :=
  SoftmaxLastAxis.unrows (m := m) (n := n) fun i =>
    logSoftmaxJvp (n := n)
      (SoftmaxLastAxis.rows (m := m) (n := n) x i)
      (SoftmaxLastAxis.rows (m := m) (n := n) dx i)

/-- VJP of `forwardMN`, computed rowwise via `logSoftmaxVjp`. -/
def vjpMN {m n : Nat} (x δ : Vec (MNSize m n)) : Vec (MNSize m n) :=
  SoftmaxLastAxis.unrows (m := m) (n := n) fun i =>
    logSoftmaxVjp (n := n)
      (SoftmaxLastAxis.rows (m := m) (n := n) x i)
      (SoftmaxLastAxis.rows (m := m) (n := n) δ i)

/-- Derivative of `forwardMN` as a continuous linear map. -/
def derivMN {m n : Nat} (x : Vec (MNSize m n)) : Vec (MNSize m n) →L[ℝ] Vec (MNSize m n) :=
  let r : Fin m → Vec n := SoftmaxLastAxis.rows (m := m) (n := n) x
  let DG : (Fin m → Vec n) →L[ℝ] (Fin m → Vec n) :=
    ContinuousLinearMap.pi (R := ℝ) fun i : Fin m =>
      (logSoftmaxDerivCLM (n := n) (r i)).comp (ContinuousLinearMap.proj (R := ℝ) i)
  (SoftmaxLastAxis.unrowsCLM (m := m) (n := n)).comp (DG.comp (SoftmaxLastAxis.rowsCLM (m := m) (n
    := n)))

/-- `HasFDerivAt` statement for `logSoftmaxVec` applied rowwise. -/
theorem hasFDerivAt_forwardMN {m n : Nat} (x : Vec (MNSize m n)) :
    HasFDerivAt (forwardMN (m := m) (n := n)) (derivMN (m := m) (n := n) x) x := by
  classical
  let G : (Fin m → Vec n) → (Fin m → Vec n) := fun r => fun i => logSoftmaxVec (n := n) (r i)
  let r0 : Fin m → Vec n := SoftmaxLastAxis.rows (m := m) (n := n) x
  let DG : (Fin m → Vec n) →L[ℝ] (Fin m → Vec n) :=
    ContinuousLinearMap.pi (R := ℝ) fun i : Fin m =>
      (logSoftmaxDerivCLM (n := n) (r0 i)).comp (ContinuousLinearMap.proj (R := ℝ) i)

  have hG : HasFDerivAt G DG r0 := by
    refine (hasFDerivAt_pi (𝕜 := ℝ)
      (φ := fun i : Fin m => fun r : (Fin m → Vec n) => logSoftmaxVec (n := n) (r i))
      (φ' := fun i : Fin m =>
        (logSoftmaxDerivCLM (n := n) (r0 i)).comp (ContinuousLinearMap.proj (R := ℝ) i))
      (x := r0)).2 ?_
    intro i
    have hlog :
        HasFDerivAt (logSoftmaxVec (n := n)) (logSoftmaxDerivCLM (n := n) (r0 i)) (r0 i) :=
      hasFDerivAt_logSoftmaxVec (n := n) (r0 i)
    have happly :
        HasFDerivAt (fun r : (Fin m → Vec n) => r i) (ContinuousLinearMap.proj (R := ℝ) i) r0 := by
      simpa using ((ContinuousLinearMap.proj (R := ℝ) i).hasFDerivAt (x := r0))
    exact hlog.comp r0 happly

  have hrows :
      HasFDerivAt (SoftmaxLastAxis.rowsCLM (m := m) (n := n)) (SoftmaxLastAxis.rowsCLM (m := m) (n
        := n)) x := by
    simpa using ((SoftmaxLastAxis.rowsCLM (m := m) (n := n)).hasFDerivAt (x := x))
  have hunrows :
      HasFDerivAt (SoftmaxLastAxis.unrowsCLM (m := m) (n := n)) (SoftmaxLastAxis.unrowsCLM (m := m)
        (n := n)) (G r0) := by
    simpa using ((SoftmaxLastAxis.unrowsCLM (m := m) (n := n)).hasFDerivAt (x := G r0))
  have hmid :
      HasFDerivAt (fun z : Vec (MNSize m n) => G ((SoftmaxLastAxis.rowsCLM (m := m) (n := n)) z))
        (DG.comp (SoftmaxLastAxis.rowsCLM (m := m) (n := n))) x := by
    exact hG.comp x hrows
  have hcomp := hunrows.comp x hmid
  simpa [forwardMN, derivMN, G, r0, DG] using hcomp

/-- JVP computed by `jvpMN` agrees with applying the derivative `derivMN`. -/
theorem jvpMN_eq_derivMN {m n : Nat} (x dx : Vec (MNSize m n)) :
    jvpMN (m := m) (n := n) x dx = (derivMN (m := m) (n := n) x) dx := by
  classical
  ext ip
  let p : Fin m × Fin n := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n)).symm ip
  simp [jvpMN, derivMN, SoftmaxLastAxis.rowsCLM, SoftmaxLastAxis.rows, SoftmaxLastAxis.unrows,
    SoftmaxLastAxis.unrowsCLM,
    MNSize, ContinuousLinearMap.comp_apply, ContinuousLinearMap.proj_apply,
    logSoftmaxJvp_eq_deriv, logSoftmaxDerivCLM]

/-- `logSoftmaxJvp` / `logSoftmaxVjp` adjointness under the inner product, lifted rowwise. -/
theorem inner_jvpMN_vjp {m n : Nat} (x dx δ : Vec (MNSize m n)) :
    inner ℝ (jvpMN (m := m) (n := n) x dx) δ =
      inner ℝ dx (vjpMN (m := m) (n := n) x δ) := by
  classical
  -- Expand to sums over `(i,j)` and apply the vector lemma rowwise.
  have hL :
      inner ℝ (jvpMN (m := m) (n := n) x dx) δ
        =
      ∑ i : Fin m,
        inner ℝ
          (logSoftmaxJvp (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)
            (SoftmaxLastAxis.rows (m := m) (n := n) dx i))
          (SoftmaxLastAxis.rows (m := m) (n := n) δ i) := by
    have :
        inner ℝ (jvpMN (m := m) (n := n) x dx) δ
          =
        ∑ p : Fin m × Fin n,
          (logSoftmaxJvp (n := n)
                (SoftmaxLastAxis.rows (m := m) (n := n) x p.1)
                (SoftmaxLastAxis.rows (m := m) (n := n) dx p.1) p.2) *
            δ (finProdFinEquiv p) := by
      let g : Fin (m * n) → ℝ := fun ip =>
        (jvpMN (m := m) (n := n) x dx ip) * δ ip
      have hsum :
          (∑ ip : Fin (m * n), g ip) = ∑ p : Fin m × Fin n, g (finProdFinEquiv p) := by
        simpa [g] using
          (Equiv.sum_comp (e := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n))) (g := g)).symm
      calc
        inner ℝ (jvpMN (m := m) (n := n) x dx) δ
            = ∑ ip : Fin (m * n), g ip := by
                simp [inner_eq_sum_mul, g]
        _ = ∑ p : Fin m × Fin n, g (finProdFinEquiv p) := hsum
        _ = ∑ p : Fin m × Fin n,
              (logSoftmaxJvp (n := n)
                    (SoftmaxLastAxis.rows (m := m) (n := n) x p.1)
                    (SoftmaxLastAxis.rows (m := m) (n := n) dx p.1) p.2) * δ (finProdFinEquiv p) :=
                      by
                classical
                refine Fintype.sum_congr _ _ ?_
                intro p
                simp [g, jvpMN, SoftmaxLastAxis.unrows, SoftmaxLastAxis.rows, MNSize, vecOfFun,
                  mul_comm]
    calc
      inner ℝ (jvpMN (m := m) (n := n) x dx) δ = ∑ p : Fin m × Fin n,
          (logSoftmaxJvp (n := n)
                (SoftmaxLastAxis.rows (m := m) (n := n) x p.1)
                (SoftmaxLastAxis.rows (m := m) (n := n) dx p.1) p.2) *
            δ (finProdFinEquiv p) := this
      _ = ∑ i : Fin m, ∑ j : Fin n,
          (logSoftmaxJvp (n := n)
                (SoftmaxLastAxis.rows (m := m) (n := n) x i)
                (SoftmaxLastAxis.rows (m := m) (n := n) dx i) j) *
            δ (finProdFinEquiv (i, j)) := by
          simp [Fintype.sum_prod_type]
      _ = ∑ i : Fin m,
          inner ℝ
            (logSoftmaxJvp (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)
              (SoftmaxLastAxis.rows (m := m) (n := n) dx i))
            (SoftmaxLastAxis.rows (m := m) (n := n) δ i) := by
          refine Finset.sum_congr rfl ?_
          intro i _hi
          simp [inner_eq_sum_mul, SoftmaxLastAxis.rows]
  have hR :
      inner ℝ dx (vjpMN (m := m) (n := n) x δ)
        =
      ∑ i : Fin m,
        inner ℝ (SoftmaxLastAxis.rows (m := m) (n := n) dx i)
          (logSoftmaxVjp (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)
            (SoftmaxLastAxis.rows (m := m) (n := n) δ i)) := by
    have :
        inner ℝ dx (vjpMN (m := m) (n := n) x δ)
          =
        ∑ p : Fin m × Fin n,
          dx (finProdFinEquiv p) *
            (logSoftmaxVjp (n := n)
                (SoftmaxLastAxis.rows (m := m) (n := n) x p.1)
                (SoftmaxLastAxis.rows (m := m) (n := n) δ p.1) p.2) := by
      let g : Fin (m * n) → ℝ := fun ip =>
        dx ip * (vjpMN (m := m) (n := n) x δ ip)
      have hsum :
          (∑ ip : Fin (m * n), g ip) = ∑ p : Fin m × Fin n, g (finProdFinEquiv p) := by
        simpa [g] using
          (Equiv.sum_comp (e := (finProdFinEquiv : Fin m × Fin n ≃ Fin (m * n))) (g := g)).symm
      calc
        inner ℝ dx (vjpMN (m := m) (n := n) x δ)
            = ∑ ip : Fin (m * n), g ip := by
                simp [inner_eq_sum_mul, g]
        _ = ∑ p : Fin m × Fin n, g (finProdFinEquiv p) := hsum
        _ = ∑ p : Fin m × Fin n,
              dx (finProdFinEquiv p) *
                (logSoftmaxVjp (n := n)
                    (SoftmaxLastAxis.rows (m := m) (n := n) x p.1)
                    (SoftmaxLastAxis.rows (m := m) (n := n) δ p.1) p.2) := by
                classical
                refine Fintype.sum_congr _ _ ?_
                intro p
                simp [g, vjpMN, SoftmaxLastAxis.unrows, SoftmaxLastAxis.rows, MNSize, vecOfFun,
                  mul_comm]
    calc
      inner ℝ dx (vjpMN (m := m) (n := n) x δ)
          = ∑ p : Fin m × Fin n,
              dx (finProdFinEquiv p) *
                (logSoftmaxVjp (n := n)
                    (SoftmaxLastAxis.rows (m := m) (n := n) x p.1)
                    (SoftmaxLastAxis.rows (m := m) (n := n) δ p.1) p.2) := this
      _ = ∑ i : Fin m, ∑ j : Fin n,
              dx (finProdFinEquiv (i, j)) *
                (logSoftmaxVjp (n := n)
                    (SoftmaxLastAxis.rows (m := m) (n := n) x i)
                    (SoftmaxLastAxis.rows (m := m) (n := n) δ i) j) := by
            simp [Fintype.sum_prod_type]
      _ = ∑ i : Fin m,
            inner ℝ (SoftmaxLastAxis.rows (m := m) (n := n) dx i)
              (logSoftmaxVjp (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)
                (SoftmaxLastAxis.rows (m := m) (n := n) δ i)) := by
            refine Finset.sum_congr rfl ?_
            intro i _hi
            simp [inner_eq_sum_mul, SoftmaxLastAxis.rows, mul_comm]
  have hrow :
      ∀ i : Fin m,
        inner ℝ
            (logSoftmaxJvp (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)
              (SoftmaxLastAxis.rows (m := m) (n := n) dx i))
            (SoftmaxLastAxis.rows (m := m) (n := n) δ i)
          =
        inner ℝ (SoftmaxLastAxis.rows (m := m) (n := n) dx i)
            (logSoftmaxVjp (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)
              (SoftmaxLastAxis.rows (m := m) (n := n) δ i)) := by
    intro i
    simpa using
      inner_logSoftmaxJvp_vjp (n := n)
        (x := SoftmaxLastAxis.rows (m := m) (n := n) x i)
        (dx := SoftmaxLastAxis.rows (m := m) (n := n) dx i)
        (δ := SoftmaxLastAxis.rows (m := m) (n := n) δ i)
  calc
    inner ℝ (jvpMN (m := m) (n := n) x dx) δ
        = ∑ i : Fin m,
            inner ℝ
              (logSoftmaxJvp (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)
                (SoftmaxLastAxis.rows (m := m) (n := n) dx i))
              (SoftmaxLastAxis.rows (m := m) (n := n) δ i) := hL
    _ = ∑ i : Fin m,
            inner ℝ (SoftmaxLastAxis.rows (m := m) (n := n) dx i)
              (logSoftmaxVjp (n := n) (SoftmaxLastAxis.rows (m := m) (n := n) x i)
                (SoftmaxLastAxis.rows (m := m) (n := n) δ i)) := by
          refine Finset.sum_congr rfl ?_
          intro i _hi
          exact hrow i
    _ = inner ℝ dx (vjpMN (m := m) (n := n) x δ) := hR.symm

end LogSoftmaxLastAxis

/-- Tape node for applying `softmaxVec` along the last axis of an `m×n` matrix (rowwise). -/
def softmaxLast {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) : Node Γ (.dim m (.dim n .scalar)) :=
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun xV =>
      castVec hsz.symm
        (SoftmaxLastAxis.forwardMN (m := m) (n := n) (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx
          xV))))
    (jvp := fun xV dxV =>
      castVec hsz.symm
        (SoftmaxLastAxis.jvpMN (m := m) (n := n)
          (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
          (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))))
    (vjp := fun xV δV =>
      CtxVec.single (Γ := Γ) (s := s) idx
        (castVec hsz.symm
          (SoftmaxLastAxis.jvpMN (m := m) (n := n)
            (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
            (castVec hsz δV))))
    (correct_inner := by
      intro xV dxV δV
      -- reduce to the matrix-vector symmetry lemma on the sliced entry, then embed into the context
      have hsymm :=
        SoftmaxLastAxis.inner_jvpMN_comm (m := m) (n := n)
          (x := castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
          (dx := castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))
          (δ := castVec hsz δV)
      have hctx :=
        CtxVec.inner_get_single (Γ := Γ) (s := s) idx dxV
          (castVec hsz.symm
            (SoftmaxLastAxis.jvpMN (m := m) (n := n)
              (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
              (castVec hsz δV)))
      -- move casts across inner products
      have hLcast :
          inner ℝ
              (castVec hsz.symm
                (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))))
              δV
            =
          inner ℝ
              (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV)))
              (castVec hsz δV) := by
        -- Insert a cancelling cast on `δV`, then use the cast-isometry lemma.
        have hδ : castVec hsz.symm (castVec hsz δV) = δV := by
          simp
        calc
          inner ℝ
              (castVec hsz.symm
                (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))))
              δV
              =
            inner ℝ
              (castVec hsz.symm
                (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))))
              (castVec hsz.symm (castVec hsz δV)) := by
                simp [hδ]
          _ =
            inner ℝ
              (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV)))
              (castVec hsz δV) := by
                simpa using
                  (inner_castVec_castVec (h := hsz.symm)
                    (x := SoftmaxLastAxis.jvpMN (m := m) (n := n)
                      (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                      (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV)))
                    (y := castVec hsz δV))
      have hRcast :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dxV)
              (castVec hsz.symm
                (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                  (castVec hsz δV)))
            =
          inner ℝ (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))
              (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                (castVec hsz δV)) := by
        -- Cast both arguments to `m*n`, use isometry, and cancel the cast on the right.
        have hδ :
            castVec hsz
                (castVec hsz.symm
                  (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                    (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                    (castVec hsz δV)))
              =
            (SoftmaxLastAxis.jvpMN (m := m) (n := n)
              (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
              (castVec hsz δV)) := by
          simp
        -- move the cast to both sides of the inner product
        have hiso :=
          (inner_castVec_castVec (h := hsz)
            (x := CtxVec.get (Γ := Γ) (s := s) idx dxV)
            (y := castVec hsz.symm
              (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                (castVec hsz δV)))).symm
        -- simplify the RHS cast using `hδ`
        simpa [hδ] using hiso
      calc
        inner ℝ
            (castVec hsz.symm
              (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))))
            δV
            = inner ℝ
                (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV)))
                (castVec hsz δV) := hLcast
        _ = inner ℝ
              (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))
              (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                (castVec hsz δV)) := hsymm
        _ = inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dxV)
              (castVec hsz.symm
                (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                  (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                  (castVec hsz δV))) := hRcast.symm
        _ = inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) idx
                (castVec hsz.symm
                  (SoftmaxLastAxis.jvpMN (m := m) (n := n)
                    (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
                    (castVec hsz δV)))) := by
              simpa using hctx.symm )

/-- Tape node for applying `logSoftmaxVec` along the last axis of an `m×n` matrix (rowwise). -/
def logSoftmaxLast {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) : Node Γ (.dim m (.dim n .scalar)) :=
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
  Node.ofVec (Γ := Γ) (τ := s)
    (f := fun xV =>
      castVec hsz.symm
        (LogSoftmaxLastAxis.forwardMN (m := m) (n := n)
          (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))))
    (jvp := fun xV dxV =>
      castVec hsz.symm
        (LogSoftmaxLastAxis.jvpMN (m := m) (n := n)
          (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
          (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV))))
    (vjp := fun xV δV =>
      CtxVec.single (Γ := Γ) (s := s) idx
        (castVec hsz.symm
          (LogSoftmaxLastAxis.vjpMN (m := m) (n := n)
            (castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV))
            (castVec hsz δV))))
    (correct_inner := by
      intro xV dxV δV
      classical
      -- Move casts across `inner` and reduce to the flattened lemma.
      let xMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV)
      let dxMN : Vec (m * n) := castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx dxV)
      let δMN : Vec (m * n) := castVec hsz δV
      have hLcast :
          inner ℝ (castVec hsz.symm (LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN)) δV
            =
          inner ℝ (LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN) δMN := by
        simpa [δMN] using
          (inner_castVec_castVec (h := hsz.symm)
            (x := LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN) (y := δMN))
      have hRcast :
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dxV) (castVec hsz.symm (LogSoftmaxLastAxis.vjpMN
            (m := m) (n := n) xMN δMN))
            =
          inner ℝ dxMN (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN) := by
        -- Cast both arguments to `m*n`, use isometry, and cancel the cast on the right.
        have hδ :
            castVec hsz (castVec hsz.symm (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN))
              =
            LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN := by
          simp
        have hiso :=
          (inner_castVec_castVec (h := hsz)
            (x := CtxVec.get (Γ := Γ) (s := s) idx dxV)
            (y := castVec hsz.symm (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN))).symm
        simpa [dxMN, hδ] using hiso
      have hctx :
          inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) idx (castVec hsz.symm (LogSoftmaxLastAxis.vjpMN (m :=
                m) (n := n) xMN δMN)))
            =
          inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dxV) (castVec hsz.symm (LogSoftmaxLastAxis.vjpMN
            (m := m) (n := n) xMN δMN)) := by
        simpa using
          (CtxVec.inner_get_single (Γ := Γ) (s := s) idx dxV
            (castVec hsz.symm (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN)))
      have hflat :
          inner ℝ (LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN) δMN
            =
          inner ℝ dxMN (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN) :=
        LogSoftmaxLastAxis.inner_jvpMN_vjp (m := m) (n := n) xMN dxMN δMN
      -- Now chain the cast rewrites + the flattened inner-product lemma.
      calc
        inner ℝ (castVec hsz.symm (LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN)) δV
            = inner ℝ (LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN) δMN := hLcast
        _ = inner ℝ dxMN (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN) := hflat
        _ = inner ℝ (CtxVec.get (Γ := Γ) (s := s) idx dxV)
              (castVec hsz.symm (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN)) := hRcast.symm
        _ = inner ℝ dxV
              (CtxVec.single (Γ := Γ) (s := s) idx
                (castVec hsz.symm (LogSoftmaxLastAxis.vjpMN (m := m) (n := n) xMN δMN))) :=
                  hctx.symm )

/-- `NodeFDerivCorrect` for `softmax_last` (rowwise softmax). -/
def softmaxLastFderiv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (softmaxLast (Γ := Γ) (m := m) (n := n) idx) := by
  classical
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
  let getMN : CtxVec Γ → Vec (m * n) :=
    fun xV => castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV)
  let getMNCLM : CtxVec Γ →L[ℝ] Vec (m * n) :=
    (Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)
  let outCast : Vec (m * n) →L[ℝ] Vec (Shape.size s) := Graph.castCLM (h := hsz.symm)
  refine
    { deriv := fun xV =>
        outCast.comp ((SoftmaxLastAxis.derivMN (m := m) (n := n) (getMN xV)).comp getMNCLM)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hget : HasFDerivAt getMN getMNCLM xV := by
      have h := getMNCLM.hasFDerivAt (x := xV)
      have hfun : getMN = fun x => getMNCLM x := by
        funext x
        simp [getMN, getMNCLM, CtxVec.getCLM_apply, Graph.castCLM]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq
    have hsoft : HasFDerivAt (SoftmaxLastAxis.forwardMN (m := m) (n := n))
        (SoftmaxLastAxis.derivMN (m := m) (n := n) (getMN xV)) (getMN xV) :=
      SoftmaxLastAxis.hasFDerivAt_forwardMN (m := m) (n := n) (getMN xV)
    have hout : HasFDerivAt (fun z : Vec (m * n) => outCast z) outCast (SoftmaxLastAxis.forwardMN (m
      := m) (n := n) (getMN xV)) :=
      outCast.hasFDerivAt (x := SoftmaxLastAxis.forwardMN (m := m) (n := n) (getMN xV))
    have hcomp := hout.comp xV (hsoft.comp xV hget)
    -- rewrite the forwardVec of the node to this composition
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := s) (softmaxLast (Γ := Γ) (m := m) (n := n) idx))
          =
        (fun xV : CtxVec Γ =>
          outCast (SoftmaxLastAxis.forwardMN (m := m) (n := n) (getMN xV))) := by
      funext xV
      ext i
      -- Do not unfold `castVec`: it expands to a private `vecOfFun` helper, and we want `simp` to
      -- use the public lemmas `castVec_apply`/`castVec_rfl` instead.
      simp [softmaxLast, Node.forwardVec_ofVec, getMN, outCast, Graph.castCLM]
    exact hcomp.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    -- Avoid unfolding `derivMN` coordinatewise: reduce to `jvpMN_eq_derivMN` plus cast/CLM
    -- simplifications.
    let xMN : Vec (m * n) := getMN xV
    let dxMN : Vec (m * n) := getMN dxV
    have hget : getMNCLM dxV = dxMN := by
      simp [getMNCLM, getMN, dxMN, CtxVec.getCLM_apply, Graph.castCLM]
    have hjvp :
        SoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN =
          (SoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN :=
      SoftmaxLastAxis.jvpMN_eq_derivMN (m := m) (n := n) xMN dxMN
    -- now unfold the node JVP and the composite CLM application, then rewrite via `hget`/`hjvp`.
    ext ip
    -- LHS: the node's JVP
    have hL :
        (Node.jvpVec (Γ := Γ) (τ := s) (softmaxLast (Γ := Γ) (m := m) (n := n) idx) xV dxV) ip
          =
        (castVec hsz.symm (SoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN)) ip := by
      simp [softmaxLast, Node.jvpVec_ofVec, getMN, xMN, dxMN]
    -- RHS: the derivative CLM applied to `dxV`
    have hR :
        ((outCast.comp ((SoftmaxLastAxis.derivMN (m := m) (n := n) xMN).comp getMNCLM)) dxV) ip
          =
        (castVec hsz.symm ((SoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN)) ip := by
      -- unfold the CLM compositions and cancel `getMNCLM dxV` to `dxMN`
      simp [outCast, Graph.castCLM, ContinuousLinearMap.comp_apply, hget]
    -- finish
    have hjvp_ip :
        (castVec hsz.symm (SoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN)) ip
          =
        (castVec hsz.symm ((SoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN)) ip :=
      congrArg (fun v => v ip) (congrArg (castVec hsz.symm) hjvp)
    calc
      (Node.jvpVec (Γ := Γ) (τ := s) (softmaxLast (Γ := Γ) (m := m) (n := n) idx) xV dxV) ip
          =
        (castVec hsz.symm (SoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN)) ip := hL
      _ =
        (castVec hsz.symm ((SoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN)) ip := hjvp_ip
      _ =
        ((outCast.comp ((SoftmaxLastAxis.derivMN (m := m) (n := n) xMN).comp getMNCLM)) dxV) ip :=
          hR.symm

/-- `NodeFDerivCorrect` for `log_softmax_last` (rowwise log-softmax). -/
def logSoftmaxLastFderiv {Γ : List Shape} {m n : Nat}
    (idx : Idx Γ (.dim m (.dim n .scalar))) :
    NodeFDerivCorrect (logSoftmaxLast (Γ := Γ) (m := m) (n := n) idx) := by
  classical
  let s : Shape := .dim m (.dim n .scalar)
  let hsz : Shape.size s = m * n := by simp [s, Shape.size]
  let getMN : CtxVec Γ → Vec (m * n) :=
    fun xV => castVec hsz (CtxVec.get (Γ := Γ) (s := s) idx xV)
  let getMNCLM : CtxVec Γ →L[ℝ] Vec (m * n) :=
    (Graph.castCLM (h := hsz)).comp (CtxVec.getCLM (Γ := Γ) (s := s) idx)
  let outCast : Vec (m * n) →L[ℝ] Vec (Shape.size s) := Graph.castCLM (h := hsz.symm)
  refine
    { deriv := fun xV =>
        outCast.comp ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) (getMN xV)).comp getMNCLM)
      hasFDerivAt := ?_
      jvp_eq := ?_ }
  · intro xV
    have hget : HasFDerivAt getMN getMNCLM xV := by
      have h := getMNCLM.hasFDerivAt (x := xV)
      have hfun : getMN = fun x => getMNCLM x := by
        funext x
        simp [getMN, getMNCLM, CtxVec.getCLM_apply, Graph.castCLM]
      exact h.congr_of_eventuallyEq hfun.eventuallyEq
    have hlog : HasFDerivAt (LogSoftmaxLastAxis.forwardMN (m := m) (n := n))
        (LogSoftmaxLastAxis.derivMN (m := m) (n := n) (getMN xV)) (getMN xV) :=
      LogSoftmaxLastAxis.hasFDerivAt_forwardMN (m := m) (n := n) (getMN xV)
    have hout : HasFDerivAt (fun z : Vec (m * n) => outCast z) outCast
        (LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (getMN xV)) :=
      outCast.hasFDerivAt (x := LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (getMN xV))
    have hcomp := hout.comp xV (hlog.comp xV hget)
    have hEq :
        (Node.forwardVec (Γ := Γ) (τ := s) (logSoftmaxLast (Γ := Γ) (m := m) (n := n) idx))
          =
        fun xV : CtxVec Γ =>
          outCast (LogSoftmaxLastAxis.forwardMN (m := m) (n := n) (getMN xV)) := by
      funext xV
      ext i
      -- Same proof structure as `softmax_last_fderiv`: keep `castVec` opaque to `simp`.
      simp [logSoftmaxLast, Node.forwardVec_ofVec, getMN, outCast, Graph.castCLM]
    exact hcomp.congr_of_eventuallyEq hEq.eventuallyEq
  · intro xV dxV
    let xMN : Vec (m * n) := getMN xV
    let dxMN : Vec (m * n) := getMN dxV
    have hget : getMNCLM dxV = dxMN := by
      simp [getMNCLM, getMN, dxMN, CtxVec.getCLM_apply, Graph.castCLM]
    have hjvp :
        LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN =
          (LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN :=
      LogSoftmaxLastAxis.jvpMN_eq_derivMN (m := m) (n := n) xMN dxMN
    ext ip
    have hL :
        (Node.jvpVec (Γ := Γ) (τ := s) (logSoftmaxLast (Γ := Γ) (m := m) (n := n) idx) xV dxV) ip
          =
        (castVec hsz.symm (LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN)) ip := by
      simp [logSoftmaxLast, Node.jvpVec_ofVec, getMN, xMN, dxMN]
    have hR :
        ((outCast.comp ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN).comp getMNCLM)) dxV) ip
          =
        (castVec hsz.symm ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN)) ip := by
      simp [outCast, Graph.castCLM, ContinuousLinearMap.comp_apply, hget]
    have hjvp_ip :
        (castVec hsz.symm (LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN)) ip
          =
        (castVec hsz.symm ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN)) ip :=
      congrArg (fun v => v ip) (congrArg (castVec hsz.symm) hjvp)
    calc
      (Node.jvpVec (Γ := Γ) (τ := s) (logSoftmaxLast (Γ := Γ) (m := m) (n := n) idx) xV dxV) ip
          =
        (castVec hsz.symm (LogSoftmaxLastAxis.jvpMN (m := m) (n := n) xMN dxMN)) ip := hL
      _ =
        (castVec hsz.symm ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN) dxMN)) ip := hjvp_ip
      _ =
        ((outCast.comp ((LogSoftmaxLastAxis.derivMN (m := m) (n := n) xMN).comp getMNCLM)) dxV) ip
          := hR.symm


end TapeNodes

end

end Autograd
end Proofs
