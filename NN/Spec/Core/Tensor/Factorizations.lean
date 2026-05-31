/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Matrix factorizations (spec layer)

This file provides **real**, shape-indexed reference implementations of the matrix
factorizations that classical / scientific-ML models (Gaussian processes, kernel ridge
regression, PCA, least squares) depend on, and which were previously missing from the spec
layer:

- `choleskySpec`     — Cholesky factorization `A = L · Lᵀ` (lower-triangular `L`), for SPD `A`.
- `qrSpec`           — QR factorization `A = Q · R` via modified Gram–Schmidt
                       (`Q` has orthonormal columns, `R` upper-triangular).
- `symEigJacobiSpec` — **full** symmetric eigendecomposition via the cyclic Jacobi algorithm
                       (all eigenpairs, not just the largest).
- `svdSpec`          — singular value decomposition `A = U · diag(σ) · Vᵀ`, built on the
                       symmetric eigendecomposition of `Aᵀ·A`.

## Relationship to `eigendecompSpec`

`Spec.eigendecompSpec` (in `NN/Spec/Models/CommonHelpers.lean`) is a power-iteration *stub* that
only recovers the **largest** eigenpair. It is intentionally left untouched (PCA depends on it).
`symEigJacobiSpec` here is the full replacement: for a symmetric matrix it returns *all* `n`
eigenvalues and an orthogonal matrix of eigenvectors.

## Intent / tradeoffs

Like the rest of the spec layer (`determinantSpec`, `inverseSpec`, `matMulSpec`), these prioritize
**mathematical clarity** and **shape safety** over performance, and are intended for small/medium
matrices and proof-oriented reference code. For large-scale numerics, use array-backed runtime
kernels.

Internally the algorithms are written over the plain function representation
`Fin n → Fin n → α` (matrices) and `Fin n → α` (vectors), then wrapped back into `Spec.Tensor`
at the boundary. This keeps the numerical formulas readable and keeps later correctness proofs
working on ordinary functions rather than on nested `Tensor` `match`es.

The iterative routines (Jacobi) take an explicit `sweeps` count: convergence of Jacobi is
asymptotic, so the caller chooses how much work to do. A dozen sweeps is ample for the small
matrices these specs target.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-! ## Boundary conversions between `Spec.Tensor` and plain functions -/

/-- View a matrix tensor as a function `Fin m → Fin n → α`. -/
def toMatFn {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) : Fin m → Fin n → α :=
  fun i j => get2 A i j

/-- Build a matrix tensor from a function `Fin m → Fin n → α`. -/
def ofMatFn {m n : Nat} (f : Fin m → Fin n → α) : Tensor α (.dim m (.dim n .scalar)) :=
  Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (f i j)))

/-- View a vector tensor as a function `Fin n → α`. -/
def toVecFn {n : Nat} (v : Tensor α (.dim n .scalar)) : Fin n → α :=
  fun i => Tensor.toScalar (get v i)

/-- Build a vector tensor from a function `Fin n → α`. -/
def ofVecFn {n : Nat} (f : Fin n → α) : Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (f i))

/-! ## Small numeric helpers on the function representation -/

/-- Dot product of two length-`p` vectors. -/
def dotFn {p : Nat} (u v : Fin p → α) : α :=
  (List.finRange p).foldl (fun s i => s + u i * v i) 0

/-- Euclidean norm of a length-`p` vector. -/
def normFn {p : Nat} (v : Fin p → α) : α :=
  MathFunctions.sqrt (dotFn v v)

/-- Decide `x < y` as a `Bool` (via the `Context`'s decidable `>`). -/
def ltBool (x y : α) : Bool := Context.gtBool y x

/-! ## Cholesky factorization

For a symmetric positive-definite `A`, compute the lower-triangular `L` with `A = L · Lᵀ`.

The columns are computed left to right. Column `j` uses only columns `0 .. j-1`:

- diagonal:  `L[j,j] = sqrt(A[j,j] - Σ_{k<j} L[j,k]²)`
- below:     `L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]`   for `i > j`
- above:     `L[i,j] = 0`                                           for `i < j`
-/

/--
The list of columns of the Cholesky factor `L`, as length-`n` vectors, computed left to right.
Element `j` of the result is column `j` of `L`. Built by a left fold so that when column `j` is
formed, `cols` already holds columns `0 .. j-1`.
-/
def choleskyColsFn {n : Nat} (A : Fin n → Fin n → α) : List (Fin n → α) :=
  (List.finRange n).foldl (fun cols j =>
    -- Σ_{k<j} L[j,k]²  (the already-computed columns evaluated at row `j`).
    let sumsq := (cols.map (fun ck => ck j)).foldl (fun s x => s + x * x) 0
    let Ljj := MathFunctions.sqrt (A j j - sumsq)
    let colj : Fin n → α := fun i =>
      if i.val < j.val then 0
      else if i.val == j.val then Ljj
      else
        -- Σ_{k<j} L[i,k]·L[j,k]
        let s := (cols.map (fun ck => ck i * ck j)).foldl (fun acc x => acc + x) 0
        (A i j - s) / Ljj
    cols ++ [colj]) []

/-- Cholesky factor as a function: `L[i,j] = (choleskyColsFn A)[j] i`. -/
def choleskyFn {n : Nat} (A : Fin n → Fin n → α) : Fin n → Fin n → α :=
  let cols := choleskyColsFn A
  fun i j => (cols.getD j.val (fun _ => 0)) i

/--
Cholesky factorization of a symmetric positive-definite matrix `A`, returning the
lower-triangular factor `L` with `A = L · Lᵀ`.

PyTorch analogue: `torch.linalg.cholesky(A)`.
-/
def choleskySpec {n : Nat} (A : Tensor α (.dim n (.dim n .scalar))) :
    Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (choleskyFn (toMatFn A))

/-! ## Triangular solves and the kernel-ridge (Tikhonov) linear solve

Once `A` is factored as `A = L · Lᵀ` (Cholesky), the linear system `A · x = b` is solved by two
triangular substitutions: forward-solve `L · z = b`, then back-solve `Lᵀ · x = z`. Each substitution
visits the unknowns in an order such that, when row `i` is reached, every unknown it depends on has
already been computed; the accumulator `acc` holds those values and `0` everywhere else, so the dot
`dotFn (row i) acc` is exactly the required partial sum (the not-yet-solved and structurally-zero
terms drop out). This is the linear solve at the heart of CHD `solve_variationnal`. -/

/-- Forward substitution: solve `L · y = b` for a lower-triangular `L` with nonzero diagonal.
Unknowns are visited `0, 1, …, n-1`; when row `i` is reached `acc` holds `y₀ … yᵢ₋₁` (and `0`
elsewhere), so `dotFn (L i) acc = Σ_{k<i} L[i,k]·yₖ` by lower-triangularity. -/
def triSolveLowerFn {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  (List.finRange n).foldl
    (fun acc i => Function.update acc i ((b i - dotFn (L i) acc) / L i i))
    (fun _ => 0)

/-- Back substitution: solve `U · x = y` for an upper-triangular `U` with nonzero diagonal.
Unknowns are visited `n-1, …, 1, 0`; when row `i` is reached `acc` holds `xᵢ₊₁ … xₙ₋₁` (and `0`
elsewhere), so `dotFn (U i) acc = Σ_{k>i} U[i,k]·xₖ` by upper-triangularity. -/
def triSolveUpperFn {n : Nat} (U : Fin n → Fin n → α) (y : Fin n → α) : Fin n → α :=
  (List.finRange n).reverse.foldl
    (fun acc i => Function.update acc i ((y i - dotFn (U i) acc) / U i i))
    (fun _ => 0)

/-- Solve `A · x = b` given a Cholesky factor `L` of `A` (so `A = L · Lᵀ`): forward-solve
`L · z = b`, then back-solve `Lᵀ · x = z`. -/
def cholSolveFn {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  triSolveUpperFn (fun i k => L k i) (triSolveLowerFn L b)

/-- The regularized matrix `K + γ·I` as a function. For a symmetric PSD kernel `K` and `γ > 0`
this is symmetric positive-definite, so its Cholesky factorization succeeds. -/
def addScaledIdFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) : Fin n → Fin n → α :=
  fun i j => K i j + (if i = j then γ else 0)

/-- The Tikhonov-regularized (kernel-ridge) solve `(K + γ·I)·x = b`, via the Cholesky factorization
of `K + γ·I`. This is the linear solve at the core of CHD `solve_variationnal`. -/
def solveRidgeFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  cholSolveFn (choleskyFn (addScaledIdFn K γ)) b

/-- Tensor-level kernel-ridge solve: `(K + γ·I)·x = b`.

PyTorch analogue: `torch.linalg.solve(K + γ·I, b)` (specialized to the SPD Cholesky path). -/
def solveRidgeSpec {n : Nat} (K : Tensor α (.dim n (.dim n .scalar))) (γ : α)
    (b : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  ofVecFn (solveRidgeFn (toMatFn K) γ (toVecFn b))

/-! ## QR factorization (modified Gram–Schmidt)

For `A : m × n`, produce `Q : m × n` with orthonormal columns and `R : n × n` upper-triangular
such that `A = Q · R`. Modified Gram–Schmidt is used for better numerical behavior than the
classical variant.
-/

/-- Internal state for the Gram–Schmidt fold: computed `Q` columns and `R` columns so far. -/
structure GSState (m n : Nat) (α : Type) where
  /-- Orthonormal `Q` columns produced so far (each of length `m`). -/
  qs : List (Fin m → α)
  /-- `R` columns produced so far (each of length `n`, upper-triangular). -/
  rcols : List (Fin n → α)

/--
Run modified Gram–Schmidt over the columns of `A`, returning the `Q` columns and `R` columns.
Column `j` is orthogonalized against the previously produced `Q` columns.
-/
def gramSchmidtFn {m n : Nat} (A : Fin m → Fin n → α) : GSState m n α :=
  (List.finRange n).foldl (fun (st : GSState m n α) j =>
    let a : Fin m → α := fun i => A i j
    -- r[k,j] = qₖ · a   for each previously computed column k
    let rkjs : List α := st.qs.map (fun qk => dotFn qk a)
    -- v = a - Σ r[k,j] qₖ
    let v : Fin m → α := fun i =>
      a i - (List.zip st.qs rkjs).foldl (fun acc (qk, r) => acc + r * qk i) 0
    let rjj := normFn v
    let qj : Fin m → α := fun i => if Context.gtBool rjj 0 then v i / rjj else 0
    let rcolj : Fin n → α := fun k =>
      if k.val < j.val then rkjs.getD k.val 0
      else if k.val == j.val then rjj
      else 0
    { qs := st.qs ++ [qj], rcols := st.rcols ++ [rcolj] }) { qs := [], rcols := [] }

/-- The `Q` factor (orthonormal columns) of the QR factorization of `A`. -/
def qrQSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) :
    Tensor α (.dim m (.dim n .scalar)) :=
  let st := gramSchmidtFn (toMatFn A)
  ofMatFn (fun i j => (st.qs.getD j.val (fun _ => 0)) i)

/-- The `R` factor (upper-triangular) of the QR factorization of `A`. -/
def qrRSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) :
    Tensor α (.dim n (.dim n .scalar)) :=
  let st := gramSchmidtFn (toMatFn A)
  ofMatFn (fun k j => (st.rcols.getD j.val (fun _ => 0)) k)

/--
QR factorization of `A : m × n` via modified Gram–Schmidt, returning `(Q, R)` with
`A = Q · R`, `Q` orthonormal columns, `R` upper-triangular.

PyTorch analogue: `torch.linalg.qr(A)`.
-/
def qrSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) :
    Tensor α (.dim m (.dim n .scalar)) × Tensor α (.dim n (.dim n .scalar)) :=
  (qrQSpec A, qrRSpec A)

/-! ## Symmetric eigendecomposition (cyclic Jacobi)

For a symmetric `A`, iteratively apply Givens rotations `J` that zero one off-diagonal entry at a
time, accumulating `A ← Jᵀ A J` and `V ← V J`. Each `J` is orthogonal, so every step is an
orthogonal similarity: the spectrum is preserved and `V` stays orthogonal. After enough sweeps the
off-diagonal mass vanishes; the diagonal holds the eigenvalues and the columns of `V` are the
eigenvectors.
-/

/-!
The iteration below runs over an `Array (Array α)` representation rather than `Fin n → Fin n → α`.
Arrays are strict values, so threading them through the rotation loop cannot build the deep closure
chains that a functional representation would (one matrix product per rotation), which is what keeps
execution cheap. We convert to/from `Spec.Tensor` only at the boundary.
-/

/-- Read entry `(i, j)` of an `Array (Array α)` matrix (`0` if out of bounds). -/
def arrGet (M : Array (Array α)) (i j : Nat) : α := (M.getD i #[]).getD j 0

/-- Materialize a matrix function into a strict `Array (Array α)`. -/
def matToArr {n : Nat} (X : Fin n → Fin n → α) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => X i j))

/-- Matrix product `X · Y` of two `n × n` array matrices. -/
def arrMatMul (n : Nat) (X Y : Array (Array α)) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n =>
    (List.finRange n).foldl (fun s k => s + arrGet X i.val k.val * arrGet Y k.val j.val) 0))

/-- Transpose of an `n × n` array matrix. -/
def arrTr (n : Nat) (X : Array (Array α)) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => arrGet X j.val i.val))

/-- `n × n` identity as an array matrix. -/
def arrId (n : Nat) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => if i.val == j.val then 1 else 0))

/--
Givens rotation in the `(p, q)` plane as an array matrix:
identity except `J[p,p]=J[q,q]=c`, `J[p,q]=s`, `J[q,p]=-s`.
-/
def arrGivens (n : Nat) (p q : Nat) (c s : α) : Array (Array α) :=
  Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n =>
    if i.val == p && j.val == p then c
    else if i.val == q && j.val == q then c
    else if i.val == p && j.val == q then s
    else if i.val == q && j.val == p then -s
    else if i.val == j.val then 1 else 0))

/--
Apply one Jacobi rotation that targets off-diagonal entry `(p, q)`, updating `(A, V)` as strict
arrays. If `A[p,q]` is already (numerically) zero, the state is returned unchanged.

The rotation parameters follow Golub & Van Loan:
`τ = (A[q,q] - A[p,p]) / (2 A[p,q])`, `t = sign(τ)/(|τ| + sqrt(1+τ²))` (or `1` if `τ = 0`),
`c = 1/sqrt(1+t²)`, `s = t·c`.
-/
def arrJacobiRotate (n : Nat) (A V : Array (Array α)) (p q : Nat) :
    Array (Array α) × Array (Array α) :=
  let apq := arrGet A p q
  if Context.gtBool (MathFunctions.abs apq) 0 then
    let τ := (arrGet A q q - arrGet A p p) / (Numbers.two * apq)
    let absτ := MathFunctions.abs τ
    let sgn : α := if ltBool τ 0 then Numbers.neg_one else 1
    let t : α :=
      if Context.gtBool absτ 0 then sgn / (absτ + MathFunctions.sqrt (1 + τ * τ)) else 1
    let c := 1 / MathFunctions.sqrt (1 + t * t)
    let s := t * c
    let J := arrGivens n p q c s
    (arrMatMul n (arrTr n J) (arrMatMul n A J), arrMatMul n V J)
  else
    (A, V)

/-- All index pairs `(p, q)` with `p < q`, in row-major order (one cyclic Jacobi sweep). -/
def jacobiPairs (n : Nat) : List (Nat × Nat) :=
  (List.range n).flatMap (fun p =>
    (List.range n).filterMap (fun q => if p < q then some (p, q) else none))

/-- One Jacobi sweep: rotate through every `(p, q)` pair with `p < q`. -/
def arrJacobiSweep (n : Nat) (st : Array (Array α) × Array (Array α)) :
    Array (Array α) × Array (Array α) :=
  (jacobiPairs n).foldl (fun s pq => arrJacobiRotate n s.1 s.2 pq.1 pq.2) st

/-- Run `sweeps` Jacobi sweeps starting from `(A, I)`, returning the rotated `A` and accumulated `V`. -/
def arrJacobiRun (n : Nat) (A : Array (Array α)) (sweeps : Nat) :
    Array (Array α) × Array (Array α) :=
  (List.range sweeps).foldl (fun st _ => arrJacobiSweep n st) (A, arrId n)

/--
Full symmetric eigendecomposition of `A` via cyclic Jacobi, returning `(eigenvalues, eigenvectors)`.

The eigenvalues are the diagonal of the rotated matrix; the eigenvectors are the **columns** of the
returned matrix `V` (so `eigenvectors[i, j]` is the `i`-th component of the `j`-th eigenvector).
`sweeps` controls how many Jacobi sweeps to run (default `12`).

Unlike `eigendecompSpec`, this recovers **all** `n` eigenpairs.

PyTorch analogue: `torch.linalg.eigh(A)`.
-/
def symEigJacobiSpec {n : Nat} (A : Tensor α (.dim n (.dim n .scalar))) (sweeps : Nat := 12) :
    Tensor α (.dim n .scalar) × Tensor α (.dim n (.dim n .scalar)) :=
  let (Af, Vf) := arrJacobiRun n (matToArr (toMatFn A)) sweeps
  (ofVecFn (fun i => arrGet Af i.val i.val), ofMatFn (fun i j => arrGet Vf i.val j.val))

/-! ## Singular value decomposition

For `A : m × n`, form the symmetric `M = Aᵀ·A : n × n`, eigendecompose it as `M = V Λ Vᵀ`,
take `σ = sqrt(max(Λ, 0))`, and recover `U` columns as `uⱼ = A vⱼ / σⱼ` (zero when `σⱼ = 0`).
Then `A = U · diag(σ) · Vᵀ`. This is the simplest reference SVD and is exact (up to the Jacobi
sweep count) for `A` of full column rank.
-/

/--
Singular value decomposition of `A : m × n` returning `(U, σ, V)` with
`A = U · diag(σ) · Vᵀ`, `U : m × n` with orthonormal columns (full-rank case), `σ : n` the singular
values, and `V : n × n` orthogonal.

`sweeps` controls the Jacobi sweep count used for the eigendecomposition of `Aᵀ·A`.

PyTorch analogue: `torch.linalg.svd(A, full_matrices=False)`.
-/
def svdSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) (sweeps : Nat := 12) :
    Tensor α (.dim m (.dim n .scalar)) × Tensor α (.dim n .scalar) ×
      Tensor α (.dim n (.dim n .scalar)) :=
  let Af := toMatFn A
  -- M = Aᵀ A  (n × n, symmetric PSD), as a strict array matrix
  let M : Array (Array α) :=
    Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n =>
      (List.finRange m).foldl (fun s k => s + Af k i * Af k j) 0))
  let (Mf, Vf) := arrJacobiRun n M sweeps
  let σ : Fin n → α := fun j =>
    let d := arrGet Mf j.val j.val
    MathFunctions.sqrt (if ltBool d 0 then 0 else d)
  let U : Fin m → Fin n → α := fun i j =>
    let sj := σ j
    if Context.gtBool sj 0 then
      ((List.finRange n).foldl (fun s k => s + Af i k * arrGet Vf k.val j.val) 0) / sj
    else 0
  (ofMatFn U, ofVecFn σ, ofMatFn (fun i j => arrGet Vf i.val j.val))

/-! ## CHD variational solve, noise, and γ-selection (eigendecomposition form)

CHD's `perform_regression_and_find_gamma` does not use the Cholesky route above; it works through the
*eigendecomposition* `K = V · diag(λ) · Vᵀ` returned by `eigh(K)` (`symEigJacobiSpec`). Three routines
share one arithmetic core — the *projected data* `Pga = Vᵀ · ga` and the *shrinkage coefficients*
`rᵢ = γ/(λᵢ + γ)`:

* `solve_variationnal` returns the solution `yb = -V·(Pga/(λ+γ))` (`= -(K+γ·I)⁻¹·ga`) and a scalar
  `noise` level;
* `find_gamma` minimises that *same* `noise` functional over `γ`;
* `Z_test` evaluates the `noise` functional on random Gaussian data to obtain a null distribution.

The definitions below mirror `interpolatory.py` verbatim, taking the eigenpairs `(Λ, V)` the solver
returns — exactly as CHD passes `eigh(K)` into them. Their algebraic identities (the solve is the
regularized inverse; the noise is a spectral quadratic-form ratio in `[0,1]`) are proved in
[`NN.Proofs.Tensor.Basic.FactorizationsVariational`](../../../Proofs/Tensor/Basic/FactorizationsVariational.lean).
-/

/-- Projected data `Pga = Vᵀ · ga`: component `i` is `⟨vᵢ, ga⟩`, the coordinate of `ga` along
eigenvector `i` (column `i` of `V`). Mirrors `np.dot(eigenvectors.T, ga)`. -/
def projFn {n : Nat} (V : Fin n → Fin n → α) (ga : Fin n → α) : Fin n → α :=
  fun i => dotFn (fun k => V k i) ga

/-- Shrinkage coefficient `rᵢ = γ/(λᵢ + γ)`. For a PSD spectrum (`λᵢ ≥ 0`) and `γ > 0` this lies in
`(0, 1]`. Mirrors `coeffs = gamma / (eigenvalues + gamma)`. -/
def ridgeCoeffFn {n : Nat} (Λ : Fin n → α) (γ : α) : Fin n → α :=
  fun i => γ / (Λ i + γ)

/-- The CHD variational solution `yb = -V·(Pga / (λ + γ))` in eigendecomposition form. Equal to
`-(K + γ·I)⁻¹·ga`. Mirrors `yb = -np.dot(eigenvectors, Pga / (eigenvalues + gamma))`. -/
def variationalSolveFn {n : Nat} (Λ : Fin n → α) (V : Fin n → Fin n → α) (γ : α) (ga : Fin n → α) :
    Fin n → α :=
  let Pga := projFn V ga
  fun i => -dotFn (V i) (fun j => Pga j / (Λ j + γ))

/-- The CHD `noise` level (also the `find_gamma` loss and the `Z_test` per-sample statistic):
`Σᵢ (Pgaᵢ·rᵢ)² / Σᵢ (Pgaᵢ·rᵢ)·Pgaᵢ`, with `rᵢ = γ/(λᵢ + γ)`. Mirrors
`noise = np.dot(Pgacoeff, Pgacoeff) / np.dot(Pgacoeff, Pga)` where `Pgacoeff = Pga * coeffs`. -/
def varNoiseFn {n : Nat} (Λ : Fin n → α) (γ : α) (Pga : Fin n → α) : α :=
  let pc := fun i => Pga i * ridgeCoeffFn Λ γ i
  dotFn pc pc / dotFn pc Pga

/-- Tensor-level CHD variational solve `yb = -(K + γ·I)⁻¹·ga`, from eigenpairs `(evals, V)`. -/
def variationalSolveSpec {n : Nat} (evals : Tensor α (.dim n .scalar))
    (V : Tensor α (.dim n (.dim n .scalar))) (γ : α) (ga : Tensor α (.dim n .scalar)) :
    Tensor α (.dim n .scalar) :=
  ofVecFn (variationalSolveFn (toVecFn evals) (toMatFn V) γ (toVecFn ga))

/-- Tensor-level CHD `noise` / `find_gamma` loss, from eigenvalues, `γ` and the data `ga` (projected
internally as `Pga = Vᵀ·ga`). -/
def varNoiseSpec {n : Nat} (evals : Tensor α (.dim n .scalar))
    (V : Tensor α (.dim n (.dim n .scalar))) (γ : α) (ga : Tensor α (.dim n .scalar)) : α :=
  varNoiseFn (toVecFn evals) γ (projFn (toMatFn V) (toVecFn ga))

/-! ## CHD mode kernels (`Modes/kernels.py`)

Everything above takes the kernel matrix `K` as input, assuming it is symmetric positive-semidefinite.
CHD *builds* `K` from data: for each pair of samples, a mode kernel evaluates a feature inner product.
The simplest is the **linear mode** (`LinearMode.vectorized_kernel`):

`K[i,j] = 1 + scale · Σ_k (which_dim_k · X[i,k]) · X[j,k]`,

a constant `1` plus a scaled Gram matrix of the (column-masked) data. For the binary selection mask
`which_dim_k ∈ {0,1}` CHD uses, `which_dim_k · X[i,k] · X[j,k] = (which_dim_k X[i,k])·(which_dim_k X[j,k])`,
so `K = 𝟙𝟙ᵀ + scale · Φ·Φᵀ` with `Φ` the masked data — manifestly symmetric positive-semidefinite for
`scale ≥ 0`. That PSD fact (proved in `FactorizationsKernels`) discharges the standing `PosSemidef`
hypothesis of the solve/`find_gamma` development for the real linear kernel. -/

/-- Column-mask the data matrix by a per-feature weight `w` (CHD `which_dim`): zero out / scale
feature `k` by `w k`. -/
def maskColsFn {n d : Nat} (X : Fin n → Fin d → α) (w : Fin d → α) : Fin n → Fin d → α :=
  fun i k => w k * X i k

/-- Linear-mode kernel matrix `K[i,j] = 1 + scale · ⟨Φ i, Φ j⟩`, `Φ = maskColsFn X w` the masked data.
For a binary selection mask this is exactly CHD `LinearMode.vectorized_kernel`. -/
def linearKernelFn {n d : Nat} (X : Fin n → Fin d → α) (w : Fin d → α) (scale : α) :
    Fin n → Fin n → α :=
  fun i j => 1 + scale * dotFn (maskColsFn X w i) (maskColsFn X w j)

/-- Tensor-level linear-mode kernel: `K = 𝟙𝟙ᵀ + scale · Φ·Φᵀ` from data `X` and selection mask `w`.

PyTorch analogue: `1 + scale * (X * which_dim).matmul(X.T)`. -/
def linearKernelSpec {n d : Nat} (X : Tensor α (.dim n (.dim d .scalar)))
    (w : Tensor α (.dim d .scalar)) (scale : α) : Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (linearKernelFn (toMatFn X) (toVecFn w) scale)

end Spec
