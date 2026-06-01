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
Strict, array-backed runtime implementation of `choleskyColsFn` (registered via `@[implemented_by]`).
Each column is *materialized* into an `Array α`, so a back-reference `L[i,k]` is an `O(1)` lookup
rather than a closure that re-evaluates the whole prefix. The closure form below is mathematically
clean (and is what the proofs reason about), but reading the full factor `L` from it re-evaluates
columns exponentially — ruinous in the interpreter (`#eval`). This computes the *same* factor strictly;
the numeric examples (`A = L·Lᵀ`, the ridge-solve residual ≈ 0) validate the two agree.
-/
def choleskyColsImpl {n : Nat} (A : Fin n → Fin n → α) : List (Fin n → α) :=
  let cols : Array (Array α) := (List.finRange n).foldl (fun cols j =>
    let jv := j.val
    -- Σ_{k<j} L[j,k]²  (previous columns at row `j`, read from the materialized arrays).
    let sumsq := (List.finRange n).foldl
      (fun s k => if k.val < jv then s + (cols.getD k.val #[]).getD jv 0 * (cols.getD k.val #[]).getD jv 0
        else s) 0
    let Ljj := MathFunctions.sqrt (A j j - sumsq)
    let colArr : Array α := Array.ofFn (fun i : Fin n =>
      if i.val < jv then 0
      else if i.val == jv then Ljj
      else
        -- Σ_{k<j} L[i,k]·L[j,k]
        let s := (List.finRange n).foldl
          (fun acc k => if k.val < jv then
            acc + (cols.getD k.val #[]).getD i.val 0 * (cols.getD k.val #[]).getD jv 0 else acc) 0
        (A i j - s) / Ljj)
    cols.push colArr) #[]
  (List.finRange n).map (fun j => fun i => (cols.getD j.val #[]).getD i.val 0)

/--
The list of columns of the Cholesky factor `L`, as length-`n` vectors, computed left to right.
Element `j` of the result is column `j` of `L`. Built by a left fold so that when column `j` is
formed, `cols` already holds columns `0 .. j-1`.

The runtime implementation is `choleskyColsImpl` (strict arrays); the closure form here is the one the
correctness proofs reason about. Both compute the same factor.
-/
@[implemented_by choleskyColsImpl]
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

/--
Strict, array-backed runtime implementation of `solveRidgeFn` (registered via `@[implemented_by]`).
It factors `K + γ·I = L·Lᵀ` and runs both triangular substitutions entirely over `Array`s, so no step
materializes the deep `Fin n → α` closures the functional definition builds — those re-evaluate
columns / the substitution accumulator exponentially, which is ruinous in the interpreter (`#eval`).
Same linear solve; the numeric examples (residual `(K+γ·I)·x − b ≈ 0`) validate the two agree.
-/
def solveRidgeImpl {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  let A : Fin n → Fin n → α := fun i j => K i j + (if i.val == j.val then γ else 0)
  -- Cholesky columns, left to right: `cols[j][i] = L[i][j]` (strict arrays, `O(1)` back-reference).
  let cols : Array (Array α) := (List.finRange n).foldl (fun cols j =>
    let jv := j.val
    let sumsq := (List.finRange n).foldl
      (fun s k => if k.val < jv then let v := (cols.getD k.val #[]).getD jv 0; s + v * v else s) 0
    let Ljj := MathFunctions.sqrt (A j j - sumsq)
    cols.push (Array.ofFn (fun i : Fin n =>
      if i.val < jv then 0
      else if i.val == jv then Ljj
      else
        let s := (List.finRange n).foldl (fun acc k =>
          if k.val < jv then
            acc + (cols.getD k.val #[]).getD i.val 0 * (cols.getD k.val #[]).getD jv 0
          else acc) 0
        (A i j - s) / Ljj))) #[]
  let Lent : Nat → Nat → α := fun i j => (cols.getD j #[]).getD i 0
  -- Forward solve `L · z = b`: `z[i] = (b[i] − Σ_{k<i} L[i,k]·z[k]) / L[i,i]`.
  let z : Array α := (List.finRange n).foldl (fun z i =>
    let iv := i.val
    let s := (List.finRange n).foldl
      (fun acc k => if k.val < iv then acc + Lent iv k.val * z.getD k.val 0 else acc) 0
    z.push ((b i - s) / Lent iv iv)) #[]
  -- Back solve `Lᵀ · x = z`: `x[i] = (z[i] − Σ_{k>i} L[k,i]·x[k]) / L[i,i]`, `i = n−1 … 0`.
  let x : Array α := (List.finRange n).reverse.foldl (fun xs i =>
    let iv := i.val
    let s := (List.finRange n).foldl
      (fun acc k => if iv < k.val then acc + Lent k.val iv * xs.getD k.val 0 else acc) 0
    xs.set! iv ((z.getD iv 0 - s) / Lent iv iv)) (Array.replicate n 0)
  fun i => x.getD i.val 0

/-- The Tikhonov-regularized (kernel-ridge) solve `(K + γ·I)·x = b`, via the Cholesky factorization
of `K + γ·I`. This is the linear solve at the core of CHD `solve_variationnal`.

The runtime implementation is `solveRidgeImpl` (strict arrays); the closure form here, built from the
verified `choleskyFn` / `triSolve*` pieces, is what the correctness proofs reason about. Both compute
the same solution. -/
@[implemented_by solveRidgeImpl]
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

/-! ## CHD `Z_test`: null-distribution significance thresholds (`interpolatory.py`)

`varNoiseFn` gives the `noise` of the *observed* data. To decide whether that noise is small *enough*
to signal a real edge, CHD compares it against the null distribution of the **same** statistic under
random data (`Z_test`): draw `N` standard-Gaussian samples, score each one's `noise`, sort them, and
take the 5th and 95th percentiles as `Z_low`/`Z_high`. An edge is significant when the observed
`noise < Z_low` — strictly below the null's lower tail.

The random draws enter the spec as an explicit family `samples : Fin N → Fin n → α` (row `j` a draw);
the randomness itself is the caller's, exactly as CHD threads a `jax.random` key into `Z_test`. The
selection guarantees (each threshold lies in `[0,1]`, and `Z_low ≤ Z_high`) are proved over `ℝ` in
[`NN.Proofs.Tensor.Basic.FactorizationsDecision`](../../../Proofs/Tensor/Basic/FactorizationsDecision.lean),
reusing the verified `noise ∈ [0,1]` bound for *every* null sample. -/

/-- The `Z_test` null sample of per-draw `noise` levels: each random draw `samples j` is projected
(`Pga = Vᵀ·sⱼ`) and scored by the **same** `varNoiseFn` as the data. Mirrors
`noises = vecdot(Pgas_coeffs, Pgas_coeffs) / vecdot(Pgas_coeffs, Pgas)` in `Z_test`. -/
def sampleNoisesFn {n N : Nat} (Λ : Fin n → α) (V : Fin n → Fin n → α) (γ : α)
    (samples : Fin N → Fin n → α) : Fin N → α :=
  fun j => varNoiseFn Λ γ (projFn V (samples j))

/-- `x ≤ y` as a `Bool` via the `Context` order (`x ≤ y` is `¬ y < x`); the sort key for the order
statistics below. -/
def leBool (x y : α) : Bool := !ltBool y x

/-- The `k`-th smallest of a finite family `a : Fin N → α`, by sorting the values (ascending, via the
`Context` order) and indexing. The `getD … 0` fallback is total; for `k < N` it is a genuine order
statistic (see `kthSmallestFn_mem`/`_mono` in `FactorizationsDecision`). -/
def kthSmallestFn {N : Nat} (a : Fin N → α) (k : Nat) : α :=
  (((List.finRange N).map a).mergeSort leBool).getD k 0

/-- The 5th-percentile index of an `N`-sample null distribution (`int(0.05·N)`, i.e. `⌊N/20⌋`). -/
def zLowIdx (N : Nat) : Nat := N / 20

/-- The 95th-percentile index of an `N`-sample null distribution (`int(0.95·N)`, i.e. `⌊19·N/20⌋`). -/
def zHighIdx (N : Nat) : Nat := 19 * N / 20

/-- CHD `Z_low`: the 5th percentile of the null `noise` distribution — the significance threshold an
observed `noise` must beat (`B_samples[int(0.05·N)]`). -/
def zLowFn {n N : Nat} (Λ : Fin n → α) (V : Fin n → Fin n → α) (γ : α)
    (samples : Fin N → Fin n → α) : α :=
  kthSmallestFn (sampleNoisesFn Λ V γ samples) (zLowIdx N)

/-- CHD `Z_high`: the 95th percentile of the null `noise` distribution (`B_samples[int(0.95·N)]`). -/
def zHighFn {n N : Nat} (Λ : Fin n → α) (V : Fin n → Fin n → α) (γ : α)
    (samples : Fin N → Fin n → α) : α :=
  kthSmallestFn (sampleNoisesFn Λ V γ samples) (zHighIdx N)

/-- CHD's per-kernel significance verdict: the observed `noise` beats the null's lower tail
(`noise < Z_low`), i.e. the edge is real. This is exactly the validity test `MinNoiseKernelChooser`
folds over (`noises < Z_lows`). -/
def zSignificantFn (noise Zlow : α) : Bool := ltBool noise Zlow

/-- Tensor-level `Z_low` threshold from eigenpairs `(evals, V)`, regularization `γ`, and a family of
null draws (the rows of `S : Tensor (.dim N (.dim n .scalar))`). -/
def zLowSpec {n N : Nat} (evals : Tensor α (.dim n .scalar))
    (V : Tensor α (.dim n (.dim n .scalar))) (γ : α)
    (S : Tensor α (.dim N (.dim n .scalar))) : α :=
  zLowFn (toVecFn evals) (toMatFn V) γ (toMatFn S)

/-- Tensor-level `Z_high` threshold from eigenpairs `(evals, V)`, regularization `γ`, and a family of
null draws (the rows of `S`). -/
def zHighSpec {n N : Nat} (evals : Tensor α (.dim n .scalar))
    (V : Tensor α (.dim n (.dim n .scalar))) (γ : α)
    (S : Tensor α (.dim N (.dim n .scalar))) : α :=
  zHighFn (toVecFn evals) (toMatFn V) γ (toMatFn S)

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

/-- Quadratic-mode kernel matrix `K[i,j] = scale · (alpha + ⟨Φ i, Φ j⟩)² + (1 − alpha²·scale)`,
`Φ = maskColsFn X w` the masked data. This is exactly CHD `QuadraticMode.vectorized_kernel`. Algebraically
it expands to `K = 𝟙𝟙ᵀ + (2·scale·alpha)·Φ·Φᵀ + scale·(Φ·Φᵀ ⊙ Φ·Φᵀ)` (the last a Hadamard square), so it
is positive-semidefinite for `scale ≥ 0` and `alpha ≥ 0` by the Schur product theorem — see
`FactorizationsKernels`. The square is written as a product to stay polymorphic over `Context α`. -/
def quadraticKernelFn {n d : Nat} (X : Fin n → Fin d → α) (w : Fin d → α) (scale alpha : α) :
    Fin n → Fin n → α :=
  fun i j =>
    let m := dotFn (maskColsFn X w i) (maskColsFn X w j)
    scale * ((alpha + m) * (alpha + m)) + (1 - alpha * alpha * scale)

/-- Tensor-level quadratic-mode kernel from data `X`, selection mask `w`, `scale`, and offset `alpha`.

PyTorch analogue: `scale * (alpha + (X * which_dim).matmul(X.T))**2 + (1 - alpha**2 * scale)`. -/
def quadraticKernelSpec {n d : Nat} (X : Tensor α (.dim n (.dim d .scalar)))
    (w : Tensor α (.dim d .scalar)) (scale alpha : α) : Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (quadraticKernelFn (toMatFn X) (toVecFn w) scale alpha)

/-- Gaussian-mode product kernel matrix
`K[i,j] = scale · ∏_dim (1 + w[dim] · exp(−(X[i,dim]−X[j,dim])²/(2·l²)))` — the Gaussian contribution of
CHD `GaussianMode` (`scale * jnp.prod(1 + which_dim * exps, axis=2)`, with
`exps[i,j,dim] = exp(−(X[i,dim]−X[j,dim])²/(2·l²))`).

Each feature factor `1 + w[dim]·k` (with `k` the per-feature Gaussian) is positive-semidefinite — `𝟙𝟙ᵀ`
plus a nonnegative multiple of the Gaussian PSD matrix — and the product over features is PSD by the
**Schur product theorem**, so `K` is PSD for `scale ≥ 0` and a nonnegative mask `w ≥ 0`
(see `FactorizationsKernels`). The product is an explicit `foldl` and the squared difference a product, to
stay polymorphic over the law-free `Context α`. -/
def gaussianKernelFn {n d : Nat} (X : Fin n → Fin d → α) (w : Fin d → α) (scale l : α) :
    Fin n → Fin n → α :=
  fun i j => scale * (List.finRange d).foldl
    (fun acc dim =>
      acc * (1 + w dim *
        MathFunctions.exp (-((X i dim - X j dim) * (X i dim - X j dim)) / ((1 + 1) * l * l)))) 1

/-- Tensor-level Gaussian-mode product kernel from data `X`, selection mask `w`, `scale`, and length
scale `l`.

PyTorch analogue: `scale * torch.prod(1 + which_dim * torch.exp(-(dx**2)/(2*l**2)), dim=2)`. -/
def gaussianKernelSpec {n d : Nat} (X : Tensor α (.dim n (.dim d .scalar)))
    (w : Tensor α (.dim d .scalar)) (scale l : α) : Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (gaussianKernelFn (toMatFn X) (toVecFn w) scale l)

/-! ## CHD discovery decision layer (`decision.py`, `_GraphDiscoveryMain.py`)

Everything above turns a kernel `K` into a `noise` level (`varNoiseFn`, proven to lie in `[0,1]`) and a
`Z_test` lower bound `Z_low`. CHD's outer *discovery loop* turns those numbers into graph-structure
decisions. Four deterministic choices, each a comparison over finite data:

* **which feature to prune next** — the least-*activated* ancestor (`min_activation = np.argmin(activations)`
  in `helper_functions.step`);
* **which kernel mode admits an edge** — `MinNoiseKernelChooser`: the valid kernel (`noise < Z_low`) of
  least `noise`, or none if no kernel is valid (`decision.py`);
* **which pruning iteration to report** — `MaxIncrementModeChooser`: the iteration of largest `noise`
  increment (`decision.py`);
* **when to stop** — `np.all(active_modes == 0)`, every ancestor pruned (`_GraphDiscoveryMain.py`).

The definitions mirror the Python verbatim; their *selection guarantees* (the chosen index really is the
least/greatest, the chooser is sound and complete against `noise < Z_low`) are proved over `ℝ` in
[`NN.Proofs.Tensor.Basic.FactorizationsDecision`](../../../Proofs/Tensor/Basic/FactorizationsDecision.lean).
Comparisons use the `Context` order test (`ltBool`/`gtBool`), so the same definitions run over `Float`. -/

/-- Index of a least element of a nonempty finite family (first on ties), by a left fold — CHD's
`np.argmin(activations)` for picking the least-activated ancestor to prune. -/
def argMinFn {n : Nat} (a : Fin (n + 1) → α) : Fin (n + 1) :=
  (List.finRange (n + 1)).foldl (fun best j => if ltBool (a j) (a best) then j else best)
    (0 : Fin (n + 1))

/-- Index of a greatest element of a nonempty finite family (first on ties), by a left fold — the
`np.argmax` underlying the mode chooser. -/
def argMaxFn {n : Nat} (a : Fin (n + 1) → α) : Fin (n + 1) :=
  (List.finRange (n + 1)).foldl (fun best j => if Context.gtBool (a j) (a best) then j else best)
    (0 : Fin (n + 1))

/-- CHD `MinNoiseKernelChooser`. Among candidate kernels with per-kernel `noise` and `Z_low`, a kernel
is *valid* (admits an edge) when `noise < Z_low`; return the valid kernel of least `noise`, or `none`
if none is valid. Mirrors `valid = noises < Z_lows; argmin(np.where(valid, noises, 2))` — the `2`
sentinel (any value above the `noise` ceiling `1`) written `1 + 1` to stay polymorphic. -/
def kernelChooserFn {n : Nat} (noises Zlows : Fin (n + 1) → α) : Option (Fin (n + 1)) :=
  let key := fun i => if ltBool (noises i) (Zlows i) then noises i else 1 + 1
  let s := argMinFn key
  if ltBool (noises s) (Zlows s) then some s else none

/-- Per-iteration `noise` increments of CHD `MaxIncrementModeChooser`:
`increments[i] = noises[i+1] − noises[i]` for an interior `i`, and `1 − noises[last]` at the end (the
gap to the `noise` ceiling, `np.append(increments, 1 - list_of_noises[-1])`). -/
def modeIncrementFn {n : Nat} (noises : Fin (n + 1) → α) : Fin (n + 1) → α :=
  fun i => if h : i.val + 1 < n + 1 then noises ⟨i.val + 1, h⟩ - noises i else 1 - noises i

/-- CHD `MaxIncrementModeChooser`: report the pruning iteration with the largest jump in `noise`
(`np.argmax(increments)`). -/
def modeChooserFn {n : Nat} (noises : Fin (n + 1) → α) : Fin (n + 1) :=
  argMaxFn (modeIncrementFn noises)

/-- CHD stopping rule `np.all(active_modes == 0)`: every ancestor has been pruned. An entry counts as
zero exactly when it is neither positive nor negative in the `Context` order. -/
def allPrunedFn {k : Nat} (m : Fin k → α) : Bool :=
  (List.finRange k).all (fun i => !Context.gtBool (m i) 0 && !Context.gtBool 0 (m i))

end Spec
