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

This file provides **real**, shape-indexed reference implementations of the two *exact, finite*
matrix factorizations that classical / scientific-ML models (Gaussian processes, kernel ridge
regression, PCA, least squares) depend on, and which were previously missing from the spec layer:

- `choleskySpec`   — Cholesky factorization `A = L · Lᵀ` (lower-triangular `L`), proved for
                     matrices with positive executable Cholesky pivots.
- `qrSpec`         — QR factorization `A = Q · R` via classical Gram–Schmidt; under positive
                     executable `R` pivots, `Q` has orthonormal columns and `R` is upper-triangular.

It also provides the linear solves that ride on the Cholesky factor:

- `triSolveLowerFn` / `triSolveUpperFn` — forward / back triangular substitution;
- `cholSolveFn`    — solve `A · x = b` from a Cholesky factor of `A`;
- `solveRidgeSpec` — the Tikhonov / kernel-ridge solve `(K + γ·I) · x = b`.

## Verification scope

The **verified** contribution is the factorizations: `choleskySpec` / `qrSpec` come with
reconstruction and structural theorems (`IsCholesky` / `IsQR`, lower- and upper-triangularity,
orthonormality) in `NN.Proofs.Tensor.Basic.Factorizations*`, under their stated positive-pivot
success hypotheses. The triangular- and ridge-solve helpers above (`triSolveLowerFn`,
`triSolveUpperFn`, `cholSolveFn`, `solveRidgeSpec`) are **executable APIs only**: this PR does *not*
yet prove their correctness (no `triSolveLower · x = b` / `solveRidge` correctness theorem has
landed). They follow the standard substitution formulas over the readable function representation
and are exercised by `#eval` examples, but should not be read as carrying a verified-correctness
guarantee.

## Intent / tradeoffs

Like the rest of the spec layer (`determinantSpec`, `inverseSpec`, `matMulSpec`), these prioritize
**mathematical clarity** and **shape safety** over performance, and are intended for small/medium
matrices and proof-oriented reference code. For large-scale numerics, use array-backed runtime
kernels.

Internally the algorithms are written over the plain function representation
`Fin n → Fin n → α` (matrices) and `Fin n → α` (vectors), then wrapped back into `Spec.Tensor`
at the boundary. This keeps the numerical formulas readable and keeps later correctness proofs
working on ordinary functions rather than on nested `Tensor` `match`es.
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

/-! ## Cholesky factorization

For an input whose executable Cholesky pivots are positive, compute the lower-triangular `L` with
`A = L · Lᵀ`. Symmetric positive-definiteness is the standard sufficient condition, but the theorem
in this file family is stated against the executable positive-pivot success condition.

The columns are computed left to right. Column `j` uses only columns `0 .. j-1`:

- diagonal:  `L[j,j] = sqrt(A[j,j] - Σ_{k<j} L[j,k]²)`
- below:     `L[i,j] = (A[i,j] - Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]`   for `i > j`
- above:     `L[i,j] = 0`                                           for `i < j`

### Trust boundary: the `@[implemented_by]` performance hooks

Several defs here (`choleskyColsFn`, `cholSolveFn`, `solveRidgeFn`) carry an `@[implemented_by …Impl]`
attribute. The clean closure form is what the correctness proofs reason about; the `…Impl` companion
is a strict, array-backed rewrite that the compiler runs instead, so `#eval` stays fast (the closure
form re-evaluates prefixes exponentially in the interpreter).

**This substitution is *trusted*, not verified.** No Lean theorem proves `choleskyColsFn = choleskyColsImpl`
(etc.), so compiled `#eval`/runtime code executes the `…Impl` body while the proofs constrain only the
closure body. The two are believed equal by construction (they transcribe the same recurrence), and the
numeric examples in `NN/Examples/Factorization` are *evidence* the compiled path is correct — but they
exercise the `…Impl` replacement, not the proof body, and are not a substitute for an equivalence proof.
Anything proved about `choleskyFn`/`solveRidgeFn` therefore transfers to `#eval` output only modulo this
unverified hook.
-/

/--
Strict, array-backed runtime implementation of `choleskyColsFn` (registered via `@[implemented_by]`).
Each column is *materialized* into an `Array α`, so a back-reference `L[i,k]` is an `O(1)` lookup
rather than a closure that re-evaluates the whole prefix. The closure form below is mathematically
clean (and is what the proofs reason about), but reading the full factor `L` from it re-evaluates
columns exponentially — ruinous in the interpreter (`#eval`). It is *intended* to compute the same
factor strictly; this equivalence is **trusted, not proved** (see the trust-boundary note above), with
the numeric examples (`A = L·Lᵀ`, the ridge-solve residual ≈ 0) as evidence rather than a proof.
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
correctness proofs reason about. The two are intended to compute the same factor — trusted, not proved;
see the trust-boundary note above.
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
Cholesky factorization candidate for `A`, returning a lower-triangular factor. Over `ℝ`, the proved
reconstruction theorem assumes symmetry and positive executable Cholesky pivots.

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
terms drop out). -/

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

/--
Strict, array-backed runtime implementation of `cholSolveFn` (registered via `@[implemented_by]`).
It materializes `L` into a strict `Array (Array α)` once, then runs both triangular substitutions over
`Array`s, so a back-reference is an `O(1)` lookup. The closure form below (`triSolveUpperFn` over
`triSolveLowerFn`) is mathematically clean — and is what the correctness proofs reason about — but reads
the `Function.update` accumulator chain on every step, which is ruinous in the interpreter (`#eval`) when
`L` is itself an unmaterialized closure (e.g. `choleskyFn` of a kernel matrix). It is *intended* to
compute the same solution strictly; this equivalence is **trusted, not proved** (see the trust-boundary
note above), with the numeric examples (the ridge residual ≈ 0) as evidence rather than a proof. -/
def cholSolveImpl {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  let La : Array (Array α) := Array.ofFn (fun i : Fin n => Array.ofFn (fun j : Fin n => L i j))
  let Lent : Nat → Nat → α := fun i j => (La.getD i #[]).getD j 0
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

/-- Solve `A · x = b` given a Cholesky factor `L` of `A` (so `A = L · Lᵀ`): forward-solve
`L · z = b`, then back-solve `Lᵀ · x = z`.

The runtime implementation is `cholSolveImpl` (strict arrays); the closure form here is what the
correctness proofs reason about. The two are intended to compute the same solution — trusted, not
proved; see the trust-boundary note above. -/
@[implemented_by cholSolveImpl]
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
Intended to be the same linear solve; this equivalence is **trusted, not proved** (see the
trust-boundary note above), with the numeric examples (residual `(K+γ·I)·x − b ≈ 0`) as evidence
rather than a proof.
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
of `K + γ·I`.

The runtime implementation is `solveRidgeImpl` (strict arrays); the closure form here, built from the
`choleskyFn` / `triSolve*` pieces the correctness proofs reason about. The two are intended to compute
the same solution — trusted, not proved; see the trust-boundary note above. -/
@[implemented_by solveRidgeImpl]
def solveRidgeFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  cholSolveFn (choleskyFn (addScaledIdFn K γ)) b

/-- Tensor-level kernel-ridge solve: `(K + γ·I)·x = b`.

PyTorch analogue: `torch.linalg.solve(K + γ·I, b)` (specialized to the SPD Cholesky path). -/
def solveRidgeSpec {n : Nat} (K : Tensor α (.dim n (.dim n .scalar))) (γ : α)
    (b : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  ofVecFn (solveRidgeFn (toMatFn K) γ (toVecFn b))

/-! ## QR factorization (classical Gram–Schmidt)

For `A : m × n`, compute classical Gram–Schmidt factors. Under positive executable `R` pivots, the
proved real theorem gives `A = Q · R`, `Q` with orthonormal columns, and upper-triangular `R`. This
uses **classical** Gram–Schmidt: each `r[k,j] = qₖ · aⱼ` is the inner product against the *original*
column `aⱼ`, and all projections are subtracted in a single pass (modified Gram–Schmidt would instead
dot each `qₖ` against the running residual). In exact real arithmetic the two coincide; the classical
form is what the recurrence below implements and what the reconstruction proof matches.
-/

/-- Internal state for the Gram–Schmidt fold: computed `Q` columns and `R` columns so far. -/
structure GSState (m n : Nat) (α : Type) where
  /-- Orthonormal `Q` columns produced so far (each of length `m`). -/
  qs : List (Fin m → α)
  /-- `R` columns produced so far (each of length `n`, upper-triangular). -/
  rcols : List (Fin n → α)

/--
Run classical Gram–Schmidt over the columns of `A`, returning the `Q` columns and `R` columns.
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

/-- The `Q` factor candidate of the QR factorization of `A`. Its columns are proved orthonormal under
positive executable `R` pivots. -/
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
QR factorization candidate of `A : m × n` via classical Gram–Schmidt. Over `ℝ`, the full
`A = Q · R`, orthonormal-column, and upper-triangular specification is proved under positive
executable `R` pivots.

PyTorch analogue: `torch.linalg.qr(A)`.
-/
def qrSpec {m n : Nat} (A : Tensor α (.dim m (.dim n .scalar))) :
    Tensor α (.dim m (.dim n .scalar)) × Tensor α (.dim n (.dim n .scalar)) :=
  (qrQSpec A, qrRSpec A)

end Spec
