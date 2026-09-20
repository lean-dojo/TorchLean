/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Constructors

/-!
# Matrix factorizations (spec layer)

This file provides **real**, shape-indexed reference implementations of the two *exact, finite*
matrix factorizations used by Gaussian processes, kernel ridge regression, PCA, and least squares:

- `choleskySpec`: Cholesky factorization $A=LL^\mathsf{T}$ (lower-triangular $L$), proved for
                     matrices with positive executable Cholesky pivots.
- `qrSpec`: QR factorization $A=QR$ via classical Gram–Schmidt; under positive executable
                     $R$ pivots, $Q$ has orthonormal columns and $R$ is upper-triangular.

It also provides the linear solves that ride on the Cholesky factor:

- `triSolveLowerFn` and `triSolveUpperFn`: forward and back triangular substitution;
- `cholSolveFn`: solve $Ax=b$ from a Cholesky factor of $A$;
- `solveRidgeSpec`: the Tikhonov / kernel-ridge solve $(K+\gamma I)x=b$.

## Verification scope

The **verified** contribution is the factorizations: `choleskySpec` / `qrSpec` come with
reconstruction and structural theorems (`IsCholesky` / `IsQR`, lower- and upper-triangularity,
orthonormality) in `NN.Proofs.Tensor.Basic.Factorizations*`, under their stated positive-pivot
success hypotheses. The triangular- and ridge-solve helpers above (`triSolveLowerFn`,
`triSolveUpperFn`, `cholSolveFn`, `solveRidgeSpec`) are **executable APIs only**: their
correctness has not been proved. They follow the standard substitution formulas over the
readable function representation
and are exercised by `#eval` examples, but should not be read as carrying a verified-correctness
guarantee.

## Intent / tradeoffs

Like the rest of the spec layer (`determinantSpec`, `inverseSpec?`, `matMulSpec`), these prioritize
**mathematical clarity** and **shape safety** over performance, and are intended for small/medium
matrices and proof-oriented reference code. For large-scale numerics, use array-backed runtime
kernels.

Internally the algorithms are written over the plain function representation
`Fin n → Fin n → α` (matrices) and `Fin n → α` (vectors), then wrapped back into `TorchLean.Tensor`
at the boundary. This keeps the numerical formulas readable and keeps later correctness proofs
working on ordinary functions rather than on nested `Tensor` `match`es.
-/

@[expose] public section


open TorchLean

namespace Spec

open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-! ## Boundary conversions between `TorchLean.Tensor` and plain functions -/

/-- View a matrix tensor as a function `Fin m → Fin n → α`. -/
def toMatFn {m n : Nat} (A : Tensor α [m, n]) : Fin m → Fin n → α :=
  fun i j => get2 A i j

/-- View a vector tensor as a function `Fin n → α`. -/
def getScalarFn {n : Nat} (v : Tensor α [n]) : Fin n → α :=
  fun i => Tensor.item (get v i)

/-! ## Small numeric helpers on the function representation -/

/-- Dot product of two length-`p` vectors. -/
def dotFn {p : Nat} (u v : Fin p → α) : α :=
  (List.finRange p).foldl (fun s i => s + u i * v i) 0

/-- Euclidean norm of a length-`p` vector. -/
def normFn {p : Nat} (v : Fin p → α) : α :=
  MathFunctions.sqrt (dotFn v v)

/-! ## Cholesky factorization

For an input whose executable Cholesky pivots are positive, compute the lower-triangular `L` with
$A=LL^\mathsf{T}$. Symmetric positive-definiteness is the standard sufficient condition, but the
theorem in this file family is stated against the executable positive-pivot success condition.

The columns are computed left to right. Column `j` uses only columns `0 .. j-1`:

- diagonal: $L_{jj}=\sqrt{A_{jj}-\sum_{k<j}L_{jk}^2}$
- below: $L_{ij}=(A_{ij}-\sum_{k<j}L_{ik}L_{jk})/L_{jj}$ for $i>j$
- above: $L_{ij}=0$ for $i<j$

### Trust boundary: the `@[implemented_by]` performance hooks

Several defs here (`choleskyColsFn`, `cholSolveFn`, `solveRidgeFn`) carry an `@[implemented_by]`
attribute. The clean closure form is what the correctness proofs reason about; the internal
companion is a strict, array-backed rewrite that the compiler runs instead, so `#eval` stays fast
(the closure form re-evaluates prefixes exponentially in the interpreter).

**This substitution is a trusted runtime boundary.** Compiled `#eval` and runtime code execute the
internal array-backed body while the proofs constrain the clean closure body. The two transcribe the
same recurrence, and the
numeric examples in `NN/Examples/Factorization` exercise the compiled path, but a future equivalence
theorem should discharge this boundary explicitly. Anything proved about `choleskyFn`/`solveRidgeFn`
therefore transfers to `#eval` output only modulo this
unverified hook.
-/

/--
Shared strict Cholesky recurrence used by factorization and ridge solving.
Each column is materialized into an `Array α`, so a back-reference `L[i,k]`
is an `O(1)` lookup rather than a closure that re-evaluates the whole prefix. The closure form below
is mathematically clean (and is what the proofs reason about), but reading the full factor `L` from
it re-evaluates columns exponentially, which is ruinous in the interpreter (`#eval`). It is
*intended* to compute the same factor strictly; this equivalence is **trusted, not proved** (see the
trust-boundary note above), with the numeric examples ($A=LL^\mathsf{T}$ and ridge-solve residual
$\approx0$) as evidence rather than a proof.
-/
def Internal.choleskyColumnsArray {n : Nat} (A : Fin n → Fin n → α) : Array (Array α) :=
  (List.finRange n).foldl (fun cols j =>
    let jv := j.val
    -- Σ_{k<j} L[j,k]²  (previous columns at row `j`, read from the materialized arrays).
    let sumsq := (List.finRange n).foldl
      (fun s k =>
        if k.val < jv then s + (cols.getD k.val #[]).getD jv 0 * (cols.getD k.val #[]).getD jv 0
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

/-- View the strictly materialized Cholesky columns as finite functions. -/
def Internal.choleskyCols {n : Nat} (A : Fin n → Fin n → α) : List (Fin n → α) :=
  let cols := Internal.choleskyColumnsArray A
  (List.finRange n).map (fun j => fun i => (cols.getD j.val #[]).getD i.val 0)

/--
The list of columns of the Cholesky factor `L`, as length-`n` vectors, computed left to right.
Element `j` of the result is column `j` of `L`. Built by a left fold so that when column `j` is
formed, `cols` already holds columns `0 .. j-1`.

The runtime implementation is `Internal.choleskyCols` (strict arrays); the closure form here is the
one used by the correctness proofs. The two are intended to compute the same factor. Their
equivalence remains part of the trust boundary described above.
-/
@[implemented_by Internal.choleskyCols]
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

/-- Cholesky factor as a function:
$L_{ij}=(\mathtt{choleskyColsFn}(A))_j(i)$. -/
def choleskyFn {n : Nat} (A : Fin n → Fin n → α) : Fin n → Fin n → α :=
  let cols := choleskyColsFn A
  fun i j => (cols.getD j.val (fun _ => 0)) i

/--
Cholesky factorization candidate for `A`, returning a lower-triangular factor. Over `ℝ`, the proved
reconstruction theorem assumes symmetry and positive executable Cholesky pivots.

PyTorch analogue: `torch.linalg.cholesky(A)`.
-/
def choleskySpec {n : Nat} (A : Tensor α [n, n]) :
    Tensor α [n, n] :=
  Tensor.matrix (choleskyFn (toMatFn A))

/-! ## Triangular solves and the kernel-ridge (Tikhonov) linear solve

Once $A$ is factored as $A=LL^\mathsf{T}$ (Cholesky), the linear system $Ax=b$ is solved by two
triangular substitutions: forward-solve $Lz=b$, then back-solve $L^\mathsf{T}x=z$. Each substitution
visits the unknowns in an order such that, when row `i` is reached, every unknown it depends on has
already been computed; the accumulator `acc` holds those values and `0` everywhere else, so the dot
`dotFn (row i) acc` is exactly the required partial sum (the not-yet-solved and structurally-zero
terms drop out). -/

/-- Forward substitution: solve $Ly=b$ for a lower-triangular $L$ with nonzero diagonal.
Unknowns are visited $0,1,\ldots,n-1$; when row $i$ is reached, `acc` holds
$y_0,\ldots,y_{i-1}$ (and $0$ elsewhere), so
$\mathtt{dotFn}(L_i,\mathtt{acc})=\sum_{k<i}L_{ik}y_k$ by lower-triangularity. -/
def triSolveLowerFn {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  (List.finRange n).foldl
    (fun acc i => Function.update acc i ((b i - dotFn (L i) acc) / L i i))
    (fun _ => 0)

/-- Back substitution: solve $Ux=y$ for an upper-triangular $U$ with nonzero diagonal.
Unknowns are visited $n-1,\ldots,1,0$; when row $i$ is reached, `acc` holds
$x_{i+1},\ldots,x_{n-1}$ (and $0$ elsewhere), so
$\mathtt{dotFn}(U_i,\mathtt{acc})=\sum_{k>i}U_{ik}x_k$ by upper-triangularity. -/
def triSolveUpperFn {n : Nat} (U : Fin n → Fin n → α) (y : Fin n → α) : Fin n → α :=
  (List.finRange n).reverse.foldl
    (fun acc i => Function.update acc i ((y i - dotFn (U i) acc) / U i i))
    (fun _ => 0)

/--
Shared forward and backward substitution loops, given a constant-time factor entry reader.
`Internal.cholSolve` supplies a materialized row array; `Internal.solveRidge` supplies the
materialized columns from the shared Cholesky recurrence. The closure form below (`triSolveUpperFn`
over `triSolveLowerFn`) is mathematically clean, and is what the correctness proofs reason about,
but reads the `Function.update` accumulator chain on every step, which is ruinous in the interpreter
(`#eval`) when `L` is itself an unmaterialized closure (e.g. `choleskyFn` of a kernel matrix). It is
*intended* to compute the same solution strictly; this equivalence is **trusted, not proved** (see
the trust-boundary note above), with the numeric examples (the ridge residual $\approx0$) as
evidence rather than a proof. -/
def Internal.solveCholeskyEntries (n : Nat) (entry : Nat → Nat → α)
    (b : Fin n → α) : Fin n → α :=
  -- Keep the full dot-product order from triSolveLowerFn, including zero-initialized entries.
  -- Dropping those terms would change IEEE behavior for infinite coefficients and signed zeros.
  let z : Array α := (List.finRange n).foldl (fun z i =>
    let iv := i.val
    let s := (List.finRange n).foldl
      (fun acc k => acc + entry iv k.val * z.getD k.val 0) 0
    z.push ((b i - s) / entry iv iv)) #[]
  -- Back solve `Lᵀ · x = z`: `x[i] = (z[i] − Σ_{k>i} L[k,i]·x[k]) / L[i,i]`, `i = n−1 … 0`.
  let x : Array α := (List.finRange n).reverse.foldl (fun xs i =>
    let iv := i.val
    let s := (List.finRange n).foldl
      (fun acc k => acc + entry k.val iv * xs.getD k.val 0) 0
    xs.set! iv ((z.getD iv 0 - s) / entry iv iv)) (Array.replicate n 0)
  fun i => x.getD i.val 0

/-- Materialize a Cholesky factor once, then use the shared strict substitution loops. -/
def Internal.cholSolve {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  let entries := Array.ofFn fun i : Fin n => Array.ofFn fun j : Fin n => L i j
  Internal.solveCholeskyEntries n (fun i j => (entries.getD i #[]).getD j 0) b

/-- Solve $Ax=b$ given a Cholesky factor $L$ of $A$ (so $A=LL^\mathsf{T}$): forward-solve
$Lz=b$, then back-solve $L^\mathsf{T}x=z$.

The runtime implementation is `Internal.cholSolve` (strict arrays); the closure form here is the
one used by the correctness proofs. Their equivalence remains part of the trust boundary described
above. -/
@[implemented_by Internal.cholSolve]
def cholSolveFn {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  triSolveUpperFn (fun i k => L k i) (triSolveLowerFn L b)

/-- The regularized matrix $K+\gamma I$ as a function. For a symmetric PSD kernel $K$ and
$\gamma>0$,
this is symmetric positive-definite, so its Cholesky factorization succeeds. -/
def addScaledIdFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) : Fin n → Fin n → α :=
  fun i j => K i j + (if i = j then γ else 0)

/--
Strict, array-backed runtime implementation of `solveRidgeFn` (registered via `@[implemented_by]`).
It factors $K+\gamma I=LL^\mathsf{T}$ and runs both triangular substitutions entirely over `Array`s,
so no step materializes the deep `Fin n → α` closures the functional definition builds; those
re-evaluate columns / the substitution accumulator exponentially, which is ruinous in the
interpreter (`#eval`). Intended to be the same linear solve; this equivalence is **trusted, not
proved** (see the trust-boundary note above), with the numeric examples (residual $(K+\gamma
I)x-b\approx0$) as evidence rather than a proof.
-/
def Internal.solveRidge {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  let columns := Internal.choleskyColumnsArray (addScaledIdFn K γ)
  Internal.solveCholeskyEntries n (fun i j => (columns.getD j #[]).getD i 0) b

/-- The Tikhonov-regularized (kernel-ridge) solve $(K+\gamma I)x=b$, via the Cholesky factorization
of $K+\gamma I$.

The runtime implementation is `Internal.solveRidge` (strict arrays); the closure form here is built
from the `choleskyFn` and `triSolve*` definitions used by the correctness proofs. Their equivalence
remains part of the trust boundary described above. -/
@[implemented_by Internal.solveRidge]
def solveRidgeFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  cholSolveFn (choleskyFn (addScaledIdFn K γ)) b

/-- Tensor-level kernel-ridge solve: $(K+\gamma I)x=b$.

PyTorch analogue: `torch.linalg.solve(K + gamma * I, b)` (specialized to the SPD Cholesky path). -/
def solveRidgeSpec {n : Nat} (K : Tensor α [n, n]) (γ : α)
    (b : Tensor α [n]) : Tensor α [n] :=
  Tensor.ofFn (solveRidgeFn (toMatFn K) γ (getScalarFn b))

/-! ## QR factorization (classical Gram–Schmidt)

For $A\in\mathbb{R}^{m\times n}$, compute classical Gram–Schmidt factors. Under positive executable
$R$ pivots, the proved real theorem gives $A=QR$, with $Q$ having orthonormal columns and $R$
upper-triangular. This uses **classical** Gram–Schmidt: each $r_{kj}=q_k^\mathsf{T}a_j$ is the inner
product against the *original* column $a_j$, and all projections are subtracted in a single pass
(modified Gram–Schmidt would instead dot each $q_k$ against the running residual). In exact real
arithmetic the two coincide; the classical
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

/-- The `Q` factor candidate of the QR factorization of `A`. Its columns are proved orthonormal
under positive executable `R` pivots. -/
def qrQSpec {m n : Nat} (A : Tensor α [m, n]) :
    Tensor α [m, n] :=
  let st := gramSchmidtFn (toMatFn A)
  Tensor.matrix (fun i j => (st.qs.getD j.val (fun _ => 0)) i)

/-- The `R` factor (upper-triangular) of the QR factorization of `A`. -/
def qrRSpec {m n : Nat} (A : Tensor α [m, n]) :
    Tensor α [n, n] :=
  let st := gramSchmidtFn (toMatFn A)
  Tensor.matrix (fun k j => (st.rcols.getD j.val (fun _ => 0)) k)

/--
QR factorization candidate of $A\in\mathbb{R}^{m\times n}$ via classical Gram–Schmidt. Over `ℝ`,
the full $A=QR$, orthonormal-column, and upper-triangular specification is proved under positive
executable $R$ pivots.

PyTorch analogue: `torch.linalg.qr(A)`.
-/
def qrSpec {m n : Nat} (A : Tensor α [m, n]) :
    Tensor α [m, n] × Tensor α [n, n] :=
  let st := gramSchmidtFn (toMatFn A)
  let Q := Tensor.matrix (fun i j => (st.qs.getD j.val (fun _ => 0)) i)
  let R := Tensor.matrix (fun k j => (st.rcols.getD j.val (fun _ => 0)) k)
  (Q, R)

/-- The first component of the QR pair is the orthonormal factor. -/
@[simp] theorem qrSpec_fst {m n : Nat} (A : Tensor α [m, n]) :
    (qrSpec A).1 = qrQSpec A := by
  rfl

/-- The second component is the upper-triangular factor.

The pair is returned as a plain product rather than a structure, so these two projections are what
callers rewrite with instead of destructuring it. -/
@[simp] theorem qrSpec_snd {m n : Nat} (A : Tensor α [m, n]) :
    (qrSpec A).2 = qrRSpec A := by
  rfl

end Spec
