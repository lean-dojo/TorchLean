/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context
public import TorchLean.Tensor.OfFn

/-!
# Matrix factorizations (spec layer)

Shape-indexed recurrences, over any scalar backend, for the two matrix factorizations behind
Gaussian processes, kernel ridge regression, PCA and least squares. The factorization theorems hold
over `ℝ`, and floating-point execution runs the same operations with the backend's rounding:

- `Tensor.cholesky`: Cholesky factorization $A=LL^\mathsf{T}$ (lower-triangular $L$), proved for
  symmetric real matrices with positive executable Cholesky pivots.
- `qr`: QR factorization $A=QR$ via classical Gram–Schmidt. Under positive executable $R$
  pivots, $Q$ has orthonormal columns and $R$ is upper-triangular.

The linear solves that ride on the Cholesky factor sit here too:

- `triSolveLowerFn` and `triSolveUpperFn`: forward and back triangular substitution.
- `cholSolveFn`: solve $Ax=b$ from a Cholesky factor of $A$.
- `Tensor.solveRidge`: the Tikhonov / kernel-ridge solve $(K+\gamma I)x=b$.

## Verification scope

`Tensor.cholesky` and `qr` come with reconstruction and structural theorems (`IsCholesky` /
`IsQR`, lower- and upper-triangularity, orthonormality) in `TorchLean.Tensor.Factorization.*`,
over `ℝ` under positive-pivot success hypotheses and, for Cholesky reconstruction, symmetry. The
triangular and ridge solves follow the standard substitution formulas over the function
representation. Over `ℝ`, `TorchLean.Tensor.Factorization.Solve` proves that they
return exact solutions: `triSolveLowerFn`, `triSolveUpperFn` and `cholSolveFn` for triangular
factors with nonzero diagonal, and `solveRidgeFn K γ b` for $(K+\gamma I)x=b$ with symmetric `K`
when the Cholesky pivots of $K+\gamma I$ are positive. Compiled code runs array versions of
`choleskyColsFn`, `cholSolveFn` and `solveRidgeFn`, and `@[csimp]` theorems prove each equal to
its definition for every scalar type `α` with `[Context α]`, `Float` and `Float32` included. We
prove no floating-point accuracy bound.

## Intent / tradeoffs

Like the rest of the spec layer (`determinant`, `inverse?`, `Tensor.matmul`), these
definitions put **mathematical clarity** and **shape safety** ahead of speed. They are meant for
small and medium matrices and for reference code in proofs, and large-scale numerics belong to the
array-backed runtime kernels.

The algorithms are written over the plain function representation, `Fin n → Fin n → α` for
matrices and `Fin n → α` for vectors, and wrapped back into `TorchLean.Tensor` at the boundary.
This keeps the numerical formulas readable and lets the proofs in
`TorchLean.Tensor.Factorization.*` work on ordinary functions rather than on nested
`Tensor` `match`es.

## Sources and PyTorch

We wrote both factorizations by hand as the textbook recurrences of Golub and Van Loan (2013):
Cholesky column by column, each column from the ones before it, and QR by classical Gram-Schmidt,
which takes each coefficient as a dot product with the original column. A column whose residual
norm isn't positive, or that comes once `min m n` basis columns exist, adds no basis column. Like
torch 2.8.0's `torch.linalg.cholesky`, `Tensor.cholesky` reads the lower triangle and never the
upper one: `[[4, 100], [2, 3]]` and `[[4, 2], [2, 3]]` both give `[[2, 0], [1, 1.4142]]`. The two
part ways at a pivot that isn't positive. The second pivot of `[[1, 2], [2, 1]]` is `1 - 4 = -3`,
and on IEEE scalars we return `[[1, 0], [2, NaN]]`, where torch raises the `RuntimeError` that its
page documents for an input that isn't positive-definite.

Our `R` pivots are norms, so none is negative, while the `torch.linalg.qr` page says "the
elements in the diagonal of R are not necessarily positive". At `[[1, 2], [3, 4]]` we get the
diagonal `3.1623, 0.6325` and torch 2.8.0 gets `-3.1623, -0.6325`, so its `Q` is the negative of
ours up to rounding. On a wide input (`m < n`) torch's default reduced mode returns `Q : [m, m]`
and `R : [m, n]`, and we keep `[m, n]` and `[n, n]`. The positivity test has no tolerance, so in
binary64 the second residual of `[[1, 1], [1, 1]]`, rounding noise of norm `3.1e-16`, enters the
basis and both columns of our `Q` come out as `[0.7071, 0.7071]`. No test compares either
factorization with torch.

## References

- Gene H. Golub and Charles F. Van Loan (2013), *Matrix Computations*, 4th ed., Johns Hopkins
  University Press, ISBN 978-1-4214-0794-4.
- `torch.linalg.cholesky` and `torch.linalg.qr` (PyTorch 2.8 docs), at
  `https://docs.pytorch.org/docs/2.8/generated/torch.linalg.cholesky.html` and
  `https://docs.pytorch.org/docs/2.8/generated/torch.linalg.qr.html`.
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Factorization

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-! ## Small numeric helpers on the function representation -/

/-- Dot product of two coordinate functions, a zero-seeded left fold in increasing index order,
the order in which `Tensor.dot` adds the products of two vectors. -/
def dotFn {α : Type} [Add α] [Mul α] [Zero α]
    {p : Nat} (u v : Fin p → α) : α :=
  (List.finRange p).foldl (fun s i => s + u i * v i) 0

/-- Euclidean norm of a length-`p` vector. -/
def normFn {p : Nat} (v : Fin p → α) : α :=
  MathFunctions.sqrt (dotFn v v)

/-! ## Fold lemmas for the `@[csimp]` theorems

The `@[csimp]` theorems below prove the array functions equal to the closure forms they replace
in compiled code. These private lemmas compare the folds over `List.finRange n` that the two forms
run.
-/

/-- If `R 0 a b` holds and step `j` of the two folds takes `R j` to `R (j + 1)`, then the two left
folds over `List.finRange n` end in `R n`. -/
private theorem _root_.TorchLean.Shape.foldl_finRange_rel {σ τ : Type*} {n : Nat}
    (R : Nat → σ → τ → Prop)
    {f : σ → Fin n → σ} {g : τ → Fin n → τ} {a : σ} {b : τ} (h0 : R 0 a b)
    (hstep : ∀ (j : Fin n) x y, R j.val x y → R (j.val + 1) (f x j) (g y j)) :
    R n ((List.finRange n).foldl f a) ((List.finRange n).foldl g b) := by
  induction n with
  | zero => simpa using h0
  | succ n ih =>
    rw [List.finRange_succ_last, List.foldl_append, List.foldl_append, List.foldl_map,
      List.foldl_map]
    exact hstep (Fin.last n) _ _ (ih (fun j x y h => hstep j.castSucc x y h))

/-- For an array `xs` of size `m ≤ n`, the fold over `List.finRange n` that applies `G` to
`xs.getD k d` when `k < m` and skips `k` otherwise equals `xs.toList.foldl G init`. -/
private theorem foldl_finRange_ite_lt {β γ : Type*} {n m : Nat} (xs : Array β) (hm : xs.size = m)
    (hmn : m ≤ n) (d : β) (G : γ → β → γ) (init : γ) :
    (List.finRange n).foldl (fun s k => if k.val < m then G s (xs.getD k.val d) else s) init
      = xs.toList.foldl G init := by
  subst hm
  suffices h : ∀ p, (List.finRange p).foldl
      (fun s k => if k.val < xs.size then G s (xs.getD k.val d) else s) init
        = (xs.toList.take p).foldl G init by
    rw [h, List.take_of_length_le (by rw [Array.length_toList]; exact hmn)]
  intro p
  induction p with
  | zero => simp
  | succ p ih =>
    rw [List.finRange_succ_last, List.foldl_append, List.foldl_map, List.take_add_one,
      List.foldl_append, Array.getElem?_toList]
    simp only [Fin.val_castSucc, Fin.val_last, List.foldl_cons, List.foldl_nil]
    rw [ih]
    by_cases hp : p < xs.size
    · rw [ite_eq_left hp, Array.getElem?_eq_getElem hp]
      simp [Array.getElem?_eq_getElem hp]
    · rw [ite_eq_right hp, Array.getElem?_eq_none (Nat.le_of_not_lt hp)]
      rfl

/-- Index `i` of `Array.ofFn f` holds `f i`. -/
private theorem getD_ofFn {β : Type*} {n : Nat} (f : Fin n → β) (i : Fin n) (d : β) :
    (Array.ofFn f).getD i.val d = f i := by
  simp

/-- Reading indices `0` to `n - 1` of an array of size `n` gives the list of its entries. -/
private theorem map_finRange_getD {β : Type*} {n : Nat} (xs : Array β) (h : xs.size = n) (d : β) :
    (List.finRange n).map (fun j => xs.getD j.val d) = xs.toList := by
  subst h
  apply List.ext_getElem (by simp)
  intro p h1 h2
  simp

/-! ## Cholesky factorization

For a symmetric real input whose executable Cholesky pivots are positive, the lower-triangular
result satisfies $A=LL^\mathsf{T}$, and those two conditions are the hypotheses of the
reconstruction theorem. On other scalar backends the definition specifies the recurrence only.

The columns are computed left to right, and column `j` reads only columns `0 .. j-1`:

- diagonal: $L_{jj}=\sqrt{A_{jj}-\sum_{k<j}L_{jk}^2}$
- below: $L_{ij}=(A_{ij}-\sum_{k<j}L_{ik}L_{jk})/L_{jj}$ for $i>j$
- above: $L_{ij}=0$ for $i<j$

### Compiled code for `choleskyColsFn`, `cholSolveFn` and `solveRidgeFn`

A `@[csimp]` theorem proves each of these three definitions equal to an array function from
`Factorization`, so compiled code, `#eval` included, runs the array function in its place:

- `choleskyColsFn` runs `choleskyCols`, which reads `choleskyColumnsArray`
  (`choleskyColsFn_eq_choleskyCols`).
- `cholSolveFn` runs `cholSolve`, which stores the entries of `L` once and calls
  `solveCholeskyEntries` (`cholSolveFn_eq_cholSolve`).
- `solveRidgeFn` runs `solveRidge`, which factors $K+\gamma I$ with
  `choleskyColumnsArray` and calls `solveCholeskyEntries` (`solveRidgeFn_eq_solveRidge`).

The proofs reason about the closure forms. The three `@[csimp]` proofs use no law of the
arithmetic, so the equalities hold on `Float` and `Float32` as well as on `ℝ`, and compiled code
returns the value of the closure form on each scalar type. Each array function does the arithmetic
of its closure form in the same order: the Cholesky sums run over the earlier columns, and both
substitutions evaluate the full `n`-term dot, products with zero included. The array functions
store each entry once. Reading an entry of the factor from the closure form of `choleskyColsFn`
evaluates the earlier columns again, at a cost exponential in `n`.
-/

/--
The Cholesky columns of `A` as arrays, computed left to right by the recurrence of `choleskyColsFn`,
so reading `L[i,k]` is an array lookup. It backs compiled `choleskyColsFn` and `solveRidgeFn`, and
read through `Factorization.choleskyCols` its columns are `choleskyColsFn A`
(`choleskyColsFn_eq_choleskyCols`).
-/
def choleskyColumnsArray {n : Nat} (A : Fin n → Fin n → α) : Array (Array α) :=
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
def choleskyCols {n : Nat} (A : Fin n → Fin n → α) : List (Fin n → α) :=
  let cols := Factorization.choleskyColumnsArray A
  (List.finRange n).map (fun j => fun i => (cols.getD j.val #[]).getD i.val 0)

/--
The columns of the Cholesky factor `L` as length-`n` vectors, computed left to right: element `j`
of the result is column `j` of `L`. A left fold builds the list, so when column `j` is formed,
`cols` already holds columns `0 .. j-1`.

The proofs reason about this closure form. Compiled code runs `choleskyCols` instead, which
`choleskyColsFn_eq_choleskyCols` proves equal to it for every scalar type.
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

/-- `choleskyColsFn` equals `choleskyCols` for every scalar type `α` with `[Context α]`.
As a `@[csimp]` lemma it makes compiled code call `choleskyCols`. -/
@[csimp] theorem choleskyColsFn_eq_choleskyCols : @choleskyColsFn = @choleskyCols := by
  funext α _ n A
  obtain ⟨hsize, hcols⟩ : (choleskyColumnsArray A).size = n ∧
      choleskyColsFn A = (choleskyColumnsArray A).toList.map
        (fun col (i : Fin n) => col.getD i.val 0) := by
    unfold choleskyColsFn choleskyColumnsArray
    refine Shape.foldl_finRange_rel (fun m (cA : Array (Array α)) (cL : List (Fin n → α)) =>
      cA.size = m ∧ cL = cA.toList.map (fun col (i : Fin n) => col.getD i.val 0)) ⟨rfl, rfl⟩ ?_
    rintro j cA _ ⟨hj, rfl⟩
    dsimp only
    refine ⟨by rw [Array.size_push, hj], ?_⟩
    rw [Array.toList_push, List.map_append, List.map_singleton]
    congr 2
    funext i
    simp only [getD_ofFn, List.map_map, List.foldl_map, Function.comp_apply]
    rw [foldl_finRange_ite_lt cA hj j.isLt.le #[]
        (fun s (col : Array α) => s + col.getD j.val 0 * col.getD j.val 0) 0,
      foldl_finRange_ite_lt cA hj j.isLt.le #[]
        (fun s (col : Array α) => s + col.getD i.val 0 * col.getD j.val 0) 0]
  rw [choleskyCols, hcols, ← map_finRange_getD _ hsize #[], List.map_map]
  rfl

/-- Cholesky factor as a function:
$L_{ij}=(\mathtt{choleskyColsFn}(A))_j(i)$. -/
def choleskyFn {n : Nat} (A : Fin n → Fin n → α) : Fin n → Fin n → α :=
  let cols := choleskyColsFn A
  fun i j => (cols.getD j.val (fun _ => 0)) i

/--
The lower-triangular Cholesky factor candidate of a square matrix. Over `ℝ` the reconstruction
theorem assumes symmetry and positive executable pivots, and compiled code computes the factor with
`choleskyCols`, which `choleskyColsFn_eq_choleskyCols` proves equal to `choleskyColsFn`.

PyTorch analogue: `torch.linalg.cholesky(A)`.
-/
def _root_.TorchLean.Tensor.cholesky {n : Nat} (matrix : Tensor α [n, n]) : Tensor α [n, n] :=
  Tensor.matrix (choleskyFn (get2 matrix))

/-! ## Triangular solves and the kernel-ridge (Tikhonov) linear solve

Once $A$ is factored as $A=LL^\mathsf{T}$ (Cholesky), the system $Ax=b$ takes two triangular
substitutions: forward-solve $Lz=b$, then back-solve $L^\mathsf{T}x=z$. Each substitution visits
the unknowns in an order where every unknown that row `i` depends on is solved before row `i`. The
accumulator `acc` holds those values and `0` everywhere else, so over `ℝ` the dot
`dotFn (row i) acc` is the required partial sum (the unsolved and structurally zero terms drop
out). On every backend the full `n`-term dot is evaluated, products with zero included. Those terms
cannot be skipped for IEEE inputs, because `∞ * 0` is NaN.
-/

/-- Forward substitution: solve $Ly=b$ for a lower-triangular $L$ with nonzero diagonal.
Unknowns are visited in the order $0,1,\ldots,n-1$. When row $i$ is reached, `acc` holds
$y_0,\ldots,y_{i-1}$ (and $0$ elsewhere), so over `ℝ`
$\mathtt{dotFn}(L_i,\mathtt{acc})=\sum_{k<i}L_{ik}y_k$, which for a lower-triangular $L$ is the
whole off-diagonal part of row $i$. -/
def triSolveLowerFn {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  (List.finRange n).foldl
    (fun acc i => Function.update acc i ((b i - dotFn (L i) acc) / L i i))
    (fun _ => 0)

/-- Back substitution: solve $Ux=y$ for an upper-triangular $U$ with nonzero diagonal.
Unknowns are visited in the order $n-1,\ldots,1,0$. When row $i$ is reached, `acc` holds
$x_{i+1},\ldots,x_{n-1}$ (and $0$ elsewhere), so over `ℝ`
$\mathtt{dotFn}(U_i,\mathtt{acc})=\sum_{k>i}U_{ik}x_k$, which for an upper-triangular $U$ is the
whole off-diagonal part of row $i$. -/
def triSolveUpperFn {n : Nat} (U : Fin n → Fin n → α) (y : Fin n → α) : Fin n → α :=
  (List.finRange n).reverse.foldl
    (fun acc i => Function.update acc i ((y i - dotFn (U i) acc) / U i i))
    (fun _ => 0)

/--
Forward substitution with `L`, then back substitution with `Lᵀ`, over arrays, where `entry i j`
reads `L[i,j]`: the arithmetic of `triSolveLowerFn` and `triSolveUpperFn` in the same order. It
backs compiled `cholSolveFn` and `solveRidgeFn`, and on every scalar type it equals
`cholSolveFn L b` when `entry i j = L i j` for all `i j : Fin n`.
-/
def solveCholeskyEntries (n : Nat) (entry : Nat → Nat → α)
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
def cholSolve {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  let entries := Array.ofFn fun i : Fin n => Array.ofFn fun j : Fin n => L i j
  Factorization.solveCholeskyEntries n (fun i j => (entries.getD i #[]).getD j 0) b

/-- Solve $Ax=b$ given a Cholesky factor $L$ of $A$ (so $A=LL^\mathsf{T}$): forward-solve
$Lz=b$, then back-solve $L^\mathsf{T}x=z$.

Compiled code runs `cholSolve` instead, which `cholSolveFn_eq_cholSolve` proves equal to
it for every scalar type. Over `ℝ`, `mulVec_cholSolveFn` proves $L(L^\mathsf{T}x)=b$ for
lower-triangular $L$ with nonzero diagonal. -/
def cholSolveFn {n : Nat} (L : Fin n → Fin n → α) (b : Fin n → α) : Fin n → α :=
  triSolveUpperFn (fun i k => L k i) (triSolveLowerFn L b)

omit [Storage α] in
/-- `solveCholeskyEntries n entry b` equals `cholSolveFn L b` when `entry i j = L i j`
for all `i j : Fin n`. -/
private theorem solveCholeskyEntries_eq {n : Nat} (entry : Nat → Nat → α) (L : Fin n → Fin n → α)
    (hentry : ∀ i j : Fin n, entry i.val j.val = L i j) (b : Fin n → α) :
    solveCholeskyEntries n entry b = cholSolveFn L b := by
  unfold solveCholeskyEntries cholSolveFn triSolveUpperFn
  extract_lets z x
  have hz : z.size = n ∧ ∀ k : Fin n, z.getD k.val 0 = triSolveLowerFn L b k := by
    unfold triSolveLowerFn
    simp only [z]
    refine Shape.foldl_finRange_rel (fun m (z : Array α) (acc : Fin n → α) =>
      z.size = m ∧ ∀ k : Fin n, z.getD k.val 0 = acc k) ⟨rfl, fun _ => rfl⟩ ?_
    rintro j z acc ⟨hzs, hza⟩
    have hs : (List.finRange n).foldl (fun s k => s + entry j.val k.val * z.getD k.val 0) 0
        = dotFn (L j) acc := by
      unfold dotFn
      simp only [hentry, hza]
    refine ⟨by rw [Array.size_push, hzs], fun k => ?_⟩
    rw [hs, hentry]
    by_cases hk : k = j
    · subst hk
      rw [Function.update_self, Array.getD_eq_getD_getElem?, Array.getElem?_push,
        ite_eq_left hzs.symm]
      rfl
    · rw [Function.update_of_ne hk, ← hza k, Array.getD_eq_getD_getElem?, Array.getElem?_push,
        ite_eq_right (fun h => hk (Fin.ext (h.trans hzs))), ← Array.getD_eq_getD_getElem?]
  funext k
  simp only [x]
  rw [List.finRange_reverse, List.foldl_map, List.foldl_map]
  refine (Shape.foldl_finRange_rel (fun _ (xs : Array α) (acc : Fin n → α) =>
      xs.size = n ∧ ∀ k : Fin n, xs.getD k.val 0 = acc k) ⟨by simp, fun k => by simp⟩ ?_).2 k
  rintro j xs acc ⟨hxs, hxa⟩
  have hs : (List.finRange n).foldl (fun s k => s + entry k.val j.rev.val * xs.getD k.val 0) 0
      = dotFn (fun k => L k j.rev) acc := by
    unfold dotFn
    simp only [hentry, hxa]
  refine ⟨by simp [hxs], fun k => ?_⟩
  rw [hs, hentry, hz.2, Array.set!_eq_setIfInBounds]
  by_cases hk : k = j.rev
  · subst hk
    rw [Function.update_self, Array.getD_eq_getD_getElem?, Array.getD_getElem?_setIfInBounds,
      ite_eq_left (hxs ▸ j.rev.isLt)]
  · rw [Function.update_of_ne hk, ← hxa k, Array.getD_eq_getD_getElem?,
      Array.getElem?_setIfInBounds_ne (fun h => hk (Fin.ext h).symm), ← Array.getD_eq_getD_getElem?]

/-- `cholSolveFn` equals `cholSolve` for every scalar type `α` with `[Context α]`.
As a `@[csimp]` lemma it makes compiled code call `cholSolve`. -/
@[csimp] theorem cholSolveFn_eq_cholSolve : @cholSolveFn = @cholSolve := by
  funext α _ n L b
  unfold cholSolve
  exact (solveCholeskyEntries_eq _ L (fun i j => by simp only [getD_ofFn]) b).symm

/-- The regularized matrix $K+\gamma I$ as a function. Over `ℝ`, without backend overflow or
rounding, a symmetric PSD kernel $K$ and $\gamma>0$ give a symmetric positive-definite matrix. -/
def addScaledIdFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) : Fin n → Fin n → α :=
  fun i j => K i j + (if i = j then γ else 0)

/--
The compiled code of `solveRidgeFn`: the Cholesky columns of $K+\gamma I$ from
`Factorization.choleskyColumnsArray`, then `Factorization.solveCholeskyEntries`, all over arrays.
`solveRidgeFn_eq_solveRidge` proves it equal to `solveRidgeFn` for every scalar type.
-/
def solveRidge {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  let columns := Factorization.choleskyColumnsArray (addScaledIdFn K γ)
  Factorization.solveCholeskyEntries n (fun i j => (columns.getD j #[]).getD i 0) b

/-- The Tikhonov-regularized (kernel-ridge) solve $(K+\gamma I)x=b$, via the Cholesky factorization
of $K+\gamma I$.

This closure form composes `choleskyFn` with `cholSolveFn`, and compiled code runs
`solveRidge` instead, which `solveRidgeFn_eq_solveRidge` proves equal to it for every
scalar type. Over `ℝ`, `mulVec_solveRidgeFn` proves the solution exact for symmetric `K` when the
Cholesky pivots of $K+\gamma I$ are positive. -/
def solveRidgeFn {n : Nat} (K : Fin n → Fin n → α) (γ : α) (b : Fin n → α) : Fin n → α :=
  cholSolveFn (choleskyFn (addScaledIdFn K γ)) b

/-- `solveRidgeFn` equals `solveRidge` for every scalar type `α` with `[Context α]`.
As a `@[csimp]` lemma it makes compiled code call `solveRidge`. -/
@[csimp] theorem solveRidgeFn_eq_solveRidge : @solveRidgeFn = @solveRidge := by
  funext α _ n K γ b
  unfold solveRidgeFn solveRidge
  refine (solveCholeskyEntries_eq _ _ (fun i j => ?_) b).symm
  simp only [choleskyFn, choleskyColsFn_eq_choleskyCols, choleskyCols]
  simp

/-- Solve `(kernel + regularization * I) x = target` through the Cholesky path, for symmetric
positive-semidefinite kernels and positive regularization. Over `ℝ`, `solveRidge_spec` proves the
solution exact for a symmetric kernel when the Cholesky pivots of `kernel + regularization * I` are
positive. Compiled code runs `solveRidge`, proved equal to `solveRidgeFn` by
`solveRidgeFn_eq_solveRidge`.

PyTorch analogue: `torch.linalg.solve(K + gamma * I, b)` (specialized to the SPD Cholesky path). -/
def _root_.TorchLean.Tensor.solveRidge {n : Nat} (kernel : Tensor α [n, n]) (regularization : α)
    (target : Tensor α [n]) : Tensor α [n] :=
  Tensor.ofFn (solveRidgeFn (get2 kernel) regularization (Tensor.getScalar target))

/-! ## QR factorization (classical Gram–Schmidt)

For $A\in\mathbb{R}^{m\times n}$ we compute the classical Gram–Schmidt factors. The method is
**classical** Gram–Schmidt: each $r_{kj}=q_k^\mathsf{T}a_j$ is the inner product with the *original*
column $a_j$, and all projections are subtracted in one pass (modified Gram–Schmidt dots each $q_k$
against the running residual). The two coincide in exact real arithmetic, and both the recurrence
below and the reconstruction proof follow the classical form.

A column enters the basis when its residual norm is positive and the basis has fewer than `min m n`
columns. Any other column, a NaN norm included, takes no basis slot: its `R` column holds only its
coefficients against the basis built so far, so a later independent column can enter. Under
positive executable $R$ pivots every column enters, and the proved real theorem gives $A=QR$ with
orthonormal columns in $Q$ and an upper-triangular $R$.
-/

/-- Internal state for the Gram–Schmidt fold: the basis columns and the `R` columns so far. -/
structure GSState (m n : Nat) (α : Type) where
  /-- The basis columns produced so far, at most `min m n` of them, each of length `m`. -/
  qs : List (Fin m → α)
  /-- One `R` column per source column processed so far, each of length `n`. -/
  rcols : List (Fin n → α)

/--
Classical Gram–Schmidt over the columns of `A`, returning the basis columns and one `R` column per
source column. Column `j` is orthogonalized against the basis columns already produced and enters
the basis, divided by its residual norm, when that norm is positive and fewer than `min m n` basis
columns exist.
-/
def gramSchmidtFn {m n : Nat} (A : Fin m → Fin n → α) : GSState m n α :=
  (List.finRange n).foldl (fun (st : GSState m n α) j =>
    let a : Fin m → α := fun i => A i j
    -- r[k,j] = qₖ · a   for each basis column k built so far
    let rkjs : List α := st.qs.map (fun qk => dotFn qk a)
    -- v = a - Σ r[k,j] qₖ, stored in a tensor so that later reads do not recompute it
    let v : Fin m → α := Tensor.getScalar <| Tensor.ofFn fun i =>
      a i - (List.zip st.qs rkjs).foldl (fun acc (qk, r) => acc + r * qk i) 0
    let rjj := normFn v
    let count := st.qs.length
    let append := Context.gtBool rjj 0 && decide (count < min m n)
    let qj : Fin m → α := Tensor.getScalar <| Tensor.ofFn fun i => v i / rjj
    let rcolj : Fin n → α := fun k =>
      if k.val < count then rkjs.getD k.val 0
      else if k.val == count && append then rjj
      else 0
    { qs := if append then st.qs ++ [qj] else st.qs, rcols := st.rcols ++ [rcolj] })
    { qs := [], rcols := [] }

/--
QR factorization candidate of $A\in\mathbb{R}^{m\times n}$ via classical Gram–Schmidt: `Q` holds
the basis columns followed by zero columns, and `R` holds one coefficient column per source column.
Over `ℝ`, the full $A=QR$, orthonormal-column, and upper-triangular specification is proved under
positive executable $R$ pivots.

PyTorch analogue: `torch.linalg.qr(A)`.
-/
def qr {m n : Nat} (A : Tensor α [m, n]) :
    Tensor α [m, n] × Tensor α [n, n] :=
  let st := gramSchmidtFn (get2 A)
  let Q := Tensor.matrix (fun i j => (st.qs.getD j.val (fun _ => 0)) i)
  let R := Tensor.matrix (fun k j => (st.rcols.getD j.val (fun _ => 0)) k)
  (Q, R)

end TorchLean.Factorization
