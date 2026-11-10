import VersoManual
import TorchLean.Tensor.Factorization.Correctness
import TorchLean.Tensor.Factorization.Orthonormal
import TorchLean.Tensor.Factorization.Reconstruction
import TorchLean.Tensor.Einops.Elab.Einsum
import TorchLean.Tensor.Einops.Elab.Pack
import TorchLean.Tensor.Literal
import TorchLean.Tensor.Einops.Elab.Transform
import TorchLean.Tensor.Tactic.Einops.Report
import TorchLean.Tensor.LinearAlgebra
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.Tensor.Constructors

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The executable tensor API, the two predicates, and the Gram-Schmidt proof layer live in three
-- namespaces. Opening them keeps each displayed `#check` on a line Verso's narrow code column can
-- hold, and it also makes the printed signatures read the way a caller would write them.
open TorchLean
open Factorization
open Tensor

-- Several signatures below print wider than this file's 100-column limit, so their `leanOutput`
-- blocks ask for `whitespace := lax` and are wrapped in the source. The rendered page still shows
-- each message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Cholesky and QR" =>
%%%
tag := "factorizations-cholesky-qr"
file := "Matrix-Factorizations___-Cholesky-and-QR"
%%%

A numerical factorization routine may return arrays of the expected sizes and still fail its
algebraic requirements: a Cholesky factor could
contain entries above the diagonal, a QR routine could reconstruct the input with a non-orthogonal
`Q`, or a zero pivot could be hidden behind a NaN. Shape safety catches none of these errors.

TorchLean therefore gives each factorization two parts:

- a public executable operation, `Tensor.cholesky` or `Tensor.qr`, that constructs tensors of a
  fixed shape
- a proposition describing the algebraic object that the program must return.

Application code uses the public `Tensor.*` factorization API, and the examples below also inspect
the generic `Factorization.*` definitions that describe the arithmetic recurrences. Their
real-valued instances are the objects of the exact theorems, while a floating-point evaluation is
a separate claim. We examine later how the runtime overrides of the Cholesky-based operations
relate to the specification.

For Cholesky, the proposition is

$$`\operatorname{IsCholesky}(A,L)
\;:\!\iff\;
\bigl(\forall i<j,\;L_{ij}=0\bigr)
\land A=LL^\top.`

For a tall or square $`m\times k` matrix, the proof-side QR proposition is

$$`\operatorname{IsQR}(A,Q,R)
\;:\!\iff\;
Q^\top Q=I
\land\bigl(\forall j<i,\;R_{ij}=0\bigr)
\land A=QR.`

The two propositions separate properties a caller may need for different reasons. In a covariance
model, Cholesky reconstruction connects the factor to the covariance, while triangularity makes
substitution possible. For least squares, QR reconstruction connects the factors to the design
matrix, and orthonormality lets one preserve Euclidean lengths in the column coordinates. A small
reconstruction residual alone does not establish that second fact. Also, `IsCholesky` as defined
here does not demand a positive diagonal: positivity belongs to the algorithm theorem's premises.
Keeping the predicate distinct from the sufficient conditions prevents us from reading an
algorithm-specific convention into every candidate factor.

The Lean blocks on this page are elaborated while the book is built. Python and shell transcripts
are separate executable examples, and the Lean document compiler doesn't check them.

# Dot Products And Norms

Both algorithms repeatedly compute inner products. Gram-Schmidt uses them to subtract projections
onto earlier columns, while Cholesky subtracts inner products of partial rows before computing a
diagonal square root. Start with the dot product and norm that implement these steps.

These vector operations live on the plain function representation `Fin p → α`, not on tensors:

```lean (name := dotSig)
-- Inspect the coordinate-function interface shared by
-- projection coefficients and residual norms.
#check @Factorization.dotFn
#check @Factorization.normFn
```

```leanOutput dotSig (whitespace := lax)
@dotFn : {α : Type} → [Add α] → [Mul α] → [Zero α] → {p : ℕ} →
  (Fin p → α) → (Fin p → α) → α
```

```leanOutput dotSig
@normFn : {α : Type} → [Context α] → {p : ℕ} → (Fin p → α) → α
```

`dotFn` needs only addition, multiplication, and zero. The norm also needs `MathFunctions.sqrt`,
supplied here through TorchLean's scalar backend class `Context α`. Instantiating `α := Float`
gives a program you can run, and instantiating `α := ℝ` gives an object you can prove things about.
That single generic definition is why the proofs and the executable path aren't two independent
transcriptions of the same recurrence.

In the body of `dotFn`, a left fold fixes the accumulation order:

```
-- Accumulate products in index order; floating addition
-- cannot freely reassociate this fold.
def dotFn {p : Nat} (u v : Fin p → α) : α :=
  (List.finRange p).foldl (fun s i => s + u i * v i) 0
```

A mathematician would write $`\sum_i u_iv_i`. The generic executable definition instead fixes a
summation order. `Float` has no `AddCommMonoid` instance supporting the usual `Finset.sum`:
rounded addition need not be associative, so $`(a+b)+c` and $`a+(b+c)` can differ
{Informal.citep goldberg1991}[]. The left fold specifies which result to compute. Over the reals,
associativity allows a theorem to identify that fold with the mathematical sum:

```lean (name := dotBridge)
-- Over reals, identify the fixed fold with the finite sum
-- used in matrix algebra.
#check @dotFn_eq_sum
```

```leanOutput dotBridge
@dotFn_eq_sum : ∀ {p : ℕ} (u v : Fin p → ℝ), dotFn u v = ∑ i, u i * v i
```

The theorem is specific to `ℝ` and doesn't remove the significance of summation order in a
floating-point run. To evaluate the fold, read the indexed coordinates of two length-three tensors:

```lean (name := dotEval)
-- Read three tensor coordinates as functions and evaluate
-- their dot product.
def uVec : Fin 3 → Float :=
  Tensor.getScalar ([1.0, 2.0, 3.0] : Tensor Float [3])

def vVec : Fin 3 → Float :=
  Tensor.getScalar ([4.0, 5.0, 6.0] : Tensor Float [3])

#eval Factorization.dotFn uVec vVec
```

```leanOutput dotEval
32.000000
```

That is $`1\cdot4+2\cdot5+3\cdot6=4+10+18=32`, and the norm of `uVec` is $`\sqrt{14}`:

```lean (name := normEval)
-- The norm takes the square root of the self-dot product,
-- whose value here is fourteen.
#eval Factorization.normFn uVec
```

```leanOutput normEval
3.741657
```

At the tensor level the same product has a shaped API, which is the one ordinary programs call.
Because the shape is in the type, the two arguments cannot have different lengths:

```lean (name := tensorDot)
-- Use the tensor API to contract matching shapes without
-- manually extracting coordinates.
def uTen : Tensor Float [3] := Tensor.from #[1.0, 2.0, 3.0]
def vTen : Tensor Float [3] := Tensor.from #[4.0, 5.0, 6.0]

#eval Tensor.dot uTen vTen
```

```leanOutput tensorDot
32.000000
```

```lean (name := tensorDotSig)
-- The shared shape parameter forces both arguments to have
-- exactly the same dimensions.
#check @Tensor.dot
```

```leanOutput tensorDotSig (whitespace := lax)
@dot : {α : Type} → [inst : Storage α] →
  [Add α] → [Mul α] → [Zero α] → {s : Shape} →
  Tensor α s → Tensor α s → α
```

The shape parameter is an arbitrary `Shape`, not `[n]`, because the definition is
`Tensor.sum (a * b)`: it contracts every scalar leaf. On `[n]` that is the vector dot product, and
on `[m, n]` it is the Frobenius inner product of two matrices.

In the signature, the last `α` is the scalar result type, so no shape remains on the returned
value. Both earlier occurrences of `Tensor α s` share one implicit `s`: Lean infers it from the
first tensor and checks the second against it. Two differently shaped tensors can't be substituted
merely because they contain the same number of scalars. An explicit reshape would be needed to
describe which coordinates are to be paired.

Two more bridge lemmas connect the real operations to Euclidean inner products and norms:

```lean (name := innerBridge)
-- Connect the real recurrence to the Euclidean inner
-- product and norm used by Gram-Schmidt.
#check @dotFn_eq_inner
#check @normFn_eq_norm
```

```leanOutput innerBridge (whitespace := lax)
@dotFn_eq_inner : ∀ {p : ℕ} (u v : Fin p → ℝ),
  dotFn u v = inner ℝ (WithLp.toLp 2 u) (WithLp.toLp 2 v)
```

```leanOutput innerBridge
@normFn_eq_norm : ∀ {p : ℕ} (v : Fin p → ℝ), normFn v = ‖WithLp.toLp 2 v‖
```

In classical Gram-Schmidt, subtracting projections makes each new column orthogonal to its
predecessors. To reuse Mathlib's proof of this fact, `dotFn_eq_inner` and `normFn_eq_norm` identify
the real fold operations with the inner product and norm of `EuclideanSpace ℝ (Fin m)`. Then
{src "TorchLean/Tensor/Factorization/Orthonormal.lean"}[the orthonormality file]
proves that, when every diagonal entry of $`R` is positive, the `j`-th executable column *is*
Mathlib's `gramSchmidtNormed ℝ` of the column map. Orthonormality then comes straight from
Mathlib's `gramSchmidtNormed_orthonormal'` {Informal.citep mathlib2020}[]. Proving that column
identity connects the implementation's lists and folds to the existing analytic theorem.

`WithLp.toLp 2` keeps the same coordinate values while viewing them in a space equipped with the
Euclidean norm. This matters because a raw function type can carry a different norm convention. A
sum-of-squares norm shouldn't be inferred from the presence of real coordinates alone. The bridge
spells out the intended geometry. After that identification, a projection coefficient computed by
`dotFn` is the inner product used by Mathlib's Gram-Schmidt theorem, and a divisor computed by
`normFn` is its Euclidean length. Both identifications are needed: matching only the projection
formula would leave open whether the resulting columns were normalized.

# Cholesky Recurrence

For a symmetric matrix, Cholesky computes a lower-triangular $`L` using

$$`L_{jj}
=\sqrt{A_{jj}-\sum_{k<j}L_{jk}^{\,2}},`

and, below the diagonal,

$$`L_{ij}
=\frac{A_{ij}-\sum_{k<j}L_{ik}L_{jk}}{L_{jj}}
\qquad(i>j).`

Entries with $`i<j` are set to zero. In the executable specification,
{src "TorchLean/Tensor/Factorization.lean"}[`choleskyColsFn`] is a left fold that
appends one column at a time. `choleskyFn` reads the resulting columns as a matrix, and
`Tensor.cholesky`, the public executable entry point, wraps that function as a shaped tensor.

This fold representation matters in the proof. Once column `j` is appended, later iterations never
change it. Generic “fold that appends” lemmas make that invariant reusable instead of reproving
list indexing at every matrix entry.

The two propositions are ordinary definitions, and their types say where they live:

```lean (name := preds)
-- Inspect the matrix-shaped propositions separately from
-- the functions that compute candidates.
#check @IsCholesky
#check @IsQR
```

```leanOutput preds (whitespace := lax)
@IsCholesky : {n : ℕ} → Matrix (Fin n) (Fin n) ℝ → Matrix (Fin n) (Fin n) ℝ → Prop
```

```leanOutput preds (whitespace := lax)
@IsQR : {m k : ℕ} → Matrix (Fin m) (Fin k) ℝ → Matrix (Fin m) (Fin k) ℝ →
  Matrix (Fin k) (Fin k) ℝ → Prop
```

Both take real-valued Mathlib `Matrix` values. Their equalities are exact, and a floating-point
residual tolerance is a separate property.

The factor itself is the generic function, so it can be read at either scalar type:

```lean (name := cholFnSig)
-- The recurrence maps an input matrix function to another
-- matrix function of the same square size.
#check @Factorization.choleskyFn
```

```leanOutput cholFnSig (whitespace := lax)
@choleskyFn : {α : Type} → [Context α] → {n : ℕ} →
  (Fin n → Fin n → α) → Fin n → Fin n → α
```

The first structural theorem needs only an index comparison:

```lean (name := lowerTri)
-- Only the above-diagonal index comparison is needed to
-- prove these entries vanish.
#check @choleskyFn_lower_triangular
```

```leanOutput lowerTri (whitespace := lax)
@choleskyFn_lower_triangular : ∀ {n : ℕ} (A : Fin n → Fin n → ℝ) {i j : Fin n},
  ↑i < ↑j → choleskyFn A i j = 0
```

That theorem applies to every real input matrix, including nonsymmetric matrices and matrices with
zero pivots. Applying it takes one line, with `norm_num` discharging the index comparison $`0<2`:

```lean
-- Specialize triangularity to entry zero-two. Its value is
-- independent of the input entries.
example (A : Fin 3 → Fin 3 → ℝ) :
    Factorization.choleskyFn A 0 2 = 0 :=
  choleskyFn_lower_triangular A (by norm_num)
```

The column builder emits `0` whenever `i < j`, independently of the input values. This proves
triangularity but says nothing about whether the remaining entries reconstruct the input.

Reconstruction needs two hypotheses:

```lean (name := cholPos)
-- Symmetry and positivity of the computed real pivots are
-- the reconstruction premises.
#check @isCholesky_of_pos
```

```leanOutput cholPos (whitespace := lax)
@isCholesky_of_pos : ∀ {n : ℕ} (A : Fin n → Fin n → ℝ),
  (∀ (i j : Fin n), A i j = A j i) →
    (∀ (j : Fin n), 0 < choleskyFn A j j) →
      IsCholesky (Matrix.of A) (Matrix.of (choleskyFn A))
```

The positive-pivot condition is the algorithm's success condition. It permits the division by
$`L_{jj}` and identifies the positive square root. A symmetric positive-definite matrix is the
standard sufficient condition, but the theorem doesn't prove

$$`\operatorname{PosDef}(A)
\Longrightarrow
\forall j,\;0<L_{jj}.`

It assumes positivity of the pivots computed by the real specification instead. Without a proof
that positive definiteness implies those premises, the theorem can't be applied using only an SPD
hypothesis.

Positivity enters the diagonal and off-diagonal equations in two related ways. A positive pivot is
nonzero, permitting cancellation after the division in the off-diagonal recurrence. On the
diagonal, the square root must recover the radicand when squared. Over the reals, square root is
total and returns zero on a negative argument, and assuming its value is positive excludes that
case. Symmetry then lets the row identities cover both halves of the input matrix. These are facts
about the computed real pivots, which explains why the theorem exposes that particular success
condition instead of merely asking for a square input tensor.

At tensor level, `cholesky_reconstruction` states the entrywise identity for
$`L=\operatorname{cholesky}(A)`:

```lean (name := cholRecon)
-- Read reconstruction through tensor entries, with the
-- transpose expressed by the second row index.
#check @cholesky_reconstruction
```

```leanOutput cholRecon (whitespace := lax)
@cholesky_reconstruction : ∀ {n : ℕ} (A : Tensor ℝ [n, n]),
  (∀ (i j : Fin n), A.get2 i j = A.get2 j i) →
    (∀ (j : Fin n), 0 < A.cholesky.get2 j j) →
      ∀ (i j : Fin n), A.get2 i j = ∑ k, A.cholesky.get2 i k * A.cholesky.get2 j k
```

In ordinary notation, the conclusion reads $`A_{ij}=\sum_k L_{ik}L_{jk}`, which is $`A=LL^\top`
written out one entry at a time. We state a tensor form because callers hold tensors, and its
hypotheses are the same two as before, restated through `Tensor.get2`.

## An Exact Cholesky Factor

To check what `IsCholesky` requires of a candidate factor, take

$$`A=
\begin{pmatrix}4&2&2\\2&5&3\\2&3&6\end{pmatrix},
\qquad
L=
\begin{pmatrix}2&0&0\\1&2&0\\1&1&2\end{pmatrix}.`

```lean (name := exactChol)
-- Check the supplied candidate factor by triangularity and
-- nine exact reconstruction equations.
def choleskyA : Matrix (Fin 3) (Fin 3) ℝ :=
  !![4, 2, 2; 2, 5, 3; 2, 3, 6]

def choleskyL : Matrix (Fin 3) (Fin 3) ℝ :=
  !![2, 0, 0; 1, 2, 0; 1, 1, 2]

example : IsCholesky choleskyA choleskyL := by
  refine ⟨?_, ?_⟩
  · intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [choleskyL]
  · ext i j
    fin_cases i <;> fin_cases j <;>
      simp [choleskyA, choleskyL, Matrix.mul_apply,
        Fin.sum_univ_three] <;> norm_num
```

The two bullets prove triangularity and reconstruction. Since the candidate contains only integers,
the proof needs no square-root computation or floating tolerance. The first bullet enumerates the
nine index pairs and discards the six where `i < j` fails (the surviving three are literal zeros),
while the second turns $`A=LL^\top` into nine scalar equations, expands the matrix product with
`Fin.sum_univ_three`, and lets `norm_num` finish. For instance the $`(2,1)` entry asks
$`1\cdot1+1\cdot2+2\cdot0=3`, using row `2` of `L` and row `1` of `L`.

Here we supply `L` ourselves. Applying `isCholesky_of_pos choleskyA` would instead require
$`0<\operatorname{choleskyFn}A\,j\,j` for each `j`, and those pivots are `MathFunctions.sqrt`
applied to real arguments. Discharging that hypothesis requires unfolding the chosen algorithm and
scalar operations, then proving the square-root identities for its diagonal entries. Our proof
above checks the candidate factor itself. It doesn't prove that `choleskyFn` returns that
candidate.

# Cholesky Execution

You can pass the same $`A` to the public executable operation `Tensor.cholesky`:

```lean (name := cholSigs)
-- The public return type fixes the square shape but does
-- not carry a reconstruction certificate.
#check @Tensor.cholesky
```

```leanOutput cholSigs (whitespace := lax)
@cholesky : {α : Type} → [inst : Storage α] → [Context α] → {n : ℕ} →
  Tensor α [n, n] → Tensor α [n, n]
```

```lean (name := cholRun)
-- Flatten the computed factor in row-major order to compare
-- it with the explicit candidate.
def spd3 : Tensor Float [3, 3] :=
  [[4, 2, 2],
   [2, 5, 3],
   [2, 3, 6]]

#eval Tensor.to (Tensor.cholesky spd3) (Array Float)
```

```leanOutput cholRun (whitespace := lax)
#[2.000000, 0.000000, 0.000000, 1.000000, 2.000000, 0.000000, 1.000000,
  1.000000, 2.000000]
```

`Tensor.to ... (Array Float)` flattens row-major, so read that as three rows of three: `2 0 0`,
then `1 2 0`, then `1 1 2`. These entries agree with the candidate `L` above.

Tracing the recurrence explains this agreement. Every intermediate value in this run is a small
integer or a dyadic rational: $`\sqrt4=2`, then $`2/2=1`, then $`\sqrt{5-1}=2`, then $`(3-1)/2=1`,
then $`\sqrt{6-1-1}=2`. Each is exactly representable in binary64 and each operation is exact, so
the float run lands on the real answer. Changing `A 0 0` from `4` to `3` makes the first pivot
$`\sqrt3`, and the floating factor then requires a rounding comparison with the real one. Our
candidate-factor proof is over $`\mathbb R`, while this `Float` output is a measurement on one
input. The executable implementation described later in the chapter is proved equal to the
definition on every scalar type, so it adds nothing to that rounding comparison.

Consider the symmetric but indefinite matrix

$$`\begin{pmatrix}1&2\\2&1\end{pmatrix},`

whose eigenvalues are `3` and `-1`. Watch where the failure lands:

```lean (name := cholFail)
-- The second pivot asks for a negative square root in this
-- symmetric indefinite example.
def indef2 : Tensor Float [2, 2] :=
  [[1, 2],
   [2, 1]]

#eval Tensor.to (Tensor.cholesky indef2) (Array Float)
```

```leanOutput cholFail
#[1.000000, 0.000000, 2.000000, NaN]
```

The first three entries are finite: $`L_{00}=\sqrt1=1`, $`L_{01}=0` by triangularity,
$`L_{10}=2/1=2`. Only the second pivot fails, because it asks for $`\sqrt{1-2^2}=\sqrt{-3}`. Index
`(1,1)`, where the NaN appears, is exactly the entry `isCholesky_of_pos` requires to be positive in
the real specification. Real and floating square roots handle a negative argument differently, so
this run illustrates a failed pivot without constituting a direct application of the theorem.

For this negative control, the example intentionally uses a sum of squared entrywise errors,
because IEEE `max` can ignore a NaN operand. That detail is part of the test's meaning: even a
diagnostic norm must choose NaN behavior deliberately.

# Cholesky In PyTorch

In the recorded transcript, `torch.linalg.cholesky` gives the same displayed factor on the
positive-definite input {Informal.citep pytorch2019}[]:

```
>>> # Compute the factor for the same small
>>> # positive-definite matrix.
>>> import torch
>>> A = torch.tensor([[4.,2.,2.],[2.,5.,3.],[2.,3.,6.]])
>>> torch.linalg.cholesky(A)
tensor([[2., 0., 0.],
        [1., 2., 0.],
        [1., 1., 2.]])
```

The factor agrees with the Lean result. On the indefinite matrix, PyTorch reports the failed pivot:

```
>>> # Inspect both the exception and the leading-minor
>>> # diagnostic for the indefinite matrix.
>>> B = torch.tensor([[1.,2.],[2.,1.]])
>>> torch.linalg.cholesky(B)
torch._C._LinAlgError: linalg.cholesky: The factorization could not be
completed because the input is not positive-definite (the leading minor
of order 2 is not positive-definite).
>>> torch.linalg.cholesky_ex(B).info
tensor(2, dtype=torch.int32)
```

The failure channels differ. PyTorch raises, while TorchLean returns a tensor containing NaN.
`Tensor.cholesky` is a total Lean function, so it must return something of type `Tensor α [n, n]`
for every input, and the scalar backend determines the result of invalid arithmetic. The output
type carries the dimensions, not a success flag or a proof of finite entries. Returning a tensor
therefore says only that the computation produced an inhabitant of that type. It doesn't discharge
the positive-pivot premise. You can see this without a large matrix in the indefinite example: its
first column is usable, but the second diagonal entry can't serve as a real positive square-root
pivot. Passing that tensor directly to a later solve would propagate the failed arithmetic beyond
the point at which it first became diagnosable. With the `Float` backend, callers must check for a
failed factorization before using its output.

`cholesky_ex` returns `info = 2`. That is the one-based index of the first non-positive leading
minor, pointing at the same pivot where the Lean run put its NaN. This is runtime diagnostic data.
TorchLean's $`\forall j,\;0<L_{jj}` is instead a hypothesis on the real specification's pivots for
a chosen matrix. It isn't an equivalence theorem between LAPACK's `info` result and those real
pivots: rounding and the different implementation would need their own analysis.

A successful numerical run is commonly described by $`A\approx LL^\top`, but applying such a
statement requires an error measure and bound. The Lean theorem here is exact over $`\mathbb R`,
while the `Float` examples only measure residuals. This chapter supplies no proved backward-error
bound for either displayed floating result. The FP32 soundness chapter describes rounding machinery
that a proof for TorchLean's factorization would need.

# Classical Gram-Schmidt As QR

For columns $`a_0,\ldots,a_{k-1}`, classical Gram-Schmidt computes

$$`\begin{aligned}
v_j &= a_j-\sum_{i<j}\langle q_i,a_j\rangle q_i,\\
r_{jj} &= \|v_j\|,\\
q_j &= v_j/r_{jj},\\
r_{ij} &= \langle q_i,a_j\rangle\quad(i<j).
\end{aligned}`

Here $`\langle\cdot,\cdot\rangle` and $`\|\cdot\|` correspond to the real instances of `dotFn` and
`normFn`. Subtracting the projections leaves the part of the new column outside the span of the
earlier columns, and normalizing that residual gives the next column of `Q`.

Throughout the projection sum, the coefficient uses the original column `a_j`. In exact arithmetic,
its component along each earlier orthonormal column can be removed independently, leaving a
residual orthogonal to all of them. That is the classical Gram-Schmidt recurrence in this source.
If the new column is already in their span, the residual is zero and there is no direction to
normalize. A positive `r_jj` rules out this degeneracy, and it is what the implementation tests
before it adds the normalized residual to the basis. Thus the pivot hypothesis describes the step
where a new independent feature direction is actually added to the factor.

TorchLean's specification uses the same column-building pattern as Cholesky. `gramSchmidtFn`
threads a list of basis columns and a list of `R` columns, and `Factorization.qr` runs it once to
return both factors as tensors, `(Factorization.qr A).1` and `(Factorization.qr A).2`. A column
whose residual norm is not positive, NaN included, takes no basis column, so a later independent
column can still enter. Its `R` column holds only its coefficients against the basis built so far.

Instead of a pair, the public operation returns a small structure, so the two factors have names:

```lean (name := qrSigs)
-- The reduced factor structure keeps the shapes of Q and R
-- tied to the input dimensions.
#check Tensor.QRFactors
#check @Tensor.qr
```

```leanOutput qrSigs
TorchLean.Tensor.QRFactors (α : Type) [Storage α] (m n : ℕ) : Type
```

```leanOutput qrSigs (whitespace := lax)
@Tensor.qr : {α : Type} → [inst : Storage α] → [Context α] → {m n : ℕ} →
  Tensor α [m, n] → QRFactors α m n
```

The fields carry the reduced shapes:

```
-- The shared inner dimension allows the two reduced factors
-- to multiply back to the input shape.
result.q : Tensor α [m, min m n]
result.r : Tensor α [min m n, n]
```

Every shape runs the same `gramSchmidtFn` schedule, which builds at most `min m n` basis columns.
The entries of `Tensor.qr` are those of `Factorization.qr` cut to the reduced shapes, and the cut
columns of `Q` and rows of `R` are zero. When `n ≤ m`, the shapes reduce to `[m, n]` and `[n, n]`
and nothing is cut. For a wide matrix, $`R_{jj}=0` for every `j ≥ m`, so the positive-pivot
hypothesis of the theorems below never holds: they don't cover wide inputs.

Three separate theorems correspond to the three conjuncts of `IsQR`. Upper-triangularity, like its
Cholesky counterpart, needs no hypothesis:

```lean (name := upperTri)
-- Entries below the R diagonal vanish by construction,
-- independently of residual norms.
#check @rMat_upper_triangular
```

```leanOutput upperTri (whitespace := lax)
@rMat_upper_triangular : ∀ {n m : ℕ} (A : Fin m → Fin n → ℝ) {k j : Fin n},
  ↑j < ↑k → rMat A k j = 0
```

`rMat` and `qMat` are the proof layer's readable views of the two factors, defined as plain
functions of the row and column index so that induction over columns does not have to travel through
a tensor. Orthonormality and reconstruction are where the rank hypothesis enters:

```lean (name := qrPos)
-- Positive residual norms support both orthonormality and
-- the complete QR predicate.
#check @qMat_transpose_mul_self
#check @isQR_of_pos
```

```leanOutput qrPos (whitespace := lax)
@qMat_transpose_mul_self : ∀ {m n : ℕ} (A : Fin m → Fin n → ℝ),
  (∀ (j : Fin n), 0 < rMat A j j) →
    ((Matrix.of fun i k => qMat A i k).transpose * Matrix.of fun i k => qMat A i k) = 1
```

```leanOutput qrPos (whitespace := lax)
@isQR_of_pos : ∀ {m n : ℕ} (A : Fin m → Fin n → ℝ),
  (∀ (j : Fin n), 0 < rMat A j j) →
    IsQR (Matrix.of A) (Matrix.of fun i k => qMat A i k) (Matrix.of fun k j => rMat A k j)
```

$`\forall j,\;0<R_{jj}` is, for classical Gram-Schmidt, the executable form of full column rank:
every new column has a nonzero component orthogonal to its predecessors. Again, the API states the
pivot condition as a hypothesis instead of deriving it from a separately formalized rank predicate,
which leaves the same missing bridge as on the Cholesky side.

At the tensor boundary the two results are restated through `Tensor.get2`:

```lean (name := qrTensor)
-- Inspect the two tensor identities separately: reconstruct
-- entries and compare column dot
-- products.
#check @qr_reconstruction
#check @qrQ_orthonormal
```

```leanOutput qrTensor (whitespace := lax)
@qr_reconstruction : ∀ {n m : ℕ} (A : Tensor ℝ [m, n]),
  (∀ (j : Fin n), 0 < (Factorization.qr A).2.get2 j j) →
    ∀ (i : Fin m) (j : Fin n), A.get2 i j =
      ∑ k, (Factorization.qr A).1.get2 i k * (Factorization.qr A).2.get2 k j
```

```leanOutput qrTensor (whitespace := lax)
@qrQ_orthonormal : ∀ {m n : ℕ} (A : Tensor ℝ [m, n]),
  (∀ (j : Fin n), 0 < (Factorization.qr A).2.get2 j j) →
    ∀ (a b : Fin n),
      ∑ i, (Factorization.qr A).1.get2 i a * (Factorization.qr A).1.get2 i b =
        if a = b then 1 else 0
```

The right-hand side `if a = b then 1 else 0` is $`Q^\top Q=I` written entrywise, and the same
hypothesis appears in both. Positivity of every diagonal residual norm supports both
orthonormality and reconstruction.

For a tall matrix, the identity is about columns: `Q` has shape `[m, n]`, so `Qᵀ Q` has shape
`[n, n]`. It does not say that `Q Qᵀ` is the `[m, m]` identity. When `m > n`, the columns span
only a subspace of the ambient row space. In a least-squares application this distinction explains
why multiplication by `Qᵀ` extracts coordinates and multiplication by `Q` reconstructs the
component in that subspace. Reading `qrQ_orthonormal` entrywise makes the orientation explicit:
the sum ranges over row index `i`, while `a` and `b` choose two columns whose dot product is being
compared with the identity matrix.

# QR Factors And Residuals

Use the following matrix to inspect both factors and their residuals:

$$`A=
\begin{pmatrix}
12&-51&4\\
6&167&-68\\
-4&24&-41
\end{pmatrix}.`

```lean (name := qrRun)
-- Print both factors so their shared diagonal normalization
-- can be inspected.
def classicQR : Tensor Float [3, 3] :=
  [[12, -51, 4],
   [6, 167, -68],
   [-4, 24, -41]]

#eval Tensor.to (Tensor.qr classicQR).q (Array Float)
#eval Tensor.to (Tensor.qr classicQR).r (Array Float)
```

```leanOutput qrRun (whitespace := lax)
#[0.857143, -0.394286, -0.331429, 0.428571, 0.902857, 0.034286, -0.285714,
  0.171429, -0.942857]
```

```leanOutput qrRun (whitespace := lax)
#[14.000000, 21.000000, -14.000000, 0.000000, 175.000000, -70.000000,
  0.000000, 0.000000, 35.000000]
```

Read `R` as three rows: `14 21 -14`, `0 175 -70`, `0 0 35`. Those are the exact textbook values,
and the two structural facts are visible in the print. The strictly lower triangle is zero, which
is `rMat_upper_triangular` and needs no hypothesis. The diagonal is `14`, `175`, `35`, all
positive, which is a runtime observation, not a proof of positivity for the theorem's real
instance. Positivity is not an accident of the example: $`r_{jj}=\|v_j\|` and `normFn` is a square
root, so the diagonal can never be negative. Over the reals it can be zero, and the positive-pivot
hypothesis excludes that case. Floating execution can also encounter NaN or overflow.

The first column of `Q` is $`(0.857143,0.428571,-0.285714)`, which is
$`(12,6,-4)/14=(6/7,3/7,-2/7)`. The divisor `14` is $`\|a_0\|=\sqrt{144+36+16}=\sqrt{196}`, so it
is also the `R` entry printed above, and the two factors share that number by construction.

Now measure the orthonormality that the theorem asserts over $`\mathbb R`:

```lean (name := qrDots)
-- Compare a column with itself and with its neighbor using
-- the same floating dot product.
def qColumn (j : Fin 3) : Fin 3 → Float :=
  fun i => Tensor.get2 (Tensor.qr classicQR).q i j

#eval Factorization.dotFn (qColumn 0) (qColumn 0)
#eval Factorization.dotFn (qColumn 0) (qColumn 1)
```

```leanOutput qrDots
1.000000
```

```leanOutput qrDots
-0.000000
```

The display `-0.000000` alone does not distinguish a small negative value from signed zero.
Scaling by $`10^{18}` reveals the nonzero residual in this case:

```lean (name := qrDotsScaled)
-- Scale only the diagnostics to expose small values hidden
-- by ordinary decimal printing.
#eval Factorization.dotFn (qColumn 0) (qColumn 1) * 1e18
#eval Factorization.dotFn (qColumn 1) (qColumn 2) * 1e18
```

```leanOutput qrDotsScaled
-20.816682
```

```leanOutput qrDotsScaled
-277.555756
```

So the two off-diagonal inner products are $`-2.08\times10^{-17}` and $`-2.78\times10^{-16}`, which
are small on the scale of a binary64 number near one. For the real factors the theorem gives zero,
but these floating factors don't satisfy $`Q^\top Q=I` exactly. Bounding their deviation requires a
rounding model, which the FP32 soundness chapter supplies for other operators and the factorization
layer lacks.

Reconstruction is a different story on this input. Taking the largest entrywise difference between
$`QR` and $`A`, and scaling it the same way to be sure nothing is hiding under the display:

```lean (name := qrResidual)
-- Form each reconstruction entry from a row of Q and a
-- column of R before taking the largest error.
def qRow (i : Fin 3) : Fin 3 → Float :=
  fun k => Tensor.get2 (Tensor.qr classicQR).q i k

def rCol (j : Fin 3) : Fin 3 → Float :=
  fun k => Tensor.get2 (Tensor.qr classicQR).r k j

def reconResidual : Float :=
  ((List.finRange 3).flatMap fun i =>
      (List.finRange 3).map fun j =>
        (Factorization.dotFn (qRow i) (rCol j) -
          Tensor.get2 classicQR i j).abs).foldl max 0.0

#eval reconResidual * 1e18
```

```leanOutput qrResidual
0.000000
```

The scaled maximum residual still displays as zero. Scaling gives a more sensitive diagnostic than
printing the residual itself, though a decimal print alone does not prove bitwise zero. It measures
this floating reconstruction of one $`3\times3` matrix.

Read the two diagnostics independently. Reconstruction forms dot products of rows of `Q` with
columns of `R`, whereas orthonormality forms dot products of pairs of columns of `Q`. Their
cancellations and rounding steps differ. A factor can therefore reproduce this input very closely
and still have nonzero cross-column dot products. Scaling a diagnostic before printing reveals
values hidden by six decimal places, but it doesn't change the computed factors or improve their
accuracy. For a model that repeatedly uses an orthogonal basis, such as a projection layer, the
cross-column error matters even when the reconstruction residual looks smaller.

In the recorded float64 `torch.linalg.qr` transcript, the factors of this matrix have different
signs:

```
>>> # Inspect the alternative factor signs together with
>>> # reconstruction and orthonormality
>>> # residuals.
>>> C = torch.tensor([[12.,-51.,4.],[6.,167.,-68.],[-4.,24.,-41.]],
...                  dtype=torch.float64)
>>> Q, R = torch.linalg.qr(C)
>>> R
tensor([[ -14.,  -21.,   14.],
        [   0., -175.,   70.],
        [   0.,    0.,  -35.]], dtype=torch.float64)
>>> (Q @ R - C).abs().max().item()
2.842170943040401e-14
>>> (Q.T @ Q - torch.eye(3, dtype=torch.float64)).abs().max().item()
2.220446049250313e-16
```

Every sign is flipped. LAPACK computes QR by Householder reflections, and the reflection at step
`j` is chosen for numerical reasons that leave $`r_{jj}` negative here. Since $`(-Q)(-R)=QR`, this
is a perfectly good factorization of the same matrix. The predicate `IsQR` itself allows negative
diagonal entries. The implementation theorems `isQR_of_pos` and `qrQ_orthonormal` concern
TorchLean's own real Gram-Schmidt factors, not arbitrary factors returned by LAPACK. Relating those
results requires more than checking or normalizing the signs: one must connect the algorithms and
account for floating arithmetic.

In the displayed run, LAPACK's reconstruction error is $`2.8\times10^{-14}`, while the Gram-Schmidt
residual displays as zero even after scaling. For LAPACK, the measured orthonormality error is
$`2.2\times10^{-16}`. For Gram-Schmidt, the larger of the two displayed off-diagonal products is
$`2.8\times10^{-16}`, but those two entries alone don't measure the full Gram-matrix error. These
measurements on one $`3\times3` matrix don't compare the algorithms' stability. Classical
Gram-Schmidt can lose orthogonality as columns become nearly dependent. A stability analysis must
relate the error to the input and the arithmetic across that wider class of cases.

# Rank-Deficient Inputs

Take the smallest rank-deficient case, a $`2\times2` matrix whose second column is twice its first:

```lean (name := deficientRun)
-- The dependent second column leaves a zero residual, so
-- it takes no basis column.
def deficient2 : Tensor Float [2, 2] :=
  [[1, 2],
   [0, 0]]

#eval Tensor.to (Tensor.qr deficient2).q (Array Float)
#eval Tensor.to (Tensor.qr deficient2).r (Array Float)
```

```leanOutput deficientRun
#[1.000000, 0.000000, 0.000000, 0.000000]
```

```leanOutput deficientRun
#[1.000000, 2.000000, 0.000000, 0.000000]
```

Reconstruction still holds: $`QR` is `[[1, 2], [0, 0]]`, which is `A`. But the second column of `Q`
is the zero vector, so its squared norm is zero instead of one:

```lean (name := deficientDot)
-- A zero column has squared norm zero, which prevents Q
-- from having orthonormal columns.
def dCol (j : Fin 2) : Fin 2 → Float :=
  fun i => Tensor.get2 (Tensor.qr deficient2).q i j

#eval Factorization.dotFn (dCol 1) (dCol 1)
```

```leanOutput deficientDot
0.000000
```

Here $`Q^\top Q` is $`\operatorname{diag}(1,0)`, differing from the identity by one in one entry.
The pivot hypothesis fails at `j = 1`, where $`r_{11}=\|v_1\|=0`. The dependent column takes no
basis column, and the second column of `Q` stays zero.

On this rank-deficient matrix, LAPACK returns a full orthogonal basis:

```
>>> # Inspect how this QR implementation completes the
>>> # orthogonal basis for a deficient input.
>>> D = torch.tensor([[1.,2.],[0.,0.]])
>>> Q, R = torch.linalg.qr(D)
>>> Q
tensor([[1., 0.],
        [-0., 1.]])
>>> R
tensor([[1., 2.],
        [0., 0.]])
```

The displayed `Q` satisfies $`Q^\top Q=I` exactly, and the rank deficiency is
visible only as $`R_{11}=0`. Householder reflections are built from the input columns but do not
normalize each residual as a new basis column, so they can supply an orthogonal completion.
In TorchLean's Gram-Schmidt, a column with zero residual norm takes no basis column, and the
unused basis columns of `Q` stay zero.

In this example, reconstruction leaves the missing column direction unconstrained. Since the second
row of `R` is zero, changing the second column of `Q` contributes nothing to `QR`. It still changes
`Qᵀ Q`, which explains how the displayed implementations can reconstruct the same matrix while
producing different orthonormality results. A consumer that needs a full basis must check that
property of the returned `Q` and not infer it from reconstruction alone.

For TorchLean's real Gram-Schmidt factors, $`0<R_{jj}` for every `j` means that every column
enters the basis, so no column of `Q` is zero.
This is a hypothesis about those particular factors. It is not a portable success test for
arbitrary QR implementations, whose nonzero diagonal entries can have either sign.

For a wide $`2\times3` input, `Tensor.qr` returns `q : Tensor Float [2, 2]` and
`r : Tensor Float [2, 3]`, and both types are known when Lean elaborates the program, not when it
runs. For a matrix with a duplicated column, the reconstruction residual may remain small while
`Qᵀ Q` differs from the identity, and floating cancellation can also leave tiny nonzero residual
columns instead of the exact zero columns predicted over the reals. Reconstruction observed on one
rank-deficient example doesn't discharge the positive-pivot hypothesis of the general theorem.

# Triangular Solves And Ridge Regression

Once $`A=LL^\top`, a linear system is two triangular substitutions: forward-solve $`Lz=b`, then
back-solve $`L^\top x=z`. TorchLean ships that path, and the ridge variant that adds
$`\gamma I` first:

```lean (name := ridgeSig)
-- The ridge interface accepts a square kernel, scalar
-- regularization, and matching target vector.
#check @Tensor.solveRidge
```

```leanOutput ridgeSig (whitespace := lax)
@Tensor.solveRidge : {α : Type} → [inst : Storage α] → [Context α] → {n : ℕ} →
  Tensor α [n, n] → α → Tensor α [n] → Tensor α [n]
```

With $`\gamma=0` it is an ordinary solve, and a $`2\times2` case can be checked by hand:

```lean (name := ridgeRun)
-- With zero regularization, compare the ordinary two-by-two
-- solve with its hand calculation.
def kernel2 : Tensor Float [2, 2] :=
  [[2, 1],
   [1, 3]]

def target2 : Tensor Float [2] := Tensor.from #[1.0, 2.0]

#eval Tensor.to
  (Tensor.solveRidge kernel2 0.0 target2) (Array Float)
```

```leanOutput ridgeRun
#[0.200000, 0.600000]
```

The determinant is $`2\cdot3-1=5`, so
$`x=\tfrac15\bigl(3\cdot1-1\cdot2,\;-1\cdot1+2\cdot2\bigr)=(1/5,3/5)`, which is what printed.

Substitution explains how the factor is used. The first triangular system computes each entry of `z`
from earlier entries, dividing by a diagonal entry of `L`. The second traverses the transposed
factor in reverse order to obtain `x`. Adding `γ I` changes only the diagonal of the matrix being
factored, not the supplied target vector. In kernel ridge regression, that vector contains the
observed targets, while the returned vector contains coefficients used for predictions through the
kernel. Over $`\mathbb R`, `solveRidge_spec` in
{src "TorchLean/Tensor/Factorization/Solve.lean"}[`FactorizationsSolve`] connects the returned
coefficients to `(K + γ I) x = b`: for symmetric `K` such that `K + γ I` has positive Cholesky
pivots, `Tensor.solveRidge K γ b` solves that system. Symmetry and positive pivots make the factor
reconstruct `K + γ I`, and the pivots form its diagonal, so both substitutions divide safely.

Each substitution has an exact theorem of its own. For a lower- or upper-triangular matrix with
nonzero diagonal, `mulVec_triSolveLowerFn` and `mulVec_triSolveUpperFn` prove that substitution
returns the solution, and `mulVec_cholSolveFn` combines them: `cholSolveFn L b` solves
$`LL^\top x=b`. The `#eval` checks one `Float` input, which these real theorems don't cover. Kernel
ridge regression and Gaussian process code use this path, so over $`\mathbb R` their linear solves
are exact under the same hypotheses.

# Exact Proofs And Floating Execution

The relevant definitions and checks provide different guarantees:

:::table +header
*
  * Object
  * Scalar
  * Guarantee
*
  * `Tensor.cholesky`, `Tensor.qr`
  * generic executable scalar backends
  * public computation with statically checked input and reduced output shapes, and `Tensor.qr` runs
    the proved `gramSchmidtFn` schedule for every shape
*
  * `IsCholesky`, `IsQR`
  * `ℝ`
  * exact algebraic specification
*
  * `Tensor.cholesky`, `Factorization.qr` in the proofs
  * `ℝ`
  * exact reconstruction under pivot hypotheses
*
  * factorization examples
  * `Float`
  * executable residual checks on concrete matrices
:::

The `Float` output is evidence that the executable definitions behave as expected on those inputs.
It isn't the proof of $`A=LL^\top` or $`Q^\top Q=I`: the $`-2.08\times10^{-17}` above is a concrete
demonstration that machine arithmetic doesn't generally satisfy those identities. Conversely, the
real theorem doesn't prove a forward-error or backward-error bound for the Float execution.

Flocq {Informal.citep flocq2011}[] provides a floating-point formalization in Rocq with rounding
predicates and error lemmas. Such machinery supports a separate question: how far a rounded
factorization can deviate from its exact-$`\mathbb R` counterpart. TorchLean's factorization
development first proves the algebra over $`\mathbb R`, where it can reuse `Matrix`,
`EuclideanSpace`, and `gramSchmidtNormed_orthonormal'`. Its comparison with floating arithmetic is
unproved, and an arithmetic formalization alone doesn't supply an algorithm's error bound.

`choleskyColsFn`, `cholSolveFn`, and `solveRidgeFn` each have a `@[csimp]` theorem
(`choleskyColsFn_eq_choleskyCols`, `cholSolveFn_eq_cholSolve`, `solveRidgeFn_eq_solveRidge`) that
proves the definition equal to an internal array implementation, so the Cholesky and ridge-solve
numerical examples run the array code while Lean proofs reason about the reference definitions. QR
uses its separate Gram-Schmidt definition. The three proofs use no law of the arithmetic, so the
equalities hold on `Float` as well as on $`\mathbb R`. The arrays retain computed columns, whereas
reading the factor from the closure representation can repeatedly recompute earlier column prefixes.

For an audit, the reconstruction identities are collected in
{src "TorchLean/Tensor/Factorization/Reconstruction.lean"}[
`Tensor.Factorization.Reconstruction`], while the orthonormal-column statements are in
{src "TorchLean/Tensor/Factorization/Orthonormal.lean"}[`Tensor.Factorization.Orthonormal`].
The executable definitions themselves remain in
{src "TorchLean/Tensor/Factorization.lean"}[`Tensor.Factorization`].
The public executable entry points are in
{src "TorchLean/Tensor/LinearAlgebra.lean"}[`Tensor.LinearAlgebra`].
Keeping those three roles separate makes it clear whether a cited result is a definition, an exact
real identity, or evidence from a concrete Float run.

# Factorization Proof Gaps

To connect the exact reconstruction results to rank conditions, solvers, and numerical stability,
we need the following theorems, which we haven't proved:

1. positive definiteness implies positive executable Cholesky pivots
2. full column rank implies positive executable Gram-Schmidt pivots
3. reconstruction and orthonormality theorems for wide and rank-deficient QR inputs, where some
   pivot is zero
4. finite-precision stability bounds, especially for loss of orthogonality in classical
   Gram-Schmidt. Comparing its factors with Householder QR also requires accounting for their
   different sign conventions.

For the exact identities, we follow the standard mathematics in Golub and Van Loan's *Matrix
Computations*. The distinction between exact factorization and floating-point stability follows
Higham's *Accuracy and Stability of Numerical Algorithms*, which is also the reference for the
loss-of-orthogonality behavior that item four would have to quantify. TorchLean proves exact
reconstruction for its real specifications and tests concrete `Float` residuals, but it doesn't
prove stability bounds.
