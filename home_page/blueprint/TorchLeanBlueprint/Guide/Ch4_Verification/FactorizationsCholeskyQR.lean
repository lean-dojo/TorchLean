import VersoManual
import NN.Proofs.Tensor.Basic.Factorizations
import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal
import NN.Proofs.Tensor.Basic.FactorizationsReconstruction
import NN.Tensor
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The executable tensor API, the two predicates, and the Gram-Schmidt proof layer live in three
-- namespaces. Opening them keeps each displayed `#check` on a line Verso's narrow code column can
-- hold, and it also makes the printed signatures read the way a caller would write them.
open TorchLean
open Spec.Factorization
open Spec.Factorization.Reconstruction

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
  fixed shape;
- a proposition describing the algebraic object that the program must return.

Application code uses the public `Tensor.*` factorization API. The examples below also inspect
the generic `Spec.*` definitions that describe the arithmetic recurrences. Their real-valued
instances are the objects of the exact theorems; a floating-point evaluation is a separate claim.
The public operations have runtime overrides, whose relation to the specification we examine later.

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
are separate executable examples; they are not checked by the Lean document compiler.

# Dot Products And Norms

Both algorithms repeatedly compute inner products. Gram-Schmidt uses them to subtract
projections onto earlier columns; Cholesky subtracts inner products of partial rows before
computing a diagonal square root. Start with the dot product and norm that implement these steps.

The vector operations live on the plain function representation `Fin p → α`, not on tensors:

```lean (name := dotSig)
-- Inspect the coordinate-function interface shared by
-- projection coefficients and residual norms.
#check @Spec.dotFn
#check @Spec.normFn
```

```leanOutput dotSig
@Spec.dotFn : {α : Type} → [Context α] → {p : ℕ} → (Fin p → α) → (Fin p → α) → α
```

```leanOutput dotSig
@Spec.normFn : {α : Type} → [Context α] → {p : ℕ} → (Fin p → α) → α
```

`Context α` is TorchLean's scalar backend class: it supplies the arithmetic, the comparisons, and
`MathFunctions.sqrt`. Instantiating `α := Float` gives a program you can run; instantiating
`α := ℝ` gives an object you can prove things about. That single generic definition is why the
proofs and the executable path are not two independent transcriptions of the same recurrence.

The body of `dotFn` fixes the accumulation order with a left fold:

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
@dotFn_eq_sum : ∀ {p : ℕ} (u v : Fin p → ℝ), Spec.dotFn u v = ∑ i, u i * v i
```

The theorem is specific to `ℝ`; it does not remove the significance of summation order in a
floating-point run. To evaluate the fold, read the indexed coordinates of two length-three tensors:

```lean (name := dotEval)
-- Read three tensor coordinates as functions and evaluate
-- their dot product.
def uVec : Fin 3 → Float :=
  Tensor.getScalar ([1.0, 2.0, 3.0] : Tensor Float [3])

def vVec : Fin 3 → Float :=
  Tensor.getScalar ([4.0, 5.0, 6.0] : Tensor Float [3])

#eval Spec.dotFn uVec vVec
```

```leanOutput dotEval
32.000000
```

That is $`1\cdot4+2\cdot5+3\cdot6=4+10+18=32`, and the norm of `uVec` is $`\sqrt{14}`:

```lean (name := normEval)
-- The norm takes the square root of the self-dot product,
-- whose value here is fourteen.
#eval Spec.normFn uVec
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

#eval Tensor.dotSpec uTen vTen
```

```leanOutput tensorDot
32.000000
```

```lean (name := tensorDotSig)
-- The shared shape parameter forces both arguments to have
-- exactly the same dimensions.
#check @Tensor.dotSpec
```

```leanOutput tensorDotSig (whitespace := lax)
@Tensor.dotSpec : {α : Type} → [inst : Storage α] → [Context α] → {s : Shape} →
  Tensor α s → Tensor α s → α
```

The shape parameter is an arbitrary `Shape`, not `[n]`, because the definition is
`sumSpec (mulSpec a b)`: it contracts every scalar leaf. On `[n]` that is the vector dot product; on
`[m, n]` it is the Frobenius inner product of two matrices.

The last `α` in the signature is the scalar result type, so no shape remains on the returned
value. The preceding two occurrences of `Tensor α s` share one implicit `s`; Lean infers it from
the first tensor and checks the second against it. Two differently shaped tensors cannot be
substituted merely because they contain the same number of scalars. An explicit reshape would
be needed to describe which coordinates are to be paired.

Two more bridge lemmas connect the real operations to Euclidean inner products and norms:

```lean (name := innerBridge)
-- Connect the real recurrence to the Euclidean inner
-- product and norm used by Gram-Schmidt.
#check @dotFn_eq_inner
#check @normFn_eq_norm
```

```leanOutput innerBridge (whitespace := lax)
@dotFn_eq_inner : ∀ {p : ℕ} (u v : Fin p → ℝ),
  Spec.dotFn u v = inner ℝ (WithLp.toLp 2 u) (WithLp.toLp 2 v)
```

```leanOutput innerBridge
@normFn_eq_norm : ∀ {p : ℕ} (v : Fin p → ℝ), Spec.normFn v = ‖WithLp.toLp 2 v‖
```

In classical Gram-Schmidt, subtracting projections makes each new column orthogonal to its
predecessors. To reuse Mathlib's proof of this fact, `dotFn_eq_inner` and `normFn_eq_norm`
identify the real fold operations with the inner product and norm of
`EuclideanSpace ℝ (Fin m)`. Then
{src "NN/Proofs/Tensor/Basic/FactorizationsOrthonormal.lean"}[the
orthonormality file] proves that the `j`-th executable column
*is* Mathlib's `gramSchmidtNormed ℝ` of the column map. Orthonormality then comes straight from
Mathlib's `gramSchmidtNormed_orthonormal'` {Informal.citep mathlib2020}[]. Proving that column
identity connects the implementation's lists and folds to the existing analytic theorem.

`WithLp.toLp 2` keeps the same coordinate values while viewing them in a space equipped with the
Euclidean norm. This matters because a raw function type can carry a different norm convention;
a sum-of-squares norm should not be inferred from the presence of real coordinates alone. The
bridge establishes the intended geometry explicitly. After that identification, a projection
coefficient computed by `dotFn` is the inner product used by Mathlib's Gram-Schmidt theorem, and
a divisor computed by `normFn` is its Euclidean length. Both identifications are needed: matching
only the projection formula would leave open whether the resulting columns were normalized.

# Cholesky Recurrence

For a symmetric matrix, Cholesky computes a lower-triangular $`L` using

$$`L_{jj}
=\sqrt{A_{jj}-\sum_{k<j}L_{jk}^{\,2}},`

and, below the diagonal,

$$`L_{ij}
=\frac{A_{ij}-\sum_{k<j}L_{ik}L_{jk}}{L_{jj}}
\qquad(i>j).`

Entries with $`i<j` are set to zero. In the executable specification,
{src "NN/Spec/Core/Tensor/Factorizations.lean"}[`choleskyColsFn`]
is a left fold that appends one column at a time; `choleskyFn` reads the resulting columns as a
matrix; `choleskySpec` wraps that function as a shaped tensor; and `Tensor.cholesky` is the public
executable entry point.

The fold representation matters in the proof. Once column `j` is appended, later iterations never
change it. Generic “fold that appends” lemmas make that invariant reusable instead of reproving list
indexing at every matrix entry.

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

Both take real-valued Mathlib `Matrix` values. Their equalities are exact; a floating-point
residual tolerance is a separate property.

The factor itself is the generic function, so it can be read at either scalar type:

```lean (name := cholFnSig)
-- The recurrence maps an input matrix function to another
-- matrix function of the same square size.
#check @Spec.choleskyFn
```

```leanOutput cholFnSig (whitespace := lax)
@Spec.choleskyFn : {α : Type} → [Context α] → {n : ℕ} →
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
  ↑i < ↑j → Spec.choleskyFn A i j = 0
```

It applies to every real input matrix, including nonsymmetric matrices and matrices with zero
pivots. Applying it takes one line, with `norm_num` discharging the
index comparison $`0<2`:

```lean
-- Specialize triangularity to entry zero-two; its value is
-- independent of the input entries.
example (A : Fin 3 → Fin 3 → ℝ) :
    Spec.choleskyFn A 0 2 = 0 :=
  choleskyFn_lower_triangular A (by norm_num)
```

The column builder emits `0` whenever `i < j`, independently of the input values. This proves
triangularity but says nothing about whether the remaining entries reconstruct the input.

In the printed triangularity type, `{i j : Fin n}` are inferred indices, and `↑i < ↑j` compares
their natural-number values. The result fixes one entry at zero. The example specializes `n` to
three and the indices to zero and two; `norm_num` proves that index comparison without looking
at any matrix entries. This is a useful kind of unconditional theorem because the recurrence
explicitly writes zeros above the diagonal. Reconstruction is different: its claim depends on
the arithmetic values written below the diagonal, so it cannot follow just from the branch that
chooses which entries are zero.

Reconstruction needs two hypotheses:

```lean (name := cholPos)
-- Symmetry and positivity of the computed real pivots are
-- the reconstruction premises.
#check @isCholesky_of_pos
```

```leanOutput cholPos (whitespace := lax)
@isCholesky_of_pos : ∀ {n : ℕ} (A : Fin n → Fin n → ℝ),
  (∀ (i j : Fin n), A i j = A j i) →
    (∀ (j : Fin n), 0 < Spec.choleskyFn A j j) →
      IsCholesky (Matrix.of A) (Matrix.of (Spec.choleskyFn A))
```

The positive-pivot condition is the algorithm's success condition. It permits the division by
$`L_{jj}` and identifies the positive square root. A symmetric positive-definite matrix is the
standard sufficient condition, but the current theorem does not prove

$$`\operatorname{PosDef}(A)
\Longrightarrow
\forall j,\;0<L_{jj}.`

It assumes positivity of the pivots computed by the real specification directly. Deriving those
premises from positive definiteness is a remaining proof obligation before the theorem can be
applied using only an SPD hypothesis.

The diagonal and off-diagonal equations use positivity in two related ways. A positive pivot is
nonzero, permitting cancellation after the division in the off-diagonal recurrence. On the
diagonal, the square root must recover the radicand when squared. Over the reals, square root is
total and returns zero on a negative argument; assuming its value is positive excludes that case.
Symmetry then lets the row identities cover both halves of the input matrix. These are facts about
the computed real pivots, which explains why the theorem exposes that particular success condition
instead of merely asking for a square input tensor.

At tensor level, `choleskySpec_reconstruction` states the entrywise identity for
$`L=\operatorname{choleskySpec}(A)`:

```lean (name := cholRecon)
-- Read reconstruction through tensor entries, with the
-- transpose expressed by the second row index.
#check @choleskySpec_reconstruction
```

```leanOutput cholRecon (whitespace := lax)
@choleskySpec_reconstruction : ∀ {n : ℕ} (A : Tensor ℝ [n, n]),
  (∀ (i j : Fin n), Spec.get2 A i j = Spec.get2 A j i) →
    (∀ (j : Fin n), 0 < Spec.get2 (Spec.choleskySpec A) j j) →
      ∀ (i j : Fin n), Spec.get2 A i j =
        ∑ k, Spec.get2 (Spec.choleskySpec A) i k * Spec.get2 (Spec.choleskySpec A) j k
```

This is $`A_{ij}=\sum_k L_{ik}L_{jk}`, which is $`A=LL^\top` written out one entry at a time. The
tensor form exists because callers hold tensors; the hypotheses are the same two as before, restated
through `Spec.get2`.

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

The two bullets are the two conjuncts. `refine ⟨?_, ?_⟩` exposes the two properties as separate
proof goals; these holes are filled by
the following bullets, not left as assumptions. `ext i j` replaces matrix equality with equality at
an arbitrary entry, after which finite case analysis is sufficient for this three-by-three example.
The literal candidate contains only integers, so its reconstruction proof needs no square-root
computation or floating tolerance. This is a convenient way to validate the meaning of the
predicate before trying to prove that a general algorithm always produces a suitable witness.
The first bullet enumerates the nine index pairs and discards the
six where `i < j` fails; the surviving three are literal zeros. The second turns
$`A=LL^\top` into nine scalar equations, expands the matrix product with `Fin.sum_univ_three`, and
lets `norm_num` finish. For instance the $`(2,1)` entry asks
$`1\cdot1+1\cdot2+2\cdot0=3`, using row `2` of `L` and row `1` of `L`.

Here `L` is supplied explicitly. Applying `isCholesky_of_pos choleskyA` would instead require
$`0<\operatorname{choleskyFn}A\,j\,j` for each `j`, and those pivots are `MathFunctions.sqrt`
applied to real arguments. Discharging that hypothesis requires unfolding the chosen algorithm and
scalar operations, then
proving the square-root identities for its diagonal entries. The proof above checks the candidate
factor directly; it does not prove that `choleskyFn` returns that candidate.

# Cholesky Execution

The same $`A` can be passed to the public executable wrapper `Tensor.cholesky`:

```lean (name := cholSigs)
-- The public return type fixes the square shape but does
-- not carry a reconstruction certificate.
#check @Tensor.cholesky
```

```leanOutput cholSigs (whitespace := lax)
@Tensor.cholesky : {α : Type} → [inst : Storage α] → [Context α] → {n : ℕ} →
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

`Tensor.to ... (Array Float)` flattens row-major, so read that as three rows of three: `2 0 0`, then
`1 2 0`, then `1 1 2`. The displayed entries agree with the candidate `L` above.

The recurrence explains this agreement. Every intermediate value in this run is
a small integer or a dyadic rational: $`\sqrt4=2`, then $`2/2=1`, then $`\sqrt{5-1}=2`, then
$`(3-1)/2=1`, then $`\sqrt{6-1-1}=2`. Each is exactly representable in binary64 and each operation
is exact, so the float run lands on the real answer. Changing `A 0 0` from `4` to `3` makes the
first pivot $`\sqrt3`; the floating factor then requires
a rounding comparison with the real one.
The candidate-factor proof is over $`\mathbb R`, while this `Float` output is a measurement on one
input. A formal connection would also need to account for the executable implementation described
later in the chapter.

From the repository root, the shipped example checks the residual instead of the entries:

```terminal
# Run the shipped concrete factorization checks from the
# repository root.
lake exe torchlean factorizations
```

The Cholesky part of the output is:

```terminal +output
Cholesky A = L·Lᵀ: OK (error = 0.000000)
Cholesky on indefinite A correctly fails (no SPD ⇒ no factor): OK (failure detected)
```

The example itself calls the public executable API:

```
-- Application code calls the public tensor wrapper on its
-- chosen input matrix.
def lowerFactor := Tensor.cholesky positiveDefiniteMatrix
```

The second line comes from the symmetric but indefinite matrix

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

The first three entries are finite: $`L_{00}=\sqrt1=1`, $`L_{01}=0` by
triangularity, $`L_{10}=2/1=2`. Only the second pivot fails, because it asks for
$`\sqrt{1-2^2}=\sqrt{-3}`. The NaN appears at index `(1,1)`, which is exactly the entry
`isCholesky_of_pos` requires to be positive in the real specification. The real square root and
the floating square root handle a negative argument differently, so this run illustrates a failed
pivot without constituting a direct application of the theorem.

The example intentionally uses a sum of squared entrywise errors for this negative control because
IEEE
`max` can ignore a NaN operand. That detail is part of the test's meaning: even a diagnostic norm
must choose NaN behavior deliberately.

# Cholesky In PyTorch

`torch.linalg.cholesky` returns the same displayed factor on the positive-definite input
{Informal.citep pytorch2019}[]:

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

Identical factor. On the indefinite one:

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

The failure channels differ. PyTorch raises; TorchLean returns a tensor containing NaN.
`Tensor.cholesky` is a total Lean function, so it must return something of type `Tensor α [n, n]`
for every input, and the scalar backend determines the result of invalid arithmetic.
The output type carries the dimensions, not a success flag or a proof of finite entries. Returning
a tensor therefore says only that the computation produced an inhabitant of that type. It does
not discharge the positive-pivot premise. The indefinite example makes this visible without a
large matrix: its first column is usable, but the second diagonal entry cannot serve as a real
positive square-root pivot. Passing that tensor directly to a later solve would propagate the
failed arithmetic beyond the point at which it first became diagnosable. With the
`Float` backend, callers must check for a failed factorization before using its output.

`cholesky_ex` returns `info = 2`. That is the one-based index of the first non-positive leading
minor, pointing at the same pivot where the Lean run put its NaN. This is runtime diagnostic
data. TorchLean's
$`\forall j,\;0<L_{jj}` is instead a hypothesis on the real specification's pivots for a chosen
matrix. It is not an equivalence theorem between LAPACK's `info` result and those real pivots;
rounding and the different implementation would need their own analysis.

A successful numerical run is commonly described by $`A\approx LL^\top`, but applying such a
statement requires an error measure and bound. The Lean theorem here is exact over
$`\mathbb R`; the `Float` examples only measure residuals. This chapter supplies no proved
backward-error bound for either displayed floating result. The FP32 soundness chapter describes
rounding machinery that a proof for TorchLean's factorization would need.

# Classical Gram-Schmidt As QR

For columns $`a_0,\ldots,a_{k-1}`, classical Gram-Schmidt computes

$$`\begin{aligned}
v_j &= a_j-\sum_{i<j}\langle q_i,a_j\rangle q_i,\\
r_{jj} &= \|v_j\|,\\
q_j &= v_j/r_{jj},\\
r_{ij} &= \langle q_i,a_j\rangle\quad(i<j).
\end{aligned}`

Here $`\langle\cdot,\cdot\rangle` and $`\|\cdot\|` correspond to the real instances of `dotFn`
and `normFn`. Subtracting the projections leaves the part of the new column outside the span of
the earlier columns; normalizing that residual gives the next column of `Q`.

The coefficient uses the original column `a_j` throughout the projection sum. In exact arithmetic,
its component along each earlier orthonormal column can be removed independently, leaving a
residual orthogonal to all of them. That is the classical Gram-Schmidt recurrence in this source.
If the new column is already in their span, the residual is zero and there is no direction to
normalize. A positive `r_jj` both rules out this degeneracy and selects the division branch of the
implementation. Thus the pivot hypothesis describes the step where a new independent feature
direction is actually added to the factor.

The TorchLean specification uses the same column-building pattern as Cholesky. `gramSchmidtFn`
threads lists of `Q` and `R` columns, while `qrQSpec`, `qrRSpec`, and `qrSpec` expose tensor-shaped
results. `qrSpec` runs Gram-Schmidt once to materialize both factors.

The public operation returns a small structure rather than a pair, so the two factors have names:

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
  Tensor α [m, n] → Tensor.QRFactors α m n
```

The fields carry the reduced shapes:

```
-- The shared inner dimension allows the two reduced factors
-- to multiply back to the input shape.
result.q : Tensor α [m, min m n]
result.r : Tensor α [min m n, n]
```

When `n ≤ m`, those shapes reduce to `[m, n]` and `[n, n]`, and the public operation uses the
theorem-backed `qrSpec` computation. For a wide matrix, it builds at most `m` independent basis
columns and does not let an early dependent source column consume a basis slot. That distinction is
necessary: merely truncating the old square-shaped factors can lose a later independent direction
and fail to reconstruct the input. The current `IsQR` theorem is about the tall/square
specification and does not silently certify the separate wide algorithm.

Three separate theorems correspond to the three conjuncts of `IsQR`. Upper-triangularity, like its
Cholesky counterpart, needs no hypothesis:

```lean (name := upperTri)
-- Entries below the R diagonal vanish by construction,
-- independently of residual norms.
#check @Rmat_upper_triangular
```

```leanOutput upperTri (whitespace := lax)
@Rmat_upper_triangular : ∀ {n m : ℕ} (A : Fin m → Fin n → ℝ) {k j : Fin n},
  ↑j < ↑k → Rmat A k j = 0
```

`Rmat` and `Qmat` are the proof layer's readable views of the two factors, defined as plain
functions of the row and column index so that induction over columns does not have to travel through
a tensor. Orthonormality and reconstruction are where the rank hypothesis enters:

```lean (name := qrPos)
-- Positive residual norms support both orthonormality and
-- the complete QR predicate.
#check @QT_mul_Q_eq_one
#check @isQR_of_pos
```

```leanOutput qrPos (whitespace := lax)
@QT_mul_Q_eq_one : ∀ {m n : ℕ} (A : Fin m → Fin n → ℝ),
  (∀ (j : Fin n), 0 < Rmat A j j) →
    ((Matrix.of fun i k => Qmat A i k).transpose * Matrix.of fun i k => Qmat A i k) = 1
```

```leanOutput qrPos (whitespace := lax)
@isQR_of_pos : ∀ {m n : ℕ} (A : Fin m → Fin n → ℝ),
  (∀ (j : Fin n), 0 < Rmat A j j) →
    IsQR (Matrix.of A) (Matrix.of fun i k => Qmat A i k) (Matrix.of fun k j => Rmat A k j)
```

$`\forall j,\;0<R_{jj}` is, for classical Gram-Schmidt, the executable form of full column rank:
every new column has a nonzero component orthogonal to its predecessors. The API again states the
pivot condition directly rather than deriving it from a separately formalized rank predicate, the
same unfinished bridge as on the Cholesky side.

At the tensor boundary the two results are restated through `Spec.get2`:

```lean (name := qrTensor)
-- Inspect the two tensor identities separately: reconstruct
-- entries and compare column dot
-- products.
#check @qrSpec_reconstruction
#check @qrSpec_orthonormal
```

```leanOutput qrTensor (whitespace := lax)
@qrSpec_reconstruction : ∀ {n m : ℕ} (A : Tensor ℝ [m, n]),
  (∀ (j : Fin n), 0 < Spec.get2 (Spec.qrRSpec A) j j) →
    ∀ (i : Fin m) (j : Fin n), Spec.get2 A i j =
      ∑ k, Spec.get2 (Spec.qrQSpec A) i k * Spec.get2 (Spec.qrRSpec A) k j
```

```leanOutput qrTensor (whitespace := lax)
@qrSpec_orthonormal : ∀ {m n : ℕ} (A : Tensor ℝ [m, n]),
  (∀ (j : Fin n), 0 < Spec.get2 (Spec.qrRSpec A) j j) →
    ∀ (a b : Fin n), ∑ i, Spec.get2 (Spec.qrQSpec A) i a * Spec.get2 (Spec.qrQSpec A) i b =
      if a = b then 1 else 0
```

The right-hand side `if a = b then 1 else 0` is $`Q^\top Q=I` written entrywise, and the same
hypothesis appears in both. Positivity of every diagonal residual norm supports both
orthonormality and reconstruction.

For a tall matrix, the identity is about columns: `Q` has shape `[m, n]`, so `Qᵀ Q` has shape
`[n, n]`. It does not say that `Q Qᵀ` is the `[m, m]` identity. When `m > n`, the columns span
only a subspace of the ambient row space. In a least-squares application this distinction explains
why multiplication by `Qᵀ` extracts coordinates and multiplication by `Q` reconstructs the
component in that subspace. Reading `qrSpec_orthonormal` entrywise makes the orientation explicit:
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

Read `R` as three rows: `14 21 -14`, `0 175 -70`, `0 0 35`. Those are the exact textbook values, and
the two structural facts are visible in the print. The strictly lower triangle is zero, which is
`Rmat_upper_triangular` and needs no hypothesis. The diagonal is `14`, `175`, `35`, all positive,
which is a runtime observation, not a proof of positivity for the theorem's real instance.
Positivity is not an accident of
the example: $`r_{jj}=\|v_j\|` and `normFn` is a square root, so the diagonal can never be negative.
Over the reals it can be zero; the positive-pivot hypothesis excludes that case. Floating
execution can also encounter NaN or overflow.

The first column of `Q` is $`(0.857143,0.428571,-0.285714)`, which is
$`(12,6,-4)/14=(6/7,3/7,-2/7)`. The divisor `14` is $`\|a_0\|=\sqrt{144+36+16}=\sqrt{196}`, so it is
also the `R` entry printed above; the two factors share that number by construction.

Now measure the orthonormality that the theorem asserts over $`\mathbb R`:

```lean (name := qrDots)
-- Compare a column with itself and with its neighbor using
-- the same floating dot product.
def qCol (j : Fin 3) : Fin 3 → Float :=
  fun i => Spec.get2 (Tensor.qr classicQR).q i j

#eval Spec.dotFn (qCol 0) (qCol 0)
#eval Spec.dotFn (qCol 0) (qCol 1)
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
#eval Spec.dotFn (qCol 0) (qCol 1) * 1e18
#eval Spec.dotFn (qCol 1) (qCol 2) * 1e18
```

```leanOutput qrDotsScaled
-20.816682
```

```leanOutput qrDotsScaled
-277.555756
```

So the two off-diagonal inner products are $`-2.08\times10^{-17}` and $`-2.78\times10^{-16}`,
which are small on the scale of a binary64 number near one. The real theorem gives zero for the
real factors; these floating factors do not satisfy $`Q^\top Q=I` exactly. Bounding their
deviation requires a
rounding model, which is what the FP32 soundness chapter supplies for other operators and what
the factorization layer does not yet have.

Reconstruction is a different story on this input. Taking the largest entrywise difference between
$`QR` and $`A`, and scaling it the same way to be sure nothing is hiding under the display:

```lean (name := qrResidual)
-- Form each reconstruction entry from a row of Q and a
-- column of R before taking the largest error.
def qRow (i : Fin 3) : Fin 3 → Float :=
  fun k => Spec.get2 (Tensor.qr classicQR).q i k

def rCol (j : Fin 3) : Fin 3 → Float :=
  fun k => Spec.get2 (Tensor.qr classicQR).r k j

def reconResidual : Float :=
  ((List.finRange 3).flatMap fun i =>
      (List.finRange 3).map fun j =>
        (Spec.dotFn (qRow i) (rCol j) -
          Spec.get2 classicQR i j).abs).foldl max 0.0

#eval reconResidual * 1e18
```

```leanOutput qrResidual
0.000000
```

The scaled maximum residual still displays as zero. This gives a more sensitive diagnostic than
the shipped example's `QR A = Q·R: OK (error = 0.000000)`, though a decimal print alone does not
prove bitwise zero. It measures this floating reconstruction of one $`3\times3` matrix.

The two diagnostics should be read independently. Reconstruction forms dot products of rows of
`Q` with columns of `R`, whereas orthonormality forms dot products of pairs of columns of `Q`.
Their cancellations and rounding steps differ. A factor can therefore reproduce this input very
closely and still have nonzero cross-column dot products. Scaling a diagnostic before printing
reveals values hidden by six decimal places; it does not change the computed factors or improve
their accuracy. For a model that repeatedly uses an orthogonal basis, such as a projection layer,
the cross-column error is directly relevant even when the reconstruction residual looks smaller.

`torch.linalg.qr` on the same matrix in float64 makes a specific and useful contrast:

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
`j` is chosen for numerical reasons that leave $`r_{jj}` negative here; $`(-Q)(-R)=QR`, so this is a
perfectly good factorization of the same matrix. The predicate `IsQR` itself allows negative
diagonal entries. The implementation theorems
`isQR_of_pos` and `qrSpec_orthonormal` concern TorchLean's own real Gram-Schmidt factors, not
arbitrary
factors returned by LAPACK. Relating those results requires more than checking or normalizing the
signs: one must connect the algorithms and account for floating arithmetic.

In the displayed run, LAPACK's reconstruction error is $`2.8\times10^{-14}`, while the
Gram-Schmidt residual displays as zero even after scaling. The measured
orthonormality is $`2.2\times10^{-16}` for LAPACK against $`2.8\times10^{-16}` for Gram-Schmidt.
These measurements on one $`3\times3` matrix do not compare the algorithms' stability.
Classical Gram-Schmidt can lose orthogonality as columns become nearly dependent. A stability
analysis must relate the error to the input and the arithmetic across that wider class of cases.

# Rank-Deficient Inputs

Take the smallest rank-deficient case, a $`2\times2` matrix whose second column is twice its first:

```lean (name := deficientRun)
-- The dependent second column leaves a zero residual for
-- the guarded normalization branch.
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
  fun i => Spec.get2 (Tensor.qr deficient2).q i j

#eval Spec.dotFn (dCol 1) (dCol 1)
```

```leanOutput deficientDot
0.000000
```

The zero column explains why the shipped example reports
`error = 1.000000` for this case: $`Q^\top Q` is $`\operatorname{diag}(1,0)`, which differs from the
identity by exactly one in one entry. The pivot hypothesis fails at `j = 1`, where
$`r_{11}=\|v_1\|=0`, and the executable code takes the guarded branch that emits `0` instead of
dividing.

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
TorchLean's guarded Gram-Schmidt branch emits a zero column when the residual norm is zero.

The missing column direction is unconstrained by reconstruction in this example. The second
row of `R` is zero, so changing the second column of `Q` contributes nothing to `QR`. It still
changes `Qᵀ Q`, which explains how the displayed implementations can reconstruct the same matrix
while producing different orthonormality results. A consumer that needs a full basis must check
that property of the returned `Q`, rather than infer it from reconstruction alone.

For TorchLean's real Gram-Schmidt factors, $`0<R_{jj}` rules out that zero-column branch.
This is a hypothesis about those particular factors. It is not a portable success test for
arbitrary QR implementations, whose nonzero diagonal entries can have either sign.

The compiled example runs four more variations, including a wide matrix whose second column depends
on its first and whose third supplies a later independent direction:

```terminal
# The same shipped executable reports wide and
# rank-deficient cases separately.
lake exe torchlean factorizations
```

```terminal +output
QR A = Q·R: OK (error = 0.000000)
QR Qᵀ·Q = I: OK (error = 0.000000)
QR(wide) A = Q·R: OK (error = 0.000000)
QR(wide) Qᵀ·Q = I: OK (error = 0.000000)
QR(rank-deficient) A = Q·R still reconstructs: OK (error = 0.000000)
QR(rank-deficient) Qᵀ·Q = I correctly fails (needs full column rank):
  OK (correctly rejected, error = 1.000000 ≥ 0.500000)
```

The wide result has `q : Tensor Float [2, 2]` and `r : Tensor Float [2, 3]`, and both types are
known when Lean elaborates the program rather than when it runs. As a further variation, duplicate
any column of the good matrix: the reconstruction diagnostic may remain small while the
orthonormality diagnostic can fail; floating cancellation can also leave tiny nonzero residual
columns instead of the exact zero columns predicted over the reals. Reconstruction observed on one
rank-deficient example does not
discharge the positive-pivot hypothesis of the general theorem.

# Triangular Solves And Ridge Regression

Once $`A=LL^\top`, a linear system is two triangular substitutions: forward-solve $`Lz=b`, then
back-solve $`L^\top x=z`. TorchLean ships that path, and the ridge variant that adds
$`\gamma I` first:

```lean (name := ridgeSig)
-- The ridge interface accepts a square kernel, scalar
-- regularization, and matching target vector.
#check @Spec.solveRidgeSpec
```

```leanOutput ridgeSig (whitespace := lax)
@Spec.solveRidgeSpec : {α : Type} → [inst : Storage α] → [Context α] → {n : ℕ} →
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

Substitution explains how the factor is used. The first triangular system computes each entry
of `z` from earlier entries, dividing by a diagonal entry of `L`. The second traverses the
transposed factor in reverse order to obtain `x`. Adding `γ I` changes only the diagonal of the
matrix being factored; it does not change the supplied target vector. In kernel ridge regression,
that vector contains the observed targets, while the returned vector contains coefficients used
for predictions through the kernel. A full correctness statement for this path would connect
the returned coefficients to `(K + γ I) x = b`, including the conditions needed for both
substitutions to divide safely.

The triangular solves, `cholSolveFn`, and `solveRidgeSpec` are currently executable APIs without
correctness theorems. The `#eval` checks one input. Kernel ridge regression and Gaussian process
code use this path, so proving the factorization alone does not establish correctness of their
linear solves.

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
  * public computation with statically checked input and reduced output shapes; the tall/square QR
    branch shares the proved reference definition
*
  * `IsCholesky`, `IsQR`
  * `ℝ`
  * exact algebraic specification
*
  * `choleskySpec`, `qrQSpec`, `qrRSpec` in the proofs
  * `ℝ`
  * exact reconstruction under pivot hypotheses
*
  * factorization examples
  * `Float`
  * executable residual checks on concrete matrices
:::

The `Float` output is evidence that the executable definitions behave as expected on those inputs.
It is not the proof of $`A=LL^\top` or $`Q^\top Q=I`; the $`-2.08\times10^{-17}` above is a concrete
demonstration that machine arithmetic does not generally satisfy those identities. Conversely, the
real theorem does not prove a forward-error or backward-error bound for the Float execution.

Flocq {Informal.citep flocq2011}[] provides a floating-point formalization in Rocq with rounding
predicates and error lemmas. Such machinery supports a separate question: how far can a rounded
factorization deviate from its exact-$`\mathbb R` counterpart? TorchLean's factorization
development first proves the algebra over $`\mathbb R`, where it can reuse `Matrix`,
`EuclideanSpace`, and `gramSchmidtNormed_orthonormal'`. The comparison with floating arithmetic
remains to be proved; an arithmetic formalization alone does not supply an algorithm's error bound.

The strict-array `@[implemented_by]` paths also need a connection to the proved definitions.
`choleskyColsFn`, `cholSolveFn`, and `solveRidgeFn` each carry that attribute,
so the Cholesky and ridge-solve numerical examples use internal array implementations while
their reference definitions remain visible to Lean proofs. QR uses its separate Gram-Schmidt
definition. The reference and array implementations are intended to compute the same recurrence,
but no equivalence theorem has been proved. The arrays retain computed columns; reading the factor
from the closure representation can repeatedly recompute earlier column prefixes.

For an audit, the reconstruction identities are collected in
{src "NN/Proofs/Tensor/Basic/FactorizationsReconstruction.lean"}[`FactorizationsReconstruction`],
while the orthonormal-column statements are in
{src "NN/Proofs/Tensor/Basic/FactorizationsOrthonormal.lean"}[`FactorizationsOrthonormal`].
The executable definitions themselves remain in
{src "NN/Spec/Core/Tensor/Factorizations.lean"}[`Tensor.Factorizations`].
The public executable entry points are in
{src "NN/Tensor/LinearAlgebra.lean"}[`Tensor.LinearAlgebra`].
Keeping those three roles separate makes it clear whether a cited result is a definition, an exact
real identity, or evidence from a concrete Float run.

# Factorization Proof Gaps

The exact reconstruction developments are substantial, but they are not a complete verified
numerical linear algebra package. The most useful next theorems are:

1. positive definiteness implies positive executable Cholesky pivots;
2. full column rank implies positive executable Gram-Schmidt pivots;
3. reconstruction and orthonormality theorems for the public wide reduced QR branch;
4. correctness of the triangular and ridge-solve helpers, which currently have none;
5. equivalence of the `@[implemented_by]` array implementations with the closure definitions the
   proofs use;
6. finite-precision stability bounds, especially for loss of orthogonality in classical
   Gram-Schmidt; comparing its factors with Householder QR also requires accounting for their
   different sign conventions.

The exact identities follow the standard mathematics in Golub and Van Loan's *Matrix
Computations*. The distinction between exact factorization and floating-point stability follows
Higham's *Accuracy and Stability of Numerical Algorithms*, which is also the reference for the
loss-of-orthogonality behavior that item six would have to quantify. TorchLean proves exact
reconstruction for its real specifications and tests concrete `Float` residuals; it does not yet
prove stability bounds.
