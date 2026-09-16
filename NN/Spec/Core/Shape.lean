/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Internal.Representation.Shape


/-!
# Shapes (`Spec.Shape`)

`Shape` is the type-level shape descriptor for tensors.

TorchLean uses *shape-indexed tensors*:

`Tensor α s`

so Lean checks shape compatibility before tensor code can run.

## Representation

A shape *is* a list of dimensions, outermost first, so model and application code writes one as
ordinary bracket notation:

- `[]` for a scalar
- `[n]` for a vector
- `[m, n]` for a matrix

`Shape` is a reducible abbreviation for `Tensor.Internal.Shape`, which is `List Nat`, so the
spec-level shape and the shape carried by a tensor buffer are the same type. There is no conversion
between them and no cast to transport a tensor across the two views.

Shape-recursive definitions and proofs are still written with the two names `Shape.scalar` and
`Shape.dim`, which are `@[match_pattern]` abbreviations for `[]` and `·  :: ·`. They may be used in
patterns exactly like constructors, and `induction`/`cases` on a shape offer the cases `scalar` and
`dim` through the eliminators registered below.

## Common utilities

- `Spec.Shape.size : Shape → Nat` is the total number of scalar elements (“numel”).
- `Spec.Shape.rank : Shape → Nat` is the number of axes.

PyTorch analogy:

- a shape itself corresponds to `tensor.shape` (a tuple of dimensions).
- `Spec.Shape.rank s` corresponds to `tensor.ndim`.
- `Spec.Shape.size s` corresponds to `tensor.numel()`.

## Broadcasting and axes

Broadcasting is encoded by the decidable proposition `CanBroadcastTo` and its typeclass wrapper
`BroadcastTo`. Because the relation is a proposition, tensor operations depend only on the two
shapes and never on how the relation was proved.

This is an intentionally *asymmetric* relation ("broadcast `s1` to `s2`"), because most tensor code
is naturally written by choosing the output shape and requiring each input to broadcast to it.

The typeclass wrapper `BroadcastTo` keeps higher-level specs readable: in many cases Lean can infer
the broadcast evidence automatically, so call sites do not have to thread proofs around by hand.

It also defines axis-validity helpers (`NonemptyAxis`) and a `wellFormed` predicate for “all
dimensions are positive”, which is useful when you want to rule out degenerate cases in proofs.
-/

@[expose] public section


namespace Spec

/--
Tensor shape descriptor used to index spec-level tensors (`TorchLean.Tensor α s`).

Use outermost-first bracket notation: `[]`, `[n]`, `[m, n]`, and so on. This is the same type as
the shape carried by a tensor buffer, `Tensor.Internal.Shape`, and therefore the same type as
`List Nat`.
-/
abbrev Shape := TorchLean.Tensor.Internal.Shape

namespace Shape

/-- The shape of a scalar: no axes. Usable in patterns, like a constructor. -/
@[match_pattern] abbrev scalar : Shape := []

/-- A shape with outermost axis of extent `n` over `s`. Usable in patterns, like a constructor. -/
@[match_pattern] abbrev dim (n : Nat) (s : Shape) : Shape := n :: s

universe u

/--
Recursion on a shape, one axis at a time.

This is the eliminator `induction s` uses, so the cases are named `scalar` and `dim` and the tail
of a `dim` is again a `Shape` rather than a `List Nat`. Ordinary list induction is unaffected: a
hypothesis whose type is spelled `List Nat` still eliminates to `nil` and `cons`.
-/
@[elab_as_elim, induction_eliminator]
protected def recAux {motive : Shape → Sort u} (scalar : motive .scalar)
    (dim : (n : Nat) → (s : Shape) → motive s → motive (.dim n s)) : (s : Shape) → motive s
  | .scalar => scalar
  | .dim n s => dim n s (Shape.recAux scalar dim s)

/-- Case analysis on a shape, with the cases named `scalar` and `dim`. -/
@[elab_as_elim, cases_eliminator]
protected def casesAux {motive : Shape → Sort u} (scalar : motive .scalar)
    (dim : (n : Nat) → (s : Shape) → motive (.dim n s)) : (s : Shape) → motive s
  | .scalar => scalar
  | .dim n s => dim n s

end Shape

/-!
Model code writes shapes as dimension lists. For example, `Tensor Float [4, 2]` is a four-by-two
tensor of Lean `Float` values, while `Trainer.Dataset [2] [1]` describes supervised samples with
two input values and one target value. The expected `Shape` type directs Lean to elaborate the list
through `Shape.ofList`.

The brackets are shape notation, not tensor storage: a value of type `Tensor Float [4, 2]` is still
a `Tensor`, never a `List`. During elaboration the notation reduces to the recursive shape below,
preserving the definitional equalities used by tensor programs and proofs. Ordinary model code
should not write the expanded `.dim` form.
-/
namespace Shape

/--
Output length of a floor-mode sliding window with symmetric padding.

For positive `kernel` and `stride` with
$\mathtt{kernel}\le\mathtt{input}+2\mathtt{padding}$, this is
$(\mathtt{input}+2\mathtt{padding}-\mathtt{kernel})/\mathtt{stride}+1$. Invalid geometry has
length zero, so saturated natural-number subtraction and division by zero cannot create a phantom
output element.
-/
def slidingWindowOutDim (input kernel stride padding : Nat) : Nat :=
  let padded := input + 2 * padding
  if kernel = 0 || stride = 0 || padded < kernel then
    0
  else
    (padded - kernel) / stride + 1

/-- Effective extent of a dilated kernel along one axis. -/
def dilatedKernelExtent (kernel dilation : Nat) : Nat :=
  if kernel = 0 then 0 else dilation * (kernel - 1) + 1

/-- Output length of a dilated sliding window with independent padding on each side.
Invalid geometry has length zero, as in `slidingWindowOutDim`. -/
def slidingWindowOutDimDilated
    (input kernel stride dilation paddingBefore paddingAfter : Nat) : Nat :=
  let effective := dilatedKernelExtent kernel dilation
  let padded := input + paddingBefore + paddingAfter
  if effective = 0 || stride = 0 || padded < effective then
    0
  else
    (padded - effective) / stride + 1

/--
Read a dimension list as a shape.

The two types are equal, so this is the identity. It is kept because a great deal of code names it
explicitly at the boundary where dimensions arrive as a list.
-/
abbrev ofList (dims : List Nat) : Shape := dims

/-- Build a shape from runtime dimensions stored outermost first. -/
def ofArray (dims : Array Nat) : Shape :=
  dims.foldr (fun extent rest => .dim extent rest) .scalar

/--
View a shape as its dimension list.

The two types are equal, so this is the identity. It is kept because front ends and bridges name it
explicitly where a list is the natural spelling.
-/
@[reducible] def toList (s : Shape) : List Nat := s

/-- Print a shape using the same dimension-list convention as model code. -/
def pretty (s : Shape) : String :=
  "[" ++ String.intercalate ", " (s.toList.map toString) ++ "]"

-- `Repr` and `ToString` come from `List Nat` and already print `[2, 3]`; a second instance on the
-- same type would shadow them for every list of naturals.

/-- Swap two adjacent dimensions at a given depth (0‑based from the outermost). -/
@[reducible] def swapAdjacentAtDepth (s : Shape) (depth : Nat) : Shape :=
  match depth, s with
  | 0, .dim m (.dim n rest) => .dim n (.dim m rest)
  | d+1, .dim m rest => .dim m (swapAdjacentAtDepth rest d)
  | _, _ => s  -- invalid depth, return unchanged

/-- At rank two, swapping at depth zero is the ordinary matrix transpose of the shape. -/
@[simp] theorem swapAdjacentAtDepth_zero_rank_two (m n : Nat) :
    swapAdjacentAtDepth [m, n] 0 = [n, m] := rfl

/-- Swapping adjacent dims at depth `depth` twice returns the original shape. -/
@[simp] theorem swapAdjacentAtDepth_involutive (s : Shape) (depth : Nat) :
    (s.swapAdjacentAtDepth depth).swapAdjacentAtDepth depth = s := by
  induction depth generalizing s with
  | zero =>
      cases s with
      | scalar => simp [swapAdjacentAtDepth]
      | dim m rest =>
          cases rest <;> simp [swapAdjacentAtDepth]
  | succ d ih =>
      cases s <;> simp [swapAdjacentAtDepth, ih]

/-- Shape obtained by applying adjacent-axis swaps from left to right. -/
def applyAdjacentSwaps : Shape → List Nat → Shape
  | s, [] => s
  | s, depth :: depths => applyAdjacentSwaps (s.swapAdjacentAtDepth depth) depths

/-- Adjacent swaps that move `axis` to the innermost position of a rank-`rank` shape. -/
def moveAxisToInnermostSwaps (rank axis : Nat) : List Nat :=
  (List.range (rank - (axis + 1))).map (axis + ·)

/-- Applying a concatenated list of swaps is applying the two halves in order.

Transposition permutations are built up as lists of adjacent swaps, so this is the lemma that lets a
composite permutation be reasoned about one factor at a time. -/
@[simp]
theorem applyAdjacentSwaps_append (s : Shape) (xs ys : List Nat) :
    applyAdjacentSwaps s (xs ++ ys) = applyAdjacentSwaps (applyAdjacentSwaps s xs) ys := by
  induction xs generalizing s with
  | nil => rfl
  | cons depth depths ih =>
      simp only [List.cons_append, applyAdjacentSwaps]
      exact ih (s.swapAdjacentAtDepth depth)

/-- Replaying adjacent-axis swaps in reverse order restores the original shape. -/
@[simp]
theorem applyAdjacentSwaps_reverse (s : Shape) (depths : List Nat) :
    applyAdjacentSwaps (applyAdjacentSwaps s depths) depths.reverse = s := by
  induction depths generalizing s with
  | nil => rfl
  | cons depth depths ih =>
      simp only [applyAdjacentSwaps, List.reverse_cons, applyAdjacentSwaps_append]
      rw [ih]
      exact swapAdjacentAtDepth_involutive s depth

/-- Append a new innermost dimension. -/
@[reducible]
def appendDim (s : Shape) (n : Nat) : Shape :=
  match s with
  | .scalar => .dim n .scalar
  | .dim m rest => .dim m (appendDim rest n)

/-- Add a new outermost dimension. -/
@[reducible]
def prependDim (s : Shape) (n : Nat) : Shape :=
  .dim n s

/-- Concatenate two shapes, preserving the dimensions of the first shape as leading axes. -/
@[reducible]
def concat : Shape → Shape → Shape
  | .scalar, suffix => suffix
  | .dim n rest, suffix => .dim n (concat rest suffix)

/--
Decide whether two shapes are equal.

`Shape` is an abbreviation for `List Nat`, so equality is decidable by the usual list instance.
Reach for this function rather than `if h : s = t` when either side is bound by a local `let`:
instance search on such a binding can get stuck, while a direct call always elaborates.
-/
def decEq (s t : Shape) : Decidable (s = t) :=
  inferInstanceAs (Decidable (s = t))

/-- Concatenating the scalar shape leaves the leading shape unchanged. -/
@[simp] theorem concat_scalar (shape : Shape) :
    shape.concat .scalar = shape := by
  induction shape with
  | scalar => rfl
  | dim n rest ih => simp only [concat, ih]

/-- Shape concatenation is associative. -/
@[simp] theorem concat_assoc (left middle right : Shape) :
    (left.concat middle).concat right = left.concat (middle.concat right) := by
  induction left with
  | scalar => rfl
  | dim n rest ih => simp only [concat, ih]

/-- Shape concatenation is list append. -/
theorem concat_eq_append (left right : Shape) : left.concat right = left ++ right := by
  induction left with
  | scalar => rfl
  | dim n rest ih => simp only [concat, ih, List.cons_append]

/--
Reversing a concatenation reverses each part and swaps them.

Shape parsers read a shape back to front (innermost axis first), so this is the lemma that turns a
`leading ++ [rows, cols]` layout into the `cols :: rows :: rest` pattern they match on. It is stated
about `reverse` rather than `concat` so that a simp set naming it leaves the other occurrences of
`concat` alone. It carries no `simp` attribute for the same reason: `concat` is the normal form for
a composed shape, and rewriting every occurrence away would strand the lemmas stated about it.
-/
theorem reverse_concat (left right : Shape) :
    (left.concat right).reverse = right.reverse ++ left.reverse := by
  rw [concat_eq_append, List.reverse_append]

/-- Appending one dimension is concatenation with a one-axis suffix. -/
theorem appendDim_eq_concat (s : Shape) (n : Nat) :
    s.appendDim n = s.concat (.dim n .scalar) := by
  induction s with
  | scalar => rfl
  | dim m rest ih => simp only [appendDim, concat, ih]

/-- Appending two dimensions is concatenation with a two-axis suffix. -/
theorem appendDim_appendDim_eq_concat (s : Shape) (m n : Nat) :
    (s.appendDim m).appendDim n = s.concat (.dim m (.dim n .scalar)) := by
  induction s with
  | scalar => rfl
  | dim k rest ih => simp only [appendDim, concat, ih]

/-- Appending a final dimension commutes with adding a fixed leading shape. -/
@[simp]
theorem concat_appendDim (leading suffix : Shape) (n : Nat) :
    (leading.concat suffix).appendDim n = leading.concat (suffix.appendDim n) := by
  induction leading with
  | scalar => rfl
  | dim m rest ih =>
      simp only [concat, appendDim, ih]

/-- Total number of scalar elements (a.k.a. “numel”). -/
def size : Shape → Nat
  | .scalar => 1
  | .dim n rest => n * size rest

/-- The number of entries is the product of the dimensions.

Not a `simp` lemma: `size` is the normal form for a shape's element count, and rewriting it to a
list product would strand the many lemmas stated about `size`. -/
theorem size_eq_prod (s : Shape) : size s = s.prod := by
  induction s with
  | scalar => rfl
  | dim n rest ih => simp [size, ih]

/--
`appendDim` multiplies the number of scalar elements by the appended dimension.

This lemma is the standard justification for reshape tricks where we:
- treat a tensor of shape `s.appendDim n` as a matrix of shape `(size s) × n`, or
- append an extra singleton dimension (`n = 1`) without changing `size`.
-/
theorem size_appendDim (s : Shape) (n : Nat) : size (appendDim s n) = size s * n := by
  induction s with
  | scalar =>
      simp [size]
  | dim m rest ih =>
      -- `appendDim` recurses to the innermost dimension; `size` is multiplicative.
      simp [size, ih, Nat.mul_assoc]

/--
`prependDim` multiplies the number of scalar elements by the new outermost dimension.

This is the counterpart of `size_appendDim` for the front of a shape, and unlike that lemma it holds
by definition: `size` already recurses on the outermost axis.
-/
theorem size_prependDim (s : Shape) (n : Nat) : size (prependDim s n) = n * size s := rfl

/-- The number of elements in a concatenated shape is the product of the two shape sizes. -/
theorem size_concat (leading suffix : Shape) :
    size (concat leading suffix) = size leading * size suffix := by
  induction leading with
  | scalar => simp [size]
  | dim n rest ih => simp [size, ih, Nat.mul_assoc]

-- Tell `grind` about the standard shape normalization lemmas.
attribute [grind =] size_appendDim

/-- Convert to an array of dimensions (outermost first). -/
def toArray (s : Shape) : Array Nat :=
  toList s |>.toArray

/-- Boolean structural equality test for shapes.

`BEq Shape` is the lawful instance on `List Nat`. This explicit recursive test is kept for code that
wants to inspect the comparison directly. -/
def areEqual : Shape → Shape → Bool
  | .scalar, .scalar => true
  | .dim n1 s1, .dim n2 s2 => n1 == n2 && areEqual s1 s2
  | _, _ => false

/-- The structural test agrees with propositional equality. -/
@[simp] theorem areEqual_eq_true_iff : ∀ {s t : Shape}, areEqual s t = true ↔ s = t
  | .scalar, .scalar => by simp [areEqual]
  | .scalar, .dim _ _ => by simp [areEqual]
  | .dim _ _, .scalar => by simp [areEqual]
  | .dim n1 s1, .dim n2 s2 => by
      simp [areEqual, areEqual_eq_true_iff (s := s1) (t := s2)]

/-- The structural test is the derived boolean equality. -/
theorem areEqual_eq_beq (s t : Shape) : areEqual s t = (s == t) := by
  by_cases h : s = t
  · subst h
    exact (areEqual_eq_true_iff.mpr rfl).trans (beq_self_eq_true s).symm
  · rw [Bool.eq_iff_iff, areEqual_eq_true_iff, beq_iff_eq]

/-- Get dimension at index `i` (0‑based), or `none` if out of bounds. -/
def getDim : Shape → Nat → Option Nat
  | .scalar, _ => none
  | .dim n _, 0 => some n
  | .dim _ rest, i+1 => getDim rest i

/- Broadcasting support -/
/-!
### Typeclass-friendly broadcasting (`BroadcastTo`)

The `CanBroadcastTo` relation is asymmetric (“broadcast `s₁` *to* `s₂`”), matching how most
operations are written: we pick a target shape and require each operand to broadcast to it.

The `BroadcastTo` wrapper lets Lean search for a broadcast proof automatically, which is convenient
for higher-level specs (layers/models) where the broadcasting details are not the point.

PyTorch analogy:

- PyTorch broadcasting aligns shapes from the *trailing* dimensions by implicitly prepending `1`s
  to the shorter shape.
- Our `Shape` is an outermost-first tree, so the corresponding operation is `expand_dims`:
  it inserts leading/outer dimensions to reach the target rank (this is the "prepend `1`s" step).
- `dim_1_to_n` corresponds to PyTorch's "dimension 1 can expand to n" rule.
-/

/-- Rank = number of dimensions (scalar has rank 0). -/
def rank : Shape → Nat
  | Shape.scalar => 0
  | Shape.dim _ rest => rank rest + 1

/-- The rank is the number of dimensions.

Not a `simp` lemma: `rank` is the normal form for a shape's number of axes. -/
public theorem rank_eq_length (s : Shape) : rank s = s.length := by
  induction s with
  | scalar => rfl
  | dim _ rest ih => simp [rank, ih]

/-- Insert a dimension at an axis, where axis `0` is outermost.

Callers that construct a tensor of this shape also carry a proof that the axis does not exceed the
input rank. The out-of-bounds scalar case is therefore unreachable in typed tensor operations.
-/
@[reducible] def insertAxis : Shape → Nat → Nat → Shape
  | shape, axis, extent =>
      match axis with
      | 0 => .dim extent shape
      | axis + 1 =>
          match shape with
          | .dim n rest => .dim n (insertAxis rest axis extent)
          | .scalar => .scalar

/-- Inserting at axis zero adds a new outermost dimension. -/
@[simp] theorem insertAxis_zero (shape : Shape) (extent : Nat) :
    insertAxis shape 0 extent = .dim extent shape := by
  cases shape <;> rfl

/-- Appending one dimension increases the rank by one. -/
@[simp] theorem rank_appendDim (s : Shape) (n : Nat) :
    rank (s.appendDim n) = rank s + 1 := by
  induction s with
  | scalar => rfl
  | dim _ rest ih => simp only [rank, ih]

/-- Prepending one dimension increases the rank by one. -/
@[simp] theorem rank_prependDim (s : Shape) (n : Nat) :
    rank (s.prependDim n) = rank s + 1 := rfl

/-- A shape decomposed into a leading prefix and a suffix of a prescribed rank. -/
structure SuffixSplit (shape : Shape) (suffixRank : Nat) where
  /-- Axes preceding the suffix. -/
  leading : Shape
  /-- Extents of the suffix axes. -/
  suffix : List Nat
  /-- The suffix has the requested number of axes. -/
  suffix_length : suffix.length = suffixRank
  /-- The decomposition reconstructs the original shape. -/
  concat_eq : leading.concat (ofList suffix) = shape

/-- Split the final `suffixRank` axes from a shape. -/
def splitSuffix (shape : Shape) (suffixRank : Nat) (h : suffixRank ≤ shape.rank) :
    SuffixSplit shape suffixRank := by
  let dims := shape.toList
  have hdims : suffixRank ≤ dims.length := by simpa [dims, rank_eq_length] using h
  let split := dims.length - suffixRank
  let suffixDims := dims.drop split
  have hsuffix : suffixDims.length = suffixRank := by
    simp [suffixDims, split, List.length_drop, Nat.sub_sub_self hdims]
  refine
    { leading := ofList (dims.take split)
      suffix := suffixDims
      suffix_length := hsuffix
      concat_eq := ?_ }
  rw [concat_eq_append]
  simp [split, suffixDims, dims]

/-- Swap the first two axes after an arbitrary fixed leading shape. -/
@[simp]
theorem swapAdjacentAtDepth_concat_rank (leading suffix : Shape) (m n : Nat) :
    swapAdjacentAtDepth (leading.concat (.dim m (.dim n suffix))) leading.rank =
      leading.concat (.dim n (.dim m suffix)) := by
  induction leading with
  | scalar => rfl
  | dim _ tail ih =>
      simp only [concat, rank, swapAdjacentAtDepth, ih]

/-- Replace every dimension by one while preserving the rank of a shape. -/
def singletonAxes : Shape → Shape
  | .scalar => .scalar
  | .dim _ rest => .dim 1 (singletonAxes rest)

/-- Collapsing every axis to length one leaves the rank alone. -/
@[simp] theorem rank_singletonAxes (s : Shape) : rank (singletonAxes s) = rank s := by
  induction s <;> simp [singletonAxes, rank, *]

/-- A shape whose every axis has length one holds exactly one element. -/
@[simp] theorem size_singletonAxes (s : Shape) : size (singletonAxes s) = 1 := by
  induction s <;> simp [singletonAxes, size, *]

/-- Proposition used by broadcast constructors that align two existing dimensions. -/
class SameRank (s₁ s₂ : Shape) : Prop where
  /-- The two shapes have the same number of dimensions. -/
  rank_eq : rank s₁ = rank s₂

instance : SameRank .scalar .scalar := ⟨rfl⟩

/-- Every shape has the same rank as itself. -/
instance sameRankRefl (s : Shape) : SameRank s s := ⟨rfl⟩

instance {s₁ s₂ : Shape} {n₁ n₂ : Nat} [tail : SameRank s₁ s₂] :
    SameRank (.dim n₁ s₁) (.dim n₂ s₂) :=
  ⟨by simp [rank, tail.rank_eq]⟩

/-!
### The broadcast relation (`CanBroadcastTo`)

`CanBroadcastTo source target` is a proposition on the two shapes, so every tensor operation that
consumes it depends only on the shapes and never on how the relation was proved. It is defined by
recursion on the target shape, which makes it decidable (`canBroadcastTo?`, `decide`), and the
structural rules of NumPy and PyTorch are recovered as theorems: `CanBroadcastTo.scalar`,
`CanBroadcastTo.dim_eq`, `CanBroadcastTo.dim_1_to_n`, and `CanBroadcastTo.expand_dims`.

The equal-dimension rules require equal-rank tails, so every rank difference is resolved by
`expand_dims` before extents are compared. The internal tensor layer states the same relation on
dimension lists (`List.Forall₂` after padding the source with leading ones);
`CanBroadcastTo.forall₂_toList` in the broadcasting module connects the two forms.
-/

/-- `CanBroadcastTo source target` holds when `source` broadcasts to `target` with right-aligned
axes: after prepending singleton axes to reach the target rank, every source extent equals the
target extent or is one. -/
def CanBroadcastTo : Shape → Shape → Prop
  | .scalar, .scalar => True
  | .dim _ _, .scalar => False
  | .scalar, .dim _ target => CanBroadcastTo .scalar target
  | .dim m source, .dim n target =>
      if source.rank = target.rank then
        (m = n ∨ m = 1) ∧ CanBroadcastTo source target
      else
        CanBroadcastTo (.dim m source) target

/-- A scalar broadcasts to a scalar. -/
@[simp] theorem canBroadcastTo_scalar_scalar : CanBroadcastTo .scalar .scalar :=
  trivial

/-- Nothing with an axis broadcasts down to a scalar; broadcasting only ever adds extent. -/
@[simp] theorem not_canBroadcastTo_dim_scalar {n : Nat} {s : Shape} :
    ¬ CanBroadcastTo (.dim n s) .scalar :=
  fun h => h

/-- A scalar broadcasts across a new outer axis exactly when it broadcasts to the tail. -/
@[simp] theorem canBroadcastTo_scalar_dim {n : Nat} {t : Shape} :
    CanBroadcastTo .scalar (.dim n t) ↔ CanBroadcastTo .scalar t :=
  Iff.rfl

/-- Equal-rank tails compare the leading extents and recurse. -/
theorem canBroadcastTo_dim_dim_of_rank_eq {m n : Nat} {s t : Shape} (hRank : s.rank = t.rank) :
    CanBroadcastTo (.dim m s) (.dim n t) ↔ (m = n ∨ m = 1) ∧ CanBroadcastTo s t := by
  show (if s.rank = t.rank then (m = n ∨ m = 1) ∧ CanBroadcastTo s t
    else CanBroadcastTo (.dim m s) t) ↔ _
  rw [ite_eq_left hRank]

/-- A target of larger rank absorbs its leading axis before the extents are compared. -/
theorem canBroadcastTo_dim_dim_of_rank_ne {m n : Nat} {s t : Shape} (hRank : s.rank ≠ t.rank) :
    CanBroadcastTo (.dim m s) (.dim n t) ↔ CanBroadcastTo (.dim m s) t := by
  show (if s.rank = t.rank then (m = n ∨ m = 1) ∧ CanBroadcastTo s t
    else CanBroadcastTo (.dim m s) t) ↔ _
  rw [ite_eq_right hRank]

/-- The broadcast relation is decidable by the same recursion that defines it. -/
instance instDecidableCanBroadcastTo : (s t : Shape) → Decidable (CanBroadcastTo s t)
  | .scalar, .scalar => isTrue trivial
  | .dim _ _, .scalar => isFalse not_canBroadcastTo_dim_scalar
  | .scalar, .dim _ target =>
      haveI := instDecidableCanBroadcastTo .scalar target
      decidable_of_iff _ canBroadcastTo_scalar_dim.symm
  | .dim m source, .dim _ target =>
      if hRank : source.rank = target.rank then
        haveI := instDecidableCanBroadcastTo source target
        decidable_of_iff _ (canBroadcastTo_dim_dim_of_rank_eq hRank).symm
      else
        haveI := instDecidableCanBroadcastTo (.dim m source) target
        decidable_of_iff _ (canBroadcastTo_dim_dim_of_rank_ne hRank).symm

/-- Decide the broadcast relation at runtime, returning the proof when it holds.

IR passes and dynamic lowerings that only know shapes at runtime use this instead of trusting that
declared input and output shapes are compatible. -/
def canBroadcastTo? (s t : Shape) : Option (PLift (CanBroadcastTo s t)) :=
  if h : CanBroadcastTo s t then some ⟨h⟩ else none

/-- `canBroadcastTo?` succeeds exactly when the relation holds. -/
theorem canBroadcastTo?_isSome_iff {s t : Shape} :
    (canBroadcastTo? s t).isSome ↔ CanBroadcastTo s t := by
  unfold canBroadcastTo?
  split <;> simp_all

/-- Broadcasting never lowers the rank. -/
theorem CanBroadcastTo.rank_le {s t : Shape} (h : CanBroadcastTo s t) : s.rank ≤ t.rank := by
  induction t generalizing s with
  | scalar =>
      cases s with
      | scalar => exact Nat.le_refl _
      | dim _ _ => exact absurd h not_canBroadcastTo_dim_scalar
  | dim n t ih =>
      cases s with
      | scalar => exact Nat.zero_le _
      | dim m s =>
          by_cases hRank : s.rank = t.rank
          · simp [rank, hRank]
          · have := ih ((canBroadcastTo_dim_dim_of_rank_ne hRank).mp h)
            simp only [rank] at this ⊢
            grind

/-- Scalar shapes agree. Higher-rank scalar broadcasts are built with `expand_dims`. -/
theorem CanBroadcastTo.scalar : CanBroadcastTo .scalar .scalar :=
  trivial

/-- Matching outer dimensions preserve broadcasting of equal-rank tails. -/
theorem CanBroadcastTo.dim_eq {n : Nat} {s₁ s₂ : Shape} [same : SameRank s₁ s₂]
    (tail : CanBroadcastTo s₁ s₂) : CanBroadcastTo (.dim n s₁) (.dim n s₂) :=
  (canBroadcastTo_dim_dim_of_rank_eq same.rank_eq).mpr ⟨Or.inl rfl, tail⟩

/-- An outer dimension of length one can expand to any target length. -/
theorem CanBroadcastTo.dim_1_to_n {n : Nat} {s₁ s₂ : Shape} [same : SameRank s₁ s₂]
    (tail : CanBroadcastTo s₁ s₂) : CanBroadcastTo (.dim 1 s₁) (.dim n s₂) :=
  (canBroadcastTo_dim_dim_of_rank_eq same.rank_eq).mpr ⟨Or.inr rfl, tail⟩

/-- A new outer target dimension aligns a source of lower rank. -/
theorem CanBroadcastTo.expand_dims {n : Nat} {s₁ s₂ : Shape} (tail : CanBroadcastTo s₁ s₂) :
    CanBroadcastTo s₁ (.dim n s₂) := by
  cases s₁ with
  | scalar => exact tail
  | dim m s =>
      have hRank : s.rank ≠ s₂.rank := by
        have := tail.rank_le
        simp only [rank] at this
        grind
      exact (canBroadcastTo_dim_dim_of_rank_ne hRank).mpr tail

/-- Removing a target axis that only aligns ranks keeps the source broadcastable. -/
theorem CanBroadcastTo.of_expand_dims {n : Nat} {s t : Shape} (hRank : s.rank ≤ t.rank)
    (h : CanBroadcastTo s (.dim n t)) : CanBroadcastTo s t := by
  cases s with
  | scalar => exact h
  | dim m s =>
      refine (canBroadcastTo_dim_dim_of_rank_ne ?_).mp h
      simp only [rank] at hRank
      grind

/-- Every shape broadcasts to itself without expanding an axis. -/
theorem CanBroadcastTo.refl : (s : Shape) → CanBroadcastTo s s
  | .scalar => trivial
  | .dim _ tail => (canBroadcastTo_dim_dim_of_rank_eq rfl).mpr ⟨Or.inl rfl, refl tail⟩

/-- A scalar broadcasts to any shape by inserting every target dimension. -/
theorem CanBroadcastTo.scalarTo : (s : Shape) → CanBroadcastTo .scalar s
  | .scalar => trivial
  | .dim _ tail => scalarTo tail

/-- A shape of singleton axes broadcasts to any shape of the same rank. -/
theorem CanBroadcastTo.singletonAxes : (s : Shape) → CanBroadcastTo (Shape.singletonAxes s) s
  | .scalar => trivial
  | .dim _ tail =>
      letI : SameRank (Shape.singletonAxes tail) tail := ⟨rank_singletonAxes tail⟩
      .dim_1_to_n (singletonAxes tail)

/-- A suffix broadcasts across an arbitrary collection of newly prepended target dimensions. -/
theorem CanBroadcastTo.prependTarget : (leading suffix : Shape) →
    CanBroadcastTo suffix (Shape.concat leading suffix)
  | .scalar, suffix => refl suffix
  | .dim _ tail, suffix => .expand_dims (prependTarget tail suffix)

/-- Typeclass wrapper for `CanBroadcastTo` so broadcast proofs can be inferred for literal
shapes. -/
class BroadcastTo (s₁ s₂ : Shape) : Prop where
  proof : CanBroadcastTo s₁ s₂

/-- Scalar shapes broadcast directly. Leading target dimensions are inferred by `expand_dims`. -/
instance broadcastToScalar : BroadcastTo Shape.scalar Shape.scalar where
  proof := CanBroadcastTo.scalar

/-- Broadcasting preserves equal leading dimensions when the tails broadcast. -/
instance broadcastToDimEq {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
    [bc : BroadcastTo s₁ s₂] : BroadcastTo (Shape.dim n s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_eq bc.proof

/-- Dimension `1` can broadcast to any `n` (PyTorch's main broadcast rule). -/
instance broadcastToDim1ToN {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
    [bc : BroadcastTo s₁ s₂] : BroadcastTo (Shape.dim 1 s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_1_to_n bc.proof

/-- Prepend an outer dimension (the "expand_dims" step used to align ranks). -/
instance broadcastToExpandDims {n : Nat} {s₁ s₂ : Shape} [bc : BroadcastTo s₁ s₂] :
    BroadcastTo s₁ (Shape.dim n s₂) where
  proof := CanBroadcastTo.expand_dims bc.proof
/-- Swap adjacent entries in an axis-ordering list, leaving invalid positions unchanged. -/
def swapAdjacentAxes (axes : List Nat) (depth : Nat) : List Nat :=
  match axes, depth with
  | [], _ => []
  | [axis], _ => [axis]
  | first :: second :: rest, 0 => second :: first :: rest
  | first :: rest, depth + 1 => first :: swapAdjacentAxes rest depth

/-- Permute axes of a shape using a zero-based structural axis ordering.

Returns `none` if the permutation is invalid. -/
def permute? (s : Shape) (perm : List Nat) : Option Shape :=
  let r := rank s
  if perm.length != r then
    none
  else if !(decide perm.Nodup) then
    none
  else
    let dims := toList s
    (perm.mapM fun i => dims[i]?).map ofList

/-- Axis permutation that exchanges `axis₁` and `axis₂` and fixes every other axis. -/
def transposePermutation (rank axis₁ axis₂ : Nat) : List Nat :=
  (List.range rank).map fun axis =>
    if axis = axis₁ then axis₂ else if axis = axis₂ then axis₁ else axis

/-- Remove one axis from a shape. Invalid axes leave the shape unchanged. -/
def eraseAxis : Shape → Nat → Shape
  | .scalar, _ => .scalar
  | .dim _ rest, 0 => rest
  | .dim n rest, axis + 1 => .dim n (eraseAxis rest axis)

/-- Replace the extent of one axis. Invalid axes leave the shape unchanged. -/
def replaceAxis : Shape → Nat → Nat → Shape
  | .scalar, _, _ => .scalar
  | .dim _ rest, 0, extent => .dim extent rest
  | .dim n rest, axis + 1, extent => .dim n (replaceAxis rest axis extent)

/-- Replace the final axis extent. A scalar shape is left unchanged. -/
@[reducible] def replaceLast : Shape → Nat → Shape
  | .scalar, _ => .scalar
  | .dim _ .scalar, extent => .dim extent .scalar
  | .dim n (.dim m rest), extent => .dim n (replaceLast (.dim m rest) extent)

/-- Structural evidence that a non-scalar shape ends in an axis with extent `extent`. -/
inductive EndsWith.Proof (extent : Nat) : Shape → Type
  | last : Proof extent (.dim extent .scalar)
  | leading {n : Nat} {rest : Shape} :
      Proof extent rest → Proof extent (.dim n rest)

/-- Typeclass evidence that a shape's final axis has extent `extent`. -/
class EndsWith (extent : Nat) (shape : Shape) where
  /-- Structural evidence consumed by final-axis operations. -/
  proof : EndsWith.Proof extent shape

/-- Replace the final extent recorded by structural evidence. -/
def EndsWith.Proof.replace {extent : Nat} {shape : Shape}
    (evidence : EndsWith.Proof extent shape) (outputExtent : Nat) : Shape :=
  match evidence with
  | .last => .dim outputExtent .scalar
  | .leading (n := n) inner => .dim n (inner.replace outputExtent)

/-- Evidence-directed final-axis replacement agrees with ordinary shape replacement. -/
theorem EndsWith.Proof.replace_eq_replaceLast {extent : Nat} {shape : Shape}
    (evidence : EndsWith.Proof extent shape) (outputExtent : Nat) :
    evidence.replace outputExtent = shape.replaceLast outputExtent := by
  induction evidence with
  | last => rfl
  | @leading n rest inner ih =>
      cases inner with
      | last => rfl
      | leading deeper =>
          simpa [EndsWith.Proof.replace, replaceLast] using congrArg (Shape.dim n) ih

/-- A one-dimensional shape ends in its only extent. -/
instance endsWithLast (extent : Nat) : EndsWith extent (.dim extent .scalar) :=
  ⟨.last⟩

/-- Adding a leading axis preserves the final extent. -/
instance endsWithLeading {extent n : Nat} {rest : Shape} [EndsWith extent rest] :
    EndsWith extent (.dim n rest) :=
  ⟨.leading (EndsWith.proof (extent := extent) (shape := rest))⟩

/-- Erasing an in-bounds axis decreases the rank by one. -/
theorem rank_eraseAxis {s : Shape} {axis : Nat} (h : axis < s.rank) :
    (s.eraseAxis axis).rank = s.rank - 1 := by
  induction s generalizing axis with
  | scalar => simp [rank] at h
  | dim n rest ih =>
      cases axis with
      | zero => simp [eraseAxis, rank]
      | succ axis =>
          have hAxis : axis < rest.rank := by
            simp only [rank] at h
            grind
          simp only [eraseAxis, rank, ih hAxis]
          grind

/-- Replacing an in-bounds axis preserves rank. -/
theorem rank_replaceAxis {s : Shape} {axis extent : Nat} (h : axis < s.rank) :
    (s.replaceAxis axis extent).rank = s.rank := by
  induction s generalizing axis with
  | scalar => simp [rank] at h
  | dim n rest ih =>
      cases axis with
      | zero => simp [replaceAxis, rank]
      | succ axis =>
          have hAxis : axis < rest.rank := by
            simp only [rank] at h
            grind
          simp [replaceAxis, rank, ih hAxis]

/-!
## Axis evidence

Axes are zero-based natural numbers. `AxisInBounds axis s` says only that the axis exists;
`HasNonemptyAxis axis s` additionally says that its extent is positive. Shape-preserving operations
such as softmax need the first condition. Reductions whose definition selects an element, such as
maximum and minimum, use the second.

Negative axes are normalized by frontends before reaching this layer. The innermost axis of a
positive-rank shape is `s.rank - 1`.
-/

/-- The zero-based `axis` names a dimension of `s`. Its extent may be zero. -/
class AxisInBounds (axis : Nat) (s : Shape) : Prop where
  /-- The axis is strictly smaller than the shape rank. -/
  proof : axis < rank s

namespace AxisInBounds

/-- Convert an ordinary rank proof into the evidence consumed by tensor operations. -/
public theorem ofRank {axis : Nat} {shape : Shape} (valid : axis < shape.rank) :
    AxisInBounds axis shape :=
  ⟨valid⟩

end AxisInBounds

/-- Looking up an axis below the rank of a shape succeeds. -/
theorem getDim_isSome_of_lt {s : Shape} {axis : Nat} (h : axis < rank s) :
    (getDim s axis).isSome = true := by
  induction s generalizing axis with
  | scalar => simp [rank] at h
  | dim n rest ih =>
      cases axis with
      | zero => rfl
      | succ axis =>
          simp only [getDim]
          apply ih
          grind [rank]

/-- Extent of a statically valid axis. -/
def axisSize (s : Shape) (axis : Nat) [h : AxisInBounds axis s] : Nat :=
  (getDim s axis).get (getDim_isSome_of_lt h.proof)

/-- The outermost dimension is axis zero, regardless of its extent. -/
instance axisInBoundsZero {n s} : AxisInBounds 0 (.dim n s) :=
  ⟨Nat.succ_pos s.rank⟩

/-- An inner axis remains in bounds under an additional outer dimension. -/
instance axisInBoundsSucc {n s axis} [h : AxisInBounds axis s] :
    AxisInBounds (axis + 1) (.dim n s) :=
  ⟨Nat.add_lt_add_right h.proof 1⟩

/-- The extent of the leading axis is its outer dimension. -/
@[simp] theorem axisSize_zero (n : Nat) (s : Shape) : axisSize (.dim n s) 0 = n := by
  rfl

/-- Looking through an outer dimension preserves the extent of an inner axis. -/
@[simp] theorem axisSize_succ (n : Nat) (s : Shape) (axis : Nat)
    [AxisInBounds axis s] [AxisInBounds (axis + 1) (.dim n s)] :
    axisSize (.dim n s) (axis + 1) = axisSize s axis := by
  simp only [axisSize, getDim]

/-- Decide whether a natural number names a dimension of `s`, returning typed evidence. -/
def axisInBounds? (axis : Nat) (s : Shape) : Option (PLift (AxisInBounds axis s)) :=
  if h : axis < rank s then some ⟨⟨h⟩⟩ else none

/-- The decision procedure succeeds whenever static in-bounds evidence is available. -/
theorem axisInBounds?_isSome {axis : Nat} {s : Shape} [h : AxisInBounds axis s] :
    (axisInBounds? axis s).isSome := by
  simp [axisInBounds?, h.proof]

/--
`NonemptyAxis axis s` says that `axis` selects a positive-length dimension of `s`.

The constructors follow the recursive representation of `Shape`, so reduction definitions can
eliminate this evidence while recursing through outer dimensions.
-/
inductive NonemptyAxis : Nat → Shape → Prop
  | zero {n : Nat} {s : Shape} : NonemptyAxis 0 (.dim (n + 1) s)
  | succ {n : Nat} {s : Shape} {axis : Nat} :
      NonemptyAxis axis s → NonemptyAxis (axis + 1) (.dim n s)

/-- A nonempty axis is, in particular, a valid axis. -/
theorem NonemptyAxis.toAxisInBounds {axis : Nat} {s : Shape} (h : NonemptyAxis axis s) :
    AxisInBounds axis s := by
  induction h with
  | zero => exact axisInBoundsZero
  | succ inner ih => exact ⟨Nat.add_lt_add_right ih.proof 1⟩

/-- Axis zero is nonempty when the outer dimension is positive. -/
@[simp] theorem nonemptyAxis_zero {n : Nat} {s : Shape} :
    NonemptyAxis 0 (.dim (n + 1) s) :=
  .zero

/-- Nonemptiness of an inner axis is unchanged by an outer dimension. -/
@[simp] theorem nonemptyAxis_succ {n : Nat} {s : Shape} {axis : Nat}
    (h : NonemptyAxis axis s) : NonemptyAxis (axis + 1) (.dim n s) :=
  .succ h

/--
Return evidence that `axis` addresses a positive dimension of `s`.

Executable consumers use this to recover the proposition required by typed tensor operations from
a raw runtime axis. Invalid axes, including axes into zero-sized dimensions, return `none`.
-/
def nonemptyAxis? (axis : Nat) : (s : Shape) → Option (PLift (NonemptyAxis axis s))
  | .scalar => none
  | .dim n rest =>
      match axis, n with
      | 0, Nat.succ k => some ⟨NonemptyAxis.zero (n := k) (s := rest)⟩
      | 0, 0 => none
      | Nat.succ inner, n =>
          (nonemptyAxis? inner rest).map (fun h =>
            ⟨NonemptyAxis.succ (n := n) (s := rest) (axis := inner) h.down⟩)

/-- Typeclass wrapper used when nonempty-axis evidence can be inferred statically. -/
class HasNonemptyAxis (axis : Nat) (s : Shape) : Prop where
  /-- The selected dimension has positive extent. -/
  proof : NonemptyAxis axis s

/-- Instance: axis `0` is valid for any positive outer dimension. -/
instance hasNonemptyAxisZero {n s} : HasNonemptyAxis 0 (.dim (n + 1) s) :=
  ⟨NonemptyAxis.zero⟩

/-- Package a proof that the outer dimension is nonzero as axis evidence. -/
theorem hasNonemptyAxisZeroOfNe {n s} (h : n ≠ 0) : HasNonemptyAxis 0 (.dim n s) :=
  let ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero h
  ⟨by rw [hm]; exact .zero⟩

/-- Package positivity of the outer dimension as axis evidence. -/
theorem hasNonemptyAxisZeroOfPos {n s} (h : 0 < n) : HasNonemptyAxis 0 (.dim n s) :=
  hasNonemptyAxisZeroOfNe (Nat.ne_of_gt h)

/-- Nonemptiness of an inner axis is unchanged by an outer dimension. -/
instance hasNonemptyAxisSucc {n s axis} [h : HasNonemptyAxis axis s] :
    HasNonemptyAxis (axis + 1) (.dim n s) :=
  ⟨NonemptyAxis.succ h.proof⟩


/-!
## Well-formedness (`wellFormed`)

`well_formed s` means "all dimensions are positive".

Why this matters (and why we designed it this way):

- Many definitions use `Fin n` indexing; if `n = 0`, there is no index and you end up with either
  vacuous truths or extra cases that obscure the intent of the lemma.
- Some common ops become awkward or partial at `n = 0`. For example, a mean typically divides by
  the number of elements, so `n = 0` needs special-case semantics.
- PyTorch *does* allow zero-sized dimensions, and most ops define a sensible result for them. We
  intentionally keep that complexity out of the core spec layer because it makes proofs much more
  case-heavy. When we need zero-dimension tensors, we introduce them with explicit
  semantics instead of relying on incidental behavior.

This is a pragmatic choice: proofs and specs are shorter, and
runtime checks can still handle edge cases separately.
-/
-- Well-formed shapes have positive dimensions.
/-- `well_formed s` means "all dimensions of `s` are positive" (recursively). -/
def wellFormed : Shape → Prop
| .scalar => True
| .dim n s => n > 0 ∧ s.wellFormed

/-!
### Size positivity

If all dimensions of a shape are positive, then the total number of scalar elements is positive.

This is a small but useful bridge lemma: many reductions are only defined for nonempty dimensions,
and `WellFormed` is our standard way of expressing that assumption.
-/

/-- If `s.wellFormed`, then `Spec.Shape.size s > 0`. -/
theorem size_pos_of_well_formed : ∀ {s : Shape}, s.wellFormed → 0 < Spec.Shape.size s
  | .scalar, _ => by
      simp [Spec.Shape.size]
  | .dim n s, hw => by
      rcases hw with ⟨hn, hs⟩
      simpa [Spec.Shape.size] using Nat.mul_pos hn (size_pos_of_well_formed (s := s) hs)

/-- A shape of positive total size has no zero dimension. -/
theorem wellFormed_of_size_pos : ∀ {s : Shape}, 0 < Spec.Shape.size s → s.wellFormed
  | .scalar, _ => trivial
  | .dim n s, h => by
      have hn : n ≠ 0 := by
        intro hn
        simp [Spec.Shape.size, hn] at h
      have hs : Spec.Shape.size s ≠ 0 := by
        intro hs
        simp [Spec.Shape.size, hs] at h
      exact ⟨Nat.pos_of_ne_zero hn, wellFormed_of_size_pos (Nat.pos_of_ne_zero hs)⟩

/-- Every in-bounds axis of a well-formed shape has positive extent. -/
theorem nonemptyAxis_of_wellFormed {s : Shape} (hw : s.wellFormed) {axis : Nat}
    (hAxis : axis < s.rank) : NonemptyAxis axis s := by
  induction s generalizing axis with
  | scalar => simp [rank] at hAxis
  | dim n rest ih =>
      rcases hw with ⟨hn, hRest⟩
      cases axis with
      | zero =>
          obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hn)
          exact .zero
      | succ axis =>
          apply NonemptyAxis.succ
          apply ih hRest
          simp only [rank] at hAxis
          grind

/-- Package a valid axis of a well-formed shape as inferred reduction-axis evidence. -/
theorem hasNonemptyAxis_of_wellFormed {s : Shape} (hw : s.wellFormed) {axis : Nat}
    (hAxis : axis < s.rank) : HasNonemptyAxis axis s :=
  ⟨nonemptyAxis_of_wellFormed hw hAxis⟩

/--
Typeclass wrapper for `Shape.wellFormed`.

We use a typeclass (instead of passing a `wellFormed` proof everywhere) because it mirrors how
other "side conditions" are handled in the library: call sites stay clean, and instances can be
provided locally (e.g. `letI : Shape.WellFormed s := ...`) when needed.
-/
class WellFormed (s : Shape) : Prop where
  proof : s.wellFormed

-- Scalars are always well-formed.
/-- Scalars are always well-formed. -/
instance : WellFormed .scalar where
  proof := trivial

/-- Adding a nonzero dimension preserves well-formedness. -/
instance {n s} [Shape.WellFormed s] [NeZero n] : Shape.WellFormed (.dim n s) :=
  ⟨⟨Nat.pos_of_ne_zero (NeZero.ne n), Shape.WellFormed.proof⟩⟩

/-- Infer nonemptiness of any valid axis from `WellFormed s`. -/
theorem inferNonemptyAxis {s : Shape} [hw : WellFormed s] {axis : Nat}
    (hAxis : axis < s.rank) : HasNonemptyAxis axis s :=
  hasNonemptyAxis_of_wellFormed hw.proof hAxis

/-!
`padLeft n s` prepends `n` singleton dimensions to a shape.

PyTorch analogy: `unsqueeze(0)` repeated `n` times (or equivalently viewing a tensor as having
extra leading dimensions of size 1). This is also the "prepend 1s" step you see in broadcasting.
-/
/-- Prepend `n` leading singleton dimensions (size `1`) to a shape. -/
def padLeft : Nat → Shape → Shape
| 0, s => s
| (n+1), s => dim 1 (padLeft n s)

/-- `padLeft n s` increases the rank by exactly `n`. -/
@[simp] theorem rank_padLeft (n : Nat) (s : Shape) : (padLeft n s).rank = s.rank + n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [padLeft, rank, ih, Nat.add_assoc]

/-- Leading singleton axes do not change the number of scalar entries. -/
@[simp] theorem size_padLeft (n : Nat) (s : Shape) : (padLeft n s).size = s.size := by
  induction n with
  | zero => rfl
  | succ n ih => simp [padLeft, size, ih]

/-- The list view of a padded shape prepends `n` ones. -/
theorem padLeft_eq_replicate_append (n : Nat) (s : Shape) :
    padLeft n s = List.replicate n 1 ++ s := by
  induction n with
  | zero => rfl
  | succ n ih => simp [padLeft, ih, List.replicate_succ]

end Shape
end Spec
