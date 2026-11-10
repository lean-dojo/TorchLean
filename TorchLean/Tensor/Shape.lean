/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Shape.Coord


/-!
# Shapes (`TorchLean.Shape`)

`Shape` is the type-level shape of a tensor. TorchLean's tensors are shape-indexed, `Tensor α s`,
so Lean checks shape compatibility before tensor code can run.

## Representation

A shape is a list of dimensions, outermost first, so model and application code writes one in
ordinary bracket notation: `[]` for a scalar, `[n]` for a vector, `[m, n]` for a matrix.

`Shape` is a reducible abbreviation for `List Nat`, so the spec-level shape and the shape carried
by a tensor buffer are the same type. There is no conversion between them and no cast to transport
a tensor across the two views.

Shape-recursive definitions and proofs use the names `Shape.scalar` and `Shape.dim`,
`@[match_pattern]` abbreviations for `[]` and `· :: ·`. They can stand in patterns like
constructors, and `induction` and `cases` on a shape offer the cases `scalar` and `dim` through the
eliminators registered below.

## Size and rank

`Shape.size` is the number of entries, the product of the dimensions, and `Shape.rank` is the
number of axes. Both are defined in `TorchLean.Tensor.Shape.Coord`. In PyTorch terms a shape is
`tensor.shape`, its rank is `tensor.ndim` and its size is `tensor.numel()`.

## Broadcasting and axes

Broadcasting is the decidable proposition `CanBroadcastTo` with its typeclass wrapper
`BroadcastTo`. It is asymmetric ("broadcast `s1` to `s2`") because most tensor code picks the
output shape and asks each input to broadcast to it. Lean can often infer the `BroadcastTo`
evidence, so call sites do not thread proofs by hand.

Axis evidence (`AxisInBounds`, `NonemptyAxis`) and the predicate `Positive`, which says that
every dimension is positive, let proofs rule out degenerate cases.
-/

@[expose] public section


open TorchLean.Shape (rank rank_cons rank_eq_length size)
open TorchLean.Shape (size_append size_eq_prod)

namespace TorchLean

namespace Shape

/-- The shape of a scalar: no axes. Usable in patterns, like a constructor. -/
@[match_pattern] abbrev scalar : Shape := []

/-- A shape with outermost axis of extent `n` over `s`. Usable in patterns, like a constructor. -/
@[match_pattern] abbrev dim (n : Nat) (s : Shape) : Shape := n :: s

universe u

/--
Recursion on a shape, one axis at a time.

`induction s` uses this eliminator, so the cases are named `scalar` and `dim` and the tail
of a `dim` is again spelled `Shape`. Ordinary list induction is unaffected: a hypothesis whose type
is spelled `List Nat` eliminates to `nil` and `cons`.
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
Model code writes shapes as dimension lists. `Tensor Float [4, 2]` is a four-by-two tensor of Lean
`Float` values, and `Trainer.Dataset [2] [1]` describes supervised samples with two input values
and one target value. Since `Shape` abbreviates `List Nat`, these brackets are ordinary list
notation.

The brackets belong to the type: a value of type `Tensor Float [4, 2]` is a `Tensor` and never a
`List`. Recursive tensor programs inspect the same list through the `.scalar` and `.dim` pattern
abbreviations, while ordinary model code can keep the bracket form.
-/
namespace Shape

/-- The extent a dilated kernel spans along one axis, `dilation * (kernel - 1) + 1`, and zero for
an empty kernel. -/
def dilatedKernelExtent (kernel dilation : Nat) : Nat :=
  if kernel = 0 then 0 else dilation * (kernel - 1) + 1

/--
Output length of a floor-mode sliding window with dilation and independent padding on each side:
$(\mathtt{input}+\mathtt{paddingBefore}+\mathtt{paddingAfter}-e)/\mathtt{stride}+1$ for the
dilated kernel extent $e$ (`dilatedKernelExtent`). An empty kernel, a zero stride, or a padded
input shorter than $e$ gives length zero, so saturated subtraction and division by zero cannot
create a phantom output element.
-/
def slidingWindowOutDimDilated
    (input kernel stride dilation paddingBefore paddingAfter : Nat) : Nat :=
  let effective := dilatedKernelExtent kernel dilation
  let padded := input + paddingBefore + paddingAfter
  if effective = 0 || stride = 0 || padded < effective then
    0
  else
    (padded - effective) / stride + 1

-- Shapes print through the `Repr` and `ToString` instances of `List Nat`, as `[2, 3]`. A second
-- instance on the same type would shadow them for every list of naturals.

/-- Swap the dimensions at depths `depth` and `depth + 1`, counted from zero at the outermost axis.
A depth without such a pair leaves the shape unchanged. -/
@[reducible] def swapAdjacentAtDepth (s : Shape) (depth : Nat) : Shape :=
  match depth, s with
  | 0, .dim m (.dim n rest) => .dim n (.dim m rest)
  | d+1, .dim m rest => .dim m (swapAdjacentAtDepth rest d)
  | _, _ => s  -- invalid depth, return unchanged

/-- At rank two, swapping at depth zero is the ordinary matrix transpose of the shape. -/
@[simp] theorem swapAdjacentAtDepth_pair_zero (m n : Nat) :
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

/--
Appending an axis of extent `n` multiplies the number of entries by `n`.

Two common reshapes rest on it: reading a tensor of shape `s ++ [n]` as a `(size s) × n`
matrix, and appending a singleton axis (`n = 1`) without changing `size`.
-/
theorem size_append_singleton (s : Shape) (n : Nat) : Shape.size (s ++ [n]) = Shape.size s * n :=
  (Shape.size_append s [n]).trans (congrArg (Shape.size s * ·) (Nat.mul_one n))

/-- A vector of length `n` has `n` entries, which lets `reshape x [_] (size_vector _).symm`
flatten `x`. -/
theorem size_vector (n : Nat) : Shape.size [n] = n := Nat.mul_one n

/-- Flattening the axes after `leading` keeps the number of entries:
`reshape x (leading ++ [suffix.size]) (size_append_eq_size_append_singleton ..)` flattens them. -/
theorem size_append_eq_size_append_singleton (leading suffix : Shape) :
    Shape.size (leading ++ suffix) = Shape.size (leading ++ [suffix.size]) := by
  rw [Shape.size_append, size_append_singleton]

/-- A shape given as the axes `lead` followed by an axis of `n` entries has as many entries as the
matrix of `m = lead.size` rows and `n` columns. -/
theorem size_eq_of_append {x lead : Shape} {m n : Nat} (hm : m = lead.size)
    (hx : x = lead ++ [n]) : x.size = Shape.size [m, n] := by
  subst hx hm
  rw [Shape.size_append]
  rfl

-- Let `grind` rewrite the size of a shape with an appended axis (`size_append_singleton`).
attribute [grind =] size_append_singleton

/-!
### Typeclass-friendly broadcasting (`BroadcastTo`)

`BroadcastTo` lets instance search find a broadcast proof, which suits layer and model specs where
the broadcasting details are not the point. The rules follow PyTorch, which aligns shapes from the
trailing dimensions by prepending `1`s to the shorter shape. On an outermost-first `Shape` that
step is the rule `expand_dims`, which lets the target carry outer dimensions the source lacks, and
`dim_one` is PyTorch's rule that a dimension of extent `1` expands to any `n`.

We took these rules by hand from the "Broadcasting semantics" note of the PyTorch 2.8 docs. Its
size rule takes "the max of the sizes", but at a zero extent torch 2.8.0 keeps the zero:
`torch.ones(1) + torch.ones(0)` has shape `[0]`, and `[1]` broadcasts to `[0]` here too. The note
also asks that "each tensor has at least one dimension", while `torch.broadcast_shapes((), (2, 3))`
is `(2, 3)` and a scalar broadcasts to every shape here.
-/

/-- The axis after a leading shape is in range. -/
theorem rank_lt_rank_append (leading suffix : Shape) (k : Nat) :
    leading.rank < (leading ++ k :: suffix).rank := by
  simp [Shape.rank_eq_length]

/-- Appending one dimension increases the rank by one. -/
@[simp] theorem rank_append_singleton (s : Shape) (n : Nat) :
    Shape.rank (s ++ [n]) = Shape.rank s + 1 := by
  simp [Shape.rank_eq_length]

/-- A shape decomposed into a leading prefix and a suffix of a prescribed rank. -/
structure SuffixSplit (shape : Shape) (suffixRank : Nat) where
  /-- Axes preceding the suffix. -/
  leading : Shape
  /-- Extents of the suffix axes. -/
  suffix : List Nat
  /-- The suffix has the requested number of axes. -/
  suffix_length : suffix.length = suffixRank
  /-- The decomposition reconstructs the original shape. -/
  append_eq : leading ++ suffix = shape

/-- Split the final `suffixRank` axes from a shape. -/
def splitSuffix (shape : Shape) (suffixRank : Nat) (h : suffixRank ≤ shape.rank) :
    SuffixSplit shape suffixRank := by
  have hdims : suffixRank ≤ shape.length := by simpa [Shape.rank_eq_length] using h
  let split := shape.length - suffixRank
  let suffixDims := shape.drop split
  have hsuffix : suffixDims.length = suffixRank := by
    simp [suffixDims, split, List.length_drop, Nat.sub_sub_self hdims]
  exact
    { leading := shape.take split
      suffix := suffixDims
      suffix_length := hsuffix
      append_eq := List.take_append_drop split shape }

/-- Swap the first two axes after an arbitrary fixed leading shape. -/
@[simp]
theorem swapAdjacentAtDepth_append_rank (leading suffix : Shape) (m n : Nat) :
    swapAdjacentAtDepth (leading ++ m :: n :: suffix) leading.rank =
      leading ++ n :: m :: suffix := by
  induction leading with
  | scalar => rfl
  | dim _ tail ih =>
      simp only [List.cons_append, Shape.rank_cons, swapAdjacentAtDepth, ih]

/-- Proposition used by broadcast constructors that align two existing dimensions. -/
class SameRank (s₁ s₂ : Shape) : Prop where
  /-- The two shapes have the same number of dimensions. -/
  rank_eq : Shape.rank s₁ = Shape.rank s₂

/-- Every shape has the same rank as itself. -/
instance sameRankRefl (s : Shape) : SameRank s s := ⟨rfl⟩

/-- One more outer axis on each side keeps the ranks equal. -/
instance {s₁ s₂ : Shape} {n₁ n₂ : Nat} [tail : SameRank s₁ s₂] :
    SameRank (.dim n₁ s₁) (.dim n₂ s₂) :=
  ⟨by simp [tail.rank_eq]⟩

/-!
### The broadcast relation (`CanBroadcastTo`)

`CanBroadcastTo source target` is a proposition on the two shapes, so every tensor operation that
consumes it depends only on the shapes and never on how the relation is proved. It is defined by
recursion on the target shape, which makes it decidable, and the structural rules of NumPy and
PyTorch are recovered as theorems: `CanBroadcastTo.scalar`, `CanBroadcastTo.dim_eq`,
`CanBroadcastTo.dim_one`, and `CanBroadcastTo.expand_dims`.

The equal-dimension rules require equal-rank tails, so every rank difference is resolved by
`expand_dims` before extents are compared. The internal tensor layer states the same relation on
dimension lists (`List.Forall₂` after padding the source with leading ones), and
`CanBroadcastTo.forall₂_toList`, proved beside `broadcast` in
`TorchLean.Tensor.Movement`, connects the two forms.
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

/-- A scalar broadcasts to a scalar. Higher-rank scalar broadcasts are built with `expand_dims`. -/
@[simp] theorem CanBroadcastTo.scalar : CanBroadcastTo .scalar .scalar :=
  trivial

/-- Nothing with an axis broadcasts down to a scalar: a broadcast never removes an axis. -/
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

/-- When the tails differ in rank, the target's leading axis only aligns ranks: the source
broadcasts to the target exactly when it broadcasts to the target's tail. -/
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
          · simp [hRank]
          · have := ih ((canBroadcastTo_dim_dim_of_rank_ne hRank).mp h)
            simp only [Shape.rank_cons] at this ⊢
            grind

/-- Matching outer dimensions preserve broadcasting of equal-rank tails. -/
theorem CanBroadcastTo.dim_eq {n : Nat} {s₁ s₂ : Shape} [same : SameRank s₁ s₂]
    (tail : CanBroadcastTo s₁ s₂) : CanBroadcastTo (.dim n s₁) (.dim n s₂) :=
  (canBroadcastTo_dim_dim_of_rank_eq same.rank_eq).mpr ⟨Or.inl rfl, tail⟩

/-- An outer dimension of length one can expand to any target length. -/
theorem CanBroadcastTo.dim_one {n : Nat} {s₁ s₂ : Shape} [same : SameRank s₁ s₂]
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
        simp only [Shape.rank_cons] at this
        grind
      exact (canBroadcastTo_dim_dim_of_rank_ne hRank).mpr tail

/-- Removing a target axis that only aligns ranks keeps the source broadcastable. -/
theorem CanBroadcastTo.of_expand_dims {n : Nat} {s t : Shape} (hRank : s.rank ≤ t.rank)
    (h : CanBroadcastTo s (.dim n t)) : CanBroadcastTo s t := by
  cases s with
  | scalar => exact h
  | dim m s =>
      refine (canBroadcastTo_dim_dim_of_rank_ne ?_).mp h
      simp only [Shape.rank_cons] at hRank
      grind

/-- Every shape broadcasts to itself without expanding an axis. -/
theorem CanBroadcastTo.refl : (s : Shape) → CanBroadcastTo s s
  | .scalar => trivial
  | .dim _ tail => (canBroadcastTo_dim_dim_of_rank_eq rfl).mpr ⟨Or.inl rfl, refl tail⟩

/-- A scalar broadcasts to any shape by inserting every target dimension. -/
theorem scalar_canBroadcastTo : (s : Shape) → CanBroadcastTo .scalar s
  | .scalar => trivial
  | .dim _ tail => scalar_canBroadcastTo tail

/-- A shape broadcasts to itself with any leading target axes prepended. -/
theorem canBroadcastTo_concat_right : (leading suffix : Shape) →
    CanBroadcastTo suffix (leading ++ suffix)
  | .scalar, suffix => CanBroadcastTo.refl suffix
  | .dim _ tail, suffix => .expand_dims (canBroadcastTo_concat_right tail suffix)

/-- Typeclass wrapper for `CanBroadcastTo`, so instance search can find broadcast proofs for
literal shapes. Computed or symbolic shapes may need the evidence written out, such as
`⟨CanBroadcastTo.refl s⟩` for a shape broadcast to itself. -/
class BroadcastTo (s₁ s₂ : Shape) : Prop where
  /-- The broadcast evidence. -/
  proof : CanBroadcastTo s₁ s₂

/-- A scalar broadcasts to a scalar, and `broadcastToExpandDims` supplies any leading target
dimensions. -/
instance broadcastToScalar : BroadcastTo Shape.scalar Shape.scalar where
  proof := CanBroadcastTo.scalar

/-- Broadcasting preserves equal leading dimensions when the tails broadcast. -/
instance broadcastToDimEq {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
    [bc : BroadcastTo s₁ s₂] : BroadcastTo (Shape.dim n s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_eq bc.proof

/-- Dimension `1` can broadcast to any `n` (PyTorch's main broadcast rule). -/
instance broadcastToDimOne {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
    [bc : BroadcastTo s₁ s₂] : BroadcastTo (Shape.dim 1 s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_one bc.proof

/-- Prepend an outer dimension (the "expand_dims" step used to align ranks). -/
instance broadcastToExpandDims {n : Nat} {s₁ s₂ : Shape} [bc : BroadcastTo s₁ s₂] :
    BroadcastTo s₁ (Shape.dim n s₂) where
  proof := CanBroadcastTo.expand_dims bc.proof

/-- Replace the final axis extent. A scalar shape is left unchanged. -/
@[reducible] def replaceLast : Shape → Nat → Shape
  | .scalar, _ => .scalar
  | .dim _ .scalar, extent => .dim extent .scalar
  | .dim n (.dim m rest), extent => .dim n (replaceLast (.dim m rest) extent)

/-- Structural evidence that a non-scalar shape ends in an axis with extent `extent`. -/
inductive EndsWith.Witness (extent : Nat) : Shape → Type
  /-- The one-axis shape `[extent]`. -/
  | last : Witness extent (.dim extent .scalar)
  /-- One more leading axis in front of a shape that ends in `extent`. -/
  | leading {n : Nat} {rest : Shape} :
      Witness extent rest → Witness extent (.dim n rest)

/-- Typeclass evidence that a shape's final axis has extent `extent`. -/
class EndsWith (extent : Nat) (shape : Shape) where
  /-- Structural evidence consumed by final-axis operations. -/
  proof : EndsWith.Witness extent shape

/-- The axes before the final one, read from structural evidence. -/
def EndsWith.Witness.leadingShape {extent : Nat} {shape : Shape} :
    EndsWith.Witness extent shape → Shape
  | .last => .scalar
  | .leading (n := n) inner => .dim n inner.leadingShape

/-- A shape that ends in `extent` is its leading axes followed by `extent`. -/
theorem EndsWith.Witness.leadingShape_append {extent : Nat} {shape : Shape}
    (evidence : EndsWith.Witness extent shape) :
    evidence.leadingShape ++ [extent] = shape := by
  induction evidence with
  | last => rfl
  | @leading n rest inner ih => exact congrArg (Shape.dim n) ih

/-- Its leading axes followed by `outputExtent` are the shape with the final extent replaced. -/
theorem EndsWith.Witness.leadingShape_append' {extent : Nat} {shape : Shape}
    (evidence : EndsWith.Witness extent shape) (outputExtent : Nat) :
    evidence.leadingShape ++ [outputExtent] = shape.replaceLast outputExtent := by
  induction evidence with
  | last => rfl
  | @leading n rest inner ih =>
      cases inner with
      | last => rfl
      | leading deeper =>
          simpa [EndsWith.Witness.leadingShape, replaceLast] using congrArg (Shape.dim n) ih

/-- A one-dimensional shape ends in its only extent. -/
instance endsWithLast (extent : Nat) : EndsWith extent (.dim extent .scalar) :=
  ⟨.last⟩

/-- Adding a leading axis preserves the final extent. -/
instance endsWithLeading {extent n : Nat} {rest : Shape} [EndsWith extent rest] :
    EndsWith extent (.dim n rest) :=
  ⟨.leading (EndsWith.proof (extent := extent) (shape := rest))⟩

/-- Erasing an in-bounds axis decreases the rank by one. -/
theorem rank_eraseIdx {s : Shape} {axis : Nat} (h : axis < s.rank) :
    Shape.rank (s.eraseIdx axis) = s.rank - 1 :=
  List.length_eraseIdx_of_lt h

/-- Replacing any axis preserves rank, including an invalid axis left unchanged. -/
theorem rank_set {s : Shape} {axis extent : Nat} :
    Shape.rank (s.set axis extent) = s.rank :=
  List.length_set

/-!
## Axis evidence

Axes are zero-based natural numbers here. `AxisInBounds axis s` says that the axis exists, and
`HasNonemptyAxis axis s` adds that its extent is positive. Graph builders and frontends that take a
`Nat` axis carry this evidence, while the specification operators take their axes as `Fin s.rank`,
such as `⟨axis, AxisInBounds.proof⟩`.

Frontends normalize negative axes before they reach this layer. The innermost axis of a
positive-rank shape is `s.rank - 1`.
-/

/-- The zero-based `axis` names a dimension of `s`. Its extent may be zero. -/
class AxisInBounds (axis : Nat) (s : Shape) : Prop where
  /-- The axis is strictly smaller than the shape rank. -/
  proof : axis < Shape.rank s

/-- Extent of a statically valid axis. -/
def axisSize (s : Shape) (axis : Nat) [h : AxisInBounds axis s] : Nat :=
  s[axis]'(Shape.rank_eq_length s ▸ h.proof)

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
@[simp] theorem axisSize_succ (n : Nat) (s : Shape) (axis : Nat) [AxisInBounds axis s] :
    axisSize (.dim n s) (axis + 1) = axisSize s axis :=
  rfl

/-- The extent of an axis is the shape's entry at that axis. -/
theorem axisSize_eq_getElem (s : Shape) (axis : Nat) [h : AxisInBounds axis s] :
    axisSize s axis = s[axis]'(Shape.rank_eq_length s ▸ h.proof) :=
  rfl

/-- Replacing an in-bounds axis by its existing extent leaves the shape unchanged. -/
@[simp] theorem set_axisSize (shape : Shape) (axis : Nat)
    [hAxis : Shape.AxisInBounds axis shape] :
    shape.set axis (shape.axisSize axis) = shape := by
  rw [axisSize_eq_getElem, List.set_getElem_self]

/-- Decide whether a natural number names a dimension of `s`, returning typed evidence. -/
def axisInBounds? (axis : Nat) (s : Shape) : Option (PLift (AxisInBounds axis s)) :=
  if h : axis < Shape.rank s then some ⟨⟨h⟩⟩ else none

/-- The decision procedure succeeds at every axis of `s`. -/
theorem axisInBounds?_isSome {s : Shape} (dim : Fin s.rank) : (axisInBounds? dim.val s).isSome := by
  simp [axisInBounds?, dim.isLt]

/--
`NonemptyAxis axis s` says that `axis` selects a positive-length dimension of `s`.

The constructors follow the recursive representation of `Shape`, so reduction definitions can
eliminate this evidence while recursing through outer dimensions.
-/
inductive NonemptyAxis : Nat → Shape → Prop
  /-- Axis zero of a shape whose outer extent is positive. -/
  | zero {n : Nat} {s : Shape} : NonemptyAxis 0 (.dim (n + 1) s)
  /-- Axis `axis + 1` of `.dim n s`, when axis `axis` of `s` is nonempty. -/
  | succ {n : Nat} {s : Shape} {axis : Nat} :
      NonemptyAxis axis s → NonemptyAxis (axis + 1) (.dim n s)

/-- A nonempty axis is, in particular, a valid axis. -/
theorem NonemptyAxis.axisInBounds {axis : Nat} {s : Shape} (h : NonemptyAxis axis s) :
    AxisInBounds axis s := by
  induction h with
  | zero => exact axisInBoundsZero
  | succ inner ih => exact ⟨Nat.add_lt_add_right ih.proof 1⟩

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

/-- A positive-length axis is a valid axis, without requiring positivity of the other axes. -/
instance axisInBoundsOfHasNonemptyAxis {axis : Nat} {shape : Shape}
    [h : HasNonemptyAxis axis shape] : AxisInBounds axis shape :=
  h.proof.axisInBounds

/-- Axis `0` is nonempty under a positive outer dimension. -/
instance hasNonemptyAxisZero {n s} : HasNonemptyAxis 0 (.dim (n + 1) s) :=
  ⟨NonemptyAxis.zero⟩

/-- Package a proof that the outer dimension is nonzero as axis evidence. -/
theorem hasNonemptyAxis_zero_of_ne {n s} (h : n ≠ 0) : HasNonemptyAxis 0 (.dim n s) :=
  let ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero h
  ⟨by rw [hm]; exact .zero⟩

/-- Nonemptiness of an inner axis is unchanged by an outer dimension. -/
instance hasNonemptyAxisSucc {n s axis} [h : HasNonemptyAxis axis s] :
    HasNonemptyAxis (axis + 1) (.dim n s) :=
  ⟨NonemptyAxis.succ h.proof⟩


/-!
## Well-formedness (`Positive`)

`Positive s` says that every dimension of `s` is positive. Shapes and tensors admit zero-sized
axes, so this predicate is an extra hypothesis for the operations and proofs that need positive
extents. `NonemptyAxis` asks for positivity of the reduced axis alone.
-/
/-- Every dimension of `s` is positive. -/
def Positive : Shape → Prop
| .scalar => True
| .dim n s => n > 0 ∧ s.Positive

/-!
### Size positivity

A shape whose dimensions are all positive has a positive number of entries, and conversely. Many
reductions are defined only for nonempty dimensions, and `WellFormed` is the library's usual way to
state that assumption.
-/

/-- If `s.Positive`, then `Shape.size s > 0`. -/
theorem size_pos_of_positive : ∀ {s : Shape}, s.Positive → 0 < Shape.size s
  | .scalar, _ => by
      simp
  | .dim n s, hw => by
      rcases hw with ⟨hn, hs⟩
      simpa using Nat.mul_pos hn (size_pos_of_positive (s := s) hs)

/-- A shape of positive total size has no zero dimension. -/
theorem positive_of_size_pos : ∀ {s : Shape}, 0 < Shape.size s → s.Positive
  | .scalar, _ => trivial
  | .dim _ _, h =>
      ⟨Nat.pos_of_mul_pos_right h, positive_of_size_pos (Nat.pos_of_mul_pos_left h)⟩

/-- Every in-bounds axis of a well-formed shape has positive extent. -/
theorem nonemptyAxis_of_positive {s : Shape} (hw : s.Positive) {axis : Nat}
    (hAxis : axis < s.rank) : NonemptyAxis axis s := by
  induction s generalizing axis with
  | scalar => simp at hAxis
  | dim n rest ih =>
      rcases hw with ⟨hn, hRest⟩
      cases axis with
      | zero =>
          obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hn)
          exact .zero
      | succ axis =>
          apply NonemptyAxis.succ
          apply ih hRest
          simp only [Shape.rank_cons] at hAxis
          grind

/-- Package a valid axis of a well-formed shape as inferred reduction-axis evidence. -/
theorem hasNonemptyAxis_of_positive {s : Shape} (hw : s.Positive) {axis : Nat}
    (hAxis : axis < s.rank) : HasNonemptyAxis axis s :=
  ⟨nonemptyAxis_of_positive hw hAxis⟩

/--
Typeclass wrapper for `Shape.Positive`.

Like the library's other side conditions it is a class, so call sites stay clean and a proof can be
supplied locally when needed (`letI : Shape.WellFormed s := ...`).
-/
class WellFormed (s : Shape) : Prop where
  /-- Every extent of `s` is positive. -/
  proof : s.Positive

/-- Scalars are always well-formed. -/
instance : WellFormed .scalar where
  proof := trivial

/-- Adding a nonzero dimension preserves well-formedness. -/
instance {n s} [Shape.WellFormed s] [NeZero n] : Shape.WellFormed (.dim n s) :=
  ⟨⟨Nat.pos_of_ne_zero (NeZero.ne n), Shape.WellFormed.proof⟩⟩

/-- Infer nonemptiness of any valid axis from `WellFormed s`. -/
theorem hasNonemptyAxis_of_lt_rank {s : Shape} [hw : WellFormed s] {axis : Nat}
    (hAxis : axis < s.rank) : HasNonemptyAxis axis s :=
  hasNonemptyAxis_of_positive hw.proof hAxis

/-!
`padLeft n s` prepends `n` singleton dimensions to a shape, as `unsqueeze(0)` applied `n` times
does in PyTorch. Broadcasting takes the same step when it pads the shorter shape with leading `1`s.
-/
/-- Prepend `n` leading singleton dimensions (size `1`) to a shape. -/
def padLeft (n : Nat) (s : Shape) : Shape :=
  List.replicate n 1 ++ s

/-- `padLeft n s` increases the rank by exactly `n`. -/
@[simp] theorem rank_padLeft (n : Nat) (s : Shape) : (padLeft n s).rank = s.rank + n := by
  simp only [padLeft, Shape.rank_eq_length, List.length_append, List.length_replicate, Nat.add_comm]

/-- Leading singleton axes do not change the number of scalar entries. -/
@[simp] theorem size_padLeft (n : Nat) (s : Shape) : (padLeft n s).size = s.size := by
  simp only [padLeft, Shape.size_eq_prod, List.prod_append, List.prod_replicate_nat, Nat.one_pow,
    Nat.one_mul]

end Shape
end TorchLean
