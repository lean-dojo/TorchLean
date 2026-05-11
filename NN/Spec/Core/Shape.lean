/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Logic.Basic

import Init.Grind
import Mathlib.Data.Nat.Init

/-!
# Shapes (`Spec.Shape`)

`Shape` is the type-level “shape descriptor” for tensors in the spec layer.

TorchLean uses *shape-indexed tensors*:

`Tensor α s`

so `Shape` is how we encode the structure of `s` in a way Lean can use for both computation and
proofs.

## Representation

`Shape` is an inductive tree:

- `.scalar`
- `.dim n s`  (a length-`n` dimension whose entries have shape `s`)

This matches the tensor definition in `NN/Spec/Core/Tensor/Core.lean`.

## Common utilities

- `Shape.size : Shape → Nat` is the total number of scalar elements (“numel”).
- `Shape.toList : Shape → List Nat` is a convenient runtime view used by front-ends and bridges.

PyTorch analogy:

- `Shape.toList s` corresponds to `tensor.shape` (a tuple of dimensions).
- `Shape.rank s` corresponds to `tensor.ndim`.
- `Shape.size s` corresponds to `tensor.numel()`.

## Broadcasting and axes

Broadcasting is encoded via `CanBroadcastTo` / `BroadcastTo`.

This is an intentionally *asymmetric* relation ("broadcast `s1` to `s2`"), because most tensor code
is naturally written by choosing the output shape and requiring each input to broadcast to it.

The typeclass wrapper `BroadcastTo` keeps higher-level specs readable: in many cases Lean can infer
the broadcast evidence automatically, so call sites don’t have to manually thread proofs around.

It also defines axis-validity helpers (`valid_axis`) and a `well_formed` predicate for “all
dimensions are positive”, which is useful when you want to rule out degenerate cases in proofs.
-/

@[expose] public section


namespace Spec

-- Shape definition
/-!
We represent shapes as an inductive tree instead of a bare `List Nat` because:

- it matches the tensor representation (`Tensor α s`) structurally, so many definitions are simple
  structural recursion,
- it keeps "scalar vs dim" cases explicit (important for proofs),
- it gives definitional equalities that are friendlier than lists in many places.
-/
/--
Tensor shape descriptor used to index spec-level tensors (`Spec.Tensor α s`).

`Shape` is an outermost-first tree:
- `.scalar` for a scalar,
- `.dim n s` for a length-`n` dimension whose entries have shape `s`.
-/
inductive Shape where
  | scalar : Shape
  | dim : Nat → Shape → Shape
deriving DecidableEq, Repr

namespace Shape

/-- Build a shape from a list of dimensions (outermost first). -/
def ofList : List Nat → Shape
  | [] => .scalar
  | n :: ns => .dim n (ofList ns)

/-- Internal helper: check that a list of axis indices is duplicate-free. -/
def nodupBool (xs : List Nat) : Bool :=
  match xs with
  | [] => true
  | x :: xs => (!xs.contains x) && nodupBool xs

/-- Internal helper: get the `i`-th entry (0-based) from a list of dimensions, defaulting to `0`. -/
def getDim! (xs : List Nat) (i : Nat) : Nat :=
  match xs, i with
  | [], _ => 0
  | x :: _, 0 => x
  | _ :: xs, i+1 => getDim! xs i

/-- Pretty-print a `Shape` for debugging / logs. -/
def pretty (s : Shape) : String :=
  match s with
  | .scalar => "scalar"
  | .dim n rest => s!"dim {n} ({pretty rest})"

/-- Swap two adjacent dimensions at a given depth (0‑based from the outermost). -/
def swapAdjacentAtDepth (s : Shape) (depth : Nat) : Shape :=
  match depth, s with
  | 0, .dim m (.dim n rest) => .dim n (.dim m rest)
  | d+1, .dim m rest => .dim m (swapAdjacentAtDepth rest d)
  | _, _ => s  -- invalid depth, return unchanged

/-- Swapping adjacent dims at depth `depth` twice returns the original shape. -/
theorem swapAdjacentAtDepth_involutive (s : Shape) (depth : Nat) :
    (s.swapAdjacentAtDepth depth).swapAdjacentAtDepth depth = s := by
  induction depth generalizing s with
  | zero =>
      cases s with
      | scalar => simp [swapAdjacentAtDepth]
      | dim m rest =>
          cases rest <;> simp [swapAdjacentAtDepth]
  | succ d ih =>
      cases s <;> simp [swapAdjacentAtDepth, ih]

/-- Append a new innermost dimension. -/
def appendDim (s : Shape) (n : Nat) : Shape :=
  match s with
  | .scalar => .dim n .scalar
  | .dim m rest => .dim m (appendDim rest n)

/-- Total number of scalar elements (a.k.a. “numel”). -/
def size : Shape → Nat
  | .scalar => 1
  | .dim n rest => n * size rest

/-- `size` for a 2D shape factors as `a * b * size s`. -/
theorem size_dim_mul (a b : Nat) (s : Shape) :
    size (dim a (dim b s)) = a * b * size s := by
  simp [size, Nat.mul_assoc]

/--
`appendDim` multiplies the number of scalar elements by the appended dimension.

This lemma is the standard justification for reshape tricks where we:
- treat a tensor of shape `s.appendDim n` as a matrix of shape `(size s) × n`, or
- append an extra singleton dimension (`n = 1`) without changing `size`.
-/
theorem size_appendDim (s : Shape) (n : Nat) : size (appendDim s n) = size s * n := by
  induction s with
  | scalar =>
      simp [appendDim, size]
  | dim m rest ih =>
      -- `appendDim` recurses to the innermost dimension; `size` is multiplicative.
      simp [appendDim, size, ih, Nat.mul_assoc]

/--
Shape-size identity used in Transformer attention reshapes.

If `dModel = numHeads * headDim`, then:
`(seqLen × dModel)` has the same `size` as `(numHeads × seqLen × headDim)`.
-/
theorem size_eq_of_dModel_eq_numHeads_mul_headDim
  (seqLen numHeads dModel headDim : Nat)
  (h : dModel = numHeads * headDim) :
  size (dim seqLen (dim dModel scalar)) = size (dim numHeads (dim seqLen (dim headDim scalar))) :=
    by
  simp [size]
  rw [h]
  -- `Nat.mul_left_comm` proves `a * b * c = b * a * c`; we just reassociate to match our goal.
  simpa [Nat.mul_assoc] using Nat.mul_left_comm seqLen numHeads headDim


/-- Size of the outermost dimension (or 1 for scalar). -/
def dimSize : Shape → Nat
  | .scalar => 1
  | .dim n _ => n

/-- Size of the innermost dimension (or 1 for scalar). -/
def innerDimSize : Shape → Nat
  | .scalar => 1
  | .dim _ inner => innerDimSize inner

/-- Convert to a list of dimensions (outermost first). -/
def toList : Shape → List Nat
  | .scalar => []
  | .dim n rest => n :: toList rest

/-- `ofList` is a left inverse of `toList`. -/
@[simp] theorem ofList_toList (s : Shape) : ofList (toList s) = s := by
  induction s with
  | scalar => rfl
  | dim n s ih =>
    simp [ofList, toList, ih]

/-- `toList` is a right inverse of `ofList`. -/
@[simp] theorem toList_ofList (xs : List Nat) : toList (ofList xs) = xs := by
  induction xs with
  | nil => rfl
  | cons n ns ih =>
    simp [ofList, toList, ih]

-- Tell `grind` about the standard shape normalization lemmas.
attribute [grind =] size_appendDim ofList_toList toList_ofList

/-- Convert to an array of dimensions (outermost first). -/
def toArray (s : Shape) : Array Nat :=
  toList s |>.toArray

/-- Boolean equality test for shapes (structural). -/
def areEqual : Shape → Shape → Bool
  | .scalar, .scalar => true
  | .dim n1 s1, .dim n2 s2 => n1 == n2 && areEqual s1 s2
  | _, _ => false

-- We keep `BEq` as an explicit structural test because it shows up in runtime checks and logs.
/-- `BEq Shape` uses the explicit structural boolean test `Shape.areEqual`. -/
instance : BEq Shape where
  beq := areEqual

/-- Default shape is scalar. -/
instance : Inhabited Shape where
  default := .scalar

/-- Check if shape is a matrix (m × n). -/
def isMatrix : Shape → Option (Nat × Nat)
  | .dim m (.dim n .scalar) => some (m, n)
  | _ => none

/-- Check if shape is a vector (n). -/
def isVector : Shape → Option Nat
  | .dim n .scalar => some n
  | _ => none

/-- Check if shape is scalar. -/
def isScalar : Shape → Bool
  | .scalar => true
  | _ => false

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

/-- Evidence that shape `s₁` can be broadcast to shape `s₂` (PyTorch-style broadcasting). -/
inductive CanBroadcastTo : Shape → Shape → Type where
  | scalar_to_any  (s : Shape) : CanBroadcastTo .scalar s
  | dim_eq {n : Nat} {s₁ s₂ : Shape} (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo (.dim n s₁) (.dim n s₂)
  | dim_1_to_n {n : Nat} {s₁ s₂ : Shape} (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo (.dim 1 s₁) (.dim n s₂)
  | expand_dims {n : Nat} {s₁ s₂ : Shape} (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo s₁ (Shape.dim n s₂)
deriving Repr

/-- Typeclass wrapper for `CanBroadcastTo` so broadcast proofs can be inferred. -/
class BroadcastTo (s₁ s₂ : Shape) where
  proof : CanBroadcastTo s₁ s₂

/-- Scalar broadcasts to any shape (analogue of "prepend 1s and expand"). -/
instance broadcastToScalarLeft (s : Shape) : BroadcastTo Shape.scalar s where
  proof := CanBroadcastTo.scalar_to_any s

/-- Broadcasting preserves equal leading dimensions when the tails broadcast. -/
instance broadcastToDimEq {n : Nat} {s₁ s₂ : Shape} [bc : BroadcastTo s₁ s₂] : BroadcastTo
  (Shape.dim n s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_eq bc.proof

/-- Dimension `1` can broadcast to any `n` (PyTorch's main broadcast rule). -/
instance broadcastToDim1ToN {n : Nat} {s₁ s₂ : Shape} [bc : BroadcastTo s₁ s₂] : BroadcastTo
  (Shape.dim 1 s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_1_to_n bc.proof

/-- Prepend an outer dimension (the "expand_dims" step used to align ranks). -/
instance broadcastToExpandDims {n : Nat} {s₁ s₂ : Shape} [bc : BroadcastTo s₁ s₂] :
    BroadcastTo s₁ (Shape.dim n s₂) where
  proof := CanBroadcastTo.expand_dims bc.proof

/-- `true` iff two shapes have the same number of elements. -/
def isValidReshape (s₁ s₂ : Shape) : Bool :=
  Shape.size s₁ == Shape.size s₂

/-- Rank = number of dimensions (scalar has rank 0). -/
def rank : Shape → Nat
  | Shape.scalar => 0
  | Shape.dim _ rest => 1 + rank rest

/-!
### Friendly aliases (PyTorch-style)

We keep the canonical names (`toList`, `rank`, `size`, `well_formed`) because they show up
throughout the spec/proof code.

For docs and examples, these aliases read more like PyTorch.
-/

/-- PyTorch-style name for `Shape.toList`. -/
abbrev dims (s : Shape) : List Nat := toList s

/-- PyTorch-style name for `Shape.rank`. -/
abbrev ndim (s : Shape) : Nat := rank s

/-- PyTorch-style name for `Shape.size` ("numel"). -/
abbrev numel (s : Shape) : Nat := size s

/-- Permute axes of a shape using a runtime permutation list (0-based). Returns `none` if invalid.
  -/
def permute? (s : Shape) (perm : List Nat) : Option Shape :=
  let r := rank s
  if perm.length != r then
    none
  else if !nodupBool perm then
    none
  else if !(perm.all (fun i => i < r)) then
    none
  else
    let dims := toList s
    -- `dims.length = r` by construction of `toList`/`rank`.
    let dims' := perm.map (fun i => getDim! dims i)
    some (ofList dims')

/-!
## Axis utilities

Why these exist:

- Reduction ops (`reduce_sum`, `reduce_mean`, etc.) need an axis argument.
- In executable code we want to reject invalid axes early, but in spec/proof code we want the axis
  validity to be available as evidence that can be carried through lemmas.

So we provide:
- `valid_axis axis s : Prop` as the core definition, and
- `valid_axis_inst axis s` as a typeclass wrapper so the common cases can be inferred.

PyTorch differences:

- PyTorch allows negative axes (e.g. `dim=-1`); here we use `Nat` axes only (0-based).
  A typical translation is: "last axis" = `Shape.rank s - 1` (when `rank s > 0`).
-/

/--
Evidence that reducing along `axis` is well-defined for a shape.

This is a small helper predicate used to rule out degenerate `0`-length dimensions when stating
laws about reductions.
-/
inductive reducibleAlong : Nat → Shape → Prop
| head {n : Nat} {s : Shape} : reducibleAlong 0 (.dim (n+1) s)  -- must be ≥ 1
| tail {n : Nat} {s : Shape} {k : Nat} :
    reducibleAlong k s → reducibleAlong (k + 1) (.dim (n+1) s)

/-- `simp` lemma: axis `0` is reducible for any positive outer dimension. -/
@[simp] theorem reducibleAlong_head {n : Nat} {s : Shape} :
    reducibleAlong 0 (.dim (n+1) s) :=
  reducibleAlong.head

/-- `simp` lemma: reducibility for inner axis lifts to the next outer axis. -/
@[simp] theorem reducibleAlong_tail {n : Nat} {s : Shape} {k : Nat} (h : reducibleAlong k s) :
    reducibleAlong (k + 1) (.dim (n+1) s) :=
  reducibleAlong.tail h

/-!
`valid_axis axis s` means that `axis` is a valid reduction axis for `s`.

We use a Prop + typeclass wrapper (`valid_axis_inst`) so proofs can be synthesized by typeclass
resolution in downstream code.
-/
/-- Axis validity predicate for reduction ops (0-based axis in `Nat`). -/
inductive valid_axis : Nat → Shape → Prop
| valid_zero {n s} : valid_axis 0 (.dim (n+1) s)
| valid_succ {n s k} (h : valid_axis k s) : valid_axis (k+1) (.dim (n+1) s)

-- Typeclass wrapper for valid_axis proof
/-- Typeclass wrapper for `valid_axis` so common axis proofs can be inferred. -/
class valid_axis_inst (axis : Nat) (s : Shape) where
  (proof : valid_axis axis s)

-- Instance for zero
/-- Instance: axis `0` is valid for any positive outer dimension. -/
instance validAxisInstZero {n s} : valid_axis_inst 0 (.dim (n+1) s) :=
  { proof := valid_axis.valid_zero }

/--
Instance: axis `0` is valid for a nonzero outer dimension `n`.

This is a convenience wrapper that turns `n ≠ 0` into the `n+1` form expected by `valid_axis`.
-/
instance validAxisInstZeroAlt {n s} (h : n ≠ 0) : valid_axis_inst 0 (.dim n s) :=
  let ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero h
  { proof := by rw [hm]; exact valid_axis.valid_zero }

-- 1 is a valid axis if the inner shape has a valid 0
/-- Instance: axis `1` is valid for a 2D shape when both outer dims are nonzero. -/
instance validAxisInstOne {n1 n2 s} (h₁ : n1 ≠ 0) (h₂ : n2 ≠ 0) :
    valid_axis_inst 1 (.dim n1 (.dim n2 s)) :=
  let ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero h₁
  { proof := by
      rw [hm]
      exact valid_axis.valid_succ (validAxisInstZeroAlt h₂).proof }

-- Instance for successor
/-- Instance: if `k` is a valid axis for `s`, then `k+1` is a valid axis for `.dim (n+1) s`. -/
instance validAxisInstSucc {n s k} [inst : valid_axis_inst k s] : valid_axis_inst (k+1) (.dim
  (n+1) s) :=
  { proof := valid_axis.valid_succ inst.proof }

-- If a caller already has `n > 0`, this instance packages it as `n ≠ 0`.
/-- Instance: axis `0` is valid if you have a positivity proof `n > 0` (converted to `n ≠ 0`). -/
instance validAxisInstZeroAlt2 {n s} (h : n > 0) : valid_axis_inst 0 (.dim n s) :=
  validAxisInstZeroAlt (n := n) (s := s) (Nat.ne_of_gt h)

-- Small lemma used at many call sites: positivity implies nonzero.
/-- Helper lemma: a positive natural is not zero. -/
theorem gt_pos_to_ne_zero {n : Nat} (h : n > 0) : n ≠ 0 :=
  Nat.ne_of_gt h


/-!
## Well-formedness (`well_formed`)

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

This is a pragmatic "make the common case pleasant" choice: proofs and specs are shorter, and
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

/-- If `s.well_formed`, then `Shape.size s > 0`. -/
theorem size_pos_of_well_formed : ∀ {s : Shape}, s.wellFormed → 0 < Shape.size s
  | .scalar, _ => by
      simp [Shape.size]
  | .dim n s, hw => by
      rcases hw with ⟨hn, hs⟩
      simpa [Shape.size] using Nat.mul_pos hn (size_pos_of_well_formed (s := s) hs)

-- Instance for the last axis (rank - 1) being valid for well-formed shapes
/--
If `rank s > 0` and `s` is well-formed, then the last axis `rank s - 1` is valid.

This powers many "reduce over last dimension" specs where the axis is computed as `rank s - 1`.
-/
instance validAxisLastInst {s : Shape} (h : Shape.rank s > 0) (hw : s.wellFormed) :
  valid_axis_inst (Shape.rank s - 1) s := {
  proof := by
    -- We'll prove this using strong induction on the rank
    suffices ∀ r : Nat, ∀ s' : Shape, Shape.rank s' = r → r > 0 → s'.wellFormed → valid_axis (r -
      1) s' by
      exact this (Shape.rank s) s rfl h hw

    intro r
    induction r using Nat.strong_induction_on with
    | h r ih =>
      intro s' hs' hr' hw'
      cases s' with
      | scalar =>
        simp [Shape.rank] at hs'
        rw [hs'] at hr'
        grind
      | dim n s'' =>
        simp [Shape.rank] at hs' ⊢
        -- Extract well-formedness properties
        have ⟨h_n_pos, hw''⟩ := hw'
        -- We know n > 0 from well-formedness
        obtain ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt h_n_pos)
        rw [hm]
        -- We have Shape.rank s'' + 1 = r, so Shape.rank s'' = r - 1
        have hs'' : Shape.rank s'' = r - 1 := by grind
        -- We need to prove valid_axis (r - 1) (.dim (m+1) s'')
        -- Since r > 0, we have r - 1 = Shape.rank s''
        rw [← hs'']

        cases Nat.eq_zero_or_pos (Shape.rank s'') with
        | inl h_zero =>
          -- If Shape.rank s'' = 0, we need valid_axis 0 (.dim (m+1) s'')
          rw [h_zero]
          exact valid_axis.valid_zero
        | inr h_pos_inner =>
          -- If Shape.rank s'' > 0, use inductive hypothesis
          have rank_eq : Shape.rank s'' = (Shape.rank s'' - 1) + 1 :=
            Eq.symm (Nat.sub_add_cancel (Nat.succ_le_of_lt h_pos_inner))
          rw [rank_eq]
          apply valid_axis.valid_succ

          -- Apply IH: we need Shape.rank s'' < r, Shape.rank s'' > 0, and s''.well_formed
          have h_lt : Shape.rank s'' < r := by
            rw [hs'']
            grind
          exact ih (Shape.rank s'') h_lt s'' rfl h_pos_inner hw''
}

/--
Typeclass wrapper for `Shape.well_formed`.

We use a typeclass (instead of passing a `well_formed` proof everywhere) because it mirrors how
other "side conditions" are handled in the library: call sites stay clean, and instances can be
provided locally (e.g. `letI : Shape.WellFormed s := ...`) when needed.
-/
class WellFormed (s : Shape) : Prop where
  proof : s.wellFormed

-- Scalars are always well-formed.
/-- Scalars are always well-formed. -/
instance : WellFormed .scalar where
  proof := trivial

-- If the inner shape is well-formed and the new dimension is positive, the result is well-formed.
/-- If `s` is well-formed and `n > 0`, then `.dim n s` is well-formed. -/
instance {n s} [WellFormed s] (h : n > 0) : WellFormed (.dim n s) where
  proof := ⟨h, WellFormed.proof⟩

-- Helper to create well-formedness for positive literals
/-
These small instances are purely about ergonomics.

In a lot of specs/examples we write shapes with concrete dimensions like `1` and `2` (bias vectors,
small CNN channels, etc.). Having `WellFormed` discharge automatically keeps call sites focused on
the math/model rather than on proof mechanics.
-/
/-- Convenience instance: `.dim 1 s` is well-formed when `s` is. -/
instance posDim1Wf {s} [Shape.WellFormed s] : Shape.WellFormed (.dim 1 s) :=
  Shape.WellFormed.mk ⟨by decide, Shape.WellFormed.proof⟩

-- Same idea as `posDim1Wf`, but for the common literal `2`.
/-- Convenience instance: `.dim 2 s` is well-formed when `s` is. -/
instance posDim2Wf {s} [Shape.WellFormed s] : Shape.WellFormed (.dim 2 s) :=
  Shape.WellFormed.mk ⟨by decide, Shape.WellFormed.proof⟩

-- General helper: if you already have a `Fact (n > 0)` in scope, lift it into `WellFormed`.
/-- If a `Fact (n > 0)` is in scope, lift it to a `Shape.WellFormed (.dim n s)` instance. -/
instance {n s} [Shape.WellFormed s] [h : Fact (n > 0)] : Shape.WellFormed (.dim n s) :=
  ⟨⟨h.out, Shape.WellFormed.proof⟩⟩

/-!
`validAxisLastAuto` is a convenience instance for the most common reduction axis:
"reduce over the last dimension".

In PyTorch this is `dim=-1` (after normalization). Here we stay in `Nat`, so the last axis is
`rank s - 1`, and we require `rank s > 0` plus well-formedness so the proof is meaningful.
-/
/-- Convenience instance: infer `valid_axis_inst (rank s - 1) s` from `WellFormed s` and `rank s >
  0`. -/
instance validAxisLastAuto {s : Shape} [h_wf : WellFormed s] (h : Shape.rank s > 0) :
  Shape.valid_axis_inst (Shape.rank s - 1) s :=
  validAxisLastInst h h_wf.proof

/-!
Bridge lemma: turn a `valid_axis` proof into a `reducibleAlong` proof.

Why both exist:
- `valid_axis` is the semantic "this axis makes sense" predicate used in public APIs.
- `reducibleAlong` is a structurally convenient predicate for recursion over tensor shapes
  (it lines up with how `Tensor.dim` is constructed).

This function is the adapter between the two views.
-/
/-- Convert a `valid_axis` proof into a structurally convenient `reducibleAlong` proof. -/
def proveReducibleAlong (axis : Nat) (s : Shape) (h : valid_axis axis s) : reducibleAlong axis s
  :=
  match h with
  | valid_axis.valid_zero => reducibleAlong.head
  | valid_axis.valid_succ h' => reducibleAlong.tail (proveReducibleAlong _ _ h')

/-!
`padLeft n s` prepends `n` singleton dimensions to a shape.

PyTorch analogy: `unsqueeze(0)` repeated `n` times (or equivalently viewing a tensor as having
extra leading dimensions of size 1). This is also the "prepend 1s" step you see in broadcasting.
-/
/-- Prepend `n` leading singleton dimensions (size `1`) to a shape. -/
def padLeft : Nat → Shape → Shape
| 0, s => s
| (n+1), s => dim 1 (padLeft n s)

-- Padding with leading `1`s increases rank by exactly `n`.
/-- `padLeft n s` increases the rank by exactly `n`. -/
theorem padLeft_rank : ∀ n s, (padLeft n s).rank = n + s.rank
| 0, s => by simp [padLeft]
| n+1, s => by
  simp [padLeft, rank]
  rw [padLeft_rank n]
  grind

end Shape
end Spec
