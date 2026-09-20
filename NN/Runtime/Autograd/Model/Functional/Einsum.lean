/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.Runtime.Autograd.Model.Program

@[expose] public section


namespace Runtime
namespace Autograd
namespace Model

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace F

/-! ## Einsum-ish building blocks -/

/-!
## Typed einsum wrappers (fast, total)

These are non-`Option` equivalents for the most common einsum contractions in ML code.
They are intended to be used directly (no string parsing), and serve as the fast-path targets for
`einsum?`.
-/

/-- `einsum("ij,jk->ik", A, B)` as a typed matmul. -/
def einsumIjJkIk {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {iDim jDim kDim : Nat}
    (a : RefTy (m := m) (α := α) [iDim, jDim])
    (b : RefTy (m := m) (α := α) [jDim, kDim]) :
    m (RefTy (m := m) (α := α) [iDim, kDim]) :=
  Runtime.Autograd.Torch.matmul (m := m) (α := α)
    (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
    (mDim := iDim) (nDim := jDim) (pDim := kDim) a b

/-- `einsum("bij,bjk->bik", A, B)` as a typed batched matmul. -/
def einsumBijBjkBik {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch iDim jDim kDim : Nat}
    (a : RefTy (m := m) (α := α) [batch, iDim, jDim])
    (b : RefTy (m := m) (α := α) [batch, jDim, kDim]) :
    m (RefTy (m := m) (α := α) [batch, iDim, kDim]) :=
  Runtime.Autograd.Torch.matmul (m := m) (α := α)
    (batchA := [batch]) (batchB := [batch]) (batch := [batch])
    (mDim := iDim) (nDim := jDim) (pDim := kDim) a b

/-- Einsum pattern used in attention: `bhid,bhjd -> bhij` (batched Q·Kᵀ per head). -/
def einsumBhidBhjdBhij {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch heads iDim jDim dDim : Nat}
    (q : RefTy (m := m) (α := α) [batch, heads, iDim, dDim])
    (k : RefTy (m := m) (α := α) [batch, heads, jDim, dDim]) :
    m (RefTy (m := m) (α := α) [batch, heads, iDim, jDim]) := do
  let bh : Nat := batch * heads
  let sQ4 : Shape := [batch, heads, iDim, dDim]
  let sK4 : Shape := [batch, heads, jDim, dDim]
  let sQ3 : Shape := [bh, iDim, dDim]
  let sK3 : Shape := [bh, jDim, dDim]
  let sKT : Shape := [bh, dDim, jDim]
  let sOut3 : Shape := [bh, iDim, jDim]
  let sOut4 : Shape := [batch, heads, iDim, jDim]
  have hQ : Spec.Shape.size sQ4 = Spec.Shape.size sQ3 := by
    simp [sQ4, sQ3, Spec.Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  have hK : Spec.Shape.size sK4 = Spec.Shape.size sK3 := by
    simp [sK4, sK3, Spec.Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  have hOut : Spec.Shape.size sOut3 = Spec.Shape.size sOut4 := by
    simp [sOut3, sOut4, Spec.Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  let q3 ← reshape (m := m) (α := α) (s₁ := sQ4) (s₂ := sQ3) q hQ
  let k3 ← reshape (m := m) (α := α) (s₁ := sK4) (s₂ := sK3) k hK
  let kt ← Runtime.Autograd.Torch.swapAdjacentAtDepth
    (m := m) (α := α) (s := sK3) 1 k3
  let out3 ← Runtime.Autograd.Torch.matmul (m := m) (α := α)
    (batchA := [bh]) (batchB := [bh]) (batch := [bh])
    (mDim := iDim) (nDim := dDim) (pDim := jDim) q3 kt
  reshape (m := m) (α := α) (s₁ := sOut3) (s₂ := sOut4) out3 hOut

/-- Einsum pattern used in attention: `bhij,bhjd -> bhid` (batched Attn·V per head). -/
def einsumBhijBhjdBhid {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch heads iDim jDim dDim : Nat}
    (attn : RefTy (m := m) (α := α) [batch, heads, iDim, jDim])
    (v : RefTy (m := m) (α := α) [batch, heads, jDim, dDim]) :
    m (RefTy (m := m) (α := α) [batch, heads, iDim, dDim]) := do
  let bh : Nat := batch * heads
  let sA4 : Shape := [batch, heads, iDim, jDim]
  let sV4 : Shape := [batch, heads, jDim, dDim]
  let sA3 : Shape := [bh, iDim, jDim]
  let sV3 : Shape := [bh, jDim, dDim]
  let sOut3 : Shape := [bh, iDim, dDim]
  let sOut4 : Shape := [batch, heads, iDim, dDim]
  have hA : Spec.Shape.size sA4 = Spec.Shape.size sA3 := by
    simp [sA4, sA3, Spec.Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  have hV : Spec.Shape.size sV4 = Spec.Shape.size sV3 := by
    simp [sV4, sV3, Spec.Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  have hOut : Spec.Shape.size sOut3 = Spec.Shape.size sOut4 := by
    simp [sOut3, sOut4, Spec.Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  let a3 ← reshape (m := m) (α := α) (s₁ := sA4) (s₂ := sA3) attn hA
  let v3 ← reshape (m := m) (α := α) (s₁ := sV4) (s₂ := sV3) v hV
  let out3 ← Runtime.Autograd.Torch.matmul (m := m) (α := α)
    (batchA := [bh]) (batchB := [bh]) (batch := [bh])
    (mDim := iDim) (nDim := jDim) (pDim := dDim) a3 v3
  reshape (m := m) (α := α) (s₁ := sOut3) (s₂ := sOut4) out3 hOut

/-! ## General einsum (PyTorch-style subscripts; runtime-checked) -/

namespace Einsum

-- Needed for runtime checks (e.g. to build a `HasNonemptyAxis` for reductions).
/-- Decidable instance for `Shape.wellFormed`, used by the dynamic einsum lowering. -/
def wellFormedDec : (s : Shape) → Decidable s.wellFormed
  | .scalar => isTrue trivial
  | .dim n s =>
      match (inferInstance : Decidable (n > 0)) with
      | isTrue hn =>
          match wellFormedDec s with
          | isTrue hs => isTrue ⟨hn, hs⟩
          | isFalse hs => isFalse (fun h => hs h.2)
      | isFalse hn =>
          isFalse (fun h => hn h.1)

/-- Local decidability instance for `Shape.wellFormed` (used by the dynamic einsum lowering). -/
instance (s : Shape) : Decidable s.wellFormed :=
  wellFormedDec s

/--
Label used by the dynamic einsum parser.

`chr c` is a concrete axis label like `'i'` or `'j'`, while `ell k` is a generated ellipsis label
that stands for "some number of unnamed batch-like axes".
-/
inductive Label where
  | chr (c : Char)
  | ell (k : Nat)
  deriving BEq, DecidableEq, Repr

/--
One operand’s subscript, split around an optional ellipsis.

For example, parsing `"ab...cd"` yields:
- `pre = ['a','b']`
- `post = ['c','d']`
- `hasEll = true`
-/
structure Subscript where
  /-- Index letters before the ellipsis. -/
  pre : List Char
  /-- Index letters after the ellipsis, empty when there is none. -/
  post : List Char
  /-- Whether an ellipsis was present. Without this flag `"ab"` and `"ab..."` would parse to the
  same pair of lists, and they mean different things once batch axes are matched up. -/
  hasEll : Bool
  deriving Repr

/-- Remove ASCII whitespace to simplify the hand-rolled parser. -/
def stripSpaces (s : String) : String :=
  String.ofList <| s.toList.filter (fun c => c != ' ' && c != '\n' && c != '\t' && c != '\r')

/-- Parse a single operand subscript (with at most one `...`). -/
def parseSubscript (raw : String) : Except String Subscript := do
  let s := stripSpaces raw
  let parts := s.splitOn "..."
  match parts with
  | .cons one .nil =>
      pure { pre := one.toList, post := [], hasEll := false }
  | .cons a (.cons b .nil) =>
      pure { pre := a.toList, post := b.toList, hasEll := true }
  | _ =>
      throw s!"einsum: invalid subscript `{raw}` (ellipsis `...` may appear at most once)"

/-- Parsed `einsum` equation: input subscripts and an optional explicit output subscript. -/
structure Parsed where
  /-- One subscript per operand, in the order they appear left of `->`. -/
  inputs : List Subscript
  /-- Explicit output subscript, or `none` for the implicit form, where the output axes are the
  letters appearing exactly once, in alphabetical order (the NumPy convention). -/
  output? : Option Subscript
  deriving Repr

/-- Parse an equation of the form `"a,b->c"` or `"a,b"` (implicit output). -/
def parseEquation (raw : String) : Except String Parsed := do
  let s := stripSpaces raw
  let parts := s.splitOn "->"
  match parts with
  | .cons lhs .nil =>
      let insRaw := lhs.splitOn ","
      let ins ← insRaw.mapM parseSubscript
      pure { inputs := ins, output? := none }
  | .cons lhs (.cons rhs .nil) =>
      let insRaw := lhs.splitOn ","
      let ins ← insRaw.mapM parseSubscript
      let out ← parseSubscript rhs
      pure { inputs := ins, output? := some out }
  | _ =>
      throw s!"einsum: invalid equation `{raw}` (expected `lhs` or `lhs->rhs`)"

/-- Detect whether a list of labels contains any duplicates (order-preserving scan). -/
def hasDupLabels (xs : List Label) : Bool :=
  let rec go (seen : List Label) : List Label → Bool
    | .nil => false
    | .cons x xs => if seen.contains x then true else go (x :: seen) xs
  go [] xs

/--
Convert a permutation of axes into a sequence of adjacent swaps.

This validates that `perm` has length `r` and only mentions axes below `r`, then delegates to the
IR lowering strategy `NN.IR.Graph.swapDepthsForPerm`: a general permutation is represented as a
list of swap depths, and swaps are implemented with `swapAdjacentAtDepth`.
-/
def swapDepthsForPerm? (perm : List Nat) (r : Nat) : Option (List Nat) :=
  if perm.length = r && perm.all (fun d => d < r) then
    (NN.IR.Graph.swapDepthsForPerm perm.toArray r).toOption.map Array.toList
  else
    none

/--
Expand an input operand’s labels to a full label list matching the operand’s rank.

If the subscript contains an ellipsis, this inserts fresh `Label.ell` labels so that the total
label count matches `Spec.Shape.rank s`.
-/
def expandInputLabels (sub : Subscript) (s : Shape) (maxEll : Nat) : Except String (List Label) :=
  do
  let r := Spec.Shape.rank s
  let fixed := sub.pre.length + sub.post.length
  if sub.hasEll then
    if r < fixed then
      throw s!"einsum: subscript has too many labels for shape rank={r}"
    let ellCount := r - fixed
    if ellCount > maxEll then
      throw s!"einsum: internal error (ellipsis count {ellCount} > maxEll {maxEll})"
    let offset := maxEll - ellCount
    let ellLabels := (List.range ellCount).map (fun i => Label.ell (offset + i))
    pure <| sub.pre.map Label.chr ++ ellLabels ++ sub.post.map Label.chr
  else
    if r != fixed then
      throw s!"einsum: subscript label count {fixed} does not match shape rank={r}"
    pure <| sub.pre.map Label.chr

/-- Expand output labels, materializing the full ellipsis range `[0..maxEll)` when present. -/
def expandOutputLabels (sub : Subscript) (maxEll : Nat) : Except String (List Label) := do
  if sub.hasEll then
    let ellLabels := (List.range maxEll).map Label.ell
    pure <| sub.pre.map Label.chr ++ ellLabels ++ sub.post.map Label.chr
  else
    pure <| sub.pre.map Label.chr

/-!
### Small association-list helpers

To keep this file dependency-light, we represent maps as association lists and use small helpers
instead of `Std.HashMap`.
-/

/-- Occurrence counts for labels, represented as an association list. -/
abbrev Counts : Type := List (Label × Nat)

/-- Look up how many times a parsed einsum label occurs in the current signature. -/
def countsFind? (cs : Counts) (x : Label) : Option Nat :=
  (cs.find? (fun p => p.1 == x)).map (fun p => p.2)

/-- Number of occurrences of a label. An absent label has count zero. -/
def labelCount (cs : Counts) (x : Label) : Nat :=
  (countsFind? cs x).getD 0

/-- Increment a label’s count (inserting it if absent). -/
def countsInc (cs : Counts) (x : Label) : Counts :=
  let rec go : Counts → Counts
    | .nil => [(x, 1)]
    | .cons (y, n) ys =>
        if y == x then
          (y, n + 1) :: ys
        else
          (y, n) :: go ys
  go cs

/-- Count all labels across a list of operands. -/
def labelCounts (xss : List (List Label)) : Counts :=
  xss.foldl (fun acc xs => xs.foldl countsInc acc) []

/-- Keep first occurrences of labels, preserving order. -/
def orderedUnique (xs : List Label) : List Label :=
  let rec go (seen : List Label) : List Label → List Label
    | .nil => []
    | .cons x xs =>
        if seen.contains x then go seen xs
        else x :: go (x :: seen) xs
  go [] xs

/-- Map each label to its concrete dimension size (association list). -/
abbrev DimMap : Type := List (Label × Nat)

/-- Lookup a label’s dimension size. -/
def dimFind? (mp : DimMap) (x : Label) : Option Nat :=
  (mp.find? (fun p => p.1 == x)).map (fun p => p.2)

/-- Insert/update a label’s dimension size in a `DimMap`. -/
def dimUpdate (mp : DimMap) (x : Label) (d : Nat) : DimMap :=
  let rec go : DimMap → DimMap
    | .nil => [(x, d)]
    | .cons (y, d0) ys =>
        if y == x then
          (y, d) :: ys
        else
          (y, d0) :: go ys
  go mp

/--
Infer a consistent label-to-dimension map from operand label lists and operand shapes.

This implements standard einsum broadcast rules: if a label is seen with both `d` and `1`, we keep
`d`; if two non-`1` sizes disagree, we error.
-/
def labelDimMap (xss : List (List Label)) (shapes : List Shape) : Except String DimMap := do
  let mut mp : DimMap := []
  for (xs, s) in List.zip xss shapes do
    let dims := Shape.toList s
    if xs.length != dims.length then
      throw "einsum: internal error (label/dim length mismatch)"
    for (lbl, d) in List.zip xs dims do
      match dimFind? mp lbl with
      | none => mp := dimUpdate mp lbl d
      | some d0 =>
          if d0 = d then
            pure ()
          else if d0 = 1 then
            mp := dimUpdate mp lbl d
          else if d = 1 then
            pure ()
          else
            throw s!"einsum: incompatible sizes for label {repr lbl}: {d0} vs {d}"
  pure mp

/--
Apply a permutation expressed as adjacent swap depths to an existentially-shaped tensor.

This is the runtime “apply swaps” primitive used by both `.permute` and the dynamic einsum output
reordering.
-/
def permuteBySwaps {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (x : Σ s : Shape, RefTy (m := m) (α := α) s) (swaps : List Nat) :
    m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
  let mut cur := x
  for d in swaps do
    -- This is definitional: each swap updates the shape index.
    let cur' : RefTy (m := m) (α := α) (cur.fst.swapAdjacentAtDepth d) ←
      swapAdjacentAtDepth (m := m) (α := α) (s := cur.fst) d cur.snd
    cur := ⟨cur.fst.swapAdjacentAtDepth d, cur'⟩
  pure cur

/-- Apply adjacent swaps while retaining the resulting shape in the return type. -/
def permuteBySwapsTyped {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)] {s : Shape}
    (x : RefTy (m := m) (α := α) s) :
    (depths : List Nat) →
      m (RefTy (m := m) (α := α) (s.applyAdjacentSwaps depths))
  | .nil => pure x
  | .cons depth depths => do
      let moved ← swapAdjacentAtDepth (m := m) (α := α) (s := s) depth x
      permuteBySwapsTyped (m := m) (α := α) moved depths

/-- Remove the element at index `n` (0-based), leaving the list unchanged if out of bounds. -/
def removeAt {α : Type} : List α → Nat → List α
  | .nil, _ => []
  | .cons _ xs, 0 => xs
  | .cons x xs, n + 1 => x :: removeAt xs n

/--
Compute a permutation that maps `src` to `tgt` when duplicates are present.

This is used for the “diagonal embedding” case when output labels contain repeats: we temporarily
expand the output with extra axes, then permute back to the requested (possibly-duplicated) order.
-/
def permForDuplicateLabels? (src tgt : List Label) : Option (List Nat) :=
  if _hLen : src.length = tgt.length then
    let rec findUnusedIndex? (used : List Nat) (l : Label) (i : Nat) : List Label → Option Nat
      | .nil => none
      | .cons x xs =>
          if x == l && !(used.contains i) then
            some i
          else
            findUnusedIndex? used l (i + 1) xs
    let rec go (used : List Nat) (tgt : List Label) (acc : List Nat) : Option (List Nat) :=
      match tgt with
      | .nil => some acc.reverse
      | .cons l ls =>
          match findUnusedIndex? used l 0 src with
          | none => none
          | some i => go (i :: used) ls (i :: acc)
    go [] tgt []
  else
    none

/-
Build a diagonal mask tensor (spec-level) for diagonal embedding/extraction.

Given axes `p` and `q`, the resulting tensor is `1` when the indices along those axes agree, and `0`
otherwise. The `ip`/`iq` parameters track the first-seen indices while recursing over dimensions.
-/
namespace Internal

/-- Recursive worker for the diagonal mask, carrying the indices seen so far on axes `p` and `q`.

Both axes are counted down as the recursion walks `dims`, so `0` means "this axis"; when the
recursion bottoms out the two remembered indices decide the entry. -/
def diagMaskSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α] :
    (dims : List Nat) → (p q : Nat) → (ip iq : Option Nat) → Tensor α (Shape.ofList dims)
  | .nil, _, _, ip, iq =>
      Tensor.scalar <|
        match ip, iq with
        | some i, some j => if i = j then 1 else 0
        | _, _ => 1
  | .cons d ds, p, q, ip, iq =>
      Tensor.dim fun i : Fin d =>
        let ip' :=
          match p with
          | 0 =>
              match ip with
              | none => some i.1
              | some v => some v
          | _ + 1 => ip
        let iq' :=
          match q with
          | 0 =>
              match iq with
              | none => some i.1
              | some v => some v
          | _ + 1 => iq
        diagMaskSpec ds (Nat.pred p) (Nat.pred q) ip' iq'

end Internal

/-- Diagonal mask specification with fresh index-tracking state. -/
def diagMaskSpec {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (dims : List Nat) (p q : Nat) :
    Tensor α (Shape.ofList dims) :=
  Internal.diagMaskSpec (α := α) dims p q none none

/-- Specialize `diagMaskSpec` to a concrete `Shape`. -/
def diagMaskForShape {α : Type} [TorchLean.Storage α] [Zero α] [One α]
    (s : Shape) (p q : Nat) : Tensor α s :=
  diagMaskSpec (α := α) (Shape.toList s) p q

/-- Return the first duplicate label in `xs`, along with its original and duplicate positions. -/
def firstDup? (xs : List Label) : Option (Label × Nat × Nat) :=
  let rec go (seen : List (Label × Nat)) (i : Nat) : List Label → Option (Label × Nat × Nat)
    | .nil => none
    | .cons x xs =>
        match seen.find? (fun p => p.1 == x) with
        | some p => some (x, p.2, i)
        | none => go ((x, i) :: seen) (i + 1) xs
  go [] 0 xs

/-- Permutation list that moves `axis` to the last position (keeping relative order of others). -/
def permMoveAxisToLast (r axis : Nat) : List Nat :=
  (List.range axis) ++ ((List.range (r - (axis + 1))).map (fun i => axis + 1 + i)) ++ [axis]

end Einsum

end F
end Model
end Autograd
end Runtime
