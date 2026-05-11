/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Float32
public import NN.Spec.Core.Complex
public import NN.Spec.Core.Utils
import Mathlib.Algebra.Order.Algebra

/-!
# Tensor API Implementation

User-facing tensor API for TorchLean.

This is the implementation leaf behind the public `NN.Tensor` umbrella. It is the ergonomic layer
that sits on top of the spec-first tensor semantics in `NN.Spec.*`. It does not introduce new math;
it provides constructors and syntax intended for examples, tests, small models, and quickstarts.

Typical responsibilities here:

- shape aliases such as `Shape.Vec` and `Shape.NCHW`,
- friendly constructors from lists and literals,
- small syntax helpers for examples/tests,
- and lightweight printing support for runtime-friendly dtypes.

Notation policy:
- keep compact literal constructors (`shape!`, `tensor!`, `tensorND!`, `tensorF!`, `tensor32!`,
  `fin!`) in this module so they are easy to find and do not leak into unrelated imports,
- keep more semantic glyphs scoped when possible (as in `NN.IR` and `NN.Floats.IEEEExec`),
- and prefer namespace-local aliases over reusing the same unscoped token in multiple layers.

Lean is statically typed, so the element type usually plays the role of “dtype”:

  `let x := NN.Tensor.tensor1d (α := Float) [0.1, 0.2]`
  `let q := NN.Tensor.tensor1d (α := ℚ) [1, 2]`

If you ever see `Tensor Float _` in code: the `_` is just “let Lean infer the shape from the RHS”.
In examples we prefer to omit the type annotation entirely unless it helps readability.

For convenience, this module also provides constructors that start from `Float` literals and cast
into common backends such as `IEEE32Exec`.

Printing: for `Tensor ℝ` we intentionally refuse to pretty-print, since that backend is meant for
proof-oriented mathematics rather than runtime IO. Use `Float`, `IEEE32Exec`, `ℚ`, etc. for values
you actually want to display.
-/

@[expose] public section


namespace NN.Tensor

open Spec

set_option linter.dupNamespace false in
/-- Local alias for the canonical spec tensor type. -/
abbrev Tensor := Spec.Tensor
/-- Local alias for the canonical spec shape type. -/
abbrev Shape := Spec.Shape

namespace Tensor

set_option linter.dupNamespace false in
/-- Scalar tensor constructor alias. -/
abbrev scalar {α : Type} (x : α) : Tensor α Spec.Shape.scalar :=
  Spec.Tensor.scalar x

set_option linter.dupNamespace false in
/-- Dimension tensor constructor alias. -/
abbrev dim {α : Type} {n : Nat} {s : Spec.Shape}
    (xs : Fin n → Tensor α s) : Tensor α (.dim n s) :=
  Spec.Tensor.dim xs

set_option linter.dupNamespace false in
/-- Shape cast alias. -/
abbrev castShape {α : Type} {s₁ s₂ : Spec.Shape} (t : Tensor α s₁) (h : s₁ = s₂) :
    Tensor α s₂ :=
  Spec.Tensor.castShape t h

end Tensor

/-! ## Scalar extraction -/

/--
Extract the scalar value from a scalar-shaped tensor.

PyTorch comparison: like `t.item()` for a 0-dim tensor.
-/
def scalarOf {α : Type} (t : Tensor α Spec.Shape.scalar) : α :=
  match t with
  | .scalar v => v

/--
Dot-notation sugar for scalar tensors: `t.item`.

This is defined at the `Spec.Tensor` namespace so that it works with the canonical tensor type.
-/
abbrev _root_.Spec.Tensor.item {α : Type} (t : Tensor α Spec.Shape.scalar) : α :=
  NN.Tensor.scalarOf t

/-- `Tensor.scalar x` round-trips through `Spec.Tensor.item`. -/
@[simp] theorem _root_.Spec.Tensor.item_scalar {α : Type} (x : α) :
    (Tensor.scalar x).item = x := rfl

/-! ## Common shape aliases -/

namespace Shape

/-- Scalar shape alias. -/
abbrev scalar : Spec.Shape := Spec.Shape.scalar

/-- Dimension shape constructor alias. -/
abbrev dim : Nat → Spec.Shape → Spec.Shape := Spec.Shape.dim

/-- Total number of scalar leaves in a shape. -/
def size (s : Spec.Shape) : Nat := Spec.Shape.size s

/-- Number of dimensions in a shape. -/
def rank (s : Spec.Shape) : Nat := Spec.Shape.rank s

/-- Vector shape `n`. -/
abbrev Vec (n : Nat) : Spec.Shape := .dim n .scalar

/-- Matrix shape `rows × cols`. -/
abbrev Mat (rows cols : Nat) : Spec.Shape := .dim rows (.dim cols .scalar)

/-- 1D signal shape `C × L` (channels-first). PyTorch analogy: `Conv1d` input without batch. -/
abbrev CL (c l : Nat) : Spec.Shape := .dim c (.dim l .scalar)

/-- Batched 1D signal shape `N × C × L`. PyTorch analogy: `Conv1d` input. -/
abbrev NCL (n c l : Nat) : Spec.Shape := .dim n (CL c l)

/-- Image shape `C × H × W` (channel-first). -/
abbrev CHW (c h w : Nat) : Spec.Shape := .dim c (.dim h (.dim w .scalar))

/-- Image shape `H × W × C` (channel-last). -/
abbrev HWC (h w c : Nat) : Spec.Shape := .dim h (.dim w (.dim c .scalar))

/-- Batched image shape `N × C × H × W` (PyTorch default for images). -/
abbrev NCHW (n c h w : Nat) : Spec.Shape := .dim n (.dim c (.dim h (.dim w .scalar)))

/-- Batched image shape `N × H × W × C` (channel-last). -/
abbrev NHWC (n h w c : Nat) : Spec.Shape := .dim n (HWC h w c)

/-- 3D volume shape `C × D × H × W` (channels-first). PyTorch analogy: `Conv3d` input without batch. -/
abbrev CDHW (c d h w : Nat) : Spec.Shape := .dim c (.dim d (.dim h (.dim w .scalar)))

/-- Batched 3D volume shape `N × C × D × H × W`. PyTorch analogy: `Conv3d` input. -/
abbrev NCDHW (n c d h w : Nat) : Spec.Shape := .dim n (CDHW c d h w)

/--
User-facing alias for a single image shape.

This is the shape you usually think of as `(C, H, W)` in PyTorch.
Internally it is the same as `Shape.CHW`.
-/
abbrev Image (c h w : Nat) : Spec.Shape := CHW c h w

/--
User-facing alias for a batch of images.

This is the shape you usually think of as `(N, C, H, W)` in PyTorch.
Internally it is the same as `Shape.NCHW`.
-/
abbrev Images (n c h w : Nat) : Spec.Shape := NCHW n c h w

/-- Conv kernel shape `OutC × InC × kH × kW`. -/
abbrev OIHW (outC inC kH kW : Nat) : Spec.Shape :=
  .dim outC (.dim inC (.dim kH (.dim kW .scalar)))

/-- 1D conv kernel shape `OutC × InC × kL`. -/
abbrev OIL (outC inC kL : Nat) : Spec.Shape :=
  .dim outC (.dim inC (.dim kL .scalar))

/-- 3D conv kernel shape `OutC × InC × kD × kH × kW`. -/
abbrev OIDHW (outC inC kD kH kW : Nat) : Spec.Shape :=
  .dim outC (.dim inC (.dim kD (.dim kH (.dim kW .scalar))))

/-- Generic leading batch dimension: `N × s`. -/
abbrev Batch (n : Nat) (s : Spec.Shape) : Spec.Shape := .dim n s

end Shape

/--
Canonical TorchLean shape alias inside the `NN` namespace.

Most user examples live under `namespace NN...`, where unqualified `Shape` resolves through `NN`.
This alias keeps those examples readable while the underlying type remains `Spec.Shape`.
-/
abbrev _root_.NN.Shape := Spec.Shape

/-- Vector shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.Vec := NN.Tensor.Shape.Vec
/-- Matrix shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.Mat := NN.Tensor.Shape.Mat
/-- 1D channel-first signal shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.CL := NN.Tensor.Shape.CL
/-- Batched 1D channel-first signal shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.NCL := NN.Tensor.Shape.NCL
/-- Channel-first image shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.CHW := NN.Tensor.Shape.CHW
/-- Channel-last image shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.HWC := NN.Tensor.Shape.HWC
/-- Batched channel-first image shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.NCHW := NN.Tensor.Shape.NCHW
/-- Batched channel-last image shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.NHWC := NN.Tensor.Shape.NHWC
/-- Channel-first 3D volume shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.CDHW := NN.Tensor.Shape.CDHW
/-- Batched channel-first 3D volume shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.NCDHW := NN.Tensor.Shape.NCDHW
/-- Single image shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.Image := NN.Tensor.Shape.Image
/-- Batched image shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.Images := NN.Tensor.Shape.Images
/-- Conv2d kernel shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.OIHW := NN.Tensor.Shape.OIHW
/-- Conv1d kernel shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.OIL := NN.Tensor.Shape.OIL
/-- Conv3d kernel shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.OIDHW := NN.Tensor.Shape.OIDHW
/-- Leading-batch shape alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.Batch := NN.Tensor.Shape.Batch
/-- Size alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.size := Spec.Shape.size
/-- Rank alias under `NN.Shape`. -/
abbrev _root_.NN.Shape.rank := Spec.Shape.rank

/-! ## Common tensor type aliases -/

/-- Vector tensor `n` (shape-indexed, like a 1-D PyTorch tensor of length `n`). -/
abbrev VecTensor (α : Type) (n : Nat) := Tensor α (Shape.Vec n)

/-- Matrix tensor `rows × cols` (shape-indexed). -/
abbrev MatTensor (α : Type) (rows cols : Nat) := Tensor α (Shape.Mat rows cols)

/-- Image tensor `C × H × W` (shape-indexed). -/
abbrev ImageTensor (α : Type) (c h w : Nat) := Tensor α (Shape.Image c h w)

/-- Batched image tensor `N × C × H × W` (shape-indexed). -/
abbrev ImagesTensor (α : Type) (n c h w : Nat) := Tensor α (Shape.Images n c h w)

/-! ## Construction -/

/-- Convert a runtime list of dimensions like `[2, 3, 4]` into a nested `Shape`.

Why this exists:
- In the spec layer we usually *carry the shape in the type* (great for safety).
- In “API land” we often start from lists (CLI args, JSON, small examples, …).

`shapeOfDims` is the bridge between those representations.
-/
@[reducible] def shapeOfDims : List Nat → Shape
  | [] => .scalar
  | n :: ns => .dim n (shapeOfDims ns)

/-- `shapeOfDims` round-trips through `Spec.Shape.toList`. -/
@[simp] theorem shapeOfDims_toList (s : Spec.Shape) :
    shapeOfDims (_root_.Spec.Shape.toList s) = s := by
  induction s with
  | scalar => rfl
  | dim n rest ih =>
      simp [_root_.Spec.Shape.toList, shapeOfDims, ih]

/-- Number of scalar elements (“numel”) implied by a runtime `dims` list. -/
def numelDims (dims : List Nat) : Nat :=
  Shape.size (shapeOfDims dims)

/-- Create a 1-D tensor from a Lean `List`.

Notes:
- The shape is computed from `xs.length`, so the type remembers the length.
- This is total and does not perform any runtime checks.

PyTorch analogy: `torch.tensor(xs)` producing a 1D tensor.
-/
def tensor1d (α : Type := Float) (xs : List α) :
    Tensor α (.dim xs.length .scalar) :=
  Spec.fromList1d xs

/-! ### One-hot vectors -/

/-- One-hot vector of length `n`, with a single `1` at index `k`. -/
def oneHot {α : Type} [Zero α] [One α] (n : Nat) (k : Fin n) : Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (if decide (i = k) then (1 : α) else 0))

/-- One-hot vector using a raw `Nat` index.

If `k ≥ n`, we return the all-zeros vector instead of failing.
This is convenient in data-conversion code where carrying a `Fin n` would obscure the caller.
-/
def oneHotNat {α : Type} [Zero α] [One α] (n k : Nat) : Tensor α (.dim n .scalar) :=
  if h : k < n then
    oneHot (α := α) n ⟨k, h⟩
  else
    Spec.fill (0 : α) (.dim n .scalar)

/-- 2-D tensor from nested lists, returning `none` for empty/ragged inputs.

This delegates to `Spec.from_list_2d`, which refuses:
- an empty outer list,
- any empty row,
- or rows with mismatched lengths.

PyTorch analogy: `torch.tensor(xss)` also refuses ragged inputs.
-/
def tensor2d? (α : Type := Float) [Inhabited α] (xss : List (List α)) :
    Option (Tensor α (.dim xss.length (.dim (if xss.isEmpty then 0 else xss.head!.length) .scalar)))
      :=
  Spec.fromList2d xss

/-- 2-D tensor from nested lists, with a human-readable error message on failure. -/
def tensor2d (α : Type := Float) [Inhabited α] (xss : List (List α)) :
    Except String (Tensor α (.dim xss.length (.dim (if xss.isEmpty then 0 else xss.head!.length)
      .scalar))) :=
  match tensor2d? (α := α) xss with
  | some t => .ok t
  | none => .error "tensor2d: empty or ragged nested lists"

/-! ### Ragged-friendly 2D constructors -/

/-- 2-D tensor from nested lists, padding/truncating each row to `nCols`.

This is the permissive sibling of `tensor2d`: it never fails, but it will silently pad with
`default` (and drop extra entries beyond `nCols`). This is useful when you *intend* ragged inputs
(e.g. batching variable-length sequences after padding).

PyTorch analogy: closer to `pad_sequence(..., batch_first=True)` followed by `torch.tensor`.
-/
def tensor2dPadTo (α : Type := Float) [Inhabited α] (nCols : Nat) (xss : List (List α)) :
    Tensor α (.dim xss.length (.dim nCols .scalar)) :=
  Spec.fromList2dPadTo (nCols := nCols) xss

/-- 2-D tensor from nested lists, padding to the maximum row length (`0` if empty).

This is convenient when you just want a rectangular tensor without precomputing `nCols`. -/
def tensor2dPadRight (α : Type := Float) [Inhabited α] (xss : List (List α)) :
    Tensor α (.dim xss.length (.dim (Spec.maxRowLength xss) .scalar)) :=
  Spec.fromList2dPadRight xss

/-! ### General N-D tensors from a flat list -/

/-!
The “real” spec constructors are shape-indexed, i.e. you usually build tensors by recursion on a
`Shape`. In example and data-loading code, though, it’s common to have:

- a runtime dims list (`List Nat`), and
- a flat list of scalars.

So we do a small amount of reshaping here.

Implementation note:
- `buildFromFlat_ofLenEq` consumes the flat list using a proof that lengths match.
- The public APIs (`tensorND` and `tensorND_ofLenEq`) are the ones that establish that proof.
-/
/--
Build a shape-indexed tensor from a flat list, given a proof that the lengths match.

This is a helper for `tensorND_ofLenEq`/`tensorND`: users typically want those APIs rather than
recursing over `Shape` directly.
-/
def buildFromFlatOfLenEq {α : Type} :
    (s : Shape) → (xs : List α) → xs.length = Shape.size s → Tensor α s
  | .scalar, xs, h => by
      have hx : xs.length = 1 := by
        have h' := h
        simp [Shape.size] at h'
        exact h'
      have h0 : 0 < xs.length := by
        simp [hx]
      exact Tensor.scalar (xs.get ⟨0, h0⟩)
  | .dim n s', xs, h => by
      let chunkSize : Nat := Shape.size s'
      have hxLen : xs.length = n * chunkSize := by
        have h' := h
        simp [Shape.size] at h'
        dsimp [chunkSize]
        exact h'
      refine Tensor.dim (fun i => ?_)
      let start : Nat := i.val * chunkSize
      let chunk : List α := (xs.drop start).take chunkSize
      have hAdd : start + chunkSize ≤ xs.length := by
        have hMul : (i.val + 1) * chunkSize ≤ n * chunkSize :=
          Nat.mul_le_mul_right chunkSize (Nat.succ_le_of_lt i.2)
        have : i.val * chunkSize + chunkSize ≤ n * chunkSize := by
          simpa [Nat.add_mul] using hMul
        simpa [hxLen, start] using this
      have hStartLe : start ≤ xs.length :=
        Nat.le_trans (Nat.le_add_right start chunkSize) hAdd
      have hChunkSub : chunkSize ≤ xs.length - start := by
        have hAdd' : chunkSize + start ≤ xs.length := by
          simpa [Nat.add_comm] using hAdd
        exact (Nat.le_sub_iff_add_le hStartLe).2 hAdd'
      have hChunkLen : chunk.length = chunkSize := by
        simp [chunk, List.length_take, List.length_drop, Nat.min_eq_left hChunkSub]
      have hChunkLen' : chunk.length = Shape.size s' := by
        simpa [chunkSize] using hChunkLen
      exact buildFromFlatOfLenEq (s := s') (xs := chunk) hChunkLen'

/-- N-D tensor from a runtime `dims` list and a flat `xs`, given a proof of matching length.

This is the “static / proof-carrying” version: if you can prove the length match, you avoid any
runtime checks and you keep a precise shape in the type.
-/
def tensorNDOfLenEq {α : Type} (dims : List Nat) (xs : List α)
    (h : xs.length = numelDims dims) : Tensor α (shapeOfDims dims) :=
  buildFromFlatOfLenEq (α := α) (s := shapeOfDims dims) (xs := xs) (by
    simpa [numelDims] using h)

/-- N-D tensor from a runtime `dims` list and a flat `xs`, with a runtime length check.

This is the “dynamic / user-friendly” version: it fails with a descriptive message if the number
of provided scalars doesn’t match the implied `numel`.
-/
def tensorND {α : Type} (dims : List Nat) (xs : List α) :
    Except String (Tensor α (shapeOfDims dims)) :=
  let expected := numelDims dims
  if h : xs.length = expected then
    .ok (tensorNDOfLenEq (α := α) (dims := dims) (xs := xs) h)
  else
    .error s!"tensorND: expected {expected} elements for dims={dims}, got {xs.length}"

/-! ### Fill/zeros/ones from runtime dims -/

/-- Fill an N-D tensor with a constant, where the shape is given as a runtime `List Nat`. -/
def fillND {α : Type} (value : α) (dims : List Nat) : Tensor α (shapeOfDims dims) :=
  Spec.fill value (shapeOfDims dims)

/-- All-zeros tensor, from a runtime `dims` list. -/
def zerosND {α : Type} [Zero α] (dims : List Nat) : Tensor α (shapeOfDims dims) :=
  fillND (α := α) 0 dims

/-- All-ones tensor, from a runtime `dims` list. -/
def onesND {α : Type} [One α] (dims : List Nat) : Tensor α (shapeOfDims dims) :=
  fillND (α := α) 1 dims

/-! ## Tactics -/

/-!
`tensorND_ofLenEq` needs a proof that your flat list has the right length.

For the common “all dimensions and lists are literals/abbrevs” case, this tactic
usually closes that goal automatically.

Usage:
  `tensorND_ofLenEq ... (by tensor_len)`
-/
macro "tensor_len" : tactic =>
  `(tactic| first
    | (simp [NN.Tensor.numelDims, NN.Tensor.shapeOfDims, Spec.Shape.size]; dsimp; decide)
    | decide
    | simp [NN.Tensor.numelDims, NN.Tensor.shapeOfDims, Spec.Shape.size])

/-!
`tensorND!` is the checked literal constructor for constants whose length proof should be solved
by `tensor_len`.

It expands to `tensorND_ofLenEq ... (by tensor_len)`, so you usually don’t have to write the proof.
If the proof can’t be solved (e.g. truly dynamic `dims`), elaboration fails; use `tensorND` in that
  case.
-/
macro "tensorND!" dims:term:max xs:term:max : term =>
  `(NN.Tensor.tensorNDOfLenEq (dims := $dims) (xs := $xs) (by tensor_len))

/--
Typed variant of `tensorND!`.

This is the same constructor as `tensorND! dims xs`, but lets you explicitly specify the element
type when numeric literals would otherwise default to an undesired type.

Example:
`def x : Tensor ℚ (shape![2, 2]) := tensorND! (ty := ℚ) [2, 2] [1, 2, 3, 4]`

Implementation note: we avoid reserving a common identifier as a syntax keyword (Lean would then
treat it as a keyword in downstream files). Instead, we parse an `ident` and sanity-check it is
`ty`.
-/
macro "tensorND!" "(" name:ident ":=" elemTy:term ")" dims:term:max xs:term:max : term => do
  if name.getId != `ty then
    Lean.Macro.throwErrorAt name "tensorND!: expected `(ty := <type>)`"
  `(NN.Tensor.tensorNDOfLenEq (α := $elemTy) (dims := $dims) (xs := $xs) (by tensor_len))

/-!
`shape!` is a compact convenience macro for examples: build a `Shape` from a bracketed dimension list.

It expands through reducible `shapeOfDims`, so it stays definitionally aligned with `tensorND!`
while still unfolding for dependent proofs and shape typeclass search.

Example:
  `def s : Shape := shape![2, 3, 4]`
-/
syntax (name := shapeBang) "shape!" "[" term,* "]" : term

macro_rules
  | `(shape![ $ds:term,* ]) =>
      `(NN.Tensor.shapeOfDims [ $ds,* ])

/-!
`tensor!` is the "nested brackets" constructor, similar to writing nested Python lists in PyTorch.

Examples:

```lean
-- shape (2, 3, 4)
def x :=
  tensor! [
    [ [0,1,2,3],   [4,5,6,7],   [8,9,10,11] ],
    [ [12,13,14,15],[16,17,18,19],[20,21,22,23] ]
  ]
```

It is fully general over rank: the nesting depth determines the rank.
Internally, it computes the dims from list lengths and flattens in row-major order
(last dimension changes fastest), then calls `tensorND!`.

If you need runtime/ragged handling, use `tensorND`/`tensorDynND` instead.
-/

open Lean

namespace TensorLit

-- Parse a nested list literal into (dims, flat-leaf-terms).
private meta partial def parseNested (t : Syntax) : MacroM (List Nat × List Syntax) := do
  match t with
  | `([$ts,*]) =>
      -- `ts` is a `TSepArray`, i.e. `#[el1, ",", el2, ",", ...]`.
      let es := ts.elemsAndSeps
      let mut elems : Array Syntax := #[]
      for i in List.finRange es.size do
        if i.val % 2 = 0 then
          elems := elems.push (es[i.val]!)
      if elems.isEmpty then
        pure ([0], [])
      else
        let parsed : List (List Nat × List Syntax) ← elems.toList.mapM parseNested
        match parsed with
        | [] => pure ([0], [])
        | (dims0, flat0) :: rest =>
            -- Refuse mixed depths like `[1, [2]]` and ragged shapes.
            for (dims, _) in rest do
              if dims != dims0 then
                Macro.throwErrorAt t "tensor!: ragged or mixed-depth nested lists are not supported"
            let flat := rest.foldl (fun acc p => acc ++ p.snd) flat0
            pure (elems.size :: dims0, flat)
  | _ =>
      -- Leaf scalar term.
      pure ([], [t])

end TensorLit

/-!
Typed variant of `tensor!`.

This is useful when the leaf literals don’t uniquely determine the element type.

Example:
```lean
def x : Tensor ℚ (shape![2, 2]) :=
  tensor! (ty := ℚ) [[1, 2], [3, 4]]
```
-/

macro "tensor!" "(" name:ident ":=" elemTy:term ")" xs:term:max : term => do
  if name.getId != `ty then
    Lean.Macro.throwErrorAt name "tensor!: expected `(ty := <type>)`"
  let (dims, flat) ← TensorLit.parseNested xs
  let dimsElems : Array (TSyntax `term) :=
    dims.toArray.map (fun n => ⟨Syntax.mkNumLit (toString n)⟩)
  let flatElems : Array (TSyntax `term) :=
    flat.toArray.map (fun stx => ⟨stx⟩)
  let dimsSep := Syntax.TSepArray.ofElems (sep := ",") dimsElems
  let flatSep := Syntax.TSepArray.ofElems (sep := ",") flatElems
  let dimsTerm ← `(term| [$(dimsSep),*])
  let flatTerm ← `(term| [$(flatSep),*])
  `(NN.Tensor.tensorNDOfLenEq (α := $elemTy) (dims := $dimsTerm) (xs := $flatTerm) (by tensor_len))

/--
Untyped variant of `tensor!`.

This is the common case: Lean infers the element type from the literal leaves (e.g. `Nat`, `Int`,
`Float`, `ℚ`, ...). If inference picks the wrong type, use the typed form
`tensor! (ty := α) ...` instead.
-/
macro "tensor!" xs:term:max : term => do
  let (dims, flat) ← TensorLit.parseNested xs
  let dimsElems : Array (TSyntax `term) :=
    dims.toArray.map (fun n => ⟨Syntax.mkNumLit (toString n)⟩)
  let flatElems : Array (TSyntax `term) :=
    flat.toArray.map (fun stx => ⟨stx⟩)
  let dimsSep := Syntax.TSepArray.ofElems (sep := ",") dimsElems
  let flatSep := Syntax.TSepArray.ofElems (sep := ",") flatElems
  let dimsTerm ← `(term| [$(dimsSep),*])
  let flatTerm ← `(term| [$(flatSep),*])
  `(NN.Tensor.tensorNDOfLenEq (dims := $dimsTerm) (xs := $flatTerm) (by tensor_len))

/-! ## Small index helpers -/

/-!
Writing `⟨0, by decide⟩` everywhere is noisy.

These macros are intended for examples/tests where the dimension is a literal or abbreviation, so
`by decide` can close the bounds proof. They are deliberately not a replacement for carrying a real
`Fin n` in library code.

Examples:
  `Tensor.vecGet v fin0!`
  `Tensor.vecGet v fin1!`
  `Tensor.vecGet v (fin! 5 3)`
-/

macro "fin0!" : term =>
  `(⟨0, by decide⟩)

/-- Convenience index `⟨1, by decide⟩` for example code where the bound proof is decidable. -/
macro "fin1!" : term =>
  `(⟨1, by decide⟩)

/-- Convenience index `⟨i, by decide⟩` packaged as `Fin n`, for examples/tests with literal bounds. -/
macro "fin!" n:term:max i:num : term =>
  `(show Fin $n from ⟨$i, by decide⟩)

/-!
`tensorF!` is a convenience macro for constants: build from **Float literals**, then cast into your
chosen runtime scalar backend using `cast : Float → α`.

Example:
  `let w : Tensor α (.dim 3 (.dim 2 .scalar)) := tensorF! cast [3, 2] [0.1, 0.2, 0.3, 0.4, 0.5,
    0.6]`

This avoids noisy lists like `[cast 0.1, cast 0.2, ...]`.
-/
macro "tensorF!" cast:term:max dims:term:max xs:term:max : term =>
  `(Spec.mapTensor $cast
      (NN.Tensor.tensorNDOfLenEq (α := Float) (dims := $dims) (xs := $xs) (by tensor_len)))

/-!
`tensor32!` is the `tensor!` macro specialized to the executable IEEE-754 binary32 backend.

It builds a `Tensor Float _` from a nested bracket literal, then casts elementwise via
`IEEE32Exec.ofFloat`.

Example:
```lean
def w : Tensor TorchLean.Floats.IEEE32Exec (shape![2, 2]) :=
  tensor32! [[0.1, 0.2], [0.3, 0.4]]
```
-/
macro "tensor32!" xs:term:max : term =>
  `(Spec.mapTensor TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat (tensor! $xs))

/-! ## Dynamic tensor wrapper (shape not in the type) -/

/-- A tensor paired with its shape, where the shape is stored as data instead of being in the type.

This is useful when:
- you need to accept arbitrary tensors at runtime (CLI / file formats / interoperability),
- but you still want to carry a `Shape` around so downstream code can inspect it.
-/
structure DynTensor (α : Type) where
  /-- Runtime shape tag carried alongside the tensor. -/
  s : Shape
  /-- Shape-indexed tensor whose type is tied to `s`. -/
  t : Tensor α s

/-- Build a `DynTensor` from runtime dims + flat data, with a runtime length check. -/
def tensorDynND {α : Type} (dims : List Nat) (xs : List α) : Except String (DynTensor α) := do
  let t ← tensorND (α := α) dims xs
  pure { s := shapeOfDims dims, t := t }

/-! ## Common dtype helpers (from Float literals) -/

/-- 1-D tensor from Float literals, cast into the executable IEEE-754 FP32 backend. -/
def tensorF321d (xs : List Float) :
    Tensor (TorchLean.Floats.F32 .ieee754Exec) (.dim xs.length .scalar) :=
  Spec.mapTensor TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat (tensor1d (α := Float) xs)

/-- 2-D tensor from Float literals, cast into the executable IEEE-754 FP32 backend. -/
def tensorF322d (xss : List (List Float)) :
    Except String (Tensor (TorchLean.Floats.F32 .ieee754Exec)
      (.dim xss.length (.dim (if xss.isEmpty then 0 else xss.head!.length) .scalar))) := do
  let t ← tensor2d (α := Float) xss
  pure (Spec.mapTensor TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat t)

/-! ## Printing -/

/-- A compact “dtype name” class used by `NN.Tensor.print`.

This is deliberately lightweight: it’s only meant for friendly IO output in examples/tests.
-/
class DTypeName (α : Type) where
  name : String

/-- Display name for `Float` tensors in `NN.Tensor.print`. -/
instance : DTypeName Float where
  name := "Float"
/-- Display name for rational tensors in `NN.Tensor.print`. -/
instance : DTypeName ℚ where
  name := "ℚ"
/-- Display name for integer tensors in `NN.Tensor.print`. -/
instance : DTypeName Int where
  name := "Int"
/-- Display name for proof-level `FP32` rounding-model tensors in `NN.Tensor.print`. -/
instance : DTypeName TorchLean.Floats.FP32 where
  name := "FP32"
/-- Display name for executable IEEE-754 FP32 tensors in `NN.Tensor.print`. -/
instance : DTypeName TorchLean.Floats.IEEE32Exec where
  name := "IEEE32Exec"
/-- Display name for proof-level real-valued tensors in `NN.Tensor.print`. -/
instance : DTypeName ℝ where
  name := "ℝ"

/-- Display name for TorchLean complex scalars in `NN.Tensor.print`. -/
instance {α : Type} [DTypeName α] : DTypeName (TorchLean.Complex α) where
  name := s!"Complex[{DTypeName.name (α := α)}]"

/-- A “pretty-printer with failure”.

We model printing as `Except String String` because some tensor element types are intentionally
non-printable (e.g. `ℝ` or proof-only floating-point models).
-/
class TensorPrintable (α : Type) where
  pretty : {s : Shape} → Tensor α s → Except String String

/-- Default printing for element types that support `ToString`. -/
instance (priority := 10) {α : Type} [ToString α] : TensorPrintable α where
  pretty := fun {_s} t => .ok (Spec.pretty t)

/-- Printing is intentionally disabled for proof-level `ℝ` tensors. -/
instance (priority := 100) : TensorPrintable ℝ where
  pretty := fun {_s} _ =>
    .error
      "Refusing to print `Tensor ℝ` (proof-level); cast to `Float`/`IEEE32Exec`/`ℚ` to display."

/-- Printing is intentionally disabled for the proof-only rounding model `FP32`. -/
instance (priority := 100) : TensorPrintable TorchLean.Floats.FP32 where
  pretty := fun {_s} _ =>
    .error
      ("Refusing to print `Tensor FP32` (proof-only rounding model); " ++
        "use `IEEE32Exec`/`Float` for runtime printing.")

/-- Print a tensor with a dtype tag, or throw an `IO.userError` if the dtype refuses to print. -/
def print {α : Type} [DTypeName α] [TensorPrintable α] {s : Shape} (t : Tensor α s) : IO Unit := do
  match (TensorPrintable.pretty (α := α) t) with
  | .ok str => IO.println s!"[{DTypeName.name (α := α)}] {str}"
  | .error msg => throw <| IO.userError msg

end NN.Tensor
