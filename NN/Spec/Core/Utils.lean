/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors

/-!
# Miscellaneous spec utilities

This file collects small helper definitions used across the spec layer.

The contents are intended to be broadly reusable: small constructors, shape-preserving maps, and
lightweight conversion/pretty-printing helpers used by examples and by higher-level APIs.

Main definitions include:

- `map_tensor` (change the scalar type of a tensor, preserving shape),
- basic constructors like `zeros`/`ones`,
- simple pretty-printing (`Spec.pretty`) for executable scalar backends,
- list ↔ tensor helpers (`from_list_1d`, `from_list_2d`).

User-facing constructors in the tensor API (for example `tensorND!`, `tensorF!`) live in
`NN/Tensor/API.lean` and are built on top of these utilities.
-/

@[expose] public section


namespace Spec

/-
Note: Most definitions in this file are kept lightweight, and we keep their
typeclass assumptions local. That way the helpers stay usable both in "executable"
backends (e.g. `Float`, `IEEE32Exec`) and in proof-level backends.
-/

/-- Map a scalar function across a tensor, changing the element type.

PyTorch analogy: `torch.Tensor.to(dtype=...)` is implemented as a scalar cast map under the hood.
We keep this explicit at the spec layer because it is a common building block. -/
def mapTensor {α β : Type} : ∀ {s : Shape}, (α → β) → Tensor α s → Tensor β s
  | .scalar, f, Tensor.scalar x => Tensor.scalar (f x)
  | .dim _ _, f, Tensor.dim g => Tensor.dim (fun i => mapTensor f (g i))

-- Tensor creation helpers

/-! ### Small constructors -/

/-- All‑zeros tensor of a given shape. -/
def zeros (α : Type) [Zero α] (s : Shape) : Tensor α s :=
  fill (0 : α) s

/-- All‑ones tensor of a given shape. -/
def ones (α : Type) [One α] (s : Shape) : Tensor α s :=
  fill (1 : α) s

/-- Fill a tensor with a constant, using the shape of an existing tensor.

PyTorch analogy: `torch.full_like(t, value)`. We implement it by structural recursion so that
the argument tensor is genuinely used (and so this stays friendly to linters). -/
def fullLike {α : Type} : ∀ {s : Shape}, α → Tensor α s → Tensor α s
  | .scalar, a, _ => Tensor.scalar a
  | .dim _ _, a, Tensor.dim f => Tensor.dim (fun i => fullLike a (f i))

/-- All‑zeros tensor with the same shape as a given tensor.

PyTorch analogy: `torch.zeros_like(t)`. -/
def zerosLike {α : Type} [Zero α] : ∀ {s : Shape}, Tensor α s → Tensor α s
  | .scalar, _ => Tensor.scalar 0
  | .dim _ _, Tensor.dim f => Tensor.dim (fun i => zerosLike (f i))

/-- All‑ones tensor with the same shape as a given tensor.

PyTorch analogy: `torch.ones_like(t)`. -/
def onesLike {α : Type} [One α] : ∀ {s : Shape}, Tensor α s → Tensor α s
  | .scalar, _ => Tensor.scalar 1
  | .dim _ _, Tensor.dim f => Tensor.dim (fun i => onesLike (f i))

/-- Zip two tensors pointwise into a tensor of pairs. -/
def zip {α : Type} {s : Shape} : Tensor α s → Tensor α s → Tensor (α × α) s
  | Tensor.scalar x, Tensor.scalar y => Tensor.scalar (x, y)
  | Tensor.dim xs, Tensor.dim ys =>
    Tensor.dim (fun i => zip (xs i) (ys i))

-- Tensor conversion utilities
/-! ### Tensor ↔ list utilities -/

/-- Convert a tensor into a flat list (row‑major by outermost dimension). -/
def toList {α : Type} : ∀ {s : Shape}, Tensor α s → List α
  | Shape.scalar, Tensor.scalar x => [x]
  | Shape.dim n _, Tensor.dim f =>
    (List.finRange n).flatMap (fun i => toList (f i))

/-- Pretty‑print a tensor using `ToString` on scalars. -/
def pretty {α : Type} [ToString α] : ∀ {s}, Tensor α s → String
  | .scalar, Tensor.scalar x => toString x
  | .dim n _, Tensor.dim f =>
      "[" ++ (String.intercalate ", " (List.finRange n |>.map (fun i => pretty (f i)))) ++ "]"

-- Advanced tensor operations
-- Use the first index of Fin n → α, given proof that n > 0
/-- Evaluate a `Fin n → α` at index 0, given a proof that `n > 0`. -/
def useFin {n : Nat} {α : Type} (values : Fin n → α) (h : 0 < n) : α :=
  let i : Fin n := ⟨0, h⟩
  values i

/-- Expand a vector into a column vector by inserting a trailing dimension of size 1. -/
def expandLastDim {α : Type} {n : Nat} (t : Tensor α (.dim n .scalar)) :
    Tensor α (.dim n (.dim 1 .scalar)) :=
  -- PyTorch analogy: `t.unsqueeze(-1)` for a 1D tensor.
  -- We spell it out at the spec layer because this "add a trailing singleton dimension"
  -- pattern shows up frequently in linear algebra helpers.
  Tensor.dim (fun i =>
    match getAtSpec t i with
    | Tensor.scalar value => singleton value)

-- List conversion functions
/-- Build a vector tensor from a list.

PyTorch analogy: `torch.tensor(xs)` producing a 1D tensor. -/
def fromList1d {α : Type} (xs : List α) : Tensor α (.dim xs.length .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (xs.get i))

/-- Build a matrix tensor from a list of rows (strict validation).

This is the "safe by default" constructor used by the user-facing API layer.
It refuses empty input and refuses ragged rows, because that usually indicates a bug at the
call site (e.g. an accidental missing column in imported weights).

PyTorch analogy: `torch.tensor(xss)` will also error if `xss` is ragged. -/
def fromList2d {α : Type} [Inhabited α] (xss : List (List α)) :
    Option (Tensor α (.dim xss.length (.dim (if xss.isEmpty then 0 else xss.head!.length) .scalar)))
      :=
  match xss with
  | [] => none
  | _ =>
    if xss.any List.isEmpty then
      none
    else
      let nCols := xss.head!.length
      if xss.all (fun row => row.length = nCols) then
        -- We already validated rectangularity, but we still use `getD` so we do not have to
        -- carry explicit length-equality proofs around in the spec.
        -- If the validation is correct, the defaults are unreachable.
        some <|
          Tensor.dim (fun i =>
            vectorTensor (fun j =>
              (xss.getD i.val []).getD j.val default))
      else
        none

/-- Compute the maximum row length of a list of rows.

This is used to build a *padded* rectangular tensor from ragged input. -/
def maxRowLength {α : Type} (xss : List (List α)) : Nat :=
  xss.foldl (fun m row => Nat.max m row.length) 0

/-- Build a matrix tensor from a list of rows by padding/truncating to `nCols`.

This is the "permissive" constructor: it never fails, but it will silently pad missing entries
with `default` (and ignore any extra entries beyond `nCols`).

This is useful when importing data that is naturally ragged, or when you intentionally want
"pad-right with zeros" semantics (common in NLP style batching).

PyTorch analogy: this is closer to a manual `pad_sequence` + `torch.tensor`, except we do it
directly as a tensor constructor at the spec layer. -/
def fromList2dPadTo {α : Type} [Inhabited α] (nCols : Nat) (xss : List (List α)) :
    Tensor α (.dim xss.length (.dim nCols .scalar)) :=
  Tensor.dim (fun i =>
    let row := xss.getD i.val []
    vectorTensor (fun j => row.getD j.val default))

/-- Build a matrix tensor from a list of rows by padding to the maximum row length.

If `xss = []`, this returns a `0 x 0` tensor. Otherwise, the number of columns is
`max_row_length xss` and shorter rows are padded with `default`. -/
def fromList2dPadRight {α : Type} [Inhabited α] (xss : List (List α)) :
    Tensor α (.dim xss.length (.dim (maxRowLength xss) .scalar)) :=
  fromList2dPadTo (nCols := maxRowLength xss) xss

end Spec
