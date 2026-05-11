/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# Directed-rounding primitives for interval propagation

TorchLean’s IBP/CROWN code represents bounds as endpoint pairs (`lo`/`hi`) inside `Box`/`FlatBox`.
To make those bounds meaningful under different numeric semantics, we abstract the *primitive*
endpoint operations (directed rounding).

Intuition:
- For pure real/interval backends, using ordinary `+`/`*` is already enclosure-safe (because the
  scalar itself is an interval type with outward rounding).
- For finite-precision backends with discrete grids (e.g. `IEEE32Exec`), we want *directed rounding*
  primitives like `addDown/addUp` and `mulDown/mulUp` so that interval propagation encloses the
  corresponding exact real operation.

This file defines a small typeclass `BoundOps` and some helper combinators. The default instance
is conservative: it just uses the scalar’s ordinary operations (no directed rounding). Specific
backends can override it (e.g. `IEEE32Exec`).

## Integration points in the current codebase

The intended usage is:

- Keep graphs/layers scalar-polymorphic over `[Context α]`.
- When a routine *propagates bounds* (IBP/affine/CROWN), also require `[BoundOps α]` and use
  `addDown/addUp/subDown/subUp/mulDown/mulUp` at the endpoints.

Concretely:

- `NN/MLTheory/CROWN/Core.lean`
  - `AffineVec.eval_on_box`: min/max over products and accumulation use `BoundOps`.
  - `IBP.linear`: interval linear layer propagation uses `BoundOps`.
- `NN.MLTheory.CROWN.Graph`
  - `box_add`, `box_sub`, `box_mul_elem`: endpoint propagation uses `BoundOps`.

To extend coverage, the next candidates are `box_inv`, `box_sqrt`, and the various nonlinearity
propagation rules in `NN.MLTheory.CROWN.Runtime.Ops` (these may need additional directed math
primitives or an Arb-backed enclosure oracle).
-/

@[expose] public section


namespace NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-!
## `BoundOps α`

`BoundOps` is intentionally focused: it supplies directed-rounding versions of the arithmetic
primitives that appear in IBP for affine/linear layers and basic arithmetic nodes.

If you want to swap in a quantized backend, the key is to provide an instance of `BoundOps` for
your scalar type.
-/
class BoundOps (α : Type) [Context α] where
  addDown : α → α → α
  addUp   : α → α → α
  subDown : α → α → α
  subUp   : α → α → α
  mulDown : α → α → α
  mulUp   : α → α → α

/-!
Default implementation: no directed rounding (use the scalar’s own arithmetic).

This keeps existing Float/ℝ/Interval instantiations working without extra adapters.
-/
instance (priority := 10) instBoundOpsDefault (α : Type) [Context α] : BoundOps α where
  addDown := (· + ·)
  addUp   := (· + ·)
  subDown := (· - ·)
  subUp   := (· - ·)
  mulDown := (· * ·)
  mulUp   := (· * ·)

namespace BoundOps

/-! Small helpers built only from `Context.decidable_gt`. -/

@[inline] def min2 (a b : α) : α :=
  if decide (a > b) then b else a

/-- Maximum of two scalars, using only `Context.decidable_gt` via `decide (a > b)`. -/
@[inline] def max2 (a b : α) : α :=
  if decide (a > b) then a else b

end BoundOps

end NN.MLTheory.CROWN
