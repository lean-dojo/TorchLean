/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Tactic.Finiteness.Attr

/-!
# Tensor Storage

TorchLean's storage boundary selects the physical tensor representation.
Arbitrary element types use ordinary `Array` storage, while types with a
specialized instance, such as `Float` and `UInt8`, use packed native buffers.

`Storage` is inferred. Ordinary users continue to write
`Tensor Float [2, 3]` or `Tensor Real [2, 3]`; generic library declarations
bind `[Storage α]` so specializing `α` also specializes its storage.
-/
