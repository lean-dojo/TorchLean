/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

/-!
# Source locations

`Einops.Span` records a source location as a half-open range measured in Unicode scalar values.
A span stores a natural-number offset and length, so its endpoint cannot precede its start. The
type records no bound tying a span to a particular source string.
-/

@[expose] public section

namespace TorchLean.Einops

universe u

/-- A half-open source range, measured in Unicode scalar values. -/
structure Span where
  /-- Number of Unicode scalar values preceding the range. -/
  offset : Nat
  /-- Number of Unicode scalar values contained in the range. -/
  length : Nat
deriving Repr, DecidableEq, BEq

namespace Span

/-- The first offset after a source span. -/
def stop (span : Span) : Nat :=
  span.offset + span.length

/-- The empty span at `offset`. -/
def point (offset : Nat) : Span :=
  ⟨offset, 0⟩

/-- The half-open range from `start` to `stop`, empty at `start` when `stop < start`. -/
def between (start stop : Nat) : Span :=
  ⟨start, stop - start⟩

/--
The range from `left.offset` to `right.stop`.

It is the smallest span containing both arguments when `left.offset ≤ right.offset`
and `left.stop ≤ right.stop`, as for the ordered delimiters the parsers supply.
-/
def join (left right : Span) : Span :=
  between left.offset right.stop

/-- A span built from an offset and a length ends at their sum. -/
@[simp] theorem stop_mk (offset length : Nat) :
    (Span.mk offset length).stop = offset + length := rfl

/-- A point span ends at its starting offset. -/
@[simp] theorem stop_point (offset : Nat) : (point offset).stop = offset := by
  simp [point, stop]

/-- The start of a span never lies after its endpoint. -/
theorem offset_le_stop (span : Span) : span.offset ≤ span.stop := by
  simp [stop]

end Span

/-- A value together with its location in source text. -/
structure Located (α : Type u) where
  /-- Parsed value. -/
  value : α
  /-- Source range of the parsed value. -/
  span : Span
deriving Repr, DecidableEq

end TorchLean.Einops
