/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Literal
public import TorchLean.Tensor.Reduce.Fold
public import TorchLean.Tensor.Tactic.Init

/-!
# Vocabulary Projection

`VocabularyProjection capacity` numbers the token ids of a stream with local ids below `capacity`,
in first-occurrence order. Local id zero is reserved for an unknown token, and every token without
a local id maps to it, including the tokens that arrive after the vocabulary is full. `fromTokens`
builds a projection from a token tensor, and `encode` and `decode` map whole streams or batches
while keeping their shape. The lookup from original to local ids is a hash map stored next to the
tensor of originals.
-/

@[expose] public section

namespace TorchLean.Text

/-- Original token ids and reverse lookup for an assigned prefix of a fixed-capacity vocabulary. -/
structure VocabularyProjection (capacity : Nat) where
  /-- Original token ids in local-id order. Only the first `size` entries are assigned. -/
  originals : Tensor Nat [capacity]
  /-- Number of assigned local ids, including the reserved unknown-token slot. -/
  size : Nat
  /-- The assigned prefix fits in the storage tensor. -/
  size_le_capacity : size ≤ capacity
  /-- Lookup from original token ids to their assigned local ids. -/
  toLocalMap : Std.HashMap Nat Nat
  /-- Original token id returned when decoding an unassigned local id. -/
  unknownToken : Nat

namespace VocabularyProjection

/-- Map an original token id to its assigned local id, defaulting to the unknown-token slot. -/
def toLocal {capacity : Nat} (vocab : VocabularyProjection capacity) (id : Nat) : Nat :=
  (vocab.toLocalMap[id]?).getD 0

/-- Map a local id back to its original token id, or to `unknownToken` when the id is at least
`size`. -/
def toOriginal {capacity : Nat} (vocab : VocabularyProjection capacity) (localId : Nat) : Nat :=
  if h : localId < vocab.size then
    vocab.originals[(⟨localId, Nat.lt_of_lt_of_le h vocab.size_le_capacity⟩ : Fin capacity)]
  else vocab.unknownToken

/-- Start a vocabulary with one reserved unknown token. -/
def singleton (capacity unknownToken : Nat) [NeZero capacity] : VocabularyProjection capacity :=
  { originals := Tensor.full [capacity] unknownToken
    size := 1
    size_le_capacity := Nat.one_le_iff_ne_zero.mpr (NeZero.ne capacity)
    toLocalMap := (Std.HashMap.emptyWithCapacity).insert unknownToken 0
    unknownToken }

/-- Give each unseen token the next free local id, in row-major order, until all `capacity` ids
are taken. -/
def extend {capacity : Nat} {shape : Shape} (vocab : VocabularyProjection capacity)
    (tokens : Tensor Nat shape) : VocabularyProjection capacity :=
  tokens.foldl (fun current token =>
    if current.toLocalMap.contains token then current
    else if h : current.size < capacity then
      { originals :=
          Tensor.set current.originals ((⟨current.size, h⟩ : Fin capacity), PUnit.unit) token
        size := current.size + 1
        size_le_capacity := h
        toLocalMap := current.toLocalMap.insert token current.size
        unknownToken := current.unknownToken }
    else current) vocab

/-- Build a vocabulary from a token tensor, reserving local id zero for `unknownToken`. -/
def fromTokens (capacity unknownToken : Nat) [NeZero capacity] {shape : Shape}
    (tokens : Tensor Nat shape) : VocabularyProjection capacity :=
  (singleton capacity unknownToken).extend tokens

/-- Project every token while retaining the shape of the original stream or batch. -/
def encode {capacity : Nat} {shape : Shape} (vocab : VocabularyProjection capacity)
    (tokens : Tensor Nat shape) : Tensor Nat shape :=
  tokens.map vocab.toLocal

/-- Decode every local token while retaining the stream or batch shape. -/
def decode {capacity : Nat} {shape : Shape} (vocab : VocabularyProjection capacity)
    (tokens : Tensor Nat shape) : Tensor Nat shape :=
  tokens.map vocab.toOriginal

end VocabularyProjection
end TorchLean.Text
