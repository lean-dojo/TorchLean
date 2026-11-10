/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Checkpoint.Format
public import TorchLean.Tensor.Shape

/-!
# Optimizer Checkpoint Schemas

An optimizer checkpoint is meaningful only for the parameter layout that produced it. A
`Schema` records that layout: the parameter shapes in module order, their trainability
flags and the slots that share storage. Optimizer-specific codecs check a file's schema against
the destination's before installing backend state.
-/

@[expose] public section

open TorchLean.Shape (size)

namespace TorchLean.Checkpoint

/-- Ordered parameter metadata shared by backend-owned optimizer checkpoints. -/
structure Schema where
  /-- Parameter shapes in module order. -/
  shapes : Array Shape
  /-- Whether each corresponding parameter is trainable. -/
  requiresGrad : Array Bool
  /--
  First trainable slot sharing each parameter's storage, with `none` at frozen slots.

  An absent map describes independent parameters. A present map keeps the original slot numbers:
  `[p, p, q]` has representatives `[0, 0, 2]`, so its moment entries have keys `0` and `2`. The
  trainer builds this map from storage identity.
  -/
  representatives : Option (Array (Option Nat)) := none

namespace Schema

/-- Optimizer slot for a trainable parameter, or `none` for a frozen or invalid slot. -/
def representative? (schema : Schema) (index : Nat) : Option Nat :=
  match schema.representatives with
  | none => if schema.requiresGrad[index]? == some true then some index else none
  | some representatives => (representatives[index]?).getD none

/--
Check the ordered metadata and the storage-sharing map, when there is one.

Every array has one entry per parameter. A slot has no representative exactly when it is frozen,
and a representative is trainable, has the same shape, comes at or before its aliases and
represents itself. Together these conditions exclude cycles and chains, and keep frozen slots out
of optimizer history.
-/
def isWellFormed (schema : Schema) : Bool := Id.run do
  unless schema.shapes.size == schema.requiresGrad.size do return false
  if let some representatives := schema.representatives then
    unless representatives.size == schema.shapes.size do return false
  for index in [0:schema.shapes.size] do
    match schema.representative? index with
    | none =>
        if schema.requiresGrad[index]? != some false then return false
    | some representative =>
        unless schema.requiresGrad[index]? == some true &&
            representative ≤ index &&
            schema.requiresGrad[representative]? == some true &&
            schema.shapes[representative]? == schema.shapes[index]? &&
            schema.representative? representative == some representative do
          return false
  return true

/-- Whether the slot `index` is its own representative and so owns a moment entry. -/
def isRepresentative (schema : Schema) (index : Nat) : Bool :=
  schema.representative? index == some index

/-- Number of distinct moment entries required by this layout. -/
def optimizerStateCount (schema : Schema) : Nat :=
  (List.range schema.shapes.size).countP schema.isRepresentative

/-- Whether any trainable slot shares an earlier slot's optimizer history. -/
def hasAliases (schema : Schema) : Bool :=
  (List.range schema.shapes.size).any fun index =>
    match schema.representative? index with
    | none => false
    | some representative => representative != index

/--
Write the storage-sharing map after the ordinary shape/trainability schema.

Each slot uses one unsigned 64-bit field: zero for a frozen parameter and `representative + 1`
otherwise. The preceding schema already fixes the number of fields and their parameter order.
-/
def writeRepresentatives
    (format : Format) (handle : IO.FS.Handle) (schema : Schema) :
    IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed in-memory parameter schema"
  for index in [0:schema.shapes.size] do
    let tag := match schema.representative? index with
      | none => 0
      | some representative => representative + 1
    writeNat64 format.name handle tag

/--
Compare every stored representative with the layout derived from the destination parameters.

Matching just the number of moment entries would accept different sharing patterns such as
`[p, q, p]` and `[p, q, q]`. Comparing all slots rejects that mismatch before moment allocation.
-/
def readRepresentatives
    (format : Format) (handle : IO.FS.Handle) (expected : Schema) :
    IO Unit := do
  unless expected.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed expected parameter schema"
  for index in [0:expected.shapes.size] do
    let tag ← readNat64 format.name handle
    let expectedTag := match expected.representative? index with
      | none => 0
      | some representative => representative + 1
    if tag != expectedTag then
      throw <| IO.userError s!"{format.name}: storage alias mismatch for parameter {index}"

/-- Write the ordered parameter schema for an optimizer-specific checkpoint format. -/
def write
    (format : Format) (handle : IO.FS.Handle) (schema : Schema) :
    IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed in-memory parameter schema"
  writeNat64 format.name handle schema.shapes.size
  for index in [0:schema.shapes.size] do
    let shape ← match schema.shapes[index]? with
      | some shape => pure shape
      | none => throw <| IO.userError s!"{format.name}: internal shape index error"
    let requiresGrad ← match schema.requiresGrad[index]? with
      | some flag => pure flag
      | none => throw <| IO.userError s!"{format.name}: internal mutability index error"
    let dims : List Nat := shape
    writeNat64 format.name handle dims.length
    for dim in dims do
      writeNat64 format.name handle dim
    writeNat64 format.name handle (Shape.size shape)
    writeNat64 format.name handle (if requiresGrad then 1 else 0)

/-- Read a parameter schema and reject any difference from the expected module layout. -/
def readAndCheck
    (format : Format) (handle : IO.FS.Handle) (expected : Schema) :
    IO Unit := do
  unless expected.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed expected parameter schema"
  let parameterCount ← readNat64 format.name handle
  if parameterCount != expected.shapes.size then
    throw <| IO.userError <|
      s!"{format.name}: parameter count mismatch " ++
        s!"(file={parameterCount}, expected={expected.shapes.size})"
  for index in [0:parameterCount] do
    let shape ← match expected.shapes[index]? with
      | some shape => pure shape
      | none => throw <| IO.userError s!"{format.name}: internal shape index error"
    let expectedRequiresGrad ← match expected.requiresGrad[index]? with
      | some flag => pure flag
      | none => throw <| IO.userError s!"{format.name}: internal mutability index error"
    let rank ← readNat64 format.name handle
    let expectedDims := shape |>.toArray
    if rank != expectedDims.size then
      throw <| IO.userError <|
        s!"{format.name}: rank mismatch for parameter {index} " ++
          s!"(file={rank}, expected={expectedDims.size})"
    let mut dims := Array.mkEmpty rank
    for _ in [0:rank] do
      dims := dims.push (← readNat64 format.name handle)
    if dims != expectedDims then
      throw <| IO.userError <|
        s!"{format.name}: shape mismatch for parameter {index} " ++
          s!"(file={dims}, expected={expectedDims})"
    let count ← readNat64 format.name handle
    if count != Shape.size shape then
      throw <| IO.userError <|
        s!"{format.name}: element count mismatch for parameter {index} " ++
          s!"(file={count}, expected={Shape.size shape})"
    let requiresGrad ← match ← readNat64 format.name handle with
      | 0 => pure false
      | 1 => pure true
      | flag => throw <| IO.userError s!"{format.name}: invalid `requiresGrad` flag {flag}"
    if requiresGrad != expectedRequiresGrad then
      throw <| IO.userError <|
        s!"{format.name}: `requiresGrad` mismatch for parameter {index}"

end Schema
end TorchLean.Checkpoint
