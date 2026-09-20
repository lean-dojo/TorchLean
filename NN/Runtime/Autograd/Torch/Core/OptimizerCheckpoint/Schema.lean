/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.CheckpointIO
public import NN.Spec.Core.Shape

/-!
# Optimizer Checkpoint Schemas

An optimizer checkpoint is meaningful only for the parameter layout that produced it. This module
stores that shared layout: parameter shapes in module order and the corresponding trainability
flags. Optimizer-specific codecs validate this schema before installing backend state.
-/

@[expose] public section

namespace Runtime.Autograd.Torch.Internal.OptimizerCheckpoint

/-- Ordered parameter metadata shared by backend-owned optimizer checkpoints. -/
structure ParameterSchema where
  /-- Parameter shapes in module order. -/
  shapes : Array Spec.Shape
  /-- Whether each corresponding parameter is trainable. -/
  requiresGrad : Array Bool
  /--
  First trainable slot sharing each parameter's storage; frozen slots contain `none`.

  An absent map describes independent parameters, as in version 2 checkpoints. An explicit map
  retains the original slot numbers: `[p, p, q]` has representatives `[0, 0, 2]`, so its moment
  entries have keys `0` and `2`. The trainer constructs this map from storage identity.
  -/
  representatives : Option (Array (Option Nat)) := none

namespace ParameterSchema

/-- Optimizer slot for a trainable parameter, or `none` for a frozen or invalid slot. -/
def representative? (schema : ParameterSchema) (index : Nat) : Option Nat :=
  match schema.representatives with
  | none => if schema.requiresGrad[index]? == some true then some index else none
  | some representatives => (representatives[index]?).getD none

/--
Check the ordered metadata and any explicit storage-sharing map.

A representative must be trainable, have the same shape, precede its aliases, and represent itself.
These checks exclude cycles and chains, and keep frozen slots out of optimizer history.
-/
def isWellFormed (schema : ParameterSchema) : Bool := Id.run do
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

/-- Number of trainable slots, including repeated uses of shared storage. -/
def trainableCount (schema : ParameterSchema) : Nat :=
  schema.requiresGrad.count true

/-- Whether this slot owns a moment entry rather than referring to another slot's history. -/
def isRepresentative (schema : ParameterSchema) (index : Nat) : Bool :=
  schema.representative? index == some index

/-- Number of distinct moment entries required by this layout. -/
def optimizerStateCount (schema : ParameterSchema) : Nat :=
  (List.range schema.shapes.size).countP schema.isRepresentative

/-- Whether any trainable slot shares an earlier slot's optimizer history. -/
def hasAliases (schema : ParameterSchema) : Bool :=
  (List.range schema.shapes.size).any fun index =>
    match schema.representative? index with
    | none => false
    | some representative => representative != index

/--
Write the version 3 sharing map after the ordinary shape/trainability schema.

Each slot uses one unsigned 64-bit field: zero for a frozen parameter and `representative + 1`
otherwise. The preceding schema already fixes the number of fields and their parameter order.
-/
def writeRepresentatives
    (format : CheckpointIO.Format) (handle : IO.FS.Handle) (schema : ParameterSchema) :
    IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed in-memory parameter schema"
  for index in [0:schema.shapes.size] do
    let tag := match schema.representative? index with
      | none => 0
      | some representative => representative + 1
    CheckpointIO.writeNat64 format.name handle tag

/--
Compare every stored representative with the layout derived from the destination parameters.

Matching just the number of moment entries would accept different sharing patterns such as
`[p, q, p]` and `[p, q, q]`. Comparing all slots rejects that mismatch before moment allocation.
-/
def readAndCheckRepresentatives
    (format : CheckpointIO.Format) (handle : IO.FS.Handle) (expected : ParameterSchema) :
    IO Unit := do
  unless expected.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed expected parameter schema"
  for index in [0:expected.shapes.size] do
    let tag ← CheckpointIO.readNat64 format.name handle
    let expectedTag := match expected.representative? index with
      | none => 0
      | some representative => representative + 1
    if tag != expectedTag then
      throw <| IO.userError s!"{format.name}: storage alias mismatch for parameter {index}"

/-- Write the ordered parameter schema for an optimizer-specific checkpoint format. -/
def write
    (format : CheckpointIO.Format) (handle : IO.FS.Handle) (schema : ParameterSchema) :
    IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed in-memory parameter schema"
  CheckpointIO.writeNat64 format.name handle schema.shapes.size
  for index in [0:schema.shapes.size] do
    let shape ← match schema.shapes[index]? with
      | some shape => pure shape
      | none => throw <| IO.userError s!"{format.name}: internal shape index error"
    let requiresGrad ← match schema.requiresGrad[index]? with
      | some flag => pure flag
      | none => throw <| IO.userError s!"{format.name}: internal mutability index error"
    let dims := Spec.Shape.toList shape
    CheckpointIO.writeNat64 format.name handle dims.length
    for dim in dims do
      CheckpointIO.writeNat64 format.name handle dim
    CheckpointIO.writeNat64 format.name handle (Spec.Shape.size shape)
    CheckpointIO.writeNat64 format.name handle (if requiresGrad then 1 else 0)

/-- Read a parameter schema and reject any difference from the expected module layout. -/
def readAndCheck
    (format : CheckpointIO.Format) (handle : IO.FS.Handle) (expected : ParameterSchema) :
    IO Unit := do
  unless expected.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed expected parameter schema"
  let parameterCount ← CheckpointIO.readNat64 format.name handle
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
    let rank ← CheckpointIO.readNat64 format.name handle
    let expectedDims := Spec.Shape.toList shape |>.toArray
    if rank != expectedDims.size then
      throw <| IO.userError <|
        s!"{format.name}: rank mismatch for parameter {index} " ++
          s!"(file={rank}, expected={expectedDims.size})"
    let mut dims := Array.mkEmpty rank
    for _ in [0:rank] do
      dims := dims.push (← CheckpointIO.readNat64 format.name handle)
    if dims != expectedDims then
      throw <| IO.userError <|
        s!"{format.name}: shape mismatch for parameter {index} " ++
          s!"(file={dims}, expected={expectedDims})"
    let count ← CheckpointIO.readNat64 format.name handle
    if count != Spec.Shape.size shape then
      throw <| IO.userError <|
        s!"{format.name}: element count mismatch for parameter {index} " ++
          s!"(file={count}, expected={Spec.Shape.size shape})"
    let requiresGrad ← match ← CheckpointIO.readNat64 format.name handle with
      | 0 => pure false
      | 1 => pure true
      | flag => throw <| IO.userError s!"{format.name}: invalid `requiresGrad` flag {flag}"
    if requiresGrad != expectedRequiresGrad then
      throw <| IO.userError <|
        s!"{format.name}: `requiresGrad` mismatch for parameter {index}"

end ParameterSchema
end Runtime.Autograd.Torch.Internal.OptimizerCheckpoint
