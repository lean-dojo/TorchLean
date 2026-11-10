/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.ScalarTrainer.ParamList
public import TorchLean.NN.Checkpoint.Schema

/-!
# Trainer Checkpoint Schema

Parameter slots outlive every forward tape, while each tape reset clears the tape's registration
maps. The trainer therefore builds its checkpoint metadata once, from its parameter pack, with the
storage identity comparison that grouped optimizer updates use.
-/

public section

namespace TorchLean

/--
The checkpoint schema of a parameter list: shapes, trainability flags and, for each trainable slot,
the first trainable slot with the same mutable storage.

Frozen slots keep their place in the schema but never own optimizer moments. A frozen view that
precedes a trainable view of the same storage is therefore not that parameter's representative.
Storage is compared on all of its cells with `Param.Cells.same`, so separately allocated
parameters with equal values are never aliases.
-/
def ParamList.checkpointSchema {α : Type} [Storage α] {shapes : List Shape}
    (parameters : ParamList α shapes) : IO Checkpoint.Schema := do
  let representatives ← ParamList.canonicalSlotIndices parameters
  pure
    { shapes := shapes.toArray
      requiresGrad := ParamList.requiresGradArray parameters
      representatives := some representatives }

end TorchLean
