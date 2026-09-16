/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Parameters
public import NN.Runtime.Autograd.Torch.Core.Session.State
public import NN.Runtime.Autograd.Torch.Core.OptimizerCheckpoint.Schema

/-!
# Trainer Checkpoint Schema

Parameter slots survive across forward tapes, while the tape's registration maps are cleared after
an update. Construct checkpoint metadata from the trainer's parameter pack once, using the same
storage identity comparison as grouped optimizer updates.
-/

public section

namespace Runtime.Autograd.Torch.Internal

/--
Record each trainable slot's first occurrence of the same mutable storage.

Frozen slots retain their place in the schema but never own optimizer moments. In particular, a
frozen view preceding a trainable view does not become that trainable parameter's representative.
We compare all storage cells through `ParameterStorage.same`; equal tensor values alone do not
make independently allocated parameters aliases.
-/
def checkpointParameterSchema {α : Type} [TorchLean.Storage α] {shapes : List Spec.Shape}
    (parameters : ParamList α shapes) : IO OptimizerCheckpoint.ParameterSchema := do
  let representatives ← ParamList.canonicalSlotIndices parameters
  let schema : OptimizerCheckpoint.ParameterSchema :=
    { shapes := shapes.toArray, requiresGrad := ParamList.requiresGradArray parameters,
      representatives := some representatives }
  -- Independent layouts retain the version 2 representation and its existing checkpoint bytes.
  return if schema.hasAliases then schema else { schema with representatives := none }

end Runtime.Autograd.Torch.Internal
