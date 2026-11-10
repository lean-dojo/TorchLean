/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.Backend
public import TorchLean.NN.Session.Recording
public import TorchLean.Tensor.Random

/-!
# Eager Random Tensors

A session counter gives repeated seeded calls distinct keys, including calls after a tape reset.
CPU and CUDA dispatch use the same key schedule.
-/

public section

namespace TorchLean

namespace Session

/-- Take the key of the next random operation: advance the session counter once per random
operation, including operations after a tape reset. -/
def takeRandomKey {α : Type} [Storage α] (session : Session α) : IO Unit :=
  session.rngCounter.modify (· + 1)

end Session

end TorchLean
