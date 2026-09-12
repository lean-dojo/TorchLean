/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Core

/-!
# RL Core Proofs

The shared tensor horizon enforces trajectory alignment in the type. The pointwise law below states
what each reconstructed return means, rather than restating a container-size invariant.
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Core

/-- Every lambda-return is its advantage plus its baseline value at the same timestep. -/
theorem returnsFromAdvantages_getScalar {α : Type} [TorchLean.Storage α] [Add α] {n : Nat}
    (advantages values : TorchLean.Tensor α [n]) (index : Fin n) :
    (Spec.RL.returnsFromAdvantages advantages values).getScalar index =
      advantages[index] + values[index] := by
  simp [Spec.RL.returnsFromAdvantages]

end Core
end RL
end Proofs
