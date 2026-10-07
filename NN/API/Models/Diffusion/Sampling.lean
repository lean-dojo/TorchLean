/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.Diffusion

/-!
# DDIM Sampling

Reverse a finite diffusion schedule with a caller-provided epsilon predictor. The predictor receives
both the schedule index and current tensor, so image time channels and other conditioning remain
independent of the sampling algorithm.
-/

@[expose] public section

namespace TorchLean.diffusion

/--
Run deterministic DDIM updates down through timestep zero.

By default, start at the last schedule entry. Pass `start := some index` to resume from a
particular timestep. An empty schedule leaves the input unchanged and never calls the predictor.

The schedule coefficients must describe the same forward process used to train `predict`.
The endpoint convention is cumulative alpha one before the first timestep.
Every update uses the supplied reconstruction postprocessor and denominator floor.
-/
def ddim {shape : Shape} {T : Nat}
    (predict : Fin T → Tensor Float shape → IO (Tensor Float shape))
    (alphaBars : Tensor Float [T]) (initial : Tensor Float shape)
    (postprocess : Tensor Float shape → Tensor Float shape :=
      fun reconstruction => Tensor.clamp reconstruction (-1) 1)
    (denominatorFloor : Float := 1e-12) (start : Option (Fin T) := none) :
    IO (Tensor Float shape) :=
  let run (start : Fin T) : IO (Tensor Float shape) := do
    let mut sample := initial
    for offset in [0:start.val + 1] do
      let index : Fin T :=
        ⟨start.val - offset, Nat.lt_of_le_of_lt (Nat.sub_le _ _) start.isLt⟩
      let previousAlpha := if index.val = 0 then 1.0 else
        let previous : Fin T :=
          ⟨index.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le _ _) index.isLt⟩
        alphaBars[previous]
      let epsilon ← predict index sample
      sample := ddimPrev previousAlpha alphaBars[index] sample epsilon postprocess denominatorFloor
    pure sample
  match start with
  | some index => run index
  | none =>
      if positive : 0 < T then
        run ⟨T - 1, Nat.sub_lt positive (by decide)⟩
      else
        pure initial

end TorchLean.diffusion
