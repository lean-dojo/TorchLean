/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Tensor.Layers.Softmax
public meta import TorchLean.Tensor.Context
public meta import TorchLean.Tensor.Layers.Softmax

/-!
# Online softmax on native floats

Checks of `Softmax.onlineStep` on native `Float32` and `Float`: its running maximum at a
NaN, signed zeros and infinities, and its normalizer at ties and at an overflowing rescale. Only
the signed zeros are compared by bits. A NaN is detected with `isNaN`, and the other values are
compared with `==`.
-/

public section

namespace TorchLean.Test.OnlineSoftmax

/-- The running maximum of `onlineStep` from `start` over `scores`, each with the value one. -/
private def step (start : Float32) (scores : Array Float32) : Float32 :=
  (Softmax.onlineStep Float32.exp id id (start, 0, 0) (scores.toList.map (·, 1))).1

-- The running maximum of `onlineStep` is the IEEE `Context.maximum`, as in `reduce .max`: a NaN
-- score propagates, `+0` is above `-0`, infinities order as numbers, tied scores weigh one each,
-- and a rescale `exp (old - new)` whose difference overflows to `-inf` is `0`.
#guard (step 0 #[0 / 0, 1]).isNaN && (step (0 / 0) #[1]).isNaN && (step 1 #[2, 0 / 0]).isNaN
#guard (step 0 #[-0]).toBits == 0 && (step (-0) #[0]).toBits == 0 &&
    (step (-0) #[-0]).toBits == (-0 : Float32).toBits
#guard step (-1 / 0) #[1 / 0] == 1 / 0 && step (-1 / 0) #[-1 / 0] == -1 / 0 &&
    step (-1 / 0) #[3] == 3
#guard (Softmax.onlineStep Float32.exp id id (2, 0, 0) [(2, 1), (2, 1)]).2.1 == 2
#guard (Softmax.onlineStep Float32.exp id id (-3e38, 1, 1) [(3e38, 1)]).2.1 == 1
#guard
  let step := fun (start : Float) (scores : Array Float) =>
    (Softmax.onlineStep Float.exp id id (start, 0, 0) (scores.toList.map (·, 1))).1
  (step 0 #[0 / 0, 1]).isNaN && (step 0 #[-0]).toBits == 0 &&
    step (-1 / 0) #[1 / 0] == 1 / 0

end TorchLean.Test.OnlineSoftmax
