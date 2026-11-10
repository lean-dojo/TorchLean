/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Arithmetic
public import TorchLean.NN.RL.Boundary
public import TorchLean.NN.RL.Transition -- shake: keep

/-!
# Checked Casts for RL Numerics

Validated boundary data arrives as host `Float`s. This module casts it into a configured FloatLib
binary format, binary32 `ExecFloat.Binary 8 23` among them, and rejects every cast whose result is
not finite: a NaN or infinite input, or a finite value that overflows the format. A JSON decimal
reaches the cast rounded to binary64 first, so the cast rounds it a second time. The checked
arithmetic is in `TorchLean.NN.RL.Numerics.Returns`.

We wrote the casts by hand, and a cast that underflows passes. In binary32 `1e-50` becomes 0 and
is accepted, while `1e39` overflows to infinity and is rejected. `castTransition` casts the
observation, then the next observation, then the reward, and reports the first failure. No theorem
or test covers these casts.

## Main declarations

* `ofFloat`: the checked cast of one `Float`.
* `castTensor` and `castTransition`: the checked casts of a tensor and of a boundary
  transition.

References:
- IEEE Std 754-2019, "IEEE Standard for Floating-Point Arithmetic", for the binary formats:
  https://doi.org/10.1109/IEEESTD.2019.8766229
- IEEE Std 1788-2015, "IEEE Standard for Interval Arithmetic", for interval endpoints:
  https://doi.org/10.1109/IEEESTD.2015.7140721
- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", ACM
  Computing Surveys 23(1):5-48 (1991), for the motivation behind fail-fast finite checks:
  https://doi.org/10.1145/103162.103163
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.ExecFloat (Binary)

namespace TorchLean
namespace RL
namespace Checked

open TorchLean.Tensor
open FloatLib.Floats.Formats.BinaryInterchange

variable {format : FloatFormat} {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]

local notation "Value" => ExecFloat (Configured.Family format code plan)

/-- The degenerate interval `[0,0]` is the default interval value. -/
instance : Inhabited (Interval Value) where
  default := Binary.Interval.point 0

/-!
## Checked casts
-/

/--
The cast `TorchLean.ofFloat x`, rounded to nearest, when it is finite, otherwise an error
naming `x` and the cast. It is the second check after `RL.Boundary` validation. A JSON
decimal arrives here already rounded to binary64, so the result rounds the decimal twice: in
binary32 it can differ by one ulp from the decimal's correctly rounded value, and a decimal just
below the overflow threshold `2^128 - 2^103` can round up to it in binary64 and be rejected.
-/
def ofFloat (x : Float) : Except String Value :=
  let y : Value := TorchLean.ofFloat x
  if Binary.isFinite y = true then
    .ok y
  else
    .error s!"RL numerics: Float cast produced a non-finite value (x={x}, y={y})."

/--
The cast of a tensor of host `Float`s, entry by entry with `ofFloat`, failing at the first
entry, in row-major order, whose cast is not finite.
-/
def castTensor {s : Shape} (t : Tensor Float s) :
    Except String (Tensor Value s) :=
  Tensor.mapM ofFloat t

/--
The cast of a validated boundary transition, failing when the cast of an entry of either
observation or of the reward is not finite. The action and both episode flags are copied.
-/
def castTransition {obsShape : Shape} {nActions : Nat}
    (t : Boundary.Transition obsShape nActions) :
    Except String
      (RL.ObservedTransition
        (Tensor Value obsShape) (Fin nActions) Value) := do
  let obs ← castTensor (s := obsShape) t.observation
  let nextObs ← castTensor (s := obsShape) t.nextObservation
  let r ← ofFloat t.reward
  pure
    { observation := obs
      action := t.action
      reward := r
      nextObservation := nextObs
      terminated := t.terminated
      truncated := t.truncated }


end Checked
end RL
end TorchLean
