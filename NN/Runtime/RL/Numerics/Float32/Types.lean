/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Boundary.Core
public import NN.Runtime.RL.Core -- shake: keep

/-!
# RL Float32 Types and Boundary Casts

This module checks host `Float` values after casting to FloatLib's `Binary 8 23`.
The RL diagnostics use that scalar type and FloatLib's interval carrier directly.
The checked arithmetic combinators live with the return/GAE recurrences in
`NN.Runtime.RL.Numerics.Float32.Returns`, where the proof layer bridge lemmas unfold them.

References: IEEE 754-2019 for binary32 arithmetic, IEEE 1788-2015 for interval endpoints, and
Goldberg's floating-point survey for the practical motivation behind fail-fast finite checks.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.ExecFloat.Binary (ofModel toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace Runtime
namespace RL
namespace Numerics
namespace Float32

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Spec.RL


/-- The degenerate interval `[0,0]` is the default interval value. -/
instance : Inhabited (Interval (Binary 8 23)) where
  default := Binary.Interval.point 0

/-!
## Checked Float → configured binary32 casting
-/

/--
Cast a host `Float` to `Binary 8 23`, rejecting NaN/Inf *and* binary64→binary32 overflow.

This is intended as a “second boundary check” after `Runtime.RL.Boundary` validation.
-/
def ofFloatChecked (x : Float) : Except String (Binary 8 23) :=
  let y : Binary 8 23 :=
    ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat x)))
  if Binary.isFinite y = true then
    .ok y
  else
    .error s!"RL float32: Float→configured binary32 cast produced non-finite value (x={x}, y={y})."

/--
Cast a tensor of host `Float`s to `Binary 8 23`, rejecting the cast if any entry becomes
non-finite.
-/
def castTensorChecked {s : Shape} (t : Tensor Float s) :
    Except String (Tensor (Binary 8 23) s) :=
  let t32 : Tensor (Binary 8 23) s :=
    TorchLean.Tensor.map (fun x =>
      (ofModel (Model.cast .binary64 .binary32 (toModel (Binary.ofFloat x))) : Binary 8 23)) t
  if Boundary.tensorAll (α := Binary 8 23) (s := s)
      (fun x => Binary.isFinite x) t32 then
    .ok t32
  else
    .error "RL float32: Float→configured binary32 tensor cast produced a non-finite entry."

/--
Cast a validated boundary transition (`Float`) to `Binary 8 23`, rejecting the cast if any
scalar
becomes non-finite.
-/
def castTransitionChecked {obsShape : Shape} {nActions : Nat}
    (t : Boundary.Transition obsShape nActions) :
    Except String
      (Spec.RL.ObservedTransition
        (Tensor (Binary 8 23) obsShape) (Fin nActions) (Binary 8 23)) := do
  let obs ← castTensorChecked (s := obsShape) t.observation
  let nextObs ← castTensorChecked (s := obsShape) t.nextObservation
  let r ← ofFloatChecked t.reward
  pure
    { observation := obs
      action := t.action
      reward := r
      nextObservation := nextObs
      terminated := t.terminated
      truncated := t.truncated }


end Float32
end Numerics
end RL
end Runtime
