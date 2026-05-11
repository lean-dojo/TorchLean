/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Runtime.RL.Boundary.Core
public import NN.Spec.Models.CommonHelpers
public import NN.Floats.Interval.IEEEExec32

/-!
# RL Float32 Types and Boundary Casts

This module is the foundation for TorchLean's explicit binary32 RL diagnostics. It defines the
`IEEE32Exec` aliases used by the runtime and checks host `Float` values after casting to binary32.
The checked arithmetic combinators live with the return/GAE recurrences in
`NN.Runtime.RL.Numerics.Float32.Returns`, where the proof-side bridge lemmas unfold them.

References: IEEE 754-2019 for binary32 arithmetic, IEEE 1788-2015 for interval endpoints, and
Goldberg's floating-point survey for the practical motivation behind fail-fast finite checks.
-/

@[expose] public section

namespace Runtime
namespace RL
namespace Numerics
namespace Float32

open Spec
open Tensor
open Spec.RL

open TorchLean.Floats
open TorchLean.Floats.IEEE754

/-- Executable IEEE-754 binary32 model used for explicit float32 semantics. -/
abbrev Float32Exec : Type := TorchLean.Floats.IEEE754.IEEE32Exec

/-- Outward-rounded interval type built on `IEEE32Exec` endpoints. -/
abbrev Interval32 : Type := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32

/--
Default inhabitant for `Interval32`.

`Array.get!` requires an `Inhabited` default; we use the degenerate interval `[0,0]`.
-/
instance : Inhabited Interval32 where
  default := TorchLean.Floats.IEEE754.IEEE32Exec.Interval32.point 0

/-!
## Checked Float → IEEE32Exec casting
-/

/--
Cast a host `Float` to `IEEE32Exec`, rejecting NaN/Inf *and* binary64→binary32 overflow.

This is intended as a “second boundary check” after `Runtime.RL.Boundary` validation.
-/
def ofFloatIEEE32ExecChecked (x : Float) : Except String Float32Exec :=
  let y : Float32Exec := TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat x
  if TorchLean.Floats.IEEE754.IEEE32Exec.isFinite y = true then
    .ok y
  else
    .error s!"RL float32: Float→IEEE32Exec cast produced non-finite value (x={x}, y={y})."

/--
Cast a tensor of host `Float`s to `IEEE32Exec`, rejecting the cast if any entry becomes
non-finite.
-/
def castTensorIEEE32ExecChecked {s : Shape} (t : Tensor Float s) :
    Except String (Tensor Float32Exec s) :=
  let t32 : Tensor Float32Exec s := Spec.mapTensor (TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat) t
  if Boundary.tensorAll (α := Float32Exec) (s := s) (fun x => TorchLean.Floats.IEEE754.IEEE32Exec.isFinite x) t32 then
    .ok t32
  else
    .error "RL float32: Float→IEEE32Exec tensor cast produced a non-finite entry."

/--
Cast a validated boundary transition (`Float`) to `IEEE32Exec`, rejecting the cast if any scalar
becomes non-finite.
-/
def castTransitionIEEE32ExecChecked {obsShape : Shape} {nActions : Nat}
    (t : Boundary.Transition obsShape nActions) :
    Except String (Spec.RL.ObservedTransition (Tensor Float32Exec obsShape) (Fin nActions) Float32Exec) := do
  let obs ← castTensorIEEE32ExecChecked (s := obsShape) t.observation
  let nextObs ← castTensorIEEE32ExecChecked (s := obsShape) t.nextObservation
  let r ← ofFloatIEEE32ExecChecked t.reward
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
