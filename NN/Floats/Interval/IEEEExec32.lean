/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Interval
public import FloatLib.Floats.Formats.IEEE754.Native

/-!
# Binary32 certificate intervals

TorchLean's IEEE execution certificates store binary32 endpoints. `Interval32` selects that
carrier from FloatLib's generic interval type; the arithmetic and soundness proofs are FloatLib's.
Other endpoint formats can use `FloatLib.Numerics.Interval` directly, with the corresponding
outward-rounding adapter. Binary intervals retain the model's whole-range fallback for unordered
or indeterminate endpoint results.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange

namespace TorchLean.Floats.IEEE754.IEEE32Exec

/-- FloatLib bounds with the binary32 endpoints used by IEEE execution certificates. -/
abbrev Interval32 : Type :=
  let format := ExecFloat.Binary.format 8 23
  let plan := Configured.StoragePlan.forKnownWidth format 32 rfl
  ExecFloat.Binary.Interval (format := format) (plan := plan) (code := Configured.Code plan)

namespace Interval32

export FloatLib.Floats.ExecFloat.Binary.Interval
  (ofModel toModel_ofModel ofModel_toModel mem point toModel_point whole toModel_whole
    leB toModel_hull toModel_add toModel_sub toModel_mul toModel_div
    toModel_neg toModel_inv toModel_relu toModel_abs toModel_sqrt)

-- Explicit binary32 arguments keep field notation usable at the certificate boundary.
-- The generic operations and all their proofs remain in FloatLib.
@[inherit_doc ExecFloat.Binary.Interval.toModel]
abbrev toModel (x : Interval32) : Model.Interval (ExecFloat.Binary.format 8 23) :=
  ExecFloat.Binary.Interval.toModel x

@[inherit_doc ExecFloat.Binary.Interval.Valid]
abbrev Valid (x : Interval32) : Prop := ExecFloat.Binary.Interval.Valid x

@[inherit_doc ExecFloat.Binary.Interval.ValidExtended]
abbrev ValidExtended (x : Interval32) : Prop := ExecFloat.Binary.Interval.ValidExtended x

@[inherit_doc ExecFloat.Binary.Interval.containsZero]
abbrev containsZero (x : Interval32) : Bool := ExecFloat.Binary.Interval.containsZero x

@[inherit_doc ExecFloat.Binary.Interval.hull]
abbrev hull (x y : Interval32) : Interval32 := ExecFloat.Binary.Interval.hull x y

@[inherit_doc ExecFloat.Binary.Interval.add]
abbrev add (x y : Interval32) : Interval32 := ExecFloat.Binary.Interval.add x y

@[inherit_doc ExecFloat.Binary.Interval.sub]
abbrev sub (x y : Interval32) : Interval32 := ExecFloat.Binary.Interval.sub x y

@[inherit_doc ExecFloat.Binary.Interval.mul]
abbrev mul (x y : Interval32) : Interval32 := ExecFloat.Binary.Interval.mul x y

@[inherit_doc ExecFloat.Binary.Interval.div]
abbrev div (x y : Interval32) : Interval32 := ExecFloat.Binary.Interval.div x y

@[inherit_doc ExecFloat.Binary.Interval.neg]
abbrev neg (x : Interval32) : Interval32 := ExecFloat.Binary.Interval.neg x

@[inherit_doc ExecFloat.Binary.Interval.inv]
abbrev inv (x : Interval32) : Interval32 := ExecFloat.Binary.Interval.inv x

@[inherit_doc ExecFloat.Binary.Interval.relu]
abbrev relu (x : Interval32) : Interval32 := ExecFloat.Binary.Interval.relu x

@[inherit_doc ExecFloat.Binary.Interval.abs]
abbrev abs (x : Interval32) : Interval32 := ExecFloat.Binary.Interval.abs x

@[inherit_doc ExecFloat.Binary.Interval.sqrt]
abbrev sqrt (x : Interval32) : Interval32 := ExecFloat.Binary.Interval.sqrt x

end Interval32
end TorchLean.Floats.IEEE754.IEEE32Exec
