/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Port

/-!
# The arguments of a registered operator in a model

A model runs a registered operator (`TorchLean.Op`) through an interpreter's references and data
(`TorchLean.Ops.apply`), with one argument per input port. A value or detached port
takes a reference, and an index, Boolean or integer port takes data. A frozen port takes either
fixed data or a recorded reference whose value the operator reads without a gradient (`Frozen`).
`Ref` and `Data` are the interpreter's reference and data types. A list of ports takes its
arguments as nested pairs ending in `Unit`, as in `Values`.
-/

@[expose] public section

namespace TorchLean

/-- The argument of a frozen port: fixed data, or a recorded reference whose value is read without
a gradient. -/
inductive Frozen (Ref Data : Shape → Type) (s : Shape) where
  /-- Fixed data. -/
  | data (x : Data s)
  /-- A recorded reference, read without a gradient. -/
  | ref (x : Ref s)

/-- The argument of a port over the references `Ref` and the data `Data` of an interpreter: a
reference for a value or detached port, `Frozen` for a frozen port, data for an index, Boolean or
integer port. -/
abbrev PortArg (α : Type) [Storage α] (Ref : Shape → Type)
    (Data : (β : Type) → [Storage β] → Shape → Type) : Port → Type :=
  Port.Slot Ref (Frozen Ref (Data α)) Ref Data

/-- The arguments of a list of ports, as nested pairs ending in `Unit`. -/
abbrev PortArgs (α : Type) [Storage α] (Ref : Shape → Type)
    (Data : (β : Type) → [Storage β] → Shape → Type) : List Port → Type :=
  List.TProd (PortArg α Ref Data)

/-- The arguments `args` with `frozen` run on each frozen argument, in port order, and every other
argument unchanged. -/
def PortArgs.mapFrozen {m : Type → Type} [Monad m] {α : Type} [Storage α] {Ref F : Shape → Type}
    {Data : (β : Type) → [Storage β] → Shape → Type} :
    (ps : List Port) → PortArgs α Ref Data ps → ({s : Shape} → Frozen Ref (Data α) s → m (F s)) →
      m (List.TProd (Port.Slot Ref F Ref Data) ps)
  | [], (), _ => pure ()
  | ⟨.value, _⟩ :: ps, (x, xs), frozen => do pure (x, ← mapFrozen ps xs frozen)
  | ⟨.frozen, _⟩ :: ps, (x, xs), frozen => do pure (← frozen x, ← mapFrozen ps xs frozen)
  | ⟨.detached, _⟩ :: ps, (x, xs), frozen => do pure (x, ← mapFrozen ps xs frozen)
  | ⟨.index _, _⟩ :: ps, (x, xs), frozen => do pure (x, ← mapFrozen ps xs frozen)
  | ⟨.bool, _⟩ :: ps, (x, xs), frozen => do pure (x, ← mapFrozen ps xs frozen)
  | ⟨.integer _ _, _⟩ :: ps, (x, xs), frozen => do pure (x, ← mapFrozen ps xs frozen)

/-- Port arguments over tensors, read as the values of the ports. A frozen port gives its tensor
whether it holds fixed data or a reference. -/
def PortArgs.values {α : Type} [Storage α] (ps : List Port)
    (args : PortArgs α (fun s => Tensor α s) (fun β _ s => Tensor β s) ps) : Values α ps :=
  Id.run <| PortArgs.mapFrozen (m := Id) (F := fun s => Tensor α s) ps args
    fun | .data x => pure x | .ref x => pure x

end TorchLean
