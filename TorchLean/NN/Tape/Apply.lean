/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.CPU
public import TorchLean.Tensor.Op.Table.Rule

/-!
# Recording a registered operator on the CPU tape

`Tape.apply` records one operator of `TorchLean.Op` on the CPU tape `Tape (TorchLean.SomeTensor α)`
through its rule `Op.rule?`. It checks the attributes and the value domain, and prepares the rule
(or the operator's alias rule, when two value positions name one node) with the parents' runtime
flags, which say which value parents need a gradient. It stores the value output, if there is one,
as one node and returns that node's id with the other outputs as tensors. The node needs a gradient
when some value-port parent does. Its backward seeds the prepared pullback and sends a contribution
to every value parent that needs a gradient, in the rule's order, so an arm can skip the products of
the others (PyTorch's `needs_input_grad`). The reverse sweeps drop a contribution to a node without
a gradient, so the sweeps are those of the node prepared with every input flagged
(`TorchLean.Tape.apply_backwardDense`).
-/
@[expose] public section

namespace TorchLean

variable {α : Type} [Storage α]

namespace Tape

/-- A parent of a recorded operator: a tape node for a value or detached port, the tensor it carries
for a frozen, index, Boolean or integer port. -/
abbrev Arg (α : Type) [Storage α] : Port → Type :=
  Port.Slot (fun _ => Nat) (fun s => Tensor α s) (fun _ => Nat)

/-- The parents of a list of ports, as nested pairs ending in `Unit`. -/
abbrev Args (α : Type) [Storage α] : List Port → Type := List.TProd (Arg α)

/-- The node id of a value or detached parent, and `none` for the other ports. -/
def Arg.id? : (p : Port) → Arg α p → Option Nat
  | ⟨.value, _⟩, id => some id
  | ⟨.frozen, _⟩, _ => none
  | ⟨.detached, _⟩, id => some id
  | ⟨.index _, _⟩, _ => none
  | ⟨.bool, _⟩, _ => none
  | ⟨.integer _ _, _⟩, _ => none

/-- The node ids of the value and detached parents, one entry per port. -/
def Args.ids : (ps : List Port) → Args α ps → List (Option Nat)
  | [], _ => []
  | p :: ps, (a, as) => Arg.id? p a :: ids ps as

/-- Read a parent's tensor: a value or detached port reads its node, checking its shape. -/
def Arg.read (t : Tape (SomeTensor α)) : (p : Port) → Arg α p → Result (p.Carrier α)
  | ⟨.value, s⟩, id => requireValue (s := s) t id
  | ⟨.frozen, _⟩, x => pure x
  | ⟨.detached, s⟩, id => requireValue (s := s) t id
  | ⟨.index _, _⟩, x => pure x
  | ⟨.bool, _⟩, x => pure x
  | ⟨.integer _ _, _⟩, x => pure x

/-- Read every parent, in port order. -/
def Args.read (t : Tape (SomeTensor α)) : (ps : List Port) → Args α ps → Result (Values α ps)
  | [], _ => pure ()
  | p :: ps, (a, as) => do
      let x ← Arg.read t p a
      pure (x, ← read t ps as)

/-- A value parent needs a gradient when its node does, and the other ports, a detached one
included, never do. -/
def Args.flags (t : Tape (SomeTensor α)) : (ps : List Port) → Args α ps → Flags ps
  | [], _ => ()
  | ⟨.value, _⟩ :: ps, (id, as) => ((t.getNode? id).any (·.requiresGrad), flags t ps as)
  | ⟨.frozen, _⟩ :: ps, (_, as) => (false, flags t ps as)
  | ⟨.detached, _⟩ :: ps, (_, as) => (false, flags t ps as)
  | ⟨.index _, _⟩ :: ps, (_, as) => (false, flags t ps as)
  | ⟨.bool, _⟩ :: ps, (_, as) => (false, flags t ps as)
  | ⟨.integer _ _, _⟩ :: ps, (_, as) => (false, flags t ps as)

/-- The contribution a prepared pullback makes to one port, as a tape contribution. -/
def Grad.contribution? : (p : Port) → Option Nat → p.Grad α → Option (Nat × SomeTensor α)
  | ⟨.value, _⟩, some id, some g => some (id, SomeTensor.ofTensor g)
  | ⟨.value, _⟩, _, _ => none
  | ⟨.frozen, _⟩, _, _ => none
  | ⟨.detached, _⟩, _, _ => none
  | ⟨.index _, _⟩, _, _ => none
  | ⟨.bool, _⟩, _, _ => none
  | ⟨.integer _ _, _⟩, _, _ => none

/-- The contributions of a prepared pullback, one entry per port. -/
def Grads.contributions : (ps : List Port) → List (Option Nat) → Grads α ps →
    List (Option (Nat × SomeTensor α))
  | [], _, _ => []
  | p :: ps, id :: ids, (g, gs) => Grad.contribution? p id g :: contributions ps ids gs
  | p :: ps, [], (g, gs) => Grad.contribution? p none g :: contributions ps [] gs

/-- The outputs of a recorded operator: the node id for its value output, the tensor for an
index, Boolean or integer output. -/
abbrev Out (α : Type) [Storage α] : Port → Type :=
  Port.Slot (fun _ => Nat) (fun s => Tensor α s) (fun s => Tensor α s)

/-- The outputs of a list of ports, as nested pairs ending in `Unit`. -/
abbrev Outs (α : Type) [Storage α] : List Port → Type := List.TProd (Out α)

/-- The number of value ports. -/
def valuePorts (ps : List Port) : Nat := ps.countP (·.kind == .value)

/-- The value output as a tape value, if there is one. -/
def Values.value? : (ps : List Port) → Values α ps → Option (SomeTensor α)
  | [], _ => none
  | ⟨.value, _⟩ :: _, (x, _) => some (SomeTensor.ofTensor x)
  | ⟨.frozen, _⟩ :: ps, (_, xs) => value? ps xs
  | ⟨.detached, _⟩ :: ps, (_, xs) => value? ps xs
  | ⟨.index _, _⟩ :: ps, (_, xs) => value? ps xs
  | ⟨.bool, _⟩ :: ps, (_, xs) => value? ps xs
  | ⟨.integer _ _, _⟩ :: ps, (_, xs) => value? ps xs

/-- The outputs, with `id` for the value output. -/
def Values.outs (id : Nat) : (ps : List Port) → Values α ps → Outs α ps
  | [], _ => ()
  | ⟨.value, _⟩ :: ps, (_, xs) => (id, outs id ps xs)
  | ⟨.frozen, _⟩ :: ps, (x, xs) => (x, outs id ps xs)
  | ⟨.detached, _⟩ :: ps, (x, xs) => (x, outs id ps xs)
  | ⟨.index _, _⟩ :: ps, (x, xs) => (x, outs id ps xs)
  | ⟨.bool, _⟩ :: ps, (x, xs) => (x, outs id ps xs)
  | ⟨.integer _ _, _⟩ :: ps, (x, xs) => (x, outs id ps xs)

/-- The output cotangents for an upstream gradient of the value output, checked against its
shape. -/
def seed (g : SomeTensor α) : (ps : List Port) → Result (Tangents α ps)
  | [] => pure ()
  | ⟨.value, s⟩ :: ps => do
      let g' ← requireGrad (τ := s) g
      pure (g', ← seed g ps)
  | ⟨.frozen, _⟩ :: ps => do pure ((), ← seed g ps)
  | ⟨.detached, _⟩ :: ps => do pure ((), ← seed g ps)
  | ⟨.index _, _⟩ :: ps => do pure ((), ← seed g ps)
  | ⟨.bool, _⟩ :: ps => do pure ((), ← seed g ps)
  | ⟨.integer _ _, _⟩ :: ps => do pure ((), ← seed g ps)

/-- The alias rule of `rule` when it is enabled and its two positions name one node. -/
def aliasOf {ins outs : List Port} (rule : Op.Rule α ins outs) (ids : List (Option Nat)) :
    Option (Op.AliasRule α ins outs rule.forward) :=
  rule.alias?.bind fun r =>
    if r.enabled && (ids[r.first.val]?).join.isSome &&
        (ids[r.first.val]?).join == (ids[r.second.val]?).join then some r else none

/-- The node prepared from `xs` with `flags` and the input position left out of the parents: the
alias rule's node and its second position when `aliasOf` finds it, the rule's node otherwise. -/
def prepared {ins outs : List Port} (rule : Op.Rule α ins outs) (ids : List (Option Nat))
    (xs : Values α ins) (flags : Flags ins) : Op.Prepared α ins outs × Option Nat :=
  match aliasOf rule ids with
  | some r => (r.prepare xs flags, some r.second.val)
  | none => (rule.prepare xs flags, none)

/-- The prepared node computes the rule's forward map, with or without the alias rule. -/
theorem prepared_value {ins outs : List Port} (rule : Op.Rule α ins outs)
    (ids : List (Option Nat)) (xs : Values α ins) (flags : Flags ins) :
    (prepared rule ids xs flags).1.value = rule.forward xs := by
  unfold prepared
  split
  · exact Op.AliasRule.prepare_value _ xs flags
  · exact rule.prepare_value xs flags

/-- The tape contributions of the input gradients `gs`, in the rule's order. -/
def emit {ins outs : List Port} (rule : Op.Rule α ins outs) (ids : List (Option Nat))
    (gs : Grads α ins) : Array (Nat × SomeTensor α) :=
  (rule.order.filterMap fun i => ((Grads.contributions ins ids gs)[i.val]?).join).toArray

/-- The node of the prepared node `p` with value `y`: it needs a gradient when some input is
flagged, its parents are the value and detached parents but the left-out position, and its backward
checks the upstream gradient and emits the prepared pullback. -/
def recordedNode {ins outs : List Port} (name : String) (rule : Op.Rule α ins outs)
    (ids : List (Option Nat)) (flags : Flags ins) (p : Op.Prepared α ins outs × Option Nat)
    (y : SomeTensor α) : Node (SomeTensor α) where
  name := some name
  value := y
  requiresGrad := (List.finRange ins.length).any fun i => Flags.get ins flags i
  parents := ((List.finRange ins.length).filterMap fun i =>
    if p.2 = some i.val then none else (ids[i.val]?).join).toArray
  backward g := do
    let gs ← seed g outs
    pure (emit rule ids (p.1.pullback gs))

/-- Append the node of the prepared node `p` when it has a value output, and return the outputs
with the appended node's id for the value output. -/
def record {ins outs : List Port} (t : Tape (SomeTensor α)) (name : String)
    (rule : Op.Rule α ins outs) (ids : List (Option Nat)) (flags : Flags ins)
    (p : Op.Prepared α ins outs × Option Nat) : Result (Tape (SomeTensor α) × Outs α outs) :=
  if valuePorts outs > 1 then throw s!"op: {name} has several value outputs" else
  match Values.value? outs p.1.value with
  | none => pure (t, Values.outs t.size outs p.1.value)
  | some y => pure ((t.addNode (recordedNode name rule ids flags p y)).1,
      Values.outs t.size outs p.1.value)

variable [Context α] [Op.Capabilities α]

/-- Record a registered operator on the CPU tape. The first failure is reported, in this order:
the attributes (`Op.check`), whether the scalar has a rule for the operator, the reads of the
parents in port order, the value domain (`validate`), then whether the operator has more than one
value output (`record`). The node is prepared with the parents' runtime flags (`Args.flags`) and
needs a gradient when some value-port parent does. -/
def apply {ins outs : List Port} (t : Tape (SomeTensor α)) (name : String) (op : Op ins outs)
    (args : Args α ins) : Result (Tape (SomeTensor α) × Outs α outs) := do
  op.check
  let some rule := op.rule? (α := α) | throw s!"op: {name} has no rule for this scalar"
  let xs ← Args.read t ins args
  rule.validate xs
  let ids := Args.ids ins args
  let flags := Args.flags t ins args
  record t name rule ids flags (prepared rule ids xs flags)

end Tape

end TorchLean
