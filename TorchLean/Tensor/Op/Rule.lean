/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Port

/-!
# The executable rule of a registered operator

`Op.Rule α ins outs` is what the CPU eager tape (`Tape.apply`), the typed-graph recorder
(`GraphM`) and `Op.invoke`, which the operator-IR evaluator uses, run for one operator: the
forward map from the values of the input ports to the values of the output ports, the dense
pullback that returns a gradient for every value input, and a prepared node that a recorder
builds once per node from the inputs and the runtime flags saying which inputs need a gradient.
A CUDA session records `LibTorch.Kernel.ofOp op` instead, related to the rule only by its
`KernelContract`, which tests check and no proof covers. A custom IR node runs the checked
program its graph stores.

The eager CPU recorder `Tape.apply` runs `validate` and then prepares the node through
`Tape.prepared` (the alias rule's `prepare` when it is enabled and its two positions name one
node, `prepare` otherwise), and never `forward`: the prepared node holds the output values and a
pullback that share the intermediate values saved while computing them (the weights of
`attentionFromScores`, the output of `logSoftmax`), so they are computed once. `check`, which runs
`validate` and then `prepare`, is the entry the evaluator `Op.invoke` uses. `forward` and
`backward` are the denotation. The pullback returns only the contributions of active inputs.
`prepare_value` and `prepare_eq` state that the prepared values are the forward values and that
the pullback returns the dense pullback's gradient at each active value input and nothing at the
others (the compact refinement of the dense pullback). A frozen, detached, index, Boolean or
integer input never receives a contribution.

`order` is the order in which a recorder emits the contributions of a node, a permutation of the
input positions. It is the port order for every operator except the fused multi-head attention,
which sends its input's gradient first. The reverse sweep adds contributions in the order it
receives them, which matters when two inputs name one node.

`validate` is the value-domain check a recorder runs before recording. The attributes' types (and
the wire decoder) check what a type can say, `Op.check` checks the rest (a pooling window) before
any input is read, and a recorder checks parent ports when it reads them. `alias?` is the rule for
two inputs that are the same node, by node identity, when its scalar supports it. The recorder then
uses its prepared node, whose value is the forward value and whose pullback is the operator's
policy for one parent at two positions. Prepared with some flags, that node sends what the node
prepared with every input flagged sends, to the inputs whose flag is set (its `prepare_eq`).
`jvp?` is the forward-mode rule where one is registered. An operator without one has no forward
mode, and no zero default stands in for it.

Separate theorems state the correctness of a rule: the algebraic, analytic and convention
certificates, which the `op_cert` attribute attaches to the operator.

In PyTorch's terms (Extending PyTorch, PyTorch 2.8 docs), a prepared node is a
`torch.autograd.Function` after its forward: the values its pullback shares are what
`ctx.save_for_backward` keeps, the flags play the part of `ctx.needs_input_grad`, and a value input
whose flag is unset gets `none`, as `backward` may return `None` for an input that needs no
gradient.
-/

@[expose] public section

namespace TorchLean

/-- One node prepared from its inputs: the output values and the pullback, which share the values
saved while computing them (softmax weights, a log-softmax output). The pullback returns the
contributions of the active inputs, in port order, for an output cotangent. -/
structure Op.Prepared (α : Type) [Storage α] (ins outs : List Port) where
  /-- The output values. -/
  value : Values α outs
  /-- The contributions for an output cotangent. -/
  pullback : Tangents α outs → Grads α ins

/-- The rule for two inputs of one node that are the same parent node, for an operator whose
forward map is `forward`. The recorder applies it when the parents at `first` and `second` are one
node, by node identity (equal values in two nodes do not qualify), and `enabled` holds for the
scalar. Otherwise it keeps the two separate contributions of the rule's pullback. -/
structure Op.AliasRule (α : Type) [Storage α] (ins outs : List Port)
    (forward : Values α ins → Values α outs) where
  /-- The position, in the input list, that receives the whole contribution. -/
  first : Fin ins.length
  /-- The position that names the same node and receives nothing. -/
  second : Fin ins.length
  /-- The two positions are distinct. -/
  distinct : first ≠ second
  /-- The two positions are the same port, so one node can feed both. -/
  same : ins.get first = ins.get second
  /-- Whether the rule applies to this scalar. -/
  enabled : Bool
  /-- The node when both positions name one node, from the inputs and their flags: one
  contribution, at `first`. -/
  prepare : Values α ins → Flags ins → Op.Prepared α ins outs
  /-- The node's values are the forward values. -/
  prepare_value : ∀ xs flags, (prepare xs flags).value = forward xs := by intros; rfl
  /-- The node prepared with `flags` sends the contributions of the node prepared with every input
  flagged to the inputs whose flag is set, and nothing to the others. -/
  prepare_eq : ∀ xs flags g,
    (prepare xs flags).pullback g = Grads.mask ins flags ((prepare xs (Flags.all ins)).pullback g)

/-- The executable rule of a registered operator. -/
structure Op.Rule (α : Type) [Storage α] (ins outs : List Port) where
  /-- The forward map from the input values to the output values. -/
  forward : Values α ins → Values α outs
  /-- The dense pullback: a gradient for every value input. -/
  backward : Values α ins → Tangents α outs → Tangents α ins
  /-- The node a recorder builds from the inputs and the flags of the inputs that need a
  gradient. -/
  prepare : Values α ins → Flags ins → Op.Prepared α ins outs :=
    fun xs flags => ⟨forward xs, fun g => Tangents.select ins flags (backward xs g)⟩
  /-- The prepared node's values are the forward values. -/
  prepare_value : ∀ xs flags, (prepare xs flags).value = forward xs := by
    intros; rfl
  /-- The prepared pullback returns the dense gradient at each active value input and nothing at
  the others. -/
  prepare_eq : ∀ xs flags g,
      (prepare xs flags).pullback g = Tangents.select ins flags (backward xs g) := by
    intros; rfl
  /-- The value-domain check run before recording. -/
  validate : Values α ins → Except String Unit := fun _ => .ok ()
  /-- The order in which a recorder emits the contributions of the node. -/
  order : List (Fin ins.length) := List.finRange ins.length
  /-- The order lists every input position once. -/
  order_perm : order.Perm (List.finRange ins.length) := by exact List.Perm.refl _
  /-- The rule for two inputs that are the same node. -/
  alias? : Option (Op.AliasRule α ins outs forward) := none
  /-- The forward-mode rule, where one is registered. -/
  jvp? : Option (Values α ins → Tangents α ins → Tangents α outs) := none

namespace Op.Rule

variable {α : Type} [Storage α] {ins outs : List Port}

/-- The checked entry used by `Op.invoke`: the value-domain check, then the prepared node. -/
def check (rule : Op.Rule α ins outs) (xs : Values α ins) (flags : Flags ins) :
    Except String (Op.Prepared α ins outs) := do
  rule.validate xs
  pure (rule.prepare xs flags)

/-- The checked entry fails exactly where the value-domain check fails. -/
theorem check_isOk (rule : Op.Rule α ins outs) (xs : Values α ins) (flags : Flags ins) :
    (rule.check xs flags).isOk = (rule.validate xs).isOk := by
  unfold check
  cases rule.validate xs <;> rfl

/-- Where the value-domain check succeeds, the checked entry returns the prepared node. -/
theorem check_ok (rule : Op.Rule α ins outs) (xs : Values α ins) (flags : Flags ins)
    (h : rule.validate xs = .ok ()) : rule.check xs flags = .ok (rule.prepare xs flags) := by
  simp only [check, h]
  rfl

end Op.Rule

end TorchLean
