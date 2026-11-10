/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Eval

/-!
# Building graphs of registered operators

`Graph.Builder` collects the nodes of an `IR.Graph` in evaluation order, and a node's id is
its position. `input`, `const` and `op` append a graph input, a constant (whose value is the
payload's constant at the node's id) and a registered operator on a list of references. Each
returns the reference of every declared output of the appended node, so later nodes can read every
output of an operator with several outputs. `constant` appends a constant node and records its
value in a payload, so a graph and its payload are built together.

Appending checks nothing. `build` returns the graph with its outputs only after the structural
check `Graph.check` accepts it, and `check_build` states that a built graph passes the check.
`Graph.mlp` builds a graph and its payload from a stack of linear layers, with an activation after
every layer but the last, and `Graph.check_mlp` states that it passes the check.
Nothing here checks the payload, the input numbering, or the runtime element types and buffer
representations of the ports.
-/

@[expose] public section

namespace TorchLean.IR

/-- The nodes of a graph of registered operators under construction, in evaluation order. -/
structure Graph.Builder where
  /-- The nodes so far. A node's id is its position. -/
  nodes : Array Node := #[]

namespace Graph.Builder

/-- Append node `n` and return the reference of each of its outputs, in order. -/
def add (b : Builder) (n : Node) : Builder × List Ref :=
  (⟨b.nodes.push n⟩, (List.range n.outs.length).map fun k => ⟨b.nodes.size, k⟩)

/-- Append graph input `index`, read at port `p`, and return its reference. -/
def input (b : Builder) (index : Nat) (p : Port) : Builder × Ref :=
  (⟨b.nodes.push (.input index p)⟩, ⟨b.nodes.size, 0⟩)

/-- Append a constant node at port `p`, whose value is the payload's constant at the node's id, and
return its reference. -/
def const (b : Builder) (p : Port) : Builder × Ref :=
  (⟨b.nodes.push (.const p)⟩, ⟨b.nodes.size, 0⟩)

/-- Append a constant node holding `v` and return `P` extended by `v` at the node's id, with the
node's reference. -/
def constant {α : Type} [Storage α] (b : Builder) (P : Payload α) (v : Op.Value α) :
    Builder × Payload α × Ref :=
  (⟨b.nodes.push (.const v.1)⟩,
    { P with const? := fun m => if m = b.nodes.size then some v else P.const? m },
    ⟨b.nodes.size, 0⟩)

/-- Append operator `o` on the references `refs`, unchecked, and return the reference of each of its
outputs, in order. -/
def op {ins outs : List Port} (b : Builder) (o : Op ins outs) (refs : List Ref) :
    Builder × List Ref :=
  b.add (.op ⟨ins, outs, o⟩ refs.toArray)

/-- The graph of the nodes with the outputs `outputs`, if the structural check `Graph.check`
accepts it, and the check's first error otherwise. -/
def build (b : Builder) (outputs : Array Ref) : Except String Graph := do
  let g : Graph := ⟨b.nodes, outputs⟩
  g.check
  pure g

/-- A built graph passes the structural check. -/
theorem check_build {b : Builder} {outputs : Array Ref} {g : Graph}
    (h : b.build outputs = .ok g) : g.check = .ok () := by
  unfold build at h
  cases hc : (⟨b.nodes, outputs⟩ : Graph).check with
  | error e => simp [hc, bind, Except.bind] at h
  | ok u =>
      simp only [hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      subst h
      cases u
      exact hc

/-- A built graph has the builder's nodes and the given outputs. -/
theorem build_eq {b : Builder} {outputs : Array Ref} {g : Graph}
    (h : b.build outputs = .ok g) : g = ⟨b.nodes, outputs⟩ := by
  unfold build at h
  cases hc : (⟨b.nodes, outputs⟩ : Graph).check with
  | error e => simp [hc, bind, Except.bind] at h
  | ok u =>
      simp only [hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
      exact h.symm

/-- A builder builds the graph of its nodes with the outputs `outputs` when that graph is well
formed. -/
theorem build_ok {b : Builder} {outputs : Array Ref} {g : Graph}
    (hg : (⟨b.nodes, outputs⟩ : Graph) = g) (h : g.WellFormed) : b.build outputs = .ok g := by
  subst hg
  exact bind_eq_ok.2 ⟨(), Graph.check_ok_iff.2 h, rfl⟩

/-- The payload `constant` returns holds `v` at the appended node's id and agrees with `P` at every
other id. -/
theorem constant_const? {α : Type} [Storage α] (b : Builder) (P : Payload α)
    (v : Op.Value α) (m : Nat) :
    (b.constant P v).2.1.const? m = if m = b.nodes.size then some v else P.const? m :=
  rfl

/-- The references `op` returns name the outputs of the appended node, in order. -/
theorem op_refs {ins outs : List Port} (b : Builder) (o : Op ins outs)
    (refs : List Ref) :
    (b.op o refs).2 = (List.range outs.length).map fun k => ⟨b.nodes.size, k⟩ :=
  rfl

end Graph.Builder

/-- The graph of the linear layers `layers`, in order, with the operator `activation n` on the `n`
outputs of every layer but the last, and its payload: graph input `0` of the first layer's input
width, then per layer its weight and bias as constants and `Op.linear`. The graph's one output is
the last layer's. An error if `layers` is empty or `Graph.check` rejects the graph
(`Graph.check_mlp`), e.g. when a layer's input width is not the previous layer's output width. -/
def Graph.mlp {α : Type} [Storage α] (activation : Nat → Operator)
    (layers : Array (SomeLinear α)) : Except String (Graph × Payload α) := do
  let some first := layers[0]?
    | throw "expected at least one layer"
  let (b, x) := ({} : Graph.Builder).input 0 (.value [first.inDim])
  let (b, P, y) := layers.zipIdx.foldl (init := (b, ({} : Payload α), x))
    fun (b, P, x) (l, i) =>
      let (b, P, w) := b.constant P ⟨.value [l.outDim, l.inDim], l.spec.weights⟩
      let (b, P, c) := b.constant P ⟨.value [l.outDim], l.spec.bias⟩
      let (b, ys) := b.op
        (Op.linear (lead := []) (inDim := l.inDim) (outDim := l.outDim)) [w, c, x]
      let y := ys.headD x
      if i + 1 < layers.size then
        let (b, zs) := b.add (.op (activation l.outDim) #[y])
        (b, P, zs.headD y)
      else (b, P, y)
  pure (← b.build #[y], P)

/-- A graph `Graph.mlp` returns passes the structural check. -/
theorem Graph.check_mlp {α : Type} [Storage α] {activation : Nat → Operator}
    {layers : Array (SomeLinear α)} {g : Graph} {P : Payload α}
    (h : Graph.mlp activation layers = .ok (g, P)) : g.check = .ok () := by
  unfold Graph.mlp at h
  split at h
  · dsimp only at h
    obtain ⟨g', hb, hg⟩ := bind_eq_ok.1 h
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hg
    exact hg.1 ▸ Graph.Builder.check_build hb
  · simp [throw, throwThe, MonadExceptOf.throw] at h

end TorchLean.IR
