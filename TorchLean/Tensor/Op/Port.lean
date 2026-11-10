/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Prod.TProd
public import TorchLean.Tensor.Datatype

/-!
# Ports of registered operators

A registered operator (`TorchLean.Op`) is indexed by the list of its input ports and the list of its
output ports. A port is an element sort and a shape: a tensor of the model's scalar, a frozen
tensor of the scalar that receives no gradient, a detached tensor of the scalar read from a recorded
node without a gradient, a tensor of indices, a tensor of Booleans, or a tensor of integer words.
Index and Boolean ports let an operator take an index tensor (`gather`, `scatter`) or a mask
(`select`), or return one (`sort`, `topk`, the comparisons), which a list of scalar tensor shapes
cannot say. An integer port holds words `BitVec width` of a signed or an unsigned type, for the
operators on the bits of integers (`bitwiseAnd`, the shifts, `bitcast`). `Port.irShape` is the
shape of the integer tensor that holds an index port's tuples, one component per entry, in the
layout of the IR and the operators `encodeIndex` and `decodeIndex`. `Port.IndexTuple.ofComponents`
builds a tuple from its components, failing at one outside its bound, and
`Port.IndexTuple.components` lists them. `Port.Kind.isScalar` holds at the sorts that carry the
model's scalar: value, frozen and detached.

`frozen` is the static role of an input that never receives a contribution: a scalar attribute of
the operator (an epsilon, a slope, clamp bounds), passed as data because the operator value does
not know the scalar type, or a value the operator reads without differentiating it (the operands of
a comparison). A recorder accepts a value where a frozen input is expected. `detached` is a value
the recorder reads from a recorded node and records as a parent, which receives no contribution
(the teacher of `distill`, as PyTorch's `detach`). An input is active when its port is a value port
and its parent requires a gradient.

`Port.Carrier`, `Port.Tangent` and `Port.Grad` are built by `Port.Slot` from one family of types for
each scalar sort and one for the index, Boolean and integer sorts. Over a list of ports, `Values`,
`Tangents`, `Grads` and `Flags` are Mathlib's `List.TProd`, nested pairs ending in `Unit`, so the
values of `[p, q]` are `p.Carrier α × (q.Carrier α × Unit)` by definition. Tangents and cotangents
exist for value ports only, and a frozen, detached, index, Boolean or integer port has `Unit` there.
`Flags` holds one Boolean per input, the runtime `requiresGrad` of each parent, and `Grads` the
contributions a prepared pullback sends: `some` for an active value port, `none` for an inactive
one. `Grads.mask` keeps the contributions of the inputs whose flag is set.
-/

@[expose] public section

namespace TorchLean

/-- The element sort of a port: the model's scalar, the scalar without a gradient, the scalar of a
recorded node read without a gradient, an index tuple with the given bounds, a Boolean, or an
integer word. The sort is semantic, independent of the buffer representation that
`TorchLean.LibTorch.Dtype` names. -/
inductive Port.Kind where
  /-- The model's scalar. -/
  | value
  /-- The model's scalar held without a gradient: an attribute such as an epsilon, a slope or a
  clamp bound, or a value the operator reads without differentiating it. -/
  | frozen
  /-- The model's scalar read from a recorded node without a gradient: the node is a parent and
  receives no contribution. -/
  | detached
  /-- A tuple of indices, one below each bound, as `IndexTuple bounds` (a single index for one
  bound). -/
  | index (bounds : List Nat)
  /-- A Boolean. -/
  | bool
  /-- A word of `width` bits of an integer type, two's complement when `signed`. -/
  | integer (signed : Bool) (width : Nat)
  deriving DecidableEq, Repr, Inhabited

/-- Whether a port sort carries the model's scalar: a value, frozen or detached port. -/
def Port.Kind.isScalar : Kind → Bool
  | .value | .frozen | .detached => true
  | _ => false

/-- The elements of an index port with `bounds`: `Fin n` for one bound `n`, so a single index is a
`Fin` as `Tensor.sort` and `Tensor.embedding` use it, and otherwise one index below each bound, as
`Tensor.gather` and `Tensor.scatter` read them over several axes. -/
abbrev Port.IndexTuple : List Nat → Type
  | [n] => Fin n
  | bounds => (k : Fin bounds.length) → Fin (bounds.get k)

namespace Port.IndexTuple

/-- The index tuple below `bounds` whose `k`-th component is `component k`, or an error when a
component is negative or reaches its bound. -/
def ofComponents : (bounds : List Nat) → (Nat → Int) → Except String (Port.IndexTuple bounds)
  | [], _ => .ok fun k => k.elim0
  | [n], component =>
      if h : 0 ≤ component 0 ∧ component 0 < n then .ok ⟨(component 0).toNat, by omega⟩
      else .error s!"index {component 0} is outside [0, {n})"
  | n :: m :: rest, component =>
      if h : ∀ k : Fin (n :: m :: rest).length,
          0 ≤ component k ∧ component k < ((n :: m :: rest).get k : Int) then
        .ok fun k => ⟨(component k).toNat, by have := h k; omega⟩
      else .error s!"an index tuple has a component outside its bounds {n :: m :: rest}"

/-- The components of an index tuple, in order. -/
def components : (bounds : List Nat) → Port.IndexTuple bounds → List Nat
  | [], _ => []
  | [_], tuple => [tuple.val]
  | n :: m :: rest, tuple => List.ofFn fun k : Fin (n :: m :: rest).length => (tuple k).val

/-- No index tuple has a component below a zero bound, so building one fails at every component
function. -/
theorem ofComponents_isOk {bounds : List Nat} (component : Nat → Int) (h : 0 ∈ bounds) :
    (ofComponents bounds component).isOk = false := by
  rcases bounds with _ | ⟨n, _ | ⟨m, rest⟩⟩
  · simp at h
  · simp only [List.mem_singleton] at h
    subst h
    simp only [ofComponents]
    split
    · omega
    · rfl
  · obtain ⟨k, hk⟩ := List.mem_iff_get.mp h
    simp only [ofComponents]
    split
    · rename_i hall
      have := hall k
      rw [hk] at this
      omega
    · rfl

/-- The tuple whose components are all zero, below positive bounds. -/
def zero : (bounds : List Nat) → (∀ b ∈ bounds, 0 < b) → Port.IndexTuple bounds
  | [], _ => fun k => k.elim0
  | [n], h => ⟨0, h n (by simp)⟩
  | _ :: _ :: _, h => fun _ => ⟨0, h _ (List.get_mem _ _)⟩

end Port.IndexTuple

/-- An input or output of an operator: an element sort and a shape. -/
structure Port where
  /-- The element sort. -/
  kind : Port.Kind
  /-- The shape. -/
  shape : Shape
  deriving DecidableEq, Repr, Inhabited

namespace Port

/-- A port of scalar values of `shape`. -/
abbrev value (shape : Shape) : Port := ⟨.value, shape⟩

/-- A port of scalar values of `shape` that receives no gradient. -/
abbrev frozen (shape : Shape) : Port := ⟨.frozen, shape⟩

/-- A port of scalar values of `shape` read from a recorded node that receives no gradient. -/
abbrev detached (shape : Shape) : Port := ⟨.detached, shape⟩

/-- A port of `shape` whose entries are index tuples below `bounds`. -/
abbrev index (bounds : List Nat) (shape : Shape) : Port := ⟨.index bounds, shape⟩

/-- A port of Booleans of `shape`. -/
abbrev bool (shape : Shape) : Port := ⟨.bool, shape⟩

/-- A port of `shape` whose entries are words of `width` bits, of a signed integer type when
`signed`. -/
abbrev integer (signed : Bool) (width : Nat) (shape : Shape) : Port :=
  ⟨.integer signed width, shape⟩

/-- The shape of the IR tensor that carries a port of a registered operator: the port's shape, with
a trailing axis holding the components of an index tuple unless the port has exactly one bound.
Component `k` of the tuple at coordinate `c` is the entry at flat position
`flat(c) * bounds.length + k`. A port with no bounds has empty tuples and a trailing axis of
extent zero. -/
def irShape : Port → Shape
  | ⟨.index bounds, s⟩ => if bounds.length = 1 then s else s ++ [bounds.length]
  | p => p.shape

/-- The family of types at a port: `value s`, `frozen s` and `detached s` at a value, frozen and
detached port of shape `s`, and `data β s` at an index, Boolean or integer port of shape `s` with
entries in `β`. -/
abbrev Slot (value frozen detached : Shape → Type)
    (data : (β : Type) → [Storage β] → Shape → Type := fun β _ s => Tensor β s) : Port → Type
  | ⟨.value, s⟩ => value s
  | ⟨.frozen, s⟩ => frozen s
  | ⟨.detached, s⟩ => detached s
  | ⟨.index bounds, s⟩ => data (IndexTuple bounds) s
  | ⟨.bool, s⟩ => data Bool s
  | ⟨.integer _ width, s⟩ => data (BitVec width) s

/-- The tensors a port carries over the scalar `α`. -/
abbrev Carrier (α : Type) [Storage α] : Port → Type :=
  Slot (fun s => Tensor α s) (fun s => Tensor α s) (fun s => Tensor α s)

/-- The tangent or cotangent of a port: a tensor of the scalar for a value port, `Unit` for a
frozen, detached, index, Boolean or integer port. -/
abbrev Tangent (α : Type) [Storage α] : Port → Type :=
  Slot (fun s => Tensor α s) (fun _ => Unit) (fun _ => Unit) (fun _ _ _ => Unit)

/-- A contribution a prepared pullback sends to one input: an optional tensor for a value port,
`none` when the input needs no gradient, and `Unit` for a frozen, detached, index, Boolean or
integer port. -/
abbrev Grad (α : Type) [Storage α] : Port → Type :=
  Slot (fun s => Option (Tensor α s)) (fun _ => Unit) (fun _ => Unit) (fun _ _ _ => Unit)

/-- The contribution of an input whose dense gradient is `g`: `some g` when `active`, `none`
otherwise. -/
def select {α : Type} [Storage α] : (p : Port) → Bool → p.Tangent α → p.Grad α
  | ⟨.value, _⟩, active, g => if active then some g else none
  | ⟨.frozen, _⟩, _, _ => ()
  | ⟨.detached, _⟩, _, _ => ()
  | ⟨.index _, _⟩, _, _ => ()
  | ⟨.bool, _⟩, _, _ => ()
  | ⟨.integer _ _, _⟩, _, _ => ()

/-- The contribution `g` to an input when `active`, nothing otherwise. -/
def mask {α : Type} [Storage α] : (p : Port) → Bool → p.Grad α → p.Grad α
  | ⟨.value, _⟩, active, g => if active then g else none
  | ⟨.frozen, _⟩, _, _ => ()
  | ⟨.detached, _⟩, _, _ => ()
  | ⟨.index _, _⟩, _, _ => ()
  | ⟨.bool, _⟩, _, _ => ()
  | ⟨.integer _ _, _⟩, _, _ => ()

end Port

/-- The values of a list of ports, as nested pairs ending in `Unit`. -/
abbrev Values (α : Type) [Storage α] : List Port → Type := List.TProd (Port.Carrier α)

/-- The tangents or cotangents of a list of ports, as nested pairs ending in `Unit`. -/
abbrev Tangents (α : Type) [Storage α] : List Port → Type := List.TProd (Port.Tangent α)

/-- The contributions of a prepared pullback to a list of input ports. -/
abbrev Grads (α : Type) [Storage α] : List Port → Type := List.TProd (Port.Grad α)

/-- One flag per input port: whether that input needs a gradient. -/
abbrev Flags : List Port → Type := List.TProd fun _ => Bool

namespace Flags

/-- The flags that mark every input as needing a gradient. -/
def all : (ps : List Port) → Flags ps
  | [] => ()
  | _ :: ps => (true, all ps)

/-- The flag of the input at a given position. -/
def get : (ps : List Port) → Flags ps → Fin ps.length → Bool
  | _ :: _, (flag, _), ⟨0, _⟩ => flag
  | _ :: ps, (_, flags), ⟨k + 1, h⟩ => get ps flags ⟨k, Nat.lt_of_succ_lt_succ h⟩

end Flags

namespace Tangents

/-- The contributions of the active inputs: each dense gradient where its flag is set. -/
def select {α : Type} [Storage α] : (ps : List Port) → Flags ps → Tangents α ps → Grads α ps
  | [], _, _ => ()
  | p :: ps, (active, flags), (g, gs) => (p.select active g, select ps flags gs)

end Tangents

namespace Grads

/-- The contributions `gs` at the inputs whose flag is set, and nothing at the others. -/
def mask {α : Type} [Storage α] : (ps : List Port) → Flags ps → Grads α ps → Grads α ps
  | [], _, _ => ()
  | p :: ps, (active, flags), (g, gs) => (p.mask active g, mask ps flags gs)

end Grads

end TorchLean
