/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Apply
public import TorchLean.NN.TypedGraph.GraphM
public import TorchLean.Tensor.Op.Args
public import TorchLean.Tensor.Op.Need

/-!
# Recording a registered operator in the typed graph

`GraphM.apply op inputVars` records the registered operator `op` (`TorchLean.Op`) on `inputVars` as
one typed graph node through its rule `Op.rule?`, run with the capability record `supported.caps` of
the call site's `Op.Supported` instance. It checks the attributes (`Op.check`), that the scalar has
a rule for `op`, that the rule has a forward-mode rule (`jvp?`) and that no value output follows the
first. Then it resolves the variables of the value, detached and frozen-reference arguments in port
order (`mkIdx`). The first failure is reported and no node is recorded. Frozen data, index,
Boolean and integer arguments are data of the environment `Δ`.

The node (`opNode`) evaluates the rule's forward map, keeps its value output and drops the index,
Boolean or integer outputs after it. Its JVP is the rule's `jvp?`, with `Unit` tangents at the
detached, frozen, index, Boolean and integer ports. Its VJP pulls the cotangent back through the
rule's node prepared with every input flagged, or through the alias rule's node when the rule has
an enabled alias rule whose two positions name one variable. It then adds one
`Contributions.single` per value input that receives a contribution, in the rule's `order`, left to
right from the first (`Contributions.zero` when there is none). Its `validate` is the rule's
value-domain check, which checked execution runs and recording skips. Its prepared node reads the
input values and keeps the prepared rule in a `Thunk` that only the value and the pullback force.
Checked execution therefore runs `validate` on the captured inputs before the rule computes
anything, and the value and the pullback share one prepared rule. `apply_run` states the recording,
and `apply_forward`, `apply_jvp`, `apply_vjp` and `apply_vjp_alias` the node's programs.

`GraphM.applyRandom op seed inputVars` draws the next key of the builder's stream under `seed` and
records `op` with that key (`Op.withKey?`). An operator without a key is refused before the draw.
-/

@[expose] public section

namespace TorchLean
namespace TypedGraph
namespace GraphM

open Tape
open TorchLean (Idx)
open TensorPack (getIdx)

/-- The arguments of an operator in the typed graph: a variable for a value or detached port, data
of the environment for an index, Boolean or integer port, and either for a frozen port. -/
abbrev Args (α Δ : Type) [Storage α] : List Port → Type :=
  PortArgs α Var (fun β _ s => Δ → Tensor β s)

/-- The arguments of an operator resolved against the context `shapes`: the index of each variable
and the data of the environment. -/
abbrev Resolved (α Δ : Type) [Storage α] (shapes : List Shape) : List Port → Type :=
  PortArgs α (Idx shapes) (fun β _ s => Δ → Tensor β s)

variable {α Δ : Type} [Storage α]

/-- Resolve the variables of `inputVars` against the context `Γ ++ ss`, in port order. -/
def resolve {Γ : List Shape} (ss : List Shape) :
    (ps : List Port) → Args α Δ ps → Result (Resolved α Δ (Γ ++ ss) ps)
  | [], () => pure ()
  | ⟨.value, _⟩ :: ps, (x, xs) => do pure (← mkIdx ss x, ← resolve ss ps xs)
  | ⟨.frozen, _⟩ :: ps, (.data x, xs) => do pure (.data x, ← resolve ss ps xs)
  | ⟨.frozen, _⟩ :: ps, (.ref x, xs) => do pure (.ref (← mkIdx ss x), ← resolve ss ps xs)
  | ⟨.detached, _⟩ :: ps, (x, xs) => do pure (← mkIdx ss x, ← resolve ss ps xs)
  | ⟨.index _, _⟩ :: ps, (x, xs) => do pure (x, ← resolve ss ps xs)
  | ⟨.bool, _⟩ :: ps, (x, xs) => do pure (x, ← resolve ss ps xs)
  | ⟨.integer _ _, _⟩ :: ps, (x, xs) => do pure (x, ← resolve ss ps xs)

namespace Resolved

/-- The input values: a value, detached or frozen-reference port reads its index of `lookup`, the
other ports read their data at `d`. -/
def read {shapes : List Shape} (lookup : TensorLookup α shapes) (d : Δ) :
    (ps : List Port) → Resolved α Δ shapes ps → Values α ps
  | [], () => ()
  | ⟨.value, _⟩ :: ps, (i, xs) => (lookup.read i, read lookup d ps xs)
  | ⟨.frozen, _⟩ :: ps, (.data x, xs) => (x d, read lookup d ps xs)
  | ⟨.frozen, _⟩ :: ps, (.ref i, xs) => (lookup.read i, read lookup d ps xs)
  | ⟨.detached, _⟩ :: ps, (i, xs) => (lookup.read i, read lookup d ps xs)
  | ⟨.index _, _⟩ :: ps, (x, xs) => (x d, read lookup d ps xs)
  | ⟨.bool, _⟩ :: ps, (x, xs) => (x d, read lookup d ps xs)
  | ⟨.integer _ _, _⟩ :: ps, (x, xs) => (x d, read lookup d ps xs)

/-- The input tangents: a value port reads its index of `lookup`, the other ports have `Unit`. -/
def tangents {shapes : List Shape} (lookup : TensorLookup α shapes) :
    (ps : List Port) → Resolved α Δ shapes ps → Tangents α ps
  | [], () => ()
  | ⟨.value, _⟩ :: ps, (i, xs) => (lookup.read i, tangents lookup ps xs)
  | ⟨.frozen, _⟩ :: ps, (_, xs) => ((), tangents lookup ps xs)
  | ⟨.detached, _⟩ :: ps, (_, xs) => ((), tangents lookup ps xs)
  | ⟨.index _, _⟩ :: ps, (_, xs) => ((), tangents lookup ps xs)
  | ⟨.bool, _⟩ :: ps, (_, xs) => ((), tangents lookup ps xs)
  | ⟨.integer _ _, _⟩ :: ps, (_, xs) => ((), tangents lookup ps xs)

/-- The context position of each value or detached port, `none` at the other ports, as the tape's
parent ids. -/
def ids {shapes : List Shape} : (ps : List Port) → Resolved α Δ shapes ps → List (Option Nat)
  | [], () => []
  | ⟨.value, _⟩ :: ps, (i, xs) => some i.i.val :: ids ps xs
  | ⟨.frozen, _⟩ :: ps, (_, xs) => none :: ids ps xs
  | ⟨.detached, _⟩ :: ps, (i, xs) => some i.i.val :: ids ps xs
  | ⟨.index _, _⟩ :: ps, (_, xs) => none :: ids ps xs
  | ⟨.bool, _⟩ :: ps, (_, xs) => none :: ids ps xs
  | ⟨.integer _ _, _⟩ :: ps, (_, xs) => none :: ids ps xs

/-- The contribution of each port: the gradient at the index of a value port that receives one,
`none` at the other ports. -/
def contributions [Zero α] {shapes : List Shape} :
    (ps : List Port) → Resolved α Δ shapes ps → Grads α ps → List (Option (Contributions α shapes))
  | [], (), () => []
  | ⟨.value, _⟩ :: ps, (i, xs), (g, gs) =>
      g.map (Contributions.single i) :: contributions ps xs gs
  | ⟨.frozen, _⟩ :: ps, (_, xs), (_, gs) => none :: contributions ps xs gs
  | ⟨.detached, _⟩ :: ps, (_, xs), (_, gs) => none :: contributions ps xs gs
  | ⟨.index _, _⟩ :: ps, (_, xs), (_, gs) => none :: contributions ps xs gs
  | ⟨.bool, _⟩ :: ps, (_, xs), (_, gs) => none :: contributions ps xs gs
  | ⟨.integer _ _, _⟩ :: ps, (_, xs), (_, gs) => none :: contributions ps xs gs

/-- The contributions of the input gradients `gs`, in the order `order`, added left to right from
the first, or `Contributions.zero` when no input receives one. -/
def emit [Add α] [Zero α] {shapes : List Shape} {ins : List Port}
    (order : List (Fin ins.length)) (xs : Resolved α Δ shapes ins) (gs : Grads α ins) :
    Contributions α shapes :=
  match order.filterMap fun i => ((contributions ins xs gs)[i.val]?).join with
  | [] => Contributions.zero
  | c :: cs => cs.foldl Contributions.add c

end Resolved

/-- `Unit` tangents at every port of a list without a value port, or `none` when the list has
one. -/
def unitTangents? : (ps : List Port) → Option (Tangents α ps)
  | [] => some ()
  | ⟨.value, _⟩ :: _ => none
  | ⟨.frozen, _⟩ :: ps => (((), ·)) <$> unitTangents? ps
  | ⟨.detached, _⟩ :: ps => (((), ·)) <$> unitTangents? ps
  | ⟨.index _, _⟩ :: ps => (((), ·)) <$> unitTangents? ps
  | ⟨.bool, _⟩ :: ps => (((), ·)) <$> unitTangents? ps
  | ⟨.integer _ _, _⟩ :: ps => (((), ·)) <$> unitTangents? ps

variable [Context α]

/-- The node of the rule `rule` of an operator with the forward-mode rule `jvp`, on the resolved
arguments `xs`, with `units` the cotangents of the outputs after the value output. -/
def opNode {shapes : List Shape} {ins rest : List Port} {s : Shape}
    (rule : Op.Rule α ins (.value s :: rest))
    (jvp : Values α ins → Tangents α ins → Tangents α (.value s :: rest))
    (units : Tangents α rest) (xs : Resolved α Δ shapes ins) : NodeData α Δ shapes s :=
  let ids := Resolved.ids ins xs
  { forward := fun ctx d => (rule.forward (Resolved.read (.ofPack ctx) d ins xs)).1
    jvp := fun ctx dctx d =>
      (jvp (Resolved.read (.ofPack ctx) d ins xs) (Resolved.tangents (.ofPack dctx) ins xs)).1
    vjp := fun ctx d g =>
      (Resolved.emit rule.order xs
        ((Tape.prepared rule ids (Resolved.read (.ofPack ctx) d ins xs) (Flags.all ins)).1.pullback
          (g, units))).dense ()
    validate := fun ctx d => rule.validate (Resolved.read (.ofPack ctx) d ins xs)
    prepare? := some ⟨fun lookup d =>
        let values := Resolved.read lookup d ins xs
        let node : Thunk (Op.Prepared α ins (.value s :: rest)) :=
          Thunk.mk fun _ => (Tape.prepared rule ids values (Flags.all ins)).1
        { value := fun _ => node.get.value.1
          vjp := fun g => (Resolved.emit rule.order xs (node.get.pullback (g, units))).dense ()
          validate := fun _ => rule.validate values
          compact? := some ⟨fun g => Resolved.emit rule.order xs (node.get.pullback (g, units)),
            fun _ => rfl⟩ },
      fun ctx d => by simp only [Thunk.get, Tape.prepared_value]⟩ }

/-- Record the registered operator `op` on `args` as one node and return its value output. The
first failure is reported, in this order: the attributes (`Op.check`), whether the scalar has a
rule for `op` at the capability record `supported.caps`, whether the rule has a forward-mode rule,
whether a value output follows the first, and the variables, in port order. -/
def apply {Γ : List Shape} {ins rest : List Port} {s : Shape} (op : Op ins (.value s :: rest))
    [supported : Op.Supported α op] (args : Args α Δ ins) : MWith α Δ Γ (Var s) := do
  liftM op.check
  let some rule := @Op.rule? α _ _ supported.caps _ _ op |
    throw s!"typed GraphM: {(Op.encode ⟨ins, _, op⟩).tag} has no rule for this scalar"
  let some jvp := rule.jvp? |
    throw s!"typed GraphM: {(Op.encode ⟨ins, _, op⟩).tag} has no forward-mode rule"
  let some units := unitTangents? (α := α) rest |
    throw s!"typed GraphM: {(Op.encode ⟨ins, _, op⟩).tag} has several value outputs"
  let ⟨ss, g, _, _⟩ ← get
  let xs ← liftM (resolve (Γ := Γ) ss ins args)
  push g (opNode rule jvp units xs)

/-- Record the random operator `op` with its key replaced (`Op.withKey?`) by the next key of the
builder's stream under `seed`. An operator without a key is refused before the draw. -/
def applyRandom {Γ : List Shape} {ins rest : List Port} {s : Shape}
    (op : Op ins (.value s :: rest)) [supported : Op.Supported α op] (seed : Nat)
    (args : Args α Δ ins) : MWith α Δ Γ (Var s) :=
  match h : op.withKey? with
  | none => throw s!"typed GraphM: {(Op.encode ⟨ins, _, op⟩).tag} takes no random key"
  | some withKey => do
      let key ← nextRandomKey seed
      apply (supported := ⟨supported.caps, by
          rw [Op.need_withKey h key]; exact supported.available⟩) (withKey key) args

/-- If the attributes pass `Op.check`, the scalar has a rule with a forward-mode rule, no value
output follows the first and the variables resolve, then `apply` pushes `opNode` on the state and
returns the variable at its position. -/
theorem apply_run {Γ : List Shape} {ins rest : List Port} {s : Shape}
    (op : Op ins (.value s :: rest)) [supported : Op.Supported α op] (args : Args α Δ ins)
    (st : StateWith α Δ Γ) {rule : Op.Rule α ins (.value s :: rest)}
    {jvp : Values α ins → Tangents α ins → Tangents α (.value s :: rest)}
    {units : Tangents α rest} {xs : Resolved α Δ (Γ ++ st.nodeShapes) ins}
    (hcheck : op.check = .ok ())
    (hspec : @Op.rule? α _ _ supported.caps _ _ op = some rule)
    (hjvp : rule.jvp? = some jvp)
    (hunits : unitTangents? (α := α) rest = some units)
    (hargs : resolve (Γ := Γ) st.nodeShapes ins args = .ok xs) :
    (apply (Γ := Γ) op args).run st =
      .ok (⟨Γ.length + st.nodeShapes.length⟩,
        { nodeShapes := st.nodeShapes ++ [s]
          data := .snoc st.data (opNode rule jvp units xs)
          bufferUpdates := st.bufferUpdates
          randomCounter := st.randomCounter }) := by
  obtain ⟨ss, g, buffers, counter⟩ := st
  simp only [apply, hcheck, hspec, hjvp, hunits, push, liftM, monadLift, MonadLift.monadLift,
    StateT.run, bind, StateT.bind, MonadState.get, getThe, MonadStateOf.get, StateT.get, modify,
    modifyGet, MonadStateOf.modifyGet, StateT.modifyGet, pure, StateT.pure, StateT.lift,
    Except.bind, Except.pure]
  simp only [hargs]

/-- The forward map of `opNode`: the value output of the rule's forward map on the input values. -/
theorem apply_forward {shapes : List Shape} {ins rest : List Port} {s : Shape}
    (rule : Op.Rule α ins (.value s :: rest))
    (jvp : Values α ins → Tangents α ins → Tangents α (.value s :: rest))
    (units : Tangents α rest) (xs : Resolved α Δ shapes ins) (ctx : TensorPack α shapes) (d : Δ) :
    (opNode rule jvp units xs).forward ctx d =
      (rule.forward (Resolved.read (.ofPack ctx) d ins xs)).1 :=
  rfl

/-- The JVP of `opNode`: the value output of the forward-mode rule on the input values and the
input tangents. -/
theorem apply_jvp {shapes : List Shape} {ins rest : List Port} {s : Shape}
    (rule : Op.Rule α ins (.value s :: rest))
    (jvp : Values α ins → Tangents α ins → Tangents α (.value s :: rest))
    (units : Tangents α rest) (xs : Resolved α Δ shapes ins) (ctx dctx : TensorPack α shapes)
    (d : Δ) :
    (opNode rule jvp units xs).jvp ctx dctx d =
      (jvp (Resolved.read (.ofPack ctx) d ins xs) (Resolved.tangents (.ofPack dctx) ins xs)).1 :=
  rfl

/-- The VJP of `opNode` where no alias rule applies: the rule's dense pullback of the cotangent
`(g, units)`, one contribution per value input, in the rule's order. -/
theorem apply_vjp {shapes : List Shape} {ins rest : List Port} {s : Shape}
    (rule : Op.Rule α ins (.value s :: rest))
    (jvp : Values α ins → Tangents α ins → Tangents α (.value s :: rest))
    (units : Tangents α rest) (xs : Resolved α Δ shapes ins)
    (halias : Tape.aliasOf rule (Resolved.ids ins xs) = none) (ctx : TensorPack α shapes)
    (d : Δ) (g : Tensor α s) :
    (opNode rule jvp units xs).vjp ctx d g =
      (Resolved.emit rule.order xs (Tangents.select ins (Flags.all ins)
        (rule.backward (Resolved.read (.ofPack ctx) d ins xs) (g, units)))).dense () := by
  simp only [opNode, Tape.prepared, halias]
  rw [rule.prepare_eq _ _ (g, units)]

/-- The VJP of `opNode` where the alias rule `r` applies: the pullback of the alias rule's node
prepared with every input flagged, in the rule's order. -/
theorem apply_vjp_alias {shapes : List Shape} {ins rest : List Port} {s : Shape}
    (rule : Op.Rule α ins (.value s :: rest))
    (jvp : Values α ins → Tangents α ins → Tangents α (.value s :: rest))
    (units : Tangents α rest) (xs : Resolved α Δ shapes ins)
    {r : Op.AliasRule α ins (.value s :: rest) rule.forward}
    (halias : Tape.aliasOf rule (Resolved.ids ins xs) = some r) (ctx : TensorPack α shapes)
    (d : Δ) (g : Tensor α s) :
    (opNode rule jvp units xs).vjp ctx d g =
      (Resolved.emit rule.order xs
        ((r.prepare (Resolved.read (.ofPack ctx) d ins xs) (Flags.all ins)).pullback
          (g, units))).dense () := by
  simp only [opNode, Tape.prepared, halias]

end GraphM
end TypedGraph
end TorchLean
