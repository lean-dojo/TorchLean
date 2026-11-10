/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ShapeErasure
public import TorchLean.NN.IR.Builder
public import TorchLean.NN.IR.Imported
public import TorchLean.NN.IR.Attr
public import TorchLean.Tensor.Op.Invoke

/-!
# Typed graphs of registered operators

`OpGraph K σ ps` is a straight-line graph of registered operators with one input of shape `σ`, and
`ps` lists the port of every value, the input first. Each later value is an output of a step on
earlier values, or a constant `c : K p` at any port `p`, interpreted by a valuation `val p c`.
A step (`IR.OpGraph.Step.op o args h`) applies a registered operator `o : Op ins outs` and appends
every output of `o`. Its arguments (`OpGraph.Args ps ins`) name, for each input port, an earlier
value at a port the input accepts (`Port.accepts`: the same port, or a value where a frozen or
detached input is expected). The proof `h` shows that `o` is admissible (`Op.Admissible`): its
attribute check passes, and its rule checks no input value, reads no optional capability and
exists at every scalar. `IR.OpGraph.eval` cannot fail. It returns the input, then the values of each
later node, where a step gives its rule's `forward` values on the values its arguments name, read
at its input ports.

`OpGraph.toOp val g outs` is the typed graph as an `IR.Imported`: a graph of registered
operators (`IR.Graph`) built node for node through `IR.Graph.Builder`, with graph
input `0` at the input's port, an operator node per step on the references of its arguments, a
constant node per constant with `val p c` in the payload, and the reference of every value. Its
outputs are the references of `outs`. `OpGraph.eval_toOp` proves that the graph returns the typed
values at `outs` for every input, and `OpGraph.eval_ok_of_admissible` that every node runs.
Neither theorem has a hypothesis.
-/

@[expose] public section

namespace TorchLean

/-- The operator has a rule at every scalar. -/
abbrev Op.HasRuleEverywhere {ins outs : List Port} (o : Op ins outs) : Prop :=
  ∀ (β : Type) [Storage β] [Context β] [Op.Capabilities β], (o.rule? (α := β)).isSome

/-- An operator a typed graph holds: it passes its attribute check, its rule checks no input value
and reads no optional capability, and it has a rule at every scalar. -/
structure Op.Admissible {ins outs : List Port} (o : Op ins outs) : Prop where
  /-- The attribute check passes. -/
  check_eq_ok : o.check = .ok ()
  /-- The rule does not check its input values. -/
  checksValues_eq_false : o.checksValues = false
  /-- The rule reads no optional capability. -/
  needsCapability_eq_false : o.needsCapability = false
  /-- The operator has a rule at every scalar. -/
  hasRuleEverywhere : o.HasRuleEverywhere

/-- The values of `ps ++ qs` from the values of `ps` and of `qs`. -/
def Values.append {α : Type} [Storage α] :
    {ps qs : List Port} → Values α ps → Values α qs → Values α (ps ++ qs)
  | [], _, (), ys => ys
  | _ :: _, _, (x, xs), ys => (x, Values.append xs ys)

namespace IR.OpGraph

/-- The positions of the values that the input ports `ins` read among values of the ports `ps`,
each a value its input port accepts. -/
abbrev Args (ps : List Port) : List Port → Type :=
  List.TProd fun p => {j : Fin ps.length // p.accepts (ps.get j) = true}

/-- A node appended after values of the ports `ps`: a registered operator on the values its
arguments name, whose outputs are appended after the values of `ps`. -/
inductive Step (ps : List Port) : List Port → Type where
  /-- Operator `o` on the values at `args`. -/
  | op {ins outs : List Port} (o : Op ins outs) (args : Args ps ins) (h : o.Admissible) :
      Step ps outs

end IR.OpGraph

/-- A typed straight-line graph of registered operators with one input of shape `σ`, where `ps`
lists the port of every value, the input first. Each later value is an output of a step on earlier
values or a constant at any port, with data in the family `K`. -/
inductive IR.OpGraph (K : Port → Type) (σ : Shape) : List Port → Type where
  /-- The graph input. -/
  | input : OpGraph K σ [.value σ]
  /-- A step on earlier values, whose outputs are appended to them. -/
  | push {ps outs : List Port} (g : OpGraph K σ ps) (step : OpGraph.Step ps outs) :
      OpGraph K σ (ps ++ outs)
  /-- A constant at port `p` with data `c`. -/
  | const {ps : List Port} (g : OpGraph K σ ps) (p : Port) (c : K p) : OpGraph K σ (ps ++ [p])

namespace IR.OpGraph

open IR (Ref)

variable {α : Type} [Storage α]

/-- Values that an input port accepts carry the input port's tensors. -/
theorem _root_.TorchLean.Port.carrier_eq_of_accepts {p q : Port} (h : p.accepts q = true) :
    q.Carrier α = p.Carrier α := by
  rcases p with ⟨pd, ps⟩
  rcases q with ⟨qd, qs⟩
  by_cases hq : (⟨qd, qs⟩ : Port) = ⟨pd, ps⟩
  · rw [hq]
  · simp only [Port.accepts, beq_iff_eq, hq, false_or, Bool.and_eq_true, Bool.or_eq_true] at h
    obtain ⟨⟨rfl, hpd⟩, rfl⟩ := h
    rcases hpd with rfl | rfl <;> rfl

/-- Reading a value at a port that accepts it returns its tensor. -/
theorem _root_.TorchLean.Port.read_of_accepts {p q : Port} (h : p.accepts q = true)
    (x : q.Carrier α) :
    p.read ⟨q, x⟩ = .ok (cast (Port.carrier_eq_of_accepts h) x) := by
  rcases p with ⟨pd, ps⟩
  rcases q with ⟨qd, qs⟩
  by_cases hq : (⟨qd, qs⟩ : Port) = ⟨pd, ps⟩
  · cases hq
    cases pd <;> simp [Port.read, pure, Except.pure]
  · have h' := h
    simp only [Port.accepts, beq_iff_eq, hq, false_or, Bool.and_eq_true, Bool.or_eq_true] at h'
    obtain ⟨⟨rfl, hpd⟩, rfl⟩ := h'
    rcases hpd with rfl | rfl <;> simp [Port.read, pure, Except.pure]

/-- The values the arguments name, read at their input ports. -/
def Args.read {ps : List Port} (vs : Values α ps) : {ins : List Port} → Args ps ins → Values α ins
  | [], () => ()
  | _ :: _, (j, js) => (cast (Port.carrier_eq_of_accepts j.2) (Values.get vs j.1), Args.read vs js)

/-- The references of the arguments among the references `refs` of the values. -/
def Args.refs {ps : List Port} (refs : Array Ref) : {ins : List Port} → Args ps ins → List Ref
  | [], () => []
  | _ :: _, (j, js) => refs[j.1.val]?.getD ⟨0, 0⟩ :: Args.refs refs js

/-- Value `j` of `ps ++ qs` is value `j` of `ps` when `j` is below the length of `ps`. -/
theorem _root_.TorchLean.Values.get_append_left :
    ∀ {ps qs : List Port} (xs : Values α ps) (ys : Values α qs)
    (j : Nat) (hj : j < ps.length) (hj' : j < (ps ++ qs).length),
    (⟨(ps ++ qs).get ⟨j, hj'⟩, Values.get (Values.append xs ys) ⟨j, hj'⟩⟩ : Op.Value α) =
      ⟨ps.get ⟨j, hj⟩, Values.get xs ⟨j, hj⟩⟩
  | _ :: _, _, (_, _), _, 0, _, _ => rfl
  | _ :: _, _, (_, xs), ys, j + 1, hj, hj' =>
      get_append_left xs ys j (Nat.lt_of_succ_lt_succ hj) (Nat.lt_of_succ_lt_succ hj')

/-- Value `ps.length + k` of `ps ++ qs` is value `k` of `qs`. -/
theorem _root_.TorchLean.Values.get_append_right :
    ∀ {ps qs : List Port} (xs : Values α ps) (ys : Values α qs)
    (j k : Nat) (hk : k < qs.length) (hj : j < (ps ++ qs).length), j = ps.length + k →
    (⟨(ps ++ qs).get ⟨j, hj⟩, Values.get (Values.append xs ys) ⟨j, hj⟩⟩ : Op.Value α) =
      ⟨qs.get ⟨k, hk⟩, Values.get ys ⟨k, hk⟩⟩
  | [], _, (), _, j, k, _, _, h => by
      obtain rfl : j = k := by simpa using h
      rfl
  | _ :: ps, _, (_, xs), ys, j, k, hk, hj, h => by
      obtain ⟨j, rfl⟩ : ∃ i, j = i + 1 := ⟨ps.length + k, by simp at h; omega⟩
      exact get_append_right xs ys j k hk (Nat.lt_of_succ_lt_succ hj) (by simp at h; omega)

/-- Output `k` of erased values is value `k` at its port. -/
theorem _root_.TorchLean.Values.getElem?_toErased :
    ∀ {ps : List Port} (ys : Values α ps) (k : Nat) (hk : k < ps.length),
    (Values.toErased ps ys)[k]? = some ⟨ps.get ⟨k, hk⟩, Values.get ys ⟨k, hk⟩⟩
  | _ :: _, (_, _), 0, _ => rfl
  | _ :: _, (_, ys), k + 1, hk => getElem?_toErased ys k (Nat.lt_of_succ_lt_succ hk)

/-- The references of the arguments read the values they name, which decode at the input ports as
the values `Args.read` reads. -/
theorem args_read {ps : List Port} (vs : Values α ps) (refs : Array Ref)
    (earlier : Array (List (Op.Value α)))
    (hrefs : ∀ (j : Nat) (hj : j < ps.length), ∃ r, refs[j]? = some r ∧
      r.read earlier = .ok ⟨ps.get ⟨j, hj⟩, Values.get vs ⟨j, hj⟩⟩) :
    ∀ {ins : List Port} (args : Args ps ins), ∃ values,
      (args.refs refs).mapM (·.read earlier) = .ok values ∧
        Values.ofErased ins values = .ok (args.read vs)
  | [], () => ⟨[], rfl, rfl⟩
  | _ :: _, (j, js) => by
      obtain ⟨r, hr, hread⟩ := hrefs j.1.val j.1.isLt
      obtain ⟨values, hm, ho⟩ := args_read vs refs earlier hrefs js
      refine ⟨⟨_, Values.get vs j.1⟩ :: values, ?_, ?_⟩
      · simp only [Args.refs, hr, Option.getD_some, List.mapM_cons, hread, hm, bind,
          Except.bind, pure, Except.pure]
      · simp only [Values.ofErased, Port.read_of_accepts j.2, ho, Args.read, bind,
        Except.bind, pure,
          Except.pure]

variable [Context α] [Op.Capabilities α]

/-- The values of a step: its rule's `forward` values on the values its arguments name. -/
def Step.eval {ps : List Port} (vs : Values α ps) : {outs : List Port} → Step ps outs →
    Values α outs
  | _, .op o args h => ((o.rule? (α := α)).get (h.hasRuleEverywhere α)).forward (args.read vs)

variable {K : Port → Type} {σ : Shape}

/-- The values of the graph on the input `x` under the constant valuation `val`: `x`, then the
outputs of each step on the earlier values, or `val p c` for a constant `c` at port `p`. -/
def eval (val : (p : Port) → K p → p.Carrier α) :
    {ps : List Port} → OpGraph K σ ps → Tensor α σ → Values α ps
  | _, .input, x => (x, ())
  | _, .push g step, x => let vs := g.eval val x; Values.append vs (step.eval vs)
  | _, .const g _ c, x => Values.append (g.eval val x) (val _ c, ())

/-- The nodes, payload and value references of the graph of registered operators, built node for
node: the input, then an operator node per step and a constant node per constant, with its value in
the payload. -/
def builder (val : (p : Port) → K p → p.Carrier α) :
    {ps : List Port} → OpGraph K σ ps → IR.Graph.Builder × IR.Payload α × Array Ref
  | _, .input =>
      let (b, r) := ({} : IR.Graph.Builder).input 0 (.value σ)
      (b, {}, #[r])
  | _, .push g (.op o args _) =>
      let (b, P, refs) := g.builder val
      let (b, outs) := b.op o (args.refs refs)
      (b, P, refs ++ outs.toArray)
  | _, .const g p c =>
      let (b, P, refs) := g.builder val
      let (b, P, r) := b.constant P ⟨p, val p c⟩
      (b, P, refs.push r)

/-- The graph of registered operators of the typed graph with the outputs `outs`: its nodes, its
payload and the reference of every value. -/
def toOp (val : (p : Port) → K p → p.Carrier α) {ps : List Port} (g : OpGraph K σ ps)
    (outs : Array (Fin ps.length)) : IR.Imported α :=
  let (b, P, refs) := g.builder val
  ⟨⟨b.nodes, outs.map fun j => refs[j.val]?.getD ⟨0, 0⟩⟩, P, refs⟩

/-- Running the built nodes of a typed graph succeeds under any payload that agrees with the built
one on the constants, and the reference of every value reads it. -/
theorem run_builder (val : (p : Port) → K p → p.Carrier α) (x : Tensor α σ) :
    ∀ {ps : List Port} (g : OpGraph K σ ps) (Q : IR.Payload α),
      (∀ m < (g.builder val).1.nodes.size, Q.const? m = (g.builder val).2.1.const? m) →
      ∃ earlier, IR.runNodes Q [⟨.value σ, x⟩] #[] (g.builder val).1.nodes.toList =
          .ok earlier ∧
        earlier.size = (g.builder val).1.nodes.size ∧ (g.builder val).2.2.size = ps.length ∧
        ∀ (j : Nat) (hj : j < ps.length), ∃ r, (g.builder val).2.2[j]? = some r ∧
          r.read earlier = .ok ⟨ps.get ⟨j, hj⟩, Values.get (g.eval val x) ⟨j, hj⟩⟩
  | _, .input, Q, _ => by
      refine ⟨#[[⟨.value σ, x⟩]], ?_, rfl, rfl, ?_⟩
      · simp [builder, IR.Graph.Builder.input, IR.runNodes, IR.Node.eval,
          Port.read_self, bind, Except.bind, pure, Except.pure]
      · intro j hj
        obtain rfl : j = 0 := by simpa using hj
        exact ⟨⟨0, 0⟩, rfl, rfl⟩
  | _, .push (ps := ps) (outs := outs) g (.op o args h), Q, hQ => by
      rcases hbg : g.builder val with ⟨b, P, refs⟩
      have hb : (OpGraph.push g (.op o args h)).builder val =
          (⟨b.nodes.push (.op ⟨_, _, o⟩ (args.refs refs).toArray)⟩, P,
            refs ++ ((List.range outs.length).map fun k => (⟨b.nodes.size, k⟩ : Ref)).toArray) := by
        simp only [builder, hbg, IR.Graph.Builder.op, IR.Graph.Builder.add,
          IR.Node.outs]
      rw [hb] at hQ ⊢
      obtain ⟨earlier, hrun, hsize, hrefsSize, hrefs⟩ := run_builder val x g Q (by
        rw [hbg]
        dsimp only
        intro m hm
        exact hQ m (by simp; omega))
      rw [hbg] at hrun hsize hrefsSize hrefs
      simp only at hrun hsize hrefsSize hrefs
      let vs := g.eval val x
      let rule := (o.rule? (α := α)).get (h.hasRuleEverywhere α)
      have hs : o.rule? (α := α) = some rule := (Option.some_get _).symm
      obtain ⟨values, hm, ho⟩ := args_read vs refs earlier hrefs args
      have hnode : (IR.Node.op ⟨_, _, o⟩ (args.refs refs).toArray).eval Q
          [⟨.value σ, x⟩] earlier earlier.size =
            .ok (Values.toErased outs (rule.forward (args.read vs))) :=
        IR.evalOp_eq o h.check_eq_ok h.checksValues_eq_false hs (by simpa using hm) ho
      refine ⟨earlier ++ #[Values.toErased outs (rule.forward (args.read vs))], ?_, ?_, ?_, ?_⟩
      · rw [Array.toList_push, IR.runNodes_append, hrun]
        exact IR.runNodes_singleton hnode
      · simp [hsize]
      · simp [hrefsSize]
      · intro j hj
        by_cases hlt : j < ps.length
        · obtain ⟨r, hr, hread⟩ := hrefs j hlt
          refine ⟨r, ?_, ?_⟩
          · rw [Array.getElem?_append_left (by omega)]
            exact hr
          · rw [IR.Ref.read_append hread]
            simp only [eval]
            exact congrArg Except.ok (Values.get_append_left _ _ j hlt hj).symm
        · have hk : j - ps.length < outs.length := by simp at hj; omega
          refine ⟨⟨b.nodes.size, j - ps.length⟩, ?_, ?_⟩
          · rw [Array.getElem?_append_right (by omega)]
            simp [hrefsSize, hk]
          · rw [← hsize]
            have hpush : earlier ++ #[Values.toErased outs (rule.forward (args.read vs))] =
                earlier.push (Values.toErased outs (rule.forward (args.read vs))) :=
              (Array.push_eq_append ..).symm
            rw [hpush, IR.Ref.read_push_getElem? (Values.getElem?_toErased _ _ hk)]
            simp only [eval, Step.eval]
            exact congrArg Except.ok
              (Values.get_append_right _ _ j (j - ps.length) hk hj (by omega)).symm
  | _, .const (ps := ps) g p c, Q, hQ => by
      rcases hbg : g.builder val with ⟨b, P, refs⟩
      have hb : (OpGraph.const g p c).builder val =
          (⟨b.nodes.push (.const p)⟩,
            { P with
              const? := fun m => if m = b.nodes.size then some ⟨p, val p c⟩ else P.const? m },
            refs.push ⟨b.nodes.size, 0⟩) := by
        simp only [builder, hbg, IR.Graph.Builder.constant]
      rw [hb] at hQ ⊢
      obtain ⟨earlier, hrun, hsize, hrefsSize, hrefs⟩ := run_builder val x g Q (by
        rw [hbg]
        dsimp only
        intro m hm
        have := hQ m (by simp; omega)
        simpa [Nat.ne_of_lt hm] using this)
      rw [hbg] at hrun hsize hrefsSize hrefs
      simp only at hrun hsize hrefsSize hrefs
      have hc : Q.const? earlier.size = some ⟨p, val p c⟩ := by
        have := hQ b.nodes.size (by simp)
        simpa [hsize] using this
      have hnode : (IR.Node.const p).eval Q [⟨.value σ, x⟩] earlier earlier.size =
          .ok [⟨p, val p c⟩] := by
        simp [IR.Node.eval, hc, Port.read_self, bind, Except.bind, pure, Except.pure]
      refine ⟨earlier ++ #[[⟨p, val p c⟩]], ?_, ?_, ?_, ?_⟩
      · rw [Array.toList_push, IR.runNodes_append, hrun]
        exact IR.runNodes_singleton hnode
      · simp [hsize]
      · simp [hrefsSize]
      · intro j hj
        by_cases hlt : j < ps.length
        · obtain ⟨r, hr, hread⟩ := hrefs j hlt
          refine ⟨r, ?_, ?_⟩
          · rw [Array.getElem?_push_lt (by omega)]
            rw [Array.getElem?_eq_getElem (by omega)] at hr
            exact hr
          · rw [IR.Ref.read_append hread]
            simp only [eval]
            exact congrArg Except.ok (Values.get_append_left _ _ j hlt hj).symm
        · have hj' : j = ps.length := by simp at hj; omega
          subst hj'
          refine ⟨⟨b.nodes.size, 0⟩, ?_, ?_⟩
          · rw [← hrefsSize, Array.getElem?_push_size]
          · rw [← hsize]
            have hpush : earlier ++ #[[⟨p, val p c⟩]] = earlier.push [⟨p, val p c⟩] :=
              (Array.push_eq_append ..).symm
            rw [hpush, IR.Ref.read_push_getElem? (v := ⟨p, val p c⟩) rfl]
            have := Values.get_append_right (ps := ps) (qs := [p]) (g.eval val x) (val p c, ())
              ps.length 0 (by simp) hj (by omega)
            simp only [eval]
            exact congrArg Except.ok this.symm

/-- Every node of the graph of registered operators of a typed graph runs, for every input. -/
theorem eval_ok_of_admissible (val : (p : Port) → K p → p.Carrier α) {ps : List Port}
    (g : OpGraph K σ ps) (outs : Array (Fin ps.length)) (x : Tensor α σ) :
    ∃ values, (g.toOp val outs).graph.run (g.toOp val outs).payload [⟨.value σ, x⟩] =
      .ok values ∧ values.size = (g.toOp val outs).graph.nodes.size := by
  rcases hbg : g.builder val with ⟨b, P, refs⟩
  obtain ⟨earlier, hrun, hsize, -, -⟩ := run_builder val x g P (fun _ _ => by rw [hbg])
  rw [hbg] at hrun hsize
  exact ⟨earlier, by simpa [toOp, hbg, IR.Graph.run] using hrun, by simpa [toOp, hbg]⟩

/-- The graph of registered operators of a typed graph returns the typed values at its outputs,
for every input. -/
theorem eval_toOp (val : (p : Port) → K p → p.Carrier α) {ps : List Port} (g : OpGraph K σ ps)
    (outs : Array (Fin ps.length)) (x : Tensor α σ) :
    (g.toOp val outs).graph.eval (g.toOp val outs).payload [⟨.value σ, x⟩] =
      .ok (outs.toList.map fun j => ⟨ps.get j, Values.get (g.eval val x) j⟩) := by
  rcases hbg : g.builder val with ⟨b, P, refs⟩
  obtain ⟨earlier, hrun, -, -, hrefs⟩ := run_builder val x g P (fun _ _ => by rw [hbg])
  rw [hbg] at hrun hrefs
  simp only at hrun hrefs
  simp only [toOp, hbg, IR.Graph.eval, IR.Graph.run, hrun, bind, Except.bind]
  rw [Array.toList_map, List.mapM_map]
  induction outs.toList with
  | nil => rfl
  | cons j js ih =>
      obtain ⟨r, hr, hread⟩ := hrefs j.val j.isLt
      simp only [List.mapM_cons, hr, Option.getD_some, Function.comp, hread, ih, bind,
        Except.bind, pure, Except.pure, List.map_cons]

end IR.OpGraph

end TorchLean
