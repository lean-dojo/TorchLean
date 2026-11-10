/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Backend.Graph
public import TorchLean.NN.Backend.Planner.Soundness

/-!
# Planning operator graphs

`IR.Graph.plan_sound` lifts `planOp_sound` from one operator to a whole graph. Every kernel of
a successful graph plan comes from an operator node of the graph, records that node's id and the
wire record of its operator, and carries a capsule that an available implementation offers for
that operator and the policy admits. The statement is about selection and covers neither native
execution nor numerical correctness.
-/

@[expose] public section

namespace TorchLean.IR

open Backend

/-- If `Node.plan?` returns a kernel, the node is an operator node on whose operator `planOp`
succeeds, and the kernel carries the node's id with the planned wire record and capsule. -/
theorem Node.plan?_eq_some {policy : KernelPolicy} {impls : Array Implementation} {id : Nat}
    {n : Node} {k : PlannedNodeKernel} (h : n.plan? policy impls id = .ok (some k)) :
    ∃ (ins outs : List Port) (o : Op ins outs) (refs : Array Ref) (pk : PlannedKernel),
      n = .op ⟨ins, outs, o⟩ refs ∧ planOp policy impls o = .ok pk ∧
      k = { nodeId := id, op := pk.op, capsule := pk.capsule } := by
  unfold Node.plan? at h
  split at h
  · simp [pure, Except.pure] at h
  · simp [pure, Except.pure] at h
  · simp [pure, Except.pure] at h
  · rename_i ins outs o refs _
    cases hp : planOp policy impls o with
    | error e => simp [hp, bind, Except.bind] at h
    | ok pk =>
      simp only [hp, bind, Except.bind, pure, Except.pure, Except.ok.injEq,
        Option.some.injEq] at h
      exact ⟨ins, outs, o, refs, pk, rfl, hp, h.symm⟩

/-- An entry of a successful `filterMapM` in `Except` is the result of `f` at an element. -/
private theorem mem_of_filterMapM_eq_ok {α β : Type} {f : α → Except String (Option β)} :
    ∀ {l : List α} {r : List β}, l.filterMapM f = .ok r → ∀ b ∈ r, ∃ a ∈ l, f a = .ok (some b)
  | [], r, h, b, hb => by
      simp only [List.filterMapM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h
      cases hb
  | a :: l, r, h, b, hb => by
      rw [List.filterMapM_cons] at h
      cases ha : f a with
      | error e => simp [ha, bind, Except.bind] at h
      | ok x =>
        cases hl : l.filterMapM f with
        | error e => cases x <;> simp [ha, hl, bind, Except.bind] at h
        | ok r' =>
          have ih := mem_of_filterMapM_eq_ok hl
          cases x with
          | none =>
              simp only [ha, hl, bind, Except.bind] at h
              cases h
              obtain ⟨a', ha', hf⟩ := ih b hb
              exact ⟨a', List.mem_cons_of_mem _ ha', hf⟩
          | some y =>
              simp only [ha, hl, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
              subst h
              rcases List.mem_cons.mp hb with rfl | hb'
              · exact ⟨a, List.mem_cons_self, ha⟩
              · obtain ⟨a', ha', hf⟩ := ih b hb'
                exact ⟨a', List.mem_cons_of_mem _ ha', hf⟩

/-- Every kernel of a successful plan belongs to an operator node of the graph: it records that
node's id and the wire record of the node's own operator, and its capsule names that operator, is
offered for it by an implementation available under `availability` and is admissible under
`policy`. -/
theorem Graph.plan_sound {g : Graph} {policy : KernelPolicy} {availability : Availability}
    {impls : Array Implementation} {plan : GraphKernelPlan}
    (h : g.plan policy availability impls = .ok plan) {k : PlannedNodeKernel}
    (hk : k ∈ plan.kernels) :
    ∃ (ins outs : List Port) (o : Op ins outs) (refs : Array Ref),
      g.nodes[k.nodeId]? = some (.op ⟨ins, outs, o⟩ refs) ∧ k.op = Op.encode ⟨ins, outs, o⟩ ∧
      Implementation.Offers (availability.filter impls) o k.capsule ∧ k.capsule.op = k.op ∧
      k.capsule.admissible policy = true := by
  unfold Graph.plan at h
  cases hc : g.check with
  | error e => simp [hc, bind, Except.bind] at h
  | ok u =>
    simp only [hc, bind, Except.bind] at h
    split at h
    · cases h
    · rename_i ks hks
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      have hl := congrArg (Array.toList <$> ·) hks
      simp only [Array.toList_filterMapM] at hl
      obtain ⟨⟨n, id⟩, hmem, hplan⟩ :=
        mem_of_filterMapM_eq_ok hl k (Array.mem_toList_iff.mpr hk)
      obtain ⟨ins, outs, o, refs, pk, rfl, hpk, rfl⟩ := Node.plan?_eq_some hplan
      obtain ⟨hop, hoffers, hcop, hadm⟩ := planOp_sound hpk
      exact ⟨ins, outs, o, refs,
        Array.mk_mem_zipIdx_iff_getElem?.mp (Array.mem_toList_iff.mp hmem), hop, hoffers, hcop,
        hadm⟩

end TorchLean.IR
