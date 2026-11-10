/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Apply
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.Tensor.Op.Table.Linear

/-!
# Recording a registered operator on the CPU tape: what is proved once

For every registered operator, a successful `Tape.apply` returns the rule's forward value at the
next node id. When the operator has a value output, it records one node whose backward checks the
upstream gradient and sends the rule's backward at every input that needs a gradient, in the
rule's order (the alias rule's pullback when two value positions name one node). Each way `apply`
can fail is stated with its error.

The reverse sweeps drop every contribution to a node that needs no gradient, so the recorded node
sweeps as the node prepared with every input flagged, under the same alias policy. By
`apply_backward_eq_filter`, the recorded backward is that node's backward, keeping only the
contributions to parents that need a gradient. On any tape that keeps the recorded tape's nodes,
replacing the recorded backward by that node's leaves each sweep unchanged (`apply_backwardDense`,
`apply_backwardDenseAll` and `apply_backwardDenseFrom`). An arm whose prepared pullback skips the
products of inputs without a gradient, as `Op.linear` does (`apply_linear_backward`), therefore
computes the sweep of its dense pullback.

For `max` and `min` of one node with itself, the alias rule is concrete. When the scalar supports
quotient arithmetic, the recorded node has that node as its only parent
(`apply_extremum_self_parents`), and its backward sends the parent one contribution when the
parent needs a gradient: the alias rule's average of the gradient with itself
(`apply_extremum_self_backward`). Without quotient arithmetic, both positions are kept
(`apply_extremum_self_parents'`).
-/

@[expose] public section

namespace TorchLean

/-- The contributions of the active inputs are the contributions of every input, masked by the
flags. -/
theorem Tangents.select_eq_mask {α : Type} [TorchLean.Storage α] :
    (ps : List TorchLean.Port) → (flags : TorchLean.Flags ps) → (gs : TorchLean.Tangents α ps) →
    TorchLean.Tangents.select ps flags gs =
      TorchLean.Grads.mask ps flags
        (TorchLean.Tangents.select ps (TorchLean.Flags.all ps) gs)
  | [], _, _ => rfl
  | ⟨d, _⟩ :: ps, (a, flags), (_, gs) => by
      cases d <;> cases a <;>
        simp [TorchLean.Tangents.select, TorchLean.Grads.mask, TorchLean.Port.select,
          TorchLean.Port.mask, TorchLean.Flags.all, TorchLean.Tangents.select_eq_mask ps flags gs]

namespace Tape

variable {α : Type} [Storage α] {ins outs : List Port}

/-- Without an alias, the prepared node is the rule's and no position is left out. -/
theorem prepared_of_aliasOf_eq_none {rule : Op.Rule α ins outs} {ids : List (Option Nat)}
    (h : aliasOf rule ids = none) (xs : Values α ins) (flags : Flags ins) :
    prepared rule ids xs flags = (rule.prepare xs flags, none) := by
  simp only [prepared, h]

/-- With an alias, the prepared node is the alias rule's and its second position is left out. -/
theorem prepared_of_aliasOf_eq_some {rule : Op.Rule α ins outs} {ids : List (Option Nat)}
    {r : Op.AliasRule α ins outs rule.forward} (h : aliasOf rule ids = some r)
    (xs : Values α ins) (flags : Flags ins) :
    prepared rule ids xs flags = (r.prepare xs flags, some r.second.val) := by
  simp only [prepared, h]

/-- The pullback prepared with `flags` is the pullback prepared with every input flagged, masked
by `flags`, with or without the alias rule. -/
theorem prepared_pullback (rule : Op.Rule α ins outs) (ids : List (Option Nat))
    (xs : Values α ins) (flags : Flags ins) (g : Tangents α outs) :
    (prepared rule ids xs flags).1.pullback g =
      Grads.mask ins flags ((prepared rule ids xs (Flags.all ins)).1.pullback g) := by
  unfold prepared
  split
  · rename_i r _
    exact r.prepare_eq xs flags g
  · rw [rule.prepare_eq, rule.prepare_eq, Tangents.select_eq_mask]

/-- Masking by the runtime flags removes exactly the contributions to parents that need no
gradient. -/
theorem contributions_mask (t : Tape (SomeTensor α)) : (ps : List Port) → (args : Args α ps) →
    (gs : Grads α ps) →
    Grads.contributions ps (Args.ids ps args) (Grads.mask ps (Args.flags t ps args) gs) =
      (Grads.contributions ps (Args.ids ps args) gs).map
        (Option.filter fun c => (t.getNode? c.1).any (·.requiresGrad))
  | [], _, _ => rfl
  | ⟨d, _⟩ :: ps, (a, args), (g, gs) => by
      have ih := contributions_mask t ps args gs
      cases d with
      | value =>
        simp only [Args.ids, Args.flags, Grads.mask, Grads.contributions, List.map_cons, ih,
          Arg.id?]
        cases g with
        | none => cases (t.getNode? a).any (·.requiresGrad) <;> rfl
        | some g =>
          cases h : (t.getNode? a).any (·.requiresGrad) <;>
            simp [Port.mask, Grad.contribution?, Option.filter, h]
      | frozen => simp [Args.ids, Args.flags, Grads.mask, Grads.contributions, ih, Arg.id?,
          Grad.contribution?, Option.filter]
      | detached => simp [Args.ids, Args.flags, Grads.mask, Grads.contributions, ih, Arg.id?,
          Grad.contribution?, Option.filter]
      | index => simp [Args.ids, Args.flags, Grads.mask, Grads.contributions, ih, Arg.id?,
          Grad.contribution?, Option.filter]
      | bool => simp [Args.ids, Args.flags, Grads.mask, Grads.contributions, ih, Arg.id?,
          Grad.contribution?, Option.filter]
      | integer => simp [Args.ids, Args.flags, Grads.mask, Grads.contributions, ih, Arg.id?,
          Grad.contribution?, Option.filter]

/-- Emitting the contributions masked by the runtime flags emits the contributions to parents
that need a gradient, in the same order. -/
theorem emit_mask (t : Tape (SomeTensor α)) (rule : Op.Rule α ins outs) (args : Args α ins)
    (gs : Grads α ins) :
    emit rule (Args.ids ins args) (Grads.mask ins (Args.flags t ins args) gs) =
      (emit rule (Args.ids ins args) gs).filter
        fun c => (t.getNode? c.1).any (·.requiresGrad) := by
  apply Array.toList_inj.mp
  simp only [Array.toList_filter, emit, contributions_mask, List.filter_filterMap]
  congr 1
  funext i
  rw [List.getElem?_map]
  cases (Grads.contributions ins (Args.ids ins args) gs)[i.val]? <;> rfl

/-- After a successful read, every parent a contribution names is a node of the tape. -/
theorem exists_getNode?_of_read_eq_ok {t : Tape (SomeTensor α)} :
      (ps : List Port) → (args : Args α ps) →
    (xs : Values α ps) → Args.read t ps args = .ok xs → (gs : Grads α ps) →
    ∀ c, some c ∈ Grads.contributions ps (Args.ids ps args) gs → ∃ n, t.getNode? c.1 = some n
  | [], _, _, _, _, c, hc => by simp [Grads.contributions] at hc
  | ⟨d, s⟩ :: ps, (a, args), xs, h, (g, gs), c, hc => by
      simp only [Args.read, bind, Except.bind] at h
      cases ha : Arg.read t ⟨d, s⟩ a with
      | error e => rw [ha] at h; cases h
      | ok x =>
        rw [ha] at h
        cases hr : Args.read t ps args with
        | error e => rw [hr] at h; cases h
        | ok xs' =>
          simp only [Args.ids, Grads.contributions, List.mem_cons] at hc
          rcases hc with hc | hc
          · cases d with
            | value =>
              cases g with
              | none => simp [Arg.id?, Grad.contribution?] at hc
              | some g =>
                simp only [Arg.id?, Grad.contribution?, Option.some.injEq] at hc
                subst hc
                simp only [Arg.read, requireValue, getValue?] at ha
                cases hn : t.getNode? a with
                | none => rw [hn] at ha; cases ha
                | some n => exact ⟨n, rfl⟩
            | frozen => simp [Grad.contribution?] at hc
            | detached => simp [Grad.contribution?] at hc
            | index => simp [Grad.contribution?] at hc
            | bool => simp [Grad.contribution?] at hc
            | integer => simp [Grad.contribution?] at hc
          · exact exists_getNode?_of_read_eq_ok ps args xs' hr gs c hc

/-- An emitted contribution is one of the per-port contributions. -/
theorem mem_contributions_of_mem_emit {rule : Op.Rule α ins outs} {ids : List (Option Nat)}
    {gs : Grads α ins}
    {c : Nat × SomeTensor α} (hc : c ∈ (emit rule ids gs).toList) :
    some c ∈ Grads.contributions ins ids gs := by
  simp only [emit, List.mem_filterMap] at hc
  obtain ⟨i, -, hi⟩ := hc
  cases h : (Grads.contributions ins ids gs)[i.val]? with
  | none => rw [h] at hi; cases hi
  | some o =>
    rw [h] at hi
    cases hi
    exact List.mem_of_getElem? h

/-- A successful `record` has at most one value output, returns the outputs of the prepared
value at the next id, and appends the recorded node exactly when there is a value output. -/
theorem record_eq_ok {t t' : Tape (SomeTensor α)} {name : String} {rule : Op.Rule α ins outs}
    {ids : List (Option Nat)} {flags : Flags ins} {p : Op.Prepared α ins outs × Option Nat}
    {o : Outs α outs} (h : record t name rule ids flags p = .ok (t', o)) :
    valuePorts outs ≤ 1 ∧ o = Values.outs t.size outs p.1.value ∧
      t' = match Values.value? outs p.1.value with
        | none => t
        | some y => (t.addNode (recordedNode name rule ids flags p y)).1 := by
  unfold record at h
  split at h
  · cases h
  · rename_i hv
    refine ⟨Nat.le_of_not_gt hv, ?_⟩
    split at h <;> rename_i hy <;>
      (simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
       obtain ⟨h1, h2⟩ := h
       simp only [hy]
       exact ⟨h2.symm, h1.symm⟩)

/-- Leaving out of a fold the elements on which the step does nothing keeps its result. -/
theorem foldlM_filter {β γ : Type} {f : β → γ → Result β} {p : γ → Bool} {l : List γ}
    (hf : ∀ c ∈ l, p c = false → ∀ a, f a c = pure a) (a : β) :
    (l.filter p).foldlM f a = l.foldlM f a := by
  induction l generalizing a with
  | nil => rfl
  | cons c l ih =>
    have ih' := ih (fun c' hc' => hf c' (List.mem_cons_of_mem _ hc'))
    by_cases hc : p c = true
    · simp only [List.filter_cons, hc, ite_true, List.foldlM_cons]
      cases f a c with
      | error e => rfl
      | ok a' => exact ih' a'
    · have hc' : p c = false := by simpa using hc
      simp only [List.filter_cons, hc', List.foldlM_cons, hf c List.mem_cons_self hc' a]
      exact ih' a

/-- Folding a step that ignores contributions to nodes without a gradient over the backward of
the node prepared with every input flagged gives the fold over the backward of the node prepared
with the runtime flags. -/
theorem recordedNode_foldlM {β : Type} {t T : Tape (SomeTensor α)} {name : String}
    {rule : Op.Rule α ins outs} {args : Args α ins} {xs : Values α ins} {flags : Flags ins}
    {y : SomeTensor α} (hxs : Args.read t ins args = .ok xs)
    (hT : ∀ id n, t.getNode? id = some n → T.getNode? id = some n)
    (f : β → Nat × SomeTensor α → Result β)
    (hf : ∀ a c n, T.getNode? c.1 = some n → n.requiresGrad = false → f a c = pure a)
    (g : SomeTensor α) (acc : β) :
    (do let cs ← (recordedNode name rule (Args.ids ins args) flags
          (prepared rule (Args.ids ins args) xs (Flags.all ins)) y).backward g
        cs.foldlM f acc) =
      (do let cs ← (recordedNode name rule (Args.ids ins args) flags
          (prepared rule (Args.ids ins args) xs (Args.flags t ins args)) y).backward g
          cs.foldlM f acc) := by
  simp only [recordedNode]
  cases seed g outs with
  | error e => rfl
  | ok gs =>
    simp only [bind, Except.bind, pure, Except.pure]
    rw [prepared_pullback rule _ xs (Args.flags t ins args), emit_mask, ← Array.foldlM_toList,
      ← Array.foldlM_toList, Array.toList_filter, foldlM_filter]
    intro c hc hp a
    obtain ⟨n, hn⟩ := exists_getNode?_of_read_eq_ok ins args xs hxs _ c
      (mem_contributions_of_mem_emit hc)
    rw [hn] at hp
    exact hf a c n (hT _ _ hn) (by simpa using hp)

section Sweep

variable {V : Type}

/-- `T` with the backward of node `k` replaced by `b`. -/
def withBackward (T : Tape V) (k : Nat) (b : V → Result (Array (Nat × V))) : Tape V :=
  { nodes := T.nodes.modify k fun n => { n with backward := b } }

/-- Node `k` of `T.withBackward k b` is node `k` of `T` with backward `b`. -/
theorem getNode?_withBackward_self (T : Tape V) (k : Nat) (b : V → Result (Array (Nat × V))) :
    (T.withBackward k b).getNode? k = (T.getNode? k).map ({ · with backward := b }) := by
  simp [withBackward, getNode?, Array.getElem?_modify]

/-- Replacing the backward of node `k` keeps every node with a different id. -/
theorem getNode?_withBackward_of_ne {T : Tape V} {k id : Nat} (b : V → Result (Array (Nat × V)))
    (h : k ≠ id) : (T.withBackward k b).getNode? id = T.getNode? id := by
  simp [withBackward, getNode?, Array.getElem?_modify, h]

/-- Replacing the backward of a node keeps the number of nodes. -/
theorem size_withBackward (T : Tape V) (k : Nat) (b : V → Result (Array (Nat × V))) :
    (T.withBackward k b).nodes.size = T.nodes.size := Array.size_modify

variable [GradValue V]

/-- Replacing the backward of a node does not change `addGradDense`. -/
theorem addGradDense_withBackward (T : Tape V) (k : Nat) (b : V → Result (Array (Nat × V))) :
    addGradDense (T.withBackward k b) = addGradDense T := by
  funext grads id g
  unfold addGradDense
  by_cases hid : k = id
  · subst hid
    rw [getNode?_withBackward_self]
    cases T.getNode? k <;> rfl
  · rw [getNode?_withBackward_of_ne b hid]

/-- Replacing the backward of a node does not change `addGradAll`. -/
theorem addGradAll_withBackward (T : Tape V) (k : Nat) (b : V → Result (Array (Nat × V))) :
    addGradAll (T.withBackward k b) = addGradAll T := by
  funext grads id g
  unfold addGradAll
  by_cases hid : k = id
  · subst hid
    rw [getNode?_withBackward_self]
    cases T.getNode? k <;> rfl
  · rw [getNode?_withBackward_of_ne b hid]

/-- Replacing the backward of node `k` by one whose contributions accumulate to the same array
keeps every step of the optional-gradient sweep. -/
theorem backwardDenseStep_withBackward {T : Tape V} {k : Nat} {n : Node V}
    {b : V → Result (Array (Nat × V))} (hk : T.getNode? k = some n)
    (hb : ∀ g (acc : Array (Option V)),
      (do let cs ← b g; cs.foldlM (fun acc2 (pid, pg) => addGradDense T acc2 pid pg) acc) =
        (do let cs ← n.backward g
            cs.foldlM (fun acc2 (pid, pg) => addGradDense T acc2 pid pg) acc)) :
    backwardDenseStep (T.withBackward k b) = backwardDenseStep T := by
  funext acc id
  unfold backwardDenseStep
  rw [addGradDense_withBackward]
  by_cases hid : k = id
  · subst hid
    rw [getNode?_withBackward_self, hk]
    rcases acc[k]? with _ | _ | d
    · rfl
    · rfl
    · simp only [Option.map_some, pure_bind]
      split
      · rfl
      · exact hb d acc
  · rw [getNode?_withBackward_of_ne b hid]

/-- Replacing the backward of node `k` by one whose contributions accumulate to the same array keeps
the optional-gradient sweep `backwardDense`. -/
theorem backwardDense_withBackward {T : Tape V} {k : Nat} {n : Node V}
    {b : V → Result (Array (Nat × V))} (hk : T.getNode? k = some n)
    (hb : ∀ g (acc : Array (Option V)),
      (do let cs ← b g; cs.foldlM (fun acc2 (pid, pg) => addGradDense T acc2 pid pg) acc) =
        (do let cs ← n.backward g
            cs.foldlM (fun acc2 (pid, pg) => addGradDense T acc2 pid pg) acc))
    (outId : Nat) (seed : V) :
    backwardDense (T.withBackward k b) outId seed = backwardDense T outId seed := by
  unfold backwardDense
  rw [backwardDenseStep_withBackward hk hb, size_withBackward]
  by_cases hid : k = outId
  · subst hid
    rw [getNode?_withBackward_self, hk]
    rfl
  · rw [getNode?_withBackward_of_ne b hid]

omit [GradValue V] in
/-- Replacing the backward of a node does not change `totalize`. -/
theorem totalize_withBackward [GradZero V] (T : Tape V) (k : Nat)
    (b : V → Result (Array (Nat × V))) (grads : Array (Option V)) :
    totalize (T.withBackward k b) grads = totalize T grads := by
  unfold totalize withBackward
  apply Array.ext_getElem?
  intro i
  simp only [Array.getElem?_mapIdx, Array.getElem?_modify]
  by_cases hid : k = i <;> simp [hid, Option.map_map, Function.comp_def]

/-- Replacing the backward of node `k` by one whose contributions accumulate to the same array
keeps every step of the total-gradient sweep. -/
theorem backwardDenseFromStep_withBackward {T : Tape V} {k : Nat} {n : Node V}
    {b : V → Result (Array (Nat × V))} (hk : T.getNode? k = some n)
    (hb : ∀ g (acc : Array V),
      (do let cs ← b g; cs.foldlM (fun acc2 (pid, pg) => addGradAll T acc2 pid pg) acc) =
        (do let cs ← n.backward g
            cs.foldlM (fun acc2 (pid, pg) => addGradAll T acc2 pid pg) acc)) :
    backwardDenseFromStep (T.withBackward k b) = backwardDenseFromStep T := by
  funext acc id
  unfold backwardDenseFromStep
  rw [addGradAll_withBackward]
  by_cases hid : k = id
  · subst hid
    rw [getNode?_withBackward_self, hk]
    simp only [Option.map_some, pure_bind]
    split
    · rfl
    · rcases acc[k]? with _ | d
      · rfl
      · simp only
        split
        · cases GradValue.check d n.value with
          | error e => rfl
          | ok u => exact hb _ acc
        · rfl
  · rw [getNode?_withBackward_of_ne b hid]

/-- Replacing the backward of node `k` by one whose contributions accumulate to the same array keeps
the total-gradient sweep `backwardDenseFrom`. -/
theorem backwardDenseFrom_withBackward {T : Tape V} {k : Nat} {n : Node V}
    {b : V → Result (Array (Nat × V))} (hk : T.getNode? k = some n)
    (hb : ∀ g (acc : Array V),
      (do let cs ← b g; cs.foldlM (fun acc2 (pid, pg) => addGradAll T acc2 pid pg) acc) =
        (do let cs ← n.backward g
            cs.foldlM (fun acc2 (pid, pg) => addGradAll T acc2 pid pg) acc))
    (grads : Array V) :
    backwardDenseFrom (T.withBackward k b) grads = backwardDenseFrom T grads := by
  have hloop : ∀ m acc, backwardDenseFromLoop (T.withBackward k b) m acc =
      backwardDenseFromLoop T m acc := by
    intro m
    induction m with
    | zero => intro acc; rfl
    | succ m ih =>
      intro acc
      simp only [backwardDenseFromLoop, backwardDenseFromStep_withBackward hk hb, ih]
  unfold backwardDenseFrom
  rw [size_withBackward, hloop]

end Sweep

variable [Context α] [Op.Capabilities α]

/-- A successful `apply` is the `record` of the rule's node prepared with the runtime flags. -/
theorem apply_eq_record {t t' : Tape (SomeTensor α)} {name : String} {op : Op ins outs}
    {args : Args α ins} {o : Outs α outs} {rule : Op.Rule α ins outs} {xs : Values α ins}
    (h : apply t name op args = .ok (t', o)) (hspec : op.rule? (α := α) = some rule)
    (hxs : Args.read t ins args = .ok xs) :
    record t name rule (Args.ids ins args) (Args.flags t ins args)
      (prepared rule (Args.ids ins args) xs (Args.flags t ins args)) = .ok (t', o) := by
  unfold apply at h
  cases hc : op.check with
  | error e => rw [hc] at h; cases h
  | ok u =>
    rw [hc, hspec, hxs] at h
    dsimp only [bind, Except.bind] at h
    cases hv : rule.validate xs with
    | error e => rw [hv] at h; cases h
    | ok u' => rw [hv] at h; exact h

/-- After a successful `apply`, the outputs are the rule's forward value at the next id. -/
theorem apply_value {t t' : Tape (SomeTensor α)} {name : String} {op : Op ins outs}
    {args : Args α ins} {o : Outs α outs} {rule : Op.Rule α ins outs} {xs : Values α ins}
    (h : apply t name op args = .ok (t', o)) (hspec : op.rule? (α := α) = some rule)
    (hxs : Args.read t ins args = .ok xs) :
    o = Values.outs t.size outs (rule.forward xs) := by
  rw [(record_eq_ok (apply_eq_record h hspec hxs)).2.1, prepared_value]

/-- After a successful `apply` with a value output, the next id holds the recorded node of the
rule prepared with the runtime flags. -/
theorem apply_node {t t' : Tape (SomeTensor α)} {name : String} {op : Op ins outs}
    {args : Args α ins} {o : Outs α outs} {rule : Op.Rule α ins outs} {xs : Values α ins}
    {y : SomeTensor α} (h : apply t name op args = .ok (t', o))
    (hspec : op.rule? (α := α) = some rule) (hxs : Args.read t ins args = .ok xs)
    (hy : Values.value? outs (rule.forward xs) = some y) :
    t'.getNode? t.size = some (recordedNode name rule (Args.ids ins args) (Args.flags t ins args)
      (prepared rule (Args.ids ins args) xs (Args.flags t ins args)) y) := by
  obtain ⟨-, -, ht⟩ := record_eq_ok (apply_eq_record h hspec hxs)
  rw [prepared_value, hy] at ht
  subst ht
  simp [addNode, getNode?, size]

/-- A successful `apply` keeps every node of the tape. -/
theorem apply_getNode? {t t' : Tape (SomeTensor α)} {name : String} {op : Op ins outs}
    {args : Args α ins} {o : Outs α outs} {rule : Op.Rule α ins outs} {xs : Values α ins}
    (h : apply t name op args = .ok (t', o)) (hspec : op.rule? (α := α) = some rule)
    (hxs : Args.read t ins args = .ok xs) {id : Nat} {n : Node (SomeTensor α)}
    (hn : t.getNode? id = some n) : t'.getNode? id = some n := by
  obtain ⟨-, -, ht⟩ := record_eq_ok (apply_eq_record h hspec hxs)
  subst ht
  split
  · exact hn
  · have hlt : id < t.nodes.size := (Array.getElem?_eq_some_iff.mp hn).1
    simpa [addNode, getNode?, Array.getElem?_push, Nat.ne_of_lt hlt] using hn

/-- Without an alias, the recorded backward checks the upstream gradient and sends the rule's
backward at every input that needs a gradient, in the rule's order. -/
theorem apply_backward {t t' : Tape (SomeTensor α)} {name : String} {op : Op ins outs}
    {args : Args α ins} {o : Outs α outs} {rule : Op.Rule α ins outs} {xs : Values α ins}
    {y : SomeTensor α} (h : apply t name op args = .ok (t', o))
    (hspec : op.rule? (α := α) = some rule) (hxs : Args.read t ins args = .ok xs)
    (hy : Values.value? outs (rule.forward xs) = some y)
    (halias : aliasOf rule (Args.ids ins args) = none) (g : SomeTensor α) :
    (t'.getNode? t.size).map (fun node => node.backward g) =
      some ((seed g outs).map fun gs => emit rule (Args.ids ins args)
        (Tangents.select ins (Args.flags t ins args) (rule.backward xs gs))) := by
  rw [apply_node h hspec hxs hy, prepared_of_aliasOf_eq_none halias]
  simp only [Option.map_some, recordedNode, rule.prepare_eq]
  cases seed g outs <;> rfl

/-- The recorded backward is the backward of the node prepared with every input flagged, keeping
the contributions to parents that need a gradient, with or without the alias rule. -/
theorem apply_backward_eq_filter {t t' : Tape (SomeTensor α)} {name : String}
    {op : Op ins outs} {args : Args α ins} {o : Outs α outs} {rule : Op.Rule α ins outs}
    {xs : Values α ins} {y : SomeTensor α} (h : apply t name op args = .ok (t', o))
    (hspec : op.rule? (α := α) = some rule) (hxs : Args.read t ins args = .ok xs)
    (hy : Values.value? outs (rule.forward xs) = some y) (g : SomeTensor α) :
    (t'.getNode? t.size).map (fun node => node.backward g) =
      some (((recordedNode name rule (Args.ids ins args) (Args.flags t ins args)
        (prepared rule (Args.ids ins args) xs (Flags.all ins)) y).backward g).map
          (·.filter fun c => (t.getNode? c.1).any (·.requiresGrad))) := by
  rw [apply_node h hspec hxs hy]
  simp only [Option.map_some, recordedNode]
  cases seed g outs with
  | error e => rfl
  | ok gs =>
    simp only [bind, Except.bind, pure, Except.pure, Except.map]
    rw [prepared_pullback rule _ xs (Args.flags t ins args), emit_mask]

/-- After a successful `apply` without a value output, the tape is unchanged. -/
theorem apply_tape_of_value?_eq_none {t t' : Tape (SomeTensor α)} {name : String}
    {op : Op ins outs} {args : Args α ins} {o : Outs α outs} {rule : Op.Rule α ins outs}
    {xs : Values α ins} (h : apply t name op args = .ok (t', o))
    (hspec : op.rule? (α := α) = some rule) (hxs : Args.read t ins args = .ok xs)
    (hy : Values.value? outs (rule.forward xs) = none) : t' = t := by
  obtain ⟨-, -, ht⟩ := record_eq_ok (apply_eq_record h hspec hxs)
  rw [prepared_value, hy] at ht
  exact ht

/-- With an alias, the recorded backward sends the alias rule's pullback, prepared with the
runtime flags. -/
theorem apply_backward_of_aliasOf_eq_some {t t' : Tape (SomeTensor α)} {name : String}
    {op : Op ins outs} {args : Args α ins} {o : Outs α outs} {rule : Op.Rule α ins outs}
    {xs : Values α ins} {y : SomeTensor α} {r : Op.AliasRule α ins outs rule.forward}
    (h : apply t name op args = .ok (t', o))
    (hspec : op.rule? (α := α) = some rule) (hxs : Args.read t ins args = .ok xs)
    (hy : Values.value? outs (rule.forward xs) = some y)
    (halias : aliasOf rule (Args.ids ins args) = some r) (g : SomeTensor α) :
    (t'.getNode? t.size).map (fun node => node.backward g) =
      some ((seed g outs).map fun gs => emit rule (Args.ids ins args)
        ((r.prepare xs (Args.flags t ins args)).pullback gs)) := by
  rw [apply_node h hspec hxs hy, prepared_of_aliasOf_eq_some halias]
  simp only [Option.map_some, recordedNode]
  cases seed g outs <;> rfl

section Errors

variable {t : Tape (SomeTensor α)} {name : String} {op : Op ins outs} {args : Args α ins}
  {e : String} {rule : Op.Rule α ins outs} {xs : Values α ins}

/-- `apply` fails with the attribute check's error. -/
theorem apply_of_check_eq_error (hc : op.check = .error e) :
    apply t name op args = .error e := by
  simp [apply, hc, bind, Except.bind]

/-- `apply` fails when the scalar has no rule for the operator. -/
theorem apply_of_rule?_eq_none (hc : op.check = .ok ()) (hs : op.rule? (α := α) = none) :
    apply t name op args = .error s!"op: {name} has no rule for this scalar" := by
  simp [apply, hc, hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw]

/-- `apply` fails with the error of reading the parents. -/
theorem apply_of_read_eq_error (hc : op.check = .ok ()) (hs : op.rule? (α := α) = some rule)
    (hr : Args.read t ins args = .error e) : apply t name op args = .error e := by
  simp [apply, hc, hs, hr, bind, Except.bind]

/-- `apply` fails with the error of the value-domain check. -/
theorem apply_of_validate_eq_error (hc : op.check = .ok ()) (hs : op.rule? (α := α) = some rule)
    (hr : Args.read t ins args = .ok xs) (hv : rule.validate xs = .error e) :
    apply t name op args = .error e := by
  simp [apply, hc, hs, hr, hv, bind, Except.bind]

/-- `apply` fails on an operator with several value outputs. -/
theorem apply_of_one_lt_valuePorts (hc : op.check = .ok ()) (hs : op.rule? (α := α) = some rule)
    (hr : Args.read t ins args = .ok xs) (hv : rule.validate xs = .ok ())
    (hn : 1 < valuePorts outs) :
    apply t name op args = .error s!"op: {name} has several value outputs" := by
  simp [apply, hc, hs, hr, hv, record, hn, bind, Except.bind, throw, throwThe,
    MonadExceptOf.throw]

end Errors

section Sweeps

variable {t t' T : Tape (SomeTensor α)} {name : String} {op : Op ins outs} {args : Args α ins}
  {o : Outs α outs} {rule : Op.Rule α ins outs} {xs : Values α ins} {y : SomeTensor α}

/-- The optional-gradient sweep of any tape `T` that keeps the nodes `apply` returns is unchanged
when the recorded node's backward is replaced by the backward of the node prepared with every
input flagged, under the same alias policy. -/
theorem apply_backwardDense (h : apply t name op args = .ok (t', o))
    (hspec : op.rule? (α := α) = some rule) (hxs : Args.read t ins args = .ok xs)
    (hy : Values.value? outs (rule.forward xs) = some y)
    (hT : ∀ id n, t'.getNode? id = some n → T.getNode? id = some n) (outId : Nat)
    (seed : SomeTensor α) :
    backwardDense (T.withBackward t.size (recordedNode name rule (Args.ids ins args)
      (Args.flags t ins args) (prepared rule (Args.ids ins args) xs (Flags.all ins)) y).backward)
      outId seed = backwardDense T outId seed := by
  apply backwardDense_withBackward (hT _ _ (apply_node h hspec hxs hy))
  intro g acc
  refine recordedNode_foldlM hxs (fun id n hn => hT id n (apply_getNode? h hspec hxs hn)) _ ?_ g
    acc
  rintro a ⟨pid, pg⟩ n hn hreq
  simp only at hn
  simp [addGradDense, hn, hreq]

/-- The sweep the eager runtime runs, `backwardDenseAll`, is unchanged by the same replacement. -/
theorem apply_backwardDenseAll (h : apply t name op args = .ok (t', o))
    (hspec : op.rule? (α := α) = some rule) (hxs : Args.read t ins args = .ok xs)
    (hy : Values.value? outs (rule.forward xs) = some y)
    (hT : ∀ id n, t'.getNode? id = some n → T.getNode? id = some n) (outId : Nat)
    (seed : SomeTensor α) :
    backwardDenseAll (T.withBackward t.size (recordedNode name rule (Args.ids ins args)
      (Args.flags t ins args) (prepared rule (Args.ids ins args) xs (Flags.all ins)) y).backward)
      outId seed = backwardDenseAll T outId seed := by
  unfold backwardDenseAll
  simp only [apply_backwardDense h hspec hxs hy hT, totalize_withBackward]

/-- The total-gradient sweep `backwardDenseFrom` is unchanged by the same replacement. -/
theorem apply_backwardDenseFrom (h : apply t name op args = .ok (t', o))
    (hspec : op.rule? (α := α) = some rule) (hxs : Args.read t ins args = .ok xs)
    (hy : Values.value? outs (rule.forward xs) = some y)
    (hT : ∀ id n, t'.getNode? id = some n → T.getNode? id = some n)
    (grads : Array (SomeTensor α)) :
    backwardDenseFrom (T.withBackward t.size (recordedNode name rule (Args.ids ins args)
      (Args.flags t ins args) (prepared rule (Args.ids ins args) xs (Flags.all ins)) y).backward)
      grads = backwardDenseFrom T grads := by
  apply backwardDenseFrom_withBackward (hT _ _ (apply_node h hspec hxs hy))
  intro g acc
  refine recordedNode_foldlM hxs (fun id n hn => hT id n (apply_getNode? h hspec hxs hn)) _ ?_ g
    acc
  rintro a ⟨pid, pg⟩ n hn hreq
  simp only at hn
  simp [addGradAll, hn, hreq]

end Sweeps

section Alias

/-- With the alias rule enabled, `max` or `min` of one node with itself records that node once,
whatever its gradient flag. -/
theorem apply_extremum_self_parents {t t' : Tape (SomeTensor α)} {name : String} {s : Shape}
    {op : Op [Port.value s, Port.value s] [Port.value s]} {i : Nat} {o : Outs α [Port.value s]}
    {x : Tensor α s} (hop : op = Op.max (s := s) ∨ op = Op.min (s := s))
    (h : apply t name op (i, i, ()) = .ok (t', o))
    (hx : Args.read t [Port.value s, Port.value s] (i, i, ()) = .ok (x, x, ()))
    (hen : (inferInstance : Numeric.QuotientArithmetic α).supported = true) :
    (t'.getNode? t.size).map (·.parents) = some #[i] := by
  rcases hop with rfl | rfl
  · rw [apply_node h Op.rule?_max hx rfl]
    simp [recordedNode, prepared, aliasOf, hen, Args.ids, Arg.id?]
    all_goals rfl
  · rw [apply_node h Op.rule?_min hx rfl]
    simp [recordedNode, prepared, aliasOf, hen, Args.ids, Arg.id?]
    all_goals rfl

/-- With the alias rule enabled, the backward of `max` or `min` of one node with itself checks the
upstream gradient and, when that node needs a gradient, sends it one contribution, the alias rule's
average of the gradient with itself. Otherwise it sends none. -/
theorem apply_extremum_self_backward {t t' : Tape (SomeTensor α)} {name : String} {s : Shape}
    {op : Op [Port.value s, Port.value s] [Port.value s]} {i : Nat} {o : Outs α [Port.value s]}
    {x : Tensor α s} (hop : op = Op.max (s := s) ∨ op = Op.min (s := s))
    (h : apply t name op (i, i, ()) = .ok (t', o))
    (hx : Args.read t [Port.value s, Port.value s] (i, i, ()) = .ok (x, x, ()))
    (hen : (inferInstance : Numeric.QuotientArithmetic α).supported = true)
    (g : SomeTensor α) :
    (t'.getNode? t.size).map (fun node => node.backward g) =
      some ((requireGrad (τ := s) g).map fun g' =>
        if (t.getNode? i).any (·.requiresGrad) then #[(i, SomeTensor.ofTensor
          (Tensor.map (fun gradient =>
            Numeric.average gradient gradient) g'))] else #[]) := by
  rcases hop with rfl | rfl
  · rw [apply_node h Op.rule?_max hx rfl]
    simp only [Option.map_some, recordedNode, prepared, aliasOf, hen]
    simp [Args.ids, Arg.id?, Args.flags, seed, emit, Grads.contributions, bind, Except.bind,
      pure, Except.pure, Except.map]
    cases requireGrad (τ := s) g <;> by_cases hb : (t.getNode? i).any (·.requiresGrad) = true <;>
      simp only [hb, ↓reduceIte] <;> rfl
  · rw [apply_node h Op.rule?_min hx rfl]
    simp only [Option.map_some, recordedNode, prepared, aliasOf, hen]
    simp [Args.ids, Arg.id?, Args.flags, seed, emit, Grads.contributions, bind, Except.bind,
      pure, Except.pure, Except.map]
    cases requireGrad (τ := s) g <;> by_cases hb : (t.getNode? i).any (·.requiresGrad) = true <;>
      simp only [hb, ↓reduceIte] <;> rfl

/-- With the alias rule disabled, `max` or `min` of one node with itself records that node at both
positions. -/
theorem apply_extremum_self_parents' {t t' : Tape (SomeTensor α)}
    {name : String} {s : Shape} {op : Op [Port.value s, Port.value s] [Port.value s]} {i : Nat}
    {o : Outs α [Port.value s]} {x : Tensor α s}
    (hop : op = Op.max (s := s) ∨ op = Op.min (s := s))
    (h : apply t name op (i, i, ()) = .ok (t', o))
    (hx : Args.read t [Port.value s, Port.value s] (i, i, ()) = .ok (x, x, ()))
    (hen : (inferInstance : Numeric.QuotientArithmetic α).supported = false) :
    (t'.getNode? t.size).map (·.parents) = some #[i, i] := by
  rcases hop with rfl | rfl
  · rw [apply_node h Op.rule?_max hx rfl]
    simp [recordedNode, prepared, aliasOf, hen, Args.ids, Arg.id?]
    all_goals rfl
  · rw [apply_node h Op.rule?_min hx rfl]
    simp [recordedNode, prepared, aliasOf, hen, Args.ids, Arg.id?]
    all_goals rfl

end Alias

/-- The backward of unbatched `linear` checks the upstream gradient and sends the weight, bias and
input gradients, in that order, to the parents that need a gradient. Its arm computes only those
products. -/
theorem apply_linear_backward {t t' : Tape (SomeTensor α)} {name : String} {inDim outDim : Nat}
    {wId bId xId : Nat} {o : Outs α [Port.value [outDim]]} {W : Tensor α [outDim, inDim]}
    {b : Tensor α [outDim]} {x : Tensor α [inDim]}
    (h : apply t name (Op.linear (lead := []) (inDim := inDim) (outDim := outDim))
      (wId, bId, xId, ()) = .ok (t', o))
    (hxs : Args.read t [.value [outDim, inDim], .value [outDim], .value [inDim]]
      (wId, bId, xId, ()) = .ok (W, b, x, ())) (g : SomeTensor α) :
    (t'.getNode? t.size).map (fun node => node.backward g) =
      some ((requireGrad (τ := [outDim]) g).map fun g' =>
        #[(wId, SomeTensor.ofTensor (Linear.weightsDeriv x g')),
          (bId, SomeTensor.ofTensor g'),
          (xId, SomeTensor.ofTensor (Linear.inputDeriv W g'))].filter
          fun c => (t.getNode? c.1).any (·.requiresGrad)) := by
  rw [apply_backward h Op.rule?_linear hxs rfl rfl g]
  simp [Args.ids, Arg.id?, Args.flags, seed, emit, Grads.contributions, Tangents.select,
    Port.select, Grad.contribution?, bind, Except.bind, pure, Except.pure, Except.map]
  cases requireGrad (τ := [outDim]) g with
  | error e => rfl
  | ok g' =>
    by_cases h1 : (t.getNode? wId).any (·.requiresGrad) = true <;>
    by_cases h2 : (t.getNode? bId).any (·.requiresGrad) = true <;>
    by_cases h3 : (t.getNode? xId).any (·.requiresGrad) = true <;>
    simp [h1, h2, h3, List.finRange]

end Tape

end TorchLean
