/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Graph
public import TorchLean.Tensor.Op.Invoke
public import TorchLean.NN.Kernel.Tensor
public import TorchLean.Tensor.ShapeErasure
public import TorchLean.Tensor.Pack

/-!
# Evaluating graphs of registered operators

`Node.eval` computes the outputs of one node of `TorchLean.IR.Graph` from the graph inputs and the
outputs of the earlier nodes, every output kept. An input node reads its graph input at its port
and a constant node its payload constant at its port (`Port.read`, so a value is accepted where a
frozen or detached port is expected). A custom node runs its checked program from the payload on
its value inputs (`runCustom`), and any other operator is invoked on the values its references
name (`Op.invoke`: the attribute check `Op.check`, the input ports, the rule's value-domain check,
then the rule's forward values). `Graph.run` evaluates the nodes in order, a node's id being the
number of nodes before it, and `Graph.eval` reads the graph's outputs. `Graph.check` is the
structural check of each node's references, ports and attributes (`Node.check`) and of the graph's
outputs.

`Node.eval_ports` proves that every output of a successful node is at the port the node declares,
and `Graph.run_size` and `Graph.run_ports` that a successful run has one entry per node, at the
node's ports. `Graph.run_step` reads node `j` of a successful run: it succeeds on a table that
agrees with the run below `j`, and its outputs are entry `j`.
`evalOp_eq` gives the outputs of an operator node whose attributes pass `Op.check`, whose rule
exists and does not check its input values (`Op.checksValues`), and whose references read values
that decode at its input ports: the rule's forward values. Per-operator equations follow from it
and the operator's `Op.rule?_<op>` lemma. `Node.eval_unary` and `Node.eval_binary` state it for one
and two value inputs. `runNodes_append` splits a run at any node, and `runNodes_cons_of_eval`
steps a run past a node with one output. `Ref.read_map` and `Ref.read_rel` read mapped outputs
and outputs related row by row.
-/

@[expose] public section

namespace TorchLean.IR

/-- A bind in `Except` succeeds exactly when its first computation does and the continuation
succeeds on its result. -/
theorem bind_eq_ok {ε β γ : Type} {x : Except ε β} {f : β → Except ε γ} {c : γ} :
    (x >>= f) = .ok c ↔ ∃ b, x = .ok b ∧ f b = .ok c := by
  cases x with
  | error e => exact ⟨fun h => (by cases h), fun ⟨_, h, _⟩ => (by cases h)⟩
  | ok b => exact ⟨fun h => ⟨b, rfl, h⟩, fun ⟨_, h, hf⟩ => (by cases h; exact hf)⟩

/-- A traversal succeeds with the results `cs` exactly when each entry's action returns its
result. -/
theorem mapM_eq_ok_iff {ε β γ : Type} {f : β → Except ε γ} :
    ∀ {l : List β} {cs : List γ}, l.mapM f = .ok cs ↔ List.Forall₂ (fun b c => f b = .ok c) l cs
  | [], cs => by
      cases cs <;> simp [pure, Except.pure]
  | b :: l, cs => by
      rw [List.mapM_cons]
      constructor
      · intro h
        obtain ⟨c, hc, h⟩ := IR.bind_eq_ok.1 h
        obtain ⟨cs', hcs, h⟩ := IR.bind_eq_ok.1 h
        cases h
        exact .cons hc (mapM_eq_ok_iff.1 hcs)
      · intro h
        cases h with
        | cons hc hcs =>
            exact IR.bind_eq_ok.2 ⟨_, hc, IR.bind_eq_ok.2
              ⟨_, mapM_eq_ok_iff.2 hcs, rfl⟩⟩

/-- A traversal by `f` then `g` succeeds exactly where the traversal by `f` does and `g` succeeds on
each of its results. -/
theorem mapM_bind_eq_ok {ε β γ δ : Type} {f : β → Except ε γ}
    {g : γ → Except ε δ} {l : List β} {zs : List δ} :
    l.mapM (fun b => f b >>= g) = .ok zs ↔
      ∃ ws, l.mapM f = .ok ws ∧ List.Forall₂ (fun w z => g w = .ok z) ws zs := by
  rw [IR.mapM_eq_ok_iff]
  induction l generalizing zs with
  | nil =>
      constructor
      · intro h
        cases h
        exact ⟨[], rfl, .nil⟩
      · rintro ⟨ws, hw, hz⟩
        cases hw
        cases hz
        exact .nil
  | cons b l ih =>
      constructor
      · intro h
        cases h with
        | cons hz hzs =>
            obtain ⟨w, hw, hwz⟩ := IR.bind_eq_ok.1 hz
            obtain ⟨ws, hws, hr⟩ := ih.1 hzs
            refine ⟨w :: ws, ?_, .cons hwz hr⟩
            rw [List.mapM_cons]
            exact IR.bind_eq_ok.2 ⟨w, hw, IR.bind_eq_ok.2 ⟨ws, hws, rfl⟩⟩
      · rintro ⟨ws, hws, hr⟩
        rw [List.mapM_cons] at hws
        obtain ⟨w, hw, hws⟩ := IR.bind_eq_ok.1 hws
        obtain ⟨ws', hws', hws⟩ := IR.bind_eq_ok.1 hws
        cases hws
        cases hr with
        | cons hwz hr => exact .cons (IR.bind_eq_ok.2 ⟨w, hw, hwz⟩) (ih.2 ⟨ws', hws', hr⟩)

/-- Related lists are related at every position. -/
theorem forall₂_getElem? {β γ : Type} {rel : β → γ → Prop} :
    ∀ {l : List β} {l' : List γ}, List.Forall₂ rel l l' → ∀ i : Nat, Option.Rel rel l[i]? l'[i]?
  | _, _, .nil, _ => .none
  | _, _, .cons h _, 0 => .some h
  | _, _, .cons _ t, i + 1 => forall₂_getElem? t i

/-- Every iteration of a successful loop that never stops early yields. -/
theorem forIn_eq_ok {ε β : Type} {f : β → PUnit → Except ε (ForInStep PUnit)}
    (hf : ∀ b, f b PUnit.unit ≠ .ok (.done PUnit.unit)) :
    ∀ {l : List β}, forIn l PUnit.unit f = .ok PUnit.unit →
      ∀ b ∈ l, f b PUnit.unit = .ok (.yield PUnit.unit)
  | [], _, b, hb => by simp at hb
  | a :: l, h, b, hb => by
      rw [List.forIn_cons] at h
      cases ha : f a PUnit.unit with
      | error e => rw [ha] at h; cases h
      | ok step =>
          cases step with
          | done u => cases u; exact absurd ha (hf a)
          | yield u =>
              cases u
              rw [ha] at h
              rcases List.mem_cons.1 hb with rfl | hb'
              · exact ha
              · exact forIn_eq_ok hf h b hb'

/-- A loop over a list succeeds where every iteration continues. -/
theorem forIn_eq_ok_of {ε β : Type} {f : β → PUnit → Except ε (ForInStep PUnit)} :
    ∀ {l : List β}, (∀ b ∈ l, f b PUnit.unit = .ok (.yield PUnit.unit)) →
      forIn l PUnit.unit f = .ok PUnit.unit
  | [], _ => rfl
  | a :: l, h => by
      simp only [List.forIn_cons, h a List.mem_cons_self, bind, Except.bind]
      exact forIn_eq_ok_of fun b hb => h b (List.mem_cons_of_mem _ hb)

/-- A traversal succeeds where its function succeeds on every entry. -/
theorem exists_mapM_eq_ok {ε β γ : Type} {f : β → Except ε γ} :
    ∀ {l : List β}, (∀ x ∈ l, ∃ y, f x = .ok y) → ∃ ys, l.mapM f = .ok ys
  | [], _ => ⟨[], rfl⟩
  | x :: l, h => by
    obtain ⟨y, hy⟩ := h x List.mem_cons_self
    obtain ⟨ys, hys⟩ := exists_mapM_eq_ok fun z hz => h z (List.mem_cons_of_mem _ hz)
    exact ⟨y :: ys, IR.mapM_eq_ok_iff.2 (.cons hy (IR.mapM_eq_ok_iff.1 hys))⟩

/-- An array traversal fails nowhere its function succeeds on every entry. -/
theorem mapM_ne_error {ε β γ : Type} {f : β → Except ε γ} {xs : Array β} {e : ε}
    (h : ∀ x ∈ xs, ∃ y, f x = .ok y) : xs.mapM f ≠ .error e := by
  rw [Array.mapM_eq_mapM_toList]
  obtain ⟨ys, hys⟩ := IR.exists_mapM_eq_ok fun x hx => h x (Array.mem_toList_iff.1 hx)
  rw [hys]
  simp [Functor.map, Except.map]

/-- A bind after an array traversal succeeds where the traversal's function succeeds on every
entry and the continuation succeeds on its result. -/
theorem exists_bind_mapM {ε β γ δ : Type} {f : β → Except ε γ} {xs : Array β}
    {k : Array γ → Except ε δ} (hf : ∀ x ∈ xs, ∃ y, f x = .ok y)
    (hk : ∀ ys, xs.mapM f = .ok ys → ∃ z, k ys = .ok z) : ∃ z, (xs.mapM f >>= k) = .ok z := by
  cases h : xs.mapM f with
  | error e => exact absurd h (IR.mapM_ne_error hf)
  | ok ys => exact hk ys h

/-- Entry `k` of list `i` of a list of lists is entry `k` past the lengths of the first `i` lists
of their concatenation. -/
theorem _root_.List.getElem?_flatten_of_getElem? {α : Type} :
    ∀ {L : List (List α)} {i k : Nat} {l : List α} {x : α}, L[i]? = some l → l[k]? = some x →
      L.flatten[((L.take i).map List.length).sum + k]? = some x
  | [], _, _, _, _, h, _ => by simp at h
  | l' :: L, 0, k, l, x, h, hx => by
      simp only [List.getElem?_cons_zero, Option.some.injEq] at h
      subst h
      simp [List.getElem?_append_left (List.getElem?_eq_some_iff.1 hx).1, hx]
  | l' :: L, i + 1, k, l, x, h, hx => by
      simp only [List.getElem?_cons_succ] at h
      have := List.getElem?_flatten_of_getElem? h hx
      simp only [List.take_succ_cons, List.map_cons, List.sum_cons, List.flatten_cons]
      rw [Nat.add_assoc, List.getElem?_append_right (by omega)]
      simpa using this

/-- A list of lists holds entry `j` of its concatenation as entry `k` of some list `i`, `j` being
`k` past the lengths of the first `i` lists. -/
theorem _root_.List.exists_of_getElem?_flatten {α : Type} :
    ∀ {L : List (List α)} {j : Nat} {x : α}, L.flatten[j]? = some x →
      ∃ i l k, L[i]? = some l ∧ l[k]? = some x ∧ j = ((L.take i).map List.length).sum + k
  | [], _, _, h => by simp at h
  | l :: L, j, x, h => by
      rw [List.flatten_cons, List.getElem?_append] at h
      split at h
      · exact ⟨0, l, j, rfl, h, by simp⟩
      · obtain ⟨i, l', k, hi, hk, hj⟩ := List.exists_of_getElem?_flatten h
        refine ⟨i + 1, l', k, by simpa using hi, hk, ?_⟩
        simp only [List.take_succ_cons, List.map_cons, List.sum_cons]
        omega

section
variable {ε σ τ α β : Type} {f : σ → α → Except ε σ} {g : τ → β → Except ε τ}

/-- A fold over the first components preserved by each second-component fold, relative to an
invariant indexed by position, gives a successful fold over their concatenation. -/
theorem foldlM_flatten_of_foldlM (I : Nat → σ → τ → Prop) :
    ∀ (l : List (α × List β)) {s : σ} {u : τ} {s' : σ},
      (∀ (i : Nat) (hi : i < l.length) (s : σ) (u : τ) (s' : σ), I i s u →
        f s l[i].1 = .ok s' → ∃ u', l[i].2.foldlM g u = .ok u' ∧ I (i + 1) s' u') →
      I 0 s u → (l.map (·.1)).foldlM f s = .ok s' →
        ∃ u', (l.map (·.2)).flatten.foldlM g u = .ok u' ∧ I l.length s' u'
  | [], s, u, s', _, h0, h => by
      simp only [List.map_nil, List.foldlM_nil, pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨u, rfl, h0⟩
  | x :: l, s, u, s', H, h0, h => by
      simp only [List.map_cons, List.foldlM_cons, List.flatten_cons, List.foldlM_append] at h ⊢
      obtain ⟨s₁, h₁, h⟩ := IR.bind_eq_ok.1 h
      obtain ⟨u₁, hu₁, hI⟩ := H 0 (by simp) s u s₁ h0 h₁
      obtain ⟨u', hu', hI'⟩ := foldlM_flatten_of_foldlM (fun i => I (i + 1)) l
        (fun i hi => H (i + 1) (by simp; omega)) hI h
      exact ⟨u', IR.bind_eq_ok.2 ⟨u₁, hu₁, hu'⟩, hI'⟩

/-- A fold over the concatenation of second components, each reflected by a step on the first
component relative to an invariant indexed by position, gives a successful first fold. -/
theorem foldlM_of_foldlM_flatten (I : Nat → σ → τ → Prop) :
    ∀ (l : List (α × List β)) {s : σ} {u : τ} {u' : τ},
      (∀ (i : Nat) (hi : i < l.length) (s : σ) (u : τ) (u' : τ), I i s u →
        l[i].2.foldlM g u = .ok u' → ∃ s', f s l[i].1 = .ok s' ∧ I (i + 1) s' u') →
      I 0 s u → (l.map (·.2)).flatten.foldlM g u = .ok u' →
        ∃ s', (l.map (·.1)).foldlM f s = .ok s' ∧ I l.length s' u'
  | [], s, u, u', _, h0, h => by
      simp only [List.map_nil, List.flatten_nil, List.foldlM_nil, pure, Except.pure,
        Except.ok.injEq] at h
      subst h
      exact ⟨s, rfl, h0⟩
  | x :: l, s, u, u', H, h0, h => by
      simp only [List.map_cons, List.foldlM_cons, List.flatten_cons, List.foldlM_append] at h ⊢
      obtain ⟨u₁, h₁, h⟩ := IR.bind_eq_ok.1 h
      obtain ⟨s₁, hs₁, hI⟩ := H 0 (by simp) s u u₁ h0 h₁
      obtain ⟨s', hs', hI'⟩ := foldlM_of_foldlM_flatten (fun i => I (i + 1)) l
        (fun i hi => H (i + 1) (by simp; omega)) hI h
      exact ⟨s', IR.bind_eq_ok.2 ⟨s₁, hs₁, hs'⟩, hI'⟩

end

/-- A loop over `l` from a flag array of size `N`, each step of which checks its id `id` (`P id`,
below `N`, flag unset) and sets its flag, succeeds only on distinct ids that pass the check with
their flags unset at the start. -/
theorem forIn_mark {ε : Type} {N : Nat} {P : Nat → Prop}
    {body : Nat → Array Bool → Except ε (ForInStep (Array Bool))}
    (hbody : ∀ id s s', body id s = .ok s' →
      P id ∧ id < N ∧ s[id]?.getD false = false ∧ s' = .yield (s.set! id true)) :
    ∀ (l : List Nat) (s r : Array Bool), s.size = N → forIn l s body = .ok r →
      l.Nodup ∧ ∀ id ∈ l, P id ∧ s[id]?.getD false = false
  | [], _, _, _, _ => by simp
  | a :: l, s, r, hs, h => by
      rw [List.forIn_cons] at h
      obtain ⟨s', ha, h⟩ := IR.bind_eq_ok.1 h
      obtain ⟨hP, hlt, hfree, rfl⟩ := hbody a s s' ha
      obtain ⟨hnd, hl⟩ := forIn_mark hbody l (s.set! a true) r (by simp [hs]) h
      have hne : ∀ id ∈ l, id ≠ a := by
        intro id hid hia
        subst hia
        have := (hl id hid).2
        simp [Array.set!, hs, hlt] at this
      refine ⟨List.nodup_cons.2 ⟨fun ha' => hne a ha' rfl, hnd⟩, ?_⟩
      intro id hid
      rcases List.mem_cons.1 hid with rfl | hid
      · exact ⟨hP, hfree⟩
      · obtain ⟨hPid, hfid⟩ := hl id hid
        refine ⟨hPid, ?_⟩
        simpa [Array.set!, Array.getElem?_setIfInBounds, Ne.symm (hne id hid)] using hfid

variable {α : Type} [Storage α]

/-- The value at reference `r` among the outputs of earlier nodes. -/
def Ref.read {β : Type} (earlier : Array (List β)) (r : Ref) : Except String β :=
  match earlier[r.node]? >>= (·[r.out]?) with
  | some v => pure v
  | none => throw s!"IR: no output {r.out} of node {r.node}"

/-- Reading output `r` of mapped outputs maps the value read. -/
theorem Ref.read_map {β γ : Type} (F : β → γ) (earlier : Array (List β)) (r : Ref) :
    r.read (earlier.map (·.map F)) = (r.read earlier).map F := by
  unfold Ref.read
  cases h : earlier[r.node]? with
  | none => simp [h, Except.map, throw, throwThe, MonadExceptOf.throw]
  | some l =>
      cases h' : l[r.out]? with
      | none => simp [h, h', Except.map, throw, throwThe, MonadExceptOf.throw]
      | some v => simp [h, h', Except.map, pure, Except.pure]

/-- Reading output `r` of outputs related row by row reads related values, or fails on both. -/
theorem Ref.read_rel {β γ : Type} {rel : β → γ → Prop} {xs : Array (List β)}
    {ys : Array (List γ)} (h : List.Forall₂ (List.Forall₂ rel) xs.toList ys.toList) (r : Ref) :
    (∀ x, r.read xs = .ok x → ∃ y, r.read ys = .ok y ∧ rel x y) ∧
    (∀ y, r.read ys = .ok y → ∃ x, r.read xs = .ok x ∧ rel x y) := by
  have h₁ := forall₂_getElem? h r.node
  simp only [Array.getElem?_toList] at h₁
  unfold Ref.read
  generalize xs[r.node]? = ox at h₁ ⊢
  generalize ys[r.node]? = oy at h₁ ⊢
  cases h₁ with
  | none => simp [throw, throwThe, MonadExceptOf.throw]
  | some hl =>
      rename_i l l'
      have h₂ := forall₂_getElem? hl r.out
      simp only [bind, Option.bind]
      generalize l[r.out]? = ox at h₂ ⊢
      generalize l'[r.out]? = oy at h₂ ⊢
      cases h₂ with
      | none => simp [throw, throwThe, MonadExceptOf.throw]
      | some hx =>
          rename_i x y
          simp only [pure, Except.pure, Except.ok.injEq]
          exact ⟨fun _ e => ⟨y, rfl, e ▸ hx⟩, fun _ e => ⟨x, rfl, e ▸ hx⟩⟩

/-- Reading references of outputs related row by row reads related values, or fails on both. -/
theorem Ref.mapM_read_rel {β γ : Type} {rel : β → γ → Prop} {xs : Array (List β)}
    {ys : Array (List γ)} (h : List.Forall₂ (List.Forall₂ rel) xs.toList ys.toList) :
    ∀ (l : List Ref),
      (∀ vs, l.mapM (·.read xs) = .ok vs → ∃ ws, l.mapM (·.read ys) = .ok ws ∧
        List.Forall₂ rel vs ws) ∧
      (∀ ws, l.mapM (·.read ys) = .ok ws → ∃ vs, l.mapM (·.read xs) = .ok vs ∧
        List.Forall₂ rel vs ws)
  | [] => ⟨fun vs hv => ⟨[], rfl, by cases hv; exact .nil⟩,
      fun ws hw => ⟨[], rfl, by cases hw; exact .nil⟩⟩
  | r :: l => by
      obtain ⟨ih₁, ih₂⟩ := Ref.mapM_read_rel h l
      obtain ⟨hr₁, hr₂⟩ := Ref.read_rel h r
      refine ⟨fun vs hv => ?_, fun ws hw => ?_⟩
      · rw [List.mapM_cons] at hv
        obtain ⟨v, hv₁, hv⟩ := bind_eq_ok.1 hv
        obtain ⟨vs', hv₂, hv⟩ := bind_eq_ok.1 hv
        cases hv
        obtain ⟨w, hw₁, hvw⟩ := hr₁ v hv₁
        obtain ⟨ws', hw₂, hvws⟩ := ih₁ vs' hv₂
        refine ⟨w :: ws', ?_, .cons hvw hvws⟩
        rw [List.mapM_cons]
        exact bind_eq_ok.2 ⟨w, hw₁, bind_eq_ok.2 ⟨ws', hw₂, rfl⟩⟩
      · rw [List.mapM_cons] at hw
        obtain ⟨w, hw₁, hw⟩ := bind_eq_ok.1 hw
        obtain ⟨ws', hw₂, hw⟩ := bind_eq_ok.1 hw
        cases hw
        obtain ⟨v, hv₁, hvw⟩ := hr₂ w hw₁
        obtain ⟨vs', hv₂, hvws⟩ := ih₂ ws' hw₂
        refine ⟨v :: vs', ?_, .cons hvw hvws⟩
        rw [List.mapM_cons]
        exact bind_eq_ok.2 ⟨v, hv₁, bind_eq_ok.2 ⟨vs', hv₂, rfl⟩⟩

/-- A value at a value port as a shape-erased tensor. -/
def someTensor? : Op.Value α → Except String (SomeTensor α)
  | ⟨⟨.value, s⟩, x⟩ => pure ⟨s, x⟩
  | ⟨p, _⟩ => throw s!"IR: a custom operation reads a value, not {repr p}"

/-- The output of the checked program of custom operation `name` on its value inputs `values`, one
per input shape. -/
def customOutput (program : Kernel.Program α) (name : String) (shapes : List Shape)
    (output : Shape) (values : Array (SomeTensor α)) : Except String (Tensor α output) := do
  let inputs ← TensorPack.ofErased values (shapes := shapes)
  (program.eval inputs output).mapError
    fun error => s!"IR: custom {name}: {repr error}"

/-- The output of custom operation `name` at node `id` reading `refs`: the payload's program at
`id` on the values of its inputs, one per input shape. -/
def runCustom (payload : Payload α) (earlier : Array (List (Op.Value α))) (id : Nat)
    (refs : Array Ref) (name : String) (shapes : List Shape) (output : Shape) :
    Except String (Tensor α output) := do
  unless refs.size = shapes.length do
    throw s!"IR: custom {name}: expected {shapes.length} inputs"
  let some program := payload.custom? id |
    throw s!"IR: custom {name}: no checked program for node {id}"
  customOutput program name shapes output (← refs.mapM fun r => do someTensor? (← r.read earlier))

variable [Context α] [Op.Capabilities α]

/-- The outputs of operator node `id` reading `refs`: a custom operation runs its program, any
other operator is invoked on the values the references name. -/
def evalOp (payload : Payload α) (earlier : Array (List (Op.Value α))) (id : Nat)
    (refs : Array Ref) : Operator → Except String (List (Op.Value α))
  | ⟨_, _, .custom name shapes output⟩ => do
      pure [⟨.value output, ← runCustom payload earlier id refs name shapes output⟩]
  | ⟨_, _, op⟩ => op.invoke (refs.toList.mapM (·.read earlier))

/-- The outputs of node `n` with id `id` from the graph inputs and the outputs of earlier
nodes. -/
def Node.eval (payload : Payload α) (inputs : List (Op.Value α))
    (earlier : Array (List (Op.Value α))) (id : Nat) (n : Node) :
    Except String (List (Op.Value α)) :=
  match n with
  | .input index p => match inputs[index]? with
      | some v => do pure [⟨p, ← p.read v⟩]
      | none => throw s!"IR: no input {index}"
  | .const p => match payload.const? id with
      | some v => do pure [⟨p, ← p.read v⟩]
      | none => throw s!"IR: no constant for node {id}"
  | .op o refs => evalOp payload earlier id refs o

/-- The outputs of `nodes` after the outputs `earlier`, each node's id the number of nodes before
it, or the first error. -/
def runNodes (payload : Payload α) (inputs : List (Op.Value α))
    (earlier : Array (List (Op.Value α))) (nodes : List Node) :
    Except String (Array (List (Op.Value α))) :=
  nodes.foldlM (fun earlier n => do
    pure (earlier.push (← n.eval payload inputs earlier earlier.size))) earlier

/-- The outputs of every node of the graph, in order, or the first error. -/
def Graph.run (g : Graph) (payload : Payload α) (inputs : List (Op.Value α)) :
    Except String (Array (List (Op.Value α))) :=
  runNodes payload inputs #[] g.nodes.toList

/-- The values of the graph's outputs. -/
def Graph.eval (g : Graph) (payload : Payload α) (inputs : List (Op.Value α)) :
    Except String (List (Op.Value α)) := do
  let values ← g.run payload inputs
  g.outputs.toList.mapM (·.read values)

/-- Every output of a successful node is at the port the node declares. -/
theorem Node.eval_ports {payload : Payload α} {inputs : List (Op.Value α)}
    {earlier : Array (List (Op.Value α))} {id : Nat} {n : Node}
    {vs : List (Op.Value α)} (h : n.eval payload inputs earlier id = .ok vs) :
    vs.map Sigma.fst = n.outs := by
  have key : ∀ {q : Port} {e : Except String (q.Carrier α)},
      (do pure [(⟨q, ← e⟩ : Op.Value α)] : Except String _) = .ok vs →
        vs.map Sigma.fst = [q] := by
    intro q e he
    cases e with
    | error => cases he
    | ok x => cases he; rfl
  cases n with
  | input index p =>
      simp only [Node.eval] at h
      split at h
      · exact key h
      · cases h
  | const p =>
      simp only [Node.eval] at h
      split at h
      · exact key h
      · cases h
  | op op refs =>
      obtain ⟨ins, outs, o⟩ := op
      simp only [Node.eval, Node.outs] at h ⊢
      cases o with
      | custom name shapes output => exact key h
      | _ => exact Op.invoke_ports h

/-- Reading reference `r` is unchanged by outputs appended after its node. -/
theorem Ref.read_push {β : Type} {earlier : Array (List β)} {r : Ref} (h : r.node < earlier.size)
    (x : List β) : r.read (earlier.push x) = r.read earlier := by
  simp only [Ref.read, Array.getElem?_push_lt h, Array.getElem?_eq_getElem h]

/-- Reading output `k` of a node just appended. -/
theorem Ref.read_push_getElem? {β : Type} {earlier : Array (List β)} {vs : List β} {k : Nat}
    {v : β} (h : vs[k]? = some v) : (⟨earlier.size, k⟩ : Ref).read (earlier.push vs) = .ok v := by
  simp [Ref.read, h, pure, Except.pure]

/-- A reference that reads a value names an earlier node. -/
theorem Ref.read_lt {β : Type} {earlier : Array (List β)} {r : Ref} {v : β}
    (h : r.read earlier = .ok v) : r.node < earlier.size := by
  unfold Ref.read at h
  by_contra hn
  simp [Array.getElem?_eq_none (Nat.le_of_not_lt hn), throw, throwThe, MonadExceptOf.throw] at h

/-- A reference that reads a value reads it after any outputs are appended. -/
theorem Ref.read_append {β : Type} {earlier : Array (List β)} {r : Ref} {v : β}
    (h : r.read earlier = .ok v) (more : Array (List β)) : r.read (earlier ++ more) = .ok v := by
  have hlt := Ref.read_lt h
  unfold Ref.read at h ⊢
  rw [Array.getElem?_append_left hlt]
  exact h

/-- Running no nodes returns the outputs so far. -/
theorem runNodes_nil (payload : Payload α) (inputs : List (Op.Value α))
    (earlier : Array (List (Op.Value α))) :
    runNodes payload inputs earlier [] = .ok earlier :=
  rfl

/-- A run splits after any node. -/
theorem runNodes_append (payload : Payload α) (inputs : List (Op.Value α))
    (earlier : Array (List (Op.Value α))) (l₁ l₂ : List Node) :
    runNodes payload inputs earlier (l₁ ++ l₂) =
      (runNodes payload inputs earlier l₁ >>= fun e => runNodes payload inputs e l₂) := by
  simp [runNodes, List.foldlM_append]

/-- A run evaluates its first node at the number of earlier node entries, then runs the rest. -/
theorem runNodes_cons (payload : Payload α) (inputs : List (Op.Value α))
    (earlier : Array (List (Op.Value α))) (n : Node) (l : List Node) :
    runNodes payload inputs earlier (n :: l) = (do
      let x ← n.eval payload inputs earlier earlier.size
      runNodes payload inputs (earlier.push x) l) := by
  simp only [runNodes, List.foldlM_cons, bind_assoc, pure_bind]

/-- One node that evaluates appends its outputs. -/
theorem runNodes_singleton {payload : Payload α} {inputs : List (Op.Value α)}
    {earlier : Array (List (Op.Value α))} {n : Node} {x : List (Op.Value α)}
    (h : n.eval payload inputs earlier earlier.size = .ok x) :
    runNodes payload inputs earlier [n] = .ok (earlier ++ #[x]) := by
  simp [runNodes_cons, h, bind, Except.bind, runNodes_nil]

/-- An operator node that passes its attribute check, has a rule that does not check its input
values, and whose references read values that decode at its input ports as `xs`, returns the
rule's forward values on `xs`. -/
theorem evalOp_eq {ins outs : List Port} (op : Op ins outs)
    (hc : op.check = .ok ()) (hv : op.checksValues = false) {rule : Op.Rule α ins outs}
    (hs : op.rule? = some rule) {payload : Payload α}
    {earlier : Array (List (Op.Value α))} {id : Nat} {refs : Array Ref}
    {values : List (Op.Value α)} {xs : Values α ins}
    (hr : refs.toList.mapM (·.read earlier) = .ok values)
    (hx : Values.ofErased ins values = .ok xs) :
    evalOp payload earlier id refs ⟨ins, outs, op⟩ =
      .ok (Values.toErased outs (rule.forward xs)) := by
  unfold evalOp
  split
  · rename_i heq
    cases heq
    cases hs
  · rename_i heq
    cases heq
    simp only [Op.invoke, hc, hr, hx, hs, Op.Rule.check,
      Op.validate_eq_ok op hv hs xs, bind, Except.bind, pure, Except.pure,
      rule.prepare_value]

/-- An operator node with one value input whose reference reads `t` returns the rule's output on
`t`. -/
theorem Node.eval_unary {payload : Payload α} {inputs : List (Op.Value α)}
    {earlier : Array (List (Op.Value α))} {id : Nat} {a b : Shape}
    (op : Op [.value a] [.value b]) (hc : op.check = .ok ())
    (hv : op.checksValues = false) {rule : Op.Rule α [.value a] [.value b]}
    (hs : op.rule? = some rule) {r : Ref} {t : Tensor α a}
    (hr : r.read earlier = .ok ⟨.value a, t⟩) :
    (Node.op ⟨_, _, op⟩ #[r]).eval payload inputs earlier id =
      .ok [⟨.value b, (rule.forward (t, ())).1⟩] :=
  evalOp_eq op hc hv hs (xs := (t, ())) (values := [⟨.value a, t⟩])
    (by simp [hr, pure, Except.pure, bind, Except.bind])
    (by simp [Values.ofErased, Port.read_self, bind, Except.bind, pure, Except.pure])

/-- An operator node with two value inputs of one shape whose references read `t` and `u` returns
the rule's output on them. -/
theorem Node.eval_binary {payload : Payload α} {inputs : List (Op.Value α)}
    {earlier : Array (List (Op.Value α))} {id : Nat} {a b : Shape}
    (op : Op [.value a, .value a] [.value b]) (hc : op.check = .ok ())
    (hv : op.checksValues = false) {rule : Op.Rule α [.value a, .value a] [.value b]}
    (hs : op.rule? = some rule) {r q : Ref} {t u : Tensor α a}
    (hr : r.read earlier = .ok ⟨.value a, t⟩) (hq : q.read earlier = .ok ⟨.value a, u⟩) :
    (Node.op ⟨_, _, op⟩ #[r, q]).eval payload inputs earlier id =
      .ok [⟨.value b, (rule.forward (t, u, ())).1⟩] :=
  evalOp_eq op hc hv hs (xs := (t, u, ())) (values := [⟨.value a, t⟩, ⟨.value a, u⟩])
    (by simp [hr, hq, pure, Except.pure, bind, Except.bind])
    (by simp [Values.ofErased, Port.read_self, bind, Except.bind, pure, Except.pure])

/-- A run whose first node returns the one output `v` continues from outputs with one more entry,
`v` at the first node's position, that keep every earlier read. -/
theorem runNodes_cons_of_eval {payload : Payload α} {inputs : List (Op.Value α)}
    {earlier : Array (List (Op.Value α))} {n : Node} {v : Op.Value α}
    (h : n.eval payload inputs earlier earlier.size = .ok [v]) (rest : List Node) :
    ∃ earlier', runNodes payload inputs earlier (n :: rest) =
        runNodes payload inputs earlier' rest ∧
      earlier'.size = earlier.size + 1 ∧ (⟨earlier.size, 0⟩ : Ref).read earlier' = .ok v ∧
      ∀ (r : Ref) (w : Op.Value α), r.read earlier = .ok w → r.read earlier' = .ok w :=
  ⟨earlier.push [v], by rw [runNodes_cons, h]; rfl, by simp,
    by simp [Ref.read, pure, Except.pure],
    fun _ _ hw => (Ref.read_push (Ref.read_lt hw) _).trans hw⟩

/-- A successful run appends one entry per node, at the node's ports, and keeps the entries before
it. -/
theorem runNodes_ports {payload : Payload α} {inputs : List (Op.Value α)} :
    ∀ (nodes : List Node) (earlier values : Array (List (Op.Value α))),
      runNodes payload inputs earlier nodes = .ok values →
      values.size = earlier.size + nodes.length ∧
      (∀ j, j < earlier.size → values[j]? = earlier[j]?) ∧
      ∀ (j : Nat) (n : Node) (vs : List (Op.Value α)), nodes[j]? = some n →
        values[earlier.size + j]? = some vs → vs.map Sigma.fst = n.outs
  | [], earlier, values, h => by
      simp only [runNodes_nil, Except.ok.injEq] at h
      subst h
      exact ⟨by simp, fun _ _ => rfl, fun j n vs hn => by simp at hn⟩
  | n :: l, earlier, values, h => by
      rw [runNodes_cons] at h
      cases hx : n.eval payload inputs earlier earlier.size with
      | error e => simp [hx, bind, Except.bind] at h
      | ok x =>
        simp only [hx, bind, Except.bind] at h
        obtain ⟨h1, h2, h3⟩ := runNodes_ports l (earlier.push x) values h
        refine ⟨by simp [h1]; omega, fun j hj => ?_, fun j m vs hm hv => ?_⟩
        · rw [h2 j (by simp; omega), Array.getElem?_push_lt hj, Array.getElem?_eq_getElem hj]
        · cases j with
          | zero =>
              simp only [List.getElem?_cons_zero, Option.some.injEq] at hm
              subst hm
              rw [Nat.add_zero, h2 earlier.size (by simp), Array.getElem?_push_size] at hv
              cases hv
              exact Node.eval_ports hx
          | succ k =>
              simp only [List.getElem?_cons_succ] at hm
              exact h3 k m vs hm (by simpa [Nat.add_assoc, Nat.add_comm 1 k] using hv)

/-- A successful run has one entry per node. -/
theorem Graph.run_size {g : Graph} {payload : Payload α} {inputs : List (Op.Value α)}
    {values : Array (List (Op.Value α))} (h : g.run payload inputs = .ok values) :
    values.size = g.nodes.size := by
  simpa using (runNodes_ports g.nodes.toList #[] values h).1

/-- Entry `j` of a successful run is at the ports of node `j`. -/
theorem Graph.run_ports {g : Graph} {payload : Payload α} {inputs : List (Op.Value α)}
    {values : Array (List (Op.Value α))} (h : g.run payload inputs = .ok values)
    {j : Nat} {n : Node} {vs : List (Op.Value α)} (hn : g.nodes[j]? = some n)
    (hv : values[j]? = some vs) : vs.map Sigma.fst = n.outs :=
  (runNodes_ports g.nodes.toList #[] values h).2.2 j n vs (by simpa using hn) (by simpa using hv)

/-- In a successful run of `nodes` from `earlier`, node `j` of the list succeeds on a table that
agrees with the run's outputs below its id `earlier.size + j`, and its outputs are the run's entry
at that id. -/
theorem runNodes_step {payload : Payload α} {inputs : List (Op.Value α)} :
    ∀ (nodes : List Node) (earlier values : Array (List (Op.Value α))),
      runNodes payload inputs earlier nodes = .ok values →
      ∀ (j : Nat) (n : Node), nodes[j]? = some n →
        ∃ (pre : Array (List (Op.Value α))) (outs : List (Op.Value α)),
          pre.size = earlier.size + j ∧ (∀ k, k < pre.size → pre[k]? = values[k]?) ∧
          n.eval payload inputs pre (earlier.size + j) = .ok outs ∧
          values[earlier.size + j]? = some outs
  | [], _, _, _, j, n, hn => by simp at hn
  | m :: rest, earlier, values, h, j, n, hn => by
      rw [runNodes_cons] at h
      cases hx : m.eval payload inputs earlier earlier.size with
      | error e => simp [hx, bind, Except.bind] at h
      | ok x =>
        simp only [hx, bind, Except.bind] at h
        obtain ⟨hsize, hpre, -⟩ := runNodes_ports rest (earlier.push x) values h
        cases j with
        | zero =>
            simp only [List.getElem?_cons_zero, Option.some.injEq] at hn
            subst hn
            refine ⟨earlier, x, by simp, fun k hk => ?_, by simpa using hx, ?_⟩
            · rw [hpre k (by simp; omega), Array.getElem?_push_lt hk, Array.getElem?_eq_getElem hk]
            · rw [Nat.add_zero, hpre earlier.size (by simp), Array.getElem?_push_size]
        | succ k =>
            simp only [List.getElem?_cons_succ] at hn
            obtain ⟨pre, outs, h1, h2, h3, h4⟩ :=
              runNodes_step rest (earlier.push x) values h k n hn
            refine ⟨pre, outs, by simp [h1]; omega, h2, ?_, ?_⟩
            · simpa [Nat.add_assoc, Nat.add_comm 1 k] using h3
            · simpa [Nat.add_assoc, Nat.add_comm 1 k] using h4

/-- In a successful run, node `j` succeeds on a table of the first `j` outputs of the run, and its
outputs are entry `j` of the run. -/
theorem Graph.run_step {g : Graph} {payload : Payload α} {inputs : List (Op.Value α)}
    {values : Array (List (Op.Value α))} (h : g.run payload inputs = .ok values)
    {j : Nat} {n : Node} (hn : g.nodes[j]? = some n) :
    ∃ (pre : Array (List (Op.Value α))) (outs : List (Op.Value α)),
      pre.size = j ∧ (∀ k, k < j → pre[k]? = values[k]?) ∧
      n.eval payload inputs pre j = .ok outs ∧ values[j]? = some outs := by
  obtain ⟨pre, outs, h1, h2, h3, h4⟩ :=
    runNodes_step g.nodes.toList #[] values h j n (by simpa using hn)
  simp only [Array.size_empty, Nat.zero_add] at h1 h3 h4
  exact ⟨pre, outs, h1, fun k hk => h2 k (by omega), h3, h4⟩

end TorchLean.IR

namespace TorchLean

/-- Whether an output at port `q` can feed the input port `p`: the same port, or a value where a
frozen or detached port of its shape is expected, as `Port.read` accepts. -/
def Port.accepts (p q : Port) : Bool :=
  q == p || (q.kind == .value && (p.kind == .frozen || p.kind == .detached) &&
    q.shape == p.shape)

/-- Every port accepts an output at itself. -/
theorem Port.accepts_self (p : Port) : p.accepts p = true := by
  simp [Port.accepts]

end TorchLean

namespace TorchLean.IR

/-- The port of output `r` of a node before node `i`. -/
def Graph.portBefore (g : Graph) (i : Nat) (r : Ref) : Option Port :=
  if r.node < i then (g.nodes[r.node]?).bind fun n => n.outs[r.out]? else none

/-- The check of node `i`, where `portBefore r` is the port of an earlier output `r`. An operator
is accepted when it has one reference per input port, each naming an earlier output at a port its
input accepts, and passes `Op.check`. An input or a constant is always accepted. -/
def Node.check (portBefore : Ref → Option Port) (i : Nat) : Node → Except String Unit
  | .op ⟨ins, _, o⟩ refs => do
      unless refs.size = ins.length do
        throw s!"IR check: node {i}: {refs.size} references for {ins.length} inputs"
      for (r, p) in refs.toList.zip ins do
        match portBefore r with
        | some q =>
            unless p.accepts q do
              throw s!"IR check: node {i}: output {r.out} of node {r.node} has port \
                {repr q}, which input port {repr p} does not accept"
        | none => throw s!"IR check: node {i}: no earlier output {r.out} of node {r.node}"
      o.check
  | _ => pure ()

/-- The structural check of a graph: each node passes `Node.check` (an operator has one reference
per input port, each naming an existing output of an earlier node at a port its input accepts, and
passes `Op.check`), and each graph output names an existing output. -/
def Graph.check (g : Graph) : Except String Unit := do
  for i in List.range g.nodes.size do
    match g.nodes[i]? with
    | some n => n.check (g.portBefore i) i
    | none => pure ()
  for r in g.outputs do
    unless (g.portBefore g.nodes.size r).isSome do
      throw s!"IR check: no output {r.out} of node {r.node}"

/-- Output `r` of a node before node `i` exists at port `q`: its node comes before node `i` and
has an output at position `r.out`, at port `q`. -/
def Graph.OutputBefore (g : Graph) (i : Nat) (r : Ref) (q : Port) : Prop :=
  r.node < i ∧ ∃ n, g.nodes[r.node]? = some n ∧ n.outs[r.out]? = some q

/-- A node is well formed where `out r q` says that output `r` exists at port `q`. An operator is
well formed when it has one reference per input port, in port order, each naming an existing
output at a port its input accepts, and passes `Op.check`. A graph input and a constant are always
well formed. -/
def Node.WellFormed (out : Ref → Port → Prop) : Node → Prop
  | .op ⟨ins, _, o⟩ refs =>
      List.Forall₂ (fun r p => ∃ q, out r q ∧ p.accepts q = true) refs.toList ins ∧
        o.check = .ok ()
  | _ => True

/-- A well-formed graph: every node is well formed with the outputs of the nodes before it, and
every graph output names an existing output. -/
def Graph.WellFormed (g : Graph) : Prop :=
  (∀ i (h : i < g.nodes.size), g.nodes[i].WellFormed (g.OutputBefore i)) ∧
    ∀ r ∈ g.outputs, ∃ q, g.OutputBefore g.nodes.size r q

/-- `portBefore` returns the port of an existing output of an earlier node. -/
theorem Graph.portBefore_eq_some_iff {g : Graph} {i : Nat} {r : Ref} {q : Port} :
    g.portBefore i r = some q ↔ g.OutputBefore i r q := by
  unfold portBefore OutputBefore
  by_cases h : r.node < i
  · simp [h, Option.bind_eq_some_iff]
  · simp [h]

/-- A node passes `Node.check` exactly when it is well formed with the outputs `portBefore`
names. -/
theorem Node.check_ok_iff {portBefore : Ref → Option Port} {i : Nat} {n : Node} :
    n.check portBefore i = .ok () ↔ n.WellFormed fun r q => portBefore r = some q := by
  rcases n with _ | _ | ⟨⟨ins, outs, o⟩, refs⟩
  · simp [check, WellFormed, pure, Except.pure]
  · simp [check, WellFormed, pure, Except.pure]
  · simp only [check, WellFormed, List.forall₂_iff_zip, Array.length_toList]
    by_cases hlen : refs.size = ins.length
    · simp only [hlen, ite_true, true_and]
      constructor
      · intro h
        obtain ⟨u, hloop, hcheck⟩ := bind_eq_ok.1 h
        refine ⟨fun {r p} hrp => ?_, hcheck⟩
        have hstep := forIn_eq_ok ?_ hloop (r, p) hrp
        · revert hstep
          cases hq : portBefore r with
          | none => simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
          | some q =>
              by_cases hacc : p.accepts q = true
              · simp [hacc]
              · simp [hacc, throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
        · intro x
          cases portBefore x.1 with
          | none => simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
          | some q =>
              by_cases hacc : x.2.accepts q = true
              · simp [hacc, pure, Except.pure]
              · simp [hacc, throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
      · rintro ⟨hall, hcheck⟩
        refine bind_eq_ok.2 ⟨PUnit.unit, forIn_eq_ok_of fun x hx => ?_, hcheck⟩
        obtain ⟨q, hq, hacc⟩ := hall (a := x.1) (b := x.2) hx
        simp [hq, hacc, pure, Except.pure]
    · simp [hlen, throw, throwThe, MonadExceptOf.throw, bind, Except.bind]

/-- A graph passes `Graph.check` exactly when it is well formed: each operator node has one
reference per input port, each naming an existing output of an earlier node at a port its input
accepts, and passes `Op.check`, and each graph output names an existing output. -/
theorem Graph.check_ok_iff {g : Graph} : g.check = .ok () ↔ g.WellFormed := by
  have hrel : ∀ i, (fun r q => g.portBefore i r = some q) = g.OutputBefore i := fun i => by
    funext r q
    exact propext portBefore_eq_some_iff
  unfold check WellFormed
  simp only [← Array.forIn_toList]
  constructor
  · intro h
    obtain ⟨u, hnodes, h⟩ := bind_eq_ok.1 h
    obtain ⟨v, houts, -⟩ := bind_eq_ok.1 h
    refine ⟨fun i hi => ?_, fun r hr => ?_⟩
    · have hstep := forIn_eq_ok ?_ hnodes i (List.mem_range.2 hi)
      · simp only [Array.getElem?_eq_getElem hi] at hstep
        obtain ⟨w, hcheck, -⟩ := bind_eq_ok.1 hstep
        rw [← hrel i]
        exact Node.check_ok_iff.1 hcheck
      · intro j
        cases g.nodes[j]? with
        | none => simp [pure, Except.pure]
        | some n =>
            cases hc : n.check (g.portBefore j) j <;>
              simp [hc, bind, Except.bind, pure, Except.pure]
    · have hstep := forIn_eq_ok ?_ houts r (Array.mem_toList_iff.2 hr)
      · by_cases hs : (g.portBefore g.nodes.size r).isSome = true
        · obtain ⟨q, hq⟩ := Option.isSome_iff_exists.1 hs
          exact ⟨q, portBefore_eq_some_iff.1 hq⟩
        · simp [hs, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hstep
      · intro x
        by_cases hs : (g.portBefore g.nodes.size x).isSome = true
        · simp [hs, pure, Except.pure]
        · simp [hs, throw, throwThe, MonadExceptOf.throw, bind, Except.bind]
  · rintro ⟨hnodes, houts⟩
    refine bind_eq_ok.2 ⟨PUnit.unit, forIn_eq_ok_of fun i hi => ?_, ?_⟩
    · have hi := List.mem_range.1 hi
      have hcheck : g.nodes[i].check (g.portBefore i) i = .ok () := by
        refine Node.check_ok_iff.2 ?_
        rw [hrel i]
        exact hnodes i hi
      simp [Array.getElem?_eq_getElem hi, hcheck, bind, Except.bind, pure, Except.pure]
    · refine bind_eq_ok.2 ⟨PUnit.unit, forIn_eq_ok_of fun r hr => ?_, rfl⟩
      obtain ⟨q, hq⟩ := houts r (Array.mem_toList_iff.1 hr)
      simp [portBefore_eq_some_iff.2 hq, pure, Except.pure]

end TorchLean.IR
