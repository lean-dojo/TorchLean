/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Program
import FloatLib.Numerics.Executable.Conversion.Proof

/-!
# Typed programs as operator graphs

A program whose steps are all operator-graph nodes, with one scalar or none, has an operator
graph (`IR.Program.graph?`). Once the program and its payload pass their checks, it runs on its
arguments as its graph. At a scalar `d` with arithmetic, every step's outputs are the graph's at
`d` with the payload's constants and registered custom programs (`Payload.graph`), and no flag is
raised (`run_graph?_ok`). Without a scalar, every step's outputs are the graph's at any scalar
type, as the same Boolean, integer and index values (`run_graph?_none`). Neither theorem assumes
that every operator has a rule: a step that fails fails in both.
-/

@[expose] public section

namespace TorchLean

open Port
open FloatLib.Floats.ExecFloat (ConversionStatus)

namespace IR.Program

open Values (mapM_ofErased_cons)

variable {p : Program} {R : Machine} {P : Payload R} {args : Array (Value R)}

/-- A value written at a scalar and read back at that scalar is unchanged. -/
theorem Value.toOp_ofOp (d : Option DType)
    (w : @Op.Value (R.CarrierAt d) (R.storageAt d)) :
    (Value.ofOp d w).toOp d = .ok w := by
  rcases w with ⟨⟨dt, s⟩, x⟩
  cases dt <;> simp [Value.ofOp, Value.toOp, pure, Except.pure]

/-- A value written at a scalar reads without one exactly where it is a Boolean, integer or index
value, as the same value. -/
theorem Value.toOp_ofOp_none (d : DType)
    (w : @Op.Value (R.CarrierAt (some d)) (R.storageAt (some d)))
    (z : @Op.Value (R.CarrierAt none) (R.storageAt none)) :
    (Value.ofOp (some d) w).toOp none = .ok z ↔ Op.NativeRel z w := by
  rcases w with ⟨⟨dt, s⟩, x⟩
  cases dt <;>
    simp only [Value.ofOp, Value.toOp, pure, Except.pure, throw, throwThe, MonadExceptOf.throw,
      reduceCtorEq, Except.ok.injEq, false_iff, reduceDIte]
  all_goals first
    | (intro h; cases h)
    | (constructor
       · rintro rfl; constructor
       · intro h; cases h; rfl)

/-- Related Boolean, integer and index values are the same program value. -/
theorem Value.ofOp_native {e e' : Option DType}
    {z : @Op.Value (R.CarrierAt e) (R.storageAt e)}
    {w : @Op.Value (R.CarrierAt e') (R.storageAt e')} (h : Op.NativeRel z w) :
    Value.ofOp e z = Value.ofOp e' w := by
  cases h <;> rfl

/-- Related lists of Boolean, integer and index values are the same program values. -/
theorem Value.map_ofOp_native {e e' : Option DType}
    {zs : List (@Op.Value (R.CarrierAt e) (R.storageAt e))}
    {ws : List (@Op.Value (R.CarrierAt e') (R.storageAt e'))}
    (h : List.Forall₂ Op.NativeRel zs ws) :
    zs.map (Value.ofOp e) = ws.map (Value.ofOp e') := by
  induction h with
  | nil => rfl
  | cons hzw _ ih => simp [Value.ofOp_native hzw, ih]

/-- A value read at a scalar is at the value's port. -/
theorem Value.toOp_port {v : Value R} {e : Option DType}
    {z : @Op.Value (R.CarrierAt e) (R.storageAt e)} (h : v.toOp e = .ok z) :
    z.1 = v.typed.port := by
  rcases v with ⟨⟨⟨dt, s⟩, sc⟩, t⟩
  cases dt <;> simp only [Value.toOp, pure, Except.pure, Except.ok.injEq] at h
  all_goals first
    | (subst h; rfl)
    | (split at h
       · rename_i hc
         subst hc
         cases h
         rfl
       · cases h)

/-- A Boolean, integer or index value reads at every scalar as the same value. -/
theorem Value.toOp_native {v : Value R} (hv : v.typed.port.kind.isScalar = false)
    {e e' : Option DType} {z : @Op.Value (R.CarrierAt e) (R.storageAt e)}
    (h : v.toOp e = .ok z) :
    ∃ z', v.toOp e' = .ok z' ∧ Op.NativeRel z z' := by
  rcases v with ⟨⟨⟨dt, s⟩, sc⟩, t⟩
  cases dt <;> simp [Kind.isScalar] at hv <;>
    simp only [Value.toOp, pure, Except.pure, Except.ok.injEq] at h ⊢ <;> subst h <;>
    exact ⟨_, rfl, by constructor⟩

/-- A value read at a Boolean, integer or index port at one scalar reads there at any scalar as
the same tensor. -/
theorem Value.read_native {a : Value R} {port : Port} (hp : port.kind.isScalar = false)
    {e e' : Option DType} {x : @Port.Carrier (R.CarrierAt e) (R.storageAt e) port}
    (h : (a.toOp e >>= @Port.read (R.CarrierAt e) (R.storageAt e) port) = .ok x) :
    ∃ y, (a.toOp e' >>= @Port.read (R.CarrierAt e') (R.storageAt e') port) = .ok y ∧
      Op.NativeRel (⟨port, x⟩ : @Op.Value (R.CarrierAt e) (R.storageAt e)) ⟨port, y⟩ := by
  obtain ⟨z, hz, hx⟩ := bind_eq_ok.1 h
  have hap : a.typed.port.kind.isScalar = false := by
    rw [← Value.toOp_port hz, Port.fst_eq_of_read hp hx]
    exact hp
  obtain ⟨z', hz', hzz⟩ := Value.toOp_native hap (e' := e') hz
  obtain ⟨y, hy, hxy⟩ := Port.read_native hzz port hx
  exact ⟨y, bind_eq_ok.2 ⟨z', hz', hy⟩, hxy⟩

/-- Boolean, integer and index values written at a scalar read without one as the same values. -/
private theorem Value.mapM_toOp_ofOp_none (d : DType) :
    ∀ {ws : List (@Op.Value (R.CarrierAt (some d)) (R.storageAt (some d)))},
      (∀ w ∈ ws, w.1.kind.isScalar = false) →
      ∃ zs, List.Forall₂ (fun w z => (Value.ofOp (some d) w).toOp none = .ok z) ws zs ∧
        List.Forall₂ Op.NativeRel zs ws
  | [], _ => ⟨[], .nil, .nil⟩
  | ⟨⟨dt, s⟩, x⟩ :: ws, h => by
      obtain ⟨zs, hzs, hr⟩ := Value.mapM_toOp_ofOp_none d (ws := ws)
        (fun v hv => h v (by simp [hv]))
      have hw := h ⟨⟨dt, s⟩, x⟩ (by simp)
      cases dt <;> simp [Kind.isScalar] at hw <;>
        exact ⟨_ :: zs, .cons ((Value.toOp_ofOp_none d _ _).2 (by constructor)) hzs,
          .cons (by constructor) hr⟩

/-- The operator-graph node of a step that is one, and none for a conversion or a call. -/
def Node.node? : Node → Option IR.Node
  | .node n => some n
  | _ => none

/-- Steps that are all operator-graph nodes are those nodes. -/
theorem Node.mapM_node? : ∀ {l : List Node} {ns : List IR.Node},
    l.mapM Node.node? = some ns → l = ns.map .node
  | [], ns, h => by simp at h; subst h; rfl
  | .node n :: l, ns, h => by
      simp only [List.mapM_cons, Node.node?] at h
      cases hl : l.mapM Node.node? with
      | none => simp [hl] at h
      | some ns' =>
          simp only [hl] at h
          cases h
          simp [Node.mapM_node? hl]
  | .cast .. :: _, _, h => by simp [List.mapM_cons, Node.node?] at h
  | .call .. :: _, _, h => by simp [List.mapM_cons, Node.node?] at h

/-- The operator graph of a program whose steps are all operator-graph nodes and whose scalars
other than none agree, paired with that scalar, or with none where every scalar is none. -/
def graph? (p : Program) : Option (Graph × Option DType) := do
  let nodes ← p.nodes.toList.mapM Node.node?
  let scalars := p.scalars.toList.filterMap id
  if scalars.all (some · == scalars.head?) then pure (⟨nodes.toArray, p.outputs⟩, scalars.head?)
  else none

/-- A program with a graph has the graph's nodes as its steps and its outputs, and every scalar
it has is the graph's. -/
theorem graph?_eq_some {g : Graph} {sd : Option DType} (h : p.graph? = some (g, sd)) :
    p.nodes = g.nodes.map .node ∧ g.outputs = p.outputs ∧
      ∀ i e, p.scalar i = some e → sd = some e := by
  unfold graph? at h
  cases hn : p.nodes.toList.mapM Node.node? with
  | none => simp [hn] at h
  | some ns =>
      simp only [hn, Option.bind_eq_bind, Option.bind_some] at h
      split at h
      · rename_i hall
        simp only [pure, Option.some.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        refine ⟨?_, rfl, fun i e he => ?_⟩
        · apply Array.ext'
          simp [Node.mapM_node? hn]
        · have hmem : e ∈ p.scalars.toList.filterMap id := by
            simp only [scalar, Option.join_eq_some_iff] at he
            exact List.mem_filterMap.2
              ⟨some e, Array.mem_toList_iff.2 (Array.mem_of_getElem? he), rfl⟩
          exact (beq_iff_eq.1 (List.all_eq_true.1 hall e hmem)).symm
      · cases h

/-- The payload of a program's operator graph at the scalar `d`: the constants read at `d` and
the registered custom programs at `d`. -/
def Payload.graph (P : Payload R) (d : DType) : @IR.Payload (R.Carrier d) (R.storage d) :=
  letI := R.storage d
  { const? := fun i => P.consts[i]?.join.bind fun v => (v.toOp (some d)).toOption
    custom? := fun i => P.customs[i]?.join.bind fun c =>
      if h : c.scalar = d then some (h ▸ c.program) else none }

/-- A checked program passes the scalar check and the step check at every step. -/
theorem check_step (h : p.check R = .ok ()) {i : Nat} {n : Node} (hn : p.nodes[i]? = some n) :
    p.checkScalar R i n = .ok () ∧ p.checkNode R i n = .ok () := by
  have hi : i < p.nodes.size := by
    by_contra hc
    rw [Array.getElem?_eq_none (Nat.le_of_not_lt hc)] at hn
    cases hn
  unfold check at h
  dsimp only at h
  split at h
  · obtain ⟨_, h₁, h⟩ := bind_eq_ok.1 h
    obtain ⟨_, h₂, -⟩ := bind_eq_ok.1 h
    have k₁ := forIn_eq_ok ?_ h₁ i (List.mem_range.2 hi)
    · have k₂ := forIn_eq_ok ?_ h₂ i (List.mem_range.2 hi)
      · simp only [hn] at k₁ k₂
        obtain ⟨_, k₁, -⟩ := bind_eq_ok.1 k₁
        obtain ⟨_, k₂, -⟩ := bind_eq_ok.1 k₂
        exact ⟨k₁, k₂⟩
      · intro b
        split
        · intro hb
          obtain ⟨_, -, hb⟩ := bind_eq_ok.1 hb
          cases hb
        · intro hb
          cases hb
    · intro b
      split
      · intro hb
        obtain ⟨_, -, hb⟩ := bind_eq_ok.1 hb
        split at hb
        · cases hb
        · obtain ⟨_, -, hb⟩ := bind_eq_ok.1 hb
          cases hb
      · intro hb
        cases hb
  · obtain ⟨_, h, -⟩ := bind_eq_ok.1 h
    cases h

/-- The scalar check of a step without a scalar: its ports are Boolean, integer or index ports. -/
theorem checkScalar_none {i : Nat} {n : Node} (h : p.checkScalar R i n = .ok ())
    (hs : p.scalar i = none) : n.ports.all (fun q => !q.kind.isScalar) := by
  unfold checkScalar at h
  rw [hs] at h
  cases hany : n.ports.any (fun q => q.kind.isScalar) with
  | false => simpa using hany
  | true =>
      rw [hany] at h
      simp [throw, throwThe, MonadExceptOf.throw] at h

/-- The step check of an operator node: one reference per input port. -/
theorem checkNode_op {i : Nat} {o : Operator} {refs : Array Ref}
    (h : p.checkNode R i (.node (.op o refs)) = .ok ()) : refs.size = o.1.length := by
  rcases o with ⟨ins, outs, op⟩
  obtain ⟨_, h, -⟩ := bind_eq_ok.1 h
  unfold IR.Node.check at h
  by_contra hne
  simp [hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- An operator whose ports are all Boolean, integer or index ports is not a custom operation,
whose output is a value. -/
private theorem not_custom_of_native {ins outs : List Port} {op : Op ins outs}
    (hnat : (ins ++ outs).all (fun q => !q.kind.isScalar)) :
    ∀ (name : String) (shapes : List Shape) (output : Shape), ins = shapes.map Port.value →
      HEq (⟨outs, op⟩ : Σ outs, Op ins outs)
        (⟨[Port.value output], Op.custom name shapes output⟩ :
          Σ outs, Op (shapes.map Port.value) outs) → False := by
  intro name shapes output hins hh
  subst hins
  obtain ⟨rfl, -⟩ := Sigma.mk.inj_iff.1 (eq_of_heq hh)
  simp [Kind.isScalar] at hnat

/-- The run of a checked program with a checked payload and its arguments is its fold over the
steps. -/
theorem run_eq_foldlM (hcheck : p.check R = .ok ()) (hP : P.check p = .ok ())
    (hsize : args.size = p.arity) :
    p.run R P args = p.nodes.toList.foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (#[], {}) := by
  simp only [run, hcheck, hP, hsize]
  rfl

/-- Running steps from the flags `f₀` merges `f₀` into the flags of running them from none: the
outputs do not depend on the flags. -/
theorem foldlM_step_merge (l : List Node) (earlier : Array (List (Value R)))
    (f₀ : ConversionStatus) :
    l.foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (earlier, f₀) =
      (fun r => (r.1, f₀.merge r.2)) <$> l.foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (earlier, {}) := by
  induction l generalizing earlier f₀ with
  | nil => simp [ConversionStatus.merge_empty]
  | cons n l ih =>
    simp only [List.foldlM_cons]
    cases hs : p.step R P args earlier earlier.size n with
    | error e => rfl
    | ok r =>
      obtain ⟨vs, raised⟩ := r
      simp only [bind, Except.bind, pure, Except.pure] at ih ⊢
      rw [ih, ih (f₀ := ConversionStatus.merge {} raised), Functor.map_map]
      congr 1
      funext r
      simp [ConversionStatus.merge_assoc]

variable {β : Type} [Storage β] [Context β] [Op.Capabilities β]

/-- The storage forward of an operator that has one succeeds exactly when its attribute check, the
reading of its inputs at its ports and the forward succeed, and returns the forward's outputs. -/
theorem runStorage_eq_ok {R : Machine} {i : Nat} {d : Option DType} {ins outs : List Port}
    {o : Op ins outs}
    {f : ∀ (α : Type) [Storage α], Values α ins → Except String (Values α outs)}
    (hf : o.storageForward? = some f)
    {inputs outputs : List (@Op.Value (R.CarrierAt d) (R.storageAt d))} :
    runStorage R i d o inputs = .ok outputs ↔
      ∃ xs ys, o.check = .ok () ∧ @Values.ofErased _ (R.storageAt d) ins inputs = .ok xs ∧
        @f _ (R.storageAt d) xs = .ok ys ∧
          outputs = @Values.toErased _ (R.storageAt d) outs ys := by
  unfold runStorage
  simp only [hf]
  cases hc : o.check with
  | error err => simp [bind, Except.bind]
  | ok u =>
    cases hx : @Values.ofErased _ (R.storageAt d) ins inputs with
    | error err => simp [bind, Except.bind]
    | ok xs =>
      cases hy : @f _ (R.storageAt d) xs with
      | error err => simp [bind, Except.bind, hy]
      | ok ys => simp [bind, Except.bind, hy, pure, Except.pure, eq_comm]

/-- The storage forward of an operator without a scalar on the inputs `xp` and its invocation on
the inputs `xg` both fail or both succeed with related outputs, when the inputs correspond. -/
private theorem runStorage_invoke {i : Nat} {ins outs : List Port} (o : Op ins outs)
    (hnat : (ins ++ outs).all (fun p => !p.kind.isScalar))
    {xp : Except String (List (@Op.Value (R.CarrierAt none) (R.storageAt none)))}
    {xg : Except String (List (Op.Value β))}
    (h₁ : ∀ zs, xp = .ok zs → ∃ ws, xg = .ok ws ∧ List.Forall₂ Op.NativeRel zs ws)
    (h₂ : ∀ ws, xg = .ok ws → (∀ w ∈ ws, w.1.kind.isScalar = false) →
      ∃ zs, xp = .ok zs ∧ List.Forall₂ Op.NativeRel zs ws) :
    (∀ rs, (xp >>= runStorage R i none o) = .ok rs →
      ∃ rs', o.invoke xg = .ok rs' ∧ List.Forall₂ Op.NativeRel rs rs') ∧
    (∀ rs', o.invoke xg = .ok rs' →
      ∃ rs, (xp >>= runStorage R i none o) = .ok rs ∧ List.Forall₂ Op.NativeRel rs rs') := by
  obtain ⟨f, hf⟩ := Option.isSome_iff_exists.1 (Op.storageForward?_isSome_of_native o hnat)
  have hrun : ∀ zs rs, runStorage R i none o zs = .ok rs ↔ ∃ xs ys, o.check = .ok () ∧
      Values.ofErased ins zs = .ok xs ∧ f _ xs = .ok ys ∧ rs = Values.toErased outs ys :=
    fun _ _ => runStorage_eq_ok hf
  refine ⟨fun rs h => ?_, fun rs' h => ?_⟩
  · obtain ⟨zs, hzs, h⟩ := bind_eq_ok.1 h
    obtain ⟨ws, hws, hzw⟩ := h₁ zs hzs
    obtain ⟨xs, ys, hc, hx, hy, rfl⟩ := (hrun zs rs).1 h
    obtain ⟨xs', hx', hxx⟩ := Values.ofErased_native ins hzw hx
    obtain ⟨ys', hy', hyy⟩ := Op.storageForward_native o hf hxx hy
    refine ⟨_, ?_, hyy⟩
    rw [hws]
    exact (Op.invoke_storageForward o hf).2 ⟨xs', ys', hc, hx', hy', rfl⟩
  · obtain ⟨ws, -, -, -, hws, -⟩ := Op.invoke_eq_ok h
    rw [hws] at h
    obtain ⟨xs', ys', hc, hx', hy', rfl⟩ := (Op.invoke_storageForward o hf).1 h
    have hins : ins.all (fun p => !p.kind.isScalar) := by
      rw [List.all_append, Bool.and_eq_true] at hnat
      exact hnat.1
    obtain ⟨zs, hzs, hzw⟩ := h₂ ws hws (Values.native_of_ofErased ins hins hx')
    obtain ⟨xs, hx, hxx⟩ := Values.ofErased_native ins (Op.NativeRel.forall₂_symm hzw) hx'
    obtain ⟨ys, hy, hyy⟩ := Op.storageForward_native o hf hxx hy'
    exact ⟨_, bind_eq_ok.2 ⟨zs, hzs, (hrun zs _).2 ⟨xs, ys, hc, hx, hy, rfl⟩⟩,
      Op.NativeRel.forall₂_symm hyy⟩

/-- A program whose steps are the nodes `l` runs as the graph run of `l` when each step does: from
outputs `Z.map (·.map F)` related row by row to the graph's outputs, each step and its node both
succeed, with related outputs and no flags, or both fail. -/
private theorem run_nodes {X : Type} (F : X → Value R) (rel : X → Op.Value β → Prop)
    (GP : IR.Payload β) (gargs : List (Op.Value β)) :
    ∀ (l : List IR.Node) (Z : Array (List X)) (W : Array (List (Op.Value β)))
      (fl : ConversionStatus), List.Forall₂ (List.Forall₂ rel) Z.toList W.toList →
    (∀ j n, l[j]? = some n → ∀ (Z' : Array (List X)) (W' : Array (List (Op.Value β))),
      Z'.size = Z.size + j → List.Forall₂ (List.Forall₂ rel) Z'.toList W'.toList →
      (∀ vs fl', p.step R P args (Z'.map (·.map F)) (Z.size + j) (.node n) = .ok (vs, fl') →
        ∃ xs ws, n.eval GP gargs W' (Z.size + j) = .ok ws ∧ List.Forall₂ rel xs ws ∧
          vs = xs.map F ∧ fl' = {}) ∧
      (∀ ws, n.eval GP gargs W' (Z.size + j) = .ok ws →
        ∃ xs, p.step R P args (Z'.map (·.map F)) (Z.size + j) (.node n) = .ok (xs.map F, {}) ∧
          List.Forall₂ rel xs ws)) →
    (∀ E fl', (l.map Node.node).foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (Z.map (·.map F), fl) = .ok (E, fl') →
      ∃ (Z' : Array (List X)) (W' : Array (List (Op.Value β))),
        runNodes GP gargs W l = .ok W' ∧
        List.Forall₂ (List.Forall₂ rel) Z'.toList W'.toList ∧ E = Z'.map (·.map F) ∧ fl' = fl) ∧
    (∀ W', runNodes GP gargs W l = .ok W' →
      ∃ Z' : Array (List X), (l.map Node.node).foldlM (fun (earlier, flags) n => do
        let (vs, raised) ← p.step R P args earlier earlier.size n
        pure (earlier.push vs, flags.merge raised)) (Z.map (·.map F), fl) =
          .ok (Z'.map (·.map F), fl) ∧ List.Forall₂ (List.Forall₂ rel) Z'.toList W'.toList)
  | [], Z, W, fl, hrel, _ => by
      refine ⟨fun E fl' h => ?_, fun W' h => ?_⟩
      · simp only [List.map_nil, List.foldlM_nil, pure, Except.pure, Except.ok.injEq,
          Prod.mk.injEq] at h
        exact ⟨Z, W, rfl, hrel, h.1.symm, h.2.symm⟩
      · simp only [runNodes_nil, Except.ok.injEq] at h
        subst h
        exact ⟨Z, rfl, hrel⟩
  | n :: l, Z, W, fl, hrel, hstep => by
      have hstep₀ := hstep 0 n rfl Z W rfl hrel
      simp only [Nat.add_zero] at hstep₀
      have hsize : (Z.map (·.map F)).size = Z.size := Array.size_map ..
      have hW : W.size = Z.size := by simpa using hrel.length_eq.symm
      have ih (xs : List X) (ws : List (Op.Value β)) (hx : List.Forall₂ rel xs ws) :=
        run_nodes F rel GP gargs l (Z.push xs) (W.push ws) fl
          (by simpa only [Array.toList_push] using List.rel_append hrel (.cons hx .nil))
          (fun j m hm Z' W' hZ' hr => by
            have := hstep (j + 1) m (by simpa using hm) Z' W' (by simp at hZ' ⊢; omega) hr
            simpa [Nat.add_assoc, Nat.add_comm 1 j] using this)
      refine ⟨fun E fl' h => ?_, fun W' h => ?_⟩
      · rw [List.map_cons, List.foldlM_cons] at h
        obtain ⟨⟨E₁, fl₁⟩, h₁, h⟩ := bind_eq_ok.1 h
        obtain ⟨⟨vs, raised⟩, hv, h₁⟩ := bind_eq_ok.1 h₁
        rw [hsize] at hv
        obtain ⟨xs, ws, hw, hx, rfl, rfl⟩ := hstep₀.1 vs raised hv
        cases h₁
        rw [ConversionStatus.merge_empty, ← Array.map_push] at h
        obtain ⟨Z', W', hW', hr, rfl, rfl⟩ := (ih xs ws hx).1 E fl' h
        refine ⟨Z', W', ?_, hr, rfl, rfl⟩
        rw [runNodes_cons, hW, hw]
        exact hW'
      · rw [runNodes_cons, hW] at h
        obtain ⟨ws, hw, h⟩ := bind_eq_ok.1 h
        obtain ⟨xs, hv, hx⟩ := hstep₀.2 ws hw
        obtain ⟨Z', hZ', hr⟩ := (ih xs ws hx).2 W' h
        refine ⟨Z', ?_, hr⟩
        rw [List.map_cons, List.foldlM_cons]
        refine bind_eq_ok.2 ⟨((Z.push xs).map (·.map F), fl), ?_, hZ'⟩
        refine bind_eq_ok.2 ⟨(xs.map F, {}), by rw [hsize]; exact hv, ?_⟩
        simp [pure, Except.pure, ConversionStatus.merge_empty, Array.map_push]

section Common

variable {d : DType} {ctx : Context (R.Carrier d)}

/-- At the common scalar `d`, a step whose scalar is `d`, or none with Boolean, integer and index
ports only, runs as its node in the graph at `d`: both fail, or both succeed with the node's outputs
at `d` and no flags. -/
private theorem step_some (hctx : R.arithmetic d = some ctx)
    {args' : List (@Op.Value (R.Carrier d) (R.storage d))}
    (hargs : List.Forall₂ (fun a v => a.toOp (some d) = .ok v) args.toList args')
    {i : Nat} {n : IR.Node}
    (hs : p.scalar i = some d ∨
      (p.scalar i = none ∧ (Node.node n).ports.all (fun q => !q.kind.isScalar)))
    (harity : ∀ o refs, n = .op o refs → refs.size = o.1.length)
    (W : Array (List (@Op.Value (R.Carrier d) (R.storage d))))
    (vs : List (Value R)) (fl : ConversionStatus) :
    p.step R P args (W.map (·.map (Value.ofOp (some d)))) i (.node n) = .ok (vs, fl) ↔
      ∃ ws, @IR.Node.eval (R.Carrier d) (R.storage d) ctx (R.capabilities d) (P.graph d)
          args' W i n = .ok ws ∧ vs = ws.map (Value.ofOp (some d)) ∧ fl = {} := by
  cases n with
  | input k port =>
      -- Both sides read argument `k`: converted at `d` when the step's scalar is `d`, read natively
      -- otherwise, where `Value.read_native` matches the two readings.
      have hk := forall₂_getElem? hargs k
      rw [Array.getElem?_toList] at hk
      simp only [step, IR.Node.eval]
      rcases hs with hs | ⟨hs, hp⟩
      · rw [hs]
        generalize args[k]? = oa at hk ⊢
        generalize args'[k]? = ov at hk ⊢
        cases hk with
        | none => simp [throw, throwThe, MonadExceptOf.throw]
        | some ha =>
            rename_i a v
            constructor
            · intro h
              obtain ⟨v', hv', h⟩ := bind_eq_ok.1 h
              rw [ha] at hv'
              cases hv'
              obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
              cases h
              exact ⟨_, bind_eq_ok.2 ⟨x, hx, rfl⟩, rfl, rfl⟩
            · rintro ⟨ws, h, rfl, rfl⟩
              obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
              cases h
              exact bind_eq_ok.2 ⟨_, ha, bind_eq_ok.2 ⟨_, hx, rfl⟩⟩
      · rw [hs]
        have hp' : port.kind.isScalar = false := by simpa [Node.ports] using hp
        generalize args[k]? = oa at hk ⊢
        generalize args'[k]? = ov at hk ⊢
        cases hk with
        | none => simp [throw, throwThe, MonadExceptOf.throw]
        | some ha =>
            rename_i a v
            constructor
            · intro h
              obtain ⟨z, hz, h⟩ := bind_eq_ok.1 h
              obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
              cases h
              obtain ⟨y, hy, hxy⟩ :=
                Value.read_native hp' (e' := some d) (bind_eq_ok.2 ⟨z, hz, hx⟩)
              obtain ⟨v', hv', hy⟩ := bind_eq_ok.1 hy
              rw [ha] at hv'
              cases hv'
              exact ⟨_, bind_eq_ok.2 ⟨y, hy, rfl⟩, by simp [Value.ofOp_native hxy], rfl⟩
            · rintro ⟨ws, h, rfl, rfl⟩
              obtain ⟨y, hy, h⟩ := bind_eq_ok.1 h
              cases h
              obtain ⟨x, hx, hyx⟩ :=
                Value.read_native hp' (e' := none) (bind_eq_ok.2 ⟨_, ha, hy⟩)
              obtain ⟨z, hz, hx⟩ := bind_eq_ok.1 hx
              refine bind_eq_ok.2 ⟨_, hz, bind_eq_ok.2 ⟨_, hx, ?_⟩⟩
              simp [Value.ofOp_native hyx, pure, Except.pure]
  | const port =>
      -- The same two cases for the payload's constant at `i`.
      simp only [step, IR.Node.eval, Payload.graph]
      rcases hs with hs | ⟨hs, hp⟩
      · rw [hs]
        cases P.consts[i]?.join with
        | none => simp [throw, throwThe, MonadExceptOf.throw]
        | some a =>
            cases ha : a.toOp (some d) with
            | error e =>
                simp only [Option.bind_some, ha, Except.toOption]
                constructor
                · intro h
                  obtain ⟨_, h, -⟩ := bind_eq_ok.1 h
                  cases h
                · rintro ⟨ws, h, -⟩
                  cases h
            | ok v =>
                simp only [Option.bind_some, ha, Except.toOption]
                constructor
                · intro h
                  obtain ⟨v', hv', h⟩ := bind_eq_ok.1 h
                  cases hv'
                  obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
                  cases h
                  exact ⟨_, bind_eq_ok.2 ⟨_, hx, rfl⟩, rfl, rfl⟩
                · rintro ⟨ws, h, rfl, rfl⟩
                  obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
                  cases h
                  exact bind_eq_ok.2 ⟨_, rfl, bind_eq_ok.2 ⟨_, hx, rfl⟩⟩
      · rw [hs]
        have hp' : port.kind.isScalar = false := by simpa [Node.ports] using hp
        cases P.consts[i]?.join with
        | none => simp [throw, throwThe, MonadExceptOf.throw]
        | some a =>
            simp only [Option.bind_some]
            constructor
            · intro h
              obtain ⟨z, hz, h⟩ := bind_eq_ok.1 h
              obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
              cases h
              obtain ⟨y, hy, hxy⟩ :=
                Value.read_native hp' (e' := some d) (bind_eq_ok.2 ⟨z, hz, hx⟩)
              obtain ⟨v, hv, hy⟩ := bind_eq_ok.1 hy
              rw [hv]
              exact ⟨_, bind_eq_ok.2 ⟨y, hy, rfl⟩, by simp [Value.ofOp_native hxy], rfl⟩
            · rintro ⟨ws, h, rfl, rfl⟩
              cases ha : a.toOp (some d) with
              | error e =>
                  simp only [ha, Except.toOption] at h
                  cases h
              | ok v =>
                  simp only [ha, Except.toOption] at h
                  obtain ⟨y, hy, h⟩ := bind_eq_ok.1 h
                  cases h
                  obtain ⟨x, hx, hyx⟩ :=
                    Value.read_native hp' (e' := none) (bind_eq_ok.2 ⟨_, ha, hy⟩)
                  obtain ⟨z, hz, hx⟩ := bind_eq_ok.1 hx
                  refine bind_eq_ok.2 ⟨_, hz, bind_eq_ok.2 ⟨_, hx, ?_⟩⟩
                  simp [Value.ofOp_native hyx, pure, Except.pure]
  | op o refs =>
      rcases o with ⟨ins, outs, op⟩
      let _ : Storage (R.Carrier d) := R.storage d
      let _ : Context (R.Carrier d) := ctx
      let _ : Op.Capabilities (R.Carrier d) := R.capabilities d
      -- Reading a reference of the converted environment and converting back reads `W`.
      have hread : ∀ r : Ref,
          (do let v ← r.read (W.map (·.map (Value.ofOp (some d)))); v.toOp (some d)) =
            r.read W := by
        intro r
        rw [Ref.read_map]
        cases r.read W with
        | error e => rfl
        | ok w => exact Value.toOp_ofOp (some d) w
      simp only [step, IR.Node.eval]
      rcases hs with hs | ⟨hs, hp⟩
      · rw [hs]
        by_cases hc : ∃ (name : String) (shapes : List Shape) (output : Shape),
            ins = shapes.map Port.value ∧
            HEq (⟨outs, op⟩ : Σ outs, Op ins outs)
              (⟨[Port.value output], Op.custom name shapes output⟩ :
                Σ outs, Op (shapes.map Port.value) outs)
        -- A custom operator runs the payload's function at `i`, which must be at scalar `d`.
        · obtain ⟨name, shapes, output, rfl, hh⟩ := hc
          obtain ⟨rfl, hop⟩ := Sigma.mk.inj_iff.1 (eq_of_heq hh)
          cases hop
          have harity' : refs.size = shapes.length := by
            simpa using harity _ refs rfl
          have hread' : ∀ r : Ref,
              (do
                let v ← r.read (W.map (·.map (Value.ofOp (some d))))
                let w ← v.toOp (some d)
                someTensor? w) = (do let w ← r.read W; someTensor? w) := by
            intro r
            rw [← hread r, bind_assoc]
          rw [runOp.eq_1, evalOp.eq_1]
          simp only [runCustom, Payload.graph, harity', hread', ↓reduceIte]
          cases P.customs[i]?.join with
          | none => simp [throw, throwThe, MonadExceptOf.throw]
          | some c =>
              by_cases h : c.scalar = d
              · subst h
                simp only [Option.bind_some, dite_true]
                constructor
                · intro h₁
                  obtain ⟨rs, h₂, h₃⟩ := bind_eq_ok.1 h₁
                  cases h₃
                  obtain ⟨vals, hm, h₄⟩ := bind_eq_ok.1 h₂
                  obtain ⟨x, hx, h₅⟩ := bind_eq_ok.1 h₄
                  cases h₅
                  exact ⟨_, bind_eq_ok.2 ⟨x, bind_eq_ok.2 ⟨vals, hm, hx⟩, rfl⟩,
                    rfl, rfl⟩
                · rintro ⟨ws, h₁, rfl, rfl⟩
                  obtain ⟨x, h₂, h₃⟩ := bind_eq_ok.1 h₁
                  cases h₃
                  obtain ⟨vals, hm, hx⟩ := bind_eq_ok.1 h₂
                  exact bind_eq_ok.2 ⟨_, bind_eq_ok.2 ⟨vals, hm,
                    bind_eq_ok.2 ⟨x, hx, rfl⟩⟩, rfl⟩
              · simp [h, throw, throwThe, MonadExceptOf.throw]
        -- Any other operator invokes the same rule on the same values on both sides.
        · have hc' : ∀ (name : String) (shapes : List Shape) (output : Shape),
              ins = shapes.map Port.value →
              HEq (⟨outs, op⟩ : Σ outs, Op ins outs)
                (⟨[Port.value output], Op.custom name shapes output⟩ :
                  Σ outs, Op (shapes.map Port.value) outs) → False :=
            fun name shapes output h₁ h₂ => hc ⟨name, shapes, output, h₁, h₂⟩
          rw [runOp.eq_2 _ _ _ _ _ _ _ _ _ hc', evalOp.eq_2 _ _ _ _ _ _ _ hc']
          simp only [hread, hctx]
          constructor
          · intro h₁
            obtain ⟨rs, h₂, h₃⟩ := bind_eq_ok.1 h₁
            cases h₃
            obtain ⟨ws, hws, h₄⟩ := bind_eq_ok.1 h₂
            split at h₄
            · cases h₄
            · refine ⟨rs, ?_, rfl, rfl⟩
              rw [hws]
              exact h₄
          · rintro ⟨ws, h₁, rfl, rfl⟩
            obtain ⟨values, xs, rule, -, hi, -, hs', -, -⟩ := Op.invoke_eq_ok h₁
            refine bind_eq_ok.2 ⟨ws, bind_eq_ok.2 ⟨values, hi, ?_⟩, rfl⟩
            rw [hs']
            rw [hi] at h₁
            exact h₁
      -- With no scalar port the operator runs on storage values (`runStorage_invoke`).
      · rw [hs]
        have hnat : (ins ++ outs).all (fun q => !q.kind.isScalar) := by
          simpa [Node.ports] using hp
        rw [runOp.eq_3, evalOp.eq_2 _ _ _ _ _ _ _ (not_custom_of_native hnat)]
        have hread₀ : ∀ r : Ref,
            (do let v ← r.read (W.map (·.map (Value.ofOp (some d)))); v.toOp none) =
              (r.read W >>= fun w => (Value.ofOp (some d) w).toOp none) := by
          intro r
          rw [Ref.read_map]
          cases r.read W <;> rfl
        simp only [hread₀]
        have key := runStorage_invoke (R := R) (i := i) op hnat
          (xp := refs.toList.mapM fun r => r.read W >>= fun w => (Value.ofOp (some d) w).toOp none)
          (xg := refs.toList.mapM (·.read W))
          (fun zs h => by
            obtain ⟨ws, hws, hr⟩ := mapM_bind_eq_ok.1 h
            refine ⟨ws, hws, ?_⟩
            exact (hr.imp fun w z h => (Value.toOp_ofOp_none d w z).1 h).flip)
          (fun ws hws hnw => by
            obtain ⟨zs, hzs, hr⟩ := Value.mapM_toOp_ofOp_none d hnw
            exact ⟨zs, mapM_bind_eq_ok.2 ⟨ws, hws, hzs⟩, hr⟩)
        constructor
        · intro h₁
          obtain ⟨rs, h₂, h₃⟩ := bind_eq_ok.1 h₁
          cases h₃
          obtain ⟨rs', h₄, hr⟩ := key.1 rs h₂
          exact ⟨rs', h₄, Value.map_ofOp_native hr, rfl⟩
        · rintro ⟨ws, h₁, rfl, rfl⟩
          obtain ⟨rs, h₂, hr⟩ := key.2 ws h₁
          refine bind_eq_ok.2 ⟨rs, h₂, ?_⟩
          simp [Value.map_ofOp_native (e := none) (e' := some d) hr, pure, Except.pure]

end Common

section Native

variable {α : Type} [Storage α] [Context α] [Op.Capabilities α]

/-- A step without a scalar, whose ports are Boolean, integer and index ports, runs as its node in
a graph at any scalar type `α`: from outputs related row by row to the graph's, both fail, or both
succeed with the same Boolean, integer and index values and no flags. -/
private theorem step_none {gargs : List (Op.Value α)}
    (hargs : ∀ k : Nat, Option.Rel (fun a w => ∃ z, a.toOp none = .ok z ∧ Op.NativeRel z w)
      args[k]? gargs[k]?)
    {GP : IR.Payload α} {i : Nat} {n : IR.Node}
    (hGP : ∀ port, n = .const port → Option.Rel Op.NativeRel
      ((P.consts[i]?.join).bind fun v => (v.toOp none).toOption) (GP.const? i))
    (hs : p.scalar i = none)
    (hp : (Node.node n).ports.all (fun q => !q.kind.isScalar))
    {Z : Array (List (@Op.Value (R.CarrierAt none) (R.storageAt none)))}
    {W : Array (List (Op.Value α))}
    (hZW : List.Forall₂ (List.Forall₂ Op.NativeRel) Z.toList W.toList) :
    (∀ vs fl, p.step R P args (Z.map (·.map (Value.ofOp none))) i (.node n) = .ok (vs, fl) →
      ∃ zs ws, n.eval GP gargs W i = .ok ws ∧ List.Forall₂ Op.NativeRel zs ws ∧
        vs = zs.map (Value.ofOp none) ∧ fl = {}) ∧
    (∀ ws, n.eval GP gargs W i = .ok ws →
      ∃ zs, p.step R P args (Z.map (·.map (Value.ofOp none))) i (.node n) =
          .ok (zs.map (Value.ofOp none), {}) ∧ List.Forall₂ Op.NativeRel zs ws) := by
  cases n with
  | input k port =>
      have hk := hargs k
      simp only [step, IR.Node.eval]
      rw [hs]
      generalize args[k]? = oa at hk ⊢
      generalize gargs[k]? = ow at hk ⊢
      -- A missing argument fails both runs; a present one reads the related value.
      cases hk with
      | none =>
          exact ⟨fun _ _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h,
            fun _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h⟩
      | some ha =>
          rename_i a w
          obtain ⟨z, hz, hzw⟩ := ha
          refine ⟨fun vs fl h => ?_, fun ws h => ?_⟩
          · obtain ⟨z', hz', h⟩ := bind_eq_ok.1 h
            rw [hz] at hz'
            cases hz'
            obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
            cases h
            obtain ⟨y, hy, hxy⟩ := Port.read_native hzw port hx
            exact ⟨_, _, bind_eq_ok.2 ⟨y, hy, rfl⟩, .cons hxy .nil, rfl, rfl⟩
          · obtain ⟨y, hy, h⟩ := bind_eq_ok.1 h
            cases h
            obtain ⟨x, hx, hyx⟩ := Port.read_native hzw.symm port hy
            exact ⟨_, bind_eq_ok.2 ⟨_, hz, bind_eq_ok.2 ⟨_, hx, rfl⟩⟩,
              .cons hyx.symm .nil⟩
  | const port =>
      have hc := hGP port rfl
      simp only [step, IR.Node.eval]
      rw [hs]
      generalize GP.const? i = og at hc ⊢
      -- A missing or unconvertible constant fails both runs; otherwise both read related values.
      cases ha : P.consts[i]?.join with
      | none =>
          rw [ha] at hc
          cases hc
          exact ⟨fun _ _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h,
            fun _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h⟩
      | some a =>
          rw [ha] at hc
          simp only [Option.bind_some] at hc
          cases hz : a.toOp none with
          | error e =>
              rw [hz] at hc
              simp only [Except.toOption] at hc
              cases hc
              refine ⟨fun _ _ h => ?_,
                fun _ h => by simp [throw, throwThe, MonadExceptOf.throw] at h⟩
              obtain ⟨z', hz', -⟩ := bind_eq_ok.1 h
              rw [hz] at hz'
              cases hz'
          | ok z =>
              rw [hz] at hc
              simp only [Except.toOption] at hc
              cases hc with
              | some hzw =>
                  rename_i w
                  refine ⟨fun vs fl h => ?_, fun ws h => ?_⟩
                  · obtain ⟨z', hz', h⟩ := bind_eq_ok.1 h
                    rw [hz] at hz'
                    cases hz'
                    obtain ⟨x, hx, h⟩ := bind_eq_ok.1 h
                    cases h
                    obtain ⟨y, hy, hxy⟩ := Port.read_native hzw port hx
                    exact ⟨_, _, bind_eq_ok.2 ⟨y, hy, rfl⟩, .cons hxy .nil, rfl, rfl⟩
                  · obtain ⟨y, hy, h⟩ := bind_eq_ok.1 h
                    cases h
                    obtain ⟨x, hx, hyx⟩ := Port.read_native hzw.symm port hy
                    exact ⟨_, bind_eq_ok.2 ⟨_, hz, bind_eq_ok.2 ⟨_, hx, rfl⟩⟩,
                      .cons hyx.symm .nil⟩
  | op o refs =>
      rcases o with ⟨ins, outs, op⟩
      have hnat : (ins ++ outs).all (fun q => !q.kind.isScalar) := by
        simpa [Node.ports] using hp
      simp only [step, IR.Node.eval]
      rw [hs, runOp.eq_3, evalOp.eq_2 _ _ _ _ _ _ _ (not_custom_of_native hnat)]
      -- Reading a lifted value and converting it back is reading `Z`.
      have hread : ∀ r : Ref,
          (do let v ← r.read (Z.map (·.map (Value.ofOp none))); v.toOp none) = r.read Z := by
        intro r
        rw [Ref.read_map]
        cases r.read Z with
        | error e => rfl
        | ok z => exact Value.toOp_ofOp none z
      simp only [hread]
      -- The arguments are related, so the storage run and `invoke` succeed together, with
      -- related results.
      obtain ⟨hm₁, hm₂⟩ := Ref.mapM_read_rel hZW refs.toList
      have key := runStorage_invoke (R := R) (i := i) op hnat
        (xp := refs.toList.mapM (·.read Z)) (xg := refs.toList.mapM (·.read W)) hm₁
        (fun ws hws _ => hm₂ ws hws)
      refine ⟨fun vs fl h => ?_, fun ws h => ?_⟩
      · obtain ⟨rs, h₂, h₃⟩ := bind_eq_ok.1 h
        cases h₃
        obtain ⟨rs', h₄, hr⟩ := key.1 rs h₂
        exact ⟨rs, rs', h₄, hr, rfl, rfl⟩
      · obtain ⟨rs, h₂, hr⟩ := key.2 ws h
        exact ⟨rs, bind_eq_ok.2 ⟨rs, h₂, rfl⟩, hr⟩

end Native

/-- Outputs related by equality row by row are equal. -/
private theorem eq_of_forall₂_eq {X : Type} {Z W : Array (List X)}
    (h : List.Forall₂ (List.Forall₂ Eq) Z.toList W.toList) : Z = W := by
  simp only [List.forall₂_eq_eq_eq] at h
  exact Array.toList_inj.1 h

/-- A checked program whose steps form the graph `g` at one scalar `d` with arithmetic `ctx` runs as
`g` at `d`, with the payload's constants and registered custom programs at `d` and the arguments
read at `d`: the outputs of every step are the graph's at `d`, and no flag is raised. -/
theorem run_graph?_ok {g : Graph} {d : DType} {ctx : Context (R.Carrier d)}
    (hg : p.graph? = some (g, some d)) (hctx : R.arithmetic d = some ctx)
    (hcheck : p.check R = .ok ()) (hP : P.check p = .ok ()) (hsize : args.size = p.arity)
    {args' : List (@Op.Value (R.Carrier d) (R.storage d))}
    (hargs : args.toList.mapM (·.toOp (some d)) = .ok args')
    (vs : Array (List (Value R))) (flags : ConversionStatus) :
    p.run R P args = .ok (vs, flags) ↔
      ∃ ws, @Graph.run (R.Carrier d) (R.storage d) ctx (R.capabilities d) g (P.graph d) args' =
          .ok ws ∧ vs = ws.map (·.map (Value.ofOp (some d))) ∧ flags = {} := by
  obtain ⟨hnodes, -, hsc⟩ := graph?_eq_some hg
  have hlist : p.nodes.toList = g.nodes.toList.map Node.node := by
    rw [hnodes, Array.toList_map]
  let _ : Storage (R.Carrier d) := R.storage d
  let _ : Context (R.Carrier d) := ctx
  let _ : Op.Capabilities (R.Carrier d) := R.capabilities d
  have key := run_nodes (p := p) (P := P) (args := args) (Value.ofOp (some d)) Eq (P.graph d)
    args' g.nodes.toList #[] #[] {} .nil ?_
  · rw [run_eq_foldlM hcheck hP hsize, hlist]
    obtain ⟨h₁, h₂⟩ := key
    constructor
    · intro h
      obtain ⟨Z', W', hW', hr, rfl, rfl⟩ := h₁ vs flags (by simpa using h)
      obtain rfl := eq_of_forall₂_eq hr
      exact ⟨Z', hW', rfl, rfl⟩
    · rintro ⟨ws, hw, rfl, rfl⟩
      obtain ⟨Z', hZ', hr⟩ := h₂ ws hw
      obtain rfl := eq_of_forall₂_eq hr
      simpa using hZ'
  · intro j n hn Z' W' hZ' hr
    obtain rfl := eq_of_forall₂_eq hr
    simp only [Array.size_empty, Nat.zero_add] at hZ' ⊢
    have hpn : p.nodes[j]? = some (.node n) := by
      rw [hnodes, Array.getElem?_map, ← Array.getElem?_toList, hn]
      rfl
    obtain ⟨hcs, hcn⟩ := check_step hcheck hpn
    have hs : p.scalar j = some d ∨
        (p.scalar j = none ∧ (Node.node n).ports.all (fun q => !q.kind.isScalar)) := by
      cases he : p.scalar j with
      | none => exact .inr ⟨rfl, checkScalar_none hcs he⟩
      | some e =>
          cases hsc j e he
          exact .inl rfl
    have harity : ∀ o refs, n = .op o refs → refs.size = o.1.length := by
      intro o refs hno
      subst hno
      exact checkNode_op hcn
    have hstep := step_some (P := P) hctx (List.mapM_eq_ok_iff_forall₂.1 hargs) hs harity Z'
    refine ⟨fun vs fl h => ?_, fun ws h => ?_⟩
    · obtain ⟨ws, hw, rfl, rfl⟩ := (hstep vs fl).1 h
      exact ⟨ws, ws, hw, List.forall₂_same.2 fun _ _ => rfl, rfl, rfl⟩
    · exact ⟨ws, (hstep _ _).2 ⟨ws, h, rfl, rfl⟩, List.forall₂_same.2 fun _ _ => rfl⟩

/-- A checked program whose steps form the graph `g` without a scalar runs as `g` at any scalar
type `α`, with constants at `g`'s constant nodes and arguments related to the program's: the
outputs of every step are the graph's, the same Boolean, integer and index values, and no flag is
raised. -/
theorem run_graph?_none {g : Graph} (hg : p.graph? = some (g, none))
    (hcheck : p.check R = .ok ()) (hP : P.check p = .ok ()) (hsize : args.size = p.arity)
    {α : Type} [Storage α] [Context α] [Op.Capabilities α] {GP : IR.Payload α}
    (hGP : ∀ j port, g.nodes[j]? = some (.const port) → Option.Rel Op.NativeRel
      ((P.consts[j]?.join).bind fun v => (v.toOp none).toOption) (GP.const? j))
    {args₀ : List (@Op.Value (R.CarrierAt none) (R.storageAt none))}
    (hargs : args.toList.mapM (·.toOp none) = .ok args₀)
    {gargs : List (Op.Value α)} (hrel : List.Forall₂ Op.NativeRel args₀ gargs)
    (vs : Array (List (Value R))) (flags : ConversionStatus) :
    p.run R P args = .ok (vs, flags) ↔
      ∃ (zs : Array (List (@Op.Value (R.CarrierAt none) (R.storageAt none))))
        (ws : Array (List (Op.Value α))), g.run GP gargs = .ok ws ∧
        List.Forall₂ (List.Forall₂ Op.NativeRel) zs.toList ws.toList ∧
        vs = zs.map (·.map (Value.ofOp none)) ∧ flags = {} := by
  obtain ⟨hnodes, -, hsc⟩ := graph?_eq_some hg
  have hlist : p.nodes.toList = g.nodes.toList.map Node.node := by
    rw [hnodes, Array.toList_map]
  have hargs' : ∀ k : Nat, Option.Rel (fun a w => ∃ z, a.toOp none = .ok z ∧ Op.NativeRel z w)
      args[k]? gargs[k]? := by
    intro k
    have h₁ := forall₂_getElem? (List.mapM_eq_ok_iff_forall₂.1 hargs) k
    have h₂ := forall₂_getElem? hrel k
    rw [Array.getElem?_toList] at h₁
    generalize args[k]? = oa at h₁ ⊢
    generalize args₀[k]? = oz at h₁ h₂
    generalize gargs[k]? = ow at h₂ ⊢
    cases h₁ with
    | none =>
        cases h₂
        exact .none
    | some ha =>
        cases h₂ with
        | some hzw => exact .some ⟨_, ha, hzw⟩
  have key := run_nodes (p := p) (P := P) (args := args) (Value.ofOp none) Op.NativeRel GP gargs
    g.nodes.toList #[] #[] {} .nil ?_
  · rw [run_eq_foldlM hcheck hP hsize, hlist]
    obtain ⟨h₁, h₂⟩ := key
    constructor
    · intro h
      obtain ⟨Z', W', hW', hr, rfl, rfl⟩ := h₁ vs flags (by simpa using h)
      exact ⟨Z', W', hW', hr, rfl, rfl⟩
    · rintro ⟨zs, ws, hw, hr, rfl, rfl⟩
      obtain ⟨Z', hZ', hr'⟩ := h₂ ws hw
      have hu : Relator.LeftUnique (@Op.NativeRel (R.CarrierAt none) α (R.storageAt none) ‹_›) :=
        fun _ _ _ => Op.NativeRel.unique
      obtain rfl := Array.toList_inj.1 (hu.forall₂.forall₂ hr' hr)
      simpa using hZ'
  · intro j n hn Z' W' hZ' hr
    simp only [Array.size_empty, Nat.zero_add] at hZ' ⊢
    have hpn : p.nodes[j]? = some (.node n) := by
      rw [hnodes, Array.getElem?_map, ← Array.getElem?_toList, hn]
      rfl
    obtain ⟨hcs, -⟩ := check_step hcheck hpn
    have hs : p.scalar j = none := by
      cases he : p.scalar j with
      | none => rfl
      | some e => cases hsc j e he
    exact step_none hargs' (fun port h => hGP j port (by rw [← Array.getElem?_toList, hn, h]))
      hs (checkScalar_none hcs hs) hr

end IR.Program

end TorchLean
