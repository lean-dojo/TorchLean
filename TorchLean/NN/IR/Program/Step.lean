/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Program.Graph

/-!
# Steps of typed programs

The steps of one operator node (`IR.Program.step`): a `bitcast` between integer words, from the
scalar to integer words and back (`step_bitcast_integer`, `step_bitcast_scalar_integer`,
`step_bitcast_integer_scalar`), `decodeIndex` and `encodeIndex` at no scalar (`step_decodeIndex`,
`step_encodeIndex`, `step_decodeIndex_eq_ok`), and an operator step as the operator's invocation:
at a scalar with arithmetic for an operator that is not a custom operation (`step_op_arithmetic`),
and at no scalar for one whose ports are all Boolean, integer or index ports (`step_op_native`).
-/

@[expose] public section

namespace TorchLean

open Port
open FloatLib.Floats.ExecFloat (ConversionStatus)

namespace IR.Program

open Values (mapM_ofErased_cons)

/-- A step of `Op.bitcast` between integer words of one width retypes its operand's words. -/
theorem step_bitcast_integer {R : Machine} {q : Program} {P : Payload R} {args : Array (Value R)}
    {earlier : Array (List (Value R))} {i : Nat} {signed signed' : Bool} {width : Nat}
    {s : Shape} {r : Ref} (hsc : q.scalar i = none)
    {x : @Port.Carrier (R.CarrierAt none) (R.storageAt none) (.integer signed width s)}
    (hr : r.read earlier = .ok ⟨⟨.integer signed width s, none⟩, x⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.bitcast (s := s) width (.integer signed)
          (.integer signed')⟩ #[r])) =
      .ok ([⟨⟨.integer signed' width s, none⟩, x⟩], {}) := by
  have hf : (Op.bitcast (s := s) width (.integer signed)
      (.integer signed')).storageForward? = some fun _ _ xs => pure (xs.1, ()) := rfl
  have hc : (Op.bitcast (s := s) width (.integer signed)
      (.integer signed')).check = .ok () := rfl
  simp only [step]
  rw [hsc]
  simp only [runOp, runStorage, hf, hc]
  simp [hr, Value.toOp, Values.ofErased, Values.toErased, bind, Except.bind, pure, Except.pure]
  erw [Port.read_self]
  rfl

/-- The rule of `Op.bitcast` from the scalar to integer words, at capabilities whose words are
`B`'s. -/
private theorem rule?_bitcast_scalar_integer_of {α : Type} {_ : Storage α} {_ : Context α}
    {caps : Op.Capabilities α} {B : Bits α} (hB : caps.bits? = some B) {s : Shape}
    {signed : Bool} :
    @Op.rule? α _ _ caps _ _ (Op.bitcast (s := s) B.width .scalar (.integer signed)) = some
      { forward := fun xs => (Tensor.map (fun (x : α) => (B.toBits x).cast rfl) xs.1, ())
        backward := fun _ _ => ((), ()) } := by
  obtain ⟨_, _, _, _, _, bits, _, _, _⟩ := caps
  obtain rfl : bits = some B := hB
  exact dite_eq_left rfl

/-- The rule of `Op.bitcast` from integer words to the scalar, at capabilities whose words are
`B`'s. -/
private theorem rule?_bitcast_integer_scalar_of {α : Type} {_ : Storage α} {_ : Context α}
    {caps : Op.Capabilities α} {B : Bits α} (hB : caps.bits? = some B) {s : Shape}
    {signed : Bool} :
    @Op.rule? α _ _ caps _ _ (Op.bitcast (s := s) B.width (.integer signed) .scalar) = some
      { forward := fun xs =>
          (Tensor.map (fun (b : BitVec B.width) => B.ofBits (b.cast rfl)) xs.1, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } := by
  obtain ⟨_, _, _, _, _, bits, _, _, _⟩ := caps
  obtain rfl : bits = some B := hB
  exact dite_eq_left rfl

/-- A step of `Op.bitcast` from the scalar to integer words at the scalar `e`, whose words have
`width` bits, reads every entry's word. -/
theorem step_bitcast_scalar_integer {R : Machine} {q : Program} {P : Payload R}
    {args : Array (Value R)} {earlier : Array (List (Value R))} {i : Nat} {e : DType}
    {signed : Bool} {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = some e)
    {ctx : Context (R.Carrier e)} (hctx : R.arithmetic e = some ctx) {B : Bits (R.Carrier e)}
    (hB : (R.capabilities e).bits? = some B) (hw : B.width = width)
    {x : @Tensor (R.Carrier e) s (R.storage e)}
    (hr : r.read earlier = .ok ⟨⟨.value s, some e⟩, x⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.bitcast (s := s) width .scalar (.integer signed)⟩ #[r])) =
      .ok ([⟨⟨.integer signed width s, none⟩,
        @Tensor.map _ (R.storage e) _ _ _ (fun y => (B.toBits y).cast hw) x⟩], {}) := by
  simp only [step]
  rw [hsc]
  simp only [runOp, hctx]
  subst hw
  have hc : (Op.bitcast (s := s) B.width .scalar (.integer signed)).check = .ok () := rfl
  simp [hr, Value.toOp, Op.invoke, hc, rule?_bitcast_scalar_integer_of hB, Values.ofErased,
    Values.toErased, Op.Rule.check, Port.read, bind, Except.bind, pure,
    Except.pure]
  rfl

/-- A step of `Op.bitcast` from integer words to the scalar at the scalar `e`, whose words have
`width` bits, reads every word as a value. -/
theorem step_bitcast_integer_scalar {R : Machine} {q : Program} {P : Payload R}
    {args : Array (Value R)} {earlier : Array (List (Value R))} {i : Nat} {e : DType}
    {signed : Bool} {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = some e)
    {ctx : Context (R.Carrier e)} (hctx : R.arithmetic e = some ctx) {B : Bits (R.Carrier e)}
    (hB : (R.capabilities e).bits? = some B) (hw : B.width = width)
    {x : Tensor (BitVec width) s} (hr : r.read earlier = .ok ⟨⟨.integer signed width s, none⟩, x⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.bitcast (s := s) width (.integer signed) .scalar⟩ #[r])) =
      .ok ([⟨⟨.value s, some e⟩,
        @Tensor.map _ _ _ (R.storage e) _ (fun b : BitVec width => B.ofBits (b.cast hw.symm)) x⟩],
        {}) := by
  simp only [step]
  rw [hsc]
  simp only [runOp, hctx]
  subst hw
  have hc : (Op.bitcast (s := s) B.width (.integer signed) .scalar).check = .ok () := rfl
  simp [hr, Value.toOp, Op.invoke, hc, rule?_bitcast_integer_scalar_of hB, Values.ofErased,
    Values.toErased, Op.Rule.check, Port.read, bind, Except.bind, pure,
    Except.pure]
  rfl

/-- A step of `Op.decodeIndex` at no scalar, on integer words of positive width, outputs the
index tuples they hold (`Op.wordIndices`) where every bound is positive or the output has no
entries, failing where a component lies outside its bound, and fails without a rule otherwise. -/
theorem step_decodeIndex {R : Machine} {q : Program} {P : Payload R}
    {args : Array (Value R)} {earlier : Array (List (Value R))} {i : Nat} {bounds : List Nat}
    {signed : Bool} {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = none) (hw : 0 < width)
    {x : @Port.Carrier (R.CarrierAt none) (R.storageAt none)
      (.integer signed width (Port.irShape ⟨.index bounds, s⟩))}
    (hr : r.read earlier =
      .ok ⟨⟨.integer signed width (Port.irShape ⟨.index bounds, s⟩), none⟩, x⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.decodeIndex (s := s) bounds signed width⟩ #[r])) =
      if (∀ b ∈ bounds, 0 < b) ∨ s.size = 0 then
        (Op.wordIndices signed x).map fun y => ([⟨⟨.index bounds s, none⟩, y⟩], {})
      else .error "op: no rule (the operator's attribute check fails, or decodeIndex has none at \
        a zero bound with entries)" := by
  have hc : (Op.decodeIndex (s := s) bounds signed width).check = .ok () :=
    ite_eq_left hw
  by_cases hb : (∀ b ∈ bounds, 0 < b) ∨ s.size = 0
  · have hf : (Op.decodeIndex (s := s) bounds signed width).storageForward? =
        some (fun _ _ xs => do
          (Op.wordIndices signed xs.1).map fun _ => ()
          pure (Op.wordIndicesOr hb signed xs.1, ())) := by
      unfold Op.storageForward?
      rw [Op.rule?_decodeIndex hb]
      exact dite_eq_left ⟨rfl, rfl⟩
    simp only [step]
    rw [hsc]
    simp only [runOp, runStorage, hf, hc, hb, ↓reduceIte]
    simp [hr, Value.toOp, Values.ofErased, Values.toErased, bind, Except.bind, pure, Except.pure]
    erw [Port.read_self]
    dsimp only
    unfold Op.wordIndicesOr
    cases Op.wordIndices signed x <;> rfl
  · have hf : (Op.decodeIndex (s := s) bounds signed width).storageForward? =
        some (fun _ _ _ => throw "op: no rule (the operator's attribute check fails, or \
          decodeIndex has none at a zero bound with entries)") := by
      unfold Op.storageForward?
      rw [Op.rule?_decodeIndex_eq_none hb]
      exact dite_eq_left ⟨rfl, rfl⟩
    simp only [step]
    rw [hsc]
    simp only [runOp, runStorage, hf, hc, hb, ↓reduceIte]
    simp [hr, Value.toOp, Values.ofErased, Values.toErased, bind, Except.bind, pure, Except.pure]
    erw [Port.read_self]
    rfl

/-- A step of `Op.encodeIndex` at no scalar, into words of positive width, outputs the words of
the components of the index tuples it reads (`Op.indexWords`), failing where a component does not
fit the integer type (`Op.checkIndexWords`). -/
theorem step_encodeIndex {R : Machine} {q : Program} {P : Payload R} {args : Array (Value R)}
    {earlier : Array (List (Value R))} {i : Nat} {bounds : List Nat} {signed : Bool}
    {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = none) (hw : 0 < width)
    {y : @Port.Carrier (R.CarrierAt none) (R.storageAt none) ⟨.index bounds, s⟩}
    (hr : r.read earlier = .ok ⟨⟨.index bounds s, none⟩, y⟩) :
    q.step R P args earlier i
        (.node (.op ⟨_, _, Op.encodeIndex (s := s) bounds signed width⟩ #[r])) =
      (Op.checkIndexWords signed width y).map fun _ =>
        ([⟨⟨.integer signed width (Port.irShape ⟨.index bounds, s⟩), none⟩,
          Op.indexWords width y⟩], {}) := by
  have hf : (Op.encodeIndex (s := s) bounds signed width).storageForward? =
      some (fun _ _ xs => do
        Op.checkIndexWords signed width xs.1
        pure (Op.indexWords width xs.1, ())) := rfl
  have hc : (Op.encodeIndex (s := s) bounds signed width).check = .ok () :=
    ite_eq_left hw
  simp only [step]
  rw [hsc]
  simp only [runOp, runStorage, hf, hc]
  simp [hr, Value.toOp, Values.ofErased, Values.toErased, bind, Except.bind, pure, Except.pure]
  erw [Port.read_self]
  dsimp only
  cases Op.checkIndexWords signed width y <;> rfl

/-- A successful step of `Op.decodeIndex` at no scalar outputs one value and no flags. -/
theorem step_decodeIndex_eq_ok {R : Machine} {q : Program} {P : Payload R}
    {args : Array (Value R)} {earlier : Array (List (Value R))} {i : Nat} {bounds : List Nat}
    {signed : Bool} {width : Nat} {s : Shape} {r : Ref} (hsc : q.scalar i = none)
    {vs : List (Value R)} {raised : ConversionStatus}
    (h : q.step R P args earlier i
      (.node (.op ⟨_, _, Op.decodeIndex (s := s) bounds signed width⟩ #[r])) =
        .ok (vs, raised)) :
    raised = {} ∧ ∃ u, vs = [u] := by
  simp only [step] at h
  rw [hsc] at h
  simp only [runOp, runStorage, Op.storageForward?] at h
  obtain ⟨results, hres, h⟩ := bind_eq_ok.1 h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  refine ⟨rfl, ?_⟩
  obtain ⟨_, -, hres⟩ := bind_eq_ok.1 hres
  obtain ⟨xs, -, hres⟩ := bind_eq_ok.1 hres
  obtain ⟨ys, -, hres⟩ := bind_eq_ok.1 hres
  obtain ⟨zs, -, hres⟩ := bind_eq_ok.1 hres
  cases hres
  exact ⟨_, rfl⟩

/-- Values read at Boolean, integer and index ports at one scalar exactly where they read there at
another, as related values. -/
theorem mapM_toOp_native {R : Machine} {γ : Type} (g : γ → Except String (Value R))
    (d₁ d₂ : Option DType) :
    ∀ (ps : List Port) (l : List γ) {xs : @Values (R.CarrierAt d₁) (R.storageAt d₁) ps},
      ps.all (fun p => !p.kind.isScalar) →
      (l.mapM (fun c => do (← g c).toOp d₁) >>= @Values.ofErased _ (R.storageAt d₁) ps) =
        .ok xs →
      ∃ ys, (l.mapM (fun c => do (← g c).toOp d₂) >>= @Values.ofErased _ (R.storageAt d₂) ps) =
          .ok ys ∧
        List.Forall₂ (@Op.NativeRel _ _ (R.storageAt d₁) (R.storageAt d₂))
          (@Values.toErased _ (R.storageAt d₁) ps xs) (@Values.toErased _ (R.storageAt d₂) ps ys)
  | [], [], _, _, _ => ⟨(), rfl, .nil⟩
  | [], a :: l, _, _, h => by
    exfalso
    revert h
    rw [List.mapM_cons]
    cases (do (← g a).toOp d₁ : Except String _) <;>
      cases l.mapM (fun c => do (← g c).toOp d₁) <;>
      simp [bind, Except.bind, pure, Except.pure, Values.ofErased]
  | _ :: _, [], _, _, h => by
    simp [bind, Except.bind, pure, Except.pure, Values.ofErased] at h
  | p :: ps, a :: l, ⟨x, xs⟩, hn, h => by
    simp only [List.all_cons, Bool.and_eq_true] at hn
    obtain ⟨ha, hl⟩ := (@mapM_ofErased_cons _ (R.storageAt d₁) _ _ p ps a l x xs).1 h
    have ha' : ((g a >>= (·.toOp d₁)) >>= @Port.read _ (R.storageAt d₁) p) = .ok x := ha
    cases hg : g a with
    | error err => simp [hg, bind, Except.bind] at ha'
    | ok w =>
      simp only [hg, bind, Except.bind] at ha'
      obtain ⟨y, hy, hxy⟩ := Value.read_native (e' := d₂) (by simpa using hn.1) ha'
      obtain ⟨ys, hys, hrel⟩ := mapM_toOp_native g d₁ d₂ ps l hn.2 hl
      refine ⟨(y, ys), (@mapM_ofErased_cons _ (R.storageAt d₂) _ _ p ps a l y ys).2 ⟨?_, hys⟩,
        .cons hxy hrel⟩
      simpa [hg, bind, Except.bind] using hy

/-- An operator step without a scalar, all of whose ports are Boolean, integer or index ports,
succeeds exactly when invoking the operator on its inputs read at any scalar does, raising no
flags. -/
theorem step_op_native {R : Machine} {q : Program} {P : Payload R} {args : Array (Value R)}
    {earlier : Array (List (Value R))} {i : Nat} {e : DType} {ins outs : List Port}
    {o : Op ins outs} {refs : Array Ref} (hsc : q.scalar i = none)
    (hn : (ins ++ outs).all (fun p => !p.kind.isScalar)) (ctx : Context (R.Carrier e))
    {vs : List (Value R)} {flags : FloatLib.Floats.ExecFloat.ConversionStatus} :
    q.step R P args earlier i (.node (.op ⟨ins, outs, o⟩ refs)) = .ok (vs, flags) ↔
      flags = {} ∧ ∃ outputs, @Op.invoke (R.Carrier e) (R.storage e) ctx
          (R.capabilities e) ins outs o
          (refs.toList.mapM fun r => do (← r.read earlier).toOp (some e)) = .ok outputs ∧
        vs = outputs.map (Value.ofOp (some e)) := by
  obtain ⟨f, hf⟩ := Option.isSome_iff_exists.1 (Op.storageForward?_isSome_of_native o hn)
  have hin : ins.all (fun p => !p.kind.isScalar) := by
    simp only [List.all_append, Bool.and_eq_true] at hn
    exact hn.1
  let _ : Storage (R.Carrier e) := R.storage e
  let _ : Context (R.Carrier e) := ctx
  let _ : Op.Capabilities (R.Carrier e) := R.capabilities e
  simp only [step]
  rw [hsc, runOp.eq_3]
  constructor
  · intro h
    cases hN : refs.toList.mapM (fun r => do (← r.read earlier).toOp none) with
    | error err =>
      rw [hN] at h
      cases h
    | ok inputs' =>
      rw [hN] at h
      simp only [bind, Except.bind] at h
      cases hR : runStorage R i none o inputs' with
      | error err =>
        rw [hR] at h
        cases h
      | ok results' =>
        rw [hR] at h
        simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        obtain ⟨xs', ys', hc, hx', hy', rfl⟩ := (runStorage_eq_ok hf).1 hR
        obtain ⟨xs, hx, hrel⟩ := mapM_toOp_native (Ref.read earlier) none (some e) ins
          refs.toList hin (by rw [hN]; exact hx')
        cases hS : refs.toList.mapM (fun r => do (← r.read earlier).toOp (some e)) with
        | error err => rw [hS] at hx; cases hx
        | ok inputs =>
          rw [hS] at hx
          obtain ⟨zs, hz, hzrel⟩ := Op.storageForward_native o hf hrel hy'
          exact ⟨rfl, _, (Op.invoke_storageForward o hf).2 ⟨xs, zs, hc, hx, hz, rfl⟩,
            Value.map_ofOp_native hzrel⟩
  · rintro ⟨rfl, outputs, h, rfl⟩
    obtain ⟨inputs, -, -, hc, hS, -, -, -, -⟩ := Op.invoke_eq_ok h
    rw [hS] at h
    obtain ⟨xs, zs, -, hx, hz, rfl⟩ := (Op.invoke_storageForward o hf).1 h
    obtain ⟨xs', hx', hrel⟩ := mapM_toOp_native (Ref.read earlier) (some e) none ins refs.toList hin
      (by rw [hS]; exact hx)
    obtain ⟨ys', hy', hyrel⟩ := Op.storageForward_native o hf hrel hz
    cases hN : refs.toList.mapM (fun r => do (← r.read earlier).toOp none) with
    | error err => rw [hN] at hx'; cases hx'
    | ok inputs' =>
      rw [hN] at hx'
      have hR := (runStorage_eq_ok (R := R) (i := i) hf).2 ⟨xs', ys', hc, hx', hy', rfl⟩
      simp only [bind, Except.bind]
      rw [hR]
      simp only [pure, Except.pure, Value.map_ofOp_native hyrel]

/-- An operator step that is not a custom operation, at a scalar with arithmetic, succeeds exactly
when invoking the operator on its inputs read at that scalar does, raising no flags. -/
theorem step_op_arithmetic {R : Machine} {q : Program} {P : Payload R} {args : Array (Value R)}
    {earlier : Array (List (Value R))} {i : Nat} {e : DType} {ins outs : List Port}
    {o : Op ins outs} {refs : Array Ref} (hsc : q.scalar i = some e)
    {ctx : Context (R.Carrier e)} (hctx : R.arithmetic e = some ctx)
    (hcus : (Node.node (.op ⟨ins, outs, o⟩ refs)).custom? = none)
    {vs : List (Value R)} {flags : ConversionStatus} :
    q.step R P args earlier i (.node (.op ⟨ins, outs, o⟩ refs)) = .ok (vs, flags) ↔
      flags = {} ∧ ∃ outputs, @Op.invoke (R.Carrier e) (R.storage e) ctx
          (R.capabilities e) ins outs o
          (refs.toList.mapM fun r => do (← r.read earlier).toOp (some e)) = .ok outputs ∧
        vs = outputs.map (Value.ofOp (some e)) := by
  have hnc : ∀ (name : String) (shapes : List Shape) (output : Shape),
      ins = shapes.map Port.value →
        (⟨outs, o⟩ : (outs : List Port) × Op ins outs) ≍
          (⟨[Port.value output], Op.custom name shapes output⟩ :
            (outs : List Port) × Op (shapes.map Port.value) outs) → False := by
    intro name shapes output h₁ h₂
    subst h₁
    obtain ⟨rfl, h₃⟩ := Sigma.mk.inj (eq_of_heq h₂)
    cases eq_of_heq h₃
    exact Option.some_ne_none _ hcus
  simp only [step]
  rw [hsc, runOp.eq_2 _ _ _ _ _ _ _ _ _ hnc]
  simp only [hctx]
  let _ : Storage (R.Carrier e) := R.storage e
  let _ : Context (R.Carrier e) := ctx
  let _ : Op.Capabilities (R.Carrier e) := R.capabilities e
  cases hm : refs.toList.mapM (fun r => do (← r.read earlier).toOp (some e)) with
  | error err =>
    simp only [bind, Except.bind, reduceCtorEq, false_iff, not_and, not_exists]
    intro _ outputs h
    obtain ⟨_, _, _, _, hi, _⟩ := Op.invoke_eq_ok h
    cases hi
  | ok inputs =>
    simp only [bind, Except.bind]
    by_cases hs : (o.rule? (α := R.Carrier e)).isNone = true
    · simp only [hs, ite_true, throw, throwThe, MonadExceptOf.throw, reduceCtorEq, false_iff,
        not_and, not_exists]
      intro _ outputs h
      obtain ⟨_, _, rule, _, _, _, hr, _⟩ := Op.invoke_eq_ok h
      simp [hr] at hs
    · simp only [hs, pure, Except.pure]
      cases o.invoke (.ok inputs) <;> simp [eq_comm, and_comm]

end IR.Program

end TorchLean
