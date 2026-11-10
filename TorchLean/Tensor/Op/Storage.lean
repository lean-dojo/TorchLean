/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Invoke
public import TorchLean.Tensor.Op.Table.Word
public import TorchLean.Tensor.Rep.Traversal.Sequence

/-!
# Operators that run without arithmetic

The operators whose ports are all Boolean, integer or index ports need no arithmetic. They are the
logical operators, the bitwise operators, shifts and bit counts of integer words and `mulhi`, a
`bitcast` between integer words, `indices`, the index adapters `encodeIndex` and `decodeIndex`, and
every operator at a Boolean or integer element. `Op.storageForward?` gives their forward at every
scalar type with storage, the value-domain check and then the forward map of their rule at `Float`
on the same tensors, so a typed program runs them without a `Context`. Every other operator has
none. `Op.storageForward?_isSome` and `Op.storageForward?_isSome_element` list operators that have
one. `Op.storageForward_eq` proves that where an operator has both a storage forward and a rule, the
storage forward is the rule's value-domain check followed by its forward map, failures of the check
included.

The operators with a storage forward and no rule are `decodeIndex` where its output has entries and
a bound is zero, so that no index tuple exists (`Op.wordIndices_isOk`), an integer true division and
a Boolean scatter other than a replacement. `Op.storageForward_none` proves that an operator with a
storage forward and no rule fails on every input. `Op.invoke_storageForward` proves that the
invocation of an operator with a storage forward succeeds exactly when its attribute check, the
reading of its inputs and its storage forward succeed, with the storage forward's outputs.

The carriers of Boolean, integer and index ports do not mention the scalar, so `Values.transfer`
reads such values over one scalar type as the same tensors over another. `Op.NativeRel` relates such
a value over one scalar type to the same value over another, and no value of the scalar is related
to one (`Values.native_of_forall₂`). An operator with only these ports that has a rule at one scalar
type with a `Context` has one at every other, with the same check on related inputs and related
outputs (`Op.rule?_native`). A storage forward that succeeds on some inputs succeeds on related
inputs, with related outputs (`Op.storageForward_native`, and the converse through
`Op.NativeRel.forall₂_symm`), and every operator whose ports are all Boolean, integer or index ports
has one (`Op.storageForward?_isSome_of_native`).
-/

@[expose] public section

namespace TorchLean

/-- Values at Boolean, integer and index ports over the scalar type `α` as the same tensors over
`β`: the carriers of these ports do not mention the scalar. -/
def Values.transfer {α β : Type} {_ : Storage α} {_ : Storage β} :
    (ps : List Port) → ps.all (fun p => !p.kind.isScalar) → Values α ps → Values β ps
  | [], _, _ => ()
  | ⟨.index _, _⟩ :: ps, h, (x, xs) =>
      (x, Values.transfer ps (by rw [List.all_cons, Bool.and_eq_true] at h; exact h.2) xs)
  | ⟨.bool, _⟩ :: ps, h, (x, xs) =>
      (x, Values.transfer ps (by rw [List.all_cons, Bool.and_eq_true] at h; exact h.2) xs)
  | ⟨.integer _ _, _⟩ :: ps, h, (x, xs) =>
      (x, Values.transfer ps (by rw [List.all_cons, Bool.and_eq_true] at h; exact h.2) xs)
  | ⟨.value, _⟩ :: _, h, _ | ⟨.frozen, _⟩ :: _, h, _ | ⟨.detached, _⟩ :: _, h, _ =>
      absurd h (by simp [Port.Kind.isScalar])

/-- The forward of an operator whose ports are all Boolean, integer or index ports, at every scalar
type with storage: the value-domain check and then the forward map of its rule at `Float` on the
same tensors. It fails where the operator has no rule, at an integer true division, a Boolean
scatter other than a replacement and a `decodeIndex` with a zero bound and entries. Every other
operator has none. -/
def Op.storageForward? {ins outs : List Port} (op : Op ins outs) :
    Option (∀ (α : Type) [Storage α], Values α ins → Except String (Values α outs)) :=
  if h : ins.all (fun p => !p.kind.isScalar) ∧ outs.all (fun p => !p.kind.isScalar) then
    some fun _ _ xs => do
      let some rule := op.rule? (α := Float) |
        throw "op: no rule (the operator's attribute check fails, or decodeIndex has none at a \
          zero bound with entries)"
      let ys : Values Float ins := Values.transfer ins h.1 xs
      rule.validate ys
      pure (Values.transfer outs h.2 (rule.forward ys))
  else none

/-- An operator whose ports are all Boolean, integer or index ports has a storage forward. -/
theorem Op.storageForward?_isSome_of_native {ins outs : List Port} (op : Op ins outs)
    (h : (ins ++ outs).all (fun p => !p.kind.isScalar)) : op.storageForward?.isSome := by
  rw [List.all_append, Bool.and_eq_true] at h
  simp only [Op.storageForward?, dite_eq_left h, Option.isSome_some]

/-- The logical operators, the integer word operators, a `bitcast` between integer words,
`indices` and the index adapters have a storage forward. `Op.storageForward?_isSome_element` lists
operators with an element argument that have one. -/
theorem Op.storageForward?_isSome {s : Shape} {signed signed' : Bool} {width : Nat}
    {bounds : List Nat} :
    (Op.logicalNot (s := s)).storageForward?.isSome ∧
    (Op.logicalAnd (s := s)).storageForward?.isSome ∧
    (Op.logicalOr (s := s)).storageForward?.isSome ∧
    (Op.logicalXor (s := s)).storageForward?.isSome ∧
    (Op.bitwiseAnd (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.bitwiseOr (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.bitwiseXor (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.bitwiseNot (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.bitwiseLeftShift (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.shiftRightArithmetic (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.shiftRightLogical (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.populationCount (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.countLeadingZeros (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.mulhi (signed := signed) (width := width) (s := s)).storageForward?.isSome ∧
    (Op.bitcast (s := s) width (.integer signed) (.integer signed')).storageForward?.isSome ∧
    (∀ dim : Fin s.rank, (Op.indices s dim).storageForward?.isSome) ∧
    (Op.encodeIndex (s := s) bounds signed width).storageForward?.isSome ∧
    (Op.decodeIndex (s := s) bounds signed width).storageForward?.isSome :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, fun _ => rfl, rfl,
    rfl⟩

/-- Operators with an element argument that have a storage forward: the comparisons and the
arithmetic operators other than true division at integer words, and every movement, `pad`, `gather`,
`select` and a replacing `scatter` at Booleans and integer words. At integer words `scatter` has one
by every rule. -/
theorem Op.storageForward?_isSome_element {s t : Shape} {signed : Bool} {width : Nat} :
    (Op.neg (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.add (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.sub (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.mul (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.max (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.min (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.div (s := s) .trunc (.integer signed width)).storageForward?.isSome ∧
    (Op.div (s := s) .floor (.integer signed width)).storageForward?.isSome ∧
    (Op.remainder (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.pow (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.eq (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.ne (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.lt (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.le (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.gt (s := s) (.integer signed width)).storageForward?.isSome ∧
    (Op.ge (s := s) (.integer signed width)).storageForward?.isSome ∧
    (∀ h : s.size = t.size, (Op.reshape h .bool).storageForward?.isSome ∧
      (Op.reshape h (.integer signed width)).storageForward?.isSome) ∧
    (∀ (perm : Equiv.Perm (Fin s.rank)) (out : Shape) (hout : out = s.permute perm),
      (Op.permute perm out hout .bool).storageForward?.isSome ∧
      (Op.permute perm out hout (.integer signed width)).storageForward?.isSome) ∧
    (∀ (dims : Fin s.rank → Fin t.rank) (h : Shape.BroadcastDims s t dims),
      (Op.broadcast dims h .bool).storageForward?.isSome ∧
      (Op.broadcast dims h (.integer signed width)).storageForward?.isSome) ∧
    (∀ dims : List (Fin s.rank), (Op.flip dims .bool).storageForward?.isSome ∧
      (Op.flip dims (.integer signed width)).storageForward?.isSome) ∧
    (∀ (dims : List (Fin s.rank)) (starts stops steps : List Nat)
      (h : Shape.SliceValid s dims starts stops steps),
      (Op.slice dims starts stops steps h .bool).storageForward?.isSome ∧
      (Op.slice dims starts stops steps h (.integer signed width)).storageForward?.isSome) ∧
    (∀ low high interior : Fin s.rank → Nat,
      (Op.pad low high interior .bool).storageForward?.isSome ∧
      (Op.pad low high interior (.integer signed width)).storageForward?.isSome) ∧
    (∀ (dims : List (Fin s.rank)) (batchDims : List (Fin s.rank × Fin t.rank))
      (h : Shape.GatherValid s t dims batchDims),
      (Op.gather dims batchDims h .bool).storageForward?.isSome ∧
      (Op.gather dims batchDims h (.integer signed width)).storageForward?.isSome) ∧
    ((Op.select (s := s) .bool).storageForward?.isSome ∧
      (Op.select (s := s) (.integer signed width)).storageForward?.isSome) ∧
    (∀ (dims : List (Fin s.rank)) (h : dims.Nodup),
      (Op.scatter (t := t) .replace dims h .bool).storageForward?.isSome ∧
      (Op.scatter (t := t) .replace dims h (.integer signed width)).storageForward?.isSome) ∧
    (∀ (dim : Fin s.rank) (sizes : List Nat), (Op.cat dim sizes .bool).storageForward?.isSome ∧
      (Op.cat dim sizes (.integer signed width)).storageForward?.isSome) ∧
    (∀ (rule : Tensor.ScatterReduce) (dims : List (Fin s.rank)) (h : dims.Nodup),
      (Op.scatter (t := t) rule dims h (.integer signed width)).storageForward?.isSome) :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    fun _ => ⟨rfl, rfl⟩, fun _ _ _ => ⟨rfl, rfl⟩, fun _ _ => ⟨rfl, rfl⟩, fun _ => ⟨rfl, rfl⟩,
    fun _ _ _ _ _ => ⟨rfl, rfl⟩, fun _ _ _ => ⟨rfl, rfl⟩, fun _ _ _ => ⟨rfl, rfl⟩, ⟨rfl, rfl⟩,
    fun _ _ => ⟨rfl, rfl⟩,
    fun _ _ => ⟨Op.storageForward?_isSome_of_native _ (by simp [Port.Kind.isScalar]),
      Op.storageForward?_isSome_of_native _ (by simp [Port.Kind.isScalar])⟩,
    fun _ _ _ => rfl⟩

/-- A tensor built by one action per entry fails where it has an entry and every action fails. -/
private theorem Tensor.generateFlatM_isOk {β : Type} [Storage β] {s : Shape}
    (f : Fin (Shape.size s) → Except String β)
    (hn : 0 < Shape.size s) (h : ∀ j, (f j).isOk = false) :
    (Tensor.generateFlatM s f).isOk = false := by
  unfold Tensor.generateFlatM
  have hseq := Tensor.sequenceFinM_isOk f hn h
  revert hseq
  cases Tensor.sequenceFinM f <;>
    simp [bind, Except.bind, pure, Except.pure, Except.isOk, Except.toBool]

/-- Decoding fails on every tensor of words where a bound is zero and the output has entries: no
index lies below zero. -/
theorem Op.wordIndices_isOk {bounds : List Nat} {s : Shape} {width : Nat} (signed : Bool)
    (h : ¬((∀ b ∈ bounds, 0 < b) ∨ s.size = 0))
    (x : Tensor (BitVec width) (Port.irShape ⟨.index bounds, s⟩)) :
    (Op.wordIndices (s := s) signed x).isOk = false := by
  have hzero : 0 ∈ bounds := by
    have : ∃ b ∈ bounds, ¬0 < b := by
      by_contra hall
      push Not at hall
      exact h (.inl hall)
    obtain ⟨b, hb, hpos⟩ := this
    exact (show b = 0 by omega) ▸ hb
  have hsize : 0 < Shape.size s := by
    omega
  exact Tensor.generateFlatM_isOk _ hsize fun _ => Port.IndexTuple.ofComponents_isOk _ hzero

section Native

/-- The same Boolean, integer or index value over two scalar types: the same port and the same
tensor, whose type does not mention the scalar. The storage instances are taken from the values,
since a scalar type's storage is not determined by the type. -/
inductive Op.NativeRel {α β : Type} {_ : Storage α} {_ : Storage β} :
    Op.Value α → Op.Value β → Prop
  /-- Booleans. -/
  | bool {s : Shape} (x : Tensor Bool s) : Op.NativeRel ⟨.bool s, x⟩ ⟨.bool s, x⟩
  /-- Integer words. -/
  | integer {signed : Bool} {width : Nat} {s : Shape} (x : Tensor (BitVec width) s) :
      Op.NativeRel ⟨.integer signed width s, x⟩ ⟨.integer signed width s, x⟩
  /-- Index tuples. -/
  | index {bounds : List Nat} {s : Shape} (x : Tensor (Port.IndexTuple bounds) s) :
      Op.NativeRel ⟨.index bounds s, x⟩ ⟨.index bounds s, x⟩

variable {α β : Type} {_ : Storage α} {_ : Storage β}

/-- The relation is symmetric. -/
theorem Op.NativeRel.symm {v : Op.Value α} {w : Op.Value β} (h : Op.NativeRel v w) :
    Op.NativeRel w v := by
  cases h <;> constructor

/-- A value is determined by the value it is related to. -/
theorem Op.NativeRel.unique {v v' : Op.Value α} {w : Op.Value β} (h : Op.NativeRel v w)
    (h' : Op.NativeRel v' w) : v = v' := by
  cases h <;> cases h' <;> rfl

/-- The entrywise relation of two lists is symmetric. -/
theorem Op.NativeRel.forall₂_symm {l : List (Op.Value α)} {l' : List (Op.Value β)}
    (h : List.Forall₂ Op.NativeRel l l') : List.Forall₂ Op.NativeRel l' l :=
  (h.imp fun _ _ h => h.symm).flip

/-- Related values are at one port and hold the same tensor. -/
theorem Op.NativeRel.heq {v : Op.Value α} {w : Op.Value β} (h : Op.NativeRel v w) :
    v.1 = w.1 ∧ HEq v.2 w.2 := by
  cases h <;> exact ⟨rfl, HEq.rfl⟩

/-- Values related to values over another scalar type are at Boolean, integer and index ports: no
value of the scalar is related to one. -/
theorem Values.native_of_forall₂ : ∀ (ps : List Port) {xs : Values α ps} {ys : Values β ps},
    List.Forall₂ Op.NativeRel (Values.toErased ps xs) (Values.toErased ps ys) →
    ps.all (fun p => !p.kind.isScalar)
  | [], _, _, _ => rfl
  | p :: ps, (_, _), (_, _), h => by
      obtain ⟨h₁, h₂⟩ := List.forall₂_cons.1 h
      rw [List.all_cons, Values.native_of_forall₂ ps h₂, Bool.and_true]
      cases h₁ <;> rfl

/-- The parts of related inputs of `cat` hold the same tensors. -/
theorem Op.catElementParts_native {e : Op.Element} {s : Shape} (dim : Fin s.rank) :
    ∀ (sizes : List Nat) {xs : Values α (sizes.map fun n => e.port .value (s.set dim n))}
      {ys : Values β (sizes.map fun n => e.port .value (s.set dim n))},
      List.Forall₂ Op.NativeRel (Values.toErased _ xs) (Values.toErased _ ys) →
      ∀ i, HEq (Op.catElementParts dim sizes xs i) (Op.catElementParts dim sizes ys i)
  | [], _, _, _, i => i.elim0
  | _ :: rest, (_, _), (_, _), h, i => by
      obtain ⟨h₁, h₂⟩ := List.forall₂_cons.1 h
      rcases i with ⟨_ | k, hk⟩
      · exact h₁.heq.2
      · exact Op.catElementParts_native dim rest h₂ ⟨k, Nat.lt_of_succ_lt_succ hk⟩

/-- If a value reads at a port, the value related to it reads there as a related tensor. -/
theorem Port.read_native {v : Op.Value α} {w : Op.Value β} (h : Op.NativeRel v w) (p : Port)
    {x : p.Carrier α} (hx : p.read v = .ok x) :
    ∃ y, p.read w = .ok y ∧ Op.NativeRel ⟨p, x⟩ ⟨p, y⟩ := by
  rcases p with ⟨pd, ps⟩
  cases h <;> cases pd <;> simp [Port.read, throw, throwThe, MonadExceptOf.throw] at hx ⊢
  case bool.bool =>
    split at hx
    · rename_i hc
      subst hc
      cases hx
      exact ⟨_, dite_eq_left rfl, by constructor⟩
    · cases hx
  case integer.integer =>
    split at hx
    · rename_i hc
      obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hc
      cases hx
      exact ⟨_, dite_eq_left ⟨⟨rfl, rfl⟩, rfl⟩, by constructor⟩
    · cases hx
  case index.index =>
    split at hx
    · rename_i hc
      obtain ⟨rfl, rfl⟩ := hc
      cases hx
      exact ⟨_, dite_eq_left ⟨rfl, rfl⟩, by constructor⟩
    · cases hx

/-- A value that reads at a Boolean, integer or index port is at that port. -/
theorem Port.fst_eq_of_read {p : Port} (hp : p.kind.isScalar = false) {v : Op.Value α}
    {x : p.Carrier α} (hx : p.read v = .ok x) : v.1 = p := by
  rcases p with ⟨pd, ps⟩
  rcases v with ⟨⟨vd, vs⟩, t⟩
  cases pd <;> simp [Kind.isScalar] at hp <;> cases vd <;>
    simp [Port.read, throw, throwThe, MonadExceptOf.throw] at hx ⊢
  all_goals
    split at hx
    · rename_i hc
      simpa using hc
    · cases hx

/-- If values read at a list of ports, the values related to them read there as related
values. -/
theorem Values.ofErased_native :
    ∀ (ps : List Port) {vs : List (Op.Value α)} {ws : List (Op.Value β)} {xs : Values α ps},
      List.Forall₂ Op.NativeRel vs ws → Values.ofErased ps vs = .ok xs →
      ∃ ys, Values.ofErased ps ws = .ok ys ∧
        List.Forall₂ Op.NativeRel (Values.toErased ps xs) (Values.toErased ps ys)
  | [], _, _, _, .nil, _ => ⟨(), rfl, .nil⟩
  | [], _, _, _, .cons _ _, h => by cases h
  | _ :: _, _, _, _, .nil, h => by cases h
  | p :: ps, v :: vs, w :: ws, xs, .cons hvw hrest, h => by
      simp only [Values.ofErased, bind, Except.bind] at h
      cases hx : p.read v with
      | error e => simp [hx] at h
      | ok x =>
        cases hxs : Values.ofErased ps vs with
        | error e => simp [hx, hxs] at h
        | ok xs' =>
          simp only [hx, hxs, pure, Except.pure] at h
          cases h
          obtain ⟨y, hy, hxy⟩ := Port.read_native hvw p hx
          obtain ⟨ys, hys, hrel⟩ := Values.ofErased_native ps hrest hxs
          exact ⟨(y, ys), by simp [Values.ofErased, hy, hys, bind, Except.bind, pure, Except.pure],
            .cons hxy hrel⟩

/-- Values that read at Boolean, integer and index ports are Boolean, integer and index values. -/
theorem Values.native_of_ofErased :
    ∀ (ps : List Port) {vs : List (Op.Value α)} {xs : Values α ps},
      ps.all (fun p => !p.kind.isScalar) → Values.ofErased ps vs = .ok xs →
      ∀ v ∈ vs, v.1.kind.isScalar = false
  | [], [], _, _, _, _, hv => by cases hv
  | [], _ :: _, _, _, h, _, _ => by cases h
  | _ :: _, [], _, _, h, _, _ => by cases h
  | p :: ps, v :: vs, xs, hn, h, w, hw => by
      simp only [List.all_cons, Bool.and_eq_true] at hn
      simp only [Values.ofErased, bind, Except.bind] at h
      cases hx : p.read v with
      | error e => simp [hx] at h
      | ok x =>
          cases hxs : Values.ofErased ps vs with
          | error e => simp [hx, hxs] at h
          | ok xs' =>
              rcases List.mem_cons.1 hw with rfl | hw
              · rw [Port.fst_eq_of_read (by simpa using hn.1) hx]
                simpa using hn.1
              · exact Values.native_of_ofErased ps hn.2 hxs w hw

/-- Values over `β` are the transfer of values over `α` exactly when they are related to them. -/
theorem Values.transfer_eq_iff :
    ∀ (ps : List Port) (hn : ps.all (fun p => !p.kind.isScalar)) {xs : Values α ps}
      {ys : Values β ps}, Values.transfer ps hn xs = ys ↔
        List.Forall₂ Op.NativeRel (Values.toErased ps xs) (Values.toErased ps ys)
  | [], _, _, _ => ⟨fun _ => .nil, fun _ => rfl⟩
  | ⟨d, s⟩ :: ps, hn, (x, xs), (y, ys) => by
      have hps : ps.all (fun p => !p.kind.isScalar) := by
        rw [List.all_cons, Bool.and_eq_true] at hn
        exact hn.2
      have ih := Values.transfer_eq_iff ps hps (xs := xs) (ys := ys)
      cases d <;> first
        | (simp [Port.Kind.isScalar] at hn; done)
        | exact ⟨fun h => by
              cases h
              exact .cons (by constructor) (ih.1 rfl),
            fun h => by
              obtain ⟨h₁, h₂⟩ := List.forall₂_cons.1 h
              cases h₁
              exact congrArg (Prod.mk _) (ih.2 h₂)⟩

/-- A transfer from `α` to `β` and on to `γ` is the transfer from `α` to `γ`. -/
theorem Values.transfer_transfer {γ : Type} {_ : Storage γ} :
    ∀ (ps : List Port) (hn : ps.all (fun p => !p.kind.isScalar)) (xs : Values α ps),
      (Values.transfer ps hn (Values.transfer ps hn xs : Values β ps) : Values γ ps) =
        Values.transfer ps hn xs
  | [], _, _ => rfl
  | ⟨d, s⟩ :: ps, hn, (x, xs) => by
      have hps : ps.all (fun p => !p.kind.isScalar) := by
        rw [List.all_cons, Bool.and_eq_true] at hn
        exact hn.2
      cases d <;> first
        | (simp [Port.Kind.isScalar] at hn; done)
        | exact congrArg (Prod.mk x) (Values.transfer_transfer ps hps xs)

/-- A storage forward that succeeds on its inputs succeeds on related inputs with related outputs:
on both it runs the rule at `Float` on the same tensors. -/
theorem Op.storageForward_native {ins outs : List Port} (op : Op ins outs)
    {f : ∀ (γ : Type) [Storage γ], Values γ ins → Except String (Values γ outs)}
    (hf : op.storageForward? = some f) {xs : Values α ins} {ys : Values β ins}
    (h : List.Forall₂ Op.NativeRel (Values.toErased ins xs) (Values.toErased ins ys))
    {zs : Values α outs} (hz : f α xs = .ok zs) :
    ∃ zs', f β ys = .ok zs' ∧
      List.Forall₂ Op.NativeRel (Values.toErased outs zs) (Values.toErased outs zs') := by
  by_cases hn : ins.all (fun p => !p.kind.isScalar) ∧ outs.all (fun p => !p.kind.isScalar)
  swap
  · rw [Op.storageForward?, dite_eq_right hn] at hf
    cases hf
  rw [Op.storageForward?, dite_eq_left hn] at hf
  cases hf
  have hxy : (Values.transfer ins hn.1 xs : Values Float ins) = Values.transfer ins hn.1 ys := by
    rw [← (Values.transfer_eq_iff ins hn.1).2 h, Values.transfer_transfer]
  dsimp only at hz ⊢
  rw [← hxy]
  revert hz
  cases Op.rule? (α := Float) op with
  | none => exact fun hz => nomatch hz
  | some rule =>
    dsimp only
    cases hv : rule.validate (Values.transfer ins hn.1 xs : Values Float ins) with
    | error e => exact fun hz => nomatch hz
    | ok u =>
      intro hz
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hz ⊢
      subst hz
      exact ⟨_, rfl, (Values.transfer_eq_iff outs hn.2).1 (Values.transfer_transfer outs hn.2 _)⟩

end Native

variable {α : Type} [Storage α] [Context α] [Op.Capabilities α]

/-- An operator whose ports are all Boolean, integer or index ports, with a rule at the scalar type
`α`, has a rule at `β` with the same value-domain check on related inputs and related outputs. -/
theorem Op.rule?_native {β : Type} [Storage β] [Context β] [Op.Capabilities β]
    {ins outs : List Port} (op : Op ins outs)
    (hn : (ins ++ outs).all (fun p => !p.kind.isScalar)) {rule : Op.Rule α ins outs}
    (hr : op.rule? = some rule) :
    ∃ rule' : Op.Rule β ins outs, op.rule? = some rule' ∧
      ∀ {xs : Values α ins} {ys : Values β ins},
        List.Forall₂ Op.NativeRel (Values.toErased ins xs) (Values.toErased ins ys) →
        rule'.validate ys = rule.validate xs ∧
          List.Forall₂ Op.NativeRel (Values.toErased outs (rule.forward xs))
            (Values.toErased outs (rule'.forward ys)) := by
  cases op
  case bitcast width source target =>
    cases source <;> cases target
    all_goals first
      | (simp [Port.Kind.isScalar] at hn; done)
      | (cases hr
         refine ⟨_, rfl, ?_⟩
         intro xs ys h
         obtain ⟨x, ⟨⟩⟩ := xs
         obtain ⟨y, ⟨⟩⟩ := ys
         simp only [Values.toErased, List.forall₂_cons] at h
         obtain ⟨h₁, -⟩ := h
         cases h₁
         exact ⟨rfl, .cons (by constructor) .nil⟩)
  case decodeIndex s bounds signed width =>
    by_cases hb : (∀ b ∈ bounds, 0 < b) ∨ s.size = 0
    · rw [Op.rule?_decodeIndex hb] at hr
      cases hr
      refine ⟨_, Op.rule?_decodeIndex hb, ?_⟩
      intro xs ys h
      obtain ⟨x, ⟨⟩⟩ := xs
      obtain ⟨y, ⟨⟩⟩ := ys
      simp only [Values.toErased, List.forall₂_cons] at h
      obtain ⟨h₁, -⟩ := h
      cases h₁
      exact ⟨rfl, .cons (by constructor) .nil⟩
    · rw [Op.rule?_decodeIndex_eq_none hb] at hr
      cases hr
  case cat dim sizes e =>
    cases e
    · simp [Port.Kind.isScalar] at hn
    all_goals
      cases hr
      refine ⟨_, rfl, ?_⟩
      intro xs ys h
      have hparts : Op.catElementParts dim sizes ys = Op.catElementParts dim sizes xs :=
        funext fun i => eq_of_heq (Op.catElementParts_native dim sizes h i).symm
      refine ⟨rfl, ?_⟩
      simp only [Values.toErased, hparts]
      exact .cons (by constructor) .nil
  all_goals (try cases ‹Op.Element›) <;> (try cases ‹Op.NumericElement›) <;>
    (try cases ‹DivRounding›) <;> (try cases ‹Tensor.ScatterReduce›)
  all_goals first
    | (simp [Port.Kind.isScalar] at hn; done)
    | (cases hr; done)
    | (cases hr
       refine ⟨_, rfl, ?_⟩
       intro xs ys h
       first
         | exact ⟨rfl, .cons (by constructor) .nil⟩
         | (obtain ⟨x₁, x₂, x₃, ⟨⟩⟩ := xs
            obtain ⟨y₁, y₂, y₃, ⟨⟩⟩ := ys
            simp only [Values.toErased, List.forall₂_cons] at h
            obtain ⟨h₁, h₂, h₃, -⟩ := h
            cases h₁
            cases h₂
            cases h₃
            exact ⟨rfl, .cons (by constructor) .nil⟩)
         | (obtain ⟨x₁, x₂, ⟨⟩⟩ := xs
            obtain ⟨y₁, y₂, ⟨⟩⟩ := ys
            simp only [Values.toErased, List.forall₂_cons] at h
            obtain ⟨h₁, h₂, -⟩ := h
            cases h₁
            cases h₂
            exact ⟨rfl, .cons (by constructor) .nil⟩)
         | (obtain ⟨x, ⟨⟩⟩ := xs
            obtain ⟨y, ⟨⟩⟩ := ys
            simp only [Values.toErased, List.forall₂_cons] at h
            obtain ⟨h₁, -⟩ := h
            cases h₁
            exact ⟨rfl, .cons (by constructor) .nil⟩))

/-- Where an operator has a storage forward and a rule, the storage forward is the rule's
value-domain check followed by its forward map. -/
theorem Op.storageForward_eq {ins outs : List Port} (op : Op ins outs)
    {f : ∀ (α : Type) [Storage α], Values α ins → Except String (Values α outs)}
    (hf : op.storageForward? = some f) {rule : Op.Rule α ins outs} (hs : op.rule? = some rule)
    (xs : Values α ins) : f α xs = (do rule.validate xs; pure (rule.forward xs)) := by
  by_cases hn : ins.all (fun p => !p.kind.isScalar) ∧ outs.all (fun p => !p.kind.isScalar)
  swap
  · rw [Op.storageForward?, dite_eq_right hn] at hf
    cases hf
  rw [Op.storageForward?, dite_eq_left hn] at hf
  cases hf
  have hio : (ins ++ outs).all (fun p => !p.kind.isScalar) := by
    rw [List.all_append, Bool.and_eq_true]
    exact hn
  obtain ⟨ruleF, hsF, hF⟩ := Op.rule?_native (β := Float) op hio hs
  obtain ⟨hvF, hyF⟩ := hF (xs := xs) ((Values.transfer_eq_iff ins hn.1).1 rfl)
  dsimp only
  simp only [hsF, hvF, (Values.transfer_eq_iff outs hn.2).2 (Op.NativeRel.forall₂_symm hyF)]

/-- An operator with a storage forward and no rule fails on every input: `decodeIndex` where a
bound is zero and its output has entries, an integer true division and a Boolean scatter other
than a replacement. -/
theorem Op.storageForward_none {ins outs : List Port} (op : Op ins outs)
    {f : ∀ (α : Type) [Storage α], Values α ins → Except String (Values α outs)}
    (hf : op.storageForward? = some f) (hs : op.rule? (α := α) = none) (xs : Values α ins) :
    (f α xs).isOk = false := by
  by_cases hn : ins.all (fun p => !p.kind.isScalar) ∧ outs.all (fun p => !p.kind.isScalar)
  swap
  · rw [Op.storageForward?, dite_eq_right hn] at hf
    cases hf
  rw [Op.storageForward?, dite_eq_left hn] at hf
  cases hf
  have hio : (ins ++ outs).all (fun p => !p.kind.isScalar) := by
    rw [List.all_append, Bool.and_eq_true]
    exact hn
  dsimp only
  cases hsF : Op.rule? (α := Float) op with
  | none => rfl
  | some ruleF =>
    obtain ⟨rule, hs', -⟩ := Op.rule?_native (α := Float) (β := α) op hio hsF
    rw [hs] at hs'
    cases hs'

/-- The invocation of an operator with a storage forward succeeds exactly when the attribute check,
the reading of its inputs at its ports and the storage forward succeed, and returns the storage
forward's outputs. -/
theorem Op.invoke_storageForward {ins outs : List Port} (op : Op ins outs)
    {f : ∀ (α : Type) [Storage α], Values α ins → Except String (Values α outs)}
    (hf : op.storageForward? = some f) {inputs outputs : List (Op.Value α)} :
    op.invoke (pure inputs) = .ok outputs ↔
      ∃ xs ys, op.check = .ok () ∧ Values.ofErased ins inputs = .ok xs ∧ f α xs = .ok ys ∧
        outputs = Values.toErased outs ys := by
  by_cases hn : ins.all (fun p => !p.kind.isScalar) ∧ outs.all (fun p => !p.kind.isScalar)
  swap
  · rw [Op.storageForward?, dite_eq_right hn] at hf
    cases hf
  rw [Op.storageForward?, dite_eq_left hn] at hf
  cases hf
  have hio : (ins ++ outs).all (fun p => !p.kind.isScalar) := by
    rw [List.all_append, Bool.and_eq_true]
    exact hn
  dsimp only
  constructor
  · intro h
    obtain ⟨values, xs, rule, hc, hi, hx, hs, hv, rfl⟩ := Op.invoke_eq_ok h
    cases hi
    obtain ⟨ruleF, hsF, hF⟩ := Op.rule?_native (β := Float) op hio hs
    obtain ⟨hvF, hyF⟩ := hF (xs := xs) ((Values.transfer_eq_iff ins hn.1).1 rfl)
    refine ⟨xs, rule.forward xs, hc, hx, ?_, rfl⟩
    simp only [hsF, hvF, hv, bind, Except.bind, pure, Except.pure,
      (Values.transfer_eq_iff outs hn.2).2 (Op.NativeRel.forall₂_symm hyF)]
  · rintro ⟨xs, ys, hc, hx, hy, rfl⟩
    revert hy
    cases hsF : Op.rule? (α := Float) op with
    | none => exact fun hy => nomatch hy
    | some ruleF =>
      dsimp only
      cases hvF : ruleF.validate (Values.transfer ins hn.1 xs : Values Float ins) with
      | error e => exact fun hy => nomatch hy
      | ok u =>
        intro hy
        simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq] at hy
        subst hy
        obtain ⟨rule, hs, hR⟩ := Op.rule?_native (α := Float) (β := α) op hio hsF
        obtain ⟨hv, hyR⟩ :=
          hR (Op.NativeRel.forall₂_symm ((Values.transfer_eq_iff ins hn.1).1 rfl))
        rw [hvF] at hv
        rw [(Values.transfer_eq_iff outs hn.2).2 hyR]
        cases u
        simp only [Op.invoke, hc, hx, hs, hv, Op.Rule.check, bind, Except.bind, pure,
          Except.pure, rule.prepare_value]

end TorchLean
