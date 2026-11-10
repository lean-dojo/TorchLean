/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Checked invocation of registered operators on erased values

An erased value is a port with a tensor it carries, `Op.Value α`. `Op.invoke` runs one registered
operator on erased inputs, checked in stages. The attribute check (`Op.check`) runs before any
input is read, then each input is read against its port (`Port.read`: element sort and shape, or a
value where a frozen or detached input is expected). Then come the value-domain check (`validate`)
and the prepared node, whose values are the outputs, an index output among them. `Op.invoke_eq_ok`
splits a successful invocation into these stages, and `Op.invoke_congr` gives the same successful
outputs on two inputs whose reads at the input ports succeed with the same values. A graph of
registered operators is an `IR.Graph`, run by `IR.Graph.run`.
-/

@[expose] public section

namespace TorchLean

variable {α : Type} [Storage α]

/-- A port with a tensor it carries. -/
abbrev Op.Value (α : Type) [Storage α] := Σ p : Port, p.Carrier α

/-- A value at a port of the scalar shape is the scalar tensor of its item. -/
theorem Op.value_eq_scalar {t : Shape} (y : Tensor α t) (h : t = .scalar) :
    (⟨.value t, y⟩ : Op.Value α) = ⟨.value .scalar,
      Tensor.scalar (Tensor.item (Tensor.castShape y h))⟩ := by
  subst h
  simp

/-- The tensor of an erased value as the tensor of port `p`: the same port, or a value where `p` is
frozen or detached at the same shape. -/
def Port.read : (p : Port) → Op.Value α → Except String (p.Carrier α)
  | ⟨.frozen, s⟩, ⟨⟨.value, t⟩, x⟩ =>
      if h : t = s then pure (h ▸ x) else throw s!"op: an input of shape {t} where {s} is expected"
  | ⟨.detached, s⟩, ⟨⟨.value, t⟩, x⟩ =>
      if h : t = s then pure (h ▸ x) else throw s!"op: an input of shape {t} where {s} is expected"
  | p, ⟨q, x⟩ =>
      if h : q = p then pure
          (h ▸ x) else throw s!"op: an input of port {repr q} where {repr p} is expected"

namespace Values

/-- The values of `ports` from erased values, one each, checked against their ports. -/
def ofErased : (ports : List Port) → List (Op.Value α) → Except String (Values α ports)
  | [], [] => pure ()
  | p :: ports, v :: vs => do
      let x ← p.read v
      pure (x, ← ofErased ports vs)
  | _, _ => throw "op: wrong number of inputs"

/-- The erased values of `ports`. -/
def toErased : (ports : List Port) → Values α ports → List (Op.Value α)
  | [], _ => []
  | p :: ports, (x, xs) => ⟨p, x⟩ :: toErased ports xs

end Values

variable [Context α] [Op.Capabilities α]

/-- One registered operator on erased inputs, checked in stages: its attributes (`Op.check`), then
the computed inputs, each input against its port, the operator's rule (`Op.rule?`) and its
value-domain check (`validate`), then the prepared node, whose values are the outputs. An attribute
error takes precedence over an error in `inputs`, which arrives already computed. -/
def Op.invoke {ins outs : List Port} (op : Op ins outs)
    (inputs : Except String (List (Op.Value α))) : Except String (List (Op.Value α)) := do
  op.check
  let xs ← Values.ofErased ins (← inputs)
  let some rule := op.rule? (α := α) |
    throw "op: no rule (the scalar lacks a capability the operator reads, a custom node runs the \
      checked program its graph stores, and a bitcast whose scalar words do not have its width, a \
      decodeIndex with a zero bound and entries, an integer true division and a Boolean scatter \
      other than a replacement have none)"
  let prepared ← rule.check xs (Flags.all ins)
  pure (Values.toErased outs prepared.value)

/-- A successful invocation passed the attribute check, read its inputs at its ports, found a rule
whose value-domain check accepts them, and returns that rule's forward values. -/
theorem Op.invoke_eq_ok {ins outs : List Port} {op : Op ins outs}
    {inputs : Except String (List (Op.Value α))} {outputs : List (Op.Value α)}
    (h : op.invoke inputs = .ok outputs) :
    ∃ values xs rule, op.check = .ok () ∧ inputs = .ok values ∧
      Values.ofErased ins values = .ok xs ∧ op.rule? = some rule ∧ rule.validate xs = .ok () ∧
      outputs = Values.toErased outs (rule.forward xs) := by
  unfold Op.invoke at h
  cases hc : op.check with
  | error e => simp [hc, bind, Except.bind] at h
  | ok u =>
    cases hi : inputs with
    | error e => simp [hc, hi, bind, Except.bind] at h
    | ok values =>
      cases hx : Values.ofErased ins values with
      | error e => simp [hc, hi, hx, bind, Except.bind] at h
      | ok xs =>
        cases hs : op.rule? (α := α) with
        | none => simp [hc, hi, hx, hs, bind, Except.bind] at h
        | some rule =>
          cases hv : rule.validate xs with
          | error e => simp [hc, hi, hx, hs, hv, Op.Rule.check, bind, Except.bind] at h
          | ok v =>
            simp only [hc, hi, hx, hs, hv, Op.Rule.check, bind, Except.bind, pure,
              Except.pure, Except.ok.injEq] at h
            cases u
            cases v
            exact ⟨values, xs, rule, rfl, rfl, hx, rfl, hv, by rw [← h, rule.prepare_value]⟩

/-- When reading two inputs at the operator's input ports succeeds with the same values, an
invocation succeeds with given outputs on one input exactly when it does on the other. -/
theorem Op.invoke_congr {ins outs : List Port} (op : Op ins outs)
    {inputs₁ inputs₂ : Except String (List (Op.Value α))}
    (h : ∀ xs, (inputs₁ >>= Values.ofErased ins) = .ok xs ↔
      (inputs₂ >>= Values.ofErased ins) = .ok xs)
    {outputs : List (Op.Value α)} :
    op.invoke inputs₁ = .ok outputs ↔ op.invoke inputs₂ = .ok outputs := by
  have half : ∀ {i₁ i₂ : Except String (List (Op.Value α))},
      (∀ xs, (i₁ >>= Values.ofErased ins) = .ok xs → (i₂ >>= Values.ofErased ins) = .ok xs) →
      op.invoke i₁ = .ok outputs → op.invoke i₂ = .ok outputs := by
    intro i₁ i₂ h₁₂ hi
    obtain ⟨values, xs, rule, hc, rfl, hx, hs, hv, rfl⟩ := Op.invoke_eq_ok hi
    have h₂ := h₁₂ xs (by simp [bind, Except.bind, hx])
    cases i₂ with
    | error e => simp [bind, Except.bind] at h₂
    | ok values₂ =>
      simp only [bind, Except.bind] at h₂
      simp only [Op.invoke, hc, h₂, hs, hv, Op.Rule.check, bind, Except.bind, pure,
        Except.pure, rule.prepare_value]
  exact ⟨half fun xs => (h xs).1, half fun xs => (h xs).2⟩

end TorchLean
