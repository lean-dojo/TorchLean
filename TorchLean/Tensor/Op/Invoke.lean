/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Run
public import TorchLean.Tensor.Op.Need

/-!
# Invoking registered operators

Facts about `Op.invoke` that hold for the whole operator table. `Op.checksValues` names the
operators whose rule checks its input values (`smoothMaxPool`, `encodeIndex`, `decodeIndex`, the
rounded integer divisions and the integer `remainder`), and `Op.validate_eq_ok` proves that every
other rule accepts every input. `Op.needsCapability` names the operators whose rule reads an
optional capability of the scalar (`Rounding`, `Erf`, `InverseTrig`, `Cbrt`, `NextAfter`, `Bits`),
and `Op.rule?_capabilities` proves that every other rule is the same for all capability instances.
Erased values read back at their ports (`Values.ofErased_toErased`), and a successful invocation
returns values at the operator's output ports (`Op.invoke_ports`).
-/

@[expose] public section

namespace TorchLean

/-- Whether the rule of an operator checks its input values before running: the smooth max pool,
which rejects a zero or nonfinite `beta`, the index adapters, which reject a component the
integer type cannot hold (`encodeIndex`) or a component outside its bound (`decodeIndex`), and the
rounded integer divisions and the integer `remainder`, which reject a zero divisor. -/
def Op.checksValues : {ins outs : List Port} → Op ins outs → Bool
  | _, _, .smoothMaxPool .. | _, _, .encodeIndex .. | _, _, .decodeIndex .. => true
  | _, _, .div .trunc (.integer ..) | _, _, .div .floor (.integer ..)
  | _, _, .remainder (.integer ..) => true
  -- The other elements of `div` and `remainder` are named, so that the match has equation lemmas.
  | _, _, .div _ .scalar | _, _, .div .none (.integer ..) | _, _, .remainder .scalar => false
  | _, _, _ => false

/-- Whether the rule of `op` reads an optional capability of the scalar (`Rounding`, `Erf`,
`InverseTrig`, `Cbrt`, `NextAfter` or `Bits`), that is, whether `Op.need op`, the capability
`Op.rule?` reads for `op`, is not `.none`. It is true at the roundings, the rounded divisions,
`fmod` and `remainder` at the scalar, the inverse trigonometric functions and `atan2`, `erf`, the
exact GELU, `cbrt`, `nextafter` and a `bitcast` with a scalar side. -/
def Op.needsCapability {ins outs : List Port} (op : Op ins outs) : Bool := op.need != .none

variable {α : Type} [Storage α]

/-- Reading an erased value at its own port returns its tensor. -/
theorem Port.read_self (p : Port) (x : p.Carrier α) : p.read ⟨p, x⟩ = .ok x := by
  rcases p with ⟨d, s⟩
  cases d <;> simp [Port.read, pure, Except.pure]

namespace Values

/-- The ports of erased values are the ports they were erased at. -/
theorem toErased_ports : (ps : List Port) → (xs : Values α ps) →
    (Values.toErased ps xs).map Sigma.fst = ps
  | [], _ => rfl
  | p :: ps, (x, xs) => by simp [Values.toErased, toErased_ports ps xs]

/-- Reading erased values back at their ports returns them. -/
theorem ofErased_toErased : (ps : List Port) → (xs : Values α ps) →
    Values.ofErased ps (Values.toErased ps xs) = .ok xs
  | [], () => rfl
  | p :: ps, (x, xs) => by
      simp [Values.ofErased, Values.toErased, Port.read_self, ofErased_toErased ps xs, bind,
        Except.bind, pure, Except.pure]

end Values

variable [Context α]

/-- An operator that reads no optional capability has the same rule under any two capability
records. -/
theorem Op.rule?_capabilities {ins outs : List Port} (op : Op ins outs)
    (h : op.needsCapability = false) (c₁ c₂ : Op.Capabilities α) :
    @Op.rule? α _ _ c₁ ins outs op = @Op.rule? α _ _ c₂ ins outs op := by
  cases op
  case div rounding e =>
    cases rounding <;> cases e <;> first | rfl | simp [Op.needsCapability] at h
  case gelu approximate => cases approximate <;> first | rfl | simp [Op.needsCapability] at h
  case linear lead _ _ => cases lead <;> rfl
  case cast source => cases source <;> rfl
  case sqrt policy => cases policy <;> rfl
  case bitcast source target =>
    cases source <;> cases target <;> first | rfl | simp [Op.needsCapability] at h
  case scatter r _ _ e => cases e <;> cases r <;> rfl
  all_goals (try cases ‹Op.Element›) <;> (try cases ‹Op.NumericElement›) <;>
    first | rfl | simp [Op.needsCapability] at h

/-- An operator that reads no capability is invoked alike under any two capability
dictionaries. -/
theorem Op.invoke_capabilities {ins outs : List Port} (op : Op ins outs)
    (h : op.needsCapability = false) (c₁ c₂ : Op.Capabilities α)
    (inputs : Except String (List (Op.Value α))) :
    @Op.invoke α _ _ c₁ ins outs op inputs = @Op.invoke α _ _ c₂ ins outs op inputs := by
  simp only [Op.invoke, Op.rule?_capabilities op h c₁ c₂]

variable [Op.Capabilities α]

/-- The rule of an operator that does not check its input values accepts every input. -/
theorem Op.validate_eq_ok {ins outs : List Port} (op : Op ins outs) (h : op.checksValues = false)
    {rule : Op.Rule α ins outs} (hs : op.rule? = some rule) (xs : Values α ins) :
    rule.validate xs = .ok () := by
  cases op
  case smoothMaxPool => simp [Op.checksValues] at h
  case encodeIndex => simp [Op.checksValues] at h
  case decodeIndex => simp [Op.checksValues] at h
  case custom => cases hs
  case div rounding e => cases rounding <;> cases e <;> first
    | (cases hs; rfl)
    | (cases hs; done)
    | (simp [Op.checksValues] at h; done)
    | (delta Op.rule? at hs; dsimp only at hs; split at hs
       · cases hs; rfl
       · cases hs)
  case gelu approximate => cases approximate <;> first
    | (cases hs; rfl)
    | (delta Op.rule? at hs; dsimp only at hs; split at hs
       · cases hs; rfl
       · cases hs)
  case linear lead _ _ => cases lead <;> (cases hs; rfl)
  case cast source => cases source <;> (cases hs; rfl)
  case sqrt policy => cases policy <;> (cases hs; rfl)
  case bitcast source target => cases source <;> cases target <;> first
    | (cases hs; rfl)
    | (delta Op.rule? at hs; dsimp only at hs; split at hs
       · split at hs
         · cases hs; rfl
         · cases hs
       · cases hs)
  case scatter r _ _ e => cases e <;> cases r <;> first | (cases hs; rfl) | cases hs
  all_goals (try cases ‹Op.Element›) <;> (try cases ‹Op.NumericElement›) <;> first
    | (cases hs; rfl)
    | (simp [Op.checksValues] at h; done)
    | (delta Op.rule? at hs; dsimp only at hs; split at hs
       · cases hs; rfl
       · cases hs)

/-- A successful invocation returns values at the operator's output ports. -/
theorem Op.invoke_ports {ins outs : List Port} {op : Op ins outs}
    {inputs : Except String (List (Op.Value α))} {vs : List (Op.Value α)}
    (h : op.invoke inputs = .ok vs) : vs.map Sigma.fst = outs := by
  obtain ⟨_, _, _, _, _, _, _, _, rfl⟩ := Op.invoke_eq_ok h
  exact Values.toErased_ports _ _

/-- The `k`-th output of a successful invocation is a tensor at the operator's `k`-th output
port. -/
theorem Op.invoke_getElem? {ins outs : List Port} {op : Op ins outs}
    {inputs : Except String (List (Op.Value α))} {vs : List (Op.Value α)}
    (h : op.invoke inputs = .ok vs) {k : Nat} {port : Port} (hp : outs[k]? = some port) :
    ∃ x, vs[k]? = some ⟨port, x⟩ := by
  have hk : (vs.map Sigma.fst)[k]? = some port := by rw [Op.invoke_ports h]; exact hp
  rw [List.getElem?_map] at hk
  cases hv : vs[k]? with
  | none => simp [hv] at hk
  | some v =>
    rcases v with ⟨q, x⟩
    simp only [hv, Option.map_some, Option.some.injEq] at hk
    subst hk
    exact ⟨x, rfl⟩



/-- Every output port of a registered operator that carries the model's scalar is a value port: no
operator returns a frozen or a detached tensor. -/
theorem Op.kind_eq_value_of_mem_outs {ins outs : List Port} (op : Op ins outs) {p : Port}
    (hp : p ∈ outs) (hs : p.kind.isScalar = true) : p.kind = .value := by
  have hw : ∀ (w : Op.Word) n s, (w.output n s).kind.isScalar = true →
      (w.output n s).kind = .value := by
    intro w n s h
    cases w
    · rfl
    · simp [Port.Kind.isScalar] at h
  have he : ∀ (e : Op.Element) s, (e.port .value s).kind.isScalar = true →
      (e.port .value s).kind = .value := by
    intro e s h
    cases e
    · rfl
    all_goals simp [Port.Kind.isScalar] at h
  have hn : ∀ (e : Op.NumericElement) s, (e.port .value s).kind.isScalar = true →
      (e.port .value s).kind = .value := by
    intro e s h
    cases e
    · rfl
    · simp [Port.Kind.isScalar] at h
  cases op <;> simp only [List.mem_cons, List.not_mem_nil, or_false] at hp <;>
    rcases hp with rfl | rfl | rfl <;>
    first
      | rfl
      | exact hw _ _ _ hs
      | exact he _ _ hs
      | exact hn _ _ hs
      | simp [Port.Kind.isScalar] at hs

end TorchLean
