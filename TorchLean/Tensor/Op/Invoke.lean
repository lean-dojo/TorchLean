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
optional capability of the scalar (`Rounding`, `Erf`, `InverseTrig`, `Cbrt`, `NextAfter`, `Bits`,
`Special`, `Frexp`, `Fma`, `ComplexFunctions`), and `Op.rule?_capabilities` proves that every
other rule is the same for all capability instances. Erased values read back at their ports
(`Values.ofErased_toErased`), and a successful invocation returns values at the operator's output
ports (`Op.invoke_ports`).
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
`InverseTrig`, `Cbrt`, `NextAfter`, `Bits`, `Special`, `Frexp`, `Fma` or `ComplexFunctions`), that
is, whether `Op.need op`, the capability `Op.rule?` reads for `op`, is not `.none`. It is true at
the roundings, the rounded divisions, `fmod` and `remainder` at the scalar, the inverse
trigonometric functions and `atan2`, `erf`, the exact GELU, `cbrt`, `nextafter`, a `bitcast` with
a scalar side, `special`, `frexp`, `fma`, `complex` and `complexPow`. -/
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
    (try cases ‹Tensor.ReduceOrder›) <;> first | rfl | simp [Op.needsCapability] at h

/-- An operator that reads no capability is invoked alike under any two capability
dictionaries. -/
theorem Op.invoke_capabilities {ins outs : List Port} (op : Op ins outs)
    (h : op.needsCapability = false) (c₁ c₂ : Op.Capabilities α)
    (inputs : Except String (List (Op.Value α))) :
    @Op.invoke α _ _ c₁ ins outs op inputs = @Op.invoke α _ _ c₂ ins outs op inputs := by
  simp only [Op.invoke, Op.rule?_capabilities op h c₁ c₂]

variable [Op.Capabilities α]

/-! The rules of the operators with a reduction order, of the complex functions, and of division,
GELU, bit casts and scatters accept every input by the theorems below, one declaration each.
Within the case split of `Op.validate_eq_ok` over the rest of the table, their cases exceed the
heartbeat budget of one declaration. -/

/-- The rule of a cumulative sum accepts every input, at every order. -/
private theorem Op.validate_cumsum_eq_ok {s : Shape} {dim : Fin s.rank}
    {order : Tensor.ReduceOrder} {rule} (hs : (Op.cumsum dim order).rule? (α := α) = some rule)
    (xs) : rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a cumulative product accepts every input, at every order. -/
private theorem Op.validate_cumprod_eq_ok {s : Shape} {dim : Fin s.rank}
    {order : Tensor.ReduceOrder} {rule} (hs : (Op.cumprod dim order).rule? (α := α) = some rule)
    (xs) : rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a cumulative maximum accepts every input, at every order. -/
private theorem Op.validate_cummax_eq_ok {s : Shape} {dim : Fin s.rank}
    {order : Tensor.ReduceOrder} {rule} (hs : (Op.cummax dim order).rule? (α := α) = some rule)
    (xs) : rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a cumulative minimum accepts every input, at every order. -/
private theorem Op.validate_cummin_eq_ok {s : Shape} {dim : Fin s.rank}
    {order : Tensor.ReduceOrder} {rule} (hs : (Op.cummin dim order).rule? (α := α) = some rule)
    (xs) : rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a matrix product accepts every input, at every order. -/
private theorem Op.validate_matmul_eq_ok {batchA batchB batch : Shape} {m n p : Nat}
    {lv rv : Bool} {hl : lv = true → m = 1} {hr : rv = true → p = 1}
    {broadcastA : Shape.CanBroadcastTo batchA batch}
    {broadcastB : Shape.CanBroadcastTo batchB batch}
    {order : Tensor.ReduceOrder}
    {rule} (hs : (Op.matmul (n := n) lv rv hl hr broadcastA broadcastB order).rule? (α := α) =
      some rule) (xs) :
    rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a scaled matrix product accepts every input, at every order. -/
private theorem Op.validate_scaledMatmul_eq_ok {batchA batchB batch : Shape} {m n p : Nat}
    {lv rv : Bool} {hl : lv = true → m = 1} {hr : rv = true → p = 1}
    {broadcastA : Shape.CanBroadcastTo batchA batch}
    {broadcastB : Shape.CanBroadcastTo batchB batch}
    {order : Tensor.ReduceOrder} {rule}
    (hs : (Op.scaledMatmul (n := n) lv rv hl hr broadcastA broadcastB order).rule? (α := α) =
      some rule) (xs) :
    rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a pool accepts every input, at every order. -/
private theorem Op.validate_pool_eq_ok {s : Shape} {op : Tensor.PoolOp} {dims : List (Fin s.rank)}
    {kernel stride padLow padHigh dilation : List Nat} {ceilMode : Bool}
    {order : Tensor.ReduceOrder}
    {rule} (hs : (Op.pool op dims kernel stride padLow padHigh dilation ceilMode order).rule?
      (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a reduction accepts every input, at every order. -/
private theorem Op.validate_reduce_eq_ok {s : Shape} {op : Tensor.ReduceOp}
    {dims : List (Fin s.rank)} {keepdim : Bool}
    {h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d}
    {order : Tensor.ReduceOrder}
    {rule} (hs : (Op.reduce op dims keepdim h order).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a mean accepts every input, at every order. -/
private theorem Op.validate_mean_eq_ok {s : Shape} {dims : List (Fin s.rank)} {keepdim : Bool}
    {order : Tensor.ReduceOrder}
    {rule} (hs : (Op.mean dims keepdim order).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a contraction accepts every input, at every order. -/
private theorem Op.validate_contract_eq_ok {s t : Shape}
    {dims batchDims : List (Fin s.rank × Fin t.rank)}
    {h : Shape.ContractValid s t dims batchDims} {order : Tensor.ReduceOrder}
    {rule} (hs : (Op.contract dims batchDims h order).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  cases order <;> (cases hs; rfl)

/-- The rule of a division accepts every input, at every rounding and element. -/
private theorem Op.validate_div_eq_ok {s : Shape} {rounding : DivRounding} {e : Op.NumericElement}
    (h : (Op.div (s := s) rounding e).checksValues = false) {rule}
    (hs : (Op.div (s := s) rounding e).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  cases rounding <;> cases e <;> first
    | (cases hs; rfl)
    | (cases hs; done)
    | (simp [Op.checksValues] at h; done)
    | (delta Op.rule? at hs; dsimp only at hs; split at hs
       · cases hs; rfl
       · cases hs)

/-- The rule of a GELU accepts every input, at both approximations. -/
private theorem Op.validate_gelu_eq_ok {s : Shape} {approximate : GeluApproximation} {rule}
    (hs : (Op.gelu (s := s) approximate).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  cases approximate <;> first
    | (cases hs; rfl)
    | (delta Op.rule? at hs; dsimp only at hs; split at hs
       · cases hs; rfl
       · cases hs)

/-- The rule of a bit cast accepts every input, between every pair of words. -/
private theorem Op.validate_bitcast_eq_ok {s : Shape} {width : Nat} {source target : Op.Word}
    {rule} (hs : (Op.bitcast (s := s) width source target).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  cases source <;> cases target <;> first
    | (cases hs; rfl)
    | (delta Op.rule? at hs; dsimp only at hs; split at hs
       · split at hs
         · cases hs; rfl
         · cases hs
       · cases hs)

/-- The rule of a scatter accepts every input, at every reduction and element. -/
private theorem Op.validate_scatter_eq_ok {s t : Shape} {r : Tensor.ScatterReduce}
    {dims : List (Fin s.rank)} {hd : dims.Nodup} {e : Op.Element} {rule}
    (hs : (Op.scatter (t := t) r dims hd e).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  cases e <;> cases r <;> first | (cases hs; rfl) | cases hs

/-- The rule of a complex function accepts every input. -/
private theorem Op.validate_complex_eq_ok {f : ComplexFn} {s : Shape} {rule}
    (hs : (Op.complex f (s := s)).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  delta Op.rule? at hs; dsimp only at hs; split at hs
  · cases hs; rfl
  · cases hs

/-- The rule of the complex power accepts every input. -/
private theorem Op.validate_complexPow_eq_ok {s : Shape} {rule}
    (hs : (Op.complexPow (s := s)).rule? (α := α) = some rule) (xs) :
    rule.validate xs = .ok () := by
  delta Op.rule? at hs; dsimp only at hs; split at hs
  · cases hs; rfl
  · cases hs

/-- The rule of an operator that does not check its input values accepts every input. -/
theorem Op.validate_eq_ok {ins outs : List Port} (op : Op ins outs) (h : op.checksValues = false)
    {rule : Op.Rule α ins outs} (hs : op.rule? = some rule) (xs : Values α ins) :
    rule.validate xs = .ok () := by
  cases op
  case smoothMaxPool => simp [Op.checksValues] at h
  case encodeIndex => simp [Op.checksValues] at h
  case decodeIndex => simp [Op.checksValues] at h
  case custom => cases hs
  case cumsum => exact Op.validate_cumsum_eq_ok hs xs
  case cumprod => exact Op.validate_cumprod_eq_ok hs xs
  case cummax => exact Op.validate_cummax_eq_ok hs xs
  case cummin => exact Op.validate_cummin_eq_ok hs xs
  case matmul => exact Op.validate_matmul_eq_ok hs xs
  case scaledMatmul => exact Op.validate_scaledMatmul_eq_ok hs xs
  case pool => exact Op.validate_pool_eq_ok hs xs
  case reduce => exact Op.validate_reduce_eq_ok hs xs
  case mean => exact Op.validate_mean_eq_ok hs xs
  case contract => exact Op.validate_contract_eq_ok hs xs
  case complex => exact Op.validate_complex_eq_ok hs xs
  case complexPow => exact Op.validate_complexPow_eq_ok hs xs
  case div => exact Op.validate_div_eq_ok h hs xs
  case gelu => exact Op.validate_gelu_eq_ok hs xs
  case linear lead _ _ => cases lead <;> (cases hs; rfl)
  case cast source => cases source <;> (cases hs; rfl)
  case sqrt policy => cases policy <;> (cases hs; rfl)
  case bitcast => exact Op.validate_bitcast_eq_ok hs xs
  case scatter => exact Op.validate_scatter_eq_ok hs xs
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
