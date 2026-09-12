/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.Semantics

/-!
# Bridge from the runtime graph evaluator to the proof-side semantics

The certificate soundness theorems are stated against the proof-side evaluator
`CertSoundness.evalNode?` (flat, `Option`-valued, over `ℝ`). The executable runtime evaluates the
same IR with `NN.IR.Graph.evalNode` (shaped `SomeTensor`, `Except`-valued). This file shows that,
at `α := ℝ`, a successful runtime evaluation of a node is reproduced by `evalNode?` once the
runtime values are flattened.

## What is bridged

The per-node theorem `evalNode_bridge` covers the node kinds `input`, `const`, `detach`, `add`,
`sub`, `mulElem`, `relu`, and `linear`. For `linear` the parent value must be a vector: the
runtime applies the affine map independently along every leading axis, whereas the flat semantics
treats the whole flattened parent as one vector, so the two only agree without leading axes.

Not bridged here: `matmul` (the runtime `matmul` is binary and shape-driven, the proof-side
`matmul` is a payload-backed unary map), the transcendental ops (`tanh`, `sigmoid`, `sin`, `cos`,
whose runtime versions go through `MathFunctions ℝ` rather than the `Real` functions used by
`evalNode?`), and the pooling, reshape, and concatenation ops.

## Parameter and input correspondence

The runtime reads parameters from a `Payload ℝ` keyed by `Node.id`, the proof-side from a
`ParamStore ℝ` keyed by array index. `PayloadMatches` requires the two stores to agree on constants
and linear layers; `InputsLift` requires the proof-side input table to hold the flattened runtime
input at every `input` node. Both are stated for the id discipline `nodes[i].id = i` that
`Graph.denoteAll` enforces.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-! ## Flattening runtime values -/

/-- Flatten a shaped runtime value into the proof-side flat value. -/
def flatOfSome (t : SomeTensor ℝ) : Val :=
  { n := t.shape.size, v := Tensor.flattenSpec t.tensor }

/-- Lift a runtime value table to the proof-side table of flattened values. -/
def liftVals (rvals : Array (SomeTensor ℝ)) : Array (Option Val) :=
  rvals.map fun t => some (flatOfSome t)

/-- The `ParamStore` mirrors the runtime payload on constants and linear layers. -/
def PayloadMatches (payload : NN.IR.Payload ℝ) (ps : ParamStore ℝ) : Prop :=
  (∀ id : Nat, ps.constVals[id]? = (payload.const? id).map fun c => { n := c.n, v := c.v }) ∧
  (∀ id : Nat, ps.linearWB[id]? =
    (payload.linear? id).map fun p => { m := p.outDim, n := p.inDim, w := p.W, b := p.b })

/-- The proof-side input table holds the flattened runtime input at every `input` node. -/
def InputsLift (nodes : Array Node) (input : SomeTensor ℝ) (inputs : Std.HashMap Nat Val) :
    Prop :=
  ∀ id : Nat, id < nodes.size → (nodes[id]!).kind = .input → inputs[id]? = some (flatOfSome input)

/-- Node kinds covered by `evalNode_bridge`. -/
def Bridged (kind : NN.IR.OpKind) : Prop :=
  match kind with
  | .input | .const _ | .detach | .add | .sub | .mulElem | .relu | .linear => True
  | _ => False

/-- A runtime value whose stored shape is a vector. -/
def IsVector (t : SomeTensor ℝ) : Prop :=
  ∃ k : Nat, t.shape = Shape.dim k Shape.scalar

/-! ## Tensor-level helper lemmas -/

/-- Flattening commutes with pointwise binary operations. -/
theorem flattenSpec_map2Spec {s : Shape} (f : ℝ → ℝ → ℝ) (a b : Tensor ℝ s) :
    Tensor.flattenSpec (Tensor.map2Spec f a b) =
      Tensor.map2Spec f (Tensor.flattenSpec a) (Tensor.flattenSpec b) := by
  unfold Tensor.flattenSpec Tensor.map2Spec
  exact (TorchLean.Tensor.Internal.Rep.zipWith_reshape _ _ _ _).symm

/-- Flattening commutes with pointwise unary operations. -/
theorem flattenSpec_mapSpec {s : Shape} (f : ℝ → ℝ) (a : Tensor ℝ s) :
    Tensor.flattenSpec (Tensor.mapSpec f a) = Tensor.mapSpec f (Tensor.flattenSpec a) := by
  unfold Tensor.flattenSpec Tensor.mapSpec
  exact (TorchLean.Tensor.Internal.Rep.map_reshape _ _ _).symm

/-- Casting back and forth along a dimension equality is the identity. -/
theorem castDimScalar_castDimScalar_symm {n n' : Nat} (h : n = n') (t : Tensor ℝ [n']) :
    castDimScalar (α := ℝ) h (castDimScalar (α := ℝ) h.symm t) = t := by
  cases h
  rfl

/-- A dimension cast keeps the underlying buffer. -/
private theorem castDimScalar_buffer {n n' : Nat} (h : n = n') (t : Tensor ℝ [n]) :
    (castDimScalar (α := ℝ) h t).buffer = t.buffer := by
  cases h
  rfl

/-- Native tensors with equal buffers are equal. -/
private theorem rep_eq_of_buffer_eq {sh : TorchLean.Tensor.Internal.Shape}
    {x y : TorchLean.Tensor.Internal.Rep ℝ sh} (h : x.buffer = y.buffer) : x = y := by
  cases x
  cases y
  cases h
  rfl

/-- Flattening a vector only changes the recorded length from `n` to `n * 1`. -/
theorem flattenSpec_vector {n : Nat} (t : Tensor ℝ [n]) :
    Tensor.flattenSpec t = castDimScalar (α := ℝ) (Nat.mul_one n).symm t := by
  apply rep_eq_of_buffer_eq
  rw [castDimScalar_buffer]
  rfl

/-- Two flat values are equal when their lengths agree and the vectors agree up to the cast. -/
theorem flatTensor_eq_of_cast {n n' : Nat} (h : n = n') (v : Tensor ℝ [n]) (v' : Tensor ℝ [n'])
    (hv : castDimScalar (α := ℝ) h v = v') :
    ({ n := n, v := v } : Val) = { n := n', v := v' } := by
  cases h
  rw [castDimScalar_self] at hv
  rw [hv]

/-! ## Lookup helper lemmas -/

/-- A present runtime parent lifts to a present flat parent. -/
theorem getVal?_liftVals {rvals : Array (SomeTensor ℝ)} {p : Nat} {r : SomeTensor ℝ}
    (h : rvals[p]? = some r) : getVal? (liftVals rvals) p = some (flatOfSome r) := by
  obtain ⟨hp, rfl⟩ := Array.getElem?_eq_some_iff.mp h
  have hp' : p < (rvals.map fun t => some (flatOfSome t)).size := by simpa using hp
  unfold getVal? liftVals
  rw [dif_pos hp', getElem!_pos (c := rvals.map fun t => some (flatOfSome t)) (i := p) hp',
    Array.getElem_map]

/-- Reading a present parent through the runtime accessor succeeds with that parent. -/
theorem getParentValue_eq_ok {rvals : Array (SomeTensor ℝ)} {i p : Nat} {n : Node}
    {r : SomeTensor ℝ} (h : rvals[p]? = some r) :
    NN.IR.Graph.getParentValue (α := ℝ) rvals i n p = .ok r := by
  simp [NN.IR.Graph.getParentValue, h, Pure.pure, Except.pure]


/-! ## Inverting successful runtime steps -/

/-- A successful `Except` bind splits into a successful action and a successful continuation. -/
private theorem bind_eq_ok {α β : Type} {x : Except String α} {f : α → Except String β} {b : β}
    (h : (x >>= f) = .ok b) : ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [Bind.bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, by simpa [Bind.bind, Except.bind] using h⟩

/-- A successful `pure` in `Except` determines its value. -/
private theorem pure_eq_ok {α : Type} {a b : α} (h : (pure a : Except String α) = .ok b) :
    a = b := by
  simpa [Pure.pure, Except.pure] using h

/-- A successful runtime parent read comes from a present array entry. -/
private theorem getParentValue_ok {rvals : Array (SomeTensor ℝ)} {i p : Nat} {n : Node}
    {r : SomeTensor ℝ} (h : NN.IR.Graph.getParentValue (α := ℝ) rvals i n p = .ok r) :
    rvals[p]? = some r := by
  unfold NN.IR.Graph.getParentValue at h
  split at h
  · rename_i r' hr'
    rw [hr']
    exact congrArg some (pure_eq_ok h)
  · cases h

/-- A successful unary parent decode is a successful `unaryParent?`. -/
private theorem unaryParentId_ok {i : Nat} {n : Node} {p : Nat}
    (h : NN.IR.Graph.unaryParentId i n = .ok p) : NN.IR.unaryParent? n.parents = some p := by
  unfold NN.IR.Graph.unaryParentId at h
  split at h
  · rename_i q hq
    rw [hq, pure_eq_ok h]
  · cases h

/-- A successful binary parent decode is a successful `binaryParents?`. -/
private theorem binaryParentIds_ok {i : Nat} {n : Node} {p : Nat × Nat}
    (h : NN.IR.Graph.binaryParentIds i n = .ok p) : NN.IR.binaryParents? n.parents = some p := by
  unfold NN.IR.Graph.binaryParentIds at h
  split at h
  · rename_i q hq
    rw [hq, pure_eq_ok h]
  · cases h

/-- A successful shape check transports the stored tensor along the shape equality. -/
private theorem expectShape_ok {s : Shape} {v : SomeTensor ℝ} {x : Tensor ℝ s}
    (h : NN.IR.Graph.expectShape (α := ℝ) s v = .ok x) :
    ∃ hs : v.shape = s, x = hs ▸ v.tensor := by
  unfold NN.IR.Graph.expectShape at h
  split at h
  · rename_i hs
    exact ⟨hs, (pure_eq_ok h).symm⟩
  · cases h

/-- A successful output normalisation transports the value to the declared shape. -/
private theorem normalizeNodeOutput_ok {i : Nat} {n : Node} {v t : SomeTensor ℝ}
    (h : NN.IR.Graph.normalizeNodeOutput (α := ℝ) i n v = .ok t) :
    ∃ hs : v.shape = n.outShape, t = ⟨n.outShape, hs ▸ v.tensor⟩ := by
  unfold NN.IR.Graph.normalizeNodeOutput at h
  split at h
  · rename_i hs
    exact ⟨hs, (pure_eq_ok h).symm⟩
  · cases h

/-- Flattening ignores the transport of a value to its own shape. -/
private theorem flatOfSome_cast {s s' : Shape} (hs : s = s') (x : Tensor ℝ s) :
    flatOfSome ⟨s', hs ▸ x⟩ = flatOfSome ⟨s, x⟩ := by
  cases hs
  rfl

/-- A successfully normalised value flattens to the value before normalisation. -/
private theorem flatOfSome_of_normalize {i : Nat} {n : Node} {v t : SomeTensor ℝ}
    (h : NN.IR.Graph.normalizeNodeOutput (α := ℝ) i n v = .ok t) :
    flatOfSome t = flatOfSome v := by
  obtain ⟨hs, rfl⟩ := normalizeNodeOutput_ok h
  obtain ⟨sh, x⟩ := v
  exact flatOfSome_cast hs x

/-- A successful constant read unflattens the payload constant. -/
private theorem evalConst_ok {payload : NN.IR.Payload ℝ} {id : Nat} {s : Shape}
    {x : Tensor ℝ s} (h : NN.IR.Graph.evalConst (α := ℝ) payload id s = .ok x) :
    ∃ (c : NN.IR.ConstFlat ℝ) (hc : c.n = s.size), payload.const? id = some c ∧
      x = Tensor.unflattenSpec s (NN.IR.Graph.castDimScalar (α := ℝ) hc c.v) := by
  unfold NN.IR.Graph.evalConst at h
  split at h
  · cases h
  · rename_i c hc
    split at h
    · rename_i hn
      exact ⟨c, hn, hc, (pure_eq_ok h).symm⟩
    · cases h

/-- The IR-side dimension cast along a proof of `n = n` is the identity. -/
private theorem ir_castDimScalar_self {n : Nat} (h : n = n) (t : Tensor ℝ [n]) :
    NN.IR.Graph.castDimScalar (α := ℝ) h t = t := by
  cases h
  rfl

/-- A successful vector-input `linear` evaluation is the affine map of the payload. -/
private theorem evalLinear_vector_ok {payload : NN.IR.Payload ℝ} {id k : Nat}
    {xr : Tensor ℝ (Shape.dim k Shape.scalar)} {outShape : Shape} {y : SomeTensor ℝ}
    (h : NN.IR.Graph.evalLinear (α := ℝ) payload id ⟨Shape.dim k Shape.scalar, xr⟩ outShape
      = .ok y) :
    ∃ (p : NN.IR.LinearWB ℝ) (hk : k = p.inDim),
      payload.linear? id = some p ∧ outShape = Shape.dim p.outDim Shape.scalar ∧
      y = ⟨Shape.dim p.outDim Shape.scalar,
        Tensor.addSpec (Spec.matVecMulSpec (α := ℝ) p.W (castDimScalar (α := ℝ) hk xr)) p.b⟩ := by
  unfold NN.IR.Graph.evalLinear at h
  split at h
  · cases h
  · rename_i p hp
    obtain ⟨xT, hxT, h⟩ := bind_eq_ok h
    obtain ⟨hs, rfl⟩ := expectShape_ok hxT
    simp only [List.dropLast, Shape.ofList, Shape.concat] at hs
    have hk : k = p.inDim := by
      cases hs
      rfl
    subst hk
    split at h
    · rename_i hOut
      simp only [List.dropLast, Shape.ofList, Shape.concat] at hOut
      subst hOut
      refine ⟨p, rfl, hp, rfl, ?_⟩
      rw [castDimScalar_self]
      exact (pure_eq_ok h).symm
    · cases h

/-! ## Per-node bridges -/

/-- The `input` node: the runtime returns the graph input at its declared shape. -/
private theorem bridge_input
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (parents : Array Nat) (outShape : Shape)
    (t : SomeTensor ℝ)
    (hin : InputsLift nodes input inputs) (hi : i < nodes.size)
    (hn : nodes[i]! = { id := i, parents := parents, kind := .input, outShape := outShape })
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i
      { id := i, parents := parents, kind := .input, outShape := outShape } = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw] at heval
  obtain ⟨x, hx, hnorm⟩ := bind_eq_ok heval
  obtain ⟨t', ht', hx'⟩ := bind_eq_ok hx
  obtain ⟨hs, rfl⟩ := expectShape_ok ht'
  have hx'' := pure_eq_ok hx'
  subst hx''
  rw [flatOfSome_of_normalize hnorm]
  obtain ⟨sh, tin⟩ := input
  simp only at hs
  subst hs
  simp only [evalNode?, hn]
  exact hin i hi (by rw [hn])

/-- The `const` node: the runtime unflattens the payload constant. -/
private theorem bridge_const
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (parents : Array Nat) (s outShape : Shape)
    (t : SomeTensor ℝ)
    (hps : PayloadMatches payload ps)
    (hn : nodes[i]! = { id := i, parents := parents, kind := .const s, outShape := outShape })
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i
      { id := i, parents := parents, kind := .const s, outShape := outShape } = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw] at heval
  obtain ⟨x, hx, hnorm⟩ := bind_eq_ok heval
  obtain ⟨t', ht', hx'⟩ := bind_eq_ok hx
  obtain ⟨c, hc, hpay, rfl⟩ := evalConst_ok ht'
  have hx'' := pure_eq_ok hx'
  subst hx''
  rw [flatOfSome_of_normalize hnorm]
  simp only [evalNode?, hn, hps.1 i, hpay, Option.map_some]
  obtain ⟨cn, cv⟩ := c
  simp only at hc
  subst hc
  simp [flatOfSome, ir_castDimScalar_self]

/-- The `detach` node: the runtime forwards its parent. -/
private theorem bridge_detach
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (parents : Array Nat) (outShape : Shape)
    (t : SomeTensor ℝ)
    (hn : nodes[i]! = { id := i, parents := parents, kind := .detach, outShape := outShape })
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i
      { id := i, parents := parents, kind := .detach, outShape := outShape } = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw] at heval
  obtain ⟨x, hx, hnorm⟩ := bind_eq_ok heval
  obtain ⟨p1, hp1, hv⟩ := bind_eq_ok hx
  obtain ⟨r, hr, hv⟩ := bind_eq_ok hv
  obtain ⟨t', ht', hx'⟩ := bind_eq_ok hv
  obtain ⟨hs, rfl⟩ := expectShape_ok ht'
  have hx'' := pure_eq_ok hx'
  subst hx''
  rw [flatOfSome_of_normalize hnorm]
  have hpar := unaryParentId_ok hp1
  have hget := getVal?_liftVals (getParentValue_ok hr)
  simp only at hpar
  simp only [evalNode?, hn, hpar, hget]
  obtain ⟨sh, x⟩ := r
  simp only at hs
  subst hs
  rfl

/-- Peel a successful binary pointwise raw runtime chain into its parents and its result. -/
private theorem binary_chain_ok {i : Nat} {n : Node} {rvals : Array (SomeTensor ℝ)} {s : Shape}
    {g : Tensor ℝ s → Tensor ℝ s → Tensor ℝ s} {x : SomeTensor ℝ}
    (h : (do
        let pp ← NN.IR.Graph.binaryParentIds i n
        let ra ← NN.IR.Graph.getParentValue (α := ℝ) rvals i n pp.1
        let a ← NN.IR.Graph.expectShape (α := ℝ) s ra
        let rb ← NN.IR.Graph.getParentValue (α := ℝ) rvals i n pp.2
        let b ← NN.IR.Graph.expectShape (α := ℝ) s rb
        pure (⟨s, g a b⟩ : SomeTensor ℝ)) = .ok x) :
    ∃ (p1 p2 : Nat) (a b : Tensor ℝ s),
      NN.IR.binaryParents? n.parents = some (p1, p2) ∧
      rvals[p1]? = some ⟨s, a⟩ ∧ rvals[p2]? = some ⟨s, b⟩ ∧
      x = ⟨s, g a b⟩ := by
  obtain ⟨⟨p1, p2⟩, hpp, hv⟩ := bind_eq_ok h
  obtain ⟨ra, hra, hv⟩ := bind_eq_ok hv
  obtain ⟨a, hsa, hv⟩ := bind_eq_ok hv
  obtain ⟨rb, hrb, hv⟩ := bind_eq_ok hv
  obtain ⟨b, hsb, hx⟩ := bind_eq_ok hv
  obtain ⟨hsa', rfl⟩ := expectShape_ok hsa
  obtain ⟨hsb', rfl⟩ := expectShape_ok hsb
  obtain ⟨sa, ta⟩ := ra
  obtain ⟨sb, tb⟩ := rb
  simp only at hsa' hsb' hra hrb
  subst hsa' hsb'
  exact ⟨p1, p2, ta, tb, binaryParentIds_ok hpp, getParentValue_ok hra,
    getParentValue_ok hrb, (pure_eq_ok hx).symm⟩

/-- Flattening a pointwise binary result agrees with the flat semantics of the operation. -/
private theorem flat_binary {s : Shape} (f : ℝ → ℝ → ℝ) (a b : Tensor ℝ s)
    (h : s.size = s.size) :
    ({ n := s.size,
        v := Tensor.map2Spec f (Tensor.flattenSpec a)
          (castDimScalar (α := ℝ) h (Tensor.flattenSpec b)) } : Val) =
      flatOfSome ⟨s, Tensor.map2Spec f a b⟩ := by
  simp [flatOfSome, flattenSpec_map2Spec]

/-- The `add` node. -/
private theorem bridge_add
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (parents : Array Nat) (outShape : Shape)
    (t : SomeTensor ℝ)
    (hn : nodes[i]! = { id := i, parents := parents, kind := .add, outShape := outShape })
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i
      { id := i, parents := parents, kind := .add, outShape := outShape } = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw] at heval
  obtain ⟨x, hx, hnorm⟩ := bind_eq_ok heval
  obtain ⟨p1, p2, a, b, hp, hra, hrb, rfl⟩ := binary_chain_ok hx
  rw [flatOfSome_of_normalize hnorm]
  simp only at hp
  simp only [evalNode?, hn, hp, getVal?_liftVals hra, getVal?_liftVals hrb]
  have hnn : (flatOfSome ⟨outShape, a⟩).n = (flatOfSome ⟨outShape, b⟩).n := rfl
  rw [dif_pos hnn]
  exact congrArg some (flat_binary (fun x y => x + y) a b hnn)

/-- The `sub` node. -/
private theorem bridge_sub
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (parents : Array Nat) (outShape : Shape)
    (t : SomeTensor ℝ)
    (hn : nodes[i]! = { id := i, parents := parents, kind := .sub, outShape := outShape })
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i
      { id := i, parents := parents, kind := .sub, outShape := outShape } = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw] at heval
  obtain ⟨x, hx, hnorm⟩ := bind_eq_ok heval
  obtain ⟨p1, p2, a, b, hp, hra, hrb, rfl⟩ := binary_chain_ok hx
  rw [flatOfSome_of_normalize hnorm]
  simp only at hp
  simp only [evalNode?, hn, hp, getVal?_liftVals hra, getVal?_liftVals hrb]
  have hnn : (flatOfSome ⟨outShape, a⟩).n = (flatOfSome ⟨outShape, b⟩).n := rfl
  rw [dif_pos hnn]
  exact congrArg some (flat_binary (fun x y => x - y) a b hnn)

/-- The `mulElem` node. -/
private theorem bridge_mulElem
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (parents : Array Nat) (outShape : Shape)
    (t : SomeTensor ℝ)
    (hn : nodes[i]! = { id := i, parents := parents, kind := .mulElem, outShape := outShape })
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i
      { id := i, parents := parents, kind := .mulElem, outShape := outShape } = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw] at heval
  obtain ⟨x, hx, hnorm⟩ := bind_eq_ok heval
  obtain ⟨p1, p2, a, b, hp, hra, hrb, rfl⟩ := binary_chain_ok hx
  rw [flatOfSome_of_normalize hnorm]
  simp only at hp
  simp only [evalNode?, hn, hp, getVal?_liftVals hra, getVal?_liftVals hrb]
  have hnn : (flatOfSome ⟨outShape, a⟩).n = (flatOfSome ⟨outShape, b⟩).n := rfl
  rw [dif_pos hnn]
  exact congrArg some (flat_binary (fun x y => x * y) a b hnn)

/-- The `relu` node. -/
private theorem bridge_relu
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (parents : Array Nat) (outShape : Shape)
    (t : SomeTensor ℝ)
    (hn : nodes[i]! = { id := i, parents := parents, kind := .relu, outShape := outShape })
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i
      { id := i, parents := parents, kind := .relu, outShape := outShape } = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw] at heval
  obtain ⟨x, hx, hnorm⟩ := bind_eq_ok heval
  obtain ⟨p1, hp1, hv⟩ := bind_eq_ok hx
  obtain ⟨r, hr, hv⟩ := bind_eq_ok hv
  obtain ⟨t', ht', hx'⟩ := bind_eq_ok hv
  obtain ⟨hs, rfl⟩ := expectShape_ok ht'
  have hx'' := pure_eq_ok hx'
  subst hx''
  rw [flatOfSome_of_normalize hnorm]
  have hpar := unaryParentId_ok hp1
  have hget := getVal?_liftVals (getParentValue_ok hr)
  simp only at hpar
  simp only [evalNode?, hn, hpar, hget]
  obtain ⟨sh, x⟩ := r
  simp only at hs
  subst hs
  simp [flatOfSome, Activation.reluSpec, flattenSpec_mapSpec]

/-- The `linear` node, for a vector-shaped parent. -/
private theorem bridge_linear
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (parents : Array Nat) (outShape : Shape)
    (t : SomeTensor ℝ)
    (hps : PayloadMatches payload ps)
    (hn : nodes[i]! = { id := i, parents := parents, kind := .linear, outShape := outShape })
    (hvec : ∀ p ∈ parents, ∀ r : SomeTensor ℝ, rvals[p]? = some r → IsVector r)
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i
      { id := i, parents := parents, kind := .linear, outShape := outShape } = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw] at heval
  obtain ⟨y, hy, hnorm⟩ := bind_eq_ok heval
  obtain ⟨p1, hp1, hv⟩ := bind_eq_ok hy
  obtain ⟨r, hr, hy⟩ := bind_eq_ok hv
  have hpar := unaryParentId_ok hp1
  simp only at hpar
  have hr' := getParentValue_ok hr
  obtain ⟨k, hk⟩ := hvec p1 (NN.IR.mem_of_unaryParent?_eq_some hpar) r hr'
  obtain ⟨sr, xr⟩ := r
  simp only at hk
  subst hk
  obtain ⟨p, hkp, hpay, hOut, rfl⟩ := evalLinear_vector_ok hy
  subst hkp
  subst hOut
  rw [flatOfSome_of_normalize hnorm]
  simp only [evalNode?, hn, hpar, getVal?_liftVals hr', hps.2 i, hpay, Option.map_some]
  have hnn : (flatOfSome ⟨Shape.dim p.inDim Shape.scalar, xr⟩).n = p.inDim :=
    Nat.mul_one p.inDim
  rw [dif_pos hnn]
  have hx :
      castDimScalar (α := ℝ) hnn (flatOfSome ⟨Shape.dim p.inDim Shape.scalar, xr⟩).v = xr := by
    simp only [flatOfSome, flattenSpec_vector]
    exact castDimScalar_castDimScalar_symm (Nat.mul_one p.inDim) xr
  rw [hx, castDimScalar_self]
  refine congrArg some (flatTensor_eq_of_cast (Nat.mul_one p.outDim).symm _ _ ?_)
  simp only [flattenSpec_vector]
  rfl

/-! ## The bridge theorem -/

/--
A successful runtime evaluation of a bridged node is reproduced by the proof-side semantics on the
flattened value table.

Hypotheses: the node record `n` sits at index `i` with `n.id = i` (the id discipline that
`Graph.denoteAll` checks), the parameter stores agree (`PayloadMatches`), the proof-side inputs are
the flattened runtime input (`InputsLift`), the node kind is bridged, and a `linear` node has a
vector-shaped parent. The conclusion is exact equality of flat values, so this theorem can be used
to establish `SemLocalOK` for the flattened runtime trace.
-/
theorem evalNode_bridge
    (nodes : Array Node) (payload : NN.IR.Payload ℝ) (ps : ParamStore ℝ)
    (input : SomeTensor ℝ) (inputs : Std.HashMap Nat Val)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (n : Node) (t : SomeTensor ℝ)
    (hps : PayloadMatches payload ps) (hin : InputsLift nodes input inputs)
    (hi : i < nodes.size) (hn : nodes[i]! = n) (hid : n.id = i)
    (hkind : Bridged n.kind)
    (hvec : n.kind = .linear →
      ∀ p ∈ n.parents, ∀ r : SomeTensor ℝ, rvals[p]? = some r → IsVector r)
    (heval : NN.IR.Graph.evalNode (α := ℝ) payload input rvals i n = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  obtain ⟨nid, parents, kind, outShape⟩ := n
  simp only at hid hkind hvec
  subst hid
  cases kind with
  | input =>
      exact bridge_input nodes ps input inputs payload rvals nid parents outShape t hin hi hn
        heval
  | const s =>
      exact bridge_const nodes ps input inputs payload rvals nid parents s outShape t hps hn
        heval
  | detach =>
      exact bridge_detach nodes ps input inputs payload rvals nid parents outShape t hn heval
  | add =>
      exact bridge_add nodes ps input inputs payload rvals nid parents outShape t hn heval
  | sub =>
      exact bridge_sub nodes ps input inputs payload rvals nid parents outShape t hn heval
  | mulElem =>
      exact bridge_mulElem nodes ps input inputs payload rvals nid parents outShape t hn heval
  | relu =>
      exact bridge_relu nodes ps input inputs payload rvals nid parents outShape t hn heval
  | linear =>
      exact bridge_linear nodes ps input inputs payload rvals nid parents outShape t hps hn
        (hvec rfl) heval
  | _ => simp [Bridged] at hkind

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
