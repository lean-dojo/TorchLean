/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphConcatPermutationBridge
public import NN.MLTheory.CROWN.Proofs.BinaryMatmulRuntimeBridge
public import NN.MLTheory.CROWN.Proofs.GraphConvBridge

/-!
# Bridge from the runtime graph evaluator to the proof-side semantics

The certificate soundness theorems are stated against the proof-side evaluator
`CertSoundness.evalNode?` (flat, `Option`-valued, over `ℝ`). The executable runtime evaluates the
same IR with `NN.IR.Graph.evalNode` (shaped `SomeTensor`, `Except`-valued). This file shows that,
at `α := ℝ`, a successful runtime evaluation of a node is reproduced by `evalNode?` once the
runtime values are flattened.

## What is bridged

The per-node theorem `evalNode_bridge` covers the node kinds `input`, `const`, `detach`, `add`,
`sub`, `mulElem`, `relu`, `linear`, binary `matmul`, `conv`, and `concat` of arbitrary arity and
valid axis. Binary `matmul` includes vector promotion and arbitrary batch broadcasting. Convolution
uses the runtime's grouped, dilated, asymmetrically padded geometry. For `linear`
the parent value must be a vector: the
runtime applies the affine map independently along every leading axis, whereas the flat semantics
treats the whole flattened parent as one vector, so the two only agree without leading axes.

Not yet included in this per-node theorem: the transcendental ops
(`tanh`, `sigmoid`, `sin`, `cos`,
whose runtime versions go through `MathFunctions ℝ` rather than the `Real` functions used by
`evalNode?`), pooling, and reshape.

## Parameter and input correspondence

The runtime reads parameters from a `Payload ℝ` keyed by `Node.id`, the proof-side from a
`ParamStore ℝ` keyed by array index. `PayloadMatches` requires the two stores to agree on constants
and linear and convolution layers; `InputsLift` requires the proof-side input table to hold the
flattened runtime input at every `input` node. Both are stated for the id discipline
`nodes[i].id = i` that `Graph.denoteAll` enforces.
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

/-- The `ParamStore` mirrors the runtime payload on constants, linear layers and convolution. -/
def PayloadMatches (payload : NN.IR.Payload ℝ) (ps : ParamStore ℝ) : Prop :=
  (∀ id : Nat, ps.constVals[id]? = (payload.const? id).map fun c => { n := c.n, v := c.v }) ∧
  (∀ id : Nat, ps.linearWB[id]? =
    (payload.linear? id).map fun p => { m := p.outDim, n := p.inDim, w := p.W, b := p.b }) ∧
  (∀ id : Nat, ps.convCfg[id]? = payload.conv? id)

/-- The proof-side input table holds the flattened runtime input at every `input` node. -/
def InputsLift (nodes : Array Node) (input : SomeTensor ℝ) (inputs : Std.HashMap Nat Val) :
    Prop :=
  ∀ id : Nat, id < nodes.size → (nodes[id]!).kind = .input → inputs[id]? = some (flatOfSome input)

/-- Node kinds covered by `evalNode_bridge`. -/
def Bridged (kind : NN.IR.OpKind) : Prop :=
  match kind with
  | .input | .const _ | .detach | .add | .sub | .mulElem | .relu | .linear | .matmul
  | .conv _ | .concat _ => True
  | _ => False

/-- A runtime value whose stored shape is a vector. -/
def IsVector (t : SomeTensor ℝ) : Prop :=
  ∃ k : Nat, t.shape = Shape.dim k Shape.scalar

/-- Present runtime parents have the shapes declared by their graph nodes. -/
def ParentShapesMatch (nodes : Array Node) (node : Node)
    (rvals : Array (SomeTensor ℝ)) : Prop :=
  ∀ p ∈ node.parents, ∀ value, rvals[p]? = some value →
    ∃ parent, nodes[p]? = some parent ∧ value.shape = parent.outShape

/-- Successful graph execution supplies the parent-shape invariant for every node. -/
theorem parentShapesMatch_of_denoteAll
    (graph : NN.IR.Graph) (payload : NN.IR.Payload ℝ) (input : SomeTensor ℝ)
    (values : Array (SomeTensor ℝ)) (node : Node)
    (heval : graph.denoteAll payload input = .ok values) :
    ParentShapesMatch graph.nodes node values := by
  obtain ⟨hsize, hshapes⟩ := NN.IR.Graph.denoteAll_shape graph payload input values heval
  intro parent _ value hvalue
  obtain ⟨hp, rfl⟩ := Array.getElem?_eq_some_iff.mp hvalue
  have hnode : parent < graph.nodes.size := by simpa [hsize] using hp
  exact ⟨graph.nodes[parent], Array.getElem?_eq_getElem hnode, hshapes parent hnode hp⟩

/-- Restricting the value table preserves agreement at every parent that remains present. -/
theorem ParentShapesMatch.of_reads
    {nodes : Array Node} {node : Node} {values initial : Array (SomeTensor ℝ)}
    (hshapes : ParentShapesMatch nodes node values)
    (hreads : ∀ (parent : Nat) (value : SomeTensor ℝ),
      initial[parent]? = some value → values[parent]? = some value) :
    ParentShapesMatch nodes node initial := by
  intro parent hparent value hvalue
  exact hshapes parent hparent value (hreads parent value hvalue)

/-- Actual evaluator prefixes inherit the shape invariant from the completed execution. -/
theorem ParentShapesMatch.of_denoteAllFrom
    {graph : NN.IR.Graph} {payload : NN.IR.Payload ℝ} {input : SomeTensor ℝ}
    {node : Node} {initial values : Array (SomeTensor ℝ)} {start : Nat}
    (hshapes : ParentShapesMatch graph.nodes node values)
    (heval : graph.denoteAllFrom payload input start initial = .ok values) :
    ParentShapesMatch graph.nodes node initial := by
  apply ParentShapesMatch.of_reads hshapes
  intro parent value hvalue
  obtain ⟨hp, rfl⟩ := Array.getElem?_eq_some_iff.mp hvalue
  exact NN.IR.Graph.denoteAllFrom_prefix graph payload input start initial values heval parent hp

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
  rw [dite_eq_left hp', getElem!_pos (c := rvals.map fun t => some (flatOfSome t)) (i := p) hp',
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

/-- Successful runtime traversal transports pointwise read correspondence to the output array. -/
private theorem array_mapM_option_of_except {α β γ : Type} (xs : Array α)
    (f : α → Except String β) (g : α → Option γ) (convert : β → γ)
    (hread : ∀ x ∈ xs, ∀ y, f x = .ok y → g x = some (convert y))
    {ys : Array β} (h : xs.mapM f = .ok ys) :
    xs.mapM g = some (ys.map convert) := by
  have hlist : ∀ (entries : List α) (results : List β),
      (∀ x ∈ entries, ∀ y, f x = .ok y → g x = some (convert y)) →
      entries.mapM f = .ok results →
      entries.mapM g = some (results.map convert) := by
    intro entries
    induction entries with
    | nil =>
        intro results _ heval
        have : results = [] := by simpa [Pure.pure, Except.pure] using heval.symm
        subst results
        rfl
    | cons x entries ih =>
        intro results hread heval
        rw [List.mapM_cons] at heval
        obtain ⟨y, hy, hcontinue⟩ := bind_eq_ok (x := f x) heval
        obtain ⟨tail, htraverse, hresult⟩ := bind_eq_ok (x := entries.mapM f) hcontinue
        have hresult := pure_eq_ok hresult
        subst results
        have hg := hread x (by simp) y hy
        have hrest := ih tail (fun z hz => hread z (by simp [hz])) htraverse
        simp [List.mapM_cons, hg, hrest]
  rw [Array.mapM_eq_mapM_toList] at h ⊢
  cases hvalues : xs.toList.mapM f with
  | error error => simp [hvalues] at h
  | ok values =>
      have hys : values.toArray = ys := by simpa [hvalues] using h
      subst ys
      rw [hlist xs.toList values
        (fun x hx => hread x (Array.mem_toList_iff.mp hx)) hvalues]
      simp

/-- Under parent-shape consistency, graph concat layout checks use the runtime parent shapes. -/
theorem concatNodeLayout?_runtime_parents
    (nodes : Array Node) (rvals parents : Array (SomeTensor ℝ)) (i : Nat)
    (node : Node) (axis : Nat) (hshapes : ParentShapesMatch nodes node rvals)
    (hparents : node.parents.mapM
      (NN.IR.Graph.getParentValue (α := ℝ) rvals i node) = .ok parents) :
    concatNodeLayout? nodes node axis =
      concatLayout? axis (parents.map (·.shape)) node.outShape := by
  have hread := array_mapM_option_of_except node.parents
    (NN.IR.Graph.getParentValue (α := ℝ) rvals i node)
    (fun p => (nodes[p]?).map (·.outShape)) (·.shape) (fun p hp value hvalue => by
      obtain ⟨parent, hnode, hshape⟩ := hshapes p hp value (getParentValue_ok hvalue)
      simp [hnode, hshape]) hparents
  simp only [concatNodeLayout?, hread, Bind.bind, Option.bind]

/-- The flat graph evaluator reproduces the shaped concat family at every declared layout. -/
theorem evalNode?_concat_flattened_family
    (nodes : Array Node) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (rvals : Array (SomeTensor ℝ)) (i axis : Nat) (layout : ConcatLayout)
    (values : (parent : Fin layout.lengths.length) → Tensor ℝ (layout.parentShape parent))
    (hkind : (nodes[i]!).kind = .concat axis)
    (hlayout : concatNodeLayout? nodes nodes[i]! axis = some layout)
    (hparents : (nodes[i]!).parents.mapM (getVal? (liftVals rvals)) =
      some (Array.ofFn fun parent => flatOfSome ⟨layout.parentShape parent, values parent⟩)) :
    evalNode? nodes ps inputs (liftVals rvals) i =
      some (flatOfSome ⟨layout.outputShape,
        Tensor.Internal.Rep.concatenateAxes layout.leading layout.trailing layout.lengths
          values⟩) := by
  simp only [evalNode?, hkind, hlayout, Bind.bind, Option.bind, hparents]
  exact concatFlatValues?_flatten layout values

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
  rw [dite_eq_left hnn]
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
  rw [dite_eq_left hnn]
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
  rw [dite_eq_left hnn]
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
  simp only [evalNode?, hn, hpar, getVal?_liftVals hr', hps.2.1 i, hpay, Option.map_some]
  have hnn : (flatOfSome ⟨Shape.dim p.inDim Shape.scalar, xr⟩).n = p.inDim :=
    Nat.mul_one p.inDim
  rw [dite_eq_left hnn]
  have hx :
      castDimScalar (α := ℝ) hnn (flatOfSome ⟨Shape.dim p.inDim Shape.scalar, xr⟩).v = xr := by
    simp only [flatOfSome, flattenSpec_vector]
    exact castDimScalar_castDimScalar_symm (Nat.mul_one p.inDim) xr
  rw [hx, castDimScalar_self]
  refine congrArg some (flatTensor_eq_of_cast (Nat.mul_one p.outDim).symm _ _ ?_)
  simp only [flattenSpec_vector]
  rfl

/-- Successful runtime matmul has binary arity and the same checked flat contraction. -/
private theorem bridge_matmul
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (node : Node) (t : SomeTensor ℝ)
    (hn : nodes[i]! = node) (hkind : node.kind = .matmul)
    (hshapes : ParentShapesMatch nodes node rvals)
    (heval : NN.IR.Graph.evalNode payload input rvals i node = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw, hkind] at heval
  obtain ⟨raw, hraw, hnorm⟩ := bind_eq_ok heval
  obtain ⟨⟨p, q⟩, hpq, hraw⟩ := bind_eq_ok hraw
  obtain ⟨left, hleft, hraw⟩ := bind_eq_ok hraw
  obtain ⟨right, hright, hraw⟩ := bind_eq_ok hraw
  have hparents := binaryParentIds_ok hpq
  have hunary : NN.IR.unaryParent? node.parents = none := by
    unfold NN.IR.binaryParents? at hparents
    split at hparents
    next hsize => simp [NN.IR.unaryParent?, hsize]
    next => cases hparents
  cases hDims : NN.IR.OpContracts.matmulDims left.shape right.shape with
  | error message =>
      simp only [hDims] at hraw
      cases hraw
  | ok dims =>
      rw [hDims] at hraw
      obtain ⟨a, ha, hraw⟩ := bind_eq_ok hraw
      obtain ⟨b, hb, hraw⟩ := bind_eq_ok hraw
      obtain ⟨hsa, rfl⟩ := expectShape_ok ha
      obtain ⟨hsb, rfl⟩ := expectShape_ok hb
      obtain ⟨sa, a⟩ := left
      obtain ⟨sb, b⟩ := right
      simp only at hsa hsb
      subst sa
      subst sb
      have hleftRead := getParentValue_ok hleft
      have hrightRead := getParentValue_ok hright
      obtain ⟨leftNode, hleftNode, hleftShape⟩ := hshapes p
        (NN.IR.fst_mem_of_binaryParents?_eq_some hparents) _ hleftRead
      obtain ⟨rightNode, hrightNode, hrightShape⟩ := hshapes q
        (NN.IR.snd_mem_of_binaryParents?_eq_some hparents) _ hrightRead
      have nodeRead {p : Nat} {parent : Node} (h : nodes[p]? = some parent) :
          nodes[p]! = parent := by
        obtain ⟨hp, rfl⟩ := Array.getElem?_eq_some_iff.mp h
        simp only [getElem!_pos nodes p hp]
      have hresult := pure_eq_ok hraw
      rw [flatOfSome_of_normalize hnorm, ← hresult]
      simp only [evalNode?, hn, hkind, hunary, hparents,
        getVal?_liftVals hleftRead, getVal?_liftVals hrightRead,
        nodeRead hleftNode, nodeRead hrightNode]
      rw [← hleftShape, ← hrightShape]
      exact evalBinaryMatmul?_matmulWithDims hDims a b

/-- Runtime convolution supplies the same validated geometry and shaped input as the flat rule. -/
private theorem bridge_conv
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (node : Node) (t : SomeTensor ℝ)
    (configuration : NN.IR.ConvConfig)
    (hps : PayloadMatches payload ps) (hn : nodes[i]! = node) (hid : node.id = i)
    (hkind : node.kind = .conv configuration)
    (hshapes : ParentShapesMatch nodes node rvals)
    (heval : NN.IR.Graph.evalNode payload input rvals i node = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw, hkind] at heval
  obtain ⟨raw, hraw, hnorm⟩ := bind_eq_ok heval
  obtain ⟨p, hp, hraw⟩ := bind_eq_ok hraw
  obtain ⟨parent, hread, hraw⟩ := bind_eq_ok hraw
  obtain ⟨output, hconv, hraw⟩ := bind_eq_ok hraw
  split at hraw
  next => cases hraw
  next =>
    have hresult : output = raw := Except.ok.inj hraw
    subst raw
    have hparents := unaryParentId_ok hp
    have hparentRead := getParentValue_ok hread
    obtain ⟨parentNode, hparentNode, hparentShape⟩ := hshapes p
      (NN.IR.mem_of_unaryParent?_eq_some hparents) parent hparentRead
    have houtputShape := (normalizeNodeOutput_ok hnorm).1
    rw [flatOfSome_of_normalize hnorm]
    simp only [evalNode?, hn, hkind, hparents, hparentNode,
      getVal?_liftVals hparentRead, Bind.bind, Option.bind]
    rw [hid] at hconv
    exact evalConvNode?_of_evalConv (hps.2.2 i) hparentShape houtputShape hconv

/-! ## The bridge theorem -/

private theorem bridge_concat
    (nodes : Array Node) (ps : ParamStore ℝ) (input : SomeTensor ℝ)
    (inputs : Std.HashMap Nat Val) (payload : NN.IR.Payload ℝ)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (node : Node) (t : SomeTensor ℝ)
    (axis : Nat) (hn : nodes[i]! = node) (hkind : node.kind = .concat axis)
    (hshapes : ParentShapesMatch nodes node rvals)
    (heval : NN.IR.Graph.evalNode payload input rvals i node = .ok t) :
    evalNode? nodes ps inputs (liftVals rvals) i = some (flatOfSome t) := by
  simp only [NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw, hkind] at heval
  obtain ⟨raw, hraw, hnorm⟩ := bind_eq_ok heval
  obtain ⟨parents, hparents, hconcat⟩ := bind_eq_ok hraw
  obtain ⟨layout, values, hlayout, hvalues, hresult⟩ :=
    evalConcat_family_of_ok i node axis parents raw hconcat
  have hchecked : concatNodeLayout? nodes nodes[i]! axis = some layout := by
    rw [hn, concatNodeLayout?_runtime_parents nodes rvals parents i node axis hshapes hparents]
    exact hlayout
  have hflat := array_mapM_option_of_except node.parents
    (NN.IR.Graph.getParentValue (α := ℝ) rvals i node)
    (getVal? (liftVals rvals)) flatOfSome
    (fun _ _ _ hread => getVal?_liftVals (getParentValue_ok hread)) hparents
  rw [hvalues, Array.map_ofFn] at hflat
  rw [flatOfSome_of_normalize hnorm, hresult]
  exact evalNode?_concat_flattened_family nodes ps inputs rvals i axis layout values
    (by rw [hn, hkind]) hchecked
    (by simpa only [hn, Function.comp_def, SomeTensor.ofTensor] using hflat)

/--
A successful runtime evaluation of a bridged node is reproduced by the proof-side semantics on the
flattened value table.

Hypotheses: the node record `n` sits at index `i` with `n.id = i` (the id discipline that
`Graph.denoteAll` checks), the parameter stores agree (`PayloadMatches`), the proof-side inputs are
the flattened runtime input (`InputsLift`), present parents have their declared shapes
(`ParentShapesMatch`), the node kind is bridged, and a `linear` node has a vector-shaped parent.
The conclusion is exact equality of flat values, so this theorem can be used
to establish `SemLocalOK` for the flattened runtime trace.
-/
theorem evalNode_bridge
    (nodes : Array Node) (payload : NN.IR.Payload ℝ) (ps : ParamStore ℝ)
    (input : SomeTensor ℝ) (inputs : Std.HashMap Nat Val)
    (rvals : Array (SomeTensor ℝ)) (i : Nat) (n : Node) (t : SomeTensor ℝ)
    (hps : PayloadMatches payload ps) (hin : InputsLift nodes input inputs)
    (hi : i < nodes.size) (hn : nodes[i]! = n) (hid : n.id = i)
    (hkind : Bridged n.kind)
    (hshapes : ParentShapesMatch nodes n rvals)
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
  | matmul =>
      exact bridge_matmul nodes ps input inputs payload rvals nid
        { id := nid, parents := parents, kind := .matmul, outShape := outShape } t hn rfl
        hshapes heval
  | conv configuration =>
      exact bridge_conv nodes ps input inputs payload rvals nid
        { id := nid, parents := parents, kind := .conv configuration, outShape := outShape }
        t configuration hps hn rfl rfl hshapes heval
  | concat axis =>
      exact bridge_concat nodes ps input inputs payload rvals nid
        { id := nid, parents := parents, kind := .concat axis, outShape := outShape } t axis hn rfl
        hshapes heval
  | _ => simp [Bridged] at hkind

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
