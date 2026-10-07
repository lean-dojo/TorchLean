/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec

/-!
# Common

Internal helper lemmas for `NN.Runtime.Autograd.IRExec.Correctness`.

These lemmas relate the typed runtime context (`TorchLean.TensorPack`) to the untyped IR
value table (`Array Spec.SomeTensor`) and provide small building-block correctness steps that are
reused across the per-op proofs.

The lemmas are grouped as follows:

* `TorchLean.TensorPack.toShapeErasedArray` lemmas: relate the typed context produced by
  `ForwardData.eval` to an untyped `Array (Spec.SomeTensor α)` (this is what the IR evaluator uses).
* `denoteAllState*` lemmas: package the IR forward evaluator (`ForwardGraph.denoteAll`) in the form
  expected by IR-style semantic equivalence proofs.

Per-op correctness files reuse these bridges. The checked matmul-layout and axis-reduction
evaluator equations also live here so their operation proofs can use the same typed witnesses.

## Main definitions

- `throw_bind_ne_ok`: eliminates impossible success branches after `throw`.
- `NoRawLog`: side condition for theorem statements that do not carry the positivity
  precondition required by raw real `log`.
- `noRawLog_of_forall_mem`: discharge `NoRawLog` from a node-array fact, so a concrete graph can
  close it with `decide`.
- `toShapeErasedArray_getElem?`, `toShapeErasedArray_getIdx?`: optional lookup bridges.
- `evalAt_matmul_dims_ok`, `evalAt_axisReduction_ok`: `evalAt` in well-typed success cases.
- `denoteAllState_*` helpers: semantic equivalence bridges between lowered state and IR denotation
  tables.

-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Runtime.Autograd.IRExec.Internal
open Proofs (Idx getIdx)

/-- The lowering and semantic evaluators agree on a successfully decoded unary parent. -/
@[simp] theorem unaryParentId_eq_ok_of_unaryParent_eq_some
    (i p : Nat) (n : NN.IR.Node) (h : unaryParent? n.parents = some p) :
    NN.IR.Graph.unaryParentId i n = .ok p := by
  simp [NN.IR.Graph.unaryParentId, h, Pure.pure, Except.pure]

/-- The lowering and semantic evaluators agree on successfully decoded binary parents. -/
@[simp] theorem binaryParentIds_eq_ok_of_binaryParents_eq_some
    (i a b : Nat) (n : NN.IR.Node) (h : binaryParents? n.parents = some (a, b)) :
    NN.IR.Graph.binaryParentIds i n = .ok (a, b) := by
  simp [NN.IR.Graph.binaryParentIds, h, Pure.pure, Except.pure]

/-!
## Shared side conditions

These predicates describe the exact fragment covered by a theorem. Keeping them in `Common` lets
the per-op lemmas, the semantic equivalence proof, and the chapter index refer to the same public
contract without import cycles.
-/

/--
Core semantic equivalence side condition: the IR graph contains no raw `.log` nodes.

The scientific raw-log domain is positive inputs. The IR denotation reports `Except.error` for
nonpositive values, while the pure lowered closure applies `Tensor.logSpec` to every input. The
end-to-end semantic-equivalence theorem therefore excludes every `.log` node. Protecting its input
with a positive clamp or a softplus-based construction can avoid the domain failure, but does not
discharge this syntactic predicate; that graph needs a separate domain-aware argument.
-/
def NoRawLog (g : NN.IR.Graph) : Prop :=
  ∀ i n, g.getNode i = .ok n → n.kind ≠ .log

/--
Discharge `NoRawLog` from a statement about the node array.

`NoRawLog` quantifies over node *ids* through `Graph.getNode`, which is the right shape for the
recursive lowering proof but an annoying shape for a caller holding a concrete graph. Every id that
`getNode` accepts reads an element of `g.nodes`, so a fact about the array is enough, and for a
literal graph the hypothesis is closed by `decide`:

```
theorem myGraphNoRawLog : NoRawLog myGraph :=
  noRawLog_of_forall_mem (by decide)
```

We added this because the alternative was for every user of
`denoteAll_eq_of_lowerToForwardGraph` to redo the same `getNode` case analysis inline.
-/
theorem noRawLog_of_forall_mem {g : NN.IR.Graph}
    (h : ∀ n ∈ g.nodes, n.kind ≠ .log) : NoRawLog g := by
  intro i n hn
  have hmem : n ∈ g.nodes := by
    unfold NN.IR.Graph.getNode at hn
    split at hn
    · contradiction
    · rename_i found hFound
      split at hn
      · contradiction
      · have hfn : found = n := by injection hn
        subst hfn
        exact Array.mem_of_getElem? hFound
  exact h n hmem

/--
If a `do`-chain begins with `throw`, it cannot produce an `.ok` result.

This lemma is used throughout the lowered-correctness proofs to close
impossible branches where lowering would have thrown an error message.
-/
theorem throw_bind_ne_ok {β γ : Type} {msg : String} {k : β → Except String γ} {v : γ}
    (h : (do
      let y ← (throw msg : Except String β)
      k y) = Except.ok v) : False := by
  simp [throw_eq_error] at h

/--
Optional lookup in `TorchLean.TensorPack.toShapeErasedArray` agrees with indexing
the underlying typed context.

This is the main bridge between the typed runtime context and the untyped IR value table.
-/
theorem toShapeErasedArray_getElem?
    {α : Type} [TorchLean.Storage α] {ss : List Shape}
    (ctx : TorchLean.TensorPack α ss) (i : Fin ss.length) :
    (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx)[i.1]? =
      some (Spec.SomeTensor.ofTensor
        (TorchLean.TensorPack.get (α := α) (ss := ss) ctx i)) := by
  let arr := TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx
  have hi : i.1 < arr.size := by
    exact Nat.lt_of_lt_of_eq i.2
      (TorchLean.TensorPack.size_toShapeErasedArray (α := α) (ss := ss) ctx).symm
  rw [show (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx)[i.1]? =
      some arr[i.1] by
    simp [arr]]
  congr 1
  simpa [arr] using
    (TorchLean.TensorPack.get_toShapeErasedArray (α := α) (ss := ss) ctx i)

/--
Optional lookup in `TorchLean.TensorPack.toShapeErasedArray` by a typed `Idx` agrees
with `getIdx` on the
underlying `TorchLean.TensorPack`.

This packages `toShapeErasedArray_getElem?` into the repository’s `Idx` wrapper.
-/
theorem toShapeErasedArray_getIdx?
    {α : Type} [TorchLean.Storage α] {ss : List Shape} {s : Shape}
    (ctx : TorchLean.TensorPack α ss) (idx : Idx ss s) :
    (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx)[idx.i.1]? =
      some (Spec.SomeTensor.ofTensor (getIdx (α := α) (xs := ctx) idx)) := by
  cases idx with
  | mk i h =>
      -- Reduce to the `Fin`-indexed lemma and then specialize with the stored shape equality.
      cases h
      simpa [getIdx, Tensor.castShape] using
        (toShapeErasedArray_getElem? (α := α) (ss := ss) ctx i)

/-- `Graph.expectShape` succeeds on a `Spec.SomeTensor` built with the same shape. -/
@[simp] theorem Graph.expectShape_mk {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (t : Tensor α s) :
    NN.IR.Graph.expectShape (α := α) (expected := s) (Spec.SomeTensor.mk (α := α) s t) = .ok t := by
  simp [NN.IR.Graph.expectShape]
  rfl

/-- `Graph.expectShape` transports a shape-erased tensor when its stored shape equals the requested
one. -/
theorem Graph.expectShape_mk_of_eq {α : Type} [TorchLean.Storage α] [Context α]
    {s t : Shape} (h : s = t) (x : Tensor α s) :
    NN.IR.Graph.expectShape (α := α) (expected := t) (Spec.SomeTensor.mk (α := α) s x) =
      .ok (x.castShape h) := by
  subst t
  simp [Tensor.castShape]

attribute [grind =] Graph.expectShape_mk throw_eq_error
  toShapeErasedArray_getElem? toShapeErasedArray_getIdx?

/--
`normalizeNodeOutput` accepts a value whose stored shape equals the declared output shape and
transports the tensor along that equality.

Stating the cast with `Tensor.castShape` lets the per-op proofs compare it with the lowered closure
through `Tensor.eqRec_eq_cast_shape` and proof irrelevance.
-/
theorem normalizeNodeOutput_mk_of_eq {α : Type} [TorchLean.Storage α] [Context α]
    {s : Shape} (i : Nat) (n : NN.IR.Node) (t : Tensor α s) (h : s = n.outShape) :
    NN.IR.Graph.normalizeNodeOutput (α := α) i n (Spec.SomeTensor.mk (α := α) s t) =
      .ok (Spec.SomeTensor.mk (α := α) n.outShape (Tensor.castShape t h)) := by
  subst h
  simp [NN.IR.Graph.normalizeNodeOutput, Pure.pure, Except.pure]

/-- Reference evaluation uses the checked matmul layout, including broadcast and vector cases. -/
theorem evalAt_matmul_dims_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α))
    (i : Nat) (n : NN.IR.Node) (aId bId : Nat) (dims : OpContracts.MatmulDims)
    (aT : Tensor α dims.leftShape) (bT : Tensor α dims.rightShape)
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul)
    (hp : binaryParents? n.parents = some (aId, bId))
    (hDims : OpContracts.matmulDims dims.leftShape dims.rightShape = .ok dims)
    (hGetA : vals[aId]? = some (Spec.SomeTensor.mk (α := α) dims.leftShape aT))
    (hGetB : vals[bId]? = some (Spec.SomeTensor.mk (α := α) dims.rightShape bT))
    (hOut : dims.outShape = n.outShape) :
    NN.IR.Graph.evalAt (α := α) g payload input vals i =
      .ok (Spec.SomeTensor.mk (α := α) n.outShape
        (hOut ▸ NN.IR.Graph.matmulWithDims dims aT bT)) := by
  simp only [NN.IR.Graph.evalAt, hN, NN.IR.Graph.evalNode, NN.IR.Graph.evalNodeRaw,
    hk, binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp,
    NN.IR.Graph.getParentValue, hGetA, hGetB, hDims, Graph.expectShape_mk,
    Spec.SomeTensor.ofTensor,
    NN.IR.Graph.normalizeNodeOutput, hOut, dite_true, Pure.pure, Except.pure,
    Bind.bind, Except.bind]

/--
`NN.IR.Graph.evalAt` for either axis-reduction node in a well-typed success case.

This helper records the selected reduction term produced by the IR evaluator once:
- the parent has the expected shape `s`,
- the axis validity check succeeds, and
- the node's declared `outShape` matches `shapeAfterSum s axis`.

The final cast to `n.outShape` comes from the `evalAt` "shape-tag normalization" step.
-/
theorem evalAt_axisReduction_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    (operation : AxisReductionKind)
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α))
    (i : Nat) (n : NN.IR.Node) (pId : Nat) (axis : Nat)
    (s : Shape) (pT : Tensor α s) (hAxisPf : PLift (Shape.NonemptyAxis axis s))
    (hN : g.getNode i = .ok n) (hk : n.kind = operation.toOpKind axis)
    (hp : unaryParent? n.parents = some pId)
    (hGet : vals[pId]? = some (Spec.SomeTensor.mk (α := α) s pT))
    (hAxis : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxisPf)
    (hOut : TorchLean.Tensor.shapeAfterSum s axis = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (Spec.SomeTensor.mk (α := α) n.outShape
        (hOut ▸ operation.denote axis pT hAxisPf.down)) := by
  cases operation <;>
    simp [AxisReductionKind.toOpKind, AxisReductionKind.denote, NN.IR.Graph.evalAt,
      NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, throw_eq_error, hAxis, hOut, Pure.pure, Except.pure]

/-- Repackage a lowered `State` as an `ForwardGraph` so we can call its evaluator helpers. -/
def execOfState {α : Type} [TorchLean.Storage α]
    (inShape : Shape) (st : State α inShape) : ForwardGraph α :=
  { inShape := inShape, ss := st.1, body := st.2 }

/-- Evaluate the lowered prefix state and convert its typed runtime context into an IR-style table.
  -/
def denoteAllState {α : Type} [TorchLean.Storage α] [Context α] (inShape : Shape)
    (st : State α inShape) (x : Tensor α inShape) : Array (Spec.SomeTensor α) :=
  ForwardGraph.denoteAll (α := α) (e := execOfState (α := α) inShape st) x

/--
`denoteAllState` commutes with extending the SSA graph by one node (`ForwardData.snoc`).

This is the key step for proving that the lowering pass’s prefix-building loop stays in semantic
equivalence with the IR denotation table.
-/
theorem denoteAllState_snoc {α : Type} [TorchLean.Storage α] [Context α]
    {inShape : Shape} {ss : List Shape} {τ : Shape}
    (gd : ForwardData α [inShape] ss)
    (nodeData : ForwardNode α ([inShape] ++ ss) τ)
    (x : Tensor α inShape) :
    let st : State α inShape := ⟨ss, gd⟩
    let st' : State α inShape := ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩
    denoteAllState (α := α) inShape st' x =
      (denoteAllState (α := α) inShape st x).push
        (Spec.SomeTensor.mk (α := α) τ
          (nodeData.eval (ForwardData.eval (ss := ss) gd (.cons x .nil)))) := by
  simp [denoteAllState, execOfState, ForwardGraph.denoteAll, ForwardGraph.eval]

/-- A successfully checked typed index retains the numeric IR parent id. -/
theorem mkIdx_ok_i_eq {inShape : Shape} {ss : List Shape} {id : Nat} {s : Shape}
    {idx : Idx ([inShape] ++ ss) s}
    (h : mkIdx (inShape := inShape) (ss := ss) id s = .ok idx) :
    idx.i.1 = id := by
  unfold mkIdx at h
  -- After unfolding, the bound check is expressed via `id ≤ ss.length` (since the ctx is `inShape
  -- :: ss`).
  by_cases hBound : id ≤ ss.length
  · have hLt : id < (inShape :: ss).length := by
      simpa using Nat.lt_succ_of_le hBound
    simp [hBound] at h
    by_cases hShape : (inShape :: ss)[id]'hLt = s
    · simp [hShape] at h
      cases h
      rfl
    · simp [hShape] at h
  · simp [hBound] at h

/--
Lookup in `denoteAllState` agrees with `getIdx` when `mkIdx pid s` succeeds.

This is used when proving correctness of the per-node lowering pass step: we translate parent ids
in the IR into typed indices into the forward-graph context.
-/
theorem denoteAllState_get_mkIdx?
    {α : Type} [TorchLean.Storage α] [Context α] {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (x : Tensor α inShape)
    {pid : Nat} {s : Shape} {idx : Idx ([inShape] ++ ss) s}
    (hIdx : mkIdx (inShape := inShape) (ss := ss) pid s = .ok idx) :
    (denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x)[pid]? =
      some (Spec.SomeTensor.mk (α := α) s
        (getIdx (α := α)
          (xs := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil))
          idx)) := by
  -- Unfold `denoteAllState` to the shape-erased context and use the checked lookup theorem.
  have hPid : pid = idx.i.1 :=
    (mkIdx_ok_i_eq (inShape := inShape) (ss := ss) (id := pid) (s := s) (idx := idx) hIdx).symm
  rw [hPid]
  change
    (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := [inShape] ++ ss)
      (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)))[idx.i.1]? =
      some (Spec.SomeTensor.mk (α := α) s
        (getIdx (α := α)
          (xs := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd
            (.cons x .nil)) idx))
  exact toShapeErasedArray_getIdx? (α := α)
    (ctx := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil))
      idx

/-- The lowering context used by `buildFrom` for node `n` at position `i`. -/
abbrev loweringContext {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (inShape : Shape) (ss : List Shape)
    (i : Nat) (n : NN.IR.Node) : NodeLoweringContext α ([inShape] ++ ss) :=
  { graph := g, payload := payload, index := i, node := n,
    parentIdx := fun pid s => mkIdx (inShape := inShape) (ss := ss) pid s }

/--
Semantic equivalence lemma for a lowering step after the typed `nodeData` has been built.

Many operator cases differ only in how they validate parents and construct the forward closure.
Once that closure and the matching `evalAt` fact are available, the tail-of-graph argument is the
same for unary, binary, and shape-changing nodes.
-/
theorem buildFrom_denoteAllFrom_nodeData_exact
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss)
    (i : Nat) (st' : State α inShape) (x : Tensor α inShape)
    (hi : i < g.nodes.size)
    (τ : Shape)
    (nodeData : ForwardNode α ([inShape] ++ ss) τ)
    (hTail :
      NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x) (i := i + 1)
          (vals := denoteAllState (α := α) inShape
            (st := (⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩ : State α inShape)) x) =
        .ok (denoteAllState (α := α) inShape st' x))
    (hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x)
          (i := i) =
        .ok
          (Spec.SomeTensor.mk (α := α) τ
            (nodeData.eval
              (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil))))) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
        (input := Spec.SomeTensor.mk (α := α) inShape x) (i := i)
        (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  unfold NN.IR.Graph.denoteAllFrom
  simp [hi, hEval]
  simpa only [denoteAllState_snoc, ForwardNode.eval] using hTail

/-- Share parent validation, typed lookup and tail composition for unary tensor operations. -/
theorem buildFrom_denoteAllFrom_unary
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hi : i < g.nodes.size)
    (label : String) (operation : Tensor α n.outShape → Tensor α n.outShape)
    (hLower : lowerNode (loweringContext g payload inShape ss i n) =
      lowerUnary (loweringContext g payload inShape ss i n) label operation)
    (hEval : ∀ (pId : Nat) (value : Tensor α n.outShape),
      unaryParent? n.parents = some pId →
      (denoteAllState (α := α) inShape ⟨ss, gd⟩ x)[pId]? =
        some (Spec.SomeTensor.mk (α := α) n.outShape value) →
      NN.IR.Graph.evalAt (α := α) g payload (Spec.SomeTensor.mk inShape x)
        (denoteAllState (α := α) inShape ⟨ss, gd⟩ x) i =
        .ok (Spec.SomeTensor.mk n.outShape (operation value)))
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  unfold buildFrom at hBuild
  simp only [hi, ↓reduceDIte, hN, Except.ok_bind] at hBuild
  rw [hLower] at hBuild
  cases hp : unaryParent? n.parents with
  | none => simp [hp, throw_eq_error] at hBuild
  | some pId =>
      cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
      | error msg => simp [hp, hIdx] at hBuild
      | ok ip =>
          simp [hp, hIdx] at hBuild
          let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
            mkForwardNode (fun values => operation (values.read ip))
          let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc gd nodeData⟩
          have hRec : buildFrom g payload inShape (i + 1) st1 = .ok st' := by
            simpa [st1, nodeData] using hBuild
          apply buildFrom_denoteAllFrom_nodeData_exact g payload gd i st' x hi
            n.outShape nodeData (ih st1 hRec)
          exact hEval pId _ hp (denoteAllState_get_mkIdx? gd x hIdx)

/-- Share binary validation and preservation while keeping the two parent shapes distinct. -/
theorem buildFrom_denoteAllFrom_binary
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (hN : g.getNode i = .ok n) (hi : i < g.nodes.size)
    (label : String) (rightShape : Shape)
    (operation : Tensor α n.outShape → Tensor α rightShape → Tensor α n.outShape)
    (hLower : lowerNode (loweringContext g payload inShape ss i n) =
      lowerBinary (loweringContext g payload inShape ss i n) label rightShape operation)
    (hEval : ∀ (aId bId : Nat) (left : Tensor α n.outShape) (right : Tensor α rightShape),
      binaryParents? n.parents = some (aId, bId) →
      (denoteAllState (α := α) inShape ⟨ss, gd⟩ x)[aId]? =
        some (Spec.SomeTensor.mk (α := α) n.outShape left) →
      (denoteAllState (α := α) inShape ⟨ss, gd⟩ x)[bId]? =
        some (Spec.SomeTensor.mk (α := α) rightShape right) →
      NN.IR.Graph.evalAt (α := α) g payload (Spec.SomeTensor.mk inShape x)
        (denoteAllState (α := α) inShape ⟨ss, gd⟩ x) i =
        .ok (Spec.SomeTensor.mk n.outShape (operation left right)))
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  unfold buildFrom at hBuild
  simp only [hi, ↓reduceDIte, hN, Except.ok_bind] at hBuild
  rw [hLower] at hBuild
  cases hp : binaryParents? n.parents with
  | none => simp [hp, throw_eq_error] at hBuild
  | some parentIds =>
      rcases parentIds with ⟨aId, bId⟩
      cases hIa : mkIdx (inShape := inShape) (ss := ss) aId n.outShape with
      | error msg => simp [hp, hIa] at hBuild
      | ok ia =>
          cases hIb : mkIdx (inShape := inShape) (ss := ss) bId rightShape with
          | error msg => simp [hp, hIa, hIb] at hBuild
          | ok ib =>
              simp [hp, hIa, hIb] at hBuild
              let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                mkForwardNode (fun values =>
                  operation (values.read ia) (values.read ib))
              let st1 : State α inShape := ⟨ss ++ [n.outShape], .snoc gd nodeData⟩
              have hRec : buildFrom g payload inShape (i + 1) st1 = .ok st' := by
                simpa [st1, nodeData] using hBuild
              apply buildFrom_denoteAllFrom_nodeData_exact g payload gd i st' x hi
                n.outShape nodeData (ih st1 hRec)
              exact hEval aId bId _ _ hp (denoteAllState_get_mkIdx? gd x hIa)
                (denoteAllState_get_mkIdx? gd x hIb)

end IRExec
end Autograd
end Runtime
