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

* `packedTensorsOfContext*` lemmas: relate the typed context produced by `ForwardData.eval` to an
  untyped `Array (Spec.SomeTensor α)` (this is what the IR evaluator uses).
* `denoteAllState*` lemmas: package the IR forward evaluator (`ForwardGraph.denoteAll`) in the form
  expected by IR-style semantic equivalence proofs.

These lemmas are infrastructure: they should not encode op-specific logic. Per-op correctness files
(Matmul/Pooling/LayerNorm/MSELoss) should depend on this module and not re-prove these bridges.

## Main definitions

- `throw_bind_ne_ok`: eliminates impossible success branches after `throw`.
- `NoRawLog`: side condition for theorem statements that do not carry the positivity
  precondition required by raw real `log`.
- `noRawLog_of_forall_mem`: discharge `NoRawLog` from a node-array fact, so a concrete graph can
  close it with `decide`.
- `packedTensorsOfContext_*`: typed-context to IR-array bridge lemmas.
- `evalAt_matmul_leading_ok`, `evalAt_axisReduction_ok`: `evalAt` in well-typed success cases.
- `denoteAllState_*` helpers: semantic equivalence bridges between lowered state and IR denotation
  tables.

## Implementation notes

- This module is shared infrastructure: predictable proof contracts
  matter more than clever proof tricks.
- Many lemmas here are proof-irrelevance/indexing bridges; these are repetitive but they remove a
  lot of friction from op-specific proofs.
- Collecting these utilities in one place gives op-specific correctness modules shared rewrite and
  indexing lemmas instead of repeated local proof scripts.
- These files can build slowly because they connect two representations at once: typed
  `TorchLean.TensorPack` contexts on the forward-graph side and dynamically shaped
  `Spec.SomeTensor` arrays on the IR side. Most of the cost is not arithmetic; it is Lean checking
  that shape casts, array indices, and proof-irrelevant casts line up exactly.
- When the same proof pattern appears in multiple operator files, prefer a named lemma with a clear
  contract over another local `simp` script.

## Tags

correctness, infrastructure, tensorpack, dval, bridge-lemmas
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Runtime.Autograd.IRExec.Internal
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
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
        -- `getNode?` is array indexing, so a successful lookup exhibits `n` as a member.
        simpa [NN.IR.Graph.getNode?] using Array.mem_of_getElem? hFound
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

/-- If two unit guards and a tail computation return `.ok`, then the first guard succeeded. -/
theorem exceptUnit_two_bind_first_ok
    {β : Type} {e₁ e₂ : Except String Unit} {next : Except String β} {v : β}
    (h : (do let _ ← e₁; let _ ← e₂; next) = Except.ok v) :
    e₁ = Except.ok () := by
  cases h₁ : e₁ <;> simp [h₁] at h
  rename_i u
  cases u
  rfl

/-- If two unit guards and a tail computation return `.ok`, then the second guard succeeded. -/
theorem exceptUnit_two_bind_second_ok
    {β : Type} {e₁ e₂ : Except String Unit} {next : Except String β} {v : β}
    (h : (do let _ ← e₁; let _ ← e₂; next) = Except.ok v) :
    e₂ = Except.ok () := by
  cases h₁ : e₁ <;> simp [h₁] at h
  rename_i u₁
  cases u₁
  cases h₂ : e₂ <;> simp [h₂] at h
  rename_i u₂
  cases u₂
  rfl

/-- If two unit guards and a tail computation return `.ok`, then the tail returned `.ok`. -/
theorem exceptUnit_two_bind_tail_ok
    {β : Type} {e₁ e₂ : Except String Unit} {next : Except String β} {v : β}
    (h : (do let _ ← e₁; let _ ← e₂; next) = Except.ok v) :
    next = Except.ok v := by
  cases h₁ : e₁ <;> simp [h₁] at h
  rename_i u₁
  cases u₁
  cases h₂ : e₂ <;> simp [h₂] at h
  rename_i u₂
  cases u₂
  exact h

/--
Array indexing is proof-irrelevant.

This is a small technical lemma: in Lean, `xs[i]'h` carries a proof `h : i < xs.size`. Different
proofs should not change the value returned by indexing.
-/
theorem array_getElem_proof_irrel {β : Type}
    (xs : Array β) (i : Nat) (h₁ h₂ : i < xs.size) : xs[i]'h₁ = xs[i]'h₂ := by
  -- `Array.getElem` is implemented via `Array.get` on a `Fin` index, and `Fin` is proof-irrelevant.
  have hFin : (⟨i, h₁⟩ : Fin xs.size) = ⟨i, h₂⟩ := by
    ext
    rfl
  -- Use `Fin` indexing (`xs[j]`) since `Array.get` is not a named constant in Lean 4.
  exact congrArg (fun j : Fin xs.size => xs[j]) hFin

/--
`packedTensorsOfContext` ignores type-level casts of the underlying `TorchLean.TensorPack`.

`ForwardData.eval` introduces a definitional cast when extending contexts; this lemma lets us erase
it before reasoning about the corresponding `Array` of `Spec.SomeTensor`s.
-/
@[simp]
theorem packedTensorsOfContext_cast {α : Type} [TorchLean.Storage α] {ss₁ ss₂ : List Shape}
    (h : ss₁ = ss₂) (ctx : TorchLean.TensorPack α ss₁) :
    packedTensorsOfContext (α := α) (ss := ss₂) (TorchLean.TensorPack.cast (α := α) h ctx) =
      packedTensorsOfContext (α := α) (ss := ss₁) ctx := by
  cases h
  simp [packedTensorsOfContext]

/-- `packedTensorsOfContext` for a snoc’d context corresponds to `Array.push` of the appended
tensor. -/
@[simp]
theorem packedTensorsOfContext_snoc {α : Type} [TorchLean.Storage α] {ss : List Shape} {τ : Shape}
    (ctx : TorchLean.TensorPack α ss) (t : Tensor α τ) :
    packedTensorsOfContext (α := α) (ss := ss ++ [τ])
        (TorchLean.TensorPack.snoc (α := α) (ss := ss) (τ := τ) ctx t) =
      (packedTensorsOfContext (α := α) (ss := ss) ctx).push
        (Spec.SomeTensor.ofTensor t) := by
  simp [packedTensorsOfContext, Spec.SomeTensor.ofTensor]

/--
Optional lookup in `packedTensorsOfContext` agrees with indexing the underlying typed context.

This is the main bridge between the typed runtime context and the untyped IR value table.
-/
theorem packedTensorsOfContext_getElem?
    {α : Type} [TorchLean.Storage α] {ss : List Shape}
    (ctx : TorchLean.TensorPack α ss) (i : Fin ss.length) :
    (packedTensorsOfContext (α := α) (ss := ss) ctx)[i.1]? =
      some (Spec.SomeTensor.ofTensor
        (TorchLean.TensorPack.get (α := α) (ss := ss) ctx i)) := by
  let arr := TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx
  have hi : i.1 < arr.size := by
    exact Nat.lt_of_lt_of_eq i.2
      (TorchLean.TensorPack.size_toShapeErasedArray (α := α) (ss := ss) ctx).symm
  rw [show (packedTensorsOfContext (α := α) (ss := ss) ctx)[i.1]? = some arr[i.1] by
    simp [arr, packedTensorsOfContext]]
  congr 1
  simpa [arr] using
    (TorchLean.TensorPack.get_toShapeErasedArray (α := α) (ss := ss) ctx i)

/--
Optional lookup in `packedTensorsOfContext` by a typed `Idx` agrees with `getIdx` on the
underlying `TorchLean.TensorPack`.

This packages `packedTensorsOfContext_getElem?` into the repository’s `Idx` wrapper.
-/
theorem packedTensorsOfContext_getIdx?
    {α : Type} [TorchLean.Storage α] {ss : List Shape} {s : Shape}
    (ctx : TorchLean.TensorPack α ss) (idx : Idx ss s) :
    (packedTensorsOfContext (α := α) (ss := ss) ctx)[idx.i.1]? =
      some (Spec.SomeTensor.ofTensor (getIdx (α := α) (xs := ctx) idx)) := by
  cases idx with
  | mk i h =>
      -- Reduce to the `Fin`-indexed lemma and then specialize with the stored shape equality.
      cases h
      simpa [getIdx, Tensor.castShape] using
        (packedTensorsOfContext_getElem? (α := α) (ss := ss) ctx i)

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

attribute [grind =] packedTensorsOfContext_cast packedTensorsOfContext_snoc Graph.expectShape_mk
  throw_eq_error array_getElem_proof_irrel
  packedTensorsOfContext_getElem? packedTensorsOfContext_getIdx?

/--
`NN.IR.Graph.evalAt` for a `.matmul` node whose parents share the leading shape
`Shape.ofList leadingRev.reverse` and end in the matrix axes `[rows, inner]` and `[inner, cols]`.

The leading shape is spelled through its reversed dimension list because that is how both the IR
evaluator and `lowerMatmul` recover it from the parent shapes. The lemma records the exact
`NN.IR.Graph.matmulLeading` term produced by the evaluator for any leading shape (plain matrices,
one batch axis, or several batch axes).
-/
theorem evalAt_matmul_leading_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α))
    (i : Nat) (n : NN.IR.Node) (aId bId : Nat) (leadingRev : List Nat) (rows inner cols : Nat)
    (aT : Tensor α ((Shape.ofList leadingRev.reverse).concat [rows, inner]))
    (bT : Tensor α ((Shape.ofList leadingRev.reverse).concat [inner, cols]))
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul)
    (hp : binaryParents? n.parents = some (aId, bId))
    (hGetA : vals[aId]? = some
      (Spec.SomeTensor.mk (α := α) ((Shape.ofList leadingRev.reverse).concat [rows, inner]) aT))
    (hGetB : vals[bId]? = some
      (Spec.SomeTensor.mk (α := α) ((Shape.ofList leadingRev.reverse).concat [inner, cols]) bT))
    (hOut : (Shape.ofList leadingRev.reverse).concat [rows, cols] = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input)
        (vals := vals) (i := i) =
      .ok (Spec.SomeTensor.mk (α := α) n.outShape
        (hOut ▸ NN.IR.Graph.matmulLeading (α := α) (Shape.ofList leadingRev.reverse) aT bT)) := by
  simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
    hN, hk, binaryParentIds_eq_ok_of_binaryParents_eq_some i aId bId n hp,
    NN.IR.Graph.expectShape, hGetA, hGetB, hOut, throw_eq_error, Pure.pure, Except.pure,
    Shape.concat_eq_append, Tensor.eqRec_eq_cast_shape]

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

/-- The two axis reductions supported by lowered IR semantic equivalence. -/
inductive AxisReductionKind where
  | sum
  | mean

/-- Convert a lowered axis-reduction case to its IR operation kind. -/
def AxisReductionKind.toOpKind (operation : AxisReductionKind) (axis : Nat) : NN.IR.OpKind :=
  match operation with
  | .sum => .reduceSum axis
  | .mean => .reduceMean axis

/-- Typed denotation of a lowered axis-reduction case. -/
def AxisReductionKind.denote
    {β : Type} [TorchLean.Storage β] [Context β] {shape : Shape}
    (operation : AxisReductionKind) (axis : Nat) (tensor : Tensor β shape)
    (axisValid : Shape.NonemptyAxis axis shape) : Tensor β (Tensor.shapeAfterSum shape axis) :=
  match operation with
  | .sum => Tensor.reduceSum (α := β) (s := shape) axis tensor
      (axisValid)
  | .mean => Tensor.reduceMean (α := β) (s := shape) axis tensor
      (axisValid)

/--
`NN.IR.Graph.evalAt` for either axis-reduction node in a well-typed success case.

This helper records the exact `Tensor.reduceSum` term produced by the IR evaluator once:
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

/-- `evalAt_axisReduction_ok` specialized to summation. -/
theorem evalAt_reduceSum_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α))
    (i : Nat) (n : NN.IR.Node) (pId : Nat) (axis : Nat)
    (s : Shape) (pT : Tensor α s) (hAxisPf : PLift (Shape.NonemptyAxis axis s))
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceSum axis)
    (hp : unaryParent? n.parents = some pId)
    (hGet : vals[pId]? = some (Spec.SomeTensor.mk (α := α) s pT))
    (hAxis : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxisPf)
    (hOut : TorchLean.Tensor.shapeAfterSum s axis = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (Spec.SomeTensor.mk (α := α) n.outShape
        (hOut ▸ Tensor.reduceSum (α := α) (s := s) axis pT
          (hAxisPf.down))) := by
  exact evalAt_axisReduction_ok .sum g payload input vals i n pId axis s pT hAxisPf
    hN hk hp hGet hAxis hOut

/--
`NN.IR.Graph.evalAt` for a `.reduceMean axis` node, specialized to a well-typed success case.

This is the mean analogue of `evalAt_reduceSum_ok`.
-/
theorem evalAt_reduceMean_ok
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.SomeTensor α)
    (vals : Array (Spec.SomeTensor α))
    (i : Nat) (n : NN.IR.Node) (pId : Nat) (axis : Nat)
    (s : Shape) (pT : Tensor α s) (hAxisPf : PLift (Shape.NonemptyAxis axis s))
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceMean axis)
    (hp : unaryParent? n.parents = some pId)
    (hGet : vals[pId]? = some (Spec.SomeTensor.mk (α := α) s pT))
    (hAxis : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxisPf)
    (hOut : TorchLean.Tensor.shapeAfterSum s axis = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (Spec.SomeTensor.mk (α := α) n.outShape
        (hOut ▸ Tensor.reduceMean (α := α) (s := s) axis pT
          (hAxisPf.down))) := by
  exact evalAt_axisReduction_ok .mean g payload input vals i n pId axis s pT hAxisPf
    hN hk hp hGet hAxis hOut

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

/--
Build a typed runtime index (`Idx`) for a numeric IR parent id.

The forward executor's context is typed by a list of shapes `[inShape] ++ ss`. `mkIdx` checks that:
- `id` is in bounds, and
- the context shape at that position matches the expected shape `s`.
-/

theorem mkIdx_ok_i_eq {inShape : Shape} {ss : List Shape} {id : Nat} {s : Shape}
    {idx : Idx ([inShape] ++ ss) s}
    (h : mkIdx (inShape := inShape) (ss := ss) id s = .ok idx) :
    idx.i.1 = id := by
  classical
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
    (packedTensorsOfContext (α := α) (ss := [inShape] ++ ss)
      (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)))[idx.i.1]? =
      some (Spec.SomeTensor.mk (α := α) s
        (getIdx (α := α)
          (xs := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd
            (.cons x .nil)) idx))
  exact packedTensorsOfContext_getIdx? (α := α)
    (ctx := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil))
      idx

/--
One-step finishing lemma for the `buildFrom`/`denoteAllFrom` semantic equivalence proof.

If we know:
- the tail recursion `i+1` is correct (`hTail`),
- the IR evaluator step at `i` matches the forward-graph node’s `forward` (`hEval`), and
- the forward-graph table at `i` is the previous table plus the pushed node value (`hStep`),
then `denoteAllFrom` at `i` returns the final forward-graph table.
-/
theorem buildFrom_denoteAllFrom_finish
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (i : Nat) (x : Tensor α inShape)
    (hi : i < g.nodes.size)
    (τ : Shape) (nodeData : ForwardNode α ([inShape] ++ ss) τ)
    (st1 st' : State α inShape)
    (ctx : TorchLean.TensorPack α ([inShape] ++ ss))
    (vals0 : Array (Spec.SomeTensor α))
    (input : Spec.SomeTensor α)
    (hTail :
      NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := input) (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
        .ok (denoteAllState (α := α) inShape st' x))
    (hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
          (input := input) (vals := vals0) (i := i) =
        .ok (Spec.SomeTensor.mk (α := α) τ (nodeData.eval ctx)))
    (hStep :
      denoteAllState (α := α) inShape st1 x =
        vals0.push (Spec.SomeTensor.mk (α := α) τ (nodeData.eval ctx))) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
        (input := input) (i := i) (vals := vals0) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  unfold NN.IR.Graph.denoteAllFrom
  simp [hi, hEval]
  simpa [hStep] using hTail
end IRExec
end Autograd
end Runtime
