/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.HardMask
public import NN.IR.OpContracts
public import NN.Spec.Core.TensorReductionShape.Reductions

/-!
# Shape Inference

Shape inference and consistency checking for `NN.IR.Graph`.

`NN.IR.Node` stores an `outShape` field because many consumers want shape metadata to be available
without re-running inference (pretty printers, exporters, verifiers, etc.).

This module provides an independent shape inference/checking procedure that recomputes the expected
output shape of each node from:
- the node's `OpKind` payload (when present), and
- the parent nodes' output shapes.

For parameterized ops whose output shape depends on external parameters (notably `OpKind.linear`),
we treat the node's declared `outShape` as an input to the checker and validate the local contracts
we can check (e.g. a linear layer preserves all leading input dimensions).

`Graph.checkShapes` uses these rules directly. Adding a new `OpKind` should extend this match before
the semantics, export, and verification passes rely on its shape contract, and should route any
nontrivial shape arithmetic through `NN.IR.OpContracts` so that `NN.IR.Semantics` computes the
same shape; `NN.IR.ShapeSoundness` proves that agreement operation by operation.

PyTorch analogy:
- `nodeOutShape` corresponds to shape propagation used when validating an FX graph.
- Where the true output shape depends on parameters, this module performs contract checking rather
  than attempting to read those parameters.

References / related systems:
- PyTorch FX (graph representation): https://pytorch.org/docs/stable/fx.html
- ONNX shape inference: https://onnx.ai/onnx/shape_inference.html
-/

@[expose] public section


namespace NN.IR

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

namespace Infer

/-!
## Node-local inference

Most IR ops are “shape transparent” (elementwise, permute, etc.). A few need special handling:
- `matmul` preserves an arbitrary shared leading shape around its final matrix axes,
- `concat` needs to merge multiple parents along an axis,
- pooling and convolution use centralized rank-polymorphic spatial arithmetic from `OpContracts`.
-/

/-- Read the sole parent shape of a unary operation. -/
def expectUnaryParent (tag : String) (parents : Array Shape) : Except String Shape :=
  if h : parents.size = 1 then
    .ok (parents[0]'(by simp [h]))
  else
    .error s!"{tag}: expected 1 parent"

/-- Read both parent shapes of a binary operation. -/
def expectBinaryParents (tag : String) (parents : Array Shape) : Except String (Shape × Shape) :=
  if h : parents.size = 2 then
    .ok (parents[0]'(by simp [h]), parents[1]'(by simp [h]))
  else
    .error s!"{tag}: expected 2 parents"

/--
Infer the output shape of a node from its kind + parent shapes.

This function is used by `Graph.checkShapes` below.
-/
def nodeOutShape (n : Node) (parentShapes : Array Shape) : Except String Shape := do
  match n.kind with
  | .input =>
      -- Nothing to infer: the input's shape is part of the graph interface.
      pure n.outShape
  | .const valueShape =>
      pure valueShape
  | .permute perm =>
      let s ← expectUnaryParent "permute" parentShapes
      match Spec.Shape.permute? s perm.toList with
      | some s' => pure s'
      | none => throw s!"permute: invalid permutation {repr perm} for shape {repr s}"
  | .transpose axis₁ axis₂ =>
      OpContracts.inferTransposeOutShape axis₁ axis₂ (← expectUnaryParent "transpose" parentShapes)
  | .detach =>
      expectUnaryParent "detach" parentShapes
  | .randUniform _seed =>
      if parentShapes.isEmpty then pure n.outShape
      else throw "rand_uniform: expected 0 parents"
  | .bernoulliMask _seed =>
      match ← expectUnaryParent "bernoulli_mask" parentShapes with
      | .scalar => pure n.outShape
      | s => throw s!"bernoulli_mask: expected scalar keepProb parent, got {repr s}"
  | .add | .sub | .mulElem | .maxElem | .minElem =>
      let (a, b) ← expectBinaryParents n.kind.tag parentShapes
      if a = b then pure a
      else throw s!"{n.kind.tag}: shape mismatch: {repr a} vs {repr b}"
  | .abs | .sqrt =>
      expectUnaryParent n.kind.tag parentShapes
  | .maxPool config =>
      OpContracts.inferWindowOutShape "max_pool" config
        (← expectUnaryParent "max_pool" parentShapes)
  | .avgPool config =>
      OpContracts.inferWindowOutShape "avg_pool" config
        (← expectUnaryParent "avg_pool" parentShapes)
  | .broadcastTo s₁ s₂ =>
      let s ← expectUnaryParent "broadcastTo" parentShapes
      if s != s₁ then
        throw s!"broadcastTo: parent shape mismatch: expected {repr s₁}, got {repr s}"
      if Spec.Shape.CanBroadcastTo s₁ s₂ then pure s₂
      else throw s!"broadcastTo: invalid broadcast from {repr s₁} to {repr s₂}"
  | .reduceSum axis =>
      let s ← expectUnaryParent "reduce_sum" parentShapes
      let _ ← OpContracts.checkReductionAxis axis s
      pure (Tensor.shapeAfterSum s axis)
  | .reduceMean axis =>
      let s ← expectUnaryParent "reduce_mean" parentShapes
      let _ ← OpContracts.checkReductionAxis axis s
      pure (Tensor.shapeAfterSum s axis)
  | .sum =>
      let _ ← expectUnaryParent "sum" parentShapes
      pure .scalar
  | .matmul =>
      let (a, b) ← expectBinaryParents "matmul" parentShapes
      OpContracts.inferMatmulOutShape a b
  | .linear =>
      -- `OpKind.linear` does not record dimensions. PyTorch's `F.linear` acts on the last
      -- dimension and preserves any leading batch/sequence dimensions, so the shape checker
      -- validates that contract and accepts the declared output last dimension.
      let s ← expectUnaryParent "linear" parentShapes
      let inDims := Shape.toList s
      let outDims := Shape.toList n.outShape
      match inDims.reverse, outDims.reverse with
      | _inLast :: inPrefixRev, _outLast :: outPrefixRev =>
          if inPrefixRev = outPrefixRev then
            pure n.outShape
          else
            throw <|
              s!"linear: leading dimensions must be preserved: input={repr s}, \
              outShape={repr n.outShape}"
      | _, _ =>
          throw s!"linear: expected rank≥1 input/output, got input={repr s}, out={repr n.outShape}"
  | .conv config =>
      OpContracts.inferConvConfigOutShape "conv" config
        (← expectUnaryParent "conv" parentShapes)
  | .batchNormEval channelAxis channels =>
      OpContracts.inferBatchNormEvalOutShape channelAxis channels
        (← expectUnaryParent "batch_norm_eval" parentShapes)
  | .relu | .tanh | .sigmoid | .softplus | .exp | .log | .inv | .sin | .cos =>
      expectUnaryParent n.kind.tag parentShapes
  | .safeLog =>
      let (s, epsilonShape) ← expectBinaryParents "safe_log" parentShapes
      if epsilonShape = .scalar then pure s
      else throw s!"safe_log: epsilon must be scalar, got {repr epsilonShape}"
  | .softmax axis =>
      let s ← expectUnaryParent "softmax" parentShapes
      let _ ← OpContracts.checkAxisValid axis s
      pure s
  | .hardMaskedSoftmax mask =>
      let s ← expectUnaryParent "hard_masked_softmax" parentShapes
      let _ ← NN.IR.HardMask.validateAs mask s
      pure s
  | .layernorm axis =>
      let s ← expectUnaryParent "layernorm" parentShapes
      let _ ← OpContracts.layerNormMatrixDims axis s
      pure s
  | .reshape inS outS =>
      let s ← expectUnaryParent "reshape" parentShapes
      if s != inS then
        throw s!"reshape: parent shape mismatch: expected {repr inS}, got {repr s}"
      if Spec.Shape.size inS != Spec.Shape.size outS then
        throw s!"reshape: numel mismatch: {Spec.Shape.size inS} vs {Spec.Shape.size outS}"
      pure outS
  | .flatten s =>
      let s' ← expectUnaryParent "flatten" parentShapes
      if s' != s then
        throw s!"flatten: parent shape mismatch: expected {repr s}, got {repr s'}"
      pure (ShapeUtil.flattenOutShape s)
  | .concat axis =>
      OpContracts.inferConcatOutShape axis parentShapes
  | .mseLoss =>
      let (a, b) ← expectBinaryParents "mse_loss" parentShapes
      if a = b then pure .scalar
      else throw s!"mse_loss: yhat/target shape mismatch: {repr a} vs {repr b}"

end Infer

namespace Graph

/--
Look up the already inferred shapes of a node's parents.

Every parent id must index a shape that has already been inferred. On a well-formed graph the
topological-order check guarantees this, and the lookup reports a readable error instead of
indexing out of bounds if a caller skips that check.
-/
def lookupParentShapes (inferred : Array Shape) : List Nat → Option (List Shape)
  | [] => some []
  | pid :: rest => do
      let shape ← inferred[pid]?
      let shapes ← lookupParentShapes inferred rest
      pure (shape :: shapes)

/--
Infer and check the declared output shapes of nodes `i, i+1, ...`, extending the table `inferred`
of shapes already inferred for the nodes below `i`.

The recursion is structural on the remaining node count so proofs can follow it in lockstep with
`Graph.denoteAllFrom`; see `NN.IR.ShapeSoundness`.
-/
def inferShapesFrom (g : Graph) (i : Nat) (inferred : Array Shape) :
    Except String (Array Shape) := do
  if _h : i < g.nodes.size then
    let n ← g.getNode i
    let parentShapes ←
      match lookupParentShapes inferred n.parents.toList with
      | some shapes => pure shapes.toArray
      | none => throw s!"IR graph: node {i}: parent id is not below the node id ({n.summary})"
    let out ← Infer.nodeOutShape n parentShapes
    if out = n.outShape then
      inferShapesFrom g (i + 1) (inferred.push out)
    else
      throw <|
        s!"IR graph: node {i}: outShape mismatch: inferred={repr out}, \
          declared={repr n.outShape} ({n.summary})"
  else
    pure inferred
termination_by g.nodes.size - i
decreasing_by
  exact Nat.sub_succ_lt_self _ _ _h

/--
Infer every node's output shape (in topo/id order) after checking structural well-formedness, and
check that each `Node.outShape` matches. Returns the inferred shapes, one per node.
-/
def inferShapes (g : Graph) : Except String (Array Shape) := do
  g.checkWellFormed
  inferShapesFrom g 0 #[]

/--
Infer shapes for every node (in topo/id order) and check that `Node.outShape` matches.

This is meant as a lowering/backend consistency check and as a clean IR invariant for the docs:
well-formed graphs have *self-consistent declared shapes*. `NN.IR.ShapeSoundness` proves that on a
graph accepted here the reference semantics computes exactly the declared shapes.
-/
def checkShapes (g : Graph) : Except String Unit := do
  let _ ← g.inferShapes
  pure ()

end Graph

end NN.IR
