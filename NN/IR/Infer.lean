/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.OpContracts
public import NN.Spec.Core.TensorReductionShape

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
we can check (e.g. input/output are vectors).

This is the single source of truth for `Graph.checkShapes`: adding a new `OpKind` should extend
this match first, then the semantics/export/verification passes can rely on the same contract.

PyTorch analogy:
- `inferNodeOutShape` corresponds to shape propagation used when validating an FX graph.
- Where the true output shape depends on parameters, this module performs contract checking rather
  than attempting to read those parameters.

References / related systems:
- PyTorch FX (graph representation): https://pytorch.org/docs/stable/fx.html
- ONNX shape inference: https://onnx.ai/onnx/shape_inference.html
-/

@[expose] public section


namespace NN.IR

open _root_.Spec
open _root_.Spec.Tensor

namespace Infer

/-!
## Node-local inference

Most IR ops are “shape transparent” (elementwise, permute, etc.). A few need special handling:
- `matmul` has rank-sensitive rules (2D and a limited 3D batched case),
- `concat` needs to merge multiple parents along an axis,
- pooling/conv ops use centralized CHW arithmetic from `OpContracts`.
-/

/--
Infer the output shape of a node from its kind + parent shapes.

This function is used by `Graph.checkInferredShapes` below.
-/
def inferNodeOutShape (n : Node) (parentShapes : List Shape) : Except String Shape := do
  match n.kind with
  | .input =>
      -- Nothing to infer: the input's shape is part of the graph interface.
      pure n.outShape
  | .const valueShape =>
      pure valueShape
  | .permute perm =>
      match parentShapes with
      | [s] =>
          match Spec.Shape.permute? s perm with
          | some s' => pure s'
          | none => throw s!"permute: invalid permutation {repr perm} for shape {repr s}"
      | _ => throw "permute: expected 1 parent"
  | .detach =>
      match parentShapes with
      | [s] => pure s
      | _ => throw "detach: expected 1 parent"
  | .randUniform _seed =>
      match parentShapes with
      | [] => pure n.outShape
      | _ => throw "rand_uniform: expected 0 parents"
  | .bernoulliMask _seed =>
      match parentShapes with
      | [.scalar] => pure n.outShape
      | [s] => throw s!"bernoulli_mask: expected scalar keepProb parent, got {repr s}"
      | _ => throw "bernoulli_mask: expected 1 parent"
  | .add | .sub | .mul_elem =>
      match parentShapes with
      | [a, b] =>
          if a = b then pure a
          else throw s!"{n.kind.tag}: shape mismatch: {repr a} vs {repr b}"
      | _ => throw s!"{n.kind.tag}: expected 2 parents"
  | .abs | .sqrt =>
      match parentShapes with
      | [s] => pure s
      | _ => throw s!"{n.kind.tag}: expected 1 parent"
  | .maxElem | .minElem =>
      match parentShapes with
      | [a, b] =>
          if a = b then pure a
          else throw s!"{n.kind.tag}: shape mismatch: {repr a} vs {repr b}"
      | _ => throw s!"{n.kind.tag}: expected 2 parents"
  | .maxPool2d kH kW stride =>
      match parentShapes with
      | [s] => OpContracts.inferPool2dCHWOutShape "max_pool2d" kH kW stride s
      | _ => throw "max_pool2d: expected 1 parent"
  | .maxPool2dPad kH kW stride padding =>
      match parentShapes with
      | [s] => OpContracts.inferPool2dCHWOutShapePad "max_pool2d_pad" kH kW stride padding s
      | _ => throw "max_pool2d_pad: expected 1 parent"
  | .avgPool2d kH kW stride =>
      match parentShapes with
      | [s] => OpContracts.inferPool2dCHWOutShape "avg_pool2d" kH kW stride s
      | _ => throw "avg_pool2d: expected 1 parent"
  | .avgPool2dPad kH kW stride padding =>
      match parentShapes with
      | [s] => OpContracts.inferPool2dCHWOutShapePad "avg_pool2d_pad" kH kW stride padding s
      | _ => throw "avg_pool2d_pad: expected 1 parent"
  | .broadcastTo s₁ s₂ =>
      match parentShapes with
      | [s] =>
          if s = s₁ then pure s₂
          else throw s!"broadcastTo: parent shape mismatch: expected {repr s₁}, got {repr s}"
      | _ => throw "broadcastTo: expected 1 parent"
  | .reduceSum axis =>
    match parentShapes with
      | [s] => OpContracts.checkAxisValid axis s *> pure (Tensor.shapeAfterSum s axis)
      | _ => throw "reduce_sum: expected 1 parent"
  | .reduceMean axis =>
    match parentShapes with
      | [s] => OpContracts.checkAxisValid axis s *> pure (Tensor.shapeAfterSum s axis)
      | _ => throw "reduce_mean: expected 1 parent"
  | .sum =>
      match parentShapes with
      | [_] => pure .scalar
      | _ => throw "sum: expected 1 parent"
  | .matmul =>
      match parentShapes with
      | [a, b] => OpContracts.inferMatmulOutShape a b
      | _ => throw "matmul: expected 2 parents"
  | .linear =>
      -- `OpKind.linear` does not record dimensions. PyTorch's `F.linear` acts on the last
      -- dimension and preserves any leading batch/sequence dimensions, so the shape checker
      -- validates that contract and accepts the declared output last dimension.
      match parentShapes with
      | [s] =>
          let inDims := Shape.toList s
          let outDims := Shape.toList n.outShape
          match inDims.reverse, outDims.reverse with
          | _inLast :: inPrefixRev, _outLast :: outPrefixRev =>
              if inPrefixRev = outPrefixRev then
                pure n.outShape
              else
                throw <|
                  s!"linear: leading dimensions must be preserved: input={repr s}, " ++
                  s!"outShape={repr n.outShape}"
          | _, _ =>
              throw s!"linear: expected rank≥1 input/output, got input={repr s}, out={repr n.outShape}"
      | _ => throw "linear: expected 1 parent"
  | .conv2d inC outC kH kW stride padding =>
      match parentShapes with
      | [s] => OpContracts.inferConv2dCHWOutShape inC outC kH kW stride padding s
      | _ => throw "conv2d: expected 1 parent"
  | .relu | .tanh | .sigmoid | .exp | .log | .inv | .sin | .cos =>
      match parentShapes with
      | [s] => pure s
      | _ => throw s!"{n.kind.tag}: expected 1 parent"
  | .softmax axis =>
      match parentShapes with
      | [s] => OpContracts.checkAxisValid axis s *> pure s
      | _ => throw "softmax: expected 1 parent"
  | .layernorm axis =>
      match parentShapes with
      | [s] =>
          -- LayerNorm preserves shape. We only validate that `axis` is in bounds; the semantics
          -- interprets it as a *suffix* normalization region (see `OpContracts.layerNorm2DParams`).
          OpContracts.checkAxisValid axis s
          pure s
      | _ => throw "layernorm: expected 1 parent"
  | .reshape inS outS =>
      match parentShapes with
      | [s] =>
          if s != inS then
            throw s!"reshape: parent shape mismatch: expected {repr inS}, got {repr s}"
          if Shape.size inS != Shape.size outS then
            throw s!"reshape: numel mismatch: {Shape.size inS} vs {Shape.size outS}"
          pure outS
      | _ => throw "reshape: expected 1 parent"
  | .flatten s =>
      match parentShapes with
      | [s'] =>
          if s' != s then
            throw s!"flatten: parent shape mismatch: expected {repr s}, got {repr s'}"
          pure (ShapeUtil.flattenOutShape s)
      | _ => throw "flatten: expected 1 parent"
  | .concat axis =>
      OpContracts.inferConcatOutShape axis parentShapes
  | .swap_first_two =>
      match parentShapes with
      | [s] =>
          match ShapeUtil.swapFirstTwoShape? s with
          | some t => pure t
          | none => throw s!"swap_first_two: expected rank≥2, got {repr s}"
      | _ => throw "swap_first_two: expected 1 parent"
  | .transpose3dLastTwo =>
      match parentShapes with
      | [s] =>
          match ShapeUtil.transpose3dLastTwoShape? s with
          | some t => pure t
          | none => throw s!"transpose3d_last_two: expected rank=3 with scalar base, got {repr s}"
      | _ => throw "transpose3d_last_two: expected 1 parent"
  | .mseLoss =>
      match parentShapes with
      | [a, b] =>
          if a = b then pure .scalar
          else throw s!"mse_loss: yhat/target shape mismatch: {repr a} vs {repr b}"
      | _ => throw "mse_loss: expected 2 parents"

end Infer

namespace Graph

/--
Infer shapes for every node (in topo/id order) and check that `Node.outShape` matches.

This is meant as a compiler/back-end sanity check and as a clean IR invariant for the docs:
well-formed graphs have *self-consistent declared shapes*.
-/
def checkInferredShapes (g : Graph) : Except String Unit := do
  g.checkWellFormed
  let mut inferred : Array Shape := #[]
  for i in [0:g.nodes.size] do
    let n ← g.getNode i
    let parentShapes := n.parents.map (fun pid => inferred[pid]!)
    let out ← Infer.inferNodeOutShape n parentShapes
    if out != n.outShape then
      throw <|
        s!"IR graph: node {i}: outShape mismatch: inferred={repr out}, " ++
          s!"declared={repr n.outShape} ({n.summary})"
    inferred := inferred.push out

end Graph

end NN.IR
