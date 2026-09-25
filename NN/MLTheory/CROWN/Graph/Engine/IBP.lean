/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Interval Bound Propagation

This module runs the flat graph IBP pass. It computes one interval box per node from input boxes,
constant tensors, and per-op interval transfer rules. The proof layer states the topological and
shape hypotheses; this executable pass is the checker-facing computation they refer to.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]
variable [NonlinearBoundOps α]

open BoundOps

/-- Compute one interval transfer. Missing parents and unavailable transfers return `none`. -/
@[expose] def ibpStepNodeAt? (nodes : Array Node) (ps : ParamStore α)
    (boxes : Array (Option (FlatBox α))) (id : Nat) (node : Node) : Option (FlatBox α) := do
  let get (pid : Nat) := (boxes[pid]?).join
  match node.kind with
  | .input => ps.inputBoxes[id]?
  | .const _ => do
    let v ← ps.constVals[id]?
    some { dim := v.n, lo := v.v, hi := v.v }
  | .detach | .reshape .. | .flatten .. => do
    let p ← unaryParent? node.parents
    get p
  | .randUniform _ | .bernoulliMask _ =>
    let d := node.outShape.size
    some { dim := d
           lo := Tensor.full (α := α) (.dim d .scalar) 0
           hi := Tensor.full (α := α) (.dim d .scalar) 1 }
  | .add => do
    let (p, q) ← binaryParents? node.parents
    let x ← get p
    let y ← get q
    if x.dim = y.dim then some (boxAdd x y) else none
  | .sub => do
    let (p, q) ← binaryParents? node.parents
    let x ← get p
    let y ← get q
    if x.dim = y.dim then some (boxSub x y) else none
  | .abs => do
    let p ← unaryParent? node.parents
    return boxAbs (← get p)
  | .sqrt => do
    let p ← unaryParent? node.parents
    boxSqrt? (← get p)
  | .inv => do
    let p ← unaryParent? node.parents
    boxInv? (← get p)
  | .maxElem => do
    let (p, q) ← binaryParents? node.parents
    let x ← get p
    let y ← get q
    if x.dim = y.dim then some (boxMaxElem x y) else none
  | .minElem => do
    let (p, q) ← binaryParents? node.parents
    let x ← get p
    let y ← get q
    if x.dim = y.dim then some (boxMinElem x y) else none
  | .maxPool config => do
    let p ← unaryParent? node.parents
    let parent ← nodes[p]?
    ibpMonotoneSomeTensor? parent.outShape node.outShape
      (NN.IR.Graph.evalMaxPool (α := α) config) (← get p)
  | .avgPool config => do
    let p ← unaryParent? node.parents
    let parent ← nodes[p]?
    ibpAvgPool? config parent.outShape node.outShape (← get p)
  | .broadcastTo s₁ s₂ => do
    let p ← unaryParent? node.parents
    ibpBroadcastTo s₁ s₂ (← get p)
  | .reduceSum axis => do
    let p ← unaryParent? node.parents
    let parent ← nodes[p]?
    ibpReduceSumAxis axis (← get p) parent.outShape
  | .reduceMean axis => do
    let p ← unaryParent? node.parents
    let parent ← nodes[p]?
    ibpReduceMeanAxis axis (← get p) parent.outShape
  | .relu => do
    let p ← unaryParent? node.parents
    return boxRelu (← get p)
  | .linear => do
    let p ← unaryParent? node.parents
    ibpLinear id ps (← get p)
  | .matmul =>
    match node.parents with
    | #[p, q] => do
      let left ← nodes[p]?
      let right ← nodes[q]?
      ibpBinaryMatmul? left.outShape right.outShape (← get p) (← get q)
    | #[p] => do
      ibpMatmul id ps (← get p)
    | _ => none
  | .transpose axis₁ axis₂ => do
    let p ← unaryParent? node.parents
    let parent ← nodes[p]?
    let transposeOp := fun value => do
      let perm ← OpContracts.transposePerm value.shape.rank axis₁ axis₂
      NN.IR.Graph.permuteSomeTensor (α := α) value perm
    ibpMonotoneSomeTensor? parent.outShape node.outShape transposeOp (← get p)
  | .permute perm => do
    let p ← unaryParent? node.parents
    let parent ← nodes[p]?
    ibpMonotoneSomeTensor? parent.outShape node.outShape
      (fun value => NN.IR.Graph.permuteSomeTensor (α := α) value perm) (← get p)
  | .mulElem => do
    let (p, q) ← binaryParents? node.parents
    boxMulElem (← get p) (← get q)
  | .sum => do
    let p ← unaryParent? node.parents
    return boxSum (← get p)
  | .mseLoss => do
    let (p, q) ← binaryParents? node.parents
    let y ← get p
    let t ← get q
    if y.dim = t.dim then boxMean? (boxSquare (boxSub y t)) else none
  | .conv config => do
    let p ← unaryParent? node.parents
    let parent ← nodes[p]?
    ibpConvNode config parent.outShape node.outShape id ps (← get p)
  | .batchNormEval channelAxis channels => do
    let p ← unaryParent? node.parents
    let parent ← nodes[p]?
    let config ← ps.batchNormEval[id]?
    let expected ←
      (OpContracts.inferBatchNormEvalOutShape channelAxis channels parent.outShape).toOption
    if expected != node.outShape || config.c != channels then none else
      ibpBatchNormEval? parent.outShape channelAxis config (← get p)
  | .exp => do
    let p ← unaryParent? node.parents
    boxUnaryEnclosure? NonlinearBoundOps.expBounds (← get p)
  | .log => do
    let p ← unaryParent? node.parents
    let input ← get p
    let lo := getDimScalarFn (α := α) input.lo
    if (List.finRange input.dim).all (fun i => decide (0 < Tensor.item (lo i))) then
      boxUnaryEnclosure? NonlinearBoundOps.logBounds input
    else none
  | .concat axis => concatNodeBoxes? nodes boxes node axis
  | .layernorm axis => do
    if !crownNodeSemanticsSupported (α := α) nodes ps id then none else do
      let p ← unaryParent? node.parents
      let input ← get p
      let s := node.outShape
      if input.dim = s.size then
        match ps.layerNorm[id]? with
        | none => ibpLayerNormBox? s input axis
        | some parameters => ibpLayerNormPayloadBox? s axis parameters input
      else none
  | .softmax axis => do
    if !crownNodeSemanticsSupported (α := α) nodes ps id then none else do
      let p ← unaryParent? node.parents
      let input ← get p
      if input.dim = node.outShape.size then
        some (ibpSoftmaxRange (α := α) node.outShape axis input.dim)
      else none
  | .hardMaskedSoftmax mask => do
    let p ← unaryParent? node.parents
    let input ← get p
    let s := node.outShape
    if hdim : input.dim = s.size then
      if hshape : mask.shape = s then
        let decoded ← (NN.IR.HardMask.toTensor? mask).toOption
        let allowed : Tensor Bool s := hshape ▸ decoded
        let sFlat : Shape := .dim input.dim .scalar
        have hsize : sFlat.size = s.size := by simp [sFlat, Spec.Shape.size, hdim]
        let xLo : Tensor α s := Tensor.reshapeSpec input.lo hsize
        let xHi : Tensor α s := Tensor.reshapeSpec input.hi hsize
        let (lo, hi) := ibpHardMaskedSoftmaxLastTensor xLo xHi allowed
        some { dim := s.size, lo := Tensor.flattenSpec lo, hi := Tensor.flattenSpec hi }
      else none
    else none
  | .tanh => do
    let p ← unaryParent? node.parents
    boxUnaryEnclosure? NonlinearBoundOps.tanhBounds (← get p)
  | .sigmoid => do
    let p ← unaryParent? node.parents
    boxUnaryEnclosure? NonlinearBoundOps.sigmoidBounds (← get p)
  | .softplus => do
    let p ← unaryParent? node.parents
    boxSoftplus? (← get p)
  | .safeLog => do
    let (p, q) ← binaryParents? node.parents
    boxSafeLog? (← get p) (← get q)
  | .sin => do
    let p ← unaryParent? node.parents
    boxUnaryEnclosure? NonlinearBoundOps.sinBounds (← get p)
  | .cos => do
    let p ← unaryParent? node.parents
    boxUnaryEnclosure? NonlinearBoundOps.cosBounds (← get p)

/-- Store the result of one transfer, clearing any previous result if the transfer fails. -/
@[expose] def propagateIBPNodeAt (nodes : Array Node) (ps : ParamStore α)
    (boxes : Array (Option (FlatBox α))) (id : Nat) (node : Node) : Array (Option (FlatBox α)) :=
  boxes.set! id (ibpStepNodeAt? nodes ps boxes id node)

/-- Propagate one node. Missing nodes or parent boxes leave the result unresolved. -/
@[expose] def propagateIBPNode (nodes : Array Node) (ps : ParamStore α)
    (boxes : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
  match nodes[id]? with
  | some node => propagateIBPNodeAt nodes ps boxes id node
  | none => boxes.set! id none

/-- Run an IBP pass over the whole graph. Caller seeds inputs via ParamStore.inputBoxes.

The body is exposed so that the proof layer can relate this pass to `CertSoundness.runIBP?`. -/
@[expose] def runIBP (g : Graph) (ps : ParamStore α) : Array (Option (FlatBox α)) :=
  let init := Array.replicate g.nodes.size none
  if crownGraphSemanticsSupported (α := α) g ps then
    (List.finRange g.nodes.size).foldl (fun acc i => propagateIBPNode (α:=α) g.nodes ps acc i)
      init
  else
    init

end NN.MLTheory.CROWN.Graph
