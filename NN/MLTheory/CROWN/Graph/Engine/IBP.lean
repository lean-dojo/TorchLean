/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base
public import NN.IR.HardMask -- shake: keep

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

/-- IBP propagation for one node, with the node record passed explicitly.

`propagateIBPNode` instantiates `node` with `nodes[id]!`. Taking the record as an argument lets the
proof layer rewrite it to a literal and evaluate the array-pattern matches below definitionally,
which is how the step is identified with its safe counterpart `CertSoundness.certStepNode?`.
-/
@[expose] def propagateIBPNodeAt (nodes : Array Node) (ps : ParamStore α)
    (boxes : Array (Option (FlatBox α))) (id : Nat) (node : Node) : Array (Option (FlatBox α)) :=
  let get! (pid : Nat) := (boxes[pid]!).get!
  match node.kind with
  | .input =>
    match ps.inputBoxes[id]? with
    | some B => boxes.set! id (some B)
    | none   => boxes
  | .const _ =>
    match ps.constVals[id]? with
    | some v => boxes.set! id (some { dim := v.n, lo := v.v, hi := v.v })
    | none   => boxes
  | .detach =>
    match node.parents with
    | #[p1] => boxes.set! id (some (get! p1))
    | _ => boxes
  | .randUniform _ | .bernoulliMask _ =>
    -- Stochastic nodes are treated as *nondeterministic-but-bounded* for verification.
    -- Sound enclosure: U[0,1) ⊆ [0,1], Bernoulli mask ⊆ [0,1].
    let d := node.outShape.size
    let lo := Tensor.full (α := α) (.dim d .scalar) 0
    let hi := Tensor.full (α := α) (.dim d .scalar) 1
    boxes.set! id (some { dim := d, lo := lo, hi := hi })
  | .add =>
    match node.parents with
    | #[p1, p2] => boxes.set! id (some (boxAdd (get! p1) (get! p2)))
    | _ => boxes
  | .sub =>
    match node.parents with
    | #[p1, p2] => boxes.set! id (some (boxSub (get! p1) (get! p2)))
    | _ => boxes
  | .abs =>
    match node.parents with
    | #[p1] => boxes.set! id (some (boxAbs (α := α) (get! p1)))
    | _ => boxes
  | .sqrt =>
    match node.parents with
    | #[p1] =>
      match boxSqrt? (α := α) (get! p1) with
      | some B => boxes.set! id (some B)
      | none => boxes
    | _ => boxes
  | .inv =>
    match node.parents with
    | #[p1] =>
        match boxInv? (α := α) (get! p1) with
        | some B => boxes.set! id (some B)
        | none => boxes
    | _ => boxes
  | .maxElem =>
    match node.parents with
    | #[p1, p2] => boxes.set! id (some (boxMaxElem (α := α) (get! p1) (get! p2)))
    | _ => boxes
  | .minElem =>
    match node.parents with
    | #[p1, p2] => boxes.set! id (some (boxMinElem (α := α) (get! p1) (get! p2)))
    | _ => boxes
  | .maxPool config =>
    match node.parents with
    | #[p1] =>
      match ibpMonotoneSomeTensor? (α := α) nodes[p1]!.outShape node.outShape
          (NN.IR.Graph.evalMaxPool (α := α) config) (get! p1) with
      | some result => boxes.set! id (some result)
      | none => boxes
    | _ => boxes
  | .avgPool config =>
    match node.parents with
    | #[p1] =>
      match ibpMonotoneSomeTensor? (α := α) nodes[p1]!.outShape node.outShape
          (NN.IR.Graph.evalAvgPool (α := α) config) (get! p1) with
      | some result => boxes.set! id (some result)
      | none => boxes
    | _ => boxes
  | .broadcastTo s₁ s₂ =>
    match node.parents with
    | #[p1] =>
      match ibpBroadcastTo (α := α) s₁ s₂ (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .reduceSum axis =>
    match node.parents with
    | #[p1] =>
      let s := nodes[p1]!.outShape
      match ibpReduceSumAxis (α := α) axis (get! p1) s with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .reduceMean axis =>
    match node.parents with
    | #[p1] =>
      let s := nodes[p1]!.outShape
      match ibpReduceMeanAxis (α := α) axis (get! p1) s with
      | some yB => boxes.set! id (some yB)
      | none => boxes
    | _ => boxes
  | .relu =>
    match node.parents with
    | #[p1] => boxes.set! id (some (boxRelu (get! p1)))
    | _ => boxes
  | .linear =>
    match node.parents with
    | #[p1] =>
      match ibpLinear (α:=α) id ps (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none    => boxes
    | _ => boxes
  | .matmul =>
    match node.parents with
    | #[p1, p2] =>
      let A := get! p1
      let B := get! p2
      let sA := nodes[p1]!.outShape
      let sB := nodes[p2]!.outShape
      let dyn2D? : Option (FlatBox α) :=
        match sA, sB with
        | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
          if hk : k = k' then
            match hk with
            | rfl =>
              if hA : A.dim = m * k then
                if hB : B.dim = k * n then
                  let outDim := m * n
                  let loT : Tensor α [outDim] :=
                    Tensor.dim (fun idx =>
                      let t := idx.val
                      let i := t / n
                      let j := t % n
                      let (sumLo, _sumHi) :=
                        (List.range k).foldl (fun (acc : α × α) kk =>
                          let (accLo, accHi) := acc
                          let aLo := getAtOrZero A.lo [i * k + kk]
                          let aHi := getAtOrZero A.hi [i * k + kk]
                          let bLo := getAtOrZero B.lo [kk * n + j]
                          let bHi := getAtOrZero B.hi [kk * n + j]
                          let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                          (addDown accLo pLo, addUp accHi pHi)
                        ) (0, 0)
                      Tensor.scalar sumLo)
                  let hiT : Tensor α [outDim] :=
                    Tensor.dim (fun idx =>
                      let t := idx.val
                      let i := t / n
                      let j := t % n
                      let (_sumLo, sumHi) :=
                        (List.range k).foldl (fun (acc : α × α) kk =>
                          let (accLo, accHi) := acc
                          let aLo := getAtOrZero A.lo [i * k + kk]
                          let aHi := getAtOrZero A.hi [i * k + kk]
                          let bLo := getAtOrZero B.lo [kk * n + j]
                          let bHi := getAtOrZero B.hi [kk * n + j]
                          let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                          (addDown accLo pLo, addUp accHi pHi)
                        ) (0, 0)
                      Tensor.scalar sumHi)
                  some { dim := outDim, lo := loT, hi := hiT }
                else none
              else none
          else none
        | _, _ => none
      let dyn3D? : Option (FlatBox α) :=
        match sA, sB with
        | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
          if hb : b = b' then
            match hb with
            | rfl =>
              if hk : k = k' then
                match hk with
                | rfl =>
                  if hA : A.dim = b * m * k then
                    if hB : B.dim = b * k * n then
                      let outDim := b * m * n
                      let block : Nat := m * n
                      let strideA : Nat := m * k
                      let strideB : Nat := k * n
                      let loT : Tensor α [outDim] :=
                        Tensor.dim (fun idx =>
                          let t := idx.val
                          let bi := t / block
                          let rem := t % block
                          let i := rem / n
                          let j := rem % n
                          let baseA := bi * strideA
                          let baseB := bi * strideB
                          let (sumLo, _sumHi) :=
                            (List.range k).foldl (fun (acc : α × α) kk =>
                              let (accLo, accHi) := acc
                              let aLo := getAtOrZero A.lo [baseA + i * k + kk]
                              let aHi := getAtOrZero A.hi [baseA + i * k + kk]
                              let bLo := getAtOrZero B.lo [baseB + kk * n + j]
                              let bHi := getAtOrZero B.hi [baseB + kk * n + j]
                              let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                              (addDown accLo pLo, addUp accHi pHi)
                            ) (0, 0)
                          Tensor.scalar sumLo)
                      let hiT : Tensor α [outDim] :=
                        Tensor.dim (fun idx =>
                          let t := idx.val
                          let bi := t / block
                          let rem := t % block
                          let i := rem / n
                          let j := rem % n
                          let baseA := bi * strideA
                          let baseB := bi * strideB
                          let (_sumLo, sumHi) :=
                            (List.range k).foldl (fun (acc : α × α) kk =>
                              let (accLo, accHi) := acc
                              let aLo := getAtOrZero A.lo [baseA + i * k + kk]
                              let aHi := getAtOrZero A.hi [baseA + i * k + kk]
                              let bLo := getAtOrZero B.lo [baseB + kk * n + j]
                              let bHi := getAtOrZero B.hi [baseB + kk * n + j]
                              let (pLo, pHi) := intervalMul (α:=α) aLo aHi bLo bHi
                              (addDown accLo pLo, addUp accHi pHi)
                            ) (0, 0)
                          Tensor.scalar sumHi)
                      some { dim := outDim, lo := loT, hi := hiT }
                    else none
                  else none
              else none
          else none
        | _, _ => none
      match dyn2D?, dyn3D? with
      | some yB, _ => boxes.set! id (some yB)
      | none, some yB => boxes.set! id (some yB)
      | none, none => boxes
    | #[p1] =>
      match ibpMatmul (α:=α) id ps (get! p1) with
      | some yB => boxes.set! id (some yB)
      | none    => boxes
    | _ => boxes
  | .reshape _ _ =>
    match node.parents with
    | #[p1] => boxes.set! id (boxes[p1]!)
    | _ => boxes
  | .flatten _ =>
    match node.parents with
    | #[p1] => boxes.set! id (boxes[p1]!)
    | _ => boxes
  | .transpose axis₁ axis₂ =>
    match node.parents with
    | #[p1] =>
      let transposeOp := fun value => do
        let perm ← OpContracts.transposePerm value.shape.rank axis₁ axis₂
        NN.IR.Graph.permuteSomeTensor (α := α) value perm
      match ibpMonotoneSomeTensor? (α := α) nodes[p1]!.outShape node.outShape transposeOp
          (get! p1) with
      | some result => boxes.set! id (some result)
      | none => boxes
    | _ => boxes
  | .permute perm =>
    match node.parents with
    | #[p1] =>
      let Xin := get! p1
      let sIn := nodes[p1]!.outShape
      if hdim : Xin.dim = sIn.size then
        let sFlat : Shape := .dim Xin.dim .scalar
        have hsize : sFlat.size = sIn.size := by
          simp [sFlat, sIn, Spec.Shape.size, hdim]
        let xLo : Tensor α sIn :=
          Tensor.reshapeSpec (α := α) (source := sFlat) (target := sIn) Xin.lo hsize
        let xHi : Tensor α sIn :=
          Tensor.reshapeSpec (α := α) (source := sFlat) (target := sIn) Xin.hi hsize
        match permuteSomeTensor? (α := α) (v := ⟨sIn, xLo⟩) perm,
            permuteSomeTensor? (α := α) (v := ⟨sIn, xHi⟩) perm with
        | some yLoV, some yHiV =>
            let sOut := Spec.SomeTensor.shape (α := α) yLoV
            if hSame : Spec.SomeTensor.shape (α := α) yHiV = sOut then
              if hOut : sOut = node.outShape then
                let yLoSOut : Tensor α sOut := Spec.SomeTensor.tensor (α := α) yLoV
                let yHiSOut : Tensor α sOut := hSame ▸ Spec.SomeTensor.tensor (α := α) yHiV
                let yLoT : Tensor α node.outShape := hOut ▸ yLoSOut
                let yHiT : Tensor α node.outShape := hOut ▸ yHiSOut
                let flatLo := Tensor.flattenSpec (α:=α) yLoT
                let flatHi := Tensor.flattenSpec (α:=α) yHiT
                boxes.set! id (some { dim := node.outShape.size, lo := flatLo, hi := flatHi })
              else
                boxes
            else
              boxes
        | _, _ => boxes
      else
        boxes
    | _ => boxes
  | .mulElem =>
    match node.parents with
    | #[p1, p2] =>
      match boxMulElem (α:=α) (get! p1) (get! p2) with
      | some prod => boxes.set! id (some prod)
      | none => boxes
    | _ => boxes
  | .sum =>
    match node.parents with
    | #[p1] =>
      boxes.set! id (some (boxSum (α := α) (get! p1)))
    | _ => boxes
  | .mseLoss =>
    match node.parents with
    | #[p1, p2] =>
      let Y := get! p1
      let T := get! p2
      if Y.dim = T.dim then
        let diff := boxSub (α := α) Y T
        let sq := boxSquare (α:=α) diff
        match boxMean? (α := α) sq with
        | some mean => boxes.set! id (some mean)
        | none => boxes
      else boxes
    | _ => boxes
  | .conv config =>
    if !crownNodeSemanticsSupported (α := α) nodes ps id then
      boxes
    else
      match node.parents with
      | #[p1] =>
        let Xin := get! p1
        match ibpConvNode (α:=α) config nodes[p1]!.outShape node.outShape id ps Xin with
        | some yB => boxes.set! id (some yB)
        | none => boxes
      | _ => boxes
  | .batchNormEval channelAxis _ =>
    match node.parents with
    | #[p1] =>
      match ps.batchNormEval[id]? with
      | some config =>
        match batchNormEvalLinear? (α := α) nodes[p1]!.outShape channelAxis config with
        | some p =>
          let Xin := get! p1
          match ibpLinearParams (α := α) p Xin with
          | some yB => boxes.set! id (some yB)
          | none => boxes
        | none => boxes
      | none => boxes
    | _ => boxes
  | .exp =>
    match node.parents with
    | #[p1] =>
      let Xin := get! p1
      match boxUnaryEnclosure? (α := α) NonlinearBoundOps.expBounds Xin with
      | some B => boxes.set! id (some B)
      | none => boxes
    | _ => boxes
  | .log =>
    match node.parents with
    | #[p1] =>
      let Xin := get! p1
      let flo := getDimScalarFn (α:=α) Xin.lo
      -- Raw IR `log` rejects nonpositive values. An interval crossing that boundary therefore has
      -- no valid transfer result; callers can use the explicitly totalized safe-log operator when
      -- epsilon clamping is intended.
      if (List.finRange Xin.dim).all (fun i =>
          decide (0 < Tensor.item (flo i))) then
        match boxUnaryEnclosure? (α := α) NonlinearBoundOps.logBounds Xin with
        | some B => boxes.set! id (some B)
        | none => boxes
      else
        boxes
    | _ => boxes
  -- layernorm/concat handled in dedicated cases below
  | .concat axis =>
    if axis != 0 then
      boxes
    else
      -- Leading-axis concatenation is contiguous in row-major flattened storage.
      match node.parents with
      | #[p1, p2] =>
        let B1 := get! p1; let B2 := get! p2
        match B1, B2 with
        | ⟨n1, lo1, hi1⟩, ⟨n2, lo2, hi2⟩ =>
          let f1lo := getDimScalarFn (α:=α) lo1
          let f2lo := getDimScalarFn (α:=α) lo2
          let f1hi := getDimScalarFn (α:=α) hi1
          let f2hi := getDimScalarFn (α:=α) hi2
          let lo :=
            Tensor.dim (fun i =>
              Fin.addCases (fun i1 => f1lo i1) (fun i2 => f2lo i2) i)
          let hi :=
            Tensor.dim (fun i =>
              Fin.addCases (fun i1 => f1hi i1) (fun i2 => f2hi i2) i)
          boxes.set! id (some { dim := n1 + n2, lo := lo, hi := hi })
      | _ => boxes
  | .layernorm axis =>
    if !crownNodeSemanticsSupported (α := α) nodes ps id then
      boxes
    else
      -- Payloads retain their affine parameters and epsilon through the directed row transfer.
      match node.parents with
      | #[p1] =>
        let Xin := get! p1
        let s := node.outShape
        if axis = Spec.Shape.rank s - 1 then
          if hdim : Xin.dim = s.size then
            let result :=
              match ps.layerNorm[id]? with
              | none => ibpLayerNormBox? (α := α) s Xin
              | some parameters => ibpLayerNormPayloadBox? (α := α) s axis parameters Xin
            match result with
            | some B => boxes.set! id (some B)
            | none => boxes
          else boxes
        else boxes
      | _ => boxes
  | .softmax axis =>
    -- Last-axis softmax bounds. We only implement `axis = rank-1`.
    match node.parents with
    | #[p1] =>
      let Xin := get! p1
      let s := node.outShape
      if axis = Spec.Shape.rank s - 1 then
        if hdim : Xin.dim = s.size then
          boxes.set! id (some (ibpSoftmaxRange (α := α) s Xin.dim))
        else boxes
      else boxes
    | _ => boxes
  | .hardMaskedSoftmax mask =>
    match node.parents with
    | #[p1] =>
      let Xin := get! p1
      let s := node.outShape
      if hdim : Xin.dim = s.size then
        if hshape : mask.shape = s then
          match NN.IR.HardMask.toTensor? mask with
          | .error _ => boxes
          | .ok decoded =>
            let allowed : Tensor Bool s := hshape ▸ decoded
            let sFlat : Shape := .dim Xin.dim .scalar
            have hsize : sFlat.size = s.size := by
              simp [sFlat, Spec.Shape.size, hdim]
            let xLo : Tensor α s :=
              Tensor.reshapeSpec (α := α) (source := sFlat) (target := s) Xin.lo hsize
            let xHi : Tensor α s :=
              Tensor.reshapeSpec (α := α) (source := sFlat) (target := s) Xin.hi hsize
            let (yLo, yHi) :=
              ibpHardMaskedSoftmaxLastTensor (α := α) xLo xHi allowed
            boxes.set! id <| some
              { dim := s.size
                lo := Tensor.flattenSpec (α := α) yLo
                hi := Tensor.flattenSpec (α := α) yHi }
        else boxes
      else boxes
    | _ => boxes
  | .tanh =>
    let Xin :=
      match node.parents with
      | #[p1] => get! p1
      | _ => get! 0
    match boxUnaryEnclosure? (α := α) NonlinearBoundOps.tanhBounds Xin with
    | some B => boxes.set! id (some B)
    | none => boxes
  | .sigmoid =>
    let Xin :=
      match node.parents with
      | #[p1] => get! p1
      | _ => get! 0
    match boxUnaryEnclosure? (α := α) NonlinearBoundOps.sigmoidBounds Xin with
    | some B => boxes.set! id (some B)
    | none => boxes
  | .softplus =>
    match node.parents with
    | #[p1] =>
      match boxSoftplus? (α := α) (get! p1) with
      | some B => boxes.set! id (some B)
      | none => boxes
    | _ => boxes
  | .safeLog =>
    match node.parents with
    | #[p1, p2] =>
      match boxSafeLog? (α := α) (get! p1) (get! p2) with
      | some B => boxes.set! id (some B)
      | none => boxes
    | _ => boxes
  | .sin =>
    let Xin :=
      match node.parents with
      | #[p1] => get! p1
      | _ => get! 0
    match boxUnaryEnclosure? (α := α) NonlinearBoundOps.sinBounds Xin with
    | some B => boxes.set! id (some B)
    | none => boxes
  | .cos =>
    let Xin :=
      match node.parents with
      | #[p1] => get! p1
      | _ => get! 0
    match boxUnaryEnclosure? (α := α) NonlinearBoundOps.cosBounds Xin with
    | some B => boxes.set! id (some B)
    | none => boxes

/-- IBP propagation for one node using `ParamStore`.

This executable function expects parents to have already been processed. The proof layer makes that
precondition explicit via `TopoSorted`; callers that execute graphs directly should use graphs whose
parents appear before their children.
-/
@[expose] def propagateIBPNode (nodes : Array Node) (ps : ParamStore α)
    (boxes : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
  propagateIBPNodeAt (α := α) nodes ps boxes id nodes[id]!

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
