/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.IBP -- shake: keep
public import NN.MLTheory.CROWN.Graph.Engine.LayerNormDerivatives

/-!
# Derivative Interval Passes

These passes propagate interval bounds for first and second derivatives through the same flat graph
used by IBP. Derivative propagation has its own chain-rule state but reuses `FlatBox` for every
intermediate enclosure.

Linear operations, pointwise arithmetic, supported activations, and selected structural operations
have explicit rules. Coupled softmax derivatives are evaluated only when the scalar instance
declares their algebra exact. LayerNorm uses independent rows, its stored affine parameters,
and a positive epsilon to bound first and mixed derivatives. A missing box is reported as a
propagation failure by the certificate consumers.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]
variable [NonlinearBoundOps α]

open BoundOps

/-- Apply the input differential of a convolution to a first or mixed-second derivative box.

Convolution is affine in its input. With weights held fixed, every input derivative passes through
the same convolution with zero bias. This keeps the original groups, dilation, padding, strides,
and leading batch shape without materializing a dense matrix. -/
@[expose] def convDerivativeBox? (nodes : Array Node) (ps : ParamStore α)
    (derivatives : Array (Option (FlatBox α))) (id : Nat) (node : Node)
    (configuration : NN.IR.ConvConfig) : Option (FlatBox α) := do
  let parentId ← unaryParent? node.parents
  let parent ← nodes[parentId]?
  let direction ← (derivatives[parentId]?).join
  let parameters ← ps.convCfg[id]?
  let derivativeParameters :=
    { parameters with
      spec := { parameters.spec with bias := Tensor.full [parameters.outChannels] 0 } }
  let derivativeStore := { ps with convCfg := ps.convCfg.insert id derivativeParameters }
  ibpConvNode configuration parent.outShape node.outShape id derivativeStore direction

/-- Global enclosure for the derivative of `tanh`. -/
private def tanhDerivBox (dim : Nat) : FlatBox α :=
  { dim := dim
    lo := Tensor.full (α := α) (.dim dim .scalar) 0
    hi := Tensor.full (α := α) (.dim dim .scalar) 1 }

/-- Global enclosure for the derivative of the logistic sigmoid. -/
private def sigmoidDerivBox (dim : Nat) : FlatBox α :=
  let quarter := BoundOps.mulUp (1 / 2) (1 / 2)
  { dim := dim
    lo := Tensor.full (α := α) (.dim dim .scalar) 0
    hi := Tensor.full (α := α) (.dim dim .scalar) quarter }

/-- A simple global enclosure for the second derivative of `tanh`. -/
private def tanhSecondDerivBox (dim : Nat) : FlatBox α :=
  { dim := dim
    lo := Tensor.full (α := α) (.dim dim .scalar) (-2)
    hi := Tensor.full (α := α) (.dim dim .scalar) 2 }

/-- A simple global enclosure for the second derivative of the logistic sigmoid. -/
private def sigmoidSecondDerivBox (dim : Nat) : FlatBox α :=
  { dim := dim
    lo := Tensor.full (α := α) (.dim dim .scalar) (-1)
    hi := Tensor.full (α := α) (.dim dim .scalar) 1 }

/-- Bounds for `δᵢⱼ - yⱼ` within one softmax normalization row. -/
private def softmaxRowDeltaMinus {n : Nat} (y : Fin n → α × α)
    (i j : Fin n) : α × α :=
  let delta : α := if i = j then 1 else 0
  (subDown delta (y j).2, subUp delta (y j).1)

/-- One row's Jacobian entry `yᵢ (δᵢⱼ - yⱼ)`. -/
private def softmaxRowJacobian {n : Nat} (y : Fin n → α × α)
    (i j : Fin n) : α × α :=
  let delta := softmaxRowDeltaMinus y i j
  intervalMul (y i).1 (y i).2 delta.1 delta.2

/-- Contract one softmax row's Jacobian with a direction enclosure. -/
private def softmaxRowFirstDerivative {n : Nat}
    (y dz : Fin n → α × α) (i : Fin n) : α × α :=
  (List.finRange n).foldl (fun acc j =>
    let jacobian := softmaxRowJacobian y i j
    let term := intervalMul jacobian.1 jacobian.2 (dz j).1 (dz j).2
    (addDown acc.1 term.1, addUp acc.2 term.2)) (0, 0)

/-- One row's Hessian entry `Jᵢⱼ (δᵢₖ - yₖ) - yᵢ Jⱼₖ`. -/
private def softmaxRowHessian {n : Nat} (y : Fin n → α × α)
    (i j k : Fin n) : α × α :=
  let jacobian := softmaxRowJacobian y i j
  let delta := softmaxRowDeltaMinus y i k
  let first := intervalMul jacobian.1 jacobian.2 delta.1 delta.2
  let innerJacobian := softmaxRowJacobian y j k
  let second := intervalMul (y i).1 (y i).2 innerJacobian.1 innerJacobian.2
  (subDown first.1 second.2, subUp first.2 second.1)

/-- The mixed chain rule `J D²z[u,v] + H[Dz[u], Dz[v]]` for one softmax row. -/
private def softmaxRowMixedSecondDerivative {n : Nat}
    (y dzLeft dzRight d2z : Fin n → α × α) (i : Fin n) : α × α :=
  let linear := softmaxRowFirstDerivative y d2z i
  let bilinear := (List.finRange n).foldl (fun acc j =>
    (List.finRange n).foldl (fun acc k =>
      let hessian := softmaxRowHessian y i j k
      let directions := intervalMul (dzLeft j).1 (dzLeft j).2 (dzRight k).1 (dzRight k).2
      let term := intervalMul hessian.1 hessian.2 directions.1 directions.2
      (addDown acc.1 term.1, addUp acc.2 term.2)) acc) (0, 0)
  (addDown linear.1 bilinear.1, addUp linear.2 bilinear.2)

/-- Read flat endpoints only after checking the tensor's declared element count. -/
private def softmaxFlatEndpoints? (s : Shape) (box : FlatBox α) :
    Option (Fin s.size → α × α) :=
  if h : box.dim = s.size then
    let lo := castDimScalar (h := h) box.lo
    let hi := castDimScalar (h := h) box.hi
    some fun i => (lo.getScalar i, hi.getScalar i)
  else
    none

/--
Gather the normalization row containing each output coordinate, then scatter its result back.

Only the selected coordinate changes while gathering a row. Outer and interior axes therefore
use the same row formulas as the final axis, without coupling independent rows. Empty tensors
have no coordinates to gather; invalid axes and inconsistent coordinates return no enclosure.
-/
private def softmaxMapRows? (s : Shape) (axis : Nat)
    (rowRule : (n : Nat) → (Fin n → Fin s.size) → Fin n → α × α) :
    Option (FlatBox α) := do
  let ⟨hAxis⟩ ← Shape.axisInBounds? axis s
  let n := s.axisSize axis (h := hAxis)
  let endpoints ← Internal.traverseFin fun i : Fin s.size => do
    let coordinates := Shape.Coord.toList s (Shape.Coord.unlinearize i)
    let column ← coordinates[axis]?
    if hColumn : column < n then
      let row ← Internal.traverseFin fun j : Fin n =>
        (Shape.Coord.ofList? s (coordinates.set axis j.val)).map Shape.Coord.linearize
      pure (rowRule n row ⟨column, hColumn⟩)
    else
      none
  return { dim := s.size
           lo := Tensor.ofFn fun i => (endpoints i).1
           hi := Tensor.ofFn fun i => (endpoints i).2 }

/-- Shape-checked first derivative for independent softmax rows along any valid axis. -/
private def softmaxFirstDerivativeBox? (s : Shape) (axis : Nat)
    (y dz : FlatBox α) : Option (FlatBox α) := do
  let yEndpoints ← softmaxFlatEndpoints? s y
  let dzEndpoints ← softmaxFlatEndpoints? s dz
  softmaxMapRows? s axis fun _ row =>
    softmaxRowFirstDerivative (yEndpoints ∘ row) (dzEndpoints ∘ row)

/-- Shape-checked mixed second derivative for independent softmax normalization rows. -/
private def softmaxMixedSecondDerivativeBox? (s : Shape) (axis : Nat)
    (y dzLeft dzRight d2z : FlatBox α) : Option (FlatBox α) := do
  let yEndpoints ← softmaxFlatEndpoints? s y
  let leftEndpoints ← softmaxFlatEndpoints? s dzLeft
  let rightEndpoints ← softmaxFlatEndpoints? s dzRight
  let secondEndpoints ← softmaxFlatEndpoints? s d2z
  softmaxMapRows? s axis fun _ row =>
    softmaxRowMixedSecondDerivative (yEndpoints ∘ row) (leftEndpoints ∘ row)
      (rightEndpoints ∘ row) (secondEndpoints ∘ row)

/--
Apply an axis permutation to both endpoints of a derivative box.

A permutation is linear, so its first and mixed second derivatives reorder the parent's
derivative coordinates in exactly the same way as its values. The IBP endpoint helper restores
the parent's shape before applying that permutation and checks the declared output shape.
Missing parents, missing derivative boxes, and invalid permutations leave the node unresolved.
-/
private def permuteDerivativeBox? (nodes : Array Node)
    (boxes : Array (Option (FlatBox α))) (node : Node) : Option (FlatBox α) := do
  let parentId ← unaryParent? node.parents
  let parent ← nodes[parentId]?
  let derivative ← (boxes[parentId]?).join
  let permutation ←
    match node.kind with
    | .transpose axis₁ axis₂ =>
      (OpContracts.transposePerm parent.outShape.rank axis₁ axis₂).toOption
    | .permute permutation => some permutation
    | _ => none
  ibpMonotoneSomeTensor? (α := α) parent.outShape node.outShape
    (fun value => NN.IR.Graph.permuteSomeTensor (α := α) value permutation) derivative

/-- Shared first-derivative propagation with a caller-supplied input seed. -/
private def runFirstDerivativeWithSeed
    (g : Graph) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (inputSeed : FlatBox α → Option (FlatBox α)) : Array (Option (FlatBox α)) :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (drs : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        match inputSeed B with
        | some seed => drs.set! id (some seed)
        | none => drs
      | none => drs
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Tensor.full (α:=α) (.dim v.n .scalar) 0
        drs.set! id (some { dim := v.n, lo := z, hi := z })
      | none => drs
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Tensor.full (α:=α) (.dim d .scalar) 0
      drs.set! id (some { dim := d, lo := z, hi := z })
    | .maxPool .. | .avgPool .. | .softplus | .safeLog =>
      -- Not supported by the derivative-bound passes (used by PINN tooling).
      drs
    | .hardMaskedSoftmax _ =>
      -- Hard-masked normalization still needs a derivative rule that handles the mask.
      drs
    | .sum =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join with
        | some dXin => drs.set! id (some (boxSum (α := α) dXin))
        | none => drs
      | _ => drs
    | .linear =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join, ps.linearWB[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Tensor.full (α:=α) (.dim p.m .scalar) 0
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .matmul =>
      match node.parents with
      | #[p1, p2] =>
        let result := do
          let leftNode ← g.nodes[p1]?
          let rightNode ← g.nodes[p2]?
          let left ← (ibp[p1]?).join
          let right ← (ibp[p2]?).join
          let dLeft ← (drs[p1]?).join
          let dRight ← (drs[p2]?).join
          let first ← ibpBinaryMatmul? leftNode.outShape rightNode.outShape dLeft right
          let second ← ibpBinaryMatmul? leftNode.outShape rightNode.outShape left dRight
          pure (boxAdd first second)
        drs.set! id result
      | #[p1] =>
        match (drs[p1]?).join, ps.matmulW[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Tensor.full (α:=α) (.dim p.m .scalar) 0
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .relu =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join with
        | some dIn =>
          let z := Tensor.full (α:=α) (.dim dIn.dim .scalar) 0
          let o := Tensor.full (α:=α) (.dim dIn.dim .scalar) 1
          let dF : FlatBox α := { dim := dIn.dim, lo := z, hi := o }
          match boxMulElem (α:=α) dIn dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .tanh =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join with
        | some dZ =>
          match boxMulElem (α := α) dZ (tanhDerivBox (α := α) dZ.dim) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .sigmoid =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join with
        | some dZ =>
          match boxMulElem (α := α) dZ (sigmoidDerivBox (α := α) dZ.dim) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .softmax axis =>
      if !NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) then drs else
      let result := do
        let parent ← unaryParent? node.parents
        let dz ← (drs[parent]?).join
        let y ← (ibp[id]?).join
        softmaxFirstDerivativeBox? node.outShape axis y dz
      drs.set! id result
    | .sin =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join, (ibp[p1]?).join with
        | some dZ, some zB =>
          match boxUnaryEnclosure? (α := α) NonlinearBoundOps.cosBounds zB with
          | some dF =>
            match boxMulElem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          | none => drs
        | _, _ => drs
      | _ => drs
    | .cos =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join, (ibp[p1]?).join with
        | some dZ, some zB =>
          match boxUnaryEnclosure? (α := α) NonlinearBoundOps.sinBounds zB with
          | some sB =>
            match boxMulElem (α:=α) dZ (boxNeg (α := α) sB) with
            | some prod => drs.set! id (some prod)
            | none => drs
          | none => drs
        | _, _ => drs
      | _ => drs
    | .exp =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join, (ibp[p1]?).join with
        | some dZ, some zB =>
          match derivBoxExp? (α := α) zB with
          | some dF =>
            match chainMul (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          | none => drs
        | _, _ => drs
      | _ => drs
    | .log =>
      match node.parents with
      | #[p1] =>
        match (drs[p1]?).join, (ibp[p1]?).join with
        | some dZ, some zB =>
          match derivBoxLog? (α := α) zB with
          | some dF =>
            match chainMul (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          | none => drs
        | _, _ => drs
      | _ => drs
    | .add =>
      match node.parents with
      | #[p1, p2] =>
        match (drs[p1]?).join, (drs[p2]?).join with
        | some d1, some d2 => some (boxAdd (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .sub =>
      match node.parents with
      | #[p1, p2] =>
        match (drs[p1]?).join, (drs[p2]?).join with
        | some d1, some d2 => some (boxSub (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .mulElem =>
      match node.parents with
      | #[p1, p2] =>
        match (drs[p1]?).join, (drs[p2]?).join, (ibp[p1]?).join, (ibp[p2]?).join with
        | some dx, some dy, some xB, some yB =>
          match boxMulElem (α:=α) dx yB, boxMulElem (α:=α) xB dy with
          | some t1, some t2 => drs.set! id (some (boxAdd (α:=α) t1 t2))
          | _, _ => drs
        | _, _, _, _ => drs
      | _ => drs
    | .layernorm axis =>
      let result := do
        let parent ← unaryParent? node.parents
        let input ← (ibp[parent]?).join
        let direction ← (drs[parent]?).join
        let zero : FlatBox α :=
          { dim := input.dim, lo := Tensor.full [input.dim] 0, hi := Tensor.full [input.dim] 0 }
        let bounds ← layerNormDerivativeBoxes? node.outShape axis ps.layerNorm[id]?
          input direction zero zero
        pure bounds.1
      drs.set! id result
    | .reshape _ _ | .flatten _ =>
      -- Reshape and flatten retain the order of the scalar coordinates.
      match node.parents with
      | #[p1] => drs.set! id ((drs[p1]?).join)
      | _ => drs
    | .transpose .. | .permute _ =>
      drs.set! id (permuteDerivativeBox? (α := α) g.nodes drs node)
    | .concat axis =>
      drs.set! id (concatNodeBoxes? (α := α) g.nodes drs node axis)
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      drs
    | .mseLoss => drs
    | .conv configuration =>
      drs.set! id (convDerivativeBox? g.nodes ps drs id node configuration)
    | .batchNormEval .. => drs
  if crownGraphSemanticsSupported (α := α) g ps then
    (List.finRange g.nodes.size).foldl propagate init
  else
    init

/--
Propagate first-derivative intervals from a scalar input.

The input derivative is the all-ones vector. The pass uses value-IBP boxes to bound activation
derivatives and leaves an entry empty when it encounters an unsupported local derivative.
-/
def runScalarDerivative
    (g : Graph) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) :
    Array (Option (FlatBox α)) :=
  runFirstDerivativeWithSeed g ps ibp fun B =>
    if B.dim = 1 then
      let one := Tensor.full (α := α) (.dim B.dim .scalar) 1
      some { dim := B.dim, lo := one, hi := one }
    else
      none

/--
Propagate a directional first-derivative enclosure from a caller-supplied input seed.

A seed whose dimension differs from an input box leaves that input unresolved. Point seeds such
as coordinate vectors recover partial derivatives; interval seeds propagate a family of
directions through the same local derivative rules.
-/
def runDirectionalDerivative
    (g : Graph) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
    (seed : FlatBox α) : Array (Option (FlatBox α)) :=
  runFirstDerivativeWithSeed g ps ibp fun B =>
    if h : seed.dim = B.dim then
      let lo := castDimScalar (α := α) (n := seed.dim) (n' := B.dim) (h := h) seed.lo
      let hi := castDimScalar (α := α) (n := seed.dim) (n' := B.dim) (h := h) seed.hi
      some { dim := B.dim, lo, hi }
    else
      none

/--
Propagate an enclosure of the mixed second derivative `D²f[u, v]`.

`dLeft` and `dRight` are first-derivative passes seeded by directions `u` and `v`. The input mixed
derivative is zero, while every nonlinear rule applies the bilinear second-order chain rule. Taking
the two arrays equal recovers the second directional derivative `D²f[v, v]`; coordinate seeds can
be paired with a fixed direction to recover the entries of a Hessian-vector product.
-/
def runMixedSecondDerivative (g : Graph) (ps : ParamStore α)
    (ibp dLeft dRight : Array (Option (FlatBox α))) : Array (Option (FlatBox α)) :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (d2s : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        let z := Tensor.full (α:=α) (.dim B.dim .scalar) 0
        d2s.set! id (some { dim := B.dim, lo := z, hi := z })
      | none => d2s
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Tensor.full (α:=α) (.dim v.n .scalar) 0
        d2s.set! id (some { dim := v.n, lo := z, hi := z })
      | none => d2s
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Tensor.full (α:=α) (.dim d .scalar) 0
      d2s.set! id (some { dim := d, lo := z, hi := z })
    | .maxPool .. | .avgPool .. | .softplus | .safeLog =>
      -- Not supported by the second-derivative bound pass.
      d2s
    | .hardMaskedSoftmax _ =>
      -- A sound row-wise Hessian rule has not yet been added for masked attention.
      d2s
    | .linear =>
      match node.parents with
      | #[p1] =>
        match (d2s[p1]?).join, ps.linearWB[id]? with
        | some d2Xin, some p =>
          if h : d2Xin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := d2Xin.lo, hi :=
              d2Xin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Tensor.full (α:=α) (.dim p.m .scalar) 0
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            d2s.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else d2s
        | _, _ => d2s
      | _ => d2s
    | .matmul =>
      match node.parents with
      | #[p1, p2] =>
        let result := do
          let leftNode ← g.nodes[p1]?
          let rightNode ← g.nodes[p2]?
          let left ← (ibp[p1]?).join
          let right ← (ibp[p2]?).join
          let duLeft ← (dLeft[p1]?).join
          let duRight ← (dLeft[p2]?).join
          let dvLeft ← (dRight[p1]?).join
          let dvRight ← (dRight[p2]?).join
          let d2Left ← (d2s[p1]?).join
          let d2Right ← (d2s[p2]?).join
          let term1 ← ibpBinaryMatmul? leftNode.outShape rightNode.outShape d2Left right
          let term2 ← ibpBinaryMatmul? leftNode.outShape rightNode.outShape duLeft dvRight
          let term3 ← ibpBinaryMatmul? leftNode.outShape rightNode.outShape dvLeft duRight
          let term4 ← ibpBinaryMatmul? leftNode.outShape rightNode.outShape left d2Right
          pure (boxAdd (boxAdd (boxAdd term1 term2) term3) term4)
        d2s.set! id result
      | #[p1] =>
        match (d2s[p1]?).join, ps.matmulW[id]? with
        | some d2Xin, some p =>
          if h : d2Xin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := d2Xin.lo, hi :=
              d2Xin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Tensor.full (α:=α) (.dim p.m .scalar) 0
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            d2s.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else d2s
        | _, _ => d2s
      | _ => d2s
    | .add =>
      match node.parents with
      | #[p1, p2] =>
        match (d2s[p1]?).join, (d2s[p2]?).join with
        | some a, some b => d2s.set! id (some (boxAdd (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .sub =>
      match node.parents with
      | #[p1, p2] =>
        match (d2s[p1]?).join, (d2s[p2]?).join with
        | some a, some b => d2s.set! id (some (boxSub (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .mulElem =>
      match node.parents with
      | #[p1, p2] =>
        match (ibp[p1]?).join, (ibp[p2]?).join, (dLeft[p1]?).join, (dRight[p1]?).join,
            (dLeft[p2]?).join, (dRight[p2]?).join, (d2s[p1]?).join, (d2s[p2]?).join with
        | some xB, some yB, some dxLeft, some dxRight, some dyLeft, some dyRight,
            some d2x, some d2y =>
          -- D²(xy)[u,v] = D²x[u,v]y + Dx[u]Dy[v] + Dx[v]Dy[u] + xD²y[u,v].
          match boxMulElem (α := α) d2x yB,
              boxMulElem (α := α) dxLeft dyRight,
              boxMulElem (α := α) dxRight dyLeft,
              boxMulElem (α := α) xB d2y with
          | some t1, some t2, some t3, some t4 =>
            d2s.set! id <| some <|
              boxAdd (α := α) (boxAdd (α := α) t1 t2) (boxAdd (α := α) t3 t4)
          | _, _, _, _ => d2s
        | _, _, _, _, _, _, _, _ => d2s
      | _ => d2s
    | .relu =>
      match node.parents with
      | #[p1] =>
        match (ibp[p1]?).join with
        | some zB =>
          let z := Tensor.full (α:=α) (.dim zB.dim .scalar) 0
          d2s.set! id (some { dim := zB.dim, lo := z, hi := z })
        | none => d2s
      | _ => d2s
    | .tanh =>
      match node.parents with
      | #[p1] =>
        match (dLeft[p1]?).join, (dRight[p1]?).join, (d2s[p1]?).join with
        | some dzLeft, some dzRight, some d2z =>
          match boxMulElem (α := α) dzLeft dzRight with
          | none => d2s
          | some dzProduct =>
            match boxMulElem (α := α)
                (tanhSecondDerivBox (α := α) dzProduct.dim) dzProduct,
              boxMulElem (α := α) (tanhDerivBox (α := α) d2z.dim) d2z with
            | some tA, some tB => d2s.set! id (some (boxAdd (α := α) tA tB))
            | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .sin =>
      match node.parents with
      | #[p1] =>
        match (ibp[p1]?).join, (dLeft[p1]?).join, (dRight[p1]?).join, (d2s[p1]?).join with
        | some zB, some dzLeft, some dzRight, some d2z =>
          -- D²sin(z)[u,v] = -sin(z) Dz[u] Dz[v] + cos(z) D²z[u,v].
          match boxUnaryEnclosure? (α := α) NonlinearBoundOps.sinBounds zB,
              boxUnaryEnclosure? (α := α) NonlinearBoundOps.cosBounds zB with
          | some sinB, some cosB =>
            match boxMulElem (α := α) dzLeft dzRight with
            | none => d2s
            | some dzProduct =>
              match boxMulElem (α:=α) (boxNeg (α := α) sinB) dzProduct,
                  boxMulElem (α:=α) cosB d2z with
              | some tA, some tB => d2s.set! id (some (boxAdd (α:=α) tA tB))
              | _, _ => d2s
          | _, _ => d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .cos =>
      match node.parents with
      | #[p1] =>
        match (ibp[p1]?).join, (dLeft[p1]?).join, (dRight[p1]?).join, (d2s[p1]?).join with
        | some zB, some dzLeft, some dzRight, some d2z =>
          -- D²cos(z)[u,v] = -cos(z) Dz[u] Dz[v] - sin(z) D²z[u,v].
          match boxUnaryEnclosure? (α := α) NonlinearBoundOps.sinBounds zB,
              boxUnaryEnclosure? (α := α) NonlinearBoundOps.cosBounds zB with
          | some sinB, some cosB =>
            match boxMulElem (α := α) dzLeft dzRight with
            | none => d2s
            | some dzProduct =>
              match boxMulElem (α:=α) (boxNeg (α := α) cosB) dzProduct,
                  boxMulElem (α:=α) (boxNeg (α := α) sinB) d2z with
              | some tA, some tB => d2s.set! id (some (boxAdd (α:=α) tA tB))
              | _, _ => d2s
          | _, _ => d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .sigmoid =>
      match node.parents with
      | #[p1] =>
        match (dLeft[p1]?).join, (dRight[p1]?).join, (d2s[p1]?).join with
        | some dzLeft, some dzRight, some d2z =>
          match boxMulElem (α := α) dzLeft dzRight with
          | none => d2s
          | some dzProduct =>
            match boxMulElem (α := α)
                (sigmoidSecondDerivBox (α := α) dzProduct.dim) dzProduct,
              boxMulElem (α := α) (sigmoidDerivBox (α := α) d2z.dim) d2z with
            | some tA, some tB => d2s.set! id (some (boxAdd (α := α) tA tB))
            | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .exp =>
      match node.parents with
      | #[p1] =>
        match (ibp[p1]?).join, (dLeft[p1]?).join, (dRight[p1]?).join, (d2s[p1]?).join with
        | some zB, some dzLeft, some dzRight, some d2z =>
          match derivBoxExp? (α := α) zB with
          | some derivative =>
            match boxMulElem (α := α) dzLeft dzRight with
            | none => d2s
            | some dzProduct =>
              match boxMulElem (α:=α) derivative dzProduct,
                  boxMulElem (α:=α) derivative d2z with
              | some tA, some tB => d2s.set! id (some (boxAdd (α:=α) tA tB))
              | _, _ => d2s
          | none => d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .log =>
      match node.parents with
      | #[p1] =>
        match (ibp[p1]?).join, (dLeft[p1]?).join, (dRight[p1]?).join, (d2s[p1]?).join with
        | some zB, some dzLeft, some dzRight, some d2z =>
          match derivBoxLog? (α := α) zB, secondDerivBoxLog? (α := α) zB with
          | some firstDerivative, some secondDerivative =>
            match boxMulElem (α := α) dzLeft dzRight with
            | none => d2s
            | some dzProduct =>
              match boxMulElem (α:=α) secondDerivative dzProduct,
                  boxMulElem (α:=α) firstDerivative d2z with
              | some tA, some tB => d2s.set! id (some (boxAdd (α:=α) tA tB))
              | _, _ => d2s
          | _, _ => d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .sum =>
      match node.parents with
      | #[p1] =>
        match (d2s[p1]?).join with
        | some d2Xin => d2s.set! id (some (boxSum (α := α) d2Xin))
        | none => d2s
      | _ => d2s
    | .reshape _ _ | .flatten _ =>
      match node.parents with
      | #[p1] => d2s.set! id ((d2s[p1]?).join)
      | _ => d2s
    | .transpose .. | .permute _ =>
      d2s.set! id (permuteDerivativeBox? (α := α) g.nodes d2s node)
    | .concat axis =>
      d2s.set! id (concatNodeBoxes? (α := α) g.nodes d2s node axis)
    | .mseLoss => d2s
    | .softmax axis =>
      -- The ideal coupled formula remains unavailable to finite-precision backends.
      if !NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) then d2s else
      let result := do
        let parent ← unaryParent? node.parents
        let y ← (ibp[id]?).join
        let dzLeft ← (dLeft[parent]?).join
        let dzRight ← (dRight[parent]?).join
        let d2z ← (d2s[parent]?).join
        softmaxMixedSecondDerivativeBox? node.outShape axis y dzLeft dzRight d2z
      d2s.set! id result
    | .layernorm axis =>
      let result := do
        let parent ← unaryParent? node.parents
        let input ← (ibp[parent]?).join
        let left ← (dLeft[parent]?).join
        let right ← (dRight[parent]?).join
        let mixed ← (d2s[parent]?).join
        let bounds ← layerNormDerivativeBoxes? node.outShape axis ps.layerNorm[id]?
          input left right mixed
        pure bounds.2
      d2s.set! id result
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      d2s
    | .conv configuration =>
      d2s.set! id (convDerivativeBox? g.nodes ps d2s id node configuration)
    | .batchNormEval .. => d2s
  if crownGraphSemanticsSupported (α := α) g ps then
    (List.finRange g.nodes.size).foldl propagate init
  else
    init

/-- Propagate the second directional derivative `D²f[v, v]` from one first-derivative pass. -/
def runSecondDirectionalDerivative (g : Graph) (ps : ParamStore α)
    (ibp dDirection : Array (Option (FlatBox α))) : Array (Option (FlatBox α)) :=
  runMixedSecondDerivative g ps ibp dDirection dDirection

/--
Compute an interval enclosure for each component of a Hessian-vector product.

`coordinateDerivatives i` is the first-derivative pass seeded by the `i`th coordinate vector;
`directionalDerivative` is seeded by the vector being multiplied by the Hessian. The result at `i`
is the mixed derivative `D²f[eᵢ, v]`, i.e. the `i`th Hessian-vector component for scalar outputs.
-/
def runHessianVectorProduct {inputDim : Nat} (g : Graph) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α)))
    (coordinateDerivatives : Fin inputDim → Array (Option (FlatBox α)))
    (directionalDerivative : Array (Option (FlatBox α))) :
    Fin inputDim → Array (Option (FlatBox α)) :=
  fun i => runMixedSecondDerivative g ps ibp (coordinateDerivatives i) directionalDerivative

/-- One-dimensional second derivatives are the all-ones directional special case. -/
def runScalarSecondDerivative (g : Graph) (ps : ParamStore α)
    (ibp d1 : Array (Option (FlatBox α))) : Array (Option (FlatBox α)) :=
  runSecondDirectionalDerivative g ps ibp d1

end NN.MLTheory.CROWN.Graph
