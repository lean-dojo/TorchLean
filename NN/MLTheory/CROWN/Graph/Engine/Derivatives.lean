/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.IBP -- shake: keep

/-!
# Derivative Interval Passes

These passes propagate interval bounds for first and second derivatives through the same flat graph
used by IBP. Derivative propagation has its own chain-rule state but reuses `FlatBox` for every
intermediate enclosure.

Linear operations, pointwise arithmetic, supported activations, and selected structural operations
have explicit rules. Coupled softmax derivatives are evaluated only when the scalar instance
declares their algebra exact. LayerNorm derivatives remain unresolved: the rowwise rule with
stored affine parameters and epsilon has not yet been connected to this interval pass. A missing
box is reported as a propagation failure by the certificate consumers.
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
      -- The current derivative pass treats softmax as one flat vector. A masked attention tensor
      -- is row structured, so propagating that rule here would mix independent rows.
      drs
    | .sum =>
      match node.parents with
      | #[p1] =>
        match drs[p1]! with
        | some dXin => drs.set! id (some (boxSum (α := α) dXin))
        | none => drs
      | _ => drs
    | .linear =>
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ps.linearWB[id]? with
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
      | #[p1] =>
        match drs[p1]!, ps.matmulW[id]? with
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
        match drs[p1]! with
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
        match drs[p1]! with
        | some dZ =>
          match boxMulElem (α := α) dZ (tanhDerivBox (α := α) dZ.dim) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .sigmoid =>
      match node.parents with
      | #[p1] =>
        match drs[p1]! with
        | some dZ =>
          match boxMulElem (α := α) dZ (sigmoidDerivBox (α := α) dZ.dim) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .softmax _ =>
      if !NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) then drs else
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          -- The formulas below construct one dense softmax Jacobian. They are sound only when
          -- the node itself is a vector, not when several last-axis rows share one flat box.
          if h : dZ.dim = yB.dim ∧ node.outShape = .dim yB.dim .scalar then
            let n := yB.dim
            -- Cast derivative tensors to dimension n for Fin alignment
            let dLo := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h.1) dZ.lo
            let dHi := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h.1) dZ.hi
            let fyLo := getDimScalarFn (α:=α) yB.lo
            let fyHi := getDimScalarFn (α:=α) yB.hi
            let fdLo := getDimScalarFn (α:=α) dLo
            let fdHi := getDimScalarFn (α:=α) dHi
            let mulI (aLo aHi bLo bHi : α) : α × α :=
              let p1 := aLo * bLo; let p2 := aLo * bHi
              let p3 := aHi * bLo; let p4 := aHi * bHi
              let lo1 := if p1 < p2 then p1 else p2
              let lo2 := if p3 < p4 then p3 else p4
              let lo  := if lo1 < lo2 then lo1 else lo2
              let hi1 := if p1 > p2 then p1 else p2
              let hi2 := if p3 > p4 then p3 else p4
              let hi  := if hi1 > hi2 then hi1 else hi2
              (lo, hi)
            let dlo :=
              Tensor.dim (fun i =>
                let yiLo := (fyLo i).item
                let yiHi := (fyHi i).item
                let (sumLo, _sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := (fyLo k).item
                    let ykHi := (fyHi k).item
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := 1 - yiHi
                        let oneMinusHi := 1 - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := (fdLo k).item
                    let dxHi := (fdHi k).item
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumLo)
            let dhi :=
              Tensor.dim (fun i =>
                let yiLo := (fyLo i).item
                let yiHi := (fyHi i).item
                let (_sumLo, sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := (fyLo k).item
                    let ykHi := (fyHi k).item
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := 1 - yiHi
                        let oneMinusHi := 1 - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := (fdLo k).item
                    let dxHi := (fdHi k).item
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumHi)
            drs.set! id (some { dim := n, lo := dlo, hi := dhi })
          else drs
        | _, _ => drs
      | _ => drs
    | .sin =>
      match node.parents with
      | #[p1] =>
        match drs[p1]!, ibp[p1]! with
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
        match drs[p1]!, ibp[p1]! with
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
        match drs[p1]!, ibp[p1]! with
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
        match drs[p1]!, ibp[p1]! with
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
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (boxAdd (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .sub =>
      match node.parents with
      | #[p1, p2] =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (boxSub (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .mulElem =>
      match node.parents with
      | #[p1, p2] =>
        match drs[p1]!, drs[p2]!, ibp[p1]!, ibp[p2]! with
        | some dx, some dy, some xB, some yB =>
          match boxMulElem (α:=α) dx yB, boxMulElem (α:=α) xB dy with
          | some t1, some t2 => drs.set! id (some (boxAdd (α:=α) t1 t2))
          | _, _ => drs
        | _, _, _, _ => drs
      | _ => drs
    | .layernorm _ =>
      -- LayerNorm couples coordinates within each normalization row. Its derivative also
      -- depends on the stored scale and epsilon, so a flattened whole-tensor rule does not
      -- describe the operation. In particular, the variance contribution contains
      -- (variance + epsilon)^(-3/2); bounding it requires a positive denominator lower bound
      -- without increasing that lower bound before taking its reciprocal.
      --
      -- The real rowwise derivative bound is proved separately in LayerNormBounds. Until
      -- that bound is connected to a directed transfer with the actual payload, leave this
      -- node unresolved for every scalar backend. Callers then report a missing derivative
      -- box instead of accepting an enclosure from an unsupported formula.
      drs.set! id none
    | .reshape _ _ | .flatten _ =>
      -- Reshape and flatten retain the order of the scalar coordinates.
      match node.parents with
      | #[p1] => drs.set! id (drs[p1]!)
      | _ => drs
    | .transpose .. | .permute _ =>
      drs.set! id (permuteDerivativeBox? (α := α) g.nodes drs node)
    | .concat _ =>
      -- Concatenation needs derivative boxes from every parent.
      drs
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      drs
    | .mseLoss => drs
    | .conv .. | .batchNormEval .. => drs
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
        match d2s[p1]!, ps.linearWB[id]? with
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
      | #[p1] =>
        match d2s[p1]!, ps.matmulW[id]? with
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
        match d2s[p1]!, d2s[p2]! with
        | some a, some b => d2s.set! id (some (boxAdd (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .sub =>
      match node.parents with
      | #[p1, p2] =>
        match d2s[p1]!, d2s[p2]! with
        | some a, some b => d2s.set! id (some (boxSub (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .mulElem =>
      match node.parents with
      | #[p1, p2] =>
        match ibp[p1]!, ibp[p2]!, dLeft[p1]!, dRight[p1]!, dLeft[p2]!, dRight[p2]!,
            d2s[p1]!, d2s[p2]! with
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
        match ibp[p1]! with
        | some zB =>
          let z := Tensor.full (α:=α) (.dim zB.dim .scalar) 0
          d2s.set! id (some { dim := zB.dim, lo := z, hi := z })
        | none => d2s
      | _ => d2s
    | .tanh =>
      match node.parents with
      | #[p1] =>
        match dLeft[p1]!, dRight[p1]!, d2s[p1]! with
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
        match ibp[p1]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
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
        match ibp[p1]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
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
        match dLeft[p1]!, dRight[p1]!, d2s[p1]! with
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
        match ibp[p1]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
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
        match ibp[p1]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
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
        match d2s[p1]! with
        | some d2Xin => d2s.set! id (some (boxSum (α := α) d2Xin))
        | none => d2s
      | _ => d2s
    | .reshape _ _ | .flatten _ =>
      match node.parents with
      | #[p1] => d2s.set! id (d2s[p1]!)
      | _ => d2s
    | .transpose .. | .permute _ =>
      d2s.set! id (permuteDerivativeBox? (α := α) g.nodes d2s node)
    | .concat _ => d2s
    | .mseLoss => d2s
    | .softmax _ =>
      -- D²y_i[u,v] = Σ_k J_ik D²z_k[u,v] + Σ_{j,k} H_ijk Dz_j[u] Dz_k[v], with
      -- J = diag(y) - y yᵀ and H derived from ∂J/∂z (bounded via y-bounds).
      if !NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) then d2s else
      match node.parents with
      | #[p1] =>
        match ibp[id]!, dLeft[p1]!, dRight[p1]!, d2s[p1]! with
        | some yB, some dzLeft, some dzRight, some d2z =>
          -- The Hessian below is for one vector-valued softmax row.
          if hLeft : dzLeft.dim = yB.dim ∧ node.outShape = .dim yB.dim .scalar then
            if hRight : dzRight.dim = yB.dim then
              if h2 : d2z.dim = yB.dim then
              let n := yB.dim
              -- Cast derivative tensors to dimension n for Fin alignment
              let dLeftLo := castDimScalar (α:=α) (n:=dzLeft.dim) (n':=n)
                (h:=hLeft.1) dzLeft.lo
              let dLeftHi := castDimScalar (α:=α) (n:=dzLeft.dim) (n':=n)
                (h:=hLeft.1) dzLeft.hi
              let dRightLo := castDimScalar (α:=α) (n:=dzRight.dim) (n':=n)
                (h:=hRight) dzRight.lo
              let dRightHi := castDimScalar (α:=α) (n:=dzRight.dim) (n':=n)
                (h:=hRight) dzRight.hi
              let d2Lo := castDimScalar (α:=α) (n:=d2z.dim) (n':=n) (h:=h2) d2z.lo
              let d2Hi := castDimScalar (α:=α) (n:=d2z.dim) (n':=n) (h:=h2) d2z.hi
              let fyLo := getDimScalarFn (α:=α) yB.lo
              let fyHi := getDimScalarFn (α:=α) yB.hi
              let fdLeftLo := getDimScalarFn (α:=α) dLeftLo
              let fdLeftHi := getDimScalarFn (α:=α) dLeftHi
              let fdRightLo := getDimScalarFn (α:=α) dRightLo
              let fdRightHi := getDimScalarFn (α:=α) dRightHi
              let fd2Lo := getDimScalarFn (α:=α) d2Lo
              let fd2Hi := getDimScalarFn (α:=α) d2Hi
              let mulI (aLo aHi bLo bHi : α) : α × α :=
                let p1 := aLo * bLo; let p2 := aLo * bHi
                let p3 := aHi * bLo; let p4 := aHi * bHi
                let lo1 := if p1 < p2 then p1 else p2
                let lo2 := if p3 < p4 then p3 else p4
                let lo  := if lo1 < lo2 then lo1 else lo2
                let hi1 := if p1 > p2 then p1 else p2
                let hi2 := if p3 > p4 then p3 else p4
                let hi  := if hi1 > hi2 then hi1 else hi2
                (lo, hi)
              -- Bounds for (δ_ik - y_k)
              let deltaMinus (i k : Fin n) : α × α :=
                if decide (i.val = k.val) then
                  let ykLo := (fyLo k).item
                  let ykHi := (fyHi k).item
                  (1 - ykHi, 1 - ykLo)
                else
                  let ykLo := (fyLo k).item
                  let ykHi := (fyHi k).item
                  ((-ykHi), (-ykLo))
              -- J*d2z term per i
              let part1_lo :=
                Tensor.dim (fun i =>
                  let yiLo := (fyLo i).item
                  let yiHi := (fyHi i).item
                  let (sumLo, _sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                      let (accLo, accHi) := acc
                      let (dmkLo, dmkHi) := deltaMinus i k
                      let d2kLo := (fd2Lo k).item
                      let d2kHi := (fd2Hi k).item
                      let (jikLo, jikHi) := mulI yiLo yiHi dmkLo dmkHi
                      let (termLo, termHi) := mulI jikLo jikHi d2kLo d2kHi
                      (accLo + termLo, accHi + termHi)
                    ) (0, 0)
                  Tensor.scalar sumLo)
              let part1_hi :=
                Tensor.dim (fun i =>
                  let yiLo := (fyLo i).item
                  let yiHi := (fyHi i).item
                  let (_sumLo, sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                      let (accLo, accHi) := acc
                      let (dmkLo, dmkHi) := deltaMinus i k
                      let d2kLo := (fd2Lo k).item
                      let d2kHi := (fd2Hi k).item
                      let (jikLo, jikHi) := mulI yiLo yiHi dmkLo dmkHi
                      let (termLo, termHi) := mulI jikLo jikHi d2kLo d2kHi
                      (accLo + termLo, accHi + termHi)
                    ) (0, 0)
                  Tensor.scalar sumHi)
              -- Quadratic term Σ_{j,k} H_ijk dz_j dz_k, use interval-bounded H from y-bounds
              let part2_lo :=
                Tensor.dim (fun i =>
                  let yiLo := (fyLo i).item
                  let yiHi := (fyHi i).item
                  let (sumLo, _sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (j : Fin n) =>
                      let (accLo, accHi) := acc
                      let yjLo := (fyLo j).item
                      let yjHi := (fyHi j).item
                      let (dijLo, dijHi) : α × α := if decide (i.val = j.val) then (1 -
                        yjHi, 1 - yjLo) else ((-yjHi), (-yjLo))
                      (List.finRange n).foldl (fun (acc2 : α × α) (k : Fin n) =>
                        let (acc2Lo, acc2Hi) := acc2
                        let ykLo := (fyLo k).item
                        let ykHi := (fyHi k).item
                        let (dikLo, dikHi) : α × α := if decide (i.val = k.val) then (1 -
                          ykHi, 1 - ykLo) else ((-ykHi), (-ykLo))
                        -- H_ijk = y_i (dij)(dik) - y_i y_j (δ_jk - y_k)
                        let (t1Lo, t1Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi dijLo dijHi
                          mulI aLo aHi dikLo dikHi
                        let (delta_jk_Lo, delta_jk_Hi) : α × α := if decide (j.val = k.val) then
                          (1 - ykHi, 1 - ykLo) else ((-ykHi), (-ykLo))
                        let (t2Lo, t2Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi yjLo yjHi
                          mulI aLo aHi delta_jk_Lo delta_jk_Hi
                        -- H interval = t1 - t2
                        let hLo := t1Lo - t2Hi
                        let hHi := t1Hi - t2Lo
                        let dzjLo := (fdLeftLo j).item
                        let dzjHi := (fdLeftHi j).item
                        let dzkLo := (fdRightLo k).item
                        let dzkHi := (fdRightHi k).item
                        let (prodLo, prodHi) := mulI dzjLo dzjHi dzkLo dzkHi
                        let (termLo, termHi) := mulI hLo hHi prodLo prodHi
                        (acc2Lo + termLo, acc2Hi + termHi)
                      ) (accLo, accHi)
                    ) (0, 0)
                  Tensor.scalar sumLo)
              let part2_hi :=
                Tensor.dim (fun i =>
                  let yiLo := (fyLo i).item
                  let yiHi := (fyHi i).item
                  let (_sumLo, sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (j : Fin n) =>
                      let (accLo, accHi) := acc
                      let yjLo := (fyLo j).item
                      let yjHi := (fyHi j).item
                      let (dijLo, dijHi) : α × α := if decide (i.val = j.val) then (1 -
                        yjHi, 1 - yjLo) else ((-yjHi), (-yjLo))
                      (List.finRange n).foldl (fun (acc2 : α × α) (k : Fin n) =>
                        let (acc2Lo, acc2Hi) := acc2
                        let ykLo := (fyLo k).item
                        let ykHi := (fyHi k).item
                        let (dikLo, dikHi) : α × α := if decide (i.val = k.val) then (1 -
                          ykHi, 1 - ykLo) else ((-ykHi), (-ykLo))
                        let (t1Lo, t1Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi dijLo dijHi
                          mulI aLo aHi dikLo dikHi
                        let (delta_jk_Lo, delta_jk_Hi) : α × α := if decide (j.val = k.val) then
                          (1 - ykHi, 1 - ykLo) else ((-ykHi), (-ykLo))
                        let (t2Lo, t2Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi yjLo yjHi
                          mulI aLo aHi delta_jk_Lo delta_jk_Hi
                        let hLo := t1Lo - t2Hi
                        let hHi := t1Hi - t2Lo
                        let dzjLo := (fdLeftLo j).item
                        let dzjHi := (fdLeftHi j).item
                        let dzkLo := (fdRightLo k).item
                        let dzkHi := (fdRightHi k).item
                        let (prodLo, prodHi) := mulI dzjLo dzjHi dzkLo dzkHi
                        let (termLo, termHi) := mulI hLo hHi prodLo prodHi
                        (acc2Lo + termLo, acc2Hi + termHi)
                      ) (accLo, accHi)
                    ) (0, 0)
                  Tensor.scalar sumHi)
              let lo := Tensor.addSpec part1_lo part2_lo
              let hi := Tensor.addSpec part1_hi part2_hi
              d2s.set! id (some { dim := n, lo := lo, hi := hi })
              else d2s
            else d2s
          else d2s
        | _, _, _, _ => d2s
      | _ => d2s
    | .layernorm _ =>
      -- The existing diagonal second-order formula does not establish the mixed bilinear term.
      -- Leave the node unresolved until a row-wise LayerNorm Hessian enclosure is available.
      d2s
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      d2s
    | .conv .. | .batchNormEval .. => d2s
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
