/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.IBP

/-!
# Derivative Interval Passes

These passes propagate interval bounds for first and second derivatives through the same flat graph
used by IBP. They are kept separate from the value IBP pass because derivative propagation has its
own chain-rule state, and it reuses the same `FlatBox` representation.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/-- Derivative IBP pass for 1D input: computes for each node an interval on dy/dx. Requires value
  IBP boxes for activations. -/
def runDeriv1D (g : Graph) (ps : ParamStore α) (ibp : Array (Option (FlatBox α))) : Array (Option
  (FlatBox α)) :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (drs : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        let one := Spec.fill (α:=α) Numbers.one (.dim B.dim .scalar)
        drs.set! id (some { dim := B.dim, lo := one, hi := one })
      | none => drs
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim v.n .scalar)
        drs.set! id (some { dim := v.n, lo := z, hi := z })
      | none => drs
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Spec.fill (α:=α) Numbers.zero (.dim d .scalar)
      drs.set! id (some { dim := d, lo := z, hi := z })
    | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad .. =>
      -- Not supported by the derivative-bound passes (used by PINN tooling).
      drs
    | .sum =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]! with
        | some dXin =>
          let loVal := Spec.Tensor.sumSpec dXin.lo
          let hiVal := Spec.Tensor.sumSpec dXin.hi
          let loT := Spec.fill (α := α) loVal (.dim 1 .scalar)
          let hiT := Spec.fill (α := α) hiVal (.dim 1 .scalar)
          drs.set! id (some { dim := 1, lo := loT, hi := hiT })
        | none => drs
      | _ => drs
    | .linear =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ps.linearWB[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .matmul =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ps.matmulW[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .relu =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]! with
        | some dIn =>
          let z := Spec.fill (α:=α) Numbers.zero (.dim dIn.dim .scalar)
          let o := Spec.fill (α:=α) Numbers.one  (.dim dIn.dim .scalar)
          let dF : FlatBox α := { dim := dIn.dim, lo := z, hi := o }
          match box_mul_elem (α:=α) dIn dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .tanh =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          -- Use tighter derivative bounds: tanh'(z) = 1 - tanh(z)^2, with tanh(z) ∈ [yl, yh]
          if dZ.dim = yB.dim then
            let fyLo := getDimScalarFn (α:=α) yB.lo
            let fyHi := getDimScalarFn (α:=α) yB.hi
            let dlo :=
              Tensor.dim (fun i =>
                match fyLo i, fyHi i with
                | .scalar yl, .scalar yh =>
                  let yl2 := yl * yl
                  let yh2 := yh * yh
                  let s_max := if yl2 > yh2 then yl2 else yh2
                  Tensor.scalar (Numbers.one - s_max))
            let dhi :=
              Tensor.dim (fun i =>
                match fyLo i, fyHi i with
                | .scalar yl, .scalar yh =>
                  let yl2 := yl * yl
                  let yh2 := yh * yh
                  let s_min :=
                    if yl < Numbers.zero then
                      if Numbers.zero < yh then Numbers.zero else (if yl2 < yh2 then yl2 else yh2)
                    else (if yl2 < yh2 then yl2 else yh2)
                  Tensor.scalar (Numbers.one - s_min))
            let dF : FlatBox α := { dim := yB.dim, lo := dlo, hi := dhi }
            match box_mul_elem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          else drs
        | _, _ => drs
      | _ => drs
    | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some sB =>
          if dZ.dim = sB.dim then
            let fsLo := getDimScalarFn (α:=α) sB.lo
            let fsHi := getDimScalarFn (α:=α) sB.hi
            let dlo :=
              Tensor.dim (fun i =>
                match fsLo i, fsHi i with
                | .scalar a, .scalar b =>
                  let fa := a * (Numbers.one - a)
                  let fb := b * (Numbers.one - b)
                  let mn := if fa < fb then fa else fb
                  Tensor.scalar mn)
            let dhi :=
              Tensor.dim (fun i =>
                match fsLo i, fsHi i with
                | .scalar a, .scalar b =>
                  let fa := a * (Numbers.one - a)
                  let fb := b * (Numbers.one - b)
                  let mxEnds := if fa > fb then fa else fb
                  let quarter := Numbers.pointfive * (Numbers.one - Numbers.pointfive)
                  let mx :=
                    if a < Numbers.pointfive then
                      if Numbers.pointfive < b then
                        let mx' := if mxEnds < quarter then quarter else mxEnds
                        mx'
                      else mxEnds
                    else mxEnds
                  Tensor.scalar mx)
            let dF : FlatBox α := { dim := sB.dim, lo := dlo, hi := dhi }
            match box_mul_elem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          else drs
        | _, _ => drs
      | _ => drs
    | .softmax _ =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          if h : dZ.dim = yB.dim then
            let n := yB.dim
            -- Cast derivative tensors to dimension n for Fin alignment
            let dLo := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h) dZ.lo
            let dHi := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h) dZ.hi
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
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (sumLo, _sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumLo)
            let dhi :=
              Tensor.dim (fun i =>
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (_sumLo, sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
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
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          let cB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let dF : FlatBox α := { dim := zB.dim, lo := cB.lo, hi := cB.hi }
          match box_mul_elem (α:=α) dZ dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .cos =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          let sB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let negSB : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) sB.hi
              hi := Tensor.mapSpec (fun x => -x) sB.lo }
          match box_mul_elem (α:=α) dZ negSB with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .exp =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match chainMul (α:=α) dZ (derivBoxExp (α:=α) zB) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .log =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match chainMul (α:=α) dZ (derivBoxLog (α:=α) zB) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (box_add (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (box_sub (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]!, ibp[p1]!, ibp[p2]! with
        | some dx, some dy, some xB, some yB =>
          match box_mul_elem (α:=α) dx yB, box_mul_elem (α:=α) xB dy with
          | some t1, some t2 => drs.set! id (some (box_add (α:=α) t1 t2))
          | _, _ => drs
        | _, _, _, _ => drs
      | _ => drs
    | .layernorm _ =>
      -- Derivative of layernorm y = (x - mean(x))/sqrt(var+eps): dy ≈ t*(dx - mean dx) + dt*u.
      -- We bound t in [t_lo,t_hi], bound v := (dx - mean dx) per-component, and bound |dt| via
      -- |dt| ≤ 0.5 * (var+eps)^(-3/2)_hi * (2/n) * Σ_j max|u_j| * max|v_j|; then add symmetric dt*u
      -- term.
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dXin, some Xin =>
          let n := Xin.dim
          if hn : dXin.dim = n then
            -- Compute mean bounds of x and dx
            let muBounds := layerNormMeanBounds (α := α) Xin.lo Xin.hi
            -- Empty vectors have no coordinates to certify; use a nonzero dummy denominator to
            -- avoid evaluating `1 / 0` in the vacuous branch.
            let nDen : Nat := if n = 0 then 1 else n
            let nA : α := (nDen : Nat)
            let mu_lo := muBounds.1
            let mu_hi := muBounds.2
            -- u_j bounds = x_j - mean(x)
            let uBounds := layerNormCenteredBounds (α := α) Xin.lo Xin.hi mu_lo mu_hi
            let u_lo := uBounds.1
            let u_hi := uBounds.2
            -- Bounds on variance and denom s = sqrt(var+eps)
            let var_hi := layerNormVarianceUpper (α := α) Xin.lo Xin.hi mu_lo mu_hi
            let s_lo := MathFunctions.sqrt Numbers.epsilon
            let tBounds := layerNormInvStdBounds (α := α) var_hi
            let t_lo := tBounds.1
            let t_hi := tBounds.2
            -- dx mean bounds and v_j = dx_j - mean(dx)
            let dmuBounds := layerNormMeanBounds (α := α) dXin.lo dXin.hi
            let vBounds := layerNormCenteredBounds (α := α) dXin.lo dXin.hi dmuBounds.1 dmuBounds.2
            let v_lo := vBounds.1
            let v_hi := vBounds.2
            -- Align v bounds to dimension n via cast
            let v_loN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_lo
            let v_hiN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_hi
            -- First term: t * v
            -- Compute base = t * v per component where t∈[t_lo,t_hi] and v_i∈[v_loN[i],v_hiN[i]]
            let vLoFn := getDimScalarFn (α:=α) v_loN
            let vHiFn := getDimScalarFn (α:=α) v_hiN
            let base_lo :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let m1 := if p1 < p2 then p1 else p2
                  let m2 := if p3 < p4 then p3 else p4
                  Tensor.scalar (if m1 < m2 then m1 else m2))
            let base_hi :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let M1 := if p1 > p2 then p1 else p2
                  let M2 := if p3 > p4 then p3 else p4
                  Tensor.scalar (if M1 > M2 then M1 else M2))
            let baseN : FlatBox α := { dim := n, lo := base_lo, hi := base_hi }
            -- Bound |dt| using t3_hi = 1/s^3 and |(2/n) Σ u_j v_j|
            let t3_hi :=
              let s_lo' := s_lo
              let s3 := s_lo' * s_lo' * s_lo'
              Numbers.one / (if s3 > Numbers.epsilon then s3 else Numbers.epsilon)
            let abs_max (l u : α) : α :=
              let al := MathFunctions.abs l
              let au := MathFunctions.abs u
              if al > au then al else au
            -- compute G = Σ max|u_j| * max|v_j|
            let u_abs := getDimScalarFn (α:=α) u_lo
            let u_abs_hi := getDimScalarFn (α:=α) u_hi
            let v_abs := getDimScalarFn (α:=α) v_loN
            let v_abs_hi := getDimScalarFn (α:=α) v_hiN
            let G : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
              match u_abs i, u_abs_hi i, v_abs i, v_abs_hi i with
              | .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                let au := abs_max ul uu
                let av := abs_max vl vu
                acc + (au * av)
            ) 0
            let V := (Numbers.two * G) / nA
            let dt_abs := (Numbers.pointfive * t3_hi) * V
            -- Add symmetric dt*u term per component: ± dt_abs * max|u_i|
            let fulo := getDimScalarFn (α:=α) u_lo
            let fuhi := getDimScalarFn (α:=α) u_hi
            let bLoFn := getDimScalarFn (α:=α) baseN.lo
            let bHiFn := getDimScalarFn (α:=α) baseN.hi
            let add_lo :=
              Tensor.dim (fun i =>
                match bLoFn i, fulo i, fuhi i with
                | .scalar bi, .scalar ul, .scalar uu =>
                  let au := abs_max ul uu
                  Tensor.scalar (bi - dt_abs * au))
            let add_hi :=
              Tensor.dim (fun i =>
                match bHiFn i, fulo i, fuhi i with
                | .scalar bi, .scalar ul, .scalar uu =>
                  let au := abs_max ul uu
                  Tensor.scalar (bi + dt_abs * au))
            drs.set! id (some { dim := n, lo := add_lo, hi := add_hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .reshape _ _ | .flatten _ | .concat _ | .swap_first_two | .transpose3dLastTwo | .permute _
      =>
      match node.parents with
      | p1 :: _ => drs.set! id (drs[p1]!)
      | _ => drs
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      drs
    | .mseLoss => drs
    | .conv2d .. | .batchNorm2dNchwEval .. => drs
  (List.finRange g.nodes.size).foldl propagate init

/-- Directional first-derivative pass: like `runDeriv1D` but seeds the derivative at the
    input node with a user-provided direction vector (as a FlatBox with lo=hi). This allows
    extracting partial derivatives for multi-dimensional inputs by choosing e_x, e_y, etc. -/
def runDerivDirectional (g : Graph) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α))) (seed : FlatBox α) : Array (Option (FlatBox α)) :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (drs : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        if h : seed.dim = B.dim then
          let dlo := castDimScalar (α:=α) (n:=seed.dim) (n':=B.dim) (h:=h) seed.lo
          let dhi := castDimScalar (α:=α) (n:=seed.dim) (n':=B.dim) (h:=h) seed.hi
          drs.set! id (some { dim := B.dim, lo := dlo, hi := dhi })
        else drs
      | none => drs
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim v.n .scalar)
        drs.set! id (some { dim := v.n, lo := z, hi := z })
      | none => drs
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Spec.fill (α:=α) Numbers.zero (.dim d .scalar)
      drs.set! id (some { dim := d, lo := z, hi := z })
    | .sin =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          let cB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let dF : FlatBox α := { dim := zB.dim, lo := cB.lo, hi := cB.hi }
          match box_mul_elem (α:=α) dZ dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .cos =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          let sB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let negSB : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) sB.hi
              hi := Tensor.mapSpec (fun x => -x) sB.lo }
          match box_mul_elem (α:=α) dZ negSB with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad .. =>
      -- Not supported by the derivative-bound passes.
      drs
    | .sum =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]! with
        | some dXin =>
          let loVal := Spec.Tensor.sumSpec dXin.lo
          let hiVal := Spec.Tensor.sumSpec dXin.hi
          let loT := Spec.fill (α := α) loVal (.dim 1 .scalar)
          let hiT := Spec.fill (α := α) hiVal (.dim 1 .scalar)
          drs.set! id (some { dim := 1, lo := loT, hi := hiT })
        | none => drs
      | _ => drs
    | .linear =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ps.linearWB[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .matmul =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ps.matmulW[id]? with
        | some dXin, some p =>
          if h : dXin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := dXin.lo, hi :=
              dXin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            drs.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else drs
        | _, _ => drs
      | _ => drs
    | .relu =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]! with
        | some dIn =>
          let z := Spec.fill (α:=α) Numbers.zero (.dim dIn.dim .scalar)
          let o := Spec.fill (α:=α) Numbers.one  (.dim dIn.dim .scalar)
          let dF : FlatBox α := { dim := dIn.dim, lo := z, hi := o }
          match box_mul_elem (α:=α) dIn dF with
          | some prod => drs.set! id (some prod)
          | none => drs
        | none => drs
      | _ => drs
    | .tanh =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          if dZ.dim = yB.dim then
            let fyLo := getDimScalarFn (α:=α) yB.lo
            let fyHi := getDimScalarFn (α:=α) yB.hi
            let dlo :=
              Tensor.dim (fun i =>
                match fyLo i, fyHi i with
                | .scalar yl, .scalar yh =>
                  let yl2 := yl * yl
                  let yh2 := yh * yh
                  let s_max := if yl2 > yh2 then yl2 else yh2
                  Tensor.scalar (Numbers.one - s_max))
            let dhi :=
              Tensor.dim (fun i =>
                match fyLo i, fyHi i with
                | .scalar yl, .scalar yh =>
                  let yl2 := yl * yl
                  let yh2 := yh * yh
                  let s_min :=
                    if yl < Numbers.zero then
                      if Numbers.zero < yh then Numbers.zero else (if yl2 < yh2 then yl2 else yh2)
                    else (if yl2 < yh2 then yl2 else yh2)
                  Tensor.scalar (Numbers.one - s_min))
            let dF : FlatBox α := { dim := yB.dim, lo := dlo, hi := dhi }
            match box_mul_elem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          else drs
        | _, _ => drs
      | _ => drs
    | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some sB =>
          if dZ.dim = sB.dim then
            let fsLo := getDimScalarFn (α:=α) sB.lo
            let fsHi := getDimScalarFn (α:=α) sB.hi
            let dlo :=
              Tensor.dim (fun i =>
                match fsLo i, fsHi i with
                | .scalar a, .scalar b =>
                  let fa := a * (Numbers.one - a)
                  let fb := b * (Numbers.one - b)
                  let mn := if fa < fb then fa else fb
                  Tensor.scalar mn)
            let dhi :=
              Tensor.dim (fun i =>
                match fsLo i, fsHi i with
                | .scalar a, .scalar b =>
                  let fa := a * (Numbers.one - a)
                  let fb := b * (Numbers.one - b)
                  let mxEnds := if fa > fb then fa else fb
                  let quarter := Numbers.pointfive * (Numbers.one - Numbers.pointfive)
                  let mx :=
                    if a < Numbers.pointfive then
                      if Numbers.pointfive < b then
                        let mx' := if mxEnds < quarter then quarter else mxEnds
                        mx'
                      else mxEnds
                    else mxEnds
                  Tensor.scalar mx)
            let dF : FlatBox α := { dim := sB.dim, lo := dlo, hi := dhi }
            match box_mul_elem (α:=α) dZ dF with
            | some prod => drs.set! id (some prod)
            | none => drs
          else drs
        | _, _ => drs
      | _ => drs
    | .softmax _ =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[id]! with
        | some dZ, some yB =>
          if h : dZ.dim = yB.dim then
            let n := yB.dim
            let dLo := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h) dZ.lo
            let dHi := castDimScalar (α:=α) (n:=dZ.dim) (n':=n) (h:=h) dZ.hi
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
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (sumLo, _sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumLo)
            let dhi :=
              Tensor.dim (fun i =>
                let yiLo := match fyLo i with | .scalar v => v
                let yiHi := match fyHi i with | .scalar v => v
                let (_sumLo, sumHi) :=
                  (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                    let (accLo, accHi) := acc
                    let ykLo := match fyLo k with | .scalar v => v
                    let ykHi := match fyHi k with | .scalar v => v
                    let (jikLo, jikHi) :=
                      if decide (i.val = k.val) then
                        let oneMinusLo := Numbers.one - yiHi
                        let oneMinusHi := Numbers.one - yiLo
                        mulI yiLo yiHi oneMinusLo oneMinusHi
                      else
                        let negLo := (-ykHi)
                        let negHi := (-ykLo)
                        mulI yiLo yiHi negLo negHi
                    let dxLo := match fdLo k with | .scalar v => v
                    let dxHi := match fdHi k with | .scalar v => v
                    let (termLo, termHi) := mulI jikLo jikHi dxLo dxHi
                    (accLo + termLo, accHi + termHi)
                  ) (0, 0)
                Tensor.scalar sumHi)
            drs.set! id (some { dim := n, lo := dlo, hi := dhi })
          else drs
        | _, _ => drs
      | _ => drs
    | .exp =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match chainMul (α:=α) dZ (derivBoxExp (α:=α) zB) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .log =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dZ, some zB =>
          match chainMul (α:=α) dZ (derivBoxLog (α:=α) zB) with
          | some prod => drs.set! id (some prod)
          | none => drs
        | _, _ => drs
      | _ => drs
    | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (box_add (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]! with
        | some d1, some d2 => some (box_sub (α:=α) d1 d2) |> fun r => drs.set! id r
        | _, _ => drs
      | _ => drs
    | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match drs[p1]!, drs[p2]!, ibp[p1]!, ibp[p2]! with
        | some dx, some dy, some xB, some yB =>
          match box_mul_elem (α:=α) dx yB, box_mul_elem (α:=α) xB dy with
          | some t1, some t2 => drs.set! id (some (box_add (α:=α) t1 t2))
          | _, _ => drs
        | _, _, _, _ => drs
      | _ => drs
    | .layernorm _ =>
      match node.parents with
      | p1 :: _ =>
        match drs[p1]!, ibp[p1]! with
        | some dXin, some Xin =>
          let n := Xin.dim
          if hn : dXin.dim = n then
            let muBounds := layerNormMeanBounds (α := α) Xin.lo Xin.hi
            let mu_lo := muBounds.1
            let mu_hi := muBounds.2
            let var_hi := layerNormVarianceUpper (α := α) Xin.lo Xin.hi mu_lo mu_hi
            let tBounds := layerNormInvStdBounds (α := α) var_hi
            let t_lo := tBounds.1
            let t_hi := tBounds.2
            let dmuBounds := layerNormMeanBounds (α := α) dXin.lo dXin.hi
            let vBounds := layerNormCenteredBounds (α := α) dXin.lo dXin.hi dmuBounds.1 dmuBounds.2
            let v_lo := vBounds.1
            let v_hi := vBounds.2
            let v_loN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_lo
            let v_hiN := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn) v_hi
            let vLoFn := getDimScalarFn (α:=α) v_loN
            let vHiFn := getDimScalarFn (α:=α) v_hiN
            let base_lo :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let m1 := if p1 < p2 then p1 else p2
                  let m2 := if p3 < p4 then p3 else p4
                  Tensor.scalar (if m1 < m2 then m1 else m2))
            let base_hi :=
              Tensor.dim (fun i =>
                match vLoFn i, vHiFn i with
                | .scalar vl, .scalar vu =>
                  let p1 := t_lo * vl
                  let p2 := t_lo * vu
                  let p3 := t_hi * vl
                  let p4 := t_hi * vu
                  let M1 := if p1 > p2 then p1 else p2
                  let M2 := if p3 > p4 then p3 else p4
                  Tensor.scalar (if M1 > M2 then M1 else M2))
            let baseN : FlatBox α := { dim := n, lo := base_lo, hi := base_hi }
            drs.set! id (some baseN)
          else drs
        | _, _ => drs
      | _ => drs
    | .reshape _ _ | .flatten _ | .concat _ | .swap_first_two | .transpose3dLastTwo | .permute _
      =>
      match node.parents with
      | p1 :: _ => drs.set! id (drs[p1]!)
      | _ => drs
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      drs
    | .mseLoss => drs
    | .conv2d .. | .batchNorm2dNchwEval .. => drs
  (List.finRange g.nodes.size).foldl propagate init

/-- Second-derivative IBP pass for 1D input: computes per node an interval on d²y/dx².
    Requires value IBP boxes and first-derivative boxes. Covers input, linear/matmul, tanh, add/sub.
      -/
def runDeriv2D (g : Graph) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α))) (d1 : Array (Option (FlatBox α))) : Array (Option (FlatBox α))
    :=
  let init : Array (Option (FlatBox α)) := Array.replicate g.nodes.size none
  let propagate (d2s : Array (Option (FlatBox α))) (id : Nat) : Array (Option (FlatBox α)) :=
    let node := g.nodes[id]!
    match node.kind with
    | .input =>
      match ps.inputBoxes[id]? with
      | some B =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim B.dim .scalar)
        d2s.set! id (some { dim := B.dim, lo := z, hi := z })
      | none => d2s
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        let z := Spec.fill (α:=α) Numbers.zero (.dim v.n .scalar)
        d2s.set! id (some { dim := v.n, lo := z, hi := z })
      | none => d2s
    | .detach | .randUniform _ | .bernoulliMask _ =>
      let d := node.outShape.size
      let z := Spec.fill (α:=α) Numbers.zero (.dim d .scalar)
      d2s.set! id (some { dim := d, lo := z, hi := z })
    | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad .. =>
      -- Not supported by the second-derivative bound pass.
      d2s
    | .linear =>
      match node.parents with
      | p1 :: _ =>
        match d2s[p1]!, ps.linearWB[id]? with
        | some d2Xin, some p =>
          if h : d2Xin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := d2Xin.lo, hi :=
              d2Xin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            d2s.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else d2s
        | _, _ => d2s
      | _ => d2s
    | .matmul =>
      match node.parents with
      | p1 :: _ =>
        match d2s[p1]!, ps.matmulW[id]? with
        | some d2Xin, some p =>
          if h : d2Xin.dim = p.n then
            let xB : Box α (.dim p.n .scalar) := castBoxDim (α:=α) (h:=h) { lo := d2Xin.lo, hi :=
              d2Xin.hi }
            let zeroB : Box α (.dim p.m .scalar) :=
              let z := Spec.fill (α:=α) Numbers.zero (.dim p.m .scalar)
              Box.point (α:=α) z
            let yB := NN.MLTheory.CROWN.IBP.linear (α:=α) (m:=p.m) (n:=p.n) p.w xB zeroB
            d2s.set! id (some { dim := p.m, lo := yB.lo, hi := yB.hi })
          else d2s
        | _, _ => d2s
      | _ => d2s
    | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match d2s[p1]!, d2s[p2]! with
        | some a, some b => d2s.set! id (some (box_add (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match d2s[p1]!, d2s[p2]! with
        | some a, some b => d2s.set! id (some (box_sub (α:=α) a b))
        | _, _ => d2s
      | _ => d2s
    | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]!, d1[p1]!, d1[p2]!, d2s[p1]!, d2s[p2]! with
        | some xB, some yB, some dx, some dy, some d2x, some d2y =>
          -- y'' = x''⊙y + 2 x'⊙y' + x⊙y''
          match box_mul_elem (α:=α) d2x yB, box_mul_elem (α:=α) dx dy, box_mul_elem (α:=α) xB d2y
            with
          | some t1, some mid, some t3 =>
            let twoMid := box_add (α:=α) mid mid
            d2s.set! id (some (box_add (α:=α) t1 (box_add (α:=α) twoMid t3)))
          | _, _, _ => d2s
        | _, _, _, _, _, _ => d2s
      | _ => d2s
    | .relu =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]! with
        | some zB =>
          let z := Spec.fill (α:=α) Numbers.zero (.dim zB.dim .scalar)
          d2s.set! id (some { dim := zB.dim, lo := z, hi := z })
        | none => d2s
      | _ => d2s
    | .tanh =>
      match node.parents with
      | p1 :: _ =>
        match ibp[id]!, d1[p1]!, d2s[p1]! with
        | some yB, some dz, some d2z =>
          if dz.dim = yB.dim then
            if d2z.dim = yB.dim then
              -- f'(z) = 1 - y^2; f''(z) = -2 y (1 - y^2)
              let fyLo := getDimScalarFn (α:=α) yB.lo
              let fyHi := getDimScalarFn (α:=α) yB.hi
              let f1_lo :=
                Tensor.dim (fun i =>
                  match fyLo i, fyHi i with
                  | .scalar yl, .scalar yh =>
                    let yl2 := yl * yl
                    let yh2 := yh * yh
                    let s_max := if yl2 > yh2 then yl2 else yh2
                    Tensor.scalar (Numbers.one - s_max))
              let f1_hi :=
                Tensor.dim (fun i =>
                  match fyLo i, fyHi i with
                  | .scalar yl, .scalar yh =>
                    let yl2 := yl * yl
                    let yh2 := yh * yh
                    let s_min :=
                      if yl < Numbers.zero then
                        if Numbers.zero < yh then Numbers.zero else (if yl2 < yh2 then yl2 else yh2)
                      else (if yl2 < yh2 then yl2 else yh2)
                    Tensor.scalar (Numbers.one - s_min))
              let f2_lo :=
                Tensor.dim (fun i =>
                  match fyLo i, fyHi i with
                  | .scalar yl, .scalar yh =>
                    let cube (v : α) := v * v * v
                    let cand1 := (-(Numbers.two) * yl) + (Numbers.two * cube yl)
                    let cand2 := (-(Numbers.two) * yh) + (Numbers.two * cube yh)
                    let rt := MathFunctions.sqrt (Numbers.one / Numbers.three)
                    -- evaluate at +/- 1/sqrt(3) when inside interval
                    let cand3 := if (yl < rt ∧ rt < yh) then ((-(Numbers.two) * rt) + (Numbers.two *
                      cube rt)) else cand1
                    let nrt := (-rt)
                    let cand4 := if (yl < nrt ∧ nrt < yh) then ((-(Numbers.two) * nrt) +
                      (Numbers.two * cube nrt)) else cand2
                    let m1 := if cand1 < cand2 then cand1 else cand2
                    let m2 := if cand3 < cand4 then cand3 else cand4
                    Tensor.scalar (if m1 < m2 then m1 else m2))
              let f2_hi :=
                Tensor.dim (fun i =>
                  match fyLo i, fyHi i with
                  | .scalar yl, .scalar yh =>
                    let cube (v : α) := v * v * v
                    let cand1 := (-(Numbers.two) * yl) + (Numbers.two * cube yl)
                    let cand2 := (-(Numbers.two) * yh) + (Numbers.two * cube yh)
                    let rt := MathFunctions.sqrt (Numbers.one / Numbers.three)
                    let cand3 := if (yl < rt ∧ rt < yh) then ((-(Numbers.two) * rt) + (Numbers.two *
                      cube rt)) else cand1
                    let nrt := (-rt)
                    let cand4 := if (yl < nrt ∧ nrt < yh) then ((-(Numbers.two) * nrt) +
                      (Numbers.two * cube nrt)) else cand2
                    let M1 := if cand1 > cand2 then cand1 else cand2
                    let M2 := if cand3 > cand4 then cand3 else cand4
                    Tensor.scalar (if M1 > M2 then M1 else M2))
              let f1B : FlatBox α := { dim := yB.dim, lo := f1_lo, hi := f1_hi }
              let f2B : FlatBox α := { dim := yB.dim, lo := f2_lo, hi := f2_hi }
              let dz2 := boxSquare (α:=α) dz
              match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
              | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
              | _, _ => d2s
            else d2s
          else d2s
        | _, _, _ => d2s
      | _ => d2s
    | .sin =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some zB, some dz, some d2z =>
          -- y = sin(z): y'' = (-sin(z))*(z')^2 + cos(z)*z''
          let sinB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let cosB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let f1B : FlatBox α := { dim := zB.dim, lo := cosB.lo, hi := cosB.hi }
          let f2B : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) sinB.hi
              hi := Tensor.mapSpec (fun x => -x) sinB.lo }
          let dz2 := boxSquare (α:=α) dz
          match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
          | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
          | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .cos =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some zB, some dz, some d2z =>
          -- y = cos(z): y'' = (-cos(z))*(z')^2 + (-sin(z))*z''
          let sinB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let cosB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α:=α) (n:=zB.dim) (ofFlatBox zB)
          let f1B : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) sinB.hi
              hi := Tensor.mapSpec (fun x => -x) sinB.lo }
          let f2B : FlatBox α :=
            { dim := zB.dim
              lo := Tensor.mapSpec (fun x => -x) cosB.hi
              hi := Tensor.mapSpec (fun x => -x) cosB.lo }
          let dz2 := boxSquare (α:=α) dz
          match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
          | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
          | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
        match ibp[id]!, d1[p1]!, d2s[p1]! with
        | some sB, some dz, some d2z =>
          if dz.dim = sB.dim then
            if d2z.dim = sB.dim then
              let fsLo := getDimScalarFn (α:=α) sB.lo
              let fsHi := getDimScalarFn (α:=α) sB.hi
              -- f'(z) = s(1-s)
              let f1_lo :=
                Tensor.dim (fun i =>
                  match fsLo i, fsHi i with
                  | .scalar a, .scalar b =>
                    let fa := a * (Numbers.one - a)
                    let fb := b * (Numbers.one - b)
                    let mn := if fa < fb then fa else fb
                    Tensor.scalar mn)
              let f1_hi :=
                Tensor.dim (fun i =>
                  match fsLo i, fsHi i with
                  | .scalar a, .scalar b =>
                    let fa := a * (Numbers.one - a)
                    let fb := b * (Numbers.one - b)
                    let quarter := Numbers.pointfive * (Numbers.one - Numbers.pointfive)
                    let mxEnds := if fa > fb then fa else fb
                    let mx := if a < Numbers.pointfive then (if Numbers.pointfive < b then (if
                      mxEnds < quarter then quarter else mxEnds) else mxEnds) else mxEnds
                    Tensor.scalar mx)
              -- f''(z) = f'(z) * (1 - 2s) with s in [a,b]
              let oneMinus2s_lo :=
                Tensor.dim (fun i =>
                  match fsLo i, fsHi i with
                  | .scalar _a, .scalar b => Tensor.scalar (Numbers.one - (Numbers.two * b)))
              let oneMinus2s_hi :=
                Tensor.dim (fun i =>
                  match fsLo i, fsHi i with
                  | .scalar a, .scalar _b => Tensor.scalar (Numbers.one - (Numbers.two * a)))
              let f1B : FlatBox α := { dim := sB.dim, lo := f1_lo, hi := f1_hi }
              let f2fac : FlatBox α := { dim := sB.dim, lo := oneMinus2s_lo, hi := oneMinus2s_hi }
              match box_mul_elem (α:=α) f1B f2fac with
              | some f2B =>
                let dz2 := boxSquare (α:=α) dz
                match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
                | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
                | _, _ => d2s
              | none => d2s
            else d2s
          else d2s
        | _, _, _ => d2s
      | _ => d2s
    | .exp =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some zB, some dz, some d2z =>
          let f1 := { dim := zB.dim, lo := Tensor.expSpec zB.lo, hi := Tensor.expSpec zB.hi }
          let f2 := f1 -- same for exp
          let dz2 := boxSquare (α:=α) dz
          match box_mul_elem (α:=α) f2 dz2, box_mul_elem (α:=α) f1 d2z with
          | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
          | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .log =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some zB, some dz, some d2z =>
          let flo := getDimScalarFn (α:=α) zB.lo
          let fhi := getDimScalarFn (α:=α) zB.hi
          let f1_lo :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar _l, .scalar u => Tensor.scalar (Numbers.one / u))
          let f1_hi :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar l, .scalar _u =>
                let l' := if l > Numbers.epsilon then l else Numbers.epsilon
                Tensor.scalar (Numbers.one / l'))
          let f2_lo :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar l, .scalar _u =>
                let l' := if l > Numbers.epsilon then l else Numbers.epsilon
                -- f'' = -1/z^2 in [-(1/l'^2), -(1/u^2)] with l' ≤ u
                Tensor.scalar (-(Numbers.one / (l' * l'))))
          let f2_hi :=
            Tensor.dim (fun i =>
              match flo i, fhi i with
              | .scalar _l, .scalar u =>
                -- upper bound (less negative): -(1/u^2)
                Tensor.scalar (-(Numbers.one / (u * u))))
          let f1B : FlatBox α := { dim := zB.dim, lo := f1_lo, hi := f1_hi }
          let f2B : FlatBox α := { dim := zB.dim, lo := f2_lo, hi := f2_hi }
          let dz2 := boxSquare (α:=α) dz
          match box_mul_elem (α:=α) f2B dz2, box_mul_elem (α:=α) f1B d2z with
          | some tA, some tB => d2s.set! id (some (box_add (α:=α) tA tB))
          | _, _ => d2s
        | _, _, _ => d2s
      | _ => d2s
    | .sum =>
      match node.parents with
      | p1 :: _ =>
        match d2s[p1]! with
        | some d2Xin =>
          let loVal := Spec.Tensor.sumSpec d2Xin.lo
          let hiVal := Spec.Tensor.sumSpec d2Xin.hi
          let loT := Spec.fill (α := α) loVal (.dim 1 .scalar)
          let hiT := Spec.fill (α := α) hiVal (.dim 1 .scalar)
          d2s.set! id (some { dim := 1, lo := loT, hi := hiT })
        | none => d2s
      | _ => d2s
    | .reshape _ _ | .flatten _ | .concat _ | .swap_first_two | .transpose3dLastTwo | .permute _
      =>
      match node.parents with
      | p1 :: _ => d2s.set! id (d2s[p1]!)
      | _ => d2s
    | .mseLoss => d2s
    | .softmax _ =>
      -- y''_i = Σ_k J_ik d2z_k + Σ_{j,k} H_ijk dz_j dz_k, with
      -- J = diag(y) - y yᵀ and H derived from ∂J/∂z (bounded via y-bounds).
      match node.parents with
      | p1 :: _ =>
        match ibp[id]!, d1[p1]!, d2s[p1]! with
        | some yB, some dz, some d2z =>
          if h1 : dz.dim = yB.dim then
            if h2 : d2z.dim = yB.dim then
              let n := yB.dim
              -- Cast derivative tensors to dimension n for Fin alignment
              let d1Lo := castDimScalar (α:=α) (n:=dz.dim) (n':=n) (h:=h1) dz.lo
              let d1Hi := castDimScalar (α:=α) (n:=dz.dim) (n':=n) (h:=h1) dz.hi
              let d2Lo := castDimScalar (α:=α) (n:=d2z.dim) (n':=n) (h:=h2) d2z.lo
              let d2Hi := castDimScalar (α:=α) (n:=d2z.dim) (n':=n) (h:=h2) d2z.hi
              let fyLo := getDimScalarFn (α:=α) yB.lo
              let fyHi := getDimScalarFn (α:=α) yB.hi
              let fd1Lo := getDimScalarFn (α:=α) d1Lo
              let fd1Hi := getDimScalarFn (α:=α) d1Hi
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
                  let ykLo := match fyLo k with | .scalar v => v
                  let ykHi := match fyHi k with | .scalar v => v
                  (Numbers.one - ykHi, Numbers.one - ykLo)
                else
                  let ykLo := match fyLo k with | .scalar v => v
                  let ykHi := match fyHi k with | .scalar v => v
                  ((-ykHi), (-ykLo))
              -- J*d2z term per i
              let part1_lo :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (sumLo, _sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                      let (accLo, accHi) := acc
                      let (dmkLo, dmkHi) := deltaMinus i k
                      let d2kLo := match fd2Lo k with | .scalar v => v
                      let d2kHi := match fd2Hi k with | .scalar v => v
                      let (jikLo, jikHi) := mulI yiLo yiHi dmkLo dmkHi
                      let (termLo, termHi) := mulI jikLo jikHi d2kLo d2kHi
                      (accLo + termLo, accHi + termHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumLo)
              let part1_hi :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (_sumLo, sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (k : Fin n) =>
                      let (accLo, accHi) := acc
                      let (dmkLo, dmkHi) := deltaMinus i k
                      let d2kLo := match fd2Lo k with | .scalar v => v
                      let d2kHi := match fd2Hi k with | .scalar v => v
                      let (jikLo, jikHi) := mulI yiLo yiHi dmkLo dmkHi
                      let (termLo, termHi) := mulI jikLo jikHi d2kLo d2kHi
                      (accLo + termLo, accHi + termHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumHi)
              -- Quadratic term Σ_{j,k} H_ijk dz_j dz_k, use interval-bounded H from y-bounds
              let part2_lo :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (sumLo, _sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (j : Fin n) =>
                      let (accLo, accHi) := acc
                      let yjLo := match fyLo j with | .scalar v => v
                      let yjHi := match fyHi j with | .scalar v => v
                      let (dijLo, dijHi) : α × α := if decide (i.val = j.val) then (Numbers.one -
                        yjHi, Numbers.one - yjLo) else ((-yjHi), (-yjLo))
                      (List.finRange n).foldl (fun (acc2 : α × α) (k : Fin n) =>
                        let (acc2Lo, acc2Hi) := acc2
                        let ykLo := match fyLo k with | .scalar v => v
                        let ykHi := match fyHi k with | .scalar v => v
                        let (dikLo, dikHi) : α × α := if decide (i.val = k.val) then (Numbers.one -
                          ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        -- H_ijk = y_i (dij)(dik) - y_i y_j (δ_jk - y_k)
                        let (t1Lo, t1Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi dijLo dijHi
                          mulI aLo aHi dikLo dikHi
                        let (delta_jk_Lo, delta_jk_Hi) : α × α := if decide (j.val = k.val) then
                          (Numbers.one - ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t2Lo, t2Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi yjLo yjHi
                          mulI aLo aHi delta_jk_Lo delta_jk_Hi
                        -- H interval = t1 - t2
                        let hLo := t1Lo - t2Hi
                        let hHi := t1Hi - t2Lo
                        let dzjLo := match fd1Lo j with | .scalar v => v
                        let dzjHi := match fd1Hi j with | .scalar v => v
                        let dzkLo := match fd1Lo k with | .scalar v => v
                        let dzkHi := match fd1Hi k with | .scalar v => v
                        let (prodLo, prodHi) := mulI dzjLo dzjHi dzkLo dzkHi
                        let (termLo, termHi) := mulI hLo hHi prodLo prodHi
                        (acc2Lo + termLo, acc2Hi + termHi)
                      ) (accLo, accHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumLo)
              let part2_hi :=
                Tensor.dim (fun i =>
                  let yiLo := match fyLo i with | .scalar v => v
                  let yiHi := match fyHi i with | .scalar v => v
                  let (_sumLo, sumHi) :=
                    (List.finRange n).foldl (fun (acc : α × α) (j : Fin n) =>
                      let (accLo, accHi) := acc
                      let yjLo := match fyLo j with | .scalar v => v
                      let yjHi := match fyHi j with | .scalar v => v
                      let (dijLo, dijHi) : α × α := if decide (i.val = j.val) then (Numbers.one -
                        yjHi, Numbers.one - yjLo) else ((-yjHi), (-yjLo))
                      (List.finRange n).foldl (fun (acc2 : α × α) (k : Fin n) =>
                        let (acc2Lo, acc2Hi) := acc2
                        let ykLo := match fyLo k with | .scalar v => v
                        let ykHi := match fyHi k with | .scalar v => v
                        let (dikLo, dikHi) : α × α := if decide (i.val = k.val) then (Numbers.one -
                          ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t1Lo, t1Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi dijLo dijHi
                          mulI aLo aHi dikLo dikHi
                        let (delta_jk_Lo, delta_jk_Hi) : α × α := if decide (j.val = k.val) then
                          (Numbers.one - ykHi, Numbers.one - ykLo) else ((-ykHi), (-ykLo))
                        let (t2Lo, t2Hi) :=
                          let (aLo, aHi) := mulI yiLo yiHi yjLo yjHi
                          mulI aLo aHi delta_jk_Lo delta_jk_Hi
                        let hLo := t1Lo - t2Hi
                        let hHi := t1Hi - t2Lo
                        let dzjLo := match fd1Lo j with | .scalar v => v
                        let dzjHi := match fd1Hi j with | .scalar v => v
                        let dzkLo := match fd1Lo k with | .scalar v => v
                        let dzkHi := match fd1Hi k with | .scalar v => v
                        let (prodLo, prodHi) := mulI dzjLo dzjHi dzkLo dzkHi
                        let (termLo, termHi) := mulI hLo hHi prodLo prodHi
                        (acc2Lo + termLo, acc2Hi + termHi)
                      ) (accLo, accHi)
                    ) (Numbers.zero, Numbers.zero)
                  Tensor.scalar sumHi)
              let lo := Tensor.addSpec part1_lo part2_lo
              let hi := Tensor.addSpec part1_hi part2_hi
              d2s.set! id (some { dim := n, lo := lo, hi := hi })
            else d2s
          else d2s
        | _, _, _ => d2s
      | _ => d2s
    | .layernorm _ =>
      -- Conservative d² for layernorm.
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]!, d1[p1]!, d2s[p1]! with
        | some Xin, some dXin, some d2Xin =>
          let n := Xin.dim
          if hn1 : dXin.dim = n then
            if hn2 : d2Xin.dim = n then
              -- Empty vectors have no coordinates to certify; use a nonzero dummy denominator to
              -- avoid evaluating `1 / 0` in the vacuous branch.
              let nDen : Nat := if n = 0 then 1 else n
              let nA : α := (nDen : Nat)
              let muBounds := layerNormMeanBounds (α := α) Xin.lo Xin.hi
              let mu_lo := muBounds.1
              let mu_hi := muBounds.2
              -- u := x - mean(x)
              let uBounds := layerNormCenteredBounds (α := α) Xin.lo Xin.hi mu_lo mu_hi
              let u_lo := uBounds.1
              let u_hi := uBounds.2
              -- s = sqrt(var+eps), bounds
              let var_hi := layerNormVarianceUpper (α := α) Xin.lo Xin.hi mu_lo mu_hi
              let s_lo := MathFunctions.sqrt Numbers.epsilon
              let tBounds := layerNormInvStdBounds (α := α) var_hi
              let t_lo := tBounds.1
              let t_hi := tBounds.2
              -- d2v = d2x - mean(d2x)
              let d2lo := castDimScalar (α:=α) (n:=d2Xin.dim) (n':=n) (h:=hn2) d2Xin.lo
              let d2hi := castDimScalar (α:=α) (n:=d2Xin.dim) (n':=n) (h:=hn2) d2Xin.hi
              let d2muBounds := layerNormMeanBounds (α := α) d2lo d2hi
              let d2vBounds := layerNormCenteredBounds (α := α) d2lo d2hi d2muBounds.1 d2muBounds.2
              let d2v_lo := d2vBounds.1
              let d2v_hi := d2vBounds.2
              -- v = dx - mean(dx)
              let dlo := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn1) dXin.lo
              let dhi := castDimScalar (α:=α) (n:=dXin.dim) (n':=n) (h:=hn1) dXin.hi
              let dmuBounds := layerNormMeanBounds (α := α) dlo dhi
              let vBounds := layerNormCenteredBounds (α := α) dlo dhi dmuBounds.1 dmuBounds.2
              let v_lo := vBounds.1
              let v_hi := vBounds.2
              -- Maximum endpoint magnitude for one scalar interval.
              let abs_max (l u : α) : α :=
                let al := MathFunctions.abs l; let au := MathFunctions.abs u
                if al > au then al else au
              let uLoFn := getDimScalarFn (α:=α) u_lo
              let uHiFn := getDimScalarFn (α:=α) u_hi
              let vLoFn := getDimScalarFn (α:=α) v_lo
              let vHiFn := getDimScalarFn (α:=α) v_hi
              let d2vLoFn := getDimScalarFn (α:=α) d2v_lo
              let d2vHiFn := getDimScalarFn (α:=α) d2v_hi
              -- global scalars for dt, d2t bounds
              let t3_hi :=
                let s3 := s_lo * s_lo * s_lo
                Numbers.one / (if s3 > Numbers.epsilon then s3 else Numbers.epsilon)
              let t5_hi :=
                let s5 := s_lo * s_lo * s_lo * s_lo * s_lo
                Numbers.one / (if s5 > Numbers.epsilon then s5 else Numbers.epsilon)
              let G : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
                match uLoFn i, uHiFn i, vLoFn i, vHiFn i with
                | .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                  let au := abs_max ul uu; let av := abs_max vl vu
                  acc + (au * av)
              ) 0
              let H : α := (List.finRange n).foldl (fun acc (i : Fin n) =>
                match uLoFn i, uHiFn i, vLoFn i, vHiFn i, d2vLoFn i, d2vHiFn i with
                | .scalar ul, .scalar uu, .scalar vl, .scalar vu, .scalar wl, .scalar wu =>
                  let au := abs_max ul uu; let av := abs_max vl vu; let aw := abs_max wl wu
                  acc + ((av * av) + (au * aw))
              ) 0
              let dvar_abs := (Numbers.two * G) / nA
              let d2var_abs := (Numbers.two * H) / nA
              let dt_abs := (Numbers.pointfive * t3_hi) * dvar_abs
              let d2t_abs := (Numbers.pointfive * Numbers.three * t5_hi) * (dvar_abs * dvar_abs) +
                (Numbers.pointfive * t3_hi) * d2var_abs
              -- base = t * d2v
              let base_lo :=
                Tensor.dim (fun i =>
                  match d2vLoFn i, d2vHiFn i with
                  | .scalar l, .scalar u =>
                    let p1 := t_lo * l; let p2 := t_lo * u
                    let p3 := t_hi * l; let p4 := t_hi * u
                    let m1 := if p1 < p2 then p1 else p2
                    let m2 := if p3 < p4 then p3 else p4
                    Tensor.scalar (if m1 < m2 then m1 else m2))
              let base_hi :=
                Tensor.dim (fun i =>
                  match d2vLoFn i, d2vHiFn i with
                  | .scalar l, .scalar u =>
                    let p1 := t_lo * l; let p2 := t_lo * u
                    let p3 := t_hi * l; let p4 := t_hi * u
                    let M1 := if p1 > p2 then p1 else p2
                    let M2 := if p3 > p4 then p3 else p4
                    Tensor.scalar (if M1 > M2 then M1 else M2))
              -- inflate by 2|dt||v_i| + |d2t||u_i|
              let baseLoFn := getDimScalarFn (α:=α) base_lo
              let baseHiFn := getDimScalarFn (α:=α) base_hi
              let lo :=
                Tensor.dim (fun i =>
                  match baseLoFn i, uLoFn i, uHiFn i, vLoFn i, vHiFn i with
                  | .scalar bi, .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                    let au := abs_max ul uu; let av := abs_max vl vu
                    Tensor.scalar (bi - (Numbers.two * dt_abs * av) - (d2t_abs * au)))
              let hi :=
                Tensor.dim (fun i =>
                  match baseHiFn i, uLoFn i, uHiFn i, vLoFn i, vHiFn i with
                  | .scalar bi, .scalar ul, .scalar uu, .scalar vl, .scalar vu =>
                    let au := abs_max ul uu; let av := abs_max vl vu
                    Tensor.scalar (bi + (Numbers.two * dt_abs * av) + (d2t_abs * au)))
              d2s.set! id (some { dim := n, lo := lo, hi := hi })
            else d2s
          else d2s
        | _, _, _ => d2s
      | _ => d2s
    | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean
      .. =>
      d2s
    | .conv2d .. | .batchNorm2dNchwEval .. => d2s
  (List.finRange g.nodes.size).foldl propagate init

end NN.MLTheory.CROWN.Graph
