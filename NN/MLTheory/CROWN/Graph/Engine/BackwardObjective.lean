/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN

/-!
# Objective-Dependent Backward CROWN

Forward CROWN gives nodewise bounds. This module handles the complementary use case: start from a
linear objective on an output node and propagate that objective backward through the graph, choosing
local relaxations from the sign of the downstream coefficients.
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

/-!
The backward pass covers the same verifier dialect as `runCROWN` where objective-dependent
relaxations are available. Unsupported nodes consume already-computed IBP boxes conservatively.
-/

private inductive BackwardDir where
  | lower
  | upper

private structure BackwardState (α : Type) [Context α] where
  coeffs : Array (Option (FlatVec α)) -- per-node objective coefficients
  cst    : α                         -- accumulated constant term

private def flatvecAdd (a b : FlatVec α) : Option (FlatVec α) :=
  if h : a.n = b.n then
    let bv : Tensor α (.dim a.n .scalar) :=
      castDimScalar (α := α) (n := b.n) (n' := a.n) h.symm b.v
    some { n := a.n, v := Tensor.addSpec a.v bv }
  else
    none

private def flatvecScale (k : α) (v : FlatVec α) : FlatVec α :=
  { n := v.n, v := Tensor.scaleSpec v.v k }

private def addCoeff (st : BackwardState α) (pid : Nat) (v : FlatVec α) : BackwardState α :=
  match st.coeffs[pid]! with
  | none => { st with coeffs := st.coeffs.set! pid (some v) }
  | some w =>
    match flatvecAdd (α:=α) w v with
    | some s => { st with coeffs := st.coeffs.set! pid (some s) }
    | none   => st

private def dotFlat {n : Nat} (a b : Tensor α (.dim n .scalar)) : α :=
  Spec.Tensor.sumSpec (Tensor.mulSpec a b)

private def consumeObjectiveFromBox (dir : BackwardDir) (aY : FlatVec α) (B : FlatBox α) : Option α
  :=
  if h : aY.n = B.dim then
    let aYv : Tensor α (.dim B.dim .scalar) :=
      castDimScalar (α := α) (n := aY.n) (n' := B.dim) h aY.v
    let fa := getDimScalarFn (α := α) aYv
    let flo := getDimScalarFn (α := α) B.lo
    let fhi := getDimScalarFn (α := α) B.hi
    let chosenProd : Tensor α (.dim B.dim .scalar) :=
      Tensor.dim (fun i =>
        match fa i, flo i, fhi i with
        | .scalar ay, .scalar l, .scalar u =>
          let y :=
            if decide (ay > Numbers.zero) then
              match dir with
              | .upper => u
              | .lower => l
            else
              match dir with
              | .upper => l
              | .lower => u
          Tensor.scalar (ay * y))
    some (Spec.Tensor.sumSpec chosenProd)
  else
    none

private def diagOfMat {n : Nat} (A : Tensor α (.dim n (.dim n .scalar))) : Tensor α (.dim n
  .scalar) :=
  match A with
  | .dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | .dim cols =>
        match cols i with
        | .scalar v => Tensor.scalar v)

-- Apply a chosen diagonal relaxation y = s ⊙ x + b for a bound on a scalar objective.
private def backwardApplyDiag {n : Nat}
  (dir : BackwardDir)
  (aY : Tensor α (.dim n .scalar))
  (sLo bLo sHi bHi : Tensor α (.dim n .scalar)) :
  (Tensor α (.dim n .scalar) × α) :=
  let fa := getDimScalarFn (α := α) aY
  let fsLo := getDimScalarFn (α := α) sLo
  let fbLo := getDimScalarFn (α := α) bLo
  let fsHi := getDimScalarFn (α := α) sHi
  let fbHi := getDimScalarFn (α := α) bHi
  let sChosen : Tensor α (.dim n .scalar) :=
    Tensor.dim (fun i =>
      match fa i, fsLo i, fsHi i with
      | .scalar ay, .scalar slo, .scalar shi =>
        let s :=
          if decide (ay > Numbers.zero) then
            match dir with
            | .upper => shi
            | .lower => slo
          else
            match dir with
            | .upper => slo
            | .lower => shi
        Tensor.scalar s)
  let bChosen : Tensor α (.dim n .scalar) :=
    Tensor.dim (fun i =>
      match fa i, fbLo i, fbHi i with
      | .scalar ay, .scalar blo, .scalar bhi =>
        let b :=
          if decide (ay > Numbers.zero) then
            match dir with
            | .upper => bhi
            | .lower => blo
          else
            match dir with
            | .upper => blo
            | .lower => bhi
        Tensor.scalar b)
  let aX := Tensor.mulSpec aY sChosen
  let cst := dotFlat (α:=α) aY bChosen
  (aX, cst)

-- Backward step for a unary op with diagonal relaxations
-- (relu/exp/log/sigmoid/tanh/softmax/layernorm).
private def backwardUnaryDiag
  (dir : BackwardDir) (preB : FlatBox α) (localB : FlatAffineBounds α)
  (aY : FlatVec α) : Option (FlatVec α × α) := by
  if h : aY.n = preB.dim then
    let n := preB.dim
    if hIn : localB.inDim = n then
      if hOut : localB.outDim = n then
        let aYv : Tensor α (.dim n .scalar) :=
          castDimScalar (α := α) (n := aY.n) (n' := n) h aY.v
        let loAffN : AffineVec α n n :=
          castAffineIn (α:=α) (n:=localB.inDim) (n':=n) (m:=n) hIn
            (castAffineOut (α:=α) (n:=localB.inDim) (m:=localB.outDim) (m':=n) hOut localB.loAff)
        let hiAffN : AffineVec α n n :=
          castAffineIn (α:=α) (n:=localB.inDim) (n':=n) (m:=n) hIn
            (castAffineOut (α:=α) (n:=localB.inDim) (m:=localB.outDim) (m':=n) hOut localB.hiAff)
        let sLo := diagOfMat (α:=α) (n:=n) loAffN.A
        let bLo := castDimScalar (α:=α) (n:=localB.outDim) (n':=n) hOut localB.loAff.c
        let sHi := diagOfMat (α:=α) (n:=n) hiAffN.A
        let bHi := castDimScalar (α:=α) (n:=localB.outDim) (n':=n) hOut localB.hiAff.c
        let (aX, cst) := backwardApplyDiag (α:=α) (n:=n) dir aYv sLo bLo sHi bHi
        exact some ({ n := n, v := aX }, cst)
      else
        exact none
    else
      exact none
  else
    exact none

private def matLeftMul {m n : Nat}
  (aY : Tensor α (.dim m .scalar)) (W : Tensor α (.dim m (.dim n .scalar))) :
  Tensor α (.dim n .scalar) :=
  match aY, W with
  | .dim aF, .dim rows =>
    Tensor.materialize <|
      Tensor.dim (fun j =>
        let s : α :=
          (List.finRange m).foldl (fun acc i =>
            match aF i, rows i with
            | .scalar ai, .dim cols =>
              match cols j with
              | .scalar wij => acc + ai * wij) Numbers.zero
        Tensor.scalar s)
  | _, _ =>
    Spec.fill (α := α) Numbers.zero (.dim n .scalar)

private def backwardLinear {m n : Nat}
  (aY : FlatVec α) (W : Tensor α (.dim m (.dim n .scalar))) (b : Tensor α (.dim m .scalar)) :
  Option (FlatVec α × α) :=
  if h : aY.n = m then
    let aYv : Tensor α (.dim m .scalar) :=
      castDimScalar (α := α) (n := aY.n) (n' := m) h aY.v
    let aX := matLeftMul (α:=α) (m:=m) (n:=n) aYv W
    let cst := dotFlat (α:=α) aYv b
    some ({ n := n, v := aX }, cst)
  else
    none

private def backwardAdd (aY : FlatVec α) : FlatVec α := aY

private def backwardSubLeft (aY : FlatVec α) : FlatVec α := aY

private def backwardSubRight (aY : FlatVec α) : FlatVec α :=
  flatvecScale (α:=α) (k := (-Numbers.one)) aY

private def backwardConcatSplit
  (aY : FlatVec α) (n1 n2 : Nat) : Option (FlatVec α × FlatVec α) :=
  if h : aY.n = n1 + n2 then
    let aYv : Tensor α (.dim (n1 + n2) .scalar) :=
      castDimScalar (α := α) (n := aY.n) (n' := n1 + n2) h aY.v
    let a1 : Tensor α (.dim n1 .scalar) :=
      Tensor.dim (fun i =>
        Tensor.scalar (getAtOrZero aYv [i.val]))
    let a2 : Tensor α (.dim n2 .scalar) :=
      Tensor.dim (fun i =>
        Tensor.scalar (getAtOrZero aYv [n1 + i.val]))
    some ({ n := n1, v := a1 }, { n := n2, v := a2 })
  else
    none

private def backwardPermuteVec {n : Nat} (perm : Fin n → Fin n) (v : Tensor α (.dim n .scalar)) :
  Tensor α (.dim n .scalar) :=
  match v with
  | .dim f => Tensor.dim (fun i => f (perm i))

private def backwardMatmul
  (dir : BackwardDir)
  (aZ : FlatVec α) (Bx By : FlatBox α)
  (sA sB : Shape) :
  Option ((FlatVec α) × (FlatVec α) × α) :=
  let dims? : Option (Nat × Nat × Nat × Nat) :=
    match sA, sB with
    | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
      if k = k' then
        some (1, m, k, n)
      else
        none
    | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
      if hb : b = b' then
        match hb with
        | rfl =>
          if k = k' then
            some (b, m, k, n)
          else
            none
      else
        none
    | _, _ => none
  match dims? with
  | none => none
  | some (batch, m, k, n) =>
    let dimA := batch * m * k
    let dimB := batch * k * n
    let outDim := batch * m * n
    if aZ.n = outDim ∧ Bx.dim = dimA ∧ By.dim = dimB then
      let (aArr, bArr, cst) : Array α × Array α × α := Id.run do
        let mut aArr : Array α := Array.replicate dimA Numbers.zero
        let mut bArr : Array α := Array.replicate dimB Numbers.zero
        let mut cst : α := Numbers.zero
        let block : Nat := m * n
        let strideA : Nat := m * k
        let strideB : Nat := k * n
        for outIdx in List.range outDim do
          let az : α := getAtOrZero aZ.v [outIdx]
          let bi := outIdx / block
          let rem := outIdx % block
          let i := rem / n
          let j := rem % n
          let baseA := bi * strideA
          let baseB := bi * strideB
          for kk in List.range k do
            let aIdx := baseA + i * k + kk
            let bIdx := baseB + kk * n + j
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive

            -- Upper plane selection.
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let axU := if u1 < u2 then ly else uy
            let ayU := if u1 < u2 then ux else lx
            let bU := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))

            -- Lower plane selection.
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let axL := if l1 > l2 then ly else uy
            let ayL := if l1 > l2 then lx else ux
            let bL := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))

            let useUpper : Bool :=
              if decide (az > Numbers.zero) then
                match dir with
                | .upper => true
                | .lower => false
              else
                match dir with
                | .upper => false
                | .lower => true

            let ax := if useUpper then axU else axL
            let ay := if useUpper then ayU else ayL
            let bb := if useUpper then bU else bL

            aArr := aArr.set! aIdx (aArr[aIdx]! + az * ax)
            bArr := bArr.set! bIdx (bArr[bIdx]! + az * ay)
            cst := cst + az * bb
        return (aArr, bArr, cst)

      let aT : Tensor α (.dim dimA .scalar) :=
        Tensor.dim (fun i => Tensor.scalar (aArr[i.val]!))
      let bT : Tensor α (.dim dimB .scalar) :=
        Tensor.dim (fun i => Tensor.scalar (bArr[i.val]!))
      some ({ n := dimA, v := aT }, { n := dimB, v := bT }, cst)
    else
      none

private def backwardMulElem
  (dir : BackwardDir)
  (aZ : FlatVec α) (Bx By : FlatBox α) :
  Option ((FlatVec α) × (FlatVec α) × α) :=
  if h : aZ.n = Bx.dim ∧ Bx.dim = By.dim then
    let n := Bx.dim
    let hZ : aZ.n = n := h.1
    let aZv : Tensor α (.dim n .scalar) :=
      castDimScalar (α := α) (n := aZ.n) (n' := n) hZ aZ.v
    let xLo := getDimScalarFn (α := α) Bx.lo
    let xHi := getDimScalarFn (α := α) Bx.hi
    let yLo := getDimScalarFn (α := α) (castDimScalar (α:=α) (n:=By.dim) (n':=n) h.2.symm By.lo)
    let yHi := getDimScalarFn (α := α) (castDimScalar (α:=α) (n:=By.dim) (n':=n) h.2.symm By.hi)
    let aF := getDimScalarFn (α := α) aZv
    -- Choose one McCormick plane per element using the interval midpoint.
    let axU : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let ax := if u1 < u2 then ly else uy
          Tensor.scalar ax)
    let ayU : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let ay := if u1 < u2 then ux else lx
          Tensor.scalar ay)
    let bU : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let u1 := ux * my + ly * mx - ux * ly
          let u2 := lx * my + uy * mx - lx * uy
          let b := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
          Tensor.scalar b)
    let axL : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let l1 := lx * my + ly * mx - lx * ly
          let l2 := ux * my + uy * mx - ux * uy
          let ax := if l1 > l2 then ly else uy
          Tensor.scalar ax)
    let ayL : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let l1 := lx * my + ly * mx - lx * ly
          let l2 := ux * my + uy * mx - ux * uy
          let ay := if l1 > l2 then lx else ux
          Tensor.scalar ay)
    let bL : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match xLo i, xHi i, yLo i, yHi i with
        | .scalar lx, .scalar ux, .scalar ly, .scalar uy =>
          let mx := (lx + ux) * Numbers.pointfive
          let my := (ly + uy) * Numbers.pointfive
          let l1 := lx * my + ly * mx - lx * ly
          let l2 := ux * my + uy * mx - ux * uy
          let b := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))
          Tensor.scalar b)
    let axUFn := getDimScalarFn (α := α) axU
    let ayUFn := getDimScalarFn (α := α) ayU
    let bUFn := getDimScalarFn (α := α) bU
    let axLFn := getDimScalarFn (α := α) axL
    let ayLFn := getDimScalarFn (α := α) ayL
    let bLFn := getDimScalarFn (α := α) bL
    let aX : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match aF i, axUFn i, axLFn i with
        | .scalar az, .scalar axu, .scalar axl =>
          let ax :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => axu
              | .lower => axl
            else
              match dir with
              | .upper => axl
              | .lower => axu
          Tensor.scalar (az * ax))
    let aY : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match aF i, ayUFn i, ayLFn i with
        | .scalar az, .scalar ayu, .scalar ayl =>
          let ay :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => ayu
              | .lower => ayl
            else
              match dir with
              | .upper => ayl
              | .lower => ayu
          Tensor.scalar (az * ay))
    let biasProd : Tensor α (.dim n .scalar) :=
      Tensor.dim (fun i =>
        match aF i, bUFn i, bLFn i with
        | .scalar az, .scalar bu, .scalar bl =>
          let b :=
            if decide (az > Numbers.zero) then
              match dir with
              | .upper => bu
              | .lower => bl
            else
              match dir with
              | .upper => bl
              | .lower => bu
          Tensor.scalar (az * b))
    let cst := Spec.Tensor.sumSpec biasProd
    some ({ n := n, v := aX }, { n := n, v := aY }, cst)
  else
    none

private def backwardNode (dir : BackwardDir)
  (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
  (ctx : AffineCtx) (st : BackwardState α) (id : Nat) : BackwardState α :=
  match st.coeffs[id]! with
  | none => st
  | some aY =>
    let node := nodes[id]!
    match node.kind with
    | .input =>
      if node.id = ctx.inputId then
        st
      else
        match ibp[id]! with
        | some Bx =>
          match consumeObjectiveFromBox (α := α) (dir := dir) aY Bx with
          | some cadd => { st with cst := st.cst + cadd }
          | none => st
        | none => st
    | .const _ =>
      match ps.constVals[id]? with
      | some v =>
        if h : aY.n = v.n then
          let aYv : Tensor α (.dim v.n .scalar) :=
            castDimScalar (α := α) (n := aY.n) (n' := v.n) h aY.v
          let add := dotFlat (α:=α) aYv v.v
          { st with cst := st.cst + add }
        else st
      | none => st
    | .detach =>
      match node.parents with
      | p1 :: _ => addCoeff (α := α) st p1 aY
      | _ => st
    | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
        let st1 := addCoeff (α:=α) st p1 (backwardAdd (α:=α) aY)
        addCoeff (α:=α) st1 p2 (backwardAdd (α:=α) aY)
      | _ => st
    | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
        let st1 := addCoeff (α:=α) st p1 (backwardSubLeft (α:=α) aY)
        addCoeff (α:=α) st1 p2 (backwardSubRight (α:=α) aY)
      | _ => st
    | .randUniform _ | .bernoulliMask _ | .abs | .sqrt | .sin | .cos | .permute _ | .maxElem |
      .minElem
    | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad ..
    | .broadcastTo .. | .reduceSum .. | .reduceMean .. =>
      match ibp[id]! with
      | some By =>
        match consumeObjectiveFromBox (α := α) (dir := dir) aY By with
        | some cadd => { st with cst := st.cst + cadd }
        | none => st
      | none => st
    | .batchNorm2dNchwEval .. =>
      match node.parents with
      | p1 :: _ =>
        match ps.batchNorm2dNchwEval[id]? with
        | some cfg =>
          match batchNorm2dNchwEvalLinear? (α := α) nodes[p1]!.outShape cfg with
          | some p =>
            match backwardLinear (α := α) (m := p.m) (n := p.n) aY p.w p.b with
            | some (aX, cadd) =>
              let st' := addCoeff (α := α) st p1 aX
              { st' with cst := st'.cst + cadd }
            | none => st
          | none => st
        | none => st
      | _ => st
    | .linear =>
      match node.parents with
      | p1 :: _ =>
        match ps.linearWB[id]? with
        | some p =>
          match backwardLinear (α:=α) (m:=p.m) (n:=p.n) aY p.w p.b with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            { st' with cst := st'.cst + cadd }
          | none => st
        | none => st
      | _ => st
    | .matmul =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]! with
        | some Bx, some By =>
          match backwardMatmul (α:=α) (dir:=dir) aY Bx By (sA := nodes[p1]!.outShape) (sB :=
            nodes[p2]!.outShape) with
          | some (aX, aY', cadd) =>
            let st1 := addCoeff (α:=α) st p1 aX
            let st2 := addCoeff (α:=α) st1 p2 aY'
            { st2 with cst := st2.cst + cadd }
          | none => st
        | _, _ => st
      | p1 :: _ =>
        match ps.matmulW[id]? with
        | some p =>
          let zb := Spec.fill (α := α) Numbers.zero (.dim p.m .scalar)
          match backwardLinear (α:=α) (m:=p.m) (n:=p.n) aY p.w zb with
          | some (aX, _cadd) =>
            addCoeff (α:=α) st p1 aX
          | none => st
        | none => st
      | _ => st
    | .conv2d .. =>
      match node.parents with
      | p1 :: _ =>
        match ps.conv2dCfg[id]? with
        | some cfg =>
          if _hs : cfg.stride = 0 then
            st
          else
            let outH := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
            let outW := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
            let outDim := cfg.outC * outH * outW
            let convAff := affOfConv2d (α:=α) cfg
            match backwardLinear (α:=α) (m:=outDim) (n:=cfg.inC * cfg.inH * cfg.inW) aY convAff.A
              convAff.c with
            | some (aX, cadd) =>
              let st' := addCoeff (α:=α) st p1 aX
              { st' with cst := st'.cst + cadd }
            | none => st
        | none => st
      | _ => st
    | .relu | .exp | .log | .inv | .sigmoid | .tanh | .softmax _ | .layernorm _ =>
      -- Unary ops: use local diagonal relaxations computed from the parent IBP box.
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]! with
        | some preB =>
          let n := preB.dim
          let idB := boundsIdentity (α:=α) n
          let localB : FlatAffineBounds α :=
            match node.kind with
            | .relu      => propagateReluBounds (α:=α) preB idB rfl
            | .exp       => propagateExpBounds (α:=α) preB idB rfl
            | .log       => propagateLogBounds (α:=α) preB idB rfl
            | .inv       =>
              let invB := boxInv (α := α) preB
              boundsConst (α:=α) n n invB.lo invB.hi
            | .sigmoid   => propagateSigmoidBounds (α:=α) preB idB rfl
            | .tanh      => propagateTanhBounds (α:=α) preB idB rfl
            | .softmax axis =>
              if axis = Shape.rank node.outShape - 1 then
                propagateSoftmaxBoundsLastAxis (α:=α) node.outShape preB idB rfl
              else
                boundsConst (α:=α) n n (Spec.fill (α:=α) Numbers.zero (.dim n .scalar)) (Spec.fill
                  (α:=α) Numbers.one (.dim n .scalar))
            | .layernorm axis =>
              if axis = Shape.rank node.outShape - 1 then
                propagateLayernormBoundsLastAxis (α:=α) node.outShape preB idB rfl
              else
                boundsConst (α:=α) n n preB.lo preB.hi
            | _ => idB
          match backwardUnaryDiag (α:=α) dir preB localB aY with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            { st' with cst := st'.cst + cadd }
          | none => st
        | none => st
      | _ => st
    | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]! with
        | some Bx, some By =>
          match backwardMulElem (α:=α) (dir:=dir) aY Bx By with
          | some (aX, aY', cadd) =>
            let st1 := addCoeff (α:=α) st p1 aX
            let st2 := addCoeff (α:=α) st1 p2 aY'
            { st2 with cst := st2.cst + cadd }
          | none => st
        | _, _ => st
      | _ => st
    | .sum =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]! with
        | some Bx =>
          if aY.n = 1 then
            let a0 : α := getAtOrZero aY.v [0]
            let out : FlatVec α :=
              { n := Bx.dim, v := Spec.fill (α := α) a0 (.dim Bx.dim .scalar) }
            addCoeff (α:=α) st p1 out
          else st
        | none => st
      | _ => st
    | .reshape _ _ | .flatten _ =>
      match node.parents with
      | p1 :: _ => addCoeff (α:=α) st p1 aY
      | _ => st
    | .concat _ =>
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]! with
        | some B1, some B2 =>
          match backwardConcatSplit (α:=α) aY B1.dim B2.dim with
          | some (a1, a2) =>
            let st1 := addCoeff (α:=α) st p1 a1
            addCoeff (α:=α) st1 p2 a2
          | none => st
        | _, _ => st
      | _ => st
    | .swap_first_two =>
      match node.parents with
      | p1 :: _ =>
        match nodes[p1]!.outShape with
        | .dim m (.dim n rest) =>
          let outDim := aY.n
          if h0 : outDim = 0 then
            addCoeff (α:=α) st p1 aY
          else
            haveI : NeZero outDim := ⟨h0⟩
            let restSize := Shape.size rest
            let block := m * restSize
            let perm : Fin outDim → Fin outDim := fun idx =>
              let t := idx.val
              let j := t / block
              let rem := t % block
              let i := rem / restSize
              let k := rem % restSize
              let tIn := i * (n * restSize) + j * restSize + k
              Fin.ofNat outDim tIn
            let aYv : Tensor α (.dim outDim .scalar) :=
              castDimScalar (α := α) (n := aY.n) (n' := outDim) rfl aY.v
            let aXv := backwardPermuteVec (α:=α) (n:=outDim) perm aYv
            addCoeff (α:=α) st p1 { n := outDim, v := aXv }
        | _ => st
      | _ => st
    | .transpose3dLastTwo =>
      match node.parents with
      | p1 :: _ =>
        match nodes[p1]!.outShape with
        | .dim _a (.dim b (.dim c .scalar)) =>
          let outDim := aY.n
          if h0 : outDim = 0 then
            addCoeff (α:=α) st p1 aY
          else
            haveI : NeZero outDim := ⟨h0⟩
            let block := c * b
            let perm : Fin outDim → Fin outDim := fun idx =>
              let t := idx.val
              let i := t / block
              let rem := t % block
              let k := rem / b
              let j := rem % b
              let tIn := i * (b * c) + j * c + k
              Fin.ofNat outDim tIn
            let aYv : Tensor α (.dim outDim .scalar) :=
              castDimScalar (α := α) (n := aY.n) (n' := outDim) rfl aY.v
            let aXv := backwardPermuteVec (α:=α) (n:=outDim) perm aYv
            addCoeff (α:=α) st p1 { n := outDim, v := aXv }
        | _ => st
      | _ => st
    | .mseLoss =>
      -- Treat mse_loss as mean(square(y - t)) using the same square relaxation as in `runCROWN`.
      match node.parents with
      | p1 :: p2 :: _ =>
        match ibp[p1]!, ibp[p2]! with
        | some Y, some T =>
          if hdim : Y.dim = T.dim then
            let n := Y.dim
            if n > 0 then
              if aY.n = 1 then
                let a0 : α := getAtOrZero aY.v [0]
                let nA : α := (n : Nat)
                let scale : α := a0 / nA
                -- Coefficients for each squared term are all `scale`.
                let aSq : Tensor α (.dim n .scalar) := Spec.fill (α := α) scale (.dim n .scalar)
                -- Diff interval box.
                let Thi := castDimScalar (α:=α) (n:=T.dim) (n':=n) hdim.symm T.hi
                let Tlo := castDimScalar (α:=α) (n:=T.dim) (n':=n) hdim.symm T.lo
                let diffLo : Tensor α (.dim n .scalar) := Tensor.subSpec Y.lo Thi
                let diffHi : Tensor α (.dim n .scalar) := Tensor.subSpec Y.hi Tlo
                let flo := getDimScalarFn (α := α) diffLo
                let fhi := getDimScalarFn (α := α) diffHi
                let slopes_hi : Tensor α (.dim n .scalar) :=
                  Tensor.dim (fun i =>
                    match flo i, fhi i with
                    | .scalar l, .scalar u => Tensor.scalar (u + l))
                let bias_hi : Tensor α (.dim n .scalar) :=
                  Tensor.dim (fun i =>
                    match flo i, fhi i with
                    | .scalar l, .scalar u => Tensor.scalar (-(u * l)))
                let slopes_lo : Tensor α (.dim n .scalar) :=
                  Tensor.dim (fun i =>
                    match flo i, fhi i with
                    | .scalar l, .scalar u =>
                      let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                        Numbers.zero
                      Tensor.scalar (Numbers.two * d))
                let bias_lo : Tensor α (.dim n .scalar) :=
                  Tensor.dim (fun i =>
                    match flo i, fhi i with
                    | .scalar l, .scalar u =>
                      let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                        Numbers.zero
                      Tensor.scalar (-(d * d)))
                -- Choose square plane per element using sign of `aSq` and `dir`.
                let (aDiff, cadd) :=
                  backwardApplyDiag (α := α) (n := n) dir aSq slopes_lo bias_lo
                    slopes_hi bias_hi
                let cst' := st.cst + cadd
                let st1 := addCoeff (α := α) { st with cst := cst' } p1 { n := n, v := aDiff }
                let st2 :=
                  addCoeff (α := α) st1 p2
                    (flatvecScale (α := α) (k := (-Numbers.one)) { n := n, v := aDiff })
                st2
              else st
            else st
          else st
        | _, _ => st
      | _ => st

private def backwardNodeWithReluAlpha (dir : BackwardDir)
  (nodes : Array Node) (ps : ParamStore α) (ibp : Array (Option (FlatBox α)))
  (ctx : AffineCtx) (reluAlpha : Array (Option (FlatVec α)))
  (st : BackwardState α) (id : Nat) : BackwardState α :=
  match st.coeffs[id]! with
  | none => st
  | some aY =>
    let node := nodes[id]!
    match node.kind with
    | .relu | .exp | .log | .inv | .sigmoid | .tanh | .softmax _ | .layernorm _ =>
      match node.parents with
      | p1 :: _ =>
        match ibp[p1]! with
        | some preB =>
          let n := preB.dim
          let idB := boundsIdentity (α:=α) n
          let localB : FlatAffineBounds α :=
            match node.kind with
            | .relu =>
              match reluAlpha[id]? with
              | some (some a) =>
                if h : a.n = n then
                  let aT : Tensor α (.dim n .scalar) :=
                    castDimScalar (α:=α) (n:=a.n) (n':=n) h a.v
                  propagateReluBoundsWithAlpha (α:=α) preB idB rfl aT
                else
                  propagateReluBounds (α:=α) preB idB rfl
              | _ =>
                propagateReluBounds (α:=α) preB idB rfl
            | .exp       => propagateExpBounds (α:=α) preB idB rfl
            | .log       => propagateLogBounds (α:=α) preB idB rfl
            | .inv       =>
              let invB := boxInv (α := α) preB
              boundsConst (α:=α) n n invB.lo invB.hi
            | .sigmoid   => propagateSigmoidBounds (α:=α) preB idB rfl
            | .tanh      => propagateTanhBounds (α:=α) preB idB rfl
            | .softmax axis =>
              if axis = Shape.rank node.outShape - 1 then
                propagateSoftmaxBoundsLastAxis (α:=α) node.outShape preB idB rfl
              else
                boundsConst (α:=α) n n (Spec.fill (α:=α) Numbers.zero (.dim n .scalar)) (Spec.fill
                  (α:=α) Numbers.one (.dim n .scalar))
            | .layernorm axis =>
              if axis = Shape.rank node.outShape - 1 then
                propagateLayernormBoundsLastAxis (α:=α) node.outShape preB idB rfl
              else
                boundsConst (α:=α) n n preB.lo preB.hi
            | _ => idB
          match backwardUnaryDiag (α:=α) dir preB localB aY with
          | some (aX, cadd) =>
            let st' := addCoeff (α:=α) st p1 aX
            { st' with cst := st'.cst + cadd }
          | none => st
        | none => st
      | _ => st
    | _ =>
      backwardNode (α:=α) dir nodes ps ibp ctx st id

private def runBackwardObjectiveDir
  (dir : BackwardDir) (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatVec α) :
  Option (AffineVec α ctx.inputDim 1) :=
  if outputId < g.nodes.size then
    let initCoeffs := (Array.replicate g.nodes.size none).set! outputId (some obj)
    let init : BackwardState α := { coeffs := initCoeffs, cst := Numbers.zero }
    let st := (List.finRange g.nodes.size).reverse.foldl (fun acc i =>
      backwardNode (α:=α) dir g.nodes ps ibp ctx acc i) init
    match st.coeffs[ctx.inputId]! with
    | some aIn =>
      if hIn : aIn.n = ctx.inputDim then
        let vIn : Tensor α (.dim ctx.inputDim .scalar) :=
          castDimScalar (α := α) (n := aIn.n) (n' := ctx.inputDim) hIn aIn.v
        let A : Tensor α (.dim 1 (.dim ctx.inputDim .scalar)) := Tensor.dim (fun _ => vIn)
        let c : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar st.cst)
        some { A := A, c := c }
      else
        none
    | none =>
      -- No dependence on the designated input: return a constant affine bound.
      let A : Tensor α (.dim 1 (.dim ctx.inputDim .scalar)) :=
        Spec.fill (α := α) Numbers.zero (.dim 1 (.dim ctx.inputDim .scalar))
      let c : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar st.cst)
      some { A := A, c := c }
  else
    none

private def runBackwardObjectiveDirWithReluAlpha
  (dir : BackwardDir) (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatVec α)
  (reluAlpha : Array (Option (FlatVec α))) :
  Option (AffineVec α ctx.inputDim 1) :=
  if outputId < g.nodes.size then
    let initCoeffs := (Array.replicate g.nodes.size none).set! outputId (some obj)
    let init : BackwardState α := { coeffs := initCoeffs, cst := Numbers.zero }
    let st := (List.finRange g.nodes.size).reverse.foldl (fun acc i =>
      backwardNodeWithReluAlpha (α:=α) dir g.nodes ps ibp ctx reluAlpha acc i) init
    match st.coeffs[ctx.inputId]! with
    | some aIn =>
      if hIn : aIn.n = ctx.inputDim then
        let vIn : Tensor α (.dim ctx.inputDim .scalar) :=
          castDimScalar (α := α) (n := aIn.n) (n' := ctx.inputDim) hIn aIn.v
        let A : Tensor α (.dim 1 (.dim ctx.inputDim .scalar)) := Tensor.dim (fun _ => vIn)
        let c : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar st.cst)
        some { A := A, c := c }
      else
        none
    | none =>
      let A : Tensor α (.dim 1 (.dim ctx.inputDim .scalar)) :=
        Spec.fill (α := α) Numbers.zero (.dim 1 (.dim ctx.inputDim .scalar))
      let c : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar st.cst)
      some { A := A, c := c }
  else
    none

/--
Objective-dependent backward CROWN bound for a scalar objective.

Given a linear objective `objᵀ * output`, this runs a backward pass that propagates the objective
coefficients through the graph, selects the relaxation attached to each node, and returns a pair of
affine bounds on the objective with respect to `ctx.inputId`.

The returned `FlatAffineBounds` always has `outDim = 1` (a scalar objective).
-/
def runCROWNBackwardObjective
  (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatVec α) :
  Option (FlatAffineBounds α) := by
  -- Upper and lower affines for the same scalar objective.
  match runBackwardObjectiveDir (α:=α) .lower g ps ctx ibp outputId obj,
        runBackwardObjectiveDir (α:=α) .upper g ps ctx ibp outputId obj with
  | some loAff, some hiAff =>
    exact some { inDim := ctx.inputDim, outDim := 1, loAff := loAff, hiAff := hiAff }
  | _, _ => exact none

/-- Evaluate already-computed backward-CROWN objective bounds on an input box. -/
def evalBackwardObjectiveBox? (bounds : FlatAffineBounds α) (xB : FlatBox α)
    (inputDim : Nat) : Except String (FlatBox α) := do
  if hIn : bounds.inDim = inputDim then
    if hXB : xB.dim = inputDim then
      if hOut : bounds.outDim = 1 then
        let outB := bounds.evalOnFlatBoxAsDim xB (by simpa [hXB] using hIn.symm) hOut
        pure { dim := 1, lo := outB.lo, hi := outB.hi }
      else
        throw s!"backward CROWN objective dimension mismatch: got {bounds.outDim}, expected 1"
    else
      throw s!"input box dimension mismatch: got {xB.dim}, expected {inputDim}"
  else
    throw s!"backward CROWN input dimension mismatch: got {bounds.inDim}, expected {inputDim}"

/--
Run objective-dependent backward CROWN and evaluate the scalar objective bounds on the input box.

The result is a `FlatBox` of dimension `1`, with `lo[0]` and `hi[0]` bounding
`objᵀ * output` over `xB`.
-/
def backwardObjectiveBox? (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) (xB : FlatBox α)
    (outputId : Nat) (obj : FlatVec α) : Except String (FlatBox α) := do
  let some bounds := runCROWNBackwardObjective (α := α) g ps ctx ibp outputId obj
    | throw "CROWN backward objective failed"
  evalBackwardObjectiveBox? (α := α) bounds xB ctx.inputDim

/--
Backward CROWN objective lower bound with externally-provided ReLU alpha slopes.

This is an integration hook for alpha-CROWN style workflows where ReLU slopes are chosen/optimized outside
TorchLean and then imported as a per-node vector in `reluAlpha`.
-/
def runCROWNBackwardObjectiveLowerWithReluAlpha
  (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
  (ibp : Array (Option (FlatBox α))) (outputId : Nat) (obj : FlatVec α)
  (reluAlpha : Array (Option (FlatVec α))) :
  Option (AffineVec α ctx.inputDim 1) :=
  runBackwardObjectiveDirWithReluAlpha (α:=α) .lower g ps ctx ibp outputId obj reluAlpha

end NN.MLTheory.CROWN.Graph
