/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base
public import NN.MLTheory.CROWN.Graph.Engine.IBP -- shake: keep

/-!
# Affine Propagation

This module contains the one-sided affine pass for the flat graph engine. It builds an upper affine
form for each node with respect to a chosen input node. Linear and structural operations preserve
affine dependence. A nonlinear operation uses the upper endpoint of its checked IBP box as a
constant affine bound unless a directed affine rule is available.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]

open BoundOps

/--
Context for affine (CROWN/DeepPoly) propagation.

Affine bounds are computed with respect to a single designated *input* node, whose flattened
dimension is `inputDim`.
-/
structure AffineCtx where
  /-- Node id treated as the input variable for affine bounds. -/
  inputId  : Nat
  /-- Flattened input dimension. -/
  inputDim : Nat

/-- Identity affine map on a flattened vector of length `n`. -/
@[expose]
def affIdentity (n : Nat) : AffineVec α n n :=
  let A :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then 1 else 0)))
  let c := Tensor.full (α:=α) (.dim n .scalar) 0
  { A := A, c := c }

/-- Pointwise addition of two affine maps with the same input and output dimensions. -/
def affAdd {n m : Nat} (a1 a2 : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.addSpec a1.A a2.A, c := Tensor.addSpec a1.c a2.c }

/-- Pointwise subtraction of two affine maps with the same input and output dimensions. -/
def affSub {n m : Nat} (a1 a2 : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.subSpec a1.A a2.A, c := Tensor.subSpec a1.c a2.c }

-- Affine helpers for linear/matmul are handled by the explicit transfer rules below.

/-- The exact affine form of a linear layer: its own weight matrix and bias. -/
private def affOfLinear (p : LinParams α) : AffineVec α p.n p.m :=
  AffineVec.ofLinear (α:=α) (inDim:=p.n) (outDim:=p.m) p.w p.b

/-- The exact affine form of a bare matmul, which is a linear layer with a zero bias. -/
private def affOfMatmul (p : MatParams α) : AffineVec α p.n p.m :=
  let zb := Tensor.full (α:=α) (.dim p.m .scalar) 0
  AffineVec.ofLinear (α:=α) (inDim:=p.n) (outDim:=p.m) p.w zb

/-- Regard the upper endpoint of a checked box as a constant upper affine bound. -/
private def upperConstAffine (inputDim : Nat) (B : FlatBox α) : FlatAffine α :=
  { inDim := inputDim
    outDim := B.dim
    aff :=
      { A := Tensor.full (α := α) (.dim B.dim (.dim inputDim .scalar)) 0
        c := B.hi } }

/--
Flatten a typed convolution into the affine map it denotes.

The CROWN pass uses this when a convolution is linear in the selected input. Keeping the conversion
here lets convolution share the same affine machinery as linear and matmul nodes.

-/
def affOfConv (config : NN.IR.ConvParams α) :
    let inShape := Shape.ofList (config.inChannels :: Tensor.to config.inputSpatial (List Nat))
    let outSpatial :=
      Spec.convOutSpatial config.inputSpatial config.kernel config.stride config.padding
    let outShape := Shape.ofList (config.outChannels :: Tensor.to outSpatial (List Nat))
    AffineVec α inShape.size outShape.size :=
  let inShape := Shape.ofList (config.inChannels :: Tensor.to config.inputSpatial (List Nat))
  let outSpatial :=
    Spec.convOutSpatial config.inputSpatial config.kernel config.stride config.padding
  let outShape := Shape.ofList (config.outChannels :: Tensor.to outSpatial (List Nat))
  let W :=
    NN.MLTheory.CROWN.convLinearMatrix (α := α) (inSpatial := config.inputSpatial) config.spec
  let b := NN.MLTheory.CROWN.convBiasBroadcast (α := α) (outSpatial := outSpatial) config.spec.bias
  AffineVec.ofLinear (α:=α)
    (inDim := inShape.size)
    (outDim := outShape.size)
    W b

/--
Propagate a single node’s affine form (CROWN/DeepPoly style) given parent affine forms.

This updates the `affs` array at index `id` when the node kind admits an affine transfer rule.
For non-affine nodes (or missing parents/params), the array is left unchanged so downstream code
can fall back to IBP boxes.
-/
def propagateAffineNode
  (nodes : Array Node) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α)))
  (affs : Array (Option (FlatAffine α)))
  (ctx : AffineCtx) (id : Nat) : Array (Option (FlatAffine α)) :=
  let node := nodes[id]!
  let getAff (pid : Nat) := (affs[pid]!)
  match node.kind with
  | .input =>
    if node.id = ctx.inputId then
      let aff := affIdentity (α:=α) ctx.inputDim
      affs.set! id (some { inDim := ctx.inputDim, outDim := ctx.inputDim, aff := aff })
    else affs
  | .const _ =>
    -- Lift constant to an affine with zero A and constant c; use ctx.inputDim for input width
    match ps.constVals[id]? with
    | some v =>
      let zA := Tensor.full (α:=α) (.dim v.n (.dim ctx.inputDim .scalar)) 0
      let aff : AffineVec α ctx.inputDim v.n := { A := zA, c := v.v }
      affs.set! id (some { inDim := ctx.inputDim, outDim := v.n, aff := aff })
    | none => affs
  | .detach =>
    match node.parents with
    | #[p1] =>
      match getAff p1 with
      | some a => affs.set! id (some a)
      | none => affs
    | _ => affs
  | .randUniform _ | .bernoulliMask _ =>
    -- Stochastic nodes are treated as non-affine; downstream passes can fall back to IBP boxes.
    affs
  | .maxPool .. | .avgPool .. =>
    -- Pooling is non-affine; downstream passes can fall back to IBP boxes.
    affs
  | .add =>
    match node.parents with
    | #[p1, p2] =>
      match getAff p1, getAff p2 with
      | some a1, some a2 =>
        if hout : a1.outDim = a2.outDim then
          if hin : a1.inDim = a2.inDim then
            let a2' := castAffineIn (α:=α) (n:=a2.inDim) (n':=a1.inDim) (m:=a2.outDim) hin.symm
              a2.aff
            let a2'' := castAffineOut (α:=α) (n:=a1.inDim) (m:=a2.outDim) (m':=a1.outDim) hout.symm
              a2'
            let outAff := affAdd (α:=α) (n:=a1.inDim) (m:=a1.outDim) a1.aff a2''
            affs.set! id (some { inDim := a1.inDim, outDim := a1.outDim, aff := outAff })
          else affs
        else affs
      | _, _ => affs
    | _ => affs
  | .sub =>
    match node.parents with
    | #[p1, p2] =>
      match getAff p1, getAff p2 with
      | some a1, some a2 =>
        if hout : a1.outDim = a2.outDim then
          if hin : a1.inDim = a2.inDim then
            let a2' := castAffineIn (α:=α) (n:=a2.inDim) (n':=a1.inDim) (m:=a2.outDim) hin.symm
              a2.aff
            let a2'' := castAffineOut (α:=α) (n:=a1.inDim) (m:=a2.outDim) (m':=a1.outDim) hout.symm
              a2'
            let outAff := affSub (α:=α) (n:=a1.inDim) (m:=a1.outDim) a1.aff a2''
            affs.set! id (some { inDim := a1.inDim, outDim := a1.outDim, aff := outAff })
          else affs
        else affs
      | _, _ => affs
    | _ => affs
  | .relu =>
    match ibp[id]! with
    | some B => affs.set! id (some (upperConstAffine (α := α) ctx.inputDim B))
    | none => affs
  | .linear =>
    match node.parents with
    | #[p1] =>
      match getAff p1, ps.linearWB[id]? with
      | some paff, some p =>
        if hdim : paff.outDim = p.n then
          let wbaff0 := affOfLinear (α:=α) p
          let wbaff  := castAffineIn (α:=α) (n:=p.n) (n':=paff.outDim) (m:=p.m) hdim.symm wbaff0
          let composed := AffineVec.compose (α:=α) (n:=paff.inDim) (h:=paff.outDim) (m:=p.m) wbaff
            paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
        else
          affs
      | _, _ => affs
    | _ => affs
  | .matmul =>
    match node.parents with
    | #[p1] =>
      match getAff p1, ps.matmulW[id]? with
      | some paff, some p =>
        if hdim : paff.outDim = p.n then
          let waff0 := affOfMatmul (α:=α) p
          let waff  := castAffineIn (α:=α) (n:=p.n) (n':=paff.outDim) (m:=p.m) hdim.symm waff0
          let composed := AffineVec.compose (α:=α) (n:=paff.inDim) (h:=paff.outDim) (m:=p.m) waff
            paff.aff
          affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
        else
          affs
      | _, _ => affs
    | _ => affs
  | .sum =>
    match node.parents with
    | #[p1] =>
      match getAff p1 with
      | some paff =>
        let onesRow : Tensor α [1, paff.outDim] :=
          Tensor.full (α := α) (.dim 1 (.dim paff.outDim .scalar)) 1
        let outAff : AffineVec α paff.inDim 1 :=
          { A := Spec.matMulSpec onesRow paff.aff.A
            c := Spec.matVecMulSpec onesRow paff.aff.c }
        affs.set! id (some { inDim := paff.inDim, outDim := 1, aff := outAff })
      | none => affs
    | _ => affs
  | .reshape _ _ => affs
  | .flatten _ => affs
  | .transpose .. => affs
  | .permute _ => affs
  | .mseLoss => affs
  | .mulElem =>
    match node.parents with
    | #[p1, p2] =>
      match getAff p1, getAff p2, ibp[p1]!, ibp[p2]! with
      | some ax, some ay, some Bx, some By =>
        -- Require matching output dims and input dims; otherwise skip
        if hout : ax.outDim = ay.outDim then
          if hin : ax.inDim = ay.inDim then
            if hbx : Bx.dim = ax.outDim then
              if hby : By.dim = ay.outDim then
                let ayOut := castAffineOut (α:=α) (n:=ay.inDim) (m:=ay.outDim) (m':=ax.outDim)
                  (h:=hout.symm) ay.aff
                let ayAligned := castAffineIn (α:=α) (n:=ay.inDim) (n':=ax.inDim) (m:=ax.outDim)
                  (h:=hin.symm) ayOut
                let bxBox := castBoxDim (α:=α) (n:=Bx.dim) (n':=ax.outDim) (h:=hbx) (ofFlatBox Bx)
                -- align By box dim to ax.outDim via ay.outDim using hout
                let hby2 : By.dim = ax.outDim := Eq.trans hby hout.symm
                let byBox := castBoxDim (α:=α) (n:=By.dim) (n':=ax.outDim) (h:=hby2) (ofFlatBox By)
                -- McCormick upper affine envelope per component i
                let A' :=
                  Tensor.dim (fun i =>
                    let lx := bxBox.lo.getScalar i
                    let ux := bxBox.hi.getScalar i
                    let ly := byBox.lo.getScalar i
                    let uy := byBox.hi.getScalar i
                    let cx := (lx + ux) * (1 / 2)
                    let cy := (ly + uy) * (1 / 2)
                    let u1_center := ux * cy + ly * cx - ux * ly
                    let u2_center := lx * cy + uy * cx - lx * uy
                    let sX := if u1_center < u2_center then ly else uy
                    let sY := if u1_center < u2_center then ux else lx
                    Tensor.dim (fun j =>
                      Tensor.scalar (sX * get2 ax.aff.A i j + sY * get2 ayAligned.A i j)))
                let c' :=
                  Tensor.dim (fun i =>
                    let lx := bxBox.lo.getScalar i
                    let ux := bxBox.hi.getScalar i
                    let ly := byBox.lo.getScalar i
                    let uy := byBox.hi.getScalar i
                    let cx := (lx + ux) * (1 / 2)
                    let cy := (ly + uy) * (1 / 2)
                    let u1_center := ux * cy + ly * cx - ux * ly
                    let u2_center := lx * cy + uy * cx - lx * uy
                    let sX := if u1_center < u2_center then ly else uy
                    let sY := if u1_center < u2_center then ux else lx
                    let off := if u1_center < u2_center then (-(ux * ly)) else (-(lx * uy))
                    Tensor.scalar
                      (sX * ax.aff.c.getScalar i + sY * ayAligned.c.getScalar i + off))
                let outAff : AffineVec α ax.inDim ax.outDim := { A := A', c := c' }
                affs.set! id (some { inDim := ax.inDim, outDim := ax.outDim, aff := outAff })
              else affs
            else affs
          else affs
        else affs
      | _, _, _, _ => affs
    | _ => affs
  | .conv .. =>
    if !crownNodeSemanticsSupported (α := α) nodes ps id then
      affs
    else
      match node.parents with
      | #[p1] =>
        match getAff p1, ps.convCfg[id]? with
        | some paff, some config =>
          let inShape :=
            Shape.ofList (config.inChannels :: Tensor.to config.inputSpatial (List Nat))
          let outSpatial :=
            Spec.convOutSpatial config.inputSpatial config.kernel config.stride config.padding
          let outShape := Shape.ofList (config.outChannels :: Tensor.to outSpatial (List Nat))
          let convIn := inShape.size
          if hdim : paff.outDim = convIn then
            let convAff0 := affOfConv (α:=α) config
            let convAff := castAffineIn (α:=α)
              (n:=convIn) (n':=paff.outDim) (m:=outShape.size)
              hdim.symm convAff0
            let composed := AffineVec.compose (α:=α)
              (n:=paff.inDim) (h:=paff.outDim) (m:=outShape.size)
              convAff paff.aff
            affs.set! id (some { inDim := paff.inDim, outDim := outShape.size, aff :=
              composed })
          else affs
        | _, _ => affs
      | _ => affs
  | .batchNormEval channelAxis _ =>
    match node.parents with
    | #[p1] =>
      match getAff p1, ps.batchNormEval[id]? with
      | some paff, some config =>
        match batchNormEvalLinear? (α := α) nodes[p1]!.outShape channelAxis config with
        | some p =>
          if hdim : paff.outDim = p.n then
            let bnAff0 := affOfLinear (α := α) p
            let bnAff := castAffineIn (α := α) (n := p.n) (n' := paff.outDim) (m := p.m)
              hdim.symm bnAff0
            let composed := AffineVec.compose (α := α)
              (n := paff.inDim) (h := paff.outDim) (m := p.m) bnAff paff.aff
            affs.set! id (some { inDim := paff.inDim, outDim := p.m, aff := composed })
          else
            affs
        | none => affs
      | _, _ => affs
    | _ => affs
  | .exp | .log | .softplus | .safeLog | .softmax _ | .hardMaskedSoftmax _ =>
    match ibp[id]! with
    | some B => affs.set! id (some (upperConstAffine (α := α) ctx.inputDim B))
    | none => affs
  | .layernorm _ =>
    if !crownNodeSemanticsSupported (α := α) nodes ps id then
      affs
    else
      match ibp[id]! with
      | some B => affs.set! id (some (upperConstAffine (α := α) ctx.inputDim B))
      | none => affs
  | .concat axis =>
    -- Concatenation on axis zero stacks the flattened output rows.
    -- For other axes/shapes, this requires stride-aware flatten/reshape bookkeeping.
    if axis != 0 then affs
    else
      let collect (parents : Array Nat) : Option (Array (FlatAffine α)) := do
        let mut result := #[]
        for parent in parents do
          let affine <- getAff parent
          result := result.push affine
        pure result
      match collect node.parents with
      | none => affs
      | some parentsAff =>
        match parentsAff[0]? with
        | none => affs
        | some first =>
          let inDim := first.inDim
          if parentsAff.all (fun a => a.inDim == inDim) then
            let totalOut := parentsAff.foldl (fun acc a => acc + a.outDim) 0
            if Spec.Shape.size node.outShape = totalOut then
              let pick (k : Nat) : FlatAffine α × Nat :=
                let result := parentsAff.foldl
                  (fun (state : Option (FlatAffine α × Nat) × Nat) a =>
                    match state with
                    | (some selected, remaining) => (some selected, remaining)
                    | (none, remaining) =>
                      if remaining < a.outDim then
                        (some (a, remaining), 0)
                      else
                        (none, remaining - a.outDim))
                  (none, k)
                result.1.getD (first, 0)
              let A' : Tensor α [totalOut, inDim] :=
                Tensor.dim (fun i =>
                  let (a, k) := pick i.val
                  Tensor.dim (fun j => Tensor.scalar (getAtOrZero a.aff.A [k, j.val])))
              let c' : Tensor α [totalOut] :=
                Tensor.dim (fun i =>
                  let (a, k) := pick i.val
                  Tensor.scalar (getAtOrZero a.aff.c [k]))
              let outAff : AffineVec α inDim totalOut := { A := A', c := c' }
              affs.set! id (some { inDim := inDim, outDim := totalOut, aff := outAff })
            else affs
          else affs
  | .abs | .sqrt | .inv | .maxElem | .minElem | .broadcastTo .. | .reduceSum .. | .reduceMean ..
  | .tanh | .sin | .cos | .sigmoid =>
    match ibp[id]! with
    | some B => affs.set! id (some (upperConstAffine (α := α) ctx.inputDim B))
    | none => affs

/--
Run the one-sided affine pass.

Linear nodes keep their affine dependence on the selected input. Nonlinear nodes use their checked
IBP upper endpoint as a constant affine bound unless this pass has a separately justified rule.
-/
def runAffine (g : Graph) (ps : ParamStore α) (ctx : AffineCtx) (ibp : Array (Option (FlatBox α))) :
  Array (Option (FlatAffine α)) :=
  let init := Array.replicate g.nodes.size none
  if crownGraphSemanticsSupported (α := α) g ps then
    (List.finRange g.nodes.size).foldl (fun acc i =>
      propagateAffineNode (α:=α) g.nodes ps ibp acc ctx i) init
  else
    init

end NN.MLTheory.CROWN.Graph
