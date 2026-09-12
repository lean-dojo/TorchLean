/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Structural
public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Linear -- shake: keep

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]

open BoundOps

/-!
# CROWN Node Transfer

Dispatch from graph operations to their affine transfer rules.
-/

/--
Propagate a single node’s *affine bounds* (lower/upper) given parent bounds.

This is the CROWN/DeepPoly-style transfer step used by `runCROWN`. For node kinds without a
dedicated rule, we fall back to the IBP enclosure (turned into a constant affine bound).
-/
def propagateCROWNNode
  (nodes : Array Node) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α)))
  (bounds : Array (Option (FlatAffineBounds α)))
  (ctx : AffineCtx) (id : Nat) : Array (Option (FlatAffineBounds α)) :=
  let node := nodes[id]!
  let getB (pid : Nat) := (bounds[pid]!)
  match node.kind with
  | .input =>
    if node.id = ctx.inputId then
      bounds.set! id (some (boundsIdentity (α:=α) ctx.inputDim))
    else bounds
  | .const _ =>
    match ps.constVals[id]? with
    | some v =>
      -- Exact constant bounds.
      bounds.set! id (some (boundsConst (α:=α) ctx.inputDim v.n v.v v.v))
    | none => bounds
  | .detach =>
    match node.parents with
    | #[p1] =>
      match getB p1 with
      | some b => bounds.set! id (some b)
      | none => bounds
    | _ => bounds
  | .randUniform _ | .bernoulliMask _ | .abs | .sqrt | .maxElem | .minElem | .sin |
    .cos | .hardMaskedSoftmax _
  | .maxPool .. | .avgPool ..
  | .broadcastTo .. | .reduceSum .. | .reduceMean .. =>
    -- Conservative fallback: use IBP box as a constant affine bound (A = 0).
    match ibp[id]! with
    | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
    | none => bounds
  | .add =>
    match node.parents with
    | #[p1, p2] =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hout : b1.outDim = b2.outDim then
          if hin : b1.inDim = b2.inDim then
            let b1Lo : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.loAff)
            let b1Hi : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.hiAff)
            let out : FlatAffineBounds α :=
              { inDim := b2.inDim
                outDim := b2.outDim
                loAff := affAdd (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Lo b2.loAff
                hiAff := affAdd (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Hi b2.hiAff }
            bounds.set! id (some out)
          else bounds
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .sub =>
    match node.parents with
    | #[p1, p2] =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hout : b1.outDim = b2.outDim then
          if hin : b1.inDim = b2.inDim then
            let b1Lo : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.loAff)
            let b1Hi : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.hiAff)
            let out : FlatAffineBounds α :=
              { inDim := b2.inDim
                outDim := b2.outDim
                loAff := affSub (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Lo b2.hiAff
                hiAff := affSub (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Hi b2.loAff }
            bounds.set! id (some out)
          else bounds
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .linear =>
    match node.parents with
    | #[p1] =>
      match getB p1, ps.linearWB[id]? with
      | some xin, some p =>
        if hout : xin.outDim = p.n then
          let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w p.b xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .matmul =>
    match node.parents with
    | #[p1, p2] =>
      -- General (batched) matmul: use McCormick relaxations per product term.
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some aAff, some bAff, some aBox, some bBox =>
        match Internal.propagateMatmulBounds (α:=α)
          (sA := nodes[p1]!.outShape) (sB := nodes[p2]!.outShape)
              aBox bBox aAff bAff with
        | some out =>
          bounds.set! id (some out)
        | none =>
          match ibp[id]! with
          | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo
            Bout.hi))
          | none => bounds
      | _, _, _, _ => bounds
    | #[p1] =>
      match getB p1, ps.matmulW[id]? with
      | some xin, some p =>
        if hout : xin.outDim = p.n then
          let zb := Tensor.full (α:=α) (.dim p.m .scalar) 0
          let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w zb xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .relu =>
    -- Computing the crossing-zero ReLU slope involves division. Until an affine-rounding
    -- capability supplies directed coefficients, retain the checked IBP enclosure.
    match ibp[id]! with
    | some B => bounds.set! id (some (boundsConst (α := α) ctx.inputDim B.dim B.lo B.hi))
    | none => bounds
  | .exp | .log | .inv | .sigmoid | .tanh | .softplus | .safeLog =>
    -- Executable nonlinear bounds come from the directed IBP pass. Turning that box into a
    -- constant affine form is less precise than an analytic relaxation, but it does not recompute
    -- transcendental values with unqualified host arithmetic. Ideal-real relaxation formulas
    -- remain available as standalone helpers in `CROWN.Activations`.
    match ibp[id]! with
    | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo Bout.hi))
    | none => bounds
  | .mulElem =>
    match node.parents with
    | #[p1, p2] =>
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some xB, some yB, some Bx, some By =>
        if hxo : xB.outDim = Bx.dim then
          if hyo : yB.outDim = By.dim then
            match propagateMulElemBounds (α:=α) Bx By xB yB hxo hyo with
            | some out => bounds.set! id (some out)
            | none =>
              match ibp[id]! with
              | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo
                Bout.hi))
              | none => bounds
          else bounds
        else bounds
      | _, _, _, _ => bounds
    | _ => bounds
  | .sum =>
    match node.parents with
    | #[p1] =>
      match getB p1 with
      | some xin =>
        let onesRow : Tensor α [1, xin.outDim] :=
          Tensor.full (α := α) (.dim 1 (.dim xin.outDim .scalar)) 1
        let loAff : AffineVec α xin.inDim 1 :=
          { A := Spec.matMulSpec onesRow xin.loAff.A
            c := Spec.matVecMulSpec onesRow xin.loAff.c }
        let hiAff : AffineVec α xin.inDim 1 :=
          { A := Spec.matMulSpec onesRow xin.hiAff.A
            c := Spec.matVecMulSpec onesRow xin.hiAff.c }
        bounds.set! id (some { inDim := xin.inDim, outDim := 1, loAff := loAff, hiAff := hiAff })
      | none => bounds
    | _ => bounds
  | .reshape _ _ =>
    -- Flattened representation preserves order; treat as identity.
    match node.parents with
    | #[p1] =>
      match getB p1 with
      | some xin => bounds.set! id (some xin)
      | none => bounds
    | _ => bounds
  | .flatten _ =>
    match node.parents with
    | #[p1] =>
      match getB p1 with
      | some xin => bounds.set! id (some xin)
      | none => bounds
    | _ => bounds
  | .concat axis =>
    if axis != 0 then
      bounds
    else
      -- Leading-axis concatenation is contiguous in row-major flattened storage.
      match node.parents with
      | #[p1, p2] =>
        match getB p1, getB p2 with
        | some b1, some b2 =>
          if hin : b1.inDim = b2.inDim then
            let b2Lo : AffineVec α b1.inDim b2.outDim :=
              castAffineIn (α := α) (n := b2.inDim) (n' := b1.inDim) (m := b2.outDim) hin.symm
                b2.loAff
            let b2Hi : AffineVec α b1.inDim b2.outDim :=
              castAffineIn (α := α) (n := b2.inDim) (n' := b1.inDim) (m := b2.outDim) hin.symm
                b2.hiAff
            let outDim := b1.outDim + b2.outDim
            let ALo : Tensor α [outDim, b1.inDim] :=
              Tensor.matrix fun i j =>
                Fin.addCases
                  (fun i1 => Spec.get2 b1.loAff.A i1 j)
                  (fun i2 => Spec.get2 b2Lo.A i2 j)
                  i
            let AHi : Tensor α [outDim, b1.inDim] :=
              Tensor.matrix fun i j =>
                Fin.addCases
                  (fun i1 => Spec.get2 b1.hiAff.A i1 j)
                  (fun i2 => Spec.get2 b2Hi.A i2 j)
                  i
            let cLo : Tensor α [outDim] :=
              Tensor.ofFn fun i =>
                Fin.addCases
                  (fun i1 => Tensor.getScalar b1.loAff.c i1)
                  (fun i2 => Tensor.getScalar b2Lo.c i2)
                  i
            let cHi : Tensor α [outDim] :=
              Tensor.ofFn fun i =>
                Fin.addCases
                  (fun i1 => Tensor.getScalar b1.hiAff.c i1)
                  (fun i2 => Tensor.getScalar b2Hi.c i2)
                  i
            bounds.set! id
              (some
                { inDim := b1.inDim
                  outDim := outDim
                  loAff := { A := ALo, c := cLo }
                  hiAff := { A := AHi, c := cHi } })
          else bounds
        | _, _ => bounds
      | _ => bounds
  | .transpose axis₁ axis₂ =>
    match node.parents with
    | #[p1] =>
      match getB p1 with
      | some xin =>
        if xin.outDim = 0 then
          bounds.set! id (some xin)
        else
          match OpContracts.transposePerm nodes[p1]!.outShape.rank axis₁ axis₂ with
          | .ok perm =>
            match permuteFlatAffineBounds? (α := α) nodes[p1]!.outShape perm xin with
            | some result => bounds.set! id (some result)
            | none => bounds
          | .error _ => bounds
      | none => bounds
    | _ => bounds
  | .permute perm =>
    match node.parents with
    | #[p1] =>
      match getB p1 with
      | some xin =>
        if xin.outDim = 0 then
          bounds.set! id (some xin)
        else
          match permuteFlatAffineBounds? (α := α) nodes[p1]!.outShape perm xin with
          | some result => bounds.set! id (some result)
          | none => bounds
      | none => bounds
    | _ => bounds
  | .layernorm _ =>
    if !crownNodeSemanticsSupported (α := α) nodes ps id then
      bounds
    else
      match ibp[id]! with
      | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
      | none => bounds
  | .softmax _ =>
    match ibp[id]! with
    | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
    | none => bounds
  | .mseLoss =>
    match ibp[id]! with
    | some B => bounds.set! id (some (boundsConst (α := α) ctx.inputDim B.dim B.lo B.hi))
    | none => bounds
  | .conv .. =>
    if !crownNodeSemanticsSupported (α := α) nodes ps id then
      bounds
    else
      match node.parents with
      | #[p1] =>
        match getB p1 with
        | some xin =>
          match ps.convCfg[id]? with
          | some config =>
            let inShape :=
              Shape.ofList (config.inChannels :: Tensor.to config.inputSpatial (List Nat))
            let outSpatial :=
              Spec.convOutSpatial config.inputSpatial config.kernel config.stride config.padding
            let outShape := Shape.ofList (config.outChannels :: Tensor.to outSpatial (List Nat))
            if hout : xin.outDim = inShape.size then
              let convAff := affOfConv (α:=α) config
              let out := propagateLinearBounds (α:=α) (n:=inShape.size) (m:=outShape.size)
                convAff.A convAff.c xin hout
              bounds.set! id (some out)
            else bounds
          | none => bounds
        | none => bounds
      | _ => bounds
  | .batchNormEval channelAxis _ =>
    match node.parents with
    | #[p1] =>
      match getB p1, ps.batchNormEval[id]? with
      | some xin, some config =>
        match batchNormEvalLinear? (α := α) nodes[p1]!.outShape channelAxis config with
        | some p =>
          if hout : xin.outDim = p.n then
            let out := propagateLinearBounds (α := α) (n := p.n) (m := p.m) p.w p.b xin hout
            bounds.set! id (some out)
          else
            bounds
        | none => bounds
      | _, _ => bounds
    | _ => bounds


end NN.MLTheory.CROWN.Graph
