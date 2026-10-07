/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Affine

/-!
# α-CROWN transfer step (graph dialect)

This file defines a *pure*, per-node transfer rule for affine bound propagation in the
`NN.MLTheory.CROWN.Graph` dialect, extended with an α-parameter for the ReLU lower relaxation
(α-CROWN).

The step function is shared by:

- the certificate checker (recompute each node from its parents and compare to a claimed bound), and
- soundness theorems of the form: "if local replay is consistent, then the claimed enclosure holds".

This module does **not** implement the outer dual-parameter optimization loop used by α/β-CROWN; it
only defines the local transfer rule for a fixed set of α-parameters.

## References

- CROWN: Zhang et al., *Efficient Neural Network Robustness Certification with General Activation
  Functions*, NeurIPS 2018. (arXiv:1811.00866)
- β-CROWN (and the α/β-CROWN toolchain): Wang et al., *Beta-CROWN: Efficient Bound Propagation with
  Provable Guarantees*, NeurIPS 2021. (arXiv:2103.06624)
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Cert

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
### Affine bounds in the graph dialect

A `FlatAffineBounds α` stores two affine maps (lower/upper) of the *flattened input vector*:
$$
  \ell(x) = A_\ell x + c_\ell,\qquad u(x) = A_u x + c_u.
$$
These are propagated through the graph using local transfer rules.

The certificate checker recomputes these affine maps node-by-node, so the functions below are
written in an executable style (using the repo’s `Tensor` operations), while still being usable
in theorem statements over `ℝ`.
-/

/-! ## α-ReLU lower relaxation -/

/-!
### α-CROWN lower relaxation for ReLU

For a pre-activation scalar $z\in[l,u]$, CROWN/DeepPoly uses:
- an *upper* linear envelope (the usual triangular relaxation), and
- a *lower* linear envelope.

In the unstable crossing case $l < 0 < u$, the lower relaxation can be parameterized by
$\alpha\in[0,1]$ to interpolate between the sound choices $y \ge 0$ and $y \ge z$.

We encode this by using `alphaRelaxLowerScalar` for the lower bound, and using
`Runtime.Ops.ReLU.relaxScalar` / `relaxVector` for the upper bound.
-/

/-- The α-CROWN lower relaxation of a scalar ReLU on `[l, u]` with slope parameter `a`.

A stable-active neuron gets the identity, a stable-inactive one gets zero, and an unstable one gets
the tunable line `y = a·x` through the origin. -/
def alphaRelaxLowerScalar (l u a : α) : NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α :=
  if u > 0 then
    if l > 0 then
      { slope := 1, bias := 0 }
    else
      -- crossing: choose y ≥ α·x (bias 0). Checker/proofs constrain 0 ≤ α ≤ 1.
      { slope := a, bias := 0 }
  else
    { slope := 0, bias := 0 }

/-- Vectorized α-CROWN lower relaxation for ReLU, applied componentwise. -/
def alphaRelaxLowerVec {n : Nat}
    (lo hi : Tensor α [n])
    (αv : Tensor α [n]) : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n] :=
  Tensor.dim fun i =>
    Tensor.scalar <|
      alphaRelaxLowerScalar (α := α) (lo.getScalar i) (hi.getScalar i) (αv.getScalar i)

/-! ## Node step function -/

/-- Safe lookup of the affine bounds recorded for node id `pid`, `none` when out of range. -/
def getAff? (cert : Array (Option (FlatAffineBounds α))) (pid : Nat) :
    Option (FlatAffineBounds α) :=
  cert.getD pid none

/-- Safe lookup of the optional α vector at node id `pid`. -/
def getAlpha? (alpha : Array (Option (FlatTensor α))) (pid : Nat) : Option (FlatTensor α) :=
  alpha.getD pid none

/--
Default α vector used when the certificate omits α values.

This matches TorchLean's default lower relaxation: pick slope `1` when `u > -l`, otherwise `0`.
-/
def defaultAlphaVec {n : Nat} (lo hi : Tensor α [n]) : Tensor α [n] :=
  Tensor.dim fun i =>
    let l := lo.getScalar i
    let u := hi.getScalar i
    Tensor.scalar <| if u > (-l) then 1 else 0

/--
One-node α-CROWN step function for a supported subset of IR ops.

The step returns `none` when required parent bounds or parameters are missing, or when dimensions
mismatch. Callers must validate node ids and the size of the IBP table before calling it; direct
array indexing relies on those invariants.

It is intended to be used for:
- executable per-node certificate checking (recompute node `id` from certificate parents), and
- proof-level soundness theorems about the checker.

## Supported node kinds

This step function handles the verifier core of the IR:
- `.input`, `.const`, `.detach`
- `.linear`, `.matmul` (ParamStore-driven linear operators in the verifier dialect)
- `.relu` (CROWN upper + α-CROWN lower)
- `.sum` (treated as a $1\times n$ linear layer)
- `.reshape`, `.flatten` (shape-only, guarded by dimensional consistency)

Custom computations return `none`, even when an IBP box is supplied: a source-evaluation proof
does not supply a real-enclosure or derivative rule. Other node kinds fall back to a conservative
**constant** affine enclosure derived from the IBP box at the same node id (if present).
The checker remains total over graphs that contain
operators outside this affine-transfer subset; end-to-end theorems account for those nodes through
the soundness assumptions attached to their IBP boxes.
-/
def alphaCrownStepNode?
    (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α)))
    (alpha : Array (Option (FlatTensor α)))
    (cert : Array (Option (FlatAffineBounds α)))
    (ctx : AffineCtx) (id : Nat) : Option (FlatAffineBounds α) :=
  let node := nodes[id]!
  match node.kind with
  | .custom .. => none
  | .input =>
      if id = ctx.inputId then
        some (boundsIdentity (α := α) ctx.inputDim)
      else
        none
  | .const _ =>
      match ps.constVals[id]? with
      | some v => some (boundsConst (α := α) ctx.inputDim v.n v.v v.v)
      | none => none
  | .detach =>
      match NN.IR.unaryParent? node.parents with
      | some p1 => getAff? (α := α) cert p1
      | none => none
  | .linear =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getAff? (α := α) cert p1, ps.linearWB[id]? with
          | some xin, some p =>
              if hout : xin.outDim = p.n then
                let out := Graph.propagateLinearBounds (α := α) (n := p.n) (m := p.m) p.w p.b xin
                  hout
                some out
              else
                none
          | _, _ => none
      | none => none
  | .matmul =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getAff? (α := α) cert p1, ps.matmulW[id]? with
          | some xin, some p =>
              if hout : xin.outDim = p.n then
                let zb : Tensor α [p.m] := Tensor.full (α := α) (.dim p.m
                  .scalar) 0
                let out := Graph.propagateLinearBounds (α := α) (n := p.n) (m := p.m) p.w zb xin
                  hout
                some out
              else none
          | _, _ => none
      | none => none
  | .relu =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getAff? (α := α) cert p1, ibp[p1]! with
          | some xin, some preB =>
              if hout : xin.outDim = preB.dim then
                let xLo := Graph.castAffineOut (α := α) hout xin.loAff
                let xHi := Graph.castAffineOut (α := α) hout xin.hiAff
                let relaxHi :=
                  NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := α) (n := preB.dim) preB.lo
                    preB.hi
                let αt : Tensor α [preB.dim] :=
                  match getAlpha? (α := α) alpha id with
                  | some αv =>
                      if hα : αv.n = preB.dim then
                        castDimScalar (α := α) (n := αv.n) (n' := preB.dim) hα αv.v
                      else
                        defaultAlphaVec (α := α) (n := preB.dim) preB.lo preB.hi
                  | none => defaultAlphaVec (α := α) (n := preB.dim) preB.lo preB.hi
                let relaxLo :=
                  alphaRelaxLowerVec (α := α) (n := preB.dim) preB.lo preB.hi αt
                let loAff :=
                  NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
                    (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                let hiAff :=
                  NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
                    (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi
                some { inDim := xin.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }
              else none
          | _, _ => none
      | none => none
  | .sum =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getAff? (α := α) cert p1 with
          | some xin =>
              -- Treat `sum` as a 1×n linear layer with all-ones weights and zero bias.
              let onesRow : Tensor α [1, xin.outDim] :=
                Tensor.full (α := α) (.dim 1 (.dim xin.outDim .scalar)) 1
              let zb : Tensor α [1] := Tensor.full (α := α) (.dim 1 .scalar) 0
              let out :=
                Graph.propagateLinearBounds (α := α) (n := xin.outDim) (m := 1) onesRow zb xin rfl
              some out
          | none => none
      | none => none
  | .reshape _ _ | .flatten _ =>
      match NN.IR.unaryParent? node.parents with
      | some p1 =>
          match getAff? (α := α) cert p1 with
          | some xin =>
              -- The semantic evaluator for reshape/flatten checks `xin.outDim = node.outShape.size`
              -- before returning a value. Mirror that here to keep transfer soundness provable.
              if hout : xin.outDim = node.outShape.size then
                let loAff := Graph.castAffineOut (α := α) (n := xin.inDim) (m := xin.outDim) (m' :=
                  node.outShape.size) hout xin.loAff
                let hiAff := Graph.castAffineOut (α := α) (n := xin.inDim) (m := xin.outDim) (m' :=
                  node.outShape.size) hout xin.hiAff
                some { inDim := xin.inDim, outDim := node.outShape.size, loAff := loAff, hiAff :=
                  hiAff }
              else
                none
          | none => none
      | none => none
  | .conv _ | .concat _ | .layernorm _ =>
      if crownNodeSemanticsSupported (α := α) nodes ps id then
        match ibp[id]! with
        | some B => some (boundsConst (α := α) ctx.inputDim B.dim B.lo B.hi)
        | none => none
      else
        none
  | _ =>
      -- Conservative fallback: allow a constant affine enclosure derived from IBP (if present).
      match ibp[id]! with
      | some B => some (boundsConst (α := α) ctx.inputDim B.dim B.lo B.hi)
      | none => none

end NN.MLTheory.CROWN.Cert
