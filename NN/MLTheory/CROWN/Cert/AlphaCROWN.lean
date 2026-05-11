/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Propagation.LinearSignsplit

/-!
# α-CROWN transfer step (graph dialect)

This file defines a *pure*, per-node transfer rule for affine bound propagation in the
`NN.MLTheory.CROWN.Graph` dialect, extended with an α-parameter for the ReLU lower relaxation
(α-CROWN).

The step function is shared by:

- the certificate checker (recompute each node from its parents and compare to a claimed bound), and
- soundness theorems of the form: "if the checker accepts, then the claimed enclosure holds".

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

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

variable {α : Type} [Context α]

/-! ## Small affine helpers (non-private) -/

/-!
### Affine bounds in the graph dialect

A `FlatAffineBounds α` stores two affine maps (lower/upper) of the *flattened input vector*:
\[
  \ell(x) = A_\ell x + c_\ell,\qquad u(x) = A_u x + c_u.
\]
These are propagated through the graph using local transfer rules.

The certificate checker recomputes these affine maps node-by-node, so the functions below are
written in an executable style (using the repo’s `Tensor` operations), while still being usable
in theorem statements over `ℝ`.
-/

/-- Cast the **input** dimension of an `AffineVec` along an equality. -/
def castAffineIn {n n' m : Nat} (h : n = n') (a : AffineVec α n m) : AffineVec α n' m := by
  simpa [h] using a

/-- Cast the **output** dimension of an `AffineVec` along an equality. -/
def castAffineOut {n m m' : Nat} (h : m = m') (a : AffineVec α n m) : AffineVec α n m' := by
  simpa [h] using a

/-- The identity affine form `x ↦ x` (as `A = I`, `c = 0`). -/
def affIdentity (n : Nat) : AffineVec α n n :=
  let A : Tensor α (.dim n (.dim n .scalar)) :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j => Tensor.scalar (if i = j then 1 else 0)))
  let c := Spec.fill (α := α) 0 (.dim n .scalar)
  { A := A, c := c }

/-- Identity affine bounds: both lower and upper are `aff_identity`. -/
def boundsIdentity (inDim : Nat) : FlatAffineBounds α :=
  { inDim := inDim, outDim := inDim, loAff := affIdentity (α := α) inDim, hiAff := affIdentity (α
    := α) inDim }

/--
Constant affine bounds for a node output.

Both maps have `A = 0`; the offsets are the provided endpoint vectors.
-/
def boundsConst (inDim outDim : Nat) (lo hi : Tensor α (.dim outDim .scalar)) : FlatAffineBounds α
  :=
  let zA : Tensor α (.dim outDim (.dim inDim .scalar)) :=
    Spec.fill (α := α) 0 (.dim outDim (.dim inDim .scalar))
  { inDim := inDim
    outDim := outDim
    loAff := { A := zA, c := lo }
    hiAff := { A := zA, c := hi } }

/-- Pointwise addition of affine forms (`A` and `c`). -/
def affAdd {n m : Nat} (a b : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.addSpec a.A b.A, c := Tensor.addSpec a.c b.c }

/-- Pointwise subtraction of affine forms (`A` and `c`). -/
def affSub {n m : Nat} (a b : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.subSpec a.A b.A, c := Tensor.subSpec a.c b.c }

/-! ## α-ReLU lower relaxation -/

/-!
### α-CROWN lower relaxation for ReLU

For a pre-activation scalar \(z\in[l,u]\), CROWN/DeepPoly uses:
- an *upper* linear envelope (the usual triangular relaxation), and
- a *lower* linear envelope.

In the unstable crossing case \(l < 0 < u\), the lower relaxation can be parameterized by
\(\alpha\in[0,1]\) to interpolate between the sound choices \(y \ge 0\) and \(y \ge z\).

We encode this by using `alphaRelaxLowerScalar` for the lower bound, and using
`Runtime.Ops.ReLU.relax_scalar` / `relax_vector` for the upper bound.
-/

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
    (lo hi : Tensor α (.dim n .scalar))
    (αv : Tensor α (.dim n .scalar)) : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim n
      .scalar) :=
  match lo, hi, αv with
  | .dim flo, .dim fhi, .dim fa =>
    Tensor.dim (fun i =>
      match flo i, fhi i, fa i with
      | .scalar l, .scalar u, .scalar a => Tensor.scalar (alphaRelaxLowerScalar (α := α) l u a))

/-! ## Node step function -/

def getAff? (cert : Array (Option (FlatAffineBounds α))) (pid : Nat) : Option (FlatAffineBounds α)
  :=
  if _h : pid < cert.size then cert[pid]! else none

/-- Safe lookup of the optional α vector at node id `pid`. -/
def getAlpha? (alpha : Array (Option (FlatVec α))) (pid : Nat) : Option (FlatVec α) :=
  if _h : pid < alpha.size then alpha[pid]! else none

/--
Default α vector used when the certificate omits α values.

This matches TorchLean's default lower relaxation: pick slope `1` when `u > -l`, otherwise `0`.
-/
def defaultAlphaVec {n : Nat} (lo hi : Tensor α (.dim n .scalar)) : Tensor α (.dim n .scalar) :=
  match lo, hi with
  | .dim flo, .dim fhi =>
      Tensor.dim (fun i =>
        match flo i, fhi i with
        | .scalar l, .scalar u =>
            -- Match TorchLean's default lower relaxation: choose slope 1 iff `u > -l`, else 0.
            let a := if u > (-l) then Numbers.one else Numbers.zero
            Tensor.scalar a)

/--
Transfer rule for affine bounds through a linear layer `y = W x + b` (sign-splitting).

The parent node provides affine lower/upper bounds in terms of the *global input*; we compose with
`W` using the standard `W = W⁺ + W⁻` decomposition (as in IBP/CROWN).
-/
def linearBoundsFromAffine
    {inDim n m : Nat}
    (W : Tensor α (.dim m (.dim n .scalar)))
    (b : Tensor α (.dim m .scalar))
    (xB : FlatAffineBounds α)
    (hout : xB.outDim = n)
    (_hin : xB.inDim = inDim := by rfl) : FlatAffineBounds α :=
  /-
  Linear transfer rule (sign-splitting).

  Suppose the parent node has affine bounds
  \[
    \ell(x) = A_\ell x + c_\ell,\qquad u(x) = A_u x + c_u
  \]
  for its output vector, expressed as functions of the *global input* \(x\).

  For a linear layer \(y = Wx + b\), we propagate affine bounds using the standard
  positive/negative decomposition \(W = W^+ + W^-\), where \(W^+\ge 0\) and \(W^-\le 0\).
  Componentwise, this gives the sound enclosure
  \[
    y \le W^+\,u(x) + W^-\,\ell(x) + b,\qquad
    y \ge W^+\,\ell(x) + W^-\,u(x) + b.
  \]

  This is exactly the same “sign split” as used in IBP, but applied to affine bounds.
  -/
  let xLo : AffineVec α xB.inDim n := castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' :=
    n) hout xB.loAff
  let xHi : AffineVec α xB.inDim n := castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' :=
    n) hout xB.hiAff
  let Wpos := NN.MLTheory.CROWN.IBP.matPos (α := α) (m := m) (n := n) W
  let Wneg := NN.MLTheory.CROWN.IBP.matNeg (α := α) (m := m) (n := n) W
  let A_hi := Tensor.addSpec (Spec.matMulSpec (α := α) Wpos xHi.A) (Spec.matMulSpec (α := α)
    Wneg xLo.A)
  let c_hi :=
    Tensor.addSpec
      (Tensor.addSpec (Spec.matVecMulSpec (α := α) Wpos xHi.c) (Spec.matVecMulSpec (α := α)
        Wneg xLo.c))
      b
  let A_lo := Tensor.addSpec (Spec.matMulSpec (α := α) Wpos xLo.A) (Spec.matMulSpec (α := α)
    Wneg xHi.A)
  let c_lo :=
    Tensor.addSpec
      (Tensor.addSpec (Spec.matVecMulSpec (α := α) Wpos xLo.c) (Spec.matVecMulSpec (α := α)
        Wneg xHi.c))
      b
  { inDim := xB.inDim
    outDim := m
    loAff := { A := A_lo, c := c_lo }
    hiAff := { A := A_hi, c := c_hi } }

/--
One-node α-CROWN step function for a supported subset of IR ops.

This is a *safe* (Option-returning) step: it returns `none` when required parent bounds or
parameters are missing, or when dimensions mismatch.

It is intended to be used for:
- executable per-node certificate checking (recompute node `id` from certificate parents), and
- proof-level soundness theorems about the checker.

## Supported node kinds

This step function handles the verifier core of the IR:
- `.input`, `.const`, `.detach`
- `.linear`, `.matmul` (ParamStore-driven linear operators in the verifier dialect)
- `.relu` (CROWN upper + α-CROWN lower)
- `.sum` (treated as a \(1\times n\) linear layer)
- `.reshape`, `.flatten` (shape-only, guarded by dimensional consistency)

All other node kinds fall back to a conservative **constant** affine enclosure derived from the
IBP box at the same node id (if present). This keeps the checker total over graphs that contain
operators outside this affine-transfer subset; end-to-end theorems then account for those nodes
through the soundness assumptions attached to their IBP boxes.
-/
def alphaCrownStepNode?
    (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α)))
    (alpha : Array (Option (FlatVec α)))
    (cert : Array (Option (FlatAffineBounds α)))
    (ctx : AffineCtx) (id : Nat) : Option (FlatAffineBounds α) :=
  let node := nodes[id]!
  match node.kind with
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
      match node.parents with
      | p1 :: _ => getAff? (α := α) cert p1
      | _ => none
  | .linear =>
      match node.parents with
      | p1 :: _ =>
          match getAff? (α := α) cert p1, ps.linearWB[id]? with
          | some xin, some p =>
              if hout : xin.outDim = p.n then
                let out := linearBoundsFromAffine (α := α) (inDim := xin.inDim) (n := p.n) (m :=
                  p.m) p.w p.b xin hout
                some out
              else
                none
          | _, _ => none
      | _ => none
  | .matmul =>
      match node.parents with
      | p1 :: _ =>
          match getAff? (α := α) cert p1, ps.matmulW[id]? with
          | some xin, some p =>
              if hout : xin.outDim = p.n then
                let zb : Tensor α (.dim p.m .scalar) := Spec.fill (α := α) Numbers.zero (.dim p.m
                  .scalar)
                let out := linearBoundsFromAffine (α := α) (inDim := xin.inDim) (n := p.n) (m :=
                  p.m) p.w zb xin hout
                some out
              else none
          | _, _ => none
      | _ => none
  | .relu =>
      match node.parents with
      | p1 :: _ =>
          match getAff? (α := α) cert p1, ibp[p1]!, getAlpha? (α := α) alpha id with
          | some xin, some preB, some αv =>
              if hout : xin.outDim = preB.dim then
                let xLo : AffineVec α xin.inDim preB.dim := by
                  simpa [hout] using xin.loAff
                let xHi : AffineVec α xin.inDim preB.dim := by
                  simpa [hout] using xin.hiAff
                let relaxHi :=
                  NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := α) (n := preB.dim) preB.lo
                    preB.hi
                let αt : Tensor α (.dim preB.dim .scalar) :=
                  if hα : αv.n = preB.dim then
                    castDimScalar (α := α) (n := αv.n) (n' := preB.dim) hα αv.v
                  else
                    defaultAlphaVec (α := α) (n := preB.dim) preB.lo preB.hi
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
          | some xin, some preB, none =>
              -- No alpha provided: use TorchLean's default 0/1 lower relaxation as a certificate-free
              -- producer.
              if hout : xin.outDim = preB.dim then
                let xLo : AffineVec α xin.inDim preB.dim := by
                  simpa [hout] using xin.loAff
                let xHi : AffineVec α xin.inDim preB.dim := by
                  simpa [hout] using xin.hiAff
                let relaxHi :=
                  NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α := α) (n := preB.dim) preB.lo
                    preB.hi
                let αt := defaultAlphaVec (α := α) (n := preB.dim) preB.lo preB.hi
                let relaxLo := alphaRelaxLowerVec (α := α) (n := preB.dim) preB.lo preB.hi αt
                let loAff :=
                  NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
                    (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                let hiAff :=
                  NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
                    (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi
                some { inDim := xin.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }
              else none
          | _, _, _ => none
      | _ => none
  | .sum =>
      match node.parents with
      | p1 :: _ =>
          match getAff? (α := α) cert p1 with
          | some xin =>
              -- Treat `sum` as a 1×n linear layer with all-ones weights and zero bias.
              let onesRow : Tensor α (.dim 1 (.dim xin.outDim .scalar)) :=
                Spec.fill (α := α) Numbers.one (.dim 1 (.dim xin.outDim .scalar))
              let zb : Tensor α (.dim 1 .scalar) := Spec.fill (α := α) Numbers.zero (.dim 1 .scalar)
              let out :=
                linearBoundsFromAffine (α := α)
                  (inDim := xin.inDim) (n := xin.outDim) (m := 1)
                  onesRow zb xin (by rfl)
              some out
          | none => none
      | _ => none
  | .reshape _ _ | .flatten _ =>
      match node.parents with
      | p1 :: _ =>
          match getAff? (α := α) cert p1 with
          | some xin =>
              -- The semantic evaluator for reshape/flatten checks `xin.outDim = node.outShape.size`
              -- before returning a value. Mirror that here to keep transfer soundness provable.
              if hout : xin.outDim = node.outShape.size then
                let loAff := castAffineOut (α := α) (n := xin.inDim) (m := xin.outDim) (m' :=
                  node.outShape.size) hout xin.loAff
                let hiAff := castAffineOut (α := α) (n := xin.inDim) (m := xin.outDim) (m' :=
                  node.outShape.size) hout xin.hiAff
                some { inDim := xin.inDim, outDim := node.outShape.size, loAff := loAff, hiAff :=
                  hiAff }
              else
                none
          | none => none
      | _ => none
  | _ =>
      -- Conservative fallback: allow a constant affine enclosure derived from IBP (if present).
      match ibp[id]! with
      | some B => some (boundsConst (α := α) ctx.inputDim B.dim B.lo B.hi)
      | none => none

end NN.MLTheory.CROWN.Cert
