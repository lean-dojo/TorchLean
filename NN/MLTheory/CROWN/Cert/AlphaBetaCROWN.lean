/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaCROWN
public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Run

/-!
# α/β-CROWN certificate step function (graph dialect)

This extends `AlphaCROWN.alphaCrownStepNode?` with explicit β-phase constraints for ReLU:

*For a ReLU node*, a certificate may additionally provide a per-neuron phase vector:
- `-1` = neuron forced inactive (`z ≤ 0`)
- `0`  = no phase constraint (unstable)
- `1`  = neuron forced active (`0 ≤ z`)

The step function checks phase consistency *against the provided IBP pre-activation bounds*:
- active requires `0 ≤ lo`
- inactive requires `hi ≤ 0`

If the phase vector is present and consistent, the propagated affine bounds use the exact
ReLU behavior for active/inactive neurons (slope 1 / slope 0), and the usual α-CROWN / CROWN
relaxations for unstable neurons.

Trust boundary: β is additional *evidence* verified against the trusted-for-this-theorem IBP bounds.

## Background / citations

This checker is kept narrow and certificate-friendly. Conceptually, β-phase
information corresponds to the phase constraints used in branch-and-bound based verifiers
(and in β-CROWN-style splitting), where additional information can turn unstable ReLUs into
provably active/inactive ones.

- CROWN: Zhang et al., *Efficient Neural Network Robustness Certification with General Activation
  Functions*, NeurIPS 2018.
- β-CROWN (splitting / phase refinement): Wang et al., *Beta-CROWN: Efficient Bound Propagation
  with Provable Guarantees*, NeurIPS 2021 (and follow-up work).

We do **not** attempt to formalize the full β-CROWN search/optimization loop here; the checker only
verifies that the provided β phases are consistent with IBP and then uses the corresponding exact
ReLU transfer rule (slope $0$ or $1$).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Cert

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph

variable {α : Type} [TorchLean.Storage α] [Context α]

/-!
## ReLU phase encoding

We represent β information using a three-valued phase:
- `inactive`: pre-activation $z \le 0$, so ReLU is exactly $0$;
- `active`:   pre-activation $0 \le z$, so ReLU is exactly $z$;
- `unstable`: no phase constraint, so we fall back to linear relaxations.

At runtime, a certificate provides an `Array Int` with entries in $\{-1,0,1\}$, which we decode
using `ReLUPhase.ofInt?`.
-/

/-- Which side of the origin a certificate claims a pre-activation stays on.

`unstable` is the honest "do not know", and it is the only phase that forces the relaxation to
spend an α parameter. -/
inductive ReLUPhase where
  /-- The pre-activation is claimed nonpositive, so the ReLU output is zero. -/
  | inactive
  /-- No claim: the pre-activation may cross zero. -/
  | unstable
  /-- The pre-activation is claimed nonnegative, so the ReLU acts as the identity. -/
  | active
  deriving DecidableEq, Repr

/-- Encode a `ReLUPhase` as the certificate integer convention `{-1, 0, 1}`. -/
@[simp] def ReLUPhase.toInt : ReLUPhase → Int
  | .inactive => -1
  | .unstable => 0
  | .active => 1

/-- Decode a runtime β phase integer (`-1/0/1`) into a `ReLUPhase`, returning `none` on invalid
  input. -/
def ReLUPhase.ofInt? (i : Int) : Option ReLUPhase :=
  if i = (-1) then some .inactive
  else if i = 0 then some .unstable
  else if i = 1 then some .active
  else none

/-- Safe lookup of the optional β vector at node id `id`. -/
def getBeta? (beta : Array (Option (Array Int))) (id : Nat) : Option (Array Int) :=
  if _h : id < beta.size then beta[id]! else none

/--
Executable phase-consistency check for a scalar pre-activation interval $[l,u]$.

This is written for a generic `Context α` (so it can run over floats or other semirings).
For proof-level soundness over `ℝ`, see theorems in
`NN/MLTheory/CROWN/Proofs/AlphaBetaReLUScalarSoundness.lean`.
-/
def phaseConsistentScalar? (l u : α) (ph : ReLUPhase) : Option Unit :=
  match ph with
  | .inactive =>
      -- `u ≤ 0` (checked via `¬ (0 < u)` to stay executable under `Context α`).
      if u > 0 then none else some ()
  | .active =>
      -- `0 ≤ l` (checked via `¬ (l < 0)` to stay executable under `Context α`).
      if l < 0 then none else some ()
  | .unstable => some ()

/-- Phase-aware **upper** (over-approx) linear relaxation for ReLU. -/
def phaseRelaxUpperScalar (l u : α) (ph : ReLUPhase) : NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α :=
  match ph with
  | .inactive => { slope := 0, bias := 0 }
  | .active   => { slope := 1, bias := 0 }
  | .unstable => NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := α) l u

/-- Phase-aware **lower** (under-approx) linear relaxation for ReLU. -/
def phaseRelaxLowerScalar (l u a : α) (ph : ReLUPhase) :
    NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α :=
  match ph with
  | .inactive => { slope := 0, bias := 0 }
  | .active   => { slope := 1, bias := 0 }
  | .unstable => alphaRelaxLowerScalar (α := α) l u a

/--
Opaque wrapper around `Array.get!` for β-phase vectors.

Why this exists:
* The executable definition `phaseRelaxVec?` indexes `phases` with `get!` after a
  length check, to avoid threading bounds proofs through the code.
* During proofs, `simp` can sometimes rewrite `get!` into the safe indexing `phases[i]` (which
  carries a proof argument). Those proof terms are definitional artifacts and can make routine
  simplification depend on irrelevant proof objects.

Keeping the indexing step opaque prevents `simp` from introducing proof-carrying indices, while
preserving executability.
-/
opaque betaAt (phases : Array Int) (i : Nat) : Int := phases[i]!

/--
Vectorized construction of ReLU relaxations under β-phase constraints.

Returns `(relaxLo, relaxHi)` if:
- the β vector has the correct length, and
- every scalar entry is decodable and phase-consistent with the IBP interval bounds.

Implementation notes:
- We first compute a boolean `ok` over all indices (so the function remains executable in a
  generic `Context α`).
- We access the β-phase vector with `get!` after checking the length. This avoids carrying bounds
  proofs while remaining total as Lean code.
-/
def phaseRelaxVec?
    {n : Nat}
    (lo hi : Tensor α [n])
    (αv : Tensor α [n])
    (phases : Array Int) :
    Option
      (Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n] ×
       Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n]) := by
  classical
  exact
    -- Check length first.
    if hlen : phases.size = n then
      -- Check consistency of each scalar constraint.
      let phaseAt : Fin n → Int := fun i => betaAt phases i
      let ok :=
        (List.finRange n).all fun i =>
          match ReLUPhase.ofInt? (phaseAt i) with
          | some ph => (phaseConsistentScalar? (α := α)
              (lo.getScalar i) (hi.getScalar i) ph).isSome
          | none => false
      if ok then
        let relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n] :=
          Tensor.dim fun i =>
            Tensor.scalar <|
              match ReLUPhase.ofInt? (phaseAt i) with
              | some ph => phaseRelaxUpperScalar (α := α)
                  (lo.getScalar i) (hi.getScalar i) ph
              | none => { slope := 0, bias := 0 }
        let relaxLo : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n] :=
          Tensor.dim fun i =>
            Tensor.scalar <|
              match ReLUPhase.ofInt? (phaseAt i) with
              | some ph => phaseRelaxLowerScalar (α := α)
                  (lo.getScalar i) (hi.getScalar i) (αv.getScalar i) ph
              | none => { slope := 0, bias := 0 }
        some (relaxLo, relaxHi)
      else
        none
    else
      none

/--
One-node α/β-CROWN step:
- delegates to `alphaCrownStepNode?` for all non-ReLU ops;
- for ReLU, optionally uses a β phase vector (if provided at `beta[id]`).

If no β info is present for this node, this is exactly `alphaCrownStepNode?`.
-/
def alphaBetaCrownStepNode?
    (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α)))
    (alpha : Array (Option (FlatTensor α)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds α)))
    (ctx : AffineCtx) (id : Nat) : Option (FlatAffineBounds α) :=
  let node := nodes[id]!
  match node.kind with
  | .relu =>
      match getBeta? (beta := beta) id with
      | none =>
          alphaCrownStepNode? (α := α) nodes ps ibp alpha cert ctx id
      | some phases =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              match getAff? (α := α) cert p1, ibp[p1]!, getAlpha? (α := α) alpha id with
              | some xin, some preB, some αv =>
                  if hout : xin.outDim = preB.dim then
                    if hα : αv.n = preB.dim then
                      let xLo : AffineVec α xin.inDim preB.dim := by simpa [hout] using xin.loAff
                      let xHi : AffineVec α xin.inDim preB.dim := by simpa [hout] using xin.hiAff
                      let αt : Tensor α [preB.dim] :=
                        castDimScalar (α := α) (n := αv.n) (n' := preB.dim) hα αv.v
                      match phaseRelaxVec? (α := α) (n := preB.dim) preB.lo preB.hi αt phases with
                      | some (relaxLo, relaxHi) =>
                          let loAff :=
                            NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
                              (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                          let hiAff :=
                            NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
                              (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi
                          some { inDim := xin.inDim, outDim := preB.dim, loAff := loAff, hiAff :=
                            hiAff }
                      | none => none
                    else none
                  else none
              | some xin, some preB, none =>
                  -- No alpha provided: follow AlphaCROWN's default lower relaxation, but still
                  -- enforce β consistency.
                  if hout : xin.outDim = preB.dim then
                    let xLo : AffineVec α xin.inDim preB.dim := by simpa [hout] using xin.loAff
                    let xHi : AffineVec α xin.inDim preB.dim := by simpa [hout] using xin.hiAff
                    let αt := defaultAlphaVec (α := α) (n := preB.dim) preB.lo preB.hi
                    match phaseRelaxVec? (α := α) (n := preB.dim) preB.lo preB.hi αt phases with
                    | some (relaxLo, relaxHi) =>
                        let loAff :=
                          NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
                            (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
                        let hiAff :=
                          NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
                            (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi
                        some { inDim := xin.inDim, outDim := preB.dim, loAff := loAff, hiAff :=
                          hiAff }
                    | none => none
                  else none
              | _, _, _ => none
          | none => none
  | _ =>
      alphaCrownStepNode? (α := α) nodes ps ibp alpha cert ctx id

/-!
## Automatic graph execution

The certificate step above accepts explicit α and β evidence. The helpers below provide the
ordinary executable path used by TorchLean's high-level verifier: α uses the checker's default
relaxation, while β records every ReLU phase already proved stable by IBP. Crossing intervals remain
unconstrained. This is a fixed-relaxation α/β-CROWN pass, not the external branch-and-bound search
performed by full Alpha-Beta-CROWN implementations.
-/

/-- Infer the strongest phase justified directly by one pre-activation interval. -/
def inferredPhase (lo hi : α) : ReLUPhase :=
  if hi > 0 then
    if lo < 0 then .unstable else .active
  else
    .inactive

/-- Infer an IBP-justified β vector for one ReLU node. -/
def inferredBetaForNode?
    (nodes : Array Node) (ibp : Array (Option (FlatBox α))) (id : Nat) :
    Option (Array Int) := do
  let node ← nodes[id]?
  match node.kind with
  | .relu => pure ()
  | _ => none
  let parent ← NN.IR.unaryParent? node.parents
  let preactivation ← ibp[parent]?
  let box ← preactivation
  pure <| (List.finRange box.dim).toArray.map fun i =>
    (inferredPhase (α := α) (box.lo.getScalar i) (box.hi.getScalar i)).toInt

/-- Infer all β phase vectors that are already justified by the IBP pass. -/
def inferredBeta
    (nodes : Array Node) (ibp : Array (Option (FlatBox α))) :
    Array (Option (Array Int)) :=
  (List.finRange nodes.size).toArray.map fun id =>
    inferredBetaForNode? (α := α) nodes ibp id.val

/--
Run the fixed-relaxation α/β-CROWN graph pass.

Stable ReLU phases come from IBP and are checked again by `alphaBetaCrownStepNode?`. Unstable
neurons use the default α-CROWN lower relaxation.
-/
def runAlphaBetaCROWN
    (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) :
    Array (Option (FlatAffineBounds α)) :=
  let alpha : Array (Option (FlatTensor α)) := Array.replicate g.nodes.size none
  let beta := inferredBeta (α := α) g.nodes ibp
  let initial : Array (Option (FlatAffineBounds α)) := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl
    (fun bounds id =>
      bounds.set! id <|
        alphaBetaCrownStepNode? (α := α)
          g.nodes ps ibp alpha beta bounds ctx id)
    initial

/-- Evaluate the automatic α/β-CROWN pass at the graph output. -/
def outputBoxAlphaBetaCROWN?
    [BoundOps α] [NonlinearBoundOps α]
    (g : Graph) (ps : ParamStore α) (xB : FlatBox α)
    (inputId outputId inputDim : Nat) : Except String (FlatBox α) := do
  let ibp := Graph.runIBP (α := α) g ps
  let ctx : AffineCtx := { inputId := inputId, inputDim := inputDim }
  let bounds := runAlphaBetaCROWN (α := α) g ps ctx ibp
  Graph.evalCROWNOutputBox? (α := α) bounds xB outputId inputDim

end NN.MLTheory.CROWN.Cert
