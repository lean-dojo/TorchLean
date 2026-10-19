/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaCROWN

/-!
# α-CROWN with ReLU phases (graph dialect)

This extends `AlphaCROWN.alphaCrownStepNode?` with optional ReLU phases. For a ReLU node, a
certificate may provide one phase per neuron:
- `-1` = neuron inactive (`z ≤ 0`)
- `0`  = no phase (unstable)
- `1`  = neuron active (`0 ≤ z`)

The step checks each phase against the IBP pre-activation bounds: active requires `0 ≤ lo`, and
inactive requires `hi ≤ 0`. When every phase is consistent, active and inactive neurons use the
exact ReLU (slope 1 or 0) and unstable neurons use the usual α-CROWN relaxation.

Because a phase is accepted only when the IBP interval already proves it, phases add no
information beyond IBP. α-CROWN already uses the identity when `0 < lo` and zero when `hi ≤ 0`, so
there the two passes agree. They differ only for an active phase with `lo = 0`: the phase uses
the identity, while α-CROWN uses the lower slope α when `0 < hi` and zero when `hi = 0`. Both lines
agree at zero, so either choice is sound. Without phases the pass is exactly
`alphaCrownStepNode?`. There are no Lagrangian β multipliers and no branch splits.

Phases have the form of the ReLU split constraints of β-CROWN (Wang et al., *Beta-CROWN: Efficient
Bound Propagation with Provable Guarantees*, NeurIPS 2021), where branch-and-bound chooses the
splits and β multipliers enforce them; neither is formalized here. CROWN itself is Zhang et al.,
*Efficient Neural Network Robustness Certification with General Activation Functions*, NeurIPS 2018.
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

A phase has three values:
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

/-- Decode a phase integer (`-1/0/1`) into a `ReLUPhase`, returning `none` on invalid input. -/
def ReLUPhase.ofInt? (i : Int) : Option ReLUPhase :=
  if i = (-1) then some .inactive
  else if i = 0 then some .unstable
  else if i = 1 then some .active
  else none

/-- The phases of node `id`, if the certificate gives any. -/
def getPhases? (phases : Array (Option (Array Int))) (id : Nat) : Option (Array Int) :=
  phases.getD id none

/--
Executable phase-consistency check for a scalar pre-activation interval $[l,u]$.

This is written for a generic `Context α` (so it can run over floats or other semirings).
For proof-level soundness over `ℝ`, see theorems in
`NN/MLTheory/CROWN/Proofs/PhasedReLUScalarSoundness.lean`.
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
Read the phase of neuron `i`, with `0` (unconstrained) past the end of the array.

`phaseRelaxVec?` checks the array length before reading, so the default is never used there. The
total read keeps the executable code free of bounds proofs, and the definition reduces in the
kernel.
-/
def phaseAt (phases : Array Int) (i : Nat) : Int := phases.getD i 0

/--
Vectorized construction of ReLU relaxations under phase constraints.

Returns `(relaxLo, relaxHi)` if:
- the phase array has the correct length, and
- every scalar entry is decodable and phase-consistent with the IBP interval bounds.

Implementation notes:
- We first compute a boolean `ok` over all indices (so the function remains executable in a
  generic `Context α`).
- Phases are read with `phaseAt` (a total `Array.getD` read) after checking the length, so no
  bounds proofs are carried and the definition reduces in the kernel.
-/
def phaseRelaxVec?
    {n : Nat}
    (lo hi : Tensor α [n])
    (αv : Tensor α [n])
    (phases : Array Int) :
    Option
      (Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n] ×
       Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n]) :=
  -- Check length first.
  if phases.size = n then
    -- Check consistency of each scalar constraint.
    let ok :=
      (List.finRange n).all fun i =>
        match ReLUPhase.ofInt? (phaseAt phases i) with
        | some ph => (phaseConsistentScalar? (α := α)
            (lo.getScalar i) (hi.getScalar i) ph).isSome
        | none => false
    if ok then
      let relaxHi : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n] :=
        Tensor.dim fun i =>
          Tensor.scalar <|
            match ReLUPhase.ofInt? (phaseAt phases i) with
            | some ph => phaseRelaxUpperScalar (α := α)
                (lo.getScalar i) (hi.getScalar i) ph
            | none => { slope := 0, bias := 0 }
      let relaxLo : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [n] :=
        Tensor.dim fun i =>
          Tensor.scalar <|
            match ReLUPhase.ofInt? (phaseAt phases i) with
            | some ph => phaseRelaxLowerScalar (α := α)
                (lo.getScalar i) (hi.getScalar i) (αv.getScalar i) ph
            | none => { slope := 0, bias := 0 }
      some (relaxLo, relaxHi)
    else
      none
  else
    none

/--
One-node α-CROWN step with ReLU phases:
- delegates to `alphaCrownStepNode?` for all non-ReLU ops;
- for ReLU, uses the node's phases if `phases[id]` provides them.

If the node has no phases, this is exactly `alphaCrownStepNode?`.
-/
def phasedCrownStepNode?
    (nodes : Array Node) (ps : ParamStore α)
    (ibp : Array (Option (FlatBox α)))
    (alpha : Array (Option (FlatTensor α)))
    (phases : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds α)))
    (ctx : AffineCtx) (id : Nat) : Option (FlatAffineBounds α) :=
  let propagate (xin : FlatAffineBounds α) (preB : FlatBox α)
      (hout : xin.outDim = preB.dim) (αt : Tensor α [preB.dim])
      (nodePhases : Array Int) : Option (FlatAffineBounds α) :=
    let xLo := Graph.castAffineOut (α := α) hout xin.loAff
    let xHi := Graph.castAffineOut (α := α) hout xin.hiAff
    match phaseRelaxVec? (α := α) (n := preB.dim) preB.lo preB.hi αt nodePhases with
    | some (relaxLo, relaxHi) =>
        let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
          (inDim := xin.inDim) (hidDim := preB.dim) relaxLo xLo
        let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α := α)
          (inDim := xin.inDim) (hidDim := preB.dim) relaxHi xHi
        some { inDim := xin.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }
    | none => none
  let node := nodes[id]!
  match node.kind with
  | .relu =>
      match getPhases? phases id with
      | none =>
          alphaCrownStepNode? (α := α) nodes ps ibp alpha cert ctx id
      | some nodePhases =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              match getAff? (α := α) cert p1, ibp[p1]!, getAlpha? (α := α) alpha id with
              | some xin, some preB, some αv =>
                  if hout : xin.outDim = preB.dim then
                    if hα : αv.n = preB.dim then
                      let αt : Tensor α [preB.dim] :=
                        castDimScalar (α := α) (n := αv.n) (n' := preB.dim) hα αv.v
                      propagate xin preB hout αt nodePhases
                    else none
                  else none
              | some xin, some preB, none =>
                  -- No alpha provided: follow AlphaCROWN's default lower relaxation, but still
                  -- check the phases.
                  if hout : xin.outDim = preB.dim then
                    let αt := defaultAlphaVec (α := α) (n := preB.dim) preB.lo preB.hi
                    propagate xin preB hout αt nodePhases
                  else none
              | _, _, _ => none
          | none => none
  | _ =>
      alphaCrownStepNode? (α := α) nodes ps ibp alpha cert ctx id

/-!
## Automatic graph execution

The certificate step above accepts explicit α vectors and phases. The helpers below provide the
ordinary executable path used by TorchLean's high-level verifier: α uses the checker's default
relaxation, and the phases record every ReLU already proved stable by IBP. Crossing intervals
remain unconstrained. This is a fixed-relaxation pass, not the branch-and-bound search of the
external alpha-beta-CROWN verifier.

The output-box API uses this affine pass only for backends that certify exact reassociation.
Rounded backends use directed CROWN instead; rounded affine coefficients alone are not sound
enclosures of the real-valued model. The requested input box seeds both paths.
-/

/-- Infer the strongest phase justified directly by one pre-activation interval. -/
def inferredPhase (lo hi : α) : ReLUPhase :=
  if hi > 0 then
    if lo < 0 then .unstable else .active
  else
    .inactive

/-- The IBP-justified phases of one ReLU node. -/
def inferredPhasesAt?
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

/-- The phases of every ReLU node that the IBP pass already justifies. -/
def inferredPhases
    (nodes : Array Node) (ibp : Array (Option (FlatBox α))) :
    Array (Option (Array Int)) :=
  (List.finRange nodes.size).toArray.map fun id =>
    inferredPhasesAt? (α := α) nodes ibp id.val

/--
Run the fixed-relaxation α-CROWN graph pass with IBP phase annotations.

Stable ReLU phases come from IBP and are checked again by `phasedCrownStepNode?`. Unstable
neurons use the default α-CROWN lower relaxation. No β multipliers are optimized and no neuron is
split.
-/
def runPhasedCROWN
    (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) :
    Array (Option (FlatAffineBounds α)) :=
  let alpha : Array (Option (FlatTensor α)) := Array.replicate g.nodes.size none
  let phases := inferredPhases (α := α) g.nodes ibp
  let initial : Array (Option (FlatAffineBounds α)) := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl
    (fun bounds id =>
      bounds.set! id <|
        phasedCrownStepNode? (α := α)
          g.nodes ps ibp alpha phases bounds ctx id)
    initial

/-- Evaluate the automatic phased α-CROWN pass at the graph output.

Rounded scalars use directed CROWN: composing nearest-rounded affine coefficients is not a
sound enclosure calculation. Both routes seed IBP with the caller's requested input box.
-/
def outputBoxPhasedCROWN?
    [BoundOps α] [NonlinearBoundOps α]
    (g : Graph) (ps : ParamStore α) (xB : FlatBox α)
    (inputId outputId inputDim : Nat) : Except String (FlatBox α) := do
  if !BoundOps.supportsExactAffineReassociation (α := α) then
    return ← Graph.outputBoxCROWN? g ps xB inputId outputId inputDim
  let ps := ps.seedInputBox inputId xB
  let ibp := Graph.runIBP (α := α) g ps
  let ctx : AffineCtx := { inputId := inputId, inputDim := inputDim }
  let bounds := runPhasedCROWN (α := α) g ps ctx ibp
  Graph.evalCROWNOutputBox? (α := α) bounds xB outputId inputDim

end NN.MLTheory.CROWN.Cert
