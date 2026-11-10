/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Transfer
public import TorchLean.Tensor.Contract

/-!
# Affine bounds of flattened values

Affine lower and upper forms of a node's flattened value in a designated input (`AffineCtx`,
`FlatAffineBounds`): the identity and constant forms, their composition through `W x + b`
(`propagateLinearBounds`), their evaluation on an input box (`evalCROWNOutputBox?`,
`evalObjectiveBox?`), and the α-CROWN and phase-aware ReLU relaxations the forward affine
pass uses. Nothing here depends on a graph representation.

Our forward pass is the forward mode of auto_LiRPA (Sec. 3.1, Algorithm 1), translated into Lean
by hand. A value carries a lower and an upper affine form in the input, and the input starts with
the identity form. `propagateLinearBounds` is the affine row of their Table 6, with `IBP.matPos`
and `IBP.matNeg` as the positive and negative parts of `W`. Over `ℝ`,
`Graph.enclosesAtInput_linear` proves that the resulting pair encloses `W v + b`.

References:
- auto_LiRPA: Xu et al.,
  "Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond",
  NeurIPS 2020, arXiv:2002.12920, Sec. 3.1 (Algorithm 1) and App. A.1 (Table 6).
- CROWN: Zhang et al.,
  "Efficient Neural Network Robustness Certification with General Activation Functions",
  NeurIPS 2018, arXiv:1811.00866, Sec. 3.2.
- α-CROWN: Xu et al., "Fast and Complete: Enabling Complete Neural Network Verification with
  Rapid and Massively Parallel Incomplete Verifiers", ICLR 2021, arXiv:2011.13824, Sec. 2.3,
  Eq. (5), and Sec. 3.1, Eq. (9).
-/

public section

namespace TorchLean.CROWN.Graph

open TorchLean.Tensor

section
variable {α : Type} [Storage α] [Context α]
variable [BoundOps α]

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
def affIdentity (n : Nat) : Linear α n n :=
  { weights := Tensor.identity n, bias := Tensor.full [n] 0 }

/-- Exact lower and upper affine bounds for the identity node. -/
@[expose] def boundsIdentity (n : Nat) : FlatAffineBounds α :=
  { inDim := n, outDim := n, loAff := affIdentity (α:=α) n, hiAff := affIdentity (α:=α) n }

/-- Constant affine bounds: a zero coefficient matrix with the given lower and upper offsets. -/
@[expose]
def boundsConst (inputDim outDim : Nat) (lo hi : Tensor α [outDim]) :
  FlatAffineBounds α :=
  let zA := Tensor.full (α:=α) (.dim outDim (.dim inputDim .scalar)) 0
  { inDim := inputDim
    outDim := outDim
    loAff := { weights := zA, bias := lo }
    hiAff := { weights := zA, bias := hi } }

/-- Algebraic composition of affine bounds through `W*x + b` on exact scalar backends.

If the parent has affine bounds `ℓ(x) = Aₗ x + cₗ ≤ v ≤ u(x) = Aᵤ x + cᵤ` in the global input `x`,
the sign split `W = W⁺ + W⁻` (`W⁺ ≥ 0`, `W⁻ ≤ 0`) gives the componentwise enclosure
`W⁺ ℓ(x) + W⁻ u(x) + b ≤ W v + b ≤ W⁺ u(x) + W⁻ ℓ(x) + b`, the same rule IBP uses on boxes.
In binary32 each sum and product rounds to nearest, and the result need not enclose `W v + b`:
with the parent form `x + 1`, the weight `1` and the bias `2 ^ -24`, the offset `1 + 2 ^ -24` is
a tie that rounds to `1`, and the upper form `x + 1` is `0` at `x = -1`, below `W v + b = 2 ^ -24`.
The α-CROWN certificate checker `Certificate.CROWNNode.check` replays this rule in binary32, and
the graph runner selects directed backward propagation for rounded backends.
-/
@[expose]
def propagateLinearBounds
  {n m : Nat}
  (W : Tensor α [m, n])
  (b : Tensor α [m])
  (xB : FlatAffineBounds α)
  (hout : xB.outDim = n) : FlatAffineBounds α :=
  -- Align parent affines to outDim=n.
  let xLo : Linear α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.loAff
  let xHi : Linear α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.hiAff
  let Wpos := IBP.matPos (α:=α) (m:=m) (n:=n) W
  let Wneg := IBP.matNeg (α:=α) (m:=m) (n:=n) W
  let A_hi :=
    Tensor.matmul (α:=α) Wpos xHi.weights + Tensor.matmul (α:=α) Wneg xLo.weights
  let c_hi :=
    (Tensor.contract Wpos xHi.bias _ _ [m] .matVec +
          Tensor.contract Wneg xLo.bias _ _ [m] .matVec) +
      b
  let A_lo :=
    Tensor.matmul (α:=α) Wpos xLo.weights + Tensor.matmul (α:=α) Wneg xHi.weights
  let c_lo :=
    (Tensor.contract Wpos xLo.bias _ _ [m] .matVec +
          Tensor.contract Wneg xHi.bias _ _ [m] .matVec) +
      b
  { inDim := xB.inDim
    outDim := m
    loAff := { weights := A_lo, bias := c_lo }
    hiAff := { weights := A_hi, bias := c_hi } }

end

section
variable {α : Type} [Storage α] [Context α]
variable [BoundOps α]
variable [NonlinearBoundOps α]

/-- Evaluate the CROWN affine bounds of node `outputId` on an input box. Fails when that node has
no affine bound or when the input dimensions disagree. -/
@[expose] def evalCROWNOutputBox? (bounds : Array (Option (FlatAffineBounds α))) (xB : FlatBox α)
    (outputId inputDim : Nat) : Except String (FlatBox α) := do
  let outAff ←
    match bounds[outputId]? with
    | some (some outAff) => pure outAff
    | some none => throw s!"CROWN produced no affine bound at output node {outputId}"
    | none => throw s!"output node {outputId} is out of bounds for {bounds.size} CROWN entries"
  if hIn : outAff.inDim = inputDim then
    if hXB : xB.dim = inputDim then
      let outB := outAff.evalOnFlatBox xB (by simpa [hXB] using hIn.symm)
      pure { dim := outAff.outDim, lo := outB.lo, hi := outB.hi }
    else
      throw s!"input box dimension mismatch: got {xB.dim}, expected {inputDim}"
  else
    throw s!"CROWN input dimension mismatch: got {outAff.inDim}, expected {inputDim}"

end

section
variable {α : Type} [Storage α] [Context α]
variable [BoundOps α]

/-- Evaluate the affine bounds of a scalar backward-CROWN objective on an input box. Fails unless
the bounds have one output and the input dimensions agree. -/
@[expose] def evalObjectiveBox? (bounds : FlatAffineBounds α) (xB : FlatBox α)
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

end

end TorchLean.CROWN.Graph

end

@[expose] public section

namespace TorchLean.CROWN.Cert

open TorchLean.Tensor
open Graph

variable {α : Type} [Storage α] [Context α]

/-! ## α-ReLU lower relaxation -/

/-!
### α-CROWN lower relaxation for ReLU

For a pre-activation scalar $z\in[l,u]$, CROWN/DeepPoly uses:
- an *upper* linear envelope (the usual triangular relaxation), and
- a *lower* linear envelope.

In the unstable crossing case $l < 0 < u$, the lower relaxation can be parameterized by
$\alpha\in[0,1]$ to interpolate between the sound choices $y \ge 0$ and $y \ge z$.

`alphaRelaxLowerScalar` gives the lower line, and `ReLU.relaxScalar` and its
componentwise form `relaxVector` give the upper one.

Eq. (5) of α-CROWN (Sec. 2.3) gives both lines, copied from the paper. Its authors stay with the
backward mode (footnote 1), and our pass runs forward. The paper tunes α by projected gradient
descent (its Eq. (9)). Nothing here tunes it: the caller hands in α per node, `defaultAlphaVec`
fills in a node it leaves out, and `Cert.alphaRelaxLowerScalar_sound` holds for any `a` in
`[0, 1]`. One case is ours. With `l = 0 < u` the test `l > 0` fails and the neuron gets `y = a·z`,
where Eq. (5) gives the identity. Both lie below ReLU on `[0, u]`. The default α there is `1`,
since `u > -l`, so the two agree unless the caller picks a smaller slope.
-/

/-- The α-CROWN lower relaxation of a scalar ReLU on `[l, u]` with slope parameter `a`.

A stable-active neuron gets the identity, a stable-inactive one gets zero, and an unstable one gets
the tunable line `y = a·x` through the origin. -/
def alphaRelaxLowerScalar (l u a : α) : CROWN.ReLURelax α :=
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
    (αv : Tensor α [n]) : Tensor (CROWN.ReLURelax α) [n] :=
  Tensor.dim fun i =>
    Tensor.scalar <|
      alphaRelaxLowerScalar (α := α) (lo.getScalar i) (hi.getScalar i) (αv.getScalar i)

/-- Safe lookup of the affine bounds recorded for node id `pid`, `none` when out of range. -/
def getAff? (cert : Array (Option (FlatAffineBounds α))) (pid : Nat) :
    Option (FlatAffineBounds α) :=
  cert.getD pid none

/-- Safe lookup of the optional α vector at node id `pid`. -/
def getAlpha? (alpha : Array (Option (FlatTensor α))) (pid : Nat) : Option (FlatTensor α) :=
  alpha.getD pid none

/--
Default α vector used when the certificate omits α values.

It gives an unstable neuron the lower relaxation slope `1` when `u > -l`, otherwise `0`. That is
CROWN's adaptive slope (Zhang et al., Sec. 3.2) in the strict form `α = I(u > |l|)` that auto_LiRPA
states (App. A.1): at `l = -1`, `u = 1` we pick `0` where CROWN picks `1`, and either line leaves a
gap of area `1/2` under ReLU. `Cert.defaultAlphaVec_range` proves every entry lies in `[0, 1]`.
-/
def defaultAlphaVec {n : Nat} (lo hi : Tensor α [n]) : Tensor α [n] :=
  Tensor.dim fun i =>
    let l := lo.getScalar i
    let u := hi.getScalar i
    Tensor.scalar <| if u > (-l) then 1 else 0

/-!
## ReLU phase encoding

A phase takes one of three values:
- `inactive`: the pre-activation satisfies $z \le 0$, so ReLU is exactly $0$.
- `active`: the pre-activation satisfies $0 \le z$, so ReLU is exactly $z$.
- `unstable`: there is no phase constraint, and the linear relaxations apply.

A certificate supplies the phases at run time as an `Array Int` with entries in $\{-1,0,1\}$,
decoded by `ReLUPhase.ofInt?`.
-/

/-- Which side of the origin a certificate claims a pre-activation stays on.

`unstable` leaves the sign open, and it is the only phase whose lower relaxation uses an α
parameter. -/
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
Executable check that a scalar pre-activation interval $[l,u]$ is consistent with a claimed phase:
`inactive` needs $u \le 0$, `active` needs $0 \le l$, and `unstable` always passes.

It is written over a generic `Context α`, so it runs on floats as well as on exact scalars. The
soundness theorems over `ℝ` are in `TorchLean.Verify.CROWN.FlatBox.Affine.Real`.
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

/-- The upper relaxation of ReLU on `[l, u]` under the phase `ph`: the line `0` when inactive, the
identity when active and `ReLU.relaxScalar l u` when unstable. -/
def phaseRelaxUpperScalar (l u : α) (ph : ReLUPhase) : CROWN.ReLURelax α :=
  match ph with
  | .inactive => { slope := 0, bias := 0 }
  | .active   => { slope := 1, bias := 0 }
  | .unstable => CROWN.ReLU.relaxScalar (α := α) l u

/-- Lower linear relaxation of a scalar ReLU under a phase: zero when inactive, the identity when
active, and `alphaRelaxLowerScalar` with slope `a` when unstable. -/
def phaseRelaxLowerScalar (l u a : α) (ph : ReLUPhase) :
    CROWN.ReLURelax α :=
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
Lower and upper ReLU relaxations of a vector under phase constraints.

Returns `(relaxLo, relaxHi)` when the phase array has length `n` and every entry decodes to a
phase consistent with the IBP interval of its neuron, and `none` otherwise.

All entries are checked in one boolean pass before either relaxation is built, which keeps the
function executable over a generic `Context α`. Phases are read with `phaseAt` once the length is
known to match.
-/
def phaseRelaxVec?
    {n : Nat}
    (lo hi : Tensor α [n])
    (αv : Tensor α [n])
    (phases : Array Int) :
    Option
      (Tensor (CROWN.ReLURelax α) [n] ×
       Tensor (CROWN.ReLURelax α) [n]) :=
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
      let relaxHi : Tensor (CROWN.ReLURelax α) [n] :=
        Tensor.dim fun i =>
          Tensor.scalar <|
            match ReLUPhase.ofInt? (phaseAt phases i) with
            | some ph => phaseRelaxUpperScalar (α := α)
                (lo.getScalar i) (hi.getScalar i) ph
            | none => { slope := 0, bias := 0 }
      let relaxLo : Tensor (CROWN.ReLURelax α) [n] :=
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


/-- Infer the strongest phase justified directly by one pre-activation interval. -/
def inferredPhase (lo hi : α) : ReLUPhase :=
  if hi > 0 then
    if lo < 0 then .unstable else .active
  else
    .inactive

end TorchLean.CROWN.Cert
