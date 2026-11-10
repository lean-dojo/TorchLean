/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN
public import TorchLean.Tensor.Layers.Activation

/-!
# Runtime CROWN operators

Executable helper operators for the graph-based CROWN/IBP engine: ReLU relaxations, their affine
propagation, and interval rules for a few scalar activations.

The module keeps runtime certificate replay apart from proof imports. Some proof-layer CROWN modules
import `Mathlib` and large theorem developments, and native executables that only replay
certificates should not pay that import cost. The graph verifier and executable certificate checks
need only this compact set of computational definitions.

The definitions live under `CROWN`, with no direct Mathlib dependency.
Proof modules cite them, and the module itself holds only the runtime support code for fast
certificate replay.

References (bound propagation background):
- Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions" (CROWN), NeurIPS 2018: https://arxiv.org/abs/1811.00866, Sec. 3.2.
- Singh, Gehr, Püschel and Vechev, "An Abstract Domain for Certifying Neural Networks"
  (DeepPoly), Proc. ACM Program. Lang. 3 (POPL), 2019, pp. 1-30, doi:10.1145/3290354.
- Gowal et al., "On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust
  Models" (IBP), 2018: https://arxiv.org/abs/1810.12715, Sec. 3, Eq. (7).
-/

@[expose] public section


open FloatLib.Numerics (MathFunctions)

namespace TorchLean.CROWN

open Tensor

variable {α : Type} [Storage α] [Context α]

/-- Parameters of a per-neuron affine relaxation `y = slope * x + bias` used in CROWN/DeepPoly. -/
structure ReLURelax (α : Type) where
  /-- Linear coefficient. -/
  slope : α
  /-- Constant offset. -/
  bias : α

namespace ReLU

/--
The triangle upper relaxation of ReLU on `[l, u]`: when `0 < u`, slope `1` and bias `0` if
`0 < l` and otherwise the chord through `(l, 0)` and `(u, u)`, with slope `s = u / (u - l)` and
bias `-s * l`, and slope and bias `0` when `0 < u` fails. Over `ℝ` the line bounds `relu x` from
above for `l ≤ x ≤ u` (`TorchLean.CROWN.ReLU.relaxScalar_sound`), and in
binary32 it need not: on `[-M, M]`, with `M` the largest finite value, `u - l` overflows to `+∞`
and the line is `0`, below `relu M = M`. The rule is CROWN's ReLU case (Zhang et al., Sec. 3.2),
written by hand. At `l = 0` the strict test `l > 0` fails and the rule takes the chord, whose slope
`u / u = 1` and bias `0` give CROWN's identity line over `ℝ`.
-/
def relaxScalar (l u : α) : ReLURelax α :=
  if u > 0 then
    if l > 0 then
      { slope := 1, bias := 0 }
    else
      let denom := (u - l)
      let αs := u / denom
      let β := -αs * l
      { slope := αs, bias := β }
  else
    { slope := 0, bias := 0 }

/-- `relaxScalar` applied to each coordinate of the vector bounds `lo` and `hi`. -/
def relaxVector {n : Nat} (lo hi : Tensor α [n]) :
    Tensor (ReLURelax α) [n] :=
  Tensor.dim (fun i =>
    Tensor.scalar (relaxScalar (lo.getScalar i) (hi.getScalar i)))

/--
Propagation of an affine form through ReLU with per-neuron relaxations.

From `y ≈ A*x + c` and per-output relaxations `(slopeᵢ, biasᵢ)` it builds the affine form
`y' ≈ diag(slope) * (A*x + c) + bias`.
-/
def propagateAffine {inDim hidDim : Nat}
    (relax : Tensor (ReLURelax α) [hidDim])
    (aff : Linear α inDim hidDim) : Linear α inDim hidDim :=
  let A' := Tensor.dim (fun i =>
    let rp := relax.getScalar i
    Tensor.dim (fun j => Tensor.scalar (get2 aff.weights i j * rp.slope)))
  let c' := Tensor.dim (fun i =>
    let rp := relax.getScalar i
    Tensor.scalar (rp.slope * aff.bias.getScalar i + rp.bias))
  { weights := A', bias := c' }

end ReLU

namespace IBP

/-!
The certificate rules of `TorchLean.CROWN.Op` (`Op.cert?`) use the endpoint-minmax and
clipped-midpoint rules below. Swapping in the engine's scalar enclosure rules (`NonlinearBoundOps`)
would change the exact `CertLocalOK` predicate, including for real-valued certificates (reversed
endpoints for tanh and sigmoid, `[0,0]` for sine and cosine).
-/

/-- Elementwise interval propagation for a monotone activation `f`: each output entry runs from the
smaller to the larger of `f l` and `f u`. Gowal et al. (2018, Eq. (7)) send an increasing
activation through the two endpoints, and comparing `f l` with `f u` handles a decreasing `f` as
well, though `IBP.mapMinmax_sound_real` assumes `Monotone f`. -/
def mapMinmax {s : Shape} (f : α → α) (xB : Box α s) : Box α s :=
  let outLo := Tensor.zipWith (fun l u =>
    let fl := f l
    let fu := f u
    if fl > fu then fu else fl) xB.lo xB.hi
  let outHi := Tensor.zipWith (fun l u =>
    let fl := f l
    let fu := f u
    if fl > fu then fl else fu) xB.lo xB.hi
  { lo := outLo, hi := outHi }

/-- Interval bound propagation for `sigmoid` (monotone, so min/max of endpoints). -/
def sigmoidMinmax {s : Shape} (xB : Box α s) : Box α s :=
  mapMinmax Context.sigmoid xB

/-- Interval bound propagation for `tanh` (monotone, so min/max of endpoints). -/
def tanhMinmax {s : Shape} (xB : Box α s) : Box α s :=
  mapMinmax MathFunctions.tanh xB

/-- The midpoint-radius enclosure clipped to `[-1, 1]` that sine and cosine share.

It is sound under exact ordered arithmetic for a 1-Lipschitz function with values in `[-1, 1]`. The
graph engine uses its own rounded scalar enclosures. -/
def midpoint {s : Shape} (f : α → α) (xB : Box α s) : Box α s :=
  { lo := Tensor.zipWith (fun l u => max (-1) (f ((l + u) / 2) - (u - l) / 2))
      xB.lo xB.hi
    hi := Tensor.zipWith (fun l u => min 1 (f ((l + u) / 2) + (u - l) / 2))
      xB.lo xB.hi }

/--
IBP for `sin` using a 1-Lipschitz enclosure:

$$
\sin([l,u])\subseteq[\sin(m)-r,\sin(m)+r]\cap[-1,1],
\qquad m=\frac{l+u}{2},\quad r=\frac{u-l}{2}.
$$

Its real-arithmetic soundness theorem, `Graph.sinMidpoint_sound_real`, needs no periodic case
split. The rule rounds nothing outward, so the graph engine's native-float path uses the scalar
backend's enclosure rules. We picked this rule ourselves to avoid that case split, and it costs
width: on `[0, π]` it gives `[1 - π/2, 1]`, about `[-0.571, 1]`, around the image `[0, 1]`.
-/
def sinMidpoint {s : Shape} (xB : Box α s) : Box α s :=
  midpoint MathFunctions.sin xB

/-- The 1-Lipschitz enclosure of `IBP.sinMidpoint`, applied to `cos`. -/
def cosMidpoint {s : Shape} (xB : Box α s) : Box α s :=
  midpoint MathFunctions.cos xB

end IBP

end TorchLean.CROWN
