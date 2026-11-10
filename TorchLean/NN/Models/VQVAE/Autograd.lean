/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Program

/-!
# VQ-VAE training

The encoder produces a continuous latent, and a discrete assignment selects a codebook vector.
Their mean squared distance appears twice in the objective, and the two terms train different
parameters: codebook loss holds the encoder output fixed and trains the embedding, while commitment
loss holds the embedding fixed and trains the encoder. Reconstruction feeds the decoder a
straight-through input, so its gradient reaches the encoder and the decoder and leaves the
embedding alone.

Every operation composes the ordinary `Ops` primitives, `detach` among them, so the same code runs
eagerly and as a typed graph. The caller supplies the code assignment and the encoder and decoder.
Selection takes a bounded index, as in `VQVAE`: the caller computes the nearest code
and breaks ties, and no argmin is differentiated.

Section 3.2 of van den Oord et al. (2017) copies the gradient at the decoder input `z_q(x)` to the
encoder output `z_e(x)`, and we wrote the estimator by hand from that rule. Their stop-gradient
`sg` in eq. (3) is `detach` here. No theorem ties this program to `VQVAE.straightThroughGrads`,
and no TorchLeanTest test runs it.

Reference: Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu (2017), "Neural Discrete
Representation Learning", NIPS 2017, arXiv:1711.00937, eq. (3) and Section 3.2.
-/

@[expose] public section

namespace TorchLean.VQVAE.Autograd

open Ops

variable {α : Type} [Storage α] [Context α]
variable {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
variable {latent obs : Shape}

/--
Select row `index` of a codebook with shape `numCodes :: latent`.

The index is non-differentiable data. The selected embedding stays connected to the codebook, so
the backward pass of the selection primitive scatters the codebook-loss gradient back to that row.
-/
def quantized {numCodes : Nat}
    (codebook : Ref (m := m) (α := α) (Shape.dim numCodes latent))
    (index : Fin numCodes) : m (Ref (m := m) (α := α) latent) :=
  Ops.apply (m := m) (α := α)
      (.indexAt (s := Shape.dim numCodes latent) ⟨0, Shape.AxisInBounds.proof⟩ index)
      (codebook, ())

/--
Pass the selected embedding to the decoder with the encoder's straight-through gradient.

The result is `detach(selected) + (encoded - detach(encoded))`, which is `selected` over the reals.
In floating point the residual is an exact zero for finite `encoded`, so the result can differ
from `selected` only in the sign of a zero, and a nonfinite encoder value makes the residual NaN.
The alternative `encoded + detach(selected - encoded)` loses the low bits of `selected` when the
two vectors have very different magnitudes. Given the backend's stop-gradient semantics, the
encoder receives the incoming cotangent and the embedding receives zero.
-/
def straightThrough (encoded selected : Ref (m := m) (α := α) latent) :
    m (Ref (m := m) (α := α) latent) := do
  let fixedCode ← Ops.apply .detach (selected, ())
  let fixedEncoder ← Ops.apply .detach (encoded, ())
  let encoderResidual ← Ops.apply .sub (encoded, fixedEncoder, ())
  Ops.apply .add (fixedCode, encoderResidual, ())

/--
Mean squared codebook loss, with the encoder output held fixed.

Only `selected` receives a gradient. If it came from `quantized`, the gradient reaches the
selected codebook row and is zero on every other row.
-/
def codebookLoss (encoded selected : Ref (m := m) (α := α) latent) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  let fixedEncoder ← Ops.apply .detach (encoded, ())
  Ops.apply .mseLoss (selected, fixedEncoder, ())

/--
Mean squared commitment loss, with the selected embedding held fixed.

Only `encoded` receives a gradient. `loss` applies the commitment weight, so this term is
unweighted and can be inspected on its own.
-/
def commitmentLoss (encoded selected : Ref (m := m) (α := α) latent) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  let fixedCode ← Ops.apply .detach (selected, ())
  Ops.apply .mseLoss (encoded, fixedCode, ())

/--
Decode the straight-through input built from `encoded` and `selected`.

`decode` can close over trainable decoder parameters, whose reconstruction gradients follow the
ordinary decoder operations. The estimator changes only the boundary between encoder and codebook.
-/
def forward
    (decode : Ref (m := m) (α := α) latent → m (Ref (m := m) (α := α) obs))
    (encoded selected : Ref (m := m) (α := α) latent) :
    m (Ref (m := m) (α := α) obs) := do
  decode (← straightThrough encoded selected)

/--
Mean squared reconstruction error through the straight-through decoder input.

The observation is held fixed as a target. Encoder and decoder parameters receive the
reconstruction gradient, while the selected codebook embedding receives none from this term.
-/
def reconstructionLoss
    (decode : Ref (m := m) (α := α) latent → m (Ref (m := m) (α := α) obs))
    (encoded selected : Ref (m := m) (α := α) latent)
    (observation : Ref (m := m) (α := α) obs) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  let reconstruction ← forward decode encoded selected
  let target ← Ops.apply .detach (observation, ())
  Ops.apply .mseLoss (reconstruction, target, ())

/--
Reconstruction loss plus codebook loss plus `beta` times commitment loss.

Each term is a mean over its own shape: observation coordinates for reconstruction and latent
coordinates for the two auxiliary terms. Pass the live encoder output and selected embedding,
because each term detaches what its role requires. `beta` is a fixed scalar hyperparameter,
usually nonnegative.
-/
def loss
    (decode : Ref (m := m) (α := α) latent → m (Ref (m := m) (α := α) obs))
    (encoded selected : Ref (m := m) (α := α) latent)
    (observation : Ref (m := m) (α := α) obs) (beta : α) :
    m (Ref (m := m) (α := α) Shape.scalar) := do
  let reconstruction ← reconstructionLoss decode encoded selected observation
  let codebook ← codebookLoss encoded selected
  let commitment ← commitmentLoss encoded selected
  let weightedCommitment ← Ops.apply .scale
      (commitment, .data (Ops.dataConst (Tensor.scalar beta)), ())
  Ops.apply .add
      ((← Ops.apply .add (reconstruction, codebook, ())), weightedCommitment,
        ())

end TorchLean.VQVAE.Autograd
