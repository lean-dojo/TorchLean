/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Program

/-!
# VQ-VAE training

The encoder produces a continuous latent, and a discrete assignment selects a codebook vector.
Their numerical distance appears twice in the objective, but the two terms train different
parameters. Codebook loss holds the encoder output fixed; commitment loss holds the embedding
fixed. Reconstruction uses a straight-through input so its gradient reaches the encoder and
decoder without updating the embedding.

These operations compose the ordinary `Ops` primitives, including `detach`, for eager and typed
graph execution. The caller supplies a code assignment and an encoder/decoder architecture.
Selection uses an explicit bounded index, as in `Generative.VQVAE`; this module does not
differentiate an argmin or choose a tie-breaking policy.
-/

@[expose] public section

namespace Runtime.Autograd.Model.VQVAE

open Spec TorchLean

variable {α : Type} [Storage α] [Context α]
variable {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
variable {latent obs : Shape}

/--
Select one embedding from a codebook with shape `numCodes :: latent`.

The index is non-differentiable data. Selection leaves the embedding connected to the codebook,
so the codebook-loss gradient is scattered back to this row by the ordinary selection primitive.
-/
def quantized {numCodes : Nat}
    (codebook : RefTy (m := m) (α := α) (latent.prependDim numCodes))
    (index : Fin numCodes) : m (RefTy (m := m) (α := α) latent) :=
  select (m := m) (α := α) (s := latent.prependDim numCodes) 0 codebook index

/--
Pass the selected embedding to the decoder with the encoder's straight-through gradient.

We compute `detach(selected) + (encoded - detach(encoded))`. For finite floating-point inputs,
subtracting the encoder value from itself avoids the cancellation in
`encoded + detach(selected - encoded)` when the two vectors have very different magnitudes.
The value is the selected embedding; the encoder cotangent is the incoming cotangent, and the
embedding cotangent is zero. These roles depend on the backend's stop-gradient semantics.
-/
def straightThrough (encoded selected : RefTy (m := m) (α := α) latent) :
    m (RefTy (m := m) (α := α) latent) := do
  let fixedCode ← detach selected
  let fixedEncoder ← detach encoded
  let encoderResidual ← sub encoded fixedEncoder
  add fixedCode encoderResidual

/--
Mean squared codebook loss, with the encoder output held fixed.

Only `selected` receives a gradient. If it came from `quantized`, the gradient updates the
selected row and leaves every other codebook row at zero.
-/
def codebookLoss (encoded selected : RefTy (m := m) (α := α) latent) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let fixedEncoder ← detach encoded
  mseLoss selected fixedEncoder

/--
Mean squared commitment loss, with the selected embedding held fixed.

Only `encoded` receives a gradient. The commitment weight is applied by `loss`, so callers can
inspect this unweighted term independently of the total objective.
-/
def commitmentLoss (encoded selected : RefTy (m := m) (α := α) latent) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let fixedCode ← detach selected
  mseLoss encoded fixedCode

/--
Decode the selected embedding using the straight-through gradient rule.

`decode` can close over trainable decoder parameters. Their reconstruction gradients follow the
ordinary decoder operations; the estimator only changes the encoder/codebook boundary.
-/
def forward
    (decode : RefTy (m := m) (α := α) latent → m (RefTy (m := m) (α := α) obs))
    (encoded selected : RefTy (m := m) (α := α) latent) :
    m (RefTy (m := m) (α := α) obs) := do
  decode (← straightThrough encoded selected)

/--
Mean squared reconstruction error through the straight-through decoder input.

The observation is held fixed as a target. Encoder and decoder parameters receive the
reconstruction gradient, while the selected codebook embedding receives none from this term.
-/
def reconstructionLoss
    (decode : RefTy (m := m) (α := α) latent → m (RefTy (m := m) (α := α) obs))
    (encoded selected : RefTy (m := m) (α := α) latent)
    (observation : RefTy (m := m) (α := α) obs) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let reconstruction ← forward decode encoded selected
  let target ← detach observation
  mseLoss reconstruction target

/--
Reconstruction loss plus codebook loss plus `beta` times commitment loss.

All three terms use mean reduction over their own shapes: observation coordinates for
reconstruction and latent coordinates for the two auxiliary terms. Pass the same live encoder
output and selected embedding to each term so their stop-gradient boundaries remain local to the
intended role. `beta` is a fixed scalar hyperparameter, usually nonnegative.
-/
def loss
    (decode : RefTy (m := m) (α := α) latent → m (RefTy (m := m) (α := α) obs))
    (encoded selected : RefTy (m := m) (α := α) latent)
    (observation : RefTy (m := m) (α := α) obs) (beta : α) :
    m (RefTy (m := m) (α := α) Shape.scalar) := do
  let reconstruction ← reconstructionLoss decode encoded selected observation
  let codebook ← codebookLoss encoded selected
  let commitment ← commitmentLoss encoded selected
  let weightedCommitment ← scale commitment beta
  add (← add reconstruction codebook) weightedCommitment

end Runtime.Autograd.Model.VQVAE
