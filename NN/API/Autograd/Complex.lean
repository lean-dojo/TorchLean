/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.State
public import NN.Runtime.Autograd.Model.Dual

/-!
# Real losses on complex parameters

An `Objective` returns a real scalar while retaining complex parameter tensors. Forward-mode
seeding differentiates both coordinates, including nonholomorphic operations such as conjugation
and squared magnitude. No complex-linear reverse rule is assumed. The gradient is represented as
`dL/dre + i*dL/dim`; multiplying it by a real learning rate gives the ordinary Euclidean update
on the two coordinates. In the convention with a factor of one half in Wirtinger derivatives,
this is twice the conjugate Wirtinger derivative.

`grad` evaluates two directional passes per complex parameter entry. This reference algorithm is
useful for small models and checking specialized backward implementations, not a claim of an
efficient large-model reverse pass. Objectives must be deterministic across those evaluations.
Scalar branch conventions still apply at nonsmooth points and complex branch cuts.
-/

@[expose] public section

namespace TorchLean.autograd.complex

open Runtime.Autograd.Model (Dual)

/-- A deterministic, real-valued objective on shape-indexed complex model state. -/
abbrev Objective (shapes : List Shape) :=
  ∀ {α : Type}, [Storage α] → [Context α] → [Atan2 α] →
    nn.State (Complex α) shapes → IO α

/--
Evaluate a real objective and its directional derivative in one forward-mode pass.

Both coordinates of `direction` are seeded independently; the tangent is the real scalar
pairing `sum (dL/dre * direction.re + dL/dim * direction.im)`.
-/
def jvp {shapes : List Shape} (objective : Objective shapes)
    {α : Type} [Storage α] [Context α] [Atan2 α]
    (state direction : nn.State (Complex α) shapes) : IO (α × α) := do
  let seeded : nn.State (Complex (Dual α)) shapes :=
    state.zipWith direction fun parameter tangent =>
      Tensor.map2Spec (fun z dz => ⟨⟨z.re, dz.re⟩, ⟨z.im, dz.im⟩⟩) parameter tangent
  let result ← objective seeded
  pure (result.re, result.du)

/--
Differentiate a real objective with respect to every complex parameter coordinate.

The returned state contains `dL/dre` and `dL/dim`, not a complex analytic derivative.
Set `value := true` to also return the real objective value. Empty tensors remain empty;
the value form still evaluates the objective when there are no parameters.
-/
def grad {shapes : List Shape} (objective : Objective shapes)
    {α : Type} [Storage α] [Context α] [Atan2 α]
    (state : nn.State (Complex α) shapes) (value : Bool := false) :
    IO (if value then nn.State (Complex α) shapes × α else nn.State (Complex α) shapes) := do
  let mut gradient : nn.State (Complex α) shapes := nn.State.zeros
  for tensorIndex in Array.finRange shapes.length do
    let shape := shapes.get tensorIndex
    let entries ← Tensor.generateFlatM shape fun index => do
      let basis : Tensor α shape :=
        (Tensor.oneHot (α := α) shape.size index).reshape shape (by simp [Shape.size])
      let realDirection : nn.State (Complex α) shapes :=
        nn.State.zeros.set tensorIndex (basis.map Complex.ofReal)
      let imagDirection : nn.State (Complex α) shapes :=
        nn.State.zeros.set tensorIndex (basis.map fun x => ⟨0, x⟩)
      let (_, realDerivative) ← jvp objective state realDirection
      let (_, imagDerivative) ← jvp objective state imagDirection
      pure (⟨realDerivative, imagDerivative⟩ : Complex α)
    gradient := gradient.set tensorIndex entries
  if h : value then
    pure (by simpa [h] using (gradient, ← objective state))
  else
    pure (by simpa [h] using gradient)

end TorchLean.autograd.complex
