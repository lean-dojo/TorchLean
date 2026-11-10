/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Pack
public import TorchLean.Tensor.Constructors
public import TorchLean.NN.Dual

/-!
# Real losses on complex parameters

An `Objective` returns a real scalar while retaining complex parameter tensors. Forward-mode
seeding differentiates both coordinates, including nonholomorphic operations such as conjugation
and squared magnitude. No complex-linear reverse rule is assumed. The gradient is represented as
`dL/dre + i*dL/dim`, so multiplying it by a real learning rate gives the ordinary Euclidean update
on the two coordinates. In the convention with a factor of one half in Wirtinger derivatives,
this is twice the conjugate Wirtinger derivative.

`autograd.complex.jvp` computes one directional derivative, and `autograd.complex.grad` evaluates
two directional passes per complex parameter entry. It is a reference algorithm, suited to small
models and to checking specialized backward implementations. Objectives must be deterministic
across those evaluations, and at nonsmooth points and on complex branch cuts the scalar branch
conventions apply.
-/

@[expose] public section

namespace TorchLean.autograd.complex

open TorchLean (Dual)

/-- A deterministic, real-valued objective on shape-indexed complex model state. -/
abbrev Objective (shapes : List Shape) :=
  ∀ {α : Type}, [Storage α] → [Context α] → [InverseTrig α] →
    TensorPack (Complex α) shapes → IO α

/--
Evaluate a real objective and its directional derivative in one forward-mode pass.

Both coordinates of `direction` are seeded independently, and the tangent is the real scalar
pairing `sum (dL/dre * direction.re + dL/dim * direction.im)`. The result is the pair
`(value, tangent)`, as from PyTorch's `torch.func.jvp`.
-/
def jvp {shapes : List Shape} (objective : Objective shapes)
    {α : Type} [Storage α] [Context α] [InverseTrig α]
    (state direction : TensorPack (Complex α) shapes) : IO (α × α) := do
  let seeded : TensorPack (Complex (Dual α)) shapes :=
    state.zipWith (fun parameter tangent =>
      Tensor.zipWith (fun z dz => ⟨⟨z.re, dz.re⟩, ⟨z.im, dz.im⟩⟩) parameter tangent) direction
  let result ← objective seeded
  pure (result.re, result.du)

/--
Differentiate a real objective with respect to every complex parameter coordinate.

Each entry of the returned state has `dL/dre` as its real part and `dL/dim` as its imaginary part,
the convention described in the module doc. The cost is two `jvp` passes per parameter entry.
Set `value := true` to also return the real objective value. Empty tensors stay empty, and the
value form evaluates the objective even when there are no parameters.
-/
def grad {shapes : List Shape} (objective : Objective shapes)
    {α : Type} [Storage α] [Context α] [InverseTrig α]
    (state : TensorPack (Complex α) shapes) (value : Bool := false) :
    IO (if value then TensorPack (Complex α) shapes × α else TensorPack (Complex α) shapes) := do
  let mut gradient : TensorPack (Complex α) shapes := .zero
  for tensorIndex in Array.finRange shapes.length do
    let shape := shapes.get tensorIndex
    let entries ← Tensor.generateFlatM shape fun index => do
      let basis : Tensor α shape :=
        (Tensor.oneHot (α := α) shape.size index).reshape shape (by simp)
      let realDirection : TensorPack (Complex α) shapes :=
        TensorPack.zero.set tensorIndex (basis.map Complex.ofReal)
      let imagDirection : TensorPack (Complex α) shapes :=
        TensorPack.zero.set tensorIndex (basis.map fun x => ⟨0, x⟩)
      let (_, realDerivative) ← jvp objective state realDirection
      let (_, imagDerivative) ← jvp objective state imagDirection
      pure (⟨realDerivative, imagDerivative⟩ : Complex α)
    gradient := gradient.set tensorIndex entries
  if h : value then
    pure (by simpa [h] using (gradient, ← objective state))
  else
    pure (by simpa [h] using gradient)

end TorchLean.autograd.complex
