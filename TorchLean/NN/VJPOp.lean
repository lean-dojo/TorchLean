/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Datatype

/-!
# Spec-level autograd operation specifications (`TorchLean.VJPOp`)

A small interface for reverse-mode differentiation. An operation spec `VJPOp` is a *pure*
`forward` map together with a *vector-Jacobian product* (VJP) `backward`, and sequential
composition (`compose`) implements the chain rule.

The interface belongs to the spec layer and is independent of any runtime autograd engine or tape
representation. A runtime may cache intermediates (PyTorch-style `ctx.save_for_backward`),
recompute them or lower to a graph, as long as it implements the same mathematical VJP.

`VJPOp` stores the two functions without a proof that `backward` is the derivative of `forward`.
That claim needs separate theorems and domain hypotheses. The interface also admits selected
subgradients and floating-point rules where a classical derivative need not exist.

PyTorch analogy:

- `VJPOp.forward` corresponds to the `forward(...)` method of a `torch.autograd.Function`.
- `VJPOp.backward` corresponds to the `backward(ctx, grad_output)` method, and TorchLean passes
  the input `x` where PyTorch passes a mutable `ctx`. At the spec level that is the same
  information: the derivative may depend on the forward inputs (and sometimes intermediate values).

We keep this module smaller than a graph IR on purpose: `VJPOp` is the math contract
(forward + VJP + composition), and the canonical graph representations live elsewhere:

- `TorchLean/NN/IR/*` (graphs of registered operators, used by verification tooling),
- `TorchLean.NN.Tape` and `TorchLean.NN.TypedGraph` (executable tape/graph engines).

When you want "a real graph", use those. When you want "the spec of an op", use `VJPOp`.

We wrote `VJPOp` and `compose` by hand. `backward` is what Baydin et al. (2018) call the transposed
Jacobian-vector product `J⊺ r` in Section 3.2, the reverse phase started from the upstream gradient
`r`, and `compose` carries these adjoints from the output back to the input as that reverse phase
does.

References:
- Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul and Jeffrey Mark Siskind,
  "Automatic Differentiation in Machine Learning: a Survey", JMLR 18(153):1-43, 2018
  (`arXiv:1502.05767`), Section 3.2
- "Extending PyTorch" (PyTorch 2.8 docs), https://docs.pytorch.org/docs/2.8/notes/extending.html
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α]

/-- Atomic operation specification (forward + VJP/backward).

`backward` takes the input `x` and an upstream gradient `dL/dy` and returns `dL/dx`, with no
differentiability or derivative-correctness law attached. Reverse-mode AD never needs the full
Jacobian, only this vector-Jacobian product. The input is an argument because many derivatives
depend on it, and the spec leaves the choice between storing intermediates and recomputing them
to the runtime.
-/
structure VJPOp (α : Type) [Storage α] (σ τ : Shape) where
  /-- The forward map from an input of shape `σ` to an output of shape `τ`. -/
  forward  : Tensor α σ → Tensor α τ
  /-- The vector-Jacobian product: given the input `x` and the upstream gradient `dL/dy`,
  return `dL/dx`. -/
  backward : Tensor α σ → Tensor α τ → Tensor α σ

namespace VJPOp

/-- The identity `VJPOp`: forward returns the input and backward the upstream gradient. -/
def id (α : Type) [Storage α] (σ : Shape) : VJPOp α σ σ :=
{ forward := fun x => x
, backward := fun _x dLdy => dLdy
}

/-- Sequential composition of two ops with the reverse-mode chain rule.

If `f : σ → τ` and `g : τ → υ`, their composition is `g ∘ f : σ → υ`. The VJPs compose in the
opposite order: given an upstream gradient `dL/dz : υ`, `g.backward` computes `dL/dy : τ` and then
`f.backward` computes `dL/dx : σ`. The forward dataflow is the chain

```
x --f.forward--> y --g.forward--> z
```

and the reverse pass walks the same chain backwards:

```
dL/dz --g.backward--> dL/dy --f.backward--> dL/dx
```

PyTorch builds this chain dynamically as a "backward graph" during the forward pass, and here it
is one pure definition. The spec's `backward` recomputes `y = f.forward x`, and a runtime engine
may cache that intermediate as long as it returns the same VJP.
-/
def compose {σ τ υ : Shape}
  (f : VJPOp α σ τ) (g : VJPOp α τ υ) : VJPOp α σ υ :=
{ forward := fun x => g.forward (f.forward x)
, backward := fun x dLdz =>
    let y := f.forward x
    let dLdy := g.backward y dLdz
    f.backward x dLdy
}

end VJPOp
end TorchLean
