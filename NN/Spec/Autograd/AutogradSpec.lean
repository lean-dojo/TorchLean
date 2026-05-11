/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core

/-!
# Spec-level autograd operation specifications (`Spec.OpSpec`)

This file defines a small interface for reverse-mode differentiation:

- an operation spec `OpSpec` is a *pure* `forward` map together with a *vector-Jacobian product*
  (VJP) `backward`;
- we provide sequential composition (`compose` / `>>>`) that implements the chain rule.

This is intentionally a spec-layer interface: it is independent of any particular runtime
autograd engine or tape representation. The runtime code is free to:

- cache intermediates (PyTorch-style `ctx.save_for_backward`), or
- recompute them, or
- compile to a graph,

as long as it implements the same mathematical VJP behavior.

PyTorch analogy:

- `OpSpec.forward` corresponds to the `forward(...)` method of a `torch.autograd.Function`.
- `OpSpec.backward` corresponds to the `backward(ctx, grad_output)` method, except that in TorchLean
  we pass the input `x` explicitly instead of a mutable `ctx`. At the spec level that is the same
  information: the derivative may depend on the forward inputs (and sometimes intermediate values).

In TorchLean we deliberately keep this file smaller than a graph IR: `OpSpec` is the math contract
(forward + VJP + composition). We do not want to invent yet another graph/IR here, because the repo
already has canonical graph representations:

- `NN/GraphSpec/*` (typed DAG structures used by graph-spec examples and some compilers),
- `NN/IR/*` (op-tagged IR used by verification tooling),
- `NN/Runtime/Autograd/*` (executable tape/graph engines).

When you want "a real graph", use those. When you want "the spec of an op", use `OpSpec`.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type}

/-- Atomic operation specification (forward + VJP/backward).

`backward` takes the input `x` and an upstream gradient `dL/dy`, and returns `dL/dx`.

Why this signature:

- Reverse-mode AD never needs a full Jacobian. What it needs is: given an upstream gradient
  `dL/dy`, compute the gradient with respect to the input, `dL/dx`. That’s exactly what `backward`
  encodes (a VJP).
- We pass `x` to `backward` because many derivatives depend on the input value. At the spec level we
  don’t force a “store intermediates vs recompute” strategy; the runtime system can choose.
-/
structure OpSpec (α : Type) (σ τ : Shape) where
  /-- forward. -/
  forward  : Tensor α σ → Tensor α τ
  /-- backward. -/
  backward : Tensor α σ → Tensor α τ → Tensor α σ

namespace OpSpec

/-- The identity `OpSpec` (forward is identity; backward returns the upstream gradient). -/
def id (α : Type) (σ : Shape) : OpSpec α σ σ :=
{ forward := fun x => x
, backward := fun _x dLdy => dLdy
}

/-- Sequential composition of two ops with the reverse-mode chain rule.

If `f : σ → τ` and `g : τ → υ`, their composition is `g ∘ f : σ → υ`.

For reverse-mode AD, we compose their VJPs:

- given an upstream gradient `dL/dz : υ`,
- compute `dL/dy : τ` using `g.backward`,
- then compute `dL/dx : σ` using `f.backward`.

You can visualize the dataflow as a compact chain:

```
x --f.forward--> y --g.forward--> z
```

and the reverse pass as the same chain walked backwards:

```
dL/dz --g.backward--> dL/dy --f.backward--> dL/dx
```

This is the core of what PyTorch builds dynamically as a "backward graph" during the forward pass,
except here we keep it as an explicit, pure definition. A runtime engine can still choose whether
to cache the intermediate `y = f.forward x` or recompute it; the spec states the mathematical VJP.
-/
def compose {σ τ υ : Shape}
  (f : OpSpec α σ τ) (g : OpSpec α τ υ) : OpSpec α σ υ :=
{ forward := fun x => g.forward (f.forward x)
, backward := fun x dLdz =>
    let y := f.forward x
    let dLdy := g.backward y dLdz
    f.backward x dLdy
}

infixr:80 " >>> " => compose

end OpSpec
end Spec
