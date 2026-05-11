# `NN.Spec.Autograd`

This directory defines TorchLean's spec level interface for reverse mode automatic differentiation.
The intent is to keep the math small and explicit:

- an `OpSpec` is forward plus VJP (vector Jacobian product),
- composition of `OpSpec`s is the chain rule.

This layer is deliberately independent of any particular runtime tape/graph engine. It is the
contract that runtime code should implement (or reuse) so the executable system stays aligned with
the spec.

PyTorch analogy:

- `OpSpec.forward` is like `torch.autograd.Function.forward`.
- `OpSpec.backward` is like `torch.autograd.Function.backward`, but expressed as a pure VJP that
  receives the input explicitly (instead of a mutable `ctx`).

Files:

- `autograd_spec.lean`: defines `Spec.OpSpec` and sequential composition (`compose` / `>>>`). This
  is the small interface for reverse mode differentiation.
- `ops.lean`: a small library of common `OpSpec`s built from the spec layer tensor primitives
  (activations, pointwise math ops, reductions, broadcasting aware wrappers, and loss functions).

Where to look for graphs and execution:

- `NN/Runtime/Autograd/*`: executable tape/graph engines and training utilities.
- `NN/GraphSpec/*` and `NN/IR/*`: typed DAG / IR representations used by compilation and verification.
- `NN/Proofs/Autograd/*`: correctness statements that relate runtime execution to spec level math.

Synchronization note: define the math once (in `NN/Spec/*`) and have runtime code call those
helpers when possible. That keeps runtime rules and spec rules tied to the same vocabulary.
