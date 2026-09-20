# `NN.Spec.Module`

`Spec.Module α σ τ` packages a pure tensor function from shape `σ` to shape `τ`. Its shapes
are part of the type, so Lean rejects a composition whose intermediate shapes do not agree.

```lean
def block : Spec.Module Float input output :=
  Spec.Module.Chain.single first
    |>.append second
```

`Spec.Module.Chain` evaluates modules from left to right. The `kind` and `pythonExpr` fields support
reports and Python source export; only `forward` determines the mathematical meaning.

The directory contains:

- `Core.lean`: the module type, typed chains, leading-dimension mapping, and selection;
- layer adapters for activations, linear maps, convolution, pooling, normalization, attention,
  embeddings, dropout, and positional encoding;
- recurrent compositions for RNNs, GRUs, and LSTMs;
- adapters for autoencoders, sequence-to-sequence models, graph networks, classical models, and
  probabilistic models.

The underlying formulas remain in `NN.Spec.Layers` and `NN.Spec.Models`. These adapters provide one
typed composition interface without duplicating those semantics.
