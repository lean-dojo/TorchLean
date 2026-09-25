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

## Recurrent stacks

`Spec.RecurrentStack Cell inputWidth outputWidth` joins cells with matching intermediate widths.
`.nil` has equal input and output widths. `.cons first rest` places one cell before the remaining
stack. For cells `2 → 3` and `3 → 1`, the stack is `.cons first (.cons second .nil)`.

`Spec.Rnn.stacked`, `Spec.Gru.stacked`, and `Spec.Lstm.stacked` take a stack and a final linear
layer, producing a `Spec.Module.Chain` over the sequence. The corresponding `StackedModel`
structures expose `layers` and `outputLayer`. Their `forward` methods accept explicit initial
states and return outputs together with final states.

The state type follows the chosen widths. RNN and GRU states for the `2 → 3 → 1` example are
`(h₃, (h₁, ()))`, where `h₃ : Tensor α [3]` and `h₁ : Tensor α [1]`. LSTM uses a
`Spec.LSTMState α width` at each position. An empty stack has state `()`, and an empty sequence
preserves the supplied states. These explicit-state reference models are separate from the
executable model builders, which initialize their recurrent states for each call.
