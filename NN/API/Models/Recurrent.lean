/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Recurrent Models

RNN, GRU, and LSTM sequence models with a linear projection at every time step.
-/

@[expose] public section

namespace TorchLean


open Spec TorchLean TorchLean.Tensor

namespace nn
namespace models

/--
Configuration for an RNN, GRU, or LSTM followed by a time-distributed linear head.

The model consumes a fixed-length sequence. Constructors accept any `batchShape` for batches,
ensembles, or other pointwise collections of sequences.
-/
structure Recurrent.Config where
  /-- Number of time steps. -/
  sequenceLength : Nat
  /-- Number of features presented at each time step. -/
  inputWidth : Nat
  /-- Width of the recurrent state. -/
  hiddenWidth : Nat
  /-- Number of features produced at each time step. -/
  outputWidth : Nat
deriving Repr

namespace Recurrent.Internal

/--
Dimension checks shared by the recurrent model family.

`kind` is the label that appears in the error message, so a failure names the model the user asked
for rather than this helper. Keeping the checks here is why `Recurrent.Config.validate` and the
sequence-to-sequence variants cannot drift into reporting different messages for the same mistake.
-/
def validateConfig (kind : String) (config : Recurrent.Config) : Except String Unit := do
  Runtime.Autograd.Model.Layers.Internal.validateRecurrentDimensions
    kind config.sequenceLength config.inputWidth config.hiddenWidth
  if config.outputWidth = 0 then
    throw s!"{kind}: output width must be positive"

end Recurrent.Internal

namespace Recurrent.Config

/-- Validate the complete recurrent model before allocating core or projection parameters. -/
def validate (config : Recurrent.Config) : Except String Unit :=
  Recurrent.Internal.validateConfig "Recurrent" config

end Recurrent.Config

/-- Input tensor shape: `batchShape × sequenceLength × inputWidth`. -/
abbrev Recurrent.Config.input (config : Recurrent.Config)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat [config.sequenceLength, config.inputWidth]

/-- Output tensor shape: `batchShape × sequenceLength × outputWidth`. -/
abbrev Recurrent.Config.output (config : Recurrent.Config)
    (batchShape : Shape := []) : Shape :=
  batchShape.concat [config.sequenceLength, config.outputWidth]

/--
Vanilla RNN core plus time-distributed linear head:

`rnn(sequenceLength, inputWidth, hiddenWidth) → linear(hiddenWidth, outputWidth)`.
-/
def rnn (config : Recurrent.Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.input batchShape) (config.output batchShape)) := by
  match Recurrent.Internal.validateConfig "RNN" config with
  | .error message =>
      exact pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.output batchShape) "RNN" message
  | .ok () =>
      have model := do
        let recurrent ←
          nn.rnn config.sequenceLength config.inputWidth config.hiddenWidth
            (batchShape := batchShape)
        let outputProjection ← linear config.hiddenWidth config.outputWidth
          (batchShape := batchShape.appendDim config.sequenceLength)
        pure (recurrent >>> outputProjection)
      simpa only [Recurrent.Config.input, Recurrent.Config.output,
        Shape.appendDim_appendDim_eq_concat] using model

/--
Gated recurrent unit plus time-distributed linear head:

`gru(sequenceLength, inputWidth, hiddenWidth) → linear(hiddenWidth, outputWidth)`.

By default the core uses the Cho-style reset-before convention. With reset gate $r_t$, update
gate $z_t$ and previous hidden state $h_{t-1}$, its candidate and update are

$$
n_t = \tanh\!\left(W_{nx}x_t + W_{nh}(r_t \odot h_{t-1}) + b_n\right),
\qquad h_t = (1-z_t)\odot n_t + z_t\odot h_{t-1}.
$$

Choose `convention := .resetAfter` for PyTorch's candidate equation:

$$
n_t = \tanh\!\left(W_{nx}x_t + b_{nx} + r_t\odot(W_{nh}h_{t-1} + b_{nh})\right).
$$

Here the reset acts after the recurrent affine map, including its bias. Moving the reset across
a general recurrent matrix changes the function, and the reset-dependent bias cannot generally
be folded into a constant bias. Repacking PyTorch weights alone therefore does not preserve a
general PyTorch GRU's behavior. The reset-after constructor implements this different recurrence
directly and stores `weight_ih, weight_hh, bias_ih, bias_hh` in PyTorch's packed gate order.

The reset-before core stores a matrix and one bias for each gate, in reset, update, candidate order.
Each matrix has shape `[hiddenWidth, inputWidth + hiddenWidth]`, with input columns followed by
hidden columns.
PyTorch stores separate input and recurrent matrices and biases, each packed in the same gate
order. Its reset and update bias pairs can be added for forward evaluation; the candidate differs
by the reset placement above.

Input and output shapes are `batchShape ++ [sequenceLength, inputWidth]` and
`batchShape ++ [sequenceLength, outputWidth]`. The core weights are shared across batch entries,
and the same linear head projects every hidden state. Each call starts every sequence at zero
hidden state and returns all projected time steps. There is no initial-state argument or separate
final-state result, and hidden state is not carried between calls.
-/
def gru (config : Recurrent.Config) (batchShape : Shape := [])
    (convention : Spec.GRUConvention := .resetBefore) :
    nn.Builder (nn.Sequential (config.input batchShape) (config.output batchShape)) := by
  match Recurrent.Internal.validateConfig "GRU" config with
  | .error message =>
      exact pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.output batchShape) "GRU" message
  | .ok () =>
      have model := do
        let recurrent ←
          nn.gru config.sequenceLength config.inputWidth config.hiddenWidth
            (batchShape := batchShape) (convention := convention)
        let outputProjection ← linear config.hiddenWidth config.outputWidth
          (batchShape := batchShape.appendDim config.sequenceLength)
        pure (recurrent >>> outputProjection)
      simpa only [Recurrent.Config.input, Recurrent.Config.output,
        Shape.appendDim_appendDim_eq_concat] using model

/--
LSTM core plus time-distributed linear head:

`lstm(sequenceLength, inputWidth, hiddenWidth) → linear(hiddenWidth, outputWidth)`.
-/
def lstm (config : Recurrent.Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.input batchShape) (config.output batchShape)) := by
  match Recurrent.Internal.validateConfig "LSTM" config with
  | .error message =>
      exact pure <| nn.Internal.invalidConfiguration
        (config.input batchShape) (config.output batchShape) "LSTM" message
  | .ok () =>
      have model := do
        let recurrent ←
          nn.lstm config.sequenceLength config.inputWidth config.hiddenWidth
            (batchShape := batchShape)
        let outputProjection ← linear config.hiddenWidth config.outputWidth
          (batchShape := batchShape.appendDim config.sequenceLength)
        pure (recurrent >>> outputProjection)
      simpa only [Recurrent.Config.input, Recurrent.Config.output,
        Shape.appendDim_appendDim_eq_concat] using model

end models
end nn

end TorchLean
