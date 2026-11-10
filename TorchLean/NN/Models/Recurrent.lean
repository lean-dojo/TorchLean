/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Builder

/-!
# Recurrent Models

`rnn`, `gru` and `lstm` build recurrent stacks from a `Recurrent.Config`, with independently chosen
hidden widths and a linear projection at every time step. An empty stack is a time-distributed
linear model and accepts empty sequences.

The cells come from `TorchLean.NN.Layer.Recurrent`, written by hand. The reset-before GRU is
eqs. (5) to (8) of Cho et al. (2014) with a bias added to each gate, since their equations carry
none, and reset-after is the `n_t` of `torch.nn.GRU`. `rnn` is the tanh recurrence of
`torch.nn.RNN` with a single bias where torch adds `b_ih` and `b_hh`. `lstm` has the gate
equations of `torch.nn.LSTM`, stored forget, input, candidate, output, where torch packs
`W_ii|W_if|W_ig|W_io`, and again one bias per gate. No TorchLeanTest test runs these models
against PyTorch.

References:
- Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger
  Schwenk, and Yoshua Bengio (2014), "Learning Phrase Representations using RNN Encoder-Decoder
  for Statistical Machine Translation", EMNLP 2014, pages 1724-1734, doi:10.3115/v1/D14-1179,
  eqs. (5) to (8).
- `torch.nn.RNN`, `torch.nn.GRU` and `torch.nn.LSTM` (PyTorch 2.8 docs).
-/

@[expose] public section

open scoped TorchLean.nn.Sequential

namespace TorchLean


open TorchLean.Tensor

namespace nn

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
  /-- Recurrent widths in execution order. An empty list builds only the output head. -/
  hiddenWidths : List Nat := []
  /-- Number of features produced at each time step. -/
  outputWidth : Nat
deriving Repr

namespace Recurrent

/--
Dimension checks shared by the recurrent model family.

`kind` labels the error message, so a failure names the model the user asked for rather than this
helper. `Recurrent.Config.validate` and `Recurrent.build` both call it.
-/
def validateConfig (kind : String) (config : Recurrent.Config) : Except String Unit := do
  match config.hiddenWidths with
  | [] =>
      Layer.requirePositive kind "input width" config.inputWidth
  | _ :: _ =>
      let mut inputWidth := config.inputWidth
      for hiddenWidth in config.hiddenWidths do
        Layer.validateRecurrent
          kind config.sequenceLength inputWidth hiddenWidth
        inputWidth := hiddenWidth
  if config.outputWidth = 0 then
    throw s!"{kind}: output width must be positive"

end Recurrent

namespace Recurrent.Config

/-- Validate the complete recurrent model before allocating core or projection parameters. -/
def validate (config : Recurrent.Config) : Except String Unit :=
  Recurrent.validateConfig "Recurrent" config

end Recurrent.Config

/-- Input tensor shape: `batchShape × sequenceLength × inputWidth`. -/
abbrev Recurrent.Config.inputShape (config : Recurrent.Config)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.sequenceLength, config.inputWidth]

/-- Output tensor shape: `batchShape × sequenceLength × outputWidth`. -/
abbrev Recurrent.Config.outputShape (config : Recurrent.Config)
    (batchShape : Shape := []) : Shape :=
  batchShape ++ [config.sequenceLength, config.outputWidth]

namespace Recurrent

/-- Stack one `core` layer per entry of `hiddenWidths`, from `inputWidth` features, then a linear
head with `outputWidth` features at every time step. Empty `hiddenWidths` gives only the head. -/
def stack (batchShape : Shape) (sequenceLength outputWidth : Nat)
    (core : (inputWidth hiddenWidth : Nat) →
      Builder (Sequential (batchShape ++ [sequenceLength, inputWidth])
        (batchShape ++ [sequenceLength, hiddenWidth]))) :
    (inputWidth : Nat) → (hiddenWidths : List Nat) →
      Builder (Sequential (batchShape ++ [sequenceLength, inputWidth])
        (batchShape ++ [sequenceLength, outputWidth]))
  | inputWidth, [] => by
      simpa only [← List.append_cons] using
        (linear inputWidth outputWidth (batchShape := batchShape ++ [sequenceLength]))
  | inputWidth, width :: rest => do
      let layer ← core inputWidth width
      let remaining ← stack batchShape sequenceLength outputWidth core width rest
      pure (layer >>> remaining)

/-- Compose the recurrent layers in order, then apply the same linear head at every time step. -/
def build (kind : String) (config : Recurrent.Config) (batchShape : Shape)
    (core : (inputWidth hiddenWidth : Nat) →
      Builder (Sequential (batchShape ++ [config.sequenceLength, inputWidth])
        (batchShape ++ [config.sequenceLength, hiddenWidth]))) :
    Builder (Sequential (config.inputShape batchShape) (config.outputShape batchShape)) := do
  if let .error message := validateConfig kind config then
    return nn.Sequential.invalid
      (config.inputShape batchShape) (config.outputShape batchShape) kind message
  stack batchShape config.sequenceLength config.outputWidth core config.inputWidth
    config.hiddenWidths

end Recurrent

/--
Vanilla RNN layers followed by a time-distributed linear head.
-/
def Recurrent.rnn (config : Recurrent.Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  Recurrent.build "RNN" config batchShape fun inputWidth hiddenWidth => by
    simpa only [← List.append_cons] using
      (nn.rnn config.sequenceLength inputWidth hiddenWidth (batchShape := batchShape))

/--
Gated recurrent layers followed by a time-distributed linear head.

By default each core uses the Cho-style reset-before convention. With reset gate $r_t$, update
gate $z_t$ and previous hidden state $h_{t-1}$, its candidate and update are

$$
n_t = \tanh\!\left(W_{nx}x_t + W_{nh}(r_t \odot h_{t-1}) + b_n\right),
\qquad h_t = (1-z_t)\odot n_t + z_t\odot h_{t-1}.
$$

Choose `convention := resetAfter` for PyTorch's candidate equation:

$$
n_t = \tanh\!\left(W_{nx}x_t + b_{nx} + r_t\odot(W_{nh}h_{t-1} + b_{nh})\right).
$$

Here the reset acts after the recurrent affine map, including its bias. Moving the reset across
a general recurrent matrix changes the function, and the reset-dependent bias cannot generally
be folded into a constant bias. Repacking PyTorch weights alone therefore does not preserve a
general PyTorch GRU's behavior. The reset-after constructor implements this recurrence as written
and stores `weight_ih, weight_hh, bias_ih, bias_hh` in PyTorch's packed gate order.

The reset-before core stores a matrix and one bias for each gate, in reset, update, candidate order.
Each matrix has shape `[hiddenWidth, inputWidth + hiddenWidth]`, with input columns followed by
hidden columns.
PyTorch stores separate input and recurrent matrices and biases, each packed in the same gate
order. Its reset and update bias pairs can be added for forward evaluation. The candidate differs
by the reset placement above.

Input and output shapes are `batchShape ++ [sequenceLength, inputWidth]` and
`batchShape ++ [sequenceLength, outputWidth]`. The core weights are shared across batch entries,
and the same linear head projects every hidden state. Each call starts every sequence at zero
hidden state and returns all projected time steps. There is no initial-state argument or separate
final-state result, and hidden state is not carried between calls.
-/
def Recurrent.gru (config : Recurrent.Config) (batchShape : Shape := [])
    (convention : GRU.Convention := .resetBefore) :
    nn.Builder (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  Recurrent.build "GRU" config batchShape fun inputWidth hiddenWidth => by
    simpa only [← List.append_cons] using
      (nn.gru config.sequenceLength inputWidth hiddenWidth
        (batchShape := batchShape) (convention := convention))

/--
LSTM layers followed by a time-distributed linear head.
-/
def Recurrent.lstm (config : Recurrent.Config) (batchShape : Shape := []) :
    nn.Builder (nn.Sequential (config.inputShape batchShape) (config.outputShape batchShape)) :=
  Recurrent.build "LSTM" config batchShape fun inputWidth hiddenWidth => by
    simpa only [← List.append_cons] using
      (nn.lstm config.sequenceLength inputWidth hiddenWidth (batchShape := batchShape))

end nn

end TorchLean
