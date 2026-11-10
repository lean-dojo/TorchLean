/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Layer.Recurrent

/-!
# Recurrent models with explicit state

A `Recurrent.Cell` is one step from an input and a state to the next state, with the hidden vector
the state exposes. `Cell.rnn`, `Cell.lstm` and `Cell.gru` are the cells of `Layer.rnnCell`,
`nn.Layer.lstmCell` and `nn.Layer.gruCell`. `run` threads a supplied initial state through a
sequence and returns every hidden vector and the final state. The models (`sequence`, `stacked`,
`classifier`, `generator`, `bidirectional`, `languageModel`, `encoderDecoder`, `residual`) are one
definition each over any cell and any head. A head, embedding or projection is a function on
references, for example the linear map
`fun x => Ops.apply (.linear (lead := [])) (weight, bias, x, ())`.

They run on every interpreter that implements their operators. The verifier recorder refuses `neg`,
so it refuses the GRU cells, and `flip`, so it refuses `bidirectional`. Their meaning under the pure
interpreter is in `TorchLean.NN.Models.Recurrent.Stack.Correctness`.

We wrote the models by hand as compositions of cells and heads. `bidirectional` lets the two cells
differ in width and puts the forward stream first, the order of directions 0 and 1 in
`torch.nn.RNN` with `bidirectional=True` (PyTorch 2.8 docs). `encoderDecoder` starts the decoder
from the encoder's final state, the `v` of eq. (1) in Ilya Sutskever, Oriol Vinyals, and Quoc V.
Le (2014), "Sequence to Sequence Learning with Neural Networks", NIPS 2014, arXiv:1409.3215. It
hands over the whole state, so an LSTM decoder starts from both `hidden` and `cell`. No
TorchLeanTest test runs these models.
-/

@[expose] public section

namespace TorchLean.Recurrent

open TorchLean.Tensor Ops
open nn.Layer

/-- A recurrent cell: one step from an input vector and a state to the next state, and the hidden
vector of a state. -/
structure Cell (m : Type → Type) (α : Type) [Storage α] [Context α] [Ops m α]
    (State : Type) (inputWidth hiddenWidth : Nat) where
  /-- The next state from the input and the previous state. -/
  step : Ref (m := m) (α := α) [inputWidth] → State → m State
  /-- The hidden vector a state exposes. -/
  hidden : State → Ref (m := m) (α := α) [hiddenWidth]

variable {α : Type} [Storage α] [Context α] {m : Type → Type} [Monad m] [Ops m α]

/-- The Elman RNN cell of `nn.Layer.rnnCell`, whose state is the hidden vector. -/
def Cell.rnn {inputWidth hiddenWidth : Nat}
    (params : Pack (Ref (m := m) (α := α)) (rnnCellShapes inputWidth hiddenWidth)) :
    Cell m α (Ref (m := m) (α := α) [hiddenWidth]) inputWidth hiddenWidth :=
  ⟨rnnCell params, id⟩

/-- The LSTM cell of `nn.Layer.lstmCell`, whose state `(hidden, cell)` exposes `hidden`. -/
def Cell.lstm {inputWidth hiddenWidth : Nat}
    (params : Pack (Ref (m := m) (α := α)) (lstmCellShapes inputWidth hiddenWidth)) :
    Cell m α (Ref (m := m) (α := α) [hiddenWidth] × Ref (m := m) (α := α) [hiddenWidth])
      inputWidth hiddenWidth :=
  ⟨lstmCell params, Prod.fst⟩

/-- The GRU cell of `nn.Layer.gruCell` under `convention`, whose state is the hidden vector. -/
def Cell.gru (convention : GRU.Convention) {inputWidth hiddenWidth : Nat}
    (params : Pack (Ref (m := m) (α := α)) (gruCellShapes convention inputWidth hiddenWidth)) :
    Cell m α (Ref (m := m) (α := α) [hiddenWidth]) inputWidth hiddenWidth :=
  ⟨gruCell convention params, id⟩

variable {State : Type}

/-- Run `cell` over the rows of `inputs` from `initial`: every hidden vector, stacked, and the final
state. An empty sequence gives a zero tensor and `initial`. -/
def run {length inputWidth hiddenWidth : Nat} (cell : Cell m α State inputWidth hiddenWidth)
    (initial : State) (inputs : Ref (m := m) (α := α) [length, inputWidth]) :
    m (Ref (m := m) (α := α) [length, hiddenWidth] × State) :=
  scan (m := m) (α := α) (fun input state => do
    let next ← cell.step input state
    pure (next, cell.hidden next)) initial inputs

/-- `f` on the rows `t` of `left` and `right` for every `t`, stacked along the leading axis. -/
def zipRows {length : Nat} {σ σ' τ : Shape}
    (f : Ref (m := m) (α := α) σ → Ref (m := m) (α := α) σ' → m (Ref (m := m) (α := α) τ))
    (left : Ref (m := m) (α := α) (Shape.dim length σ))
    (right : Ref (m := m) (α := α) (Shape.dim length σ')) :
    m (Ref (m := m) (α := α) (Shape.dim length τ)) :=
  Prod.fst <$> nn.Layer.scanRows (m := m) (α := α) () fun t (u : Unit) => do
    let x ← Ops.apply (m := m) (α := α)
      (.indexAt (s := Shape.dim length σ) ⟨0, Shape.AxisInBounds.proof⟩ t) (left, ())
    let y ← Ops.apply (m := m) (α := α)
      (.indexAt (s := Shape.dim length σ') ⟨0, Shape.AxisInBounds.proof⟩ t) (right, ())
    let output ← f x y
    pure (u, output)

/-- A stack of recurrent cells whose widths match, each with a state indexed by its hidden width.
-/
inductive Stack (m : Type → Type) (α : Type) [Storage α] [Context α] [Ops m α]
    (State : Nat → Type) : Nat → Nat → Type where
  /-- No recurrent layers. -/
  | nil {width : Nat} : Stack m α State width width
  /-- One cell followed by a stack that reads its hidden stream. -/
  | cons {inputWidth hiddenWidth outputWidth : Nat}
      (cell : Cell m α (State hiddenWidth) inputWidth hiddenWidth)
      (rest : Stack m α State hiddenWidth outputWidth) : Stack m α State inputWidth outputWidth

namespace Stack

variable {StateOf : Nat → Type}

/-- One state per layer, and no state for an empty stack. -/
def States {inputWidth outputWidth : Nat} :
    Stack m α StateOf inputWidth outputWidth → Type
  | .nil => Unit
  | .cons (hiddenWidth := width) _ rest => StateOf width × rest.States

/-- Run each layer on the hidden stream of the previous one, each from its own state: the last
hidden stream and every final state. An empty stack returns its inputs. -/
def run {length : Nat} : {inputWidth outputWidth : Nat} →
    (layers : Stack m α StateOf inputWidth outputWidth) → layers.States →
    Ref (m := m) (α := α) [length, inputWidth] →
      m (Ref (m := m) (α := α) [length, outputWidth] × layers.States)
  | _, _, .nil, states, inputs => pure (inputs, states)
  | _, _, .cons cell rest, states, inputs => do
      let first ← Recurrent.run cell states.1 inputs
      let others ← rest.run states.2 first.1
      pure (others.1, ((first.2, others.2) : _ × rest.States))

/-- The stack of the cells of `cells`, in order, all of width `width`. -/
def ofList {width : Nat} :
    List (Cell m α State width width) → Stack m α (fun _ => State) width width
  | [] => .nil
  | cell :: cells => .cons cell (ofList cells)

/-- The states of `layers` from a list with one state per layer, or `none` for a list of another
length. -/
def states? {inputWidth outputWidth : Nat} :
    (layers : Stack m α (fun _ => State) inputWidth outputWidth) → List State →
      Option layers.States
  | .nil, [] => some ()
  | .cons _ rest, state :: states =>
      (rest.states? states).map fun tail => ((state, tail) : State × rest.States)
  | _, _ => none

/-- The states of `layers` as a list, in layer order. -/
def statesList {inputWidth outputWidth : Nat} :
    (layers : Stack m α (fun _ => State) inputWidth outputWidth) → layers.States → List State
  | .nil, _ => []
  | .cons _ rest, states => states.1 :: rest.statesList states.2

end Stack

/-- Run `cell` from `initial`, then `head` on every hidden vector: the outputs and the final state.
-/
def sequence {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell m α State inputWidth hiddenWidth)
    (head : Ref (m := m) (α := α) [hiddenWidth] → m (Ref (m := m) (α := α) [outputWidth]))
    (initial : State) (inputs : Ref (m := m) (α := α) [length, inputWidth]) :
    m (Ref (m := m) (α := α) [length, outputWidth] × State) := do
  let result ← run cell initial inputs
  let outputs ← Ops.mapBatch (m := m) (α := α) (σ := [hiddenWidth]) head result.1
  pure (outputs, result.2)

/-- Run `layers` from one state each, then `head` on every vector of the last hidden stream: the
outputs and every final state. -/
def stacked {StateOf : Nat → Type} {length inputWidth hiddenWidth outputWidth : Nat}
    (layers : Stack m α StateOf inputWidth hiddenWidth)
    (head : Ref (m := m) (α := α) [hiddenWidth] → m (Ref (m := m) (α := α) [outputWidth]))
    (initial : layers.States) (inputs : Ref (m := m) (α := α) [length, inputWidth]) :
    m (Ref (m := m) (α := α) [length, outputWidth] × layers.States) := do
  let result ← layers.run initial inputs
  let outputs ← Ops.mapBatch (m := m) (α := α) (σ := [hiddenWidth]) head result.1
  pure (outputs, result.2)

/-- `head` on the hidden vector of the final state of `cell` run from `initial`: for an empty
sequence, on the hidden vector of `initial`. -/
def classifier {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell m α State inputWidth hiddenWidth)
    (head : Ref (m := m) (α := α) [hiddenWidth] → m (Ref (m := m) (α := α) [outputWidth]))
    (initial : State) (inputs : Ref (m := m) (α := α) [length, inputWidth]) :
    m (Ref (m := m) (α := α) [outputWidth]) := do
  let result ← run cell initial inputs
  head (cell.hidden result.2)

/-- `embedding` on every token row, then `sequence`. -/
def generator {length tokenWidth inputWidth hiddenWidth outputWidth : Nat}
    (embedding : Ref (m := m) (α := α) [tokenWidth] → m (Ref (m := m) (α := α) [inputWidth]))
    (cell : Cell m α State inputWidth hiddenWidth)
    (head : Ref (m := m) (α := α) [hiddenWidth] → m (Ref (m := m) (α := α) [outputWidth]))
    (initial : State) (tokens : Ref (m := m) (α := α) [length, tokenWidth]) :
    m (Ref (m := m) (α := α) [length, outputWidth] × State) := do
  let embedded ← Ops.mapBatch (m := m) (α := α) (σ := [tokenWidth]) embedding tokens
  sequence cell head initial embedded

/-- Run `forwardCell` over `inputs` and `backwardCell` over them reversed in time, reverse the
second hidden stream back, and apply `head` to `[forward; backward]` at every step. -/
def bidirectional {State' : Type} {length inputWidth hiddenWidth hiddenWidth' outputWidth : Nat}
    (forwardCell : Cell m α State inputWidth hiddenWidth)
    (backwardCell : Cell m α State' inputWidth hiddenWidth')
    (head : Ref (m := m) (α := α) [hiddenWidth + hiddenWidth'] →
      m (Ref (m := m) (α := α) [outputWidth]))
    (forwardInitial : State) (backwardInitial : State')
    (inputs : Ref (m := m) (α := α) [length, inputWidth]) :
    m (Ref (m := m) (α := α) [length, outputWidth]) := do
  let forward ← run forwardCell forwardInitial inputs
  let reversed ← Ops.apply (m := m) (α := α) (.flip (s := [length, inputWidth]) [⟨0, by simp⟩])
    (inputs, ())
  let backward ← run backwardCell backwardInitial reversed
  let backwardHidden ← Ops.apply (m := m) (α := α)
    (.flip (s := [length, hiddenWidth']) [⟨0, by simp⟩]) (backward.1, ())
  zipRows (m := m) (α := α) (σ := [hiddenWidth]) (σ' := [hiddenWidth']) (fun a b => do
    let both ← Ops.apply (m := m) (α := α)
      (.cat (s := .dim 0 .scalar) ⟨0, by simp⟩ [hiddenWidth, hiddenWidth']) (a, b, ())
    head both) forward.1 backwardHidden

/-- `embedding` on every token row, then the cells of `layers` in order from one state each, then
`head` on every vector: the outputs and the final states, or `none` unless `initial` has one state
per layer. -/
def languageModel {length tokenWidth width outputWidth : Nat}
    (embedding : Ref (m := m) (α := α) [tokenWidth] → m (Ref (m := m) (α := α) [width]))
    (layers : List (Cell m α State width width))
    (head : Ref (m := m) (α := α) [width] → m (Ref (m := m) (α := α) [outputWidth]))
    (initial : List State) (tokens : Ref (m := m) (α := α) [length, tokenWidth]) :
    m (Option (Ref (m := m) (α := α) [length, outputWidth] × List State)) :=
  match (Stack.ofList layers).states? initial with
  | none => pure none
  | some states => do
      let embedded ← Ops.mapBatch (m := m) (α := α) (σ := [tokenWidth]) embedding tokens
      let result ← stacked (Stack.ofList layers) head states embedded
      pure (some (result.1, (Stack.ofList layers).statesList result.2))

/-- Encode the embedded `source` with `encoder` from `initial`, then run `generator` with `decoder`
on `target` from the encoder's final state: the outputs, the encoder's final state, and the
decoder's final state. -/
def encoderDecoder {sourceLength targetLength sourceWidth targetWidth sourceInput targetInput
      hiddenWidth outputWidth : Nat}
    (sourceEmbedding : Ref (m := m) (α := α) [sourceWidth] →
      m (Ref (m := m) (α := α) [sourceInput]))
    (encoder : Cell m α State sourceInput hiddenWidth)
    (targetEmbedding : Ref (m := m) (α := α) [targetWidth] →
      m (Ref (m := m) (α := α) [targetInput]))
    (decoder : Cell m α State targetInput hiddenWidth)
    (head : Ref (m := m) (α := α) [hiddenWidth] → m (Ref (m := m) (α := α) [outputWidth]))
    (initial : State) (source : Ref (m := m) (α := α) [sourceLength, sourceWidth])
    (target : Ref (m := m) (α := α) [targetLength, targetWidth]) :
    m (Ref (m := m) (α := α) [targetLength, outputWidth] × State × State) := do
  let embedded ← Ops.mapBatch (m := m) (α := α) (σ := [sourceWidth]) sourceEmbedding source
  let encoded ← run encoder initial embedded
  let decoded ← generator targetEmbedding decoder head encoded.2 target
  pure (decoded.1, encoded.2, decoded.2)

/-- Run `cell` from `initial`, add `projection` of each input to its hidden vector (`h + p`), and
apply `head` at every step: the outputs and the last of those residual features (for an empty
sequence, the hidden vector of `initial`). -/
def residual {length inputWidth hiddenWidth outputWidth : Nat}
    (cell : Cell m α State inputWidth hiddenWidth)
    (projection : Ref (m := m) (α := α) [inputWidth] → m (Ref (m := m) (α := α) [hiddenWidth]))
    (head : Ref (m := m) (α := α) [hiddenWidth] → m (Ref (m := m) (α := α) [outputWidth]))
    (initial : State) (inputs : Ref (m := m) (α := α) [length, inputWidth]) :
    m (Ref (m := m) (α := α) [length, outputWidth] × Ref (m := m) (α := α) [hiddenWidth]) := do
  let result ← run cell initial inputs
  let features ← zipRows (m := m) (α := α) (σ := [hiddenWidth]) (σ' := [inputWidth])
    (fun h x => do
      let projected ← projection x
      Ops.apply (m := m) (α := α) (.add (s := [hiddenWidth])) (h, projected, ())) result.1 inputs
  let outputs ← Ops.mapBatch (m := m) (α := α) (σ := [hiddenWidth]) head features
  if hLength : 0 < length then do
    let last ← Ops.apply (m := m) (α := α)
      (.indexAt (s := [length, hiddenWidth]) ⟨0, Shape.AxisInBounds.proof⟩
        ⟨length - 1, Nat.sub_one_lt (Nat.ne_of_gt hLength)⟩) (features, ())
    pure (outputs, last)
  else
    pure (outputs, cell.hidden initial)

end TorchLean.Recurrent
