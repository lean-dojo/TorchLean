/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Core

/-!
# Recurrent Stacks with Independent Widths

Adjacent cell dimensions agree by construction. The empty stack preserves the input width, and
its state is empty. A nonempty stack carries one state of the appropriate width per layer.
-/

@[expose] public section

namespace Spec

open TorchLean

/-- A sequence of recurrent cells with independently chosen intermediate widths. -/
inductive RecurrentStack (Cell : Nat → Nat → Type) : Nat → Nat → Type where
  /-- No recurrent layers. -/
  | nil {width : Nat} : RecurrentStack Cell width width
  /-- One cell followed by a stack consuming its hidden stream. -/
  | cons {inputWidth hiddenWidth outputWidth : Nat}
      (cell : Cell inputWidth hiddenWidth) (rest : RecurrentStack Cell hiddenWidth outputWidth) :
      RecurrentStack Cell inputWidth outputWidth

namespace RecurrentStack

/-- One state per layer, with no states for an empty stack. -/
def States {Cell : Nat → Nat → Type} {inputWidth outputWidth : Nat}
    (layers : RecurrentStack Cell inputWidth outputWidth) (State : Nat → Type) : Type :=
  match layers with
  | .nil => Unit
  | .cons (hiddenWidth := width) _ rest => State width × rest.States State

/-- Run each cell on the preceding layer's entire sequence and retain every final state. -/
def run {α : Type} [Storage α] {Cell : Nat → Nat → Type} {State : Nat → Type}
    {sequenceLength inputWidth outputWidth : Nat}
    (layers : RecurrentStack Cell inputWidth outputWidth)
    (runCell : {input hidden : Nat} → Cell input hidden →
      Tensor α [sequenceLength, input] → State hidden →
      Tensor α [sequenceLength, hidden] × State hidden)
    (inputs : Tensor α [sequenceLength, inputWidth]) (states : layers.States State) :
    Tensor α [sequenceLength, outputWidth] × layers.States State :=
  match layers with
  | .nil => (inputs, states)
  | .cons cell rest =>
      let (output, state) := runCell cell inputs states.1
      let (finalOutput, finalStates) := rest.run runCell output states.2
      (finalOutput, (state, finalStates))

/-- Compose the recurrent modules and a head, including a head with no recurrent layers. -/
def toChain {α : Type} [Storage α] {Cell : Nat → Nat → Type}
    {sequenceLength inputWidth hiddenWidth outputWidth : Nat}
    (layers : RecurrentStack Cell inputWidth hiddenWidth)
    (toModule : {input hidden : Nat} → Cell input hidden →
      Module α [sequenceLength, input] [sequenceLength, hidden])
    (head : Module α [sequenceLength, hiddenWidth] [sequenceLength, outputWidth]) :
    Module.Chain α [sequenceLength, inputWidth] [sequenceLength, outputWidth] :=
  match layers with
  | .nil => .single head
  | .cons cell rest => .comp (.single (toModule cell)) (rest.toChain toModule head)

end RecurrentStack
end Spec
