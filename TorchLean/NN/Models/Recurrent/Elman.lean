/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.DGraph
public import TorchLean.NN.Tape.RealNode.Arm
public import TorchLean.NN.Op.Pointwise

/-!
# Elman RNN Cell VJP

`cellDGraph` is one step of a vanilla tanh RNN as a proof-carrying graph:

`h' = tanh(W [x; h] + b)`.

The graph concatenates the input and the hidden state, applies a fixed affine map and takes
`tanh`. Its bundled certificate, read at the input with `.at`, gives the cell's reverse-mode
theorem through `Tape.Graph.backpropVec_eq_fderiv`, and `differentiableAt_unroll`
proves differentiability of mathematical unrolls with supplied differentiable transitions. No
derivative formula or backward pass is given for the unroll. Connecting unrolls to the runtime
sequence layer, its gradient accumulation and its trainable parameters needs separate
correspondence proofs.

The cell is the tanh recurrence of `torch.nn.RNN`, written by hand and stacked into one map. Our
`W` is `[W_ih | W_hh]` with the input columns first, the order in which `concat` builds `[x; h]`,
and `b` is torch's `b_ih + b_hh`. A torch cell with 3 inputs and 4 hidden units becomes a 4 × 7
`W` here, and the two give the same step over the reals. No TorchLeanTest test runs `cellDGraph`.

References:

* Jeffrey L. Elman (1990), "Finding Structure in Time", Cognitive Science 14(2):179-211,
  doi:10.1207/s15516709cog1402_1.
* `torch.nn.RNN` (PyTorch 2.8 docs):
  https://docs.pytorch.org/docs/2.8/generated/torch.nn.RNN.html
-/

@[expose] public section

namespace TorchLean
namespace RNN

open Tape.RealNode

noncomputable section

open Tape (DGraph)

/-- Context for a one-step Elman cell: current input and previous hidden state. -/
abbrev ctx (inputSize hiddenSize : Nat) : List Shape :=
  [[inputSize], [hiddenSize]]

/-- Saved tensors: concatenated `[x; h]`, affine preactivation, and next hidden state. -/
abbrev ssCell (inputSize hiddenSize : Nat) : List Shape :=
  [[inputSize + hiddenSize], [hiddenSize], [hiddenSize]]

/-- Index of the current input vector in the Elman-cell context. -/
def idxInput {inputSize hiddenSize : Nat} {ss : List Shape} :
    Idx (ctx inputSize hiddenSize ++ ss) [inputSize] :=
  ⟨⟨0, by simp [ctx]⟩, by simp [ctx]⟩

/-- Index of the previous hidden state in the Elman-cell context. -/
def idxHidden {inputSize hiddenSize : Nat} {ss : List Shape} :
    Idx (ctx inputSize hiddenSize ++ ss) [hiddenSize] :=
  ⟨⟨1, by simp [ctx]⟩, by simp [ctx]⟩

/-- Index of the concatenated `[x; h]` vector. -/
def idxConcat {inputSize hiddenSize : Nat} :
    Idx (ctx inputSize hiddenSize ++ [.dim (inputSize + hiddenSize) .scalar])
      (.dim (inputSize + hiddenSize) .scalar) :=
  Idx.last (Γ := ctx inputSize hiddenSize) (ss := [])
    (τ := .dim (inputSize + hiddenSize) .scalar)

/-- Index of the affine preactivation. -/
def idxPre {inputSize hiddenSize : Nat} :
    Idx (ctx inputSize hiddenSize ++
      [[inputSize + hiddenSize], [hiddenSize]]) [hiddenSize] :=
  Idx.last (Γ := ctx inputSize hiddenSize)
    (ss := [.dim (inputSize + hiddenSize) .scalar]) (τ := [hiddenSize])

/--
Proof-carrying graph for one Elman RNN cell: concatenate `[x; h]`, apply `cell`, take `tanh`.

The affine map `cell` is a fixed `Linear` outside the context, so the graph's VJP is the
gradient with respect to the cell inputs `(x, h)` only.
-/
def cellDGraph {inputSize hiddenSize : Nat}
    (cell : Linear ℝ (inputSize + hiddenSize) hiddenSize) :
    DGraph (ctx inputSize hiddenSize) (ssCell inputSize hiddenSize) :=
  let dg0 : DGraph (ctx inputSize hiddenSize) [] := DGraph.nil
  let dg1 :
      DGraph (ctx inputSize hiddenSize) [.dim (inputSize + hiddenSize) .scalar] :=
    DGraph.snoc (dg := dg0)
      (node := concat
        (Γ := ctx inputSize hiddenSize) (n := inputSize) (m := hiddenSize) (s := .scalar)
        (idxInput (inputSize := inputSize) (hiddenSize := hiddenSize) (ss := []))
        (idxHidden (inputSize := inputSize) (hiddenSize := hiddenSize) (ss := [])))
      (hn := concatFDeriv
        (Γ := ctx inputSize hiddenSize) (n := inputSize) (m := hiddenSize) (s := .scalar)
        (idxInput (inputSize := inputSize) (hiddenSize := hiddenSize) (ss := []))
        (idxHidden (inputSize := inputSize) (hiddenSize := hiddenSize) (ss := [])))
  let dg2 :
      DGraph (ctx inputSize hiddenSize)
        [[inputSize + hiddenSize], [hiddenSize]] :=
    DGraph.snoc (dg := dg1)
      (node := linear
        (Γ := ctx inputSize hiddenSize ++ [.dim (inputSize + hiddenSize) .scalar])
        (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
        (idxConcat (inputSize := inputSize) (hiddenSize := hiddenSize)) cell)
      (hn := linearFDeriv
        (Γ := ctx inputSize hiddenSize ++ [.dim (inputSize + hiddenSize) .scalar])
        (inDim := inputSize + hiddenSize) (outDim := hiddenSize)
        (idxConcat (inputSize := inputSize) (hiddenSize := hiddenSize)) cell)
  DGraph.snoc (dg := dg2)
    (node := Tape.LawfulNode.ofArm (Op.tanh (s := [hiddenSize])) rfl
      (Γ := ctx inputSize hiddenSize ++ [[inputSize + hiddenSize], [hiddenSize]])
      (idxPre (inputSize := inputSize) (hiddenSize := hiddenSize), ())
      (Arm.adj_of_cert₁ fun xs δ => Op.tanh_backward_eq_adjoint xs.1 δ))
    (hn := FDerivCorrect.ofArm₁
      (Γ := ctx inputSize hiddenSize ++ [[inputSize + hiddenSize], [hiddenSize]])
      (s := [hiddenSize]) (idx := idxPre (inputSize := inputSize) (hiddenSize := hiddenSize))
      (cert := Op.tanh_backward_eq_adjoint))

/-- Forward evaluation of one Elman cell, returning the input context followed by the saved
tensors. -/
def cellEval {inputSize hiddenSize : Nat}
    (cell : Linear ℝ (inputSize + hiddenSize) hiddenSize) :
    CtxVec (ctx inputSize hiddenSize) →
      CtxVec (ctx inputSize hiddenSize ++ ssCell inputSize hiddenSize) :=
  Tape.GraphData.evalVec
    (Γ := ctx inputSize hiddenSize)
    (ss := ssCell inputSize hiddenSize)
    (cellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g.toData ()

/-- Forward evaluation of one Elman cell is differentiable at every input context. -/
theorem differentiableAt_cellEval
    {inputSize hiddenSize : Nat}
    (cell : Linear ℝ (inputSize + hiddenSize) hiddenSize)
    (xV : CtxVec (ctx inputSize hiddenSize)) :
    DifferentiableAt ℝ (cellEval cell) xV := by
  obtain ⟨D, hD, -⟩ := Tape.GraphData.hasFDerivAt_evalVec_and_jvp
    (cellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).g.toData () xV
    ((cellDGraph (inputSize := inputSize) (hiddenSize := hiddenSize) cell).hg.at xV)
  exact hD.differentiableAt

/--
Unroll an Elman cell through a list of transitions, returning the trace of the last cell.

Each transition maps the full trace of one cell to the context of the next. In the usual sequence
case the transition selects the next input vector and passes on the hidden state from the
previous step.
-/
def unroll {inputSize hiddenSize : Nat}
    (cell : Linear ℝ (inputSize + hiddenSize) hiddenSize) :
    List
      (CtxVec (ctx inputSize hiddenSize ++ ssCell inputSize hiddenSize) →
        CtxVec (ctx inputSize hiddenSize)) →
      CtxVec (ctx inputSize hiddenSize) →
      CtxVec (ctx inputSize hiddenSize ++ ssCell inputSize hiddenSize)
  | [], x => cellEval cell x
  | step :: rest, x => unroll cell rest (step (cellEval cell x))

/--
`TransitionsDifferentiableAt cell steps x` holds when each transition in `steps` is
differentiable at the trace the previous cell produces, starting from the cell context `x`.

The predicate recurses along `steps`, so the induction in `differentiableAt_unroll` is
independent of how a sequence is stored.
-/
def TransitionsDifferentiableAt {inputSize hiddenSize : Nat}
    (cell : Linear ℝ (inputSize + hiddenSize) hiddenSize) :
    List
      (CtxVec (ctx inputSize hiddenSize ++ ssCell inputSize hiddenSize) →
        CtxVec (ctx inputSize hiddenSize)) →
      CtxVec (ctx inputSize hiddenSize) → Prop
  | [], _ => True
  | step :: rest, x =>
      DifferentiableAt ℝ step (cellEval cell x) ∧
      TransitionsDifferentiableAt cell rest (step (cellEval cell x))

/--
Differentiability of an arbitrary-length mathematical Elman RNN unroll.

If every transition between cells is differentiable at the trace reached during the forward pass,
then the whole recurrence is differentiable at the initial cell context.
-/
theorem differentiableAt_unroll
    {inputSize hiddenSize : Nat}
    (cell : Linear ℝ (inputSize + hiddenSize) hiddenSize)
    (steps :
      List
        (CtxVec (ctx inputSize hiddenSize ++ ssCell inputSize hiddenSize) →
          CtxVec (ctx inputSize hiddenSize)))
    (x : CtxVec (ctx inputSize hiddenSize))
    (hSteps : TransitionsDifferentiableAt cell steps x) :
    DifferentiableAt ℝ (unroll cell steps) x := by
  induction steps generalizing x with
  | nil =>
      -- `unroll cell []` unfolds to `cellEval cell`.
      exact differentiableAt_cellEval cell x
  | cons step rest ih =>
      rcases hSteps with ⟨hStep, hRest⟩
      have hcomp := (ih (step (cellEval cell x)) hRest).comp x
        (hStep.comp x (differentiableAt_cellEval cell x))
      exact hcomp.congr_of_eventuallyEq (Filter.Eventually.of_forall fun _ => rfl)

end

end RNN
end TorchLean
