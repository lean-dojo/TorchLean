/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Linear
public import NN.Spec.Module.SpecModule

/-!
# Linear module wrappers

`NN/Spec/Layers/Linear.lean` defines the math-level linear layer spec (parameters + gradients).

This file provides `NNModuleSpec` wrappers so linear layers can be:

- composed in a `SpecChain` with compile-time shape checking, and
- annotated with lightweight export metadata (PyTorch pretty-printing).

These wrappers are thin on purpose: the meaning is still the underlying `linear_spec`; the extra
fields are just metadata.

Most readers can map these directly to PyTorch: `LinearModuleSpec` is `nn.Linear`, and the sequence
helpers are the common pattern "apply a linear layer to each timestep, then maybe select the last
step for classification".
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec


variable {α : Type} [Add α] [Mul α] [Zero α]
/-- Wrap LinearSpec forward as an NNModuleSpec. -/
def LinearModuleSpec {inDim outDim : Nat}
  (m : LinearSpec α inDim outDim) :
  NNModuleSpec α (.dim inDim .scalar) (.dim outDim .scalar) :=
{ forward := fun x => linearSpec (α:=α) m x, kind := "Linear", export_func := {
  toPyTorch := s!"nn.Linear({inDim}, {outDim})",
  dimensions := (inDim, outDim)
} }

/-- Apply a linear layer independently at each timestep (sequence lift). -/
def LinearSeqModuleSpec {α : Type} [Context α]
  {seqLen hiddenSize outputSize : Nat}
  (m : LinearSpec α hiddenSize outputSize) :
  NNModuleSpec α (Shape.dim seqLen (Shape.dim hiddenSize .scalar))
                  (Shape.dim seqLen (Shape.dim outputSize .scalar)) :=
  SpecChain.mapEach (LinearModuleSpec m)

-- Helper function to get the last timestep from a sequence
/-- Extract the last timestep of a sequence (requires `seqLen ≠ 0`).

In PyTorch terms: `seq[-1]` when `seq` has shape `(seqLen, hiddenSize)`. -/
def getLastTimestepSpec {α : Type} [Context α] {seqLen hiddenSize : Nat}
  (seq : Tensor α (Shape.dim seqLen (Shape.dim hiddenSize .scalar)))
  (h : seqLen ≠ 0) : Tensor α (Shape.dim hiddenSize .scalar) :=
  match seq with
  | Tensor.dim seq_fn =>
    -- Get the last valid index
    let last_idx : Fin seqLen := ⟨Nat.pred seqLen, Nat.pred_lt h⟩
    seq_fn last_idx

-- Linear classifier that extracts last timestep from sequence for classification
/-- Sequence classifier: take last timestep, then apply a linear projection to `numClasses`. -/
def LinearClassifierModuleSpec {α : Type} [Context α]
  {seqLen hiddenSize numClasses : Nat}
  (m : LinearSpec α hiddenSize numClasses)
  (h : seqLen ≠ 0) :
  NNModuleSpec α (Shape.dim seqLen (Shape.dim hiddenSize .scalar))
                  (Shape.dim numClasses .scalar) :=
{
  forward := fun x =>
    -- Extract the last timestep for classification
    let last_timestep := getLastTimestepSpec x h
    Spec.linearSpec (α:=α) m last_timestep,
  kind := "LinearClassifier",
  export_func := {
    toPyTorch := s!"nn.Sequential(SelectLast(), nn.Linear({hiddenSize}, {numClasses}))",
    dimensions := (hiddenSize, numClasses)
  }
}

/-- Example: Linear → Linear composition via SpecChain. -/
def twoLayerLinear
  {inDim hidDim outDim : Nat}
  (l1 : LinearSpec α inDim hidDim)
  (l2 : LinearSpec α hidDim outDim) :
  ModSpec.SpecChain α (.dim inDim .scalar) (.dim outDim .scalar) :=
  let m1 := LinearModuleSpec (α:=α) l1
  let m2 := LinearModuleSpec (α:=α) l2
  ModSpec.SpecChain.comp (ModSpec.SpecChain.single m1) (ModSpec.SpecChain.single m2)

end Spec
