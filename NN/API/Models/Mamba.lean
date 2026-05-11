/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.Spec.Models.Mamba

/-!
# Mamba Model Helpers (API)

Reusable configuration, model constructors, and text helpers for Mamba-style sequence models.

The trainable model path uses TorchLean autograd layers and therefore runs on the CPU and CUDA
backends.  The spec-backed deterministic helpers below are kept as small mathematical reference
utilities; runnable training examples use the autograd constructor.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for byte-level Mamba-style language models. -/
structure MambaTextConfig where
  vocab : Nat
  stateDim : Nat
  ssmStateDim : Nat
  convWidth : Nat
deriving Repr

/-- One-hot token vector shape. -/
abbrev mambaTokenVec (cfg : MambaTextConfig) : Shape :=
  NN.Tensor.Shape.Vec cfg.vocab

/-- Compact hidden-state shape. -/
abbrev mambaStateVec (cfg : MambaTextConfig) : Shape :=
  NN.Tensor.Shape.Vec cfg.stateDim

/-- Full selective-scan state shape. -/
abbrev mambaFullState (cfg : MambaTextConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.stateDim cfg.ssmStateDim

/-- Sequence-major one-hot token matrix shape. -/
abbrev mambaTokenMat (cfg : MambaTextConfig) (seqLen : Nat) : Shape :=
  NN.Tensor.Shape.Mat seqLen cfg.vocab

/-- Output logits shape for byte-level causal language modeling. -/
abbrev mambaLogitMat (cfg : MambaTextConfig) (seqLen : Nat) : Shape :=
  NN.Tensor.Shape.Mat seqLen cfg.vocab

/--
Trainable Mamba-style causal language model over one-hot token inputs.

Architecture:

`mamba(seqLen, vocab, stateDim) → linear(stateDim → vocab)` applied at every time step.

The recurrent core is a gated diagonal state-space update implemented with autograd-covered
TorchLean ops.  Passing `--cuda` to a runner that instantiates this model trains the same parameters
on the CUDA backend.
-/
def mambaTextLm (cfg : MambaTextConfig) (seqLen : Nat) :
    nn.M (nn.Sequential (mambaTokenMat cfg seqLen) (mambaLogitMat cfg seqLen)) :=
  nn.sequential![
    nn.mamba seqLen cfg.vocab cfg.stateDim,
    nn.linear cfg.stateDim cfg.vocab (pfx := NN.Tensor.Shape.Vec seqLen)
  ]

/-- Small deterministic initializer for spec-level reference blocks. -/
def mambaCenteredHash (seed modulus : Nat) : Float :=
  (Float.ofNat (seed % modulus) - Float.ofNat (modulus / 2)) / Float.ofNat modulus

/-- Build a vector tensor from an index function. -/
def mambaVectorFloat {n : Nat} (f : Fin n → Float) : _root_.Spec.Tensor Float (shape![n]) :=
  _root_.Spec.Tensor.dim (fun i => _root_.Spec.Tensor.scalar (f i))

/-- Build a matrix tensor from an index function. -/
def mambaMatrixFloat {m n : Nat} (f : Fin m → Fin n → Float) :
    _root_.Spec.Tensor Float (shape![m, n]) :=
  _root_.Spec.Tensor.dim (fun i =>
    _root_.Spec.Tensor.dim (fun j => _root_.Spec.Tensor.scalar (f i j)))

/-- Compact diagonal Mamba-style block for spec-level reference evaluation. -/
def compactMambaFloat (cfg : MambaTextConfig) :
    _root_.Models.MambaBlockSpec Float cfg.vocab cfg.stateDim cfg.vocab :=
  { inProj := mambaMatrixFloat
      (fun i j => mambaCenteredHash (i.val * 17 + j.val * 31 + 3) 47 / 4.0)
    gateProj := mambaMatrixFloat
      (fun i j => mambaCenteredHash (i.val * 13 + j.val * 19 + 7) 43 / 5.0)
    outProj := mambaMatrixFloat
      (fun i j => mambaCenteredHash (i.val * 29 + j.val * 11 + 5) 53 / 3.0)
    ssm :=
      { A := mambaVectorFloat (fun i => 0.82 + Float.ofNat (i.val % 5) * 0.025)
        B := mambaVectorFloat (fun i => 0.12 + Float.ofNat (i.val % 3) * 0.015)
        C := mambaVectorFloat (fun i => 0.90 - Float.ofNat (i.val % 4) * 0.03)
        D := mambaVectorFloat (fun i => 0.08 + Float.ofNat (i.val % 2) * 0.02) } }

/--
Full selective Mamba-style block with causal depthwise convolution and token-dependent scan
parameters.  This deterministic initializer is meant for reference evaluation rather than
checkpoint-quality training.
-/
def selectiveMambaFloat (cfg : MambaTextConfig) :
    _root_.Models.SelectiveMambaBlockSpec Float
      cfg.vocab cfg.stateDim cfg.ssmStateDim cfg.vocab cfg.convWidth :=
  { xProj := mambaMatrixFloat
      (fun i j => mambaCenteredHash (i.val * 17 + j.val * 31 + 3) 47 * 2.0)
    zProj := mambaMatrixFloat
      (fun i j => mambaCenteredHash (i.val * 13 + j.val * 19 + 7) 43 * 2.0)
    convKernel := mambaMatrixFloat
      (fun tap i => 0.4 + mambaCenteredHash (tap.val * 23 + i.val * 7 + 11) 41 * 0.2)
    convBias := mambaVectorFloat
      (fun i => mambaCenteredHash (i.val * 5 + 3) 37 * 0.2)
    dtProj := mambaMatrixFloat
      (fun i j => mambaCenteredHash (i.val * 19 + j.val * 17 + 5) 47 * 0.2)
    dtBias := mambaVectorFloat
      (fun i => -1.5 + Float.ofNat (i.val % 5) * 0.1)
    A := mambaMatrixFloat
      (fun i n => 0.2 + Float.ofNat ((i.val + n.val) % 7) * 0.03)
    bProj := mambaMatrixFloat
      (fun i n => mambaCenteredHash (i.val * 11 + n.val * 29 + 13) 53)
    cProj := mambaMatrixFloat
      (fun i n => mambaCenteredHash (i.val * 31 + n.val * 7 + 17) 59)
    dSkip := mambaVectorFloat
      (fun i => 0.35 + Float.ofNat (i.val % 3) * 0.03)
    outProj := mambaMatrixFloat
      (fun i j => mambaCenteredHash (i.val * 29 + j.val * 11 + 5) 53 / 3.0) }

def mambaTrainingOffsets (tokenCount seqLen windows : Nat) : List Nat :=
  let usable := if tokenCount > seqLen + 1 then tokenCount - seqLen - 1 else 1
  let stride := Nat.max 1 (usable / Nat.max 1 windows)
  (List.range windows).map (fun i => (i * stride) % usable)

end models
end nn

end API
end NN
