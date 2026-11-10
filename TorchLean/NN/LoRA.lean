/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Operations

/-!
# Low-Rank Adapters

LoRA represents a linear-weight update as two smaller matrices. For a base weight
$W : \mathbb{R}^{d_{in}\times d_{out}}$, an adapter of rank $r$ uses an input factor
$A : \mathbb{R}^{d_{in}\times r}$ and an output factor
$B : \mathbb{R}^{r\times d_{out}}$:

$$W_{eff}=W+sAB.$$

Weights are stored input by output, the transpose of the `[outputWidth, inputWidth]` weights of
`nn.linear` and `Tensor.linear`, so $A$ and $B$ are the transposes of the `lora_A` and `lora_B`
weights in Hugging Face PEFT. The paper's scale is $s = \alpha / r$. `Parameters` holds the two
factors, `weightUpdate` and `effectiveWeight` form $sAB$ and $W + sAB$, and `linear` maps each
input row $x$ of a batch to $xW_{eff}$ through `Tensor.vecmat`. Which parameters to train is up to
the training code.

Reference: Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models” (2021),
https://arxiv.org/abs/2106.09685.
-/

@[expose] public section

namespace TorchLean.LoRA

open TorchLean.Tensor

/-- LoRA factors for a linear weight of shape `inputWidth × outputWidth`. -/
structure Parameters (α : Type) [TorchLean.Storage α]
    (inputWidth rank outputWidth : Nat) where
  /-- The factor $A$, projecting the input dimension to the adapter rank. -/
  inputFactor : Tensor α [inputWidth, rank]
  /-- The factor $B$, projecting the adapter rank to the output dimension. -/
  outputFactor : Tensor α [rank, outputWidth]

/-- The scaled low-rank update $sAB$. -/
def weightUpdate {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {inputWidth rank outputWidth : Nat}
    (parameters : Parameters α inputWidth rank outputWidth) (scale : α) :
    Tensor α [inputWidth, outputWidth] :=
  Tensor.scale (Tensor.matmul parameters.inputFactor parameters.outputFactor) scale

/-- `effectiveWeight W p s` is the adapted weight $W + sAB$, where `p` holds $A$ and $B$. -/
def effectiveWeight {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {inputWidth rank outputWidth : Nat}
    (baseWeight : Tensor α [inputWidth, outputWidth])
    (parameters : Parameters α inputWidth rank outputWidth) (scale : α) :
    Tensor α [inputWidth, outputWidth] :=
  Tensor.add baseWeight (weightUpdate parameters scale)

/--
`linear input W p s` multiplies each row of `input` by the adapted weight $W + sAB$, for any
leading batch shape, with no bias. In PyTorch terms it is `input @ (W + s * A @ B)`. The full
`inputWidth × outputWidth` weight is formed once and shared by all rows.
-/
def linear {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {batchShape : Shape} {inputWidth rank outputWidth : Nat}
    (input : Tensor α (batchShape ++ [inputWidth]))
    (baseWeight : Tensor α [inputWidth, outputWidth])
    (parameters : Parameters α inputWidth rank outputWidth) (scale : α) :
    Tensor α (batchShape ++ [outputWidth]) :=
  let weight := effectiveWeight baseWeight parameters scale
  TorchLean.Tensor.mapLeading batchShape
    (fun row => Tensor.vecmat row weight)
    input

end TorchLean.LoRA
