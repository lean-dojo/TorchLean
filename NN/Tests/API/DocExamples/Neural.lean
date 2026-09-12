/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.API.Neural.Summary

/-!
# Compiled Docstring Examples: Model Building

Every `Example:` block in a `TorchLean.nn` docstring appears here verbatim, wrapped in a namespace
so that the same short names can be reused entry by entry. Nothing in this file runs: the point is
that the snippets a reader copies out of the API documentation are the snippets Lean typechecks on
every build, so an API change that invalidates a docstring breaks the build instead of quietly
shipping.

`repo_lint.py` compares the fenced `lean` block inside each `Example:` against the namespace body
carrying the matching `doc-example:` marker, so the two cannot drift apart. Adding an example means
writing it here first, running `python3 scripts/checks/repo_lint.py --sync-doc-examples`, and
letting that command copy the checked text into the docstring.
-/

@[expose] public section

namespace NN.Tests.API.DocExamples.Neural

open TorchLean

-- doc-example: NN/API/Seeded.lean :: def linear (inputWidth outputWidth : Nat)
namespace Linear

-- Two affine layers around a ReLU: `2 -> 8 -> 1`.
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

-- With a batch axis in front, the same call reads `[16, 2] -> [16, 8]`.
def batched : nn.Builder (nn.Sequential [16, 2] [16, 8]) :=
  nn.linear 2 8 (batchShape := [16])

end Linear

-- doc-example: NN/API/Seeded.lean :: def build
namespace Build

def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

-- Same seed, same weights, on every machine and every run.
def built : nn.Sequential [2] [1] := nn.build 7 model

end Build

-- doc-example: NN/API/Seeded.lean :: def mlp
namespace Mlp

-- `16 -> 32 -> 32 -> 1`, ReLU between hidden layers, dropout after each one.
def model : nn.Builder (nn.Sequential [16] [1]) :=
  nn.mlp 16 1 { hiddenWidths := [32, 32], activation := .relu, dropout? := some 0.1 }

end Mlp

-- doc-example: NN/API/Seeded.lean :: def conv
namespace Conv

def spatial : Tensor Nat [2] := [8, 8]

def convolution : nn.Convolution.Config 2 :=
  { outChannels := 4, kernelSize := [3, 3] }

-- One `8 x 8` channel in, four `6 x 6` feature maps out: no padding, so the kernel eats a
-- one-pixel border on each side.
def model : nn.Builder (nn.Sequential [1, 8, 8] [4, 6, 6]) :=
  nn.conv spatial convolution (inputChannels := 1)

end Conv

-- doc-example: NN/API/Seeded.lean :: def maxPool
namespace MaxPool

def spatial : Tensor Nat [2] := [8, 8]

def pooling : nn.Pooling.Config 2 :=
  { kernelSize := [2, 2], stride := [2, 2] }

-- Non-overlapping `2 x 2` windows halve both spatial axes and leave the channel count alone.
def model : nn.Builder (nn.Sequential [3, 8, 8] [3, 4, 4]) :=
  nn.maxPool spatial pooling (channels := 3)

end MaxPool

-- doc-example: NN/API/Seeded.lean :: def globalAvgPool
namespace GlobalAvgPool

-- Average every `8 x 8` map down to one number per channel, the usual last step before a
-- classifier head.
def model : nn.Builder (nn.Sequential [3, 8, 8] [3]) :=
  nn.globalAvgPool [8, 8] (channels := 3)

end GlobalAvgPool

-- doc-example: NN/API/Seeded.lean :: def batchNorm
namespace BatchNorm

-- Shape in equals shape out. What changes is the running mean and variance this layer keeps as
-- persistent buffers, updated in `.train` mode and only read in `.eval` mode.
def model : nn.Builder (nn.Sequential [3, 8, 8] [3, 8, 8]) :=
  nn.batchNorm [8, 8] (momentum := 0.1) (channels := 3)

end BatchNorm

-- doc-example: NN/API/Seeded.lean :: def layerNorm
namespace LayerNorm

-- Normalizes across the final axis of each `[16, 64]` row, the Transformer convention.
def model : nn.Builder (nn.Sequential [16, 64] [16, 64]) :=
  nn.layerNorm [16] (width := 64)

end LayerNorm

-- doc-example: NN/API/Seeded.lean :: def dropout
namespace Dropout

-- Active in `.train` mode and the identity in `.eval` mode, which the trainer selects for you.
def model : nn.Builder (nn.Sequential [64] [64]) :=
  nn.dropout 0.1

end Dropout

-- doc-example: NN/API/Seeded.lean :: def softmax
namespace Softmax

-- Axis `0` of a rank-one shape: a probability vector over ten classes.
def model : nn.Builder (nn.Sequential [10] [10]) :=
  nn.softmax 0

end Softmax

-- doc-example: NN/API/Seeded.lean :: def embedding
namespace Embedding

-- A vocabulary of 256 byte tokens, each mapped to a 32-dimensional row.
def table : nn.Embedding 256 32 :=
  nn.build 0 (nn.embedding 256 32)

-- `table.model` fixes the index shape: a length-16 token window becomes `[16, 32]`.
def model : nn.IndexedModel [16] [16, 32] (Fin 256) :=
  table.model [16]

end Embedding

-- doc-example: NN/API/Seeded.lean :: def multiHeadAttention
namespace MultiHeadAttention

-- Two heads of width 4 give an internal attention width of 8, which here happens to match the
-- model width; the two are independent, so `headCount * headWidth` may differ from it.
def model : nn.Builder (nn.Sequential [4, 8] [4, 8]) :=
  nn.multiHeadAttention { headCount := 2, headWidth := 4 }
    (sequenceLength := 4) (modelWidth := 8)

-- Causal masking is a separate argument rather than a config field, because the mask is a value
-- with the sequence length in its type.
def causal : nn.Builder (nn.Sequential [4, 8] [4, 8]) :=
  nn.multiHeadAttention { headCount := 2, headWidth := 4 }
    (mask := some (Spec.causalMask 4)) (sequenceLength := 4) (modelWidth := 8)

end MultiHeadAttention

-- doc-example: NN/API/Seeded.lean :: def transformerEncoderBlock
namespace TransformerEncoderBlock

-- Pre-norm block, GELU feed-forward, no dropout: attention and feed-forward each sit inside their
-- own residual connection, so shapes in and out agree.
def model : nn.Builder (nn.Sequential [16, 64] [16, 64]) :=
  nn.transformerEncoderBlock
    { headCount := 4
      headWidth := 16
      feedForwardWidth := 256
      normalizeFirst := true }
    (sequenceLength := 16) (modelWidth := 64)

end TransformerEncoderBlock

-- doc-example: NN/API/Seeded.lean :: def lstm
namespace Lstm

-- Sixteen timesteps of width 8 in, sixteen hidden states of width 32 out.
def model : nn.Builder (nn.Sequential [16, 8] [16, 32]) :=
  nn.lstm 16 8 32

end Lstm

-- doc-example: NN/API/Seeded.lean :: def withModel
namespace WithModel

def builder : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

-- `nn.Sequential` lives in `Type 1`, so it cannot be returned from `IO`. Drawing the seed in `IO`
-- and handing the model to a continuation keeps model building pure.
def main : IO Unit :=
  nn.withModel builder fun model => nn.printSummary model

end WithModel

-- doc-example: NN/API/Neural/Summary.lean :: def printSummary
namespace PrintSummary

def model : nn.Sequential [2] [1] :=
  nn.build 0 nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

-- Prints one row per layer with its kind, shapes, and parameter count, then the totals. The
-- counterpart of `print(model)` plus `torchinfo.summary`.
def main : IO Unit := nn.printSummary model

end PrintSummary

-- doc-example: NN/API/Neural/Blocks.lean :: def residual
namespace Residual

def inner : nn.Builder (nn.Sequential [64] [64]) :=
  nn.Sequential![nn.linear 64 64, nn.relu]

-- `x + inner x`, the connection that made deep stacks trainable.
def model : nn.Builder (nn.Sequential [64] [64]) := do
  pure (nn.residual (← inner))

end Residual

-- doc-example: NN/API/Neural/Blocks.lean :: def addBranches
namespace AddBranches

-- Two branches over the same input, outputs summed. Parameters are stored as the first branch's
-- list followed by the second's.
def model : nn.Builder (nn.Sequential [32] [8]) := do
  let wide ← nn.Sequential![nn.linear 32 8, nn.relu]
  let shortcut ← nn.linear 32 8
  pure (nn.addBranches wide shortcut)

end AddBranches

-- doc-example: NN/API/Neural/Blocks.lean :: def concatBranches
namespace ConcatBranches

-- Concatenation along the first output axis: `[4, 8]` next to `[6, 8]` gives `[10, 8]`, and the
-- addition happens in the type rather than in a runtime shape check.
def model : nn.Builder (nn.Sequential [16] [10, 8]) := do
  let left ← nn.Sequential![nn.linear 16 (4 * 8), nn.reshape [4 * 8] [4, 8]]
  let right ← nn.Sequential![nn.linear 16 (6 * 8), nn.reshape [6 * 8] [6, 8]]
  pure (nn.concatBranches left right)

end ConcatBranches

-- doc-example: NN/API/Neural/Leading.lean :: opaque mapLeading
namespace MapLeading

def perSample : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![nn.linear 2 8, nn.relu, nn.linear 8 1]

-- One model, applied at each of five positions of a new leading axis, sharing its parameters.
def model : nn.Builder (nn.Sequential [5, 2] [5, 1]) := do
  pure (nn.mapLeading [5] (← perSample))

end MapLeading

-- doc-example: NN/API/Macros.lean :: scoped syntax (name := nnSequentialBangLit)
namespace SequentialLiteral

-- As close to `torch.nn.Sequential([...])` as a shape-indexed model can get: the entries are
-- builders, they run in order, and the shapes have to line up or the model does not compile.
def model : nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

end SequentialLiteral

-- doc-example: NN/API/Macros.lean :: scoped syntax (name := nnComposeBangLit)
namespace ComposeLiteral

-- The same list spelling for models that are already built, with no monad in the way.
def stack (first : nn.Sequential [4] [8]) (second : nn.Sequential [8] [2]) :
    nn.Sequential [4] [2] :=
  nn.compose![first, second]

end ComposeLiteral

end NN.Tests.API.DocExamples.Neural
