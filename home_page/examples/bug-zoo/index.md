---
title: Bug Zoo
---

# Bug Zoo

Bug Zoo collects mistakes that are easy to miss in ordinary machine-learning tests. The program
still runs and returns a tensor, loss, or token sequence, but the result no longer has the meaning
the caller assumed.

Each example is deliberately small. It states the intended behavior as a Lean definition or
theorem, shows where an implementation can depart from it, and identifies any runtime assumption
that remains outside the proof. Together they cover attention, decoding, data boundaries,
normalization, losses, compilation, floating point, and geometry.

## Attention and Autoregressive Decoding

A causal mask should exclude future keys exactly. Replacing $-\infty$ with a large finite negative
number only approximates that behavior and can fail when logits leave the expected range.
[`AttentionMask.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/AttentionMask.lean)
uses hard-mask semantics and proves that every strict-future attention weight is zero.

Incremental decoding introduces a different problem. A key/value cache must contain the same keys
and values that full-sequence attention would have seen, in the same positions.
[`KVCache.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/KVCache.lean)
checks the append operation, while
[`RoPEPosition.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/RoPEPosition.lean)
records the position assigned to the new token. These contracts isolate the two common off-by-one
errors instead of hiding them inside a generation loop.

## Data, Batches, and Normalization State

Tokenizer errors often appear much earlier than the model. A checkpoint may expect one vocabulary
or special-token convention while the data loader supplies another.
[`TokenizerBoundary.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/TokenizerBoundary.lean)
requires imported token ids to inhabit `Fin vocabularySize`, making the vocabulary bound part of the
object passed to the network. Tokenizer identity and special-token meanings still need their own
agreement checks.

Batching should normally change throughput, not the prediction for an individual sample.
[`BatchInvariance.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/BatchInvariance.lean)
states that selecting a row from a batched reference run agrees with evaluating that row alone.
The theorem concerns independent row-wise evaluation; connecting a batching kernel to that
reference remains a separate obligation.

Normalization has its own hidden state. Batch normalization uses learned affine parameters and
running statistics at inference time; layer normalization has a degenerate one-feature case that
is easy to mishandle. The normalization examples make the axes, epsilon placement, running state,
and zero-gradient corner cases explicit. These exact normalization identities are over the reals;
floating kernels can have nonzero residuals. The BatchNorm theorem chooses one scale and bias from
fixed inference statistics and proves the same affine map works for every input.

## Losses, Floating Point, and Compilation

Several examples concern computations that are mathematically familiar but numerically unsafe.
Masking a quotient after division does not repair a division by zero, and a direct implementation
of a logit loss can overflow even when its stable form is finite. `AutogradDomain.lean` and
`StableLoss.lean` expose the denominator policy and stable formula before reverse mode is considered.
Adding epsilon shifts a denominator; it does not make every input safe, since a denominator of
`-epsilon` still becomes zero.

Real-number proofs do not automatically describe a binary32 run.
[`FloatBoundary.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/FloatBoundary.lean)
uses FloatLib's native import/export, finite-input addition/subtraction, and square-root theorems
directly. `CompilerBoundary.lean` proves that successful lowering of
the supported IR fragment preserves node denotations in a Lean reference evaluator. Its hypothesis
excludes raw logarithm nodes; external compiler and native-kernel conformance remain separate obligations.

The final geometry case starts from tensors exported by a detector. Lean recomputes camera
projection and positive depth, then checks that the reported two-dimensional box encloses every
projected corner. The detector remains an external producer; the enclosure claim does not.

## Run the Examples

Build the complete collection with:

```bash
lake build NN.Examples.BugZoo.All
```

The geometry case has a registered certificate checker:

```bash
lake exe verify -- camera-box3d-cert
```

The other entries are inspected through their checked Lean definitions and theorems. The separate
`lake exe verify -- all` command runs the ten bundled checks opted in through `includeInAll`.
External, interactive, and longer workflows are excluded; `verify -- list` shows the complete registry.

For a runnable PyTorch comparison of one-feature LayerNorm and constant normalization slices:

```bash
python3 scripts/verification/normalization_contract_probe.py --device cpu
```

This optional probe requires PyTorch. It prints version metadata and the output/input-gradient/
scale-gradient residuals from the real-valued reference; `--device cuda` selects an available GPU.
Residuals depend on the installed provider and hardware and are not proof certificates.

The source files and the contracts they expose are listed below.

| Source file | Bug family | Contract exposed |
| --- | --- | --- |
| `AttentionMask.lean` | Causal masks, mask polarity, finite sentinels standing in for $-\infty$ | Future positions receive exactly zero attention weight under hard-mask semantics. |
| `KVCache.lean` | Shifted or malformed key/value caches in autoregressive decoding | The appended key/value vector is exactly the final cache entry. |
| `RoPEPosition.lean` | Off-by-one or mismatched rotary/absolute positions | Appending a token assigns the next sequence position. |
| `TokenizerBoundary.lean` | Vocabulary-size and special-token mismatches | Imported token ids inhabit `Fin vocabularySize`. |
| `BatchInvariance.lean` | Dynamic batching changing per-sample outputs | Selecting one row from a batched reference run equals evaluating that row alone. |
| `NormalizationState.lean` | BatchNorm formula/state mistakes | Epsilon placement and eval-time running statistics are explicit objects. |
| `LayerNormDegenerateAxis.lean` | One-feature LayerNorm corner cases | Over the reals, the output is the bias, with zero input and scale-gradient contribution. |
| `ConstantNormalizationSlice.lean` | Cancellation in normalization kernels on constant slices | Over the reals, affine normalization returns the bias and contributes zero scale gradient. |
| `IgnoredLabelLoss.lean` | All-ignored cross-entropy reductions | Ignored labels and the empty-reduction policy are named. |
| `AutogradDomain.lean` | Masking after undefined division | The reference expression shifts the denominator by epsilon before masking. |
| `StableLoss.lean` | Numerically unstable losses and domain-sensitive ops | Logit losses use the stable log-softmax path. |
| `ShapeAndBroadcast.lean` | Missing axes and silent broadcasts | Dimension changes are explicit terms with shape evidence. |
| `CompilerBoundary.lean` | Optimized graphs silently changing semantics | Successful supported IR lowering preserves reference node denotations, assuming no raw logarithm nodes. |
| `FloatBoundary.lean` | Real-valued reasoning applied to Float32 runs | FloatLib proves native round trips, finite-input add/sub agreement, and total square-root agreement through native export; configured division has a separate software-model refinement. |
| `Geometry3DProjection.lean` | Camera convention, depth, layout, and projection-box errors | The checker recomputes projection, positive depth, and 2D box enclosure. |

## Two Checked Statements

Under hard-mask semantics, every strict-future key receives exactly zero attention weight.
The source theorem has this signature:

```text
theorem trueInfinityMask_future_attention_weight_zero
    {n : Nat} (scores : TorchLean.Tensor ℝ [n, n])
    (i j : Fin n) (hij : i.val < j.val) :
    (Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask n))[(i, j)] = 0
```

Lean 4.34.0 defines core `Float32` operations through `Float32.Model`. The scalar connection comes
from FloatLib directly. This complete example proves addition agreement for finite operands:

```lean
import FloatLib.Floats.Formats.IEEE754
open FloatLib.Floats

example (a b : Float32) (ha : a.isFinite = true) (hb : b.isFinite = true) :
    ExecFloat.Binary.ofFloat32 (a + b) =
      ExecFloat.Binary.ofFloat32 a + ExecFloat.Binary.ofFloat32 b :=
  ExecFloat.Binary.ofFloat32_add_of_isFinite a b ha hb
```

The finite inputs include signed zeros and subnormals. No premise requires the result to be finite,
so overflow is included. A NaN or infinite input lies outside this addition theorem. The separate
square-root export theorem covers every configured input; it canonicalizes NaNs at the native
boundary because Lean stores one NaN representation. These logical equalities do not certify
compiled CPU instructions, CUDA kernels, or a library's reduction order.
