---
title: Bug Zoo
---

# Bug Zoo

Here we'll look at mistakes that are easy to miss in ordinary machine-learning tests. Our program
may still return a tensor, loss, or token sequence even when it computes something different from
what we intended.

We'll keep each example small: state the intended behavior in Lean, examine how an implementation
can depart from it, and identify any runtime assumptions left outside the proof. Between them, the
examples cover attention, decoding, data boundaries, normalization, losses, compilation, floating
point, and geometry.

## Attention and Autoregressive Decoding

Let's start with causal attention, where we want the mask to exclude future keys exactly. Replacing
$-\infty$ with a large finite negative number only approximates that behavior, and it can fail when
logits leave the expected range.
[`AttentionMask.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/BugZoo/AttentionMask.lean)
uses hard-mask semantics and proves that every strict-future attention weight is zero.

Incremental decoding has a different problem: a key/value cache must contain the same keys and
values that full-sequence attention would have seen, in the same positions.
[`KVCache.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/BugZoo/KVCache.lean)
checks the append operation, while
[`RoPEPosition.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/BugZoo/RoPEPosition.lean)
records the position assigned to the new token, so we can check the cache update and the token
position separately from the generation loop.

## Data, Batches, and Normalization State

Before a token reaches our model, we need to check how it was encoded: a checkpoint may expect one
vocabulary or special-token convention while the data loader supplies another.
[`Tokenizer.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/BugZoo/Tokenizer.lean)
requires imported token ids to inhabit `Fin vocabularySize` (which makes the vocabulary bound part
of the object passed to the network). Tokenizer identity and special-token meanings still need
their own agreement checks.

Batching should normally change throughput, not the prediction for an individual sample.
[`Tensor.unstack_mapLeading`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/Tensor/Operations.lean)
states that selecting a row from a batched reference run agrees with evaluating that row alone.
That theorem concerns independent row-wise evaluation, and connecting a batching kernel to that
reference remains a separate obligation.

Normalization has its own hidden state. Batch normalization uses learned affine parameters and
running statistics at inference time, and layer normalization has a degenerate one-feature case
that is easy to mishandle. Our normalization examples spell out the axes, epsilon placement,
running state, and zero-gradient corner cases. These exact normalization identities hold over the
reals (floating kernels can have nonzero residuals). Our BatchNorm theorem chooses one scale and
bias from fixed inference statistics and proves the same affine map works for every input.

## Losses, Floating Point, and Compilation

Familiar formulas can also fail when we evaluate them numerically. Masking a quotient after
division doesn't repair a division by zero, and a direct implementation of a logit loss can
overflow even when its stable form is finite. `AutogradDomain.lean` and `StableLoss.lean` expose
the denominator policy and stable formula before reverse mode is considered. Adding epsilon shifts
a denominator, but it doesn't make every input safe (a denominator of `-epsilon` still becomes
zero).

Real-number proofs do not automatically describe a binary32 run.
FloatLib's native import/export, finite-input addition/subtraction, and square-root theorems
relate the two directly. External compiler and native-kernel conformance remain separate
obligations.

Some formulas fail from rounding alone.
[`RoundingBugs.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/DeepDives/Floats/RoundingBugs.lean)
takes standard formulas such as softmax cross-entropy at the logits `[9, -9, -9]` in binary32, the
cosine loss, and the probability clamp at `1 - epsilon`. For each one, the kernel checks an input
on which it fails, and a theorem covers the stable formula TorchLean computes.
[`LibraryBugs.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/DeepDives/Floats/LibraryBugs.lean)
and its two continuations do the same for formulas from PyTorch, JAX, Hugging Face Transformers,
TRL, Stable-Baselines3, diffusers, PyTorch Geometric, TensorFlow Probability, torchvision and
xLSTM, each beside a formula with the same real value that doesn't fail there. These files check
the formulas as written in Lean, but they don't establish that an installed library executes
exactly these operations.

Last comes the geometry case, which starts from tensors exported by a detector. Lean recomputes
camera projection and positive depth, then checks that the reported two-dimensional box (expanded
by its stated tolerance) encloses every projected corner. That's a check of the exported geometry,
not of the detector's ability to identify the right object.

## Run the Examples

We can check all the Lean examples by building the collection:

```bash
scripts/lake.sh build TorchLeanExamples
```

For the geometry case, run its registered certificate checker:

```bash
scripts/lake.sh exe verify -- camera-box3d-cert
```

You inspect the other entries through their checked Lean definitions and theorems. A separate
command, `scripts/lake.sh exe verify -- all`, runs the five bundled checks opted in through
`includeInAll` and leaves out external, interactive, and longer workflows (`verify -- list` shows
the complete registry).

Here are the source files and the contracts they expose.

| Source file | Bug family | Contract exposed |
| --- | --- | --- |
| `AttentionMask.lean` | Causal masks, mask polarity, finite sentinels standing in for $-\infty$ | Future positions receive exactly zero attention weight under hard-mask semantics. |
| `KVCache.lean` | Shifted or malformed key/value caches in autoregressive decoding | The appended key/value vector is exactly the final cache entry. |
| `RoPEPosition.lean` | Off-by-one or mismatched rotary/absolute positions | Appending a token assigns the next sequence position. |
| `Tokenizer.lean` | Vocabulary-size and special-token mismatches | Imported token ids inhabit `Fin vocabularySize`. |
| `BatchNorm.lean` | BatchNorm formula/state mistakes | Epsilon placement and eval-time running statistics are explicit objects. |
| `LayerNormAxis.lean` | One-feature LayerNorm corner cases | Over the reals, one-feature LayerNorm outputs the bias, and its backward rule returns zero scale and input gradients. |
| `ConstantSlice.lean` | Cancellation in normalization kernels on constant slices | Over the reals, `Tensor.batchNorm` of the constant tensor `x` with running mean `x` and running variance zero returns the bias, and a scalar identity gives a zero scale-gradient term. |
| `IgnoredLabel.lean` | All-ignored cross-entropy reductions | Ignored labels and the empty-reduction policy are named. |
| `AutogradDomain.lean` | Masking after undefined division | The reference expression shifts the denominator by epsilon before masking. |
| `StableLoss.lean` | Numerically unstable losses and domain-sensitive ops | Logit losses use the stable log-softmax path. |
| `Broadcasting.lean` | Missing axes and silent broadcasts | Dimension changes are explicit terms with shape evidence. |

## Two Checked Statements

Let's look more closely at two of the statements. For hard-masked attention, we prove that every
strict-future key receives exactly zero weight:

```text
theorem TorchLean.Softmax.hardMasked_causalMask_eq_zero
    {n : Nat} (scores : TorchLean.Tensor ℝ [n, n])
    (i j : Fin n) (hij : i.val < j.val) :
    TorchLean.Tensor.get2 (TorchLean.Softmax.hardMasked scores (TorchLean.Attention.causalMask n)) i j = 0
```

Lean defines core `Float32` operations through `Float32.Model`. FloatLib supplies the scalar
connection, and we can use its theorem to prove addition agreement for finite operands:

```lean
import FloatLib.Floats.Formats.IEEE754
open FloatLib.Floats

example (a b : Float32) (ha : a.isFinite = true) (hb : b.isFinite = true) :
    ExecFloat.Binary.ofFloat32 (a + b) =
      ExecFloat.Binary.ofFloat32 a + ExecFloat.Binary.ofFloat32 b :=
  ExecFloat.Binary.ofFloat32_add_of_isFinite a b ha hb
```

Signed zeros and subnormals count as finite inputs, and no premise requires the result to be
finite, so overflow is included. A NaN or infinite input lies outside this addition theorem. Square
root has its own export theorem, which covers every configured input and canonicalizes NaNs at the
native boundary because Lean stores one NaN representation. These logical equalities don't certify
compiled CPU instructions, CUDA kernels, or a library's reduction order.
