# TorchLean BugZoo

BugZoo is a collection of small checked case studies for common neural-network failure modes. Each
example starts from a real class of bugs, then asks what the corresponding TorchLean contract should
be.

Many ML bugs are not crashes. The code still returns tensors, logits, losses, or tokens, but the
computation no longer means what the user intended. BugZoo turns those cases into small artifacts: a
shape contract, a mask theorem, a cache invariant, a tokenizer boundary, a Float32 bridge, or a
compiler-preservation obligation.

Build all fifteen case studies from the repository root with:

```bash
lake build NN.Examples.BugZoo.All
```

Then open an individual file in Lean's Infoview to inspect its definitions and theorem statements.
These are checked case studies rather than training executables. The geometry checker also runs
with `lake exe verify -- camera-box3d-cert`; `verify -- all` runs the broader registered
verification suite, not every BugZoo file.

With PyTorch installed, compare native normalization results against the constant-slice reference:

```bash
python3 scripts/verification/normalization_contract_probe.py --device cpu
```

The probe prints version metadata and residuals for outputs and gradients across two dtypes and
four input magnitudes. Use `--device cuda` to inspect a GPU implementation. These are measurements
of the installed provider, separate from the Lean theorems.

## Bug Families And Contracts

| BugZoo file | Real bug family | What Lean makes explicit |
| --- | --- | --- |
| `NormalizationState.lean` | CRADLE reported a CNTK/Keras BatchNorm backend bug where epsilon was placed outside the square root; LEMON found BatchNormalization moving-stat bugs and NaN-producing BatchNorm layers. | The BatchNorm formula, epsilon placement, and eval-time running statistics become explicit Lean objects instead of ambient framework state. |
| `AttentionMask.lean` | PyTorch `MultiheadAttention` has had causal mask and fully masked head bugs around `need_weights`; attention masks are also easy to get wrong by polarity, layout, or finite sentinel values standing in for $-\infty$. | The causal mask theorem says strict future positions get exactly zero attention weight under the stated mask semantics. |
| `BatchInvariance.lean` | Production LLM inference can change outputs when dynamic batching changes reduction behavior, even with randomness disabled. | The reference theorem states that selecting one example from a batched run equals evaluating that example alone. Runtime kernels can be checked against this target. |
| `KVCache.lean` | LLM engines have real cache management bugs: shifted caches, wrong positions, shape/config mismatches, and resource/cache scheduling faults. | Cache append specs prove the newly appended key/value is exactly the final cache entry. |
| `RoPEPosition.lean` | LLM inference engine studies report RoPE/position mismatches alongside cache and tokenizer/config bugs. | Decode positions are explicit schedules; appending a token assigns the next position by theorem, not by ambient mutable state. |
| `TokenizerBoundary.lean` | Tokenizer/config mismatches can silently disagree about vocabulary size or special token IDs before tensors ever reach the model. | Token IDs are represented as `Fin vocabularySize`, making out of vocabulary IDs unrepresentable inside the verified fragment. |
| `CompilerBoundary.lean` | DL compiler studies and a PyTorch compiler study report silent wrong outputs from optimized graphs without crashes or warnings. | Successful supported IR lowering preserves reference node denotations, assuming no raw logarithm nodes; external compilers need separate conformance evidence. |
| `ShapeAndBroadcast.lean` | Tensor shape faults are common, and many wrong broadcasts do not crash. | Shape indexed tensors make intended axes part of the type/spec instead of relying on late runtime checks. |
| `StableLoss.lean` | TensorFuzz targeted rare numerical failures and broken losses; numerical studies repeatedly find bad domains for `log`, `sqrt`, division, and reductions. | Stable loss and domain sensitive ops become named specs with stated finite value obligations. |
| `IgnoredLabelLoss.lean` | PyTorch issue #75181 reported `CrossEntropyLoss(ignore_index=...)` returning `nan` for an all-ignored target case. | Ignored labels become explicit zero contributions, and the empty-reduction policy is named instead of hidden in a backend kernel. |
| `AutogradDomain.lean` | PyTorch's autograd docs show that masking after `x / 0` can still leave `nan` gradients because the undefined division remains in the backward graph. | The reference expression shifts the denominator by epsilon before masking; nonzero denominators and finite runtime values remain obligations. |
| `FloatBoundary.lean` | Floating-point verification attacks show why real-valued proofs need an explicit bridge before they describe deployed `Float32` behavior. | FloatLib proves agreement with configured software arithmetic for Lean's logical `Float32.Model`: addition/subtraction on finite operands and square root modulo NaN canonicalization. Compiled CPU and CUDA execution require separate conformance evidence. |
| `LayerNormDegenerateAxis.lean` | Backend LayerNorm kernels can mishandle the `normalized_shape=(1,)` case even though the math is a constant function. | The one-feature LayerNorm contract says the output is the bias, with zero input gradient and zero scale gradient; the repro script checks PyTorch against this contract. |
| `ConstantNormalizationSlice.lean` | GroupNorm, InstanceNorm, and BatchNorm kernels can suffer cancellation on large constant slices even when their saved mean/rstd imply zero normalized activation. | The constant-slice normalization theorem says affine normalization returns the bias and has zero scale-gradient contribution. |
| `Geometry3DProjection.lean` | 3D perception glue fails through camera convention mismatches, negative depth, swapped `xyxy`/axis layouts, malformed corner tensors, and projected 3D boxes that do not actually enclose their 2D claims. | The Geometry3D checker recomputes tensor native projection, checks positive depth and bbox enclosure, includes a theorem for homogeneous projection intervals, and renders accepted/rejected PNG overlays for human inspection. |

## Source Trail

The case studies are motivated by published bug studies and systems reports:

- [CRADLE](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf):
  cross backend validation found backend semantic bugs including BatchNorm epsilon placement.
- [LEMON](https://lingming.cs.illinois.edu/courses/cs598ast-f20/paper-dnn-lib-testing.pdf):
  model generation testing found library inconsistencies, moving stat errors, and NaN bugs localized
  to BatchNormalization.
- [TensorFuzz](https://arxiv.org/abs/1807.10875): coverage guided fuzzing targeted rare numerical
  errors and neural network debugging failures.
- [PyTorch MultiheadAttention issue #99282](https://github.com/pytorch/pytorch/issues/99282):
  `is_causal=True` was reported ignored when `need_weights=True`.
- [PyTorch MultiheadAttention issue #160064](https://github.com/pytorch/pytorch/issues/160064):
  fully masked attention heads were reported to produce NaNs when attention weights were requested.
- [PyTorch CrossEntropyLoss issue #75181](https://github.com/pytorch/pytorch/issues/75181):
  an `ignore_index` all-ignored-label case was reported returning `nan`.
- [PyTorch autograd notes](https://docs.pytorch.org/docs/main/notes/autograd.html#division-by-zero-in-autograd):
  masking after division by zero does not remove the undefined operation from the backward graph.
- [FreeFuzz](https://github.com/ise-uiuc/FreeFuzz) and
  [NNSmith](https://arxiv.org/abs/2207.13066): fuzzing and generated valid models expose
  framework/API and compiler semantic mismatches.
- [DL compiler bug studies](https://haoyang9804.github.io/papers/fse21.pdf): wrong-code bugs call
  for semantic oracles alongside crash tests.
- [PyTorch compiler correctness study](https://arxiv.org/abs/2604.08720): silent `torch.compile`
  correctness bugs can produce incorrect outputs without exceptions, crashes, or warnings.
- [Mobile deployment fault studies](https://arxiv.org/abs/2101.04930): model conversion fails
  through shapes, tensor names, unsupported ops, version mismatches, and registration details.
- [LLM inference engine bug studies](https://arxiv.org/abs/2506.09713): serving stacks fail through
  cache, RoPE, tokenizer/config, resource, batching, and multi device boundaries.
- [Batch invariant inference work](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/):
  reproducibility can require proving outputs are independent of server batch composition.
- 3D projection glue reports including
  [PyTorch3D #522](https://github.com/facebookresearch/pytorch3d/issues/522),
  [PyTorch3D #596](https://github.com/facebookresearch/pytorch3d/issues/596),
  [PyTorch3D #1105](https://github.com/facebookresearch/pytorch3d/issues/1105),
  [PyTorch3D #1183](https://github.com/facebookresearch/pytorch3d/issues/1183),
  [PyTorch3D #1427](https://github.com/facebookresearch/pytorch3d/issues/1427),
  [Detectron2 #2402](https://github.com/facebookresearch/detectron2/issues/2402),
  [Omni3D #60](https://github.com/facebookresearch/omni3d/issues/60), and
  [BlenderProc #1150](https://github.com/DLR-RM/BlenderProc/issues/1150):
  camera conventions, tensor layouts, and bbox projection checks are a real boundary problem,
  not a synthetic TorchLean only example.
