# TorchLean BugZoo

BugZoo collects small checked case studies of common neural-network failure modes. Each one starts
from a real class of bugs and states the TorchLean contract for it.

Many ML bugs are not crashes. The code returns tensors, logits, losses, or tokens, but they do not
mean what the user intended. BugZoo turns those cases into small artifacts: a shape contract, a mask
theorem, a cache invariant, or a tokenizer boundary. The Compiler Boundary section of the
[BugZoo guide chapter][guide] covers compiler preservation, with the verifier's lowering theorem
`IR.lowerForward_eq_ok`.

[guide]: ../../home_page/blueprint/TorchLeanBlueprint/Guide/Ch5_Applications/BugZooCatalog.lean

Build every case study, with the other examples, from the repository root with:

```bash
scripts/lake.sh build TorchLeanExamples
```

Then open a file in Lean's Infoview to read its definitions and theorem statements. The case
studies are checked when they build and have no executables of their own. The geometry checker
also runs as `scripts/lake.sh exe verify -- camera-box3d-cert`, which `verify -- all` includes.

## Bug Families And Contracts

| BugZoo file | Real bug family | TorchLean contract |
| --- | --- | --- |
| `BatchNorm.lean` | CRADLE reported a CNTK/Keras BatchNorm backend bug where epsilon was placed outside the square root. LEMON found BatchNormalization moving-stat bugs and NaN-producing BatchNorm layers. | The BatchNorm formula, epsilon placement, and eval-time running statistics become explicit Lean objects instead of ambient framework state. |
| `AttentionMask.lean` | PyTorch `MultiheadAttention` has had causal mask and fully masked head bugs around `need_weights`. Attention masks are also easy to get wrong by polarity, layout, or finite sentinel values standing in for $-\infty$. | The causal mask theorem says strict future positions get exactly zero attention weight under the stated mask semantics. |
| `KVCache.lean` | LLM engines have real cache management bugs: shifted caches, wrong positions, shape/config mismatches, and resource/cache scheduling faults. | Cache append specs prove the newly appended key/value is exactly the final cache entry. |
| `RoPEPosition.lean` | LLM inference engine studies report RoPE/position mismatches alongside cache and tokenizer/config bugs. | Decode positions are explicit schedules: appending a token assigns the next position by theorem, not by ambient mutable state. |
| `Tokenizer.lean` | Tokenizer/config mismatches can silently disagree about vocabulary size or special token IDs before tensors ever reach the model. | Token IDs are represented as `Fin vocabularySize`, making out of vocabulary IDs unrepresentable inside the verified fragment. |
| `Broadcasting.lean` | Tensor shape faults are common, and many wrong broadcasts do not crash. | Shape indexed tensors make intended axes part of the type/spec instead of relying on late runtime checks. |
| `StableLoss.lean` | TensorFuzz targeted rare numerical failures and broken losses. Numerical studies repeatedly find bad domains for `log`, `sqrt`, division, and reductions. | Stable loss and domain sensitive ops become named specs with stated finite value obligations. |
| `IgnoredLabel.lean` | PyTorch issue #75181 reported `CrossEntropyLoss(ignore_index=...)` returning `nan` for an all-ignored target case. | A local helper makes an ignored label contribute zero and names one empty-reduction policy, an epsilon-shifted mean. TorchLean's cross-entropy losses take no ignore mask. |
| `AutogradDomain.lean` | PyTorch's autograd docs show that masking after `x / 0` can still leave `nan` gradients because the undefined division remains in the backward graph. | The reference expression shifts the denominator by epsilon before masking. Nonzero denominators and finite runtime values remain obligations. |
| `LayerNormAxis.lean` | Backend LayerNorm kernels can mishandle the `normalized_shape=(1,)` case even though the math is a constant function. | The one-feature LayerNorm contract says the output is the bias, with zero input gradient and zero scale gradient. |
| `ConstantSlice.lean` | GroupNorm, InstanceNorm, and BatchNorm kernels can suffer cancellation on large constant slices even when their saved mean/rstd imply zero normalized activation. | Over the reals, `Tensor.batchNorm` on a tensor whose entries all equal `x`, with running mean `x` and running variance zero, returns the bias in every entry. On the same input, the reverse rule `BatchNorm.runningBackward` gives a zero scale gradient. |

The [Geometry3D example](../Verification/Geometry3D/README.md) is the 3D camera-projection case
study. Its bug family is 3D perception glue: camera convention mismatches, negative depth, swapped
`xyxy` layouts, malformed corner tensors, and projected boxes that miss their 2D claims. In
`TorchLeanExamples/Verification/Geometry3D/Box3D.lean`, `checkCert_sound` proves that an accepted
certificate satisfies `Verified`: positive image size, an ordered bbox inside the image,
positive depths, and projected corners inside the image and, up to `tol`, inside the bbox. Over an
ordered field, `ofHomogeneous_inside_bbox` takes intervals for
the homogeneous camera coordinates. If the numerator intervals are nonnegative, the depth interval
has a positive lower end and the quotient intervals lie inside the claimed box, then every
`(uNum / z, vNum / z)` with values in those intervals lies inside the box.

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
- [PyTorch autograd
  notes](https://docs.pytorch.org/docs/main/notes/autograd.html#division-by-zero-in-autograd):
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
