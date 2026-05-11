# TorchLean BugZoo

BugZoo is a collection of small checked case studies for common neural-network failure modes. Each
card starts from a real class of bugs, then asks what the corresponding TorchLean contract should be.

Many ML bugs are not crashes. The code still returns tensors, logits, losses, or tokens, but the
computation no longer means what the user intended. BugZoo turns those cases into small artifacts: a
shape contract, a mask theorem, a cache invariant, a tokenizer boundary, a Float32 bridge, or a
compiler-preservation obligation.

## Why These Bugs Matter

The useful TorchLean point is not "Python cannot run neural nets." Python can run them very well.
The problem is that many serious ML bugs still return tensors, losses, logits, or tokens.
They are semantic bugs: the code runs, but it is no longer the math people think they deployed.

| BugZoo file | Real bug family | What Lean makes explicit |
| --- | --- | --- |
| `NormalizationState.lean` | CRADLE reported a CNTK/Keras BatchNorm backend bug where epsilon was placed outside the square root; LEMON found BatchNormalization moving-stat bugs and NaN-producing BatchNorm layers. | The BatchNorm formula, epsilon placement, and eval-time running statistics become explicit Lean objects instead of ambient framework state. |
| `AttentionMask.lean` | PyTorch `MultiheadAttention` has had causal mask and fully masked head bugs around `need_weights`; attention masks are also easy to get wrong by polarity, layout, or finite sentinel values standing in for `-∞`. | The causal mask theorem says strict future positions get exactly zero attention weight under the stated mask semantics. |
| `BatchInvariance.lean` | Production LLM inference can change outputs when dynamic batching changes reduction behavior, even with randomness disabled. | The reference theorem states that selecting one example from a batched run equals evaluating that example alone. Runtime kernels can be checked against this target. |
| `KVCache.lean` | LLM engines have real cache management bugs: shifted caches, wrong positions, shape/config mismatches, and resource/cache scheduling faults. | Cache append specs prove the newly appended key/value is exactly the final cache entry. |
| `RoPEPosition.lean` | LLM inference engine studies report RoPE/position mismatches alongside cache and tokenizer/config bugs. | Decode positions are explicit schedules; appending a token assigns the next position by theorem, not by ambient mutable state. |
| `TokenizerBoundary.lean` | Tokenizer/config mismatches can silently disagree about vocabulary size or special token IDs before tensors ever reach the model. | Token IDs are represented as `Fin vocabSize`, making out of vocabulary IDs unrepresentable inside the verified fragment. |
| `CompilerBoundary.lean` | DL compiler studies and a PyTorch compiler study report silent wrong outputs from optimized graphs without crashes or warnings. | Import/export boundaries become contracts: accepted backends must preserve source op semantics, shapes, dtypes, weights, and buffers. |
| `ShapeAndBroadcast.lean` | Tensor shape faults are common, and many wrong broadcasts do not crash. | Shape indexed tensors make intended axes part of the type/spec instead of relying on late runtime checks. |
| `StableLoss.lean` | TensorFuzz targeted rare numerical failures and broken losses; numerical studies repeatedly find bad domains for `log`, `sqrt`, division, and reductions. | Stable loss and domain sensitive ops become named specs with stated finite value obligations. |
| `IgnoredLabelLoss.lean` | PyTorch issue #75181 reported `CrossEntropyLoss(ignore_index=...)` returning `nan` for an all-ignored target case. | Ignored labels become explicit zero contributions, and the empty-reduction policy is named instead of hidden in a backend kernel. |
| `AutogradDomain.lean` | PyTorch's autograd docs show that masking after `x / 0` can still leave `nan` gradients because the undefined division remains in the backward graph. | The safe domain graph records epsilon protected division before masking, so importers can distinguish it from "divide first, mask later." |
| `FloatBoundary.lean` | Floating point verification attacks show that real valued proofs do not automatically imply deployed Float32 behavior. | Runtime Float32 operations are tied back to explicit IEEE style semantics rather than silently borrowing real number reasoning. |
| `Geometry3DProjection.lean` | 3D perception glue fails through camera convention mismatches, negative depth, swapped `xyxy`/axis layouts, malformed corner tensors, and projected 3D boxes that do not actually enclose their 2D claims. | The Geometry3D checker recomputes tensor native projection, checks positive depth and bbox enclosure, includes a theorem for homogeneous projection intervals, and renders accepted/rejected PNG overlays for human inspection. |

## Contract Shapes

| Bug family | Contract shape |
| --- | --- |
| Causal attention | future positions receive zero attention weight |
| KV cache | appended key/value appears at the final cache slot |
| RoPE position | appended token receives the next position |
| Tokenizer boundary | token ids inhabit `Fin vocabSize` |
| Batch invariance | selecting a row from batched evaluation equals evaluating that row alone |
| Compiler boundary | target graph output equals source graph output |
| Float boundary | runtime Float32 agrees with the named IEEE-style model under stated assumptions |
| Stable loss | logits loss uses the stable log-softmax path |
| Geometry3D projection | projected 3D corners have positive depth and enclose the claimed 2D box |

Scope: this BugZoo pass does not verify distributed training
setups, NCCL/collective semantics, paged attention allocators, mixed quantization, or arbitrary CUDA
kernels. Those can be separate boundary cards, but they are not part of the checked TorchLean scope
shown here.

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
- [DL compiler bug studies](https://haoyang9804.github.io/papers/fse21.pdf): wrong code bugs are
  common enough to need semantic oracles, not just crash tests.
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

## Reading guide

BugZoo files should read like small case studies. Each file should answer, in order:

- what real bug family is being modeled;
- what the bad framework side pattern looks like;
- what exact TorchLean object is the trusted contract;
- what the theorem proves, and what it does not prove.

That keeps the prose explanation close to the checked Lean artifact without claiming more than the
example proves.
