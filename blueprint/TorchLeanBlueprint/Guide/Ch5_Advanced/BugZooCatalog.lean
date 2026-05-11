import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "BugZoo Catalog" =>
%%%
tag := "bugzoo-catalog"
%%%

BugZoo is where we turn "this class of ML bug happens in real systems" into a small, named
TorchLean contract. We built it as a verification catalog, not as a collection of artificial
failures. Each card points at a bug family that has appeared in frameworks, compilers, deployment
tools, or LLM serving stacks, then asks a narrow question: what object would have made the intended
behavior explicit?

The [BugZoo catalog API](https://github.com/lean-dojo/TorchLean/tree/main/NN/Examples/BugZoo/) and
[BugZoo overview](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/README.md) focus on the TorchLean fragment itself:
once a computation enters a typed TorchLean spec, shape changes, masks, token bounds, finite domain
choices, stateful normalization parameters, and backend semantics become named objects that can be
checked.

BugZoo should be read as the smallest public proof-of-concept for TorchLean's philosophy. Each card
starts from a failure mode that can occur in ordinary ML code, then asks what object would have made
the intended behavior explicit.

That is why every BugZoo file is small. A tiny theorem is often a better public contract than a
large demo. It lets us say "this is the exact thing we checked" beside the paper or issue from the
real world that motivated the card.

# Card Anatomy

A good BugZoo card has four parts:

- the pattern in the framework that goes wrong;
- the TorchLean object that names the intended behavior;
- the theorem, structure, or definition that marks the checked boundary;
- the external conformance obligation or unsupported scope that remains outside the checked claim.

This format is useful because most ML bugs are semantic rather than syntactic. The program often
still returns a tensor. The loss may still be a scalar. A compiled graph may still run. An LLM
server may still emit tokens. BugZoo asks whether those tensors and tokens still mean what the user
thought they meant.

The common contract shape is:

$$`\text{bug pattern}
\;\leadsto\;
\text{TorchLean object}
\;\leadsto\;
\text{checked claim}`

That is why the cards are short. The goal is not to recreate an entire framework bug; the goal is
to isolate the semantic invariant that would have made the bug harder to miss.

Some representative contract shapes:

| Card | Contract shape |
|---|---|
| Attention mask | $`j>i\Rightarrow A_{ij}=0` |
| Batch invariance | $`\operatorname{select}(\operatorname{mapBatch}(f,X),i)=f(X_i)` |
| Tokenizer boundary | token ids inhabit `Fin vocabSize` |
| KV cache | appended key/value appears at the final slot |
| Float boundary | runtime Float32 agrees with `IEEE32Exec` under a named agreement |
| Compiler boundary | target output equals source output |
| Stable loss | logits path uses log-softmax semantics |
| Ignored labels | inactive labels contribute zero |

# The Cards

## Shape And Broadcast

[NN.Examples.BugZoo.ShapeAndBroadcast API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/ShapeAndBroadcast.lean)
records one of TorchLean's core design choices: shapes belong in the object, not as an
afterthought. The card uses a missing batch dimension and a reduce then broadcast pattern as its
running examples. In NumPy style tensor libraries, a reduced vector can silently expand back across a
matrix and still produce a plausible loss. In TorchLean, ordinary elementwise operations require the
same shape, and explicit broadcasting carries `Shape.CanBroadcastTo` evidence.

The local contract is deliberately concrete: `addSingletonBatch` names the batch insertion,
`reduceRows` names the reduction, and `broadcastRowToMatrix` names the expansion. The theorem
`broadcastRowToMatrix_firstRow` does not claim to solve every shape bug in Python. It shows the
preferred style: if a dimension is added, removed, or reintroduced, the documentation and proof
script can point at a term that did it.

The empirical motivation is tensor shape fault work such as
[SFData](https://doi.org/10.1145/3533767.3534383), plus
[numerical bug studies](https://conf.researchr.org/details/ase-2022/ase-2022-nier-track/18/An-Empirical-Study-on-Numerical-Bugs-in-Deep-Learning-Programs)
that found bad reductions and accidental broadcasting in real DL programs.

The contract shape is:

$$`\operatorname{add} :
\operatorname{Tensor}(\alpha,s)\to
\operatorname{Tensor}(\alpha,s)\to
\operatorname{Tensor}(\alpha,s)`

and explicit broadcast operations carry evidence that the source shape can be broadcast to the
target shape.

## Stable Loss

[NN.Examples.BugZoo.StableLoss API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/StableLoss.lean) is about losses that
look mathematically harmless but fail numerically. The classic sketch is `softmax` followed by
`log`: if a probability rounds to zero, the log path can produce infinities and downstream NaNs.
TorchLean keeps two APIs separate. Logits should use `crossEntropyLogitsSpec`, which unfolds through
`logSoftmaxSpec`; probability inputs use the clipped probability form `crossEntropySpec`.

The checked hooks are small but important. `crossEntropyLogits_uses_logSoftmax` says that the logits
loss really takes the stable logits path. `crossEntropyProbabilities_clips_before_log` says that the
probability path clamps before `log`. `safeDivSpec_unfold` gives division with domain assumptions an
explicit node protected by epsilon rather than hiding it in an optimizer or backend.

This card is motivated by [TensorFuzz](https://proceedings.mlr.press/v97/odena19a.html), which
targeted rare numerical failures, and by
[empirical studies of numerical bugs](https://conf.researchr.org/details/ase-2022/ase-2022-nier-track/18/An-Empirical-Study-on-Numerical-Bugs-in-Deep-Learning-Programs)
involving `log`, `sqrt`, division, `exp`, and reductions.

## Ignored Labels

[NN.Examples.BugZoo.IgnoredLabelLoss API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/IgnoredLabelLoss.lean) turns a
corner case that is easy to dismiss into a reduction contract.
[PyTorch issue #75181](https://github.com/pytorch/pytorch/issues/75181) reported an `ignore_index`
case where all labels were ignored and the result was `nan`.

The TorchLean lesson is not "we copied every branch of a framework kernel." TorchLean exposes the
policy. `labelContribution false loss = 0` states that ignored labels contribute no scalar loss, and
the card's helper for empty reductions names one possible policy for the all ignored case. That
matters because the bug is not in the idea of ignoring labels; it is in letting the empty active set
pass through an unnamed backend reduction.

The policy is the definition:

$$`\operatorname{labelContribution}(active,loss)
=
\begin{cases}
loss, & active\\
0, & \neg active
\end{cases}`

The all ignored case is then a declared reduction policy, not an accidental division by zero.

## Autograd Domain

[NN.Examples.BugZoo.AutogradDomain API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/AutogradDomain.lean) follows
[PyTorch's own autograd note about division by zero](https://docs.pytorch.org/docs/main/notes/autograd.html#division-by-zero-in-autograd).
If a graph computes `x / 0` and masks the bad value afterward, the forward result may look hidden
while the backward graph still contains the undefined operation.

TorchLean's card names the difference between "divide first, mask later" and "safe divide, then
mask." `maskAfterSafeDiv` records `safedivSpec` before the mask, and
`maskAfterSafeDiv_uses_epsilon_denominator` unfolds to division by `denominator + epsilon`. The
contrast definition, `unsafeDivThenMask`, is kept in the file deliberately so importers and reviewers
can see the risky graph shape rather than treating every masked expression as safe.

## Attention Mask

[NN.Examples.BugZoo.AttentionMask API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/AttentionMask.lean) is the catalog
entry for causal mask semantics. Attention masks fail by polarity, layout, fake negative infinity,
fully masked rows, and interactions with API flags. PyTorch has had relevant reports for
`MultiheadAttention`, including
[`is_causal=True` being ignored when `need_weights=True`](https://github.com/pytorch/pytorch/issues/99282)
and [fully masked heads producing NaNs](https://github.com/pytorch/pytorch/issues/160064) when
weights are requested.

The card connects the runtime convention to the math. Lean's real numbers do not contain literal
negative infinity, so the file first uses `EReal` to state the exact fact:
`exp(-infinity) = 0`. TorchLean's ordinary attention spec then uses the equivalent hard masked
softmax numerator. The theorem `trueInfinityMask_future_attention_weight_zero` says that strict
future positions receive exactly zero attention mass under the causal mask. That is the property we
want when reasoning about autoregressive output causality.

In formula form:

$$`j>i \quad\Longrightarrow\quad
\operatorname{attentionWeight}_{causal}(i,j)=0`

This card also points back to the original transformer paper,
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

## Compiler Boundary

[NN.Examples.BugZoo.CompilerBoundary API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/CompilerBoundary.lean) is the
wrong code card. It is about optimized graphs that run, return tensors, and are nevertheless not
the same computation as the source graph. [NNSmith](https://arxiv.org/abs/2207.13066) found
compiler bugs across TVM, TensorRT, ONNXRuntime, and PyTorch,
[FreeFuzz](https://arxiv.org/abs/2201.06589) found framework/API bugs by mining real snippets, and
a recent [PyTorch compiler correctness study](https://arxiv.org/abs/2604.08720) focuses directly
on silent `torch.compile` wrong outputs.

The local object is the `SemanticBoundary` structure. It has a source evaluator, a target
evaluator, an implementation relation, and a preservation field. This lightweight shape is the
reusable claim behind heavier IR compiler correctness theorems: accepted target code must agree with
the source semantics on every input. A smoke test can make us more confident, but it is not the same
kind of artifact as a semantic boundary.

The reusable statement is:

$$`\operatorname{implements}(source,target)
\quad\Longrightarrow\quad
\forall x,\; target(x)=source(x)`

## Float Boundary

[NN.Examples.BugZoo.FloatBoundary API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/FloatBoundary.lean) is where we
refuse to let proofs over real numbers silently masquerade as float32 deployment guarantees.
Floating point verification attacks show why that matters: a property proved over reals can be
invalidated by finite precision, exceptional values, fused operations, denorm behavior, or changed
reduction order. The card cites
[Jia and Rinard's warning paper](https://doi.org/10.1109/SPW50608.2020.00058).

TorchLean's answer is explicit modeling. `IEEE32Exec` is the executable bit level float32 model.
Runtime `Float32` primitives are not transparent to the Lean kernel, so the theorem
`runtimeFloat32_add_rewrites_to_ieee32` requires the named assumption
`RuntimeFloat32MatchesIEEE32Exec`. We built that friction intentionally. If a proof uses the
float32 model, the boundary says where runtime conformance entered.

The boundary is therefore visible in the theorem shape:

$$`\operatorname{RuntimeFloat32MatchesIEEE32Exec}
\quad\Longrightarrow\quad
\operatorname{runtimeAdd}(x,y)
=
\operatorname{IEEE32Exec.add}(x,y)`

## Normalization State

[NN.Examples.BugZoo.NormalizationState API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/NormalizationState.lean)
covers BatchNorm style bugs where the formula or state is wrong but the layer still emits a tensor.
[CRADLE](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf) reported a
BatchNorm epsilon placement issue across backends, while
[LEMON](https://lingming.cs.illinois.edu/courses/cs598ast-f20/paper-dnn-lib-testing.pdf) found
BatchNormalization moving stat bugs and BatchNorm layers that produce NaNs.

The card splits the concern in two. First, `normalizeCore_scalar_uses_variance_plus_epsilon` shows
that TorchLean's scalar normalization puts epsilon inside the variance term before square root.
Second, `RunningStats` packages inference time mean and variance as explicit inputs. This style is
useful because train/eval mode bugs are often state bugs. If running statistics are ambient mutable
framework state, the proof cannot see them. If they are arguments to `batchNormEvalWithStats`, the
state boundary is visible.

## Batch Invariance

[NN.Examples.BugZoo.BatchInvariance API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/BatchInvariance.lean) is about
serving systems that change outputs depending on which other requests share a batch. This can come
from dynamic batching, kernel selection, reduction order, and scheduling details even when user
randomness is off. The catalog points at recent LLM serving discussions and studies:
[Thinking Machines on inference nondeterminism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
and an [LLM inference engine bug study](https://arxiv.org/abs/2506.09713).

The checked reference semantics is `mapBatch`: apply the same function to each example independently
to each row. The theorem `mapBatch_select_eq_single` says that selecting one row from the batched
result equals evaluating that row alone. This is not a claim that every deployed kernel is
invariant under batching. It is the semantic target a runtime path should refine, with any deliberate
float32 tolerance or reduction order difference stated separately.

## KV Cache

[NN.Examples.BugZoo.KVCache API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/KVCache.lean) models cache accounting in
autoregressive inference. LLM engines fail through shifted caches, wrong cache slots, config/shape
mismatches, resource scheduling, and interaction with positions or tokenizers. The broader source
trail is the [LLM inference engine bug study](https://arxiv.org/abs/2506.09713).

The local contract is intentionally humble. A cache append operation should preserve existing
entries and put the newly decoded key/value in the final slot. That sounds obvious, but it is the
kind of invariant that becomes fragile when a serving stack layers paged attention, batching, and
mutable buffers. The BugZoo card says what the reference operation means before any external cache
manager is trusted to implement it.

## RoPE Position

[NN.Examples.BugZoo.RoPEPosition API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/RoPEPosition.lean) pairs naturally
with the KV cache card. Rotary position embeddings make position accounting part of the model's
meaning. A decode position off by one can be hard to notice because the tensor shapes still line
up and the model still produces tokens.

The file introduces `PositionSchedule` and `appendNextPosition`. The theorem
`appendNextPosition_last` states that the newly appended token gets exactly the next sequence index.
The schedule is explicit rather than derived from ambient mutable state. That is the same design
move as the normalization card: if the state affects semantics, it should appear in the object we
inspect.

## Tokenizer Boundary

[NN.Examples.BugZoo.TokenizerBoundary API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/BugZoo/TokenizerBoundary.lean) marks
the boundary before tensors even reach the model. Tokenizer/config mismatches can disagree about
vocabulary size, padding, EOS, or special token IDs while the neural network code itself looks
ordinary. The [LLM inference engine bug study](https://arxiv.org/abs/2506.09713) lists
tokenizer/config bugs as a real serving class.

TorchLean's current contract is small but valuable: token IDs inside the verified fragment can be
represented as `Fin vocabSize`. `padId_in_vocab` and `tokenAt_in_vocab` are almost tautological,
which means token IDs outside the vocabulary are no longer a late runtime condition once data has
crossed into this representation. The remaining producer step is the importer or tokenizer bridge
that constructs those `Fin` values from external bytes.

# Catalog Value

BugZoo makes proof engineering more concrete. Instead of saying "TorchLean helps with ML bugs," we
can point at a card and ask what changed:

- shape bugs become typed shapes and explicit broadcast evidence;
- unstable losses become named stable specs and operators with explicit domains;
- mask bugs become theorems that future positions have exactly zero weight;
- state bugs become explicit arguments;
- tokenizer and cache bugs become import or append contracts;
- compiler and float bugs become semantic preservation or runtime conformance boundaries.

Future BugZoo entries should follow the same rhythm: cite the real bug family, name the TorchLean
object that captures the intended behavior, and say plainly where the checked statement stops.
