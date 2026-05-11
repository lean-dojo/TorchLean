---
title: Bug Zoo Walkthrough
usemathjax: true
---

Bug Zoo is TorchLean’s collection of small case studies for semantic bugs that can pass ordinary
runtime checks. These examples focus on cases where code still returns tensors, losses, or tokens,
but the returned value no longer satisfies the intended contract.

Bug Zoo is the easiest place to understand the motivation for TorchLean: many ML failures are not
type errors or crashes, but silent changes in meaning.

Each card starts with a bug pattern, then isolates the small mathematical contract that would have
made the intended behavior explicit. Some cards make a mistake unrepresentable in the checked
fragment. Others turn the mistake into a theorem obligation or a runtime agreement that has to be
named.

## A Few Case Studies

<div class="showcase-grid bug-zoo-grid">
  <a class="showcase-card showcase-image-card" href="{{ '/docs/NN/Examples/BugZoo/AttentionMask.html' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/attention-masks.png' | relative_url }}" alt="Attention mask Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Attention Masks</span>
      <span class="showcase-text">Checks that masked future positions receive exactly zero attention weight under the stated causal mask semantics.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/docs/NN/Examples/BugZoo/KVCache.html' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/kv-cache-rope.png' | relative_url }}" alt="KV cache and RoPE Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">KV Cache and RoPE</span>
      <span class="showcase-text">Makes the cache-position contract explicit so incremental decoding agrees with the intended full-sequence computation.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/docs/NN/Examples/BugZoo/TokenizerBoundary.html' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/tokenizer-boundaries.png' | relative_url }}" alt="Tokenizer boundary Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Tokenizer Boundaries</span>
      <span class="showcase-text">Separates byte/token assumptions from model assumptions, so text preprocessing cannot silently change the checked claim.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/docs/NN/Examples/BugZoo/NormalizationState.html' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/normalization-state.png' | relative_url }}" alt="Normalization state Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Normalization State</span>
      <span class="showcase-text">Tracks which statistics are training state, inference state, or explicit inputs rather than treating normalization as a black box.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/docs/NN/Examples/BugZoo/BatchInvariance.html' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/batch-invariance.png' | relative_url }}" alt="Batch invariance Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Batch Invariance</span>
      <span class="showcase-text">States when processing one sample alone should agree with processing it as part of a batch.</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/docs/NN/Examples/BugZoo/FloatBoundary.html' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/bug-zoo/float-autograd-boundaries.png' | relative_url }}" alt="Float and autograd boundaries Bug Zoo case study"/>
    <span class="showcase-body">
      <span class="showcase-title">Float and Autograd Boundaries</span>
      <span class="showcase-text">Shows how runtime `Float32` and reverse-mode claims are connected through named assumptions and proof-facing statements.</span>
    </span>
  </a>
</div>

## What “A Checked Claim” Looks Like Here

Each case study should end in a precise statement.

Here is the attention-mask claim in one line: strict-future keys get exactly zero attention weight
under the stated causal mask semantics.

```lean
theorem trueInfinityMask_future_attention_weight_zero :
  Spec.get2 (Spec.hardMaskedSoftmaxSpec scores (Spec.causalMask n)) i j = 0
```

And here is the Float32 boundary claim: runtime arithmetic rewrites to the explicit `IEEE32Exec`
model only under a named assumption.

```lean
theorem runtimeFloat32_add_rewrites_to_ieee32
    [RuntimeFloat32MatchesIEEE32Exec] (a b : F32) :
    toIEEE32Exec (a + b) = IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)
```

Those are the statement shapes Bug Zoo is meant to make routine.
