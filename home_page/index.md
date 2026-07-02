---
# layout: home
usemathjax: true
---

<div class="hero home-hero">
  <div class="hero-text">
    <p class="lede">
      TorchLean is a Lean 4 library for executable neural networks with formal specifications,
      verified graph semantics, CUDA trust boundaries, and proof-checkable ML artifacts.
    </p>

    <div class="home-actions" aria-label="Primary links">
      <a class="primary-link" href="{{ '/blueprint/Introduction/' | relative_url }}">Start reading</a>
      <a class="secondary-link" href="{{ '/examples/' | relative_url }}">View examples</a>
      <a class="secondary-link" href="https://arxiv.org/abs/2602.22631">Read the paper</a>
    </div>
  </div>
</div>

## What TorchLean Is

<div class="home-pillars">
  <section>
    <h3>Specifications</h3>
    <p>
      Typed tensors, layer contracts, losses, attention, normalization, and model definitions live
      in Lean as mathematical objects.
    </p>
  </section>
  <section>
    <h3>Runtime</h3>
    <p>
      Executable autograd, graph IR lowering, PyTorch-style interop, and CUDA kernels are exposed
      through explicit trust boundaries.
    </p>
  </section>
  <section>
    <h3>Verification</h3>
    <p>
      Proof-backed compiler pieces, CROWN and IBP certificates, floating-point models, and bug-zoo
      contracts make ML behavior inspectable.
    </p>
  </section>
</div>

<div class="home-overview">
  <img
    src="{{ '/assets/media/figures/torchlean-layout.png' | relative_url }}"
    alt="TorchLean overview: typed tensors, shared graph IR, verified reverse mode autograd, IEEE-754 semantics, certificate checking, PyTorch round trip, CUDA trust boundary, approximation theorems, and Lean verification."
    loading="lazy" />
</div>

## Workflows

<div class="workflow-list">
  <a href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/Autograd-Walkthrough/' | relative_url }}">
    <span>01</span>
    <strong>Write and run models</strong>
    <em>Use Lean-native training loops, tensors, and autograd examples.</em>
  </a>
  <a href="{{ '/blueprint/Semantics-and-Graphs/Graphs-and-IR/' | relative_url }}">
    <span>02</span>
    <strong>Lower to graph IR</strong>
    <em>Inspect shapes, payloads, graph semantics, and executable traces.</em>
  </a>
  <a href="{{ '/blueprint/Floating-Point-and-Native-Boundaries/GPU-and-CUDA-Boundaries/' | relative_url }}">
    <span>03</span>
    <strong>Name the trusted boundary</strong>
    <em>Separate proved specs from floating-point, PyTorch, and CUDA runtime assumptions.</em>
  </a>
  <a href="{{ '/examples/bug-zoo/' | relative_url }}">
    <span>04</span>
    <strong>Turn bugs into contracts</strong>
    <em>Study small checked examples for masks, losses, normalization, caches, and runtime edges.</em>
  </a>
</div>
