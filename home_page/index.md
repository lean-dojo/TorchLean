---
# layout: home
description: "Tensor computation, floating-point verification, and machine learning in Lean 4."
---

<section class="home-intro">
  <figure class="home-overview">
    <a
      href="{{ '/assets/media/figures/torchlean-layout.png' | relative_url }}"
      aria-label="Open the full TorchLean system diagram">
      <img
        src="{{ '/assets/media/figures/torchlean-layout.png' | relative_url }}"
        alt="TorchLean overview: typed tensors, shared graph IR, autograd proofs, IEEE-754 semantics, certificate checking, PyTorch interoperability, CUDA providers, and model analysis."
        loading="eager" />
    </a>
    <figcaption>From a typed model to execution, analysis, and proof.</figcaption>
  </figure>

  <div class="home-intro-copy">
    <p>
      TorchLean brings tensor computation, machine learning, and formal verification together in
      Lean 4. You can use it for general numerical programming, build and train models, or study
      the mathematical properties of an algorithm.
    </p>

    <p>
      The tensor library provides general array operations and linear algebra, with shapes checked
      in the types. For floating-point work, TorchLean integrates FloatLib's configurable binary
      formats, executable arithmetic, and proofs about rounding, numerical error, and interval
      bounds. The tensor and scalar APIs can be used independently of the machine learning components.
    </p>

    <p>
      The machine learning tools cover classical models, deep learning, reinforcement learning,
      and neural network verification. You can fit a regression model, train a transformer, build
      a generative model, or develop an agent that learns from interaction, using automatic
      differentiation, optimizers, and CPU or GPU execution.
    </p>

    <p>
      You can also write mathematical specifications and prove theorems about the operations and
      algorithms you use. For a classifier, that can mean establishing conditions under which its
      prediction stays the same for every input in a region; for a numerical computation, it can
      mean proving a bound on its rounding error.
    </p>
  </div>
</section>

## Explore TorchLean

<div class="workflow-list">
  <a href="{{ '/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/' | relative_url }}">
    <span>01</span>
    <strong>Tensors and linear algebra</strong>
    <em>Compute with arrays and matrices, and prove identities about their operations.</em>
  </a>
  <a href="{{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }}">
    <span>02</span>
    <strong>Floating-point verification</strong>
    <em>Study rounding, prove numerical error bounds, and work with interval arithmetic.</em>
  </a>
  <a href="{{ '/blueprint/Building-Models/The-TorchLean-API/#TorchLean--Building-Models--TorchLean-API--Classical-Models' | relative_url }}">
    <span>03</span>
    <strong>Classical machine learning</strong>
    <em>Work with regression, nearest neighbors, forests, and probabilistic models.</em>
  </a>
  <a href="{{ '/blueprint/Examples-and-Applications/Modern-Models/' | relative_url }}">
    <span>04</span>
    <strong>Deep learning</strong>
    <em>Build and train neural networks, from multilayer perceptrons to transformers.</em>
  </a>
  <a href="{{ '/blueprint/Examples-and-Applications/Reinforcement-Learning/' | relative_url }}">
    <span>05</span>
    <strong>Reinforcement learning</strong>
    <em>Define environments, collect experience, and train policies with PPO.</em>
  </a>
  <a href="{{ '/blueprint/Verification-and-Certificates/Neural-Network-Verification/' | relative_url }}">
    <span>06</span>
    <strong>Neural network verification</strong>
    <em>Check whether input uncertainty can change a classifier's prediction.</em>
  </a>
  <a href="{{ '/blueprint/Formalization-Map/' | relative_url }}">
    <span>07</span>
    <strong>Specifications and proofs</strong>
    <em>State mathematical properties and follow the Lean theorems that establish them.</em>
  </a>
</div>
