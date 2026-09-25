---
# layout: home
description: "Tensor computation, floating-point verification, and machine learning in Lean 4."
---

<section class="home-intro">
  <div class="home-intro-copy">
    <p>
      TorchLean brings tensor computation, machine learning, and formal verification together in
      Lean 4. The cool part is that Lean is both a functional programming language and a theorem
      prover, so you can write computations, build and train models, and prove mathematical
      properties in the same language. You can start with tensors and linear algebra, use the
      library for general numerical programming, or work directly with the specifications and proofs.
    </p>
  </div>

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

</section>

<section class="home-highlights" markdown="1">

A few highlights we're excited about:

- **Tensors that carry their shapes.** Lean checks that the dimensions fit when you compose tensor operations. You can use the [array and linear algebra library]({{ '/blueprint/Building-Models/Tensors-That-Remember-Their-Shapes/' | relative_url }}) on its own, without building a neural network.
- **Build and train models in Lean.** Start with [regression or other classical models]({{ '/blueprint/Building-Models/The-TorchLean-API/#TorchLean--Building-Models--TorchLean-API--Classical-Models' | relative_url }}), train a [transformer or generative model]({{ '/blueprint/Examples-and-Applications/Modern-Models/' | relative_url }}), or develop a [reinforcement learning agent]({{ '/blueprint/Examples-and-Applications/Reinforcement-Learning/' | relative_url }}). TorchLean owns automatic differentiation and optimizers, with CPU execution and a GPU backend built on LibTorch's tensor operations.
- **A shared graph for computation and proofs.** A typed computation graph records the operations in a model and the shapes of their inputs and outputs. We use it to describe the calculation, execute it, and state mathematical properties about it. The [formalization map]({{ '/blueprint/Formalization-Map/' | relative_url }}) connects the definitions to their proofs.
- **Floating-point behavior is part of the mathematics.** Through [FloatLib](https://lean-dojo.github.io/FloatLib/), TorchLean supports configurable binary formats, executable arithmetic, and proofs about rounding and numerical error. The [floating-point guide]({{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }}) explains the arithmetic models and how they relate to native execution.
- **Ask questions about a whole range of inputs.** For a classifier, we can establish conditions under which its prediction stays the same throughout an input region. The [verification tools]({{ '/blueprint/Verification-and-Certificates/Neural-Network-Verification/' | relative_url }}) include interval bounds and certificate checking, with Lean proofs for the mathematical guarantees.

The [guide]({{ '/blueprint/' | relative_url }}) walks through the library step by step. If you'd rather start by running something, try the [examples]({{ '/examples/' | relative_url }}).

</section>
