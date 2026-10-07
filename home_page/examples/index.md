---
title: Examples
---

Let's train a model, compute some gradients, or check a numerical claim in Lean.
Pick an example below for the code and commands.

## Featured Examples

The card images are illustrations. Each example page includes its own results and assumptions.

<div class="showcase-grid showcase-grid-featured">
  <a class="showcase-card showcase-image-card" href="{{ '/examples/custom-computations/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/custom-computations.svg' | relative_url }}" alt="A Lean square function applied to a tensor on CPU or GPU, mapping 1, 2, 3 to 1, 4, 9"/>
    <span class="showcase-body">
      <span class="showcase-title">Custom Tensor Computations</span>
      <span class="showcase-text">Let's square a tensor with an ordinary Lean function, then choose CPU or GPU when we run it. Try the CPU example directly in Lean's Infoview.</span>
      <span class="showcase-link">Open the example</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Semantics-and-Graphs/The-Canonical-Graph-IR/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/graph-ir-bounds-new.png' | relative_url }}" alt="TorchLean graph IR to interval bounds example"/>
    <span class="showcase-body">
      <span class="showcase-title">Graph IR and Bounds</span>
      <span class="showcase-text">Follow a small model as it becomes a graph with named operations, then use that graph for shape checks, execution traces, and interval bounds.</span>
      <span class="showcase-link">Open guide page</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/Differentiation-By-Example/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/autograd-basics-new.png' | relative_url }}" alt="Autograd basics example"/>
    <span class="showcase-body">
      <span class="showcase-title">Autograd Basics</span>
      <span class="showcase-text">Compute gradients for small tensor functions and inspect the recorded operations and local gradient calculations.</span>
      <span class="showcase-link">Open guide page</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Building-Models/Training___-One-State-Transition-At-A-Time/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/supervised-training-new.png' | relative_url }}" alt="Illustration of batches, a model, loss, optimizer, and training curves"/>
    <span class="showcase-body">
      <span class="showcase-title">Supervised Training</span>
      <span class="showcase-text">Build a model, load its training data, and train it in Lean. Save the loss curve to see how the run went.</span>
      <span class="showcase-link">Open training guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/diffusion/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/diffusion-new.png' | relative_url }}" alt="Illustration of iterative denoising from noise to an image, not a measured sample"/>
    <span class="showcase-body">
      <span class="showcase-title">Diffusion</span>
      <span class="showcase-text">Train a small denoiser, run deterministic DDIM sampling, and inspect both the generated images and the saved loss log.</span>
      <span class="showcase-link">Open diffusion walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/text-models/#gpt-2' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/gpt-text-new.png' | relative_url }}" alt="Illustration of token embeddings, transformer blocks, and next-token scores"/>
    <span class="showcase-body">
      <span class="showcase-title">GPT-Style Text</span>
      <span class="showcase-text">Tokenize bytes, build next-token examples, train a small causal transformer, save a checkpoint, and sample continuations.</span>
      <span class="showcase-link">Open text walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/scientific-ml/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/scientific-ml-new.png' | relative_url }}" alt="Illustration of a PDE, Fourier neural operator, and training loss; residual checks are separate"/>
    <span class="showcase-body">
      <span class="showcase-title">Scientific ML</span>
      <span class="showcase-text">Train a Fourier neural operator on Burgers data or a PINN from an equation, then explore separate checks for PDE residuals and datasets.</span>
      <span class="showcase-link">Open scientific ML pipeline</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/PyTorch-Round-Trip/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/pytorch-roundtrip-new.png' | relative_url }}" alt="PyTorch round-trip example"/>
    <span class="showcase-body">
      <span class="showcase-title">PyTorch Round Trip</span>
      <span class="showcase-text">Export PyTorch weights, load them into a TorchLean model, and check that their shapes match.</span>
      <span class="showcase-link">Open interop guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/float32-ieee-new.png' | relative_url }}" alt="Floating-point formats and IEEE arithmetic illustration"/>
    <span class="showcase-body">
      <span class="showcase-title">Floating-Point Formats and Proofs</span>
      <span class="showcase-text">Choose a floating-point format with FloatLib, compare rounded calculations, and see how we prove their numerical properties.</span>
      <span class="showcase-link">Open floating-point guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/numerical-runtime/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/graph-ir-bounds-new.png' | relative_url }}" alt="Numerical certificate and binary32 replay for a two-layer MLP"/>
    <span class="showcase-body">
      <span class="showcase-title">Numerical Runtime Certificates</span>
      <span class="showcase-text">Compute intervals for a small MLP, then replay its binary32 calculations and check each intermediate result.</span>
      <span class="showcase-link">Open the complete run</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Examples-and-Applications/Reinforcement-Learning/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/reinforcement-learning-new.png' | relative_url }}" alt="An agent selects actions, receives states and rewards, and gathers rollouts for policy updates"/>
    <span class="showcase-body">
      <span class="showcase-title">Reinforcement Learning</span>
      <span class="showcase-text">Run PPO on Lean-native and Gymnasium environments, then inspect the rollout, reward, and policy artifacts that enter training.</span>
      <span class="showcase-link">Open RL guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/bug-zoo/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/bug-zoo-new.png' | relative_url }}" alt="Bug Zoo case studies"/>
    <span class="showcase-body">
      <span class="showcase-title">Bug Zoo</span>
      <span class="showcase-text">See how common ML bugs become small Lean contracts: causal masks, KV caches, token ids, normalization state, batching, and Float32 behavior.</span>
      <span class="showcase-link">Open Bug Zoo walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/3d-vision/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/geometry3d-vision-new.png' | relative_url }}" alt="Illustration of camera projection and a cuboid; the example checks projected-point enclosure"/>
    <span class="showcase-body">
      <span class="showcase-title">3D Vision Certificates</span>
      <span class="showcase-text">Export camera and box tensors from a detector, recompute projection in Lean, and reject boxes that do not enclose projected corners.</span>
      <span class="showcase-link">Open 3D vision tutorial</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/verification/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/verification-bounds-new.png' | relative_url }}" alt="Interval bounds propagated through a network; the shaded region illustrates a nonnegative-output property."/>
    <span class="showcase-body">
      <span class="showcase-title">IBP and CROWN Verification</span>
      <span class="showcase-text">Bound a model's outputs over an input region using IBP or CROWN, and check certificates exported by other verification tools.</span>
      <span class="showcase-link">Open verification tutorial</span>
    </span>
  </a>
</div>
