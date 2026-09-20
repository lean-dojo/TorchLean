---
title: Examples
---

These examples show TorchLean at work on real machine-learning problems: training models,
differentiating tensor programs, moving weights through PyTorch, and checking numerical or
verification claims in Lean. Open an example for the runnable code and the theorem or contract
behind it.

## Featured Examples

<div class="showcase-grid showcase-grid-featured">
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
      <span class="showcase-text">Compute gradients for small tensor functions, then inspect the tape and VJP objects that make reverse mode explicit.</span>
      <span class="showcase-link">Open guide page</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Building-Models/Training___-One-State-Transition-At-A-Time/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/supervised-training-new.png' | relative_url }}" alt="MLP and CNN training example"/>
    <span class="showcase-body">
      <span class="showcase-title">Supervised Training</span>
      <span class="showcase-text">Instantiate supervised models, build loaders, fit for multiple epochs or fixed steps, and save loss curves from the same Lean runner.</span>
      <span class="showcase-link">Open training guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/diffusion/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/diffusion-new.png' | relative_url }}" alt="Diffusion on real images example"/>
    <span class="showcase-body">
      <span class="showcase-title">Diffusion</span>
      <span class="showcase-text">Train a small denoiser, run deterministic DDIM sampling, and inspect both the generated images and the saved loss log.</span>
      <span class="showcase-link">Open diffusion walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/text-models/#gpt-2' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/gpt-text-new.png' | relative_url }}" alt="GPT-2 style text example"/>
    <span class="showcase-body">
      <span class="showcase-title">GPT-Style Text</span>
      <span class="showcase-text">Tokenize bytes, build next-token examples, train a small causal transformer, save a checkpoint, and sample continuations.</span>
      <span class="showcase-link">Open text walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/scientific-ml/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/scientific-ml-new.png' | relative_url }}" alt="Scientific ML pipeline from Burgers data to FNO training and Lean checks"/>
    <span class="showcase-body">
      <span class="showcase-title">Scientific ML</span>
      <span class="showcase-text">Prepare the Burgers dataset, train a 1D Fourier neural operator, export prediction artifacts, and connect PDE residual checks to Lean.</span>
      <span class="showcase-link">Open scientific ML pipeline</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/PyTorch-Round-Trip/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/pytorch-roundtrip-new.png' | relative_url }}" alt="PyTorch round-trip example"/>
    <span class="showcase-body">
      <span class="showcase-title">PyTorch Round Trip</span>
      <span class="showcase-text">Move weights across the Python boundary while keeping tensor shapes, parameter packs, and import checks visible.</span>
      <span class="showcase-link">Open interop guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/floatlib-binary-formats.svg' | relative_url }}" alt="FloatLib configured binary32 and binary128 formats: sign, exponent, fraction, and normal significand precision."/>
    <span class="showcase-body">
      <span class="showcase-title">Floating-Point Formats and Proofs</span>
      <span class="showcase-text">Compare FloatLib's executable binary arithmetic with rounded <code>FP32</code> models and TorchLean's proofs of agreement with Lean's logical <code>Float32</code> operations.</span>
      <span class="showcase-link">Open floating-point guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/numerical-runtime/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/graph-ir-bounds-new.png' | relative_url }}" alt="Numerical certificate and binary32 replay for a two-layer MLP"/>
    <span class="showcase-body">
      <span class="showcase-title">Numerical Runtime Certificates</span>
      <span class="showcase-text">Run a two-layer MLP through operation coverage, interval propagation, backend-capsule audit, and bit-level binary32 replay.</span>
      <span class="showcase-link">Open the complete run</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Examples-and-Applications/Reinforcement-Learning/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/reinforcement-learning-new.png' | relative_url }}" alt="Reinforcement learning examples"/>
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
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/geometry3d-vision-new.png' | relative_url }}" alt="3D Vision Certificates example"/>
    <span class="showcase-body">
      <span class="showcase-title">3D Vision Certificates</span>
      <span class="showcase-text">Export camera and box tensors from a detector, recompute projection in Lean, and reject boxes that do not enclose projected corners.</span>
      <span class="showcase-link">Open 3D vision tutorial</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/verification/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/verification-bounds-new.png' | relative_url }}" alt="IBP and alpha-CROWN verification example"/>
    <span class="showcase-body">
      <span class="showcase-title">IBP and CROWN Verification</span>
      <span class="showcase-text">Attach input boxes to an IR graph, propagate interval or affine bounds, and check small external certificates through Lean. PINN examples use the same artifact-first style.</span>
      <span class="showcase-link">Open verification tutorial</span>
    </span>
  </a>
</div>
