---
title: Examples
usemathjax: true
---

The examples are the quickest way to see TorchLean as a working system. Each page starts from
something concrete: a model run, a graph, a tensor artifact, a saved log, a verifier result, or a
bug pattern. The goal is to show how runnable ML code becomes an object that Lean can inspect,
lower, check, or relate to a theorem.

Start with a small training run or the verification bounds page. For a systems-oriented view, read
the text-model and diffusion walkthroughs. For the clearest motivation, read Bug Zoo.

## Featured Examples

<div class="showcase-grid showcase-grid-featured">
  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Semantics-and-Graphs/Graphs-and-IR/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/ir-bounds.png' | relative_url }}" alt="TorchLean graph IR to interval bounds example"/>
    <span class="showcase-body">
      <span class="showcase-title">Graph IR and Bounds</span>
      <span class="showcase-text">Follow a small model as it becomes an op-tagged graph, then use that graph for shape checks, execution traces, and interval bounds.</span>
      <span class="showcase-link">Open guide page</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/Autograd-Walkthrough/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/autograd-basics.png' | relative_url }}" alt="Autograd basics example"/>
    <span class="showcase-body">
      <span class="showcase-title">Autograd Basics</span>
      <span class="showcase-text">Compute gradients for small tensor functions, then inspect the tape and VJP objects that make reverse mode explicit.</span>
      <span class="showcase-link">Open guide page</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/docs/NN/Examples/Models.html' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/mlp-cnn-training.png' | relative_url }}" alt="MLP and CNN training example"/>
    <span class="showcase-body">
      <span class="showcase-title">Supervised Training</span>
      <span class="showcase-text">Instantiate supervised models, build loaders, fit for multiple epochs or fixed steps, and save loss curves from the same Lean runner.</span>
      <span class="showcase-link">Open model examples docs</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/diffusion/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/diffusion-real-images.png' | relative_url }}" alt="Diffusion on real images example"/>
    <span class="showcase-body">
      <span class="showcase-title">Diffusion</span>
      <span class="showcase-text">Train a small denoiser, run deterministic DDIM sampling, and inspect both the generated images and the saved loss log.</span>
      <span class="showcase-link">Open diffusion walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/text-models/#gpt-2' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/gpt2-text.png' | relative_url }}" alt="GPT-2 style text example"/>
    <span class="showcase-body">
      <span class="showcase-title">GPT-Style Text</span>
      <span class="showcase-text">Tokenize bytes, build next-token examples, train a small causal transformer, save parameters, and sample continuations.</span>
      <span class="showcase-link">Open text walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/text-models/#mamba' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/mamba-text.png' | relative_url }}" alt="Mamba text model example"/>
    <span class="showcase-body">
      <span class="showcase-title">Mamba Text Model</span>
      <span class="showcase-text">Use the same corpus and logging path with a compact state-space sequence model instead of attention.</span>
      <span class="showcase-link">Open text walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/PyTorch-Round-Trip/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/pytorch-roundtrip.png' | relative_url }}" alt="PyTorch round-trip example"/>
    <span class="showcase-body">
      <span class="showcase-title">PyTorch Round Trip</span>
      <span class="showcase-text">Move weights across the Python boundary while keeping tensor shapes, parameter packs, and import checks visible.</span>
      <span class="showcase-link">Open interop guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/float32-ieee.png' | relative_url }}" alt="Float32 and IEEE-754 checks example"/>
    <span class="showcase-body">
      <span class="showcase-title">Float32 and IEEE-754</span>
      <span class="showcase-text">Compare real specifications, rounded `FP32` models, executable IEEE bit semantics, and runtime `Float32` bridge assumptions.</span>
      <span class="showcase-link">Open floating-point guide</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/docs/NN/Examples/Models/RL.html' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/reinforcement-learning.png' | relative_url }}" alt="Reinforcement learning examples"/>
    <span class="showcase-body">
      <span class="showcase-title">Reinforcement Learning</span>
      <span class="showcase-text">Run PPO on Lean-native and Gymnasium environments, then inspect the rollout, reward, and policy artifacts that enter training.</span>
      <span class="showcase-link">Open RL example docs</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/bug-zoo/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/bug-zoo.png' | relative_url }}" alt="Bug Zoo case studies"/>
    <span class="showcase-body">
      <span class="showcase-title">Bug Zoo</span>
      <span class="showcase-text">See how common ML bugs become small Lean contracts: causal masks, KV caches, token ids, normalization state, batching, and Float32 behavior.</span>
      <span class="showcase-link">Open Bug Zoo walkthrough</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/3d-vision/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/geometry3d-vision-certificates.png' | relative_url }}" alt="3D Vision Certificates example"/>
    <span class="showcase-body">
      <span class="showcase-title">3D Vision Certificates</span>
      <span class="showcase-text">Export camera and box tensors from a detector, recompute projection in Lean, and reject boxes that do not enclose projected corners.</span>
      <span class="showcase-link">Open 3D vision tutorial</span>
    </span>
  </a>

  <a class="showcase-card showcase-image-card" href="{{ '/examples/verification/' | relative_url }}">
    <img class="showcase-media" src="{{ '/assets/media/examples/showcase/verification-bounds.png' | relative_url }}" alt="IBP and alpha-CROWN verification example"/>
    <span class="showcase-body">
      <span class="showcase-title">IBP and CROWN Verification</span>
      <span class="showcase-text">Attach input boxes to an IR graph, propagate interval or affine bounds, and check small external certificates through Lean.</span>
      <span class="showcase-link">Open verification tutorial</span>
    </span>
  </a>
</div>

CUDA is opt-in. The build flags, runtime path, and agreement assumptions are explained in
[GPU and CUDA]({{ '/blueprint/Floating-Point-and-Native-Boundaries/GPU-and-CUDA-Boundaries/' | relative_url }}).
