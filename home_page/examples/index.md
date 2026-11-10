---
title: Examples
---

Let's train a model, compute some gradients, or check a numerical claim in Lean.
Pick an example below for the code and commands.

- [Custom Tensor Computations]({{ '/examples/custom-computations/' | relative_url }}) Let's square
  a tensor with an ordinary Lean function, then choose CPU or GPU when we run it. Try the CPU
  example in Lean's Infoview.

- [Graph IR and Bounds]({{ '/blueprint/Semantics-and-Graphs/The-Canonical-Graph-IR/' | relative_url }})
  Follow a small model as it becomes a graph with named operations, then use that graph for shape checks, execution traces, and interval bounds.

- [Autograd Basics]({{ '/blueprint/Runtime___-Autograd___-and-Interop/Differentiation-By-Example/' | relative_url }})
  Compute gradients for small tensor functions and inspect the recorded operations and local gradient calculations.

- [Supervised Training]({{ '/blueprint/Building-Models/Training___-One-State-Transition-At-A-Time/' | relative_url }})
  Build a model, load its training data, and train it in Lean. Save the loss curve to see how the run went.

- [Diffusion]({{ '/examples/diffusion/' | relative_url }})
  Train a small denoiser, run deterministic DDIM sampling, and inspect both the generated images and the saved loss log.

- [GPT-Style Text]({{ '/examples/text-models/#gpt-2' | relative_url }})
  Tokenize bytes, build next-token examples, train a small causal transformer, save a checkpoint, and sample continuations.

- [Scientific ML]({{ '/examples/scientific-ml/' | relative_url }})
  Train a Fourier neural operator on Burgers data or a PINN from an equation, then explore separate checks for PDE residuals and datasets.

- [PyTorch Round Trip]({{ '/blueprint/Runtime___-Autograd___-and-Interop/PyTorch-Round-Trip/' | relative_url }})
  Export PyTorch weights, load them into a TorchLean model, and check that their shapes match.

- [Checking a Captured Computation]({{ '/examples/verify/' | relative_url }})
  Use `torchlean.verify` with a PyTorch or JAX function. Check the graph's types and shapes, then compare its mixed-precision outputs on an example.

- [JAX Computations and Lean Proofs]({{ '/examples/jaxpr/' | relative_url }})
  Bring a JAX function into the same graph IR, compare its outputs, and prove that a sum of squares is unchanged by permuting its inputs.

- [Floating-Point Formats and Proofs]({{ '/blueprint/Floating-Point-and-Native-Boundaries/Floating-Point-Semantics/' | relative_url }})
  Choose a floating-point format with FloatLib, compare rounded calculations, and see how we prove their numerical properties.

- [Numerical Runtime Certificates]({{ '/examples/numerical-runtime/' | relative_url }})
  Compute intervals for a small MLP, then replay its binary32 calculations and check each intermediate result.

- [Reinforcement Learning]({{ '/blueprint/Examples-and-Applications/Reinforcement-Learning/' | relative_url }})
  Run PPO on Lean-native and Gymnasium environments, then inspect the rollout, reward, and policy artifacts that enter training.

- [Bug Zoo]({{ '/examples/bug-zoo/' | relative_url }})
  See how common ML bugs become small Lean contracts: causal masks, KV caches, token ids,
  normalization state, batching, Float32 behavior, and formulas that fail from rounding.

- [3D Vision Certificates]({{ '/examples/3d-vision/' | relative_url }})
  Export camera and box tensors from a detector, recompute projection in Lean, and reject boxes that do not enclose projected corners.

- [IBP and CROWN Verification]({{ '/examples/verification/' | relative_url }})
  Bound a model's outputs over an input region using IBP or CROWN, and check certificates exported by other verification tools.
