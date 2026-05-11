/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Core.RealCorrectness
public import NN.Proofs.Autograd.Coverage
public import NN.Proofs.Autograd.Runtime.Link
public import NN.Proofs.Autograd.Tape.Core.Soundness

/-!
# Autograd proofs: overview and map to PyTorch

TorchLean’s autograd development is split into **spec**, **runtime**, and **proofs** layers.
This folder (`NN/Proofs/Autograd/*`) is the main correctness/soundness argument for the runtime
autograd engine, written in a way that stays close to the structure of PyTorch Autograd.

## Big picture

1. **Spec layer (`NN/Spec/Autograd/*`)**
   - Defines small `OpSpec`s: each op is a pure `forward` function plus an explicit VJP
     (`backward`).
   - These are the “local backward rules” analogous to implementing a small
     `torch.autograd.Function` with a hand-written `backward`.

2. **Correctness layer (`NN/Proofs/Autograd/Core/RealCorrectness.lean` and
   `NN/Proofs/Autograd/Core/SemiringCorrectness.lean`)**
   - Proves the core adjointness law (VJP/JVP duality) for those `OpSpec`s:
     `⟪JVP(x, dx), δ⟫ = ⟪dx, VJP(x, δ)⟫`.
   - This is enough to justify the reverse-mode chain rule over a graph/tape.

3. **Tape/graph soundness (`NN/Proofs/Autograd/Tape/*`)**
   - Models a dynamic SSA/DAG tape: nodes can reference any previously produced values, matching
     the “tape of nodes” view used by most reverse-mode engines.
   - Proves that the global reverse-mode accumulation algorithm is sound assuming local
     adjointness at each node.

4. **Runtime link (`NN/Proofs/Autograd/Runtime/Link.lean`)**
   - Connects the executable runtime tape in `NN/Runtime/Autograd/*` to the proved tape model by
     compiling proved nodes into runtime nodes with baked-in backward closures.

## How this compares to PyTorch Autograd

- **Dynamic graphs**: both systems support DAG structure with sharing/fan-out.
- **VJP-first**: PyTorch’s backward is VJP-based; TorchLean proofs are organized around the same
  VJP/JVP
  adjointness statement.
- **Pure semantics**: TorchLean uses pure functions and typed shapes in the spec/proof layers;
  PyTorch
  uses an imperative engine with runtime shapes and a mutable `ctx` for custom Functions.
- **Trust boundary**: TorchLean can swap “runtime semantics” (exact, rounded models, etc.) depending
  on
  the backend, whereas PyTorch executes on IEEE-754 hardware by default.

## References / citations
- PyTorch Autograd docs: https://pytorch.org/docs/stable/autograd.html
- `torch.autograd.Function`: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
- AD survey: Baydin et al. (JMLR 2018), https://arxiv.org/abs/1502.05767
- Foundations: Griewank & Walther, *Evaluating Derivatives* (SIAM, 2008).
-/

@[expose] public section
