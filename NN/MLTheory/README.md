# NN/MLTheory

This directory contains TorchLean's formal ML theory layer: specifications, executable checkers, and
theorem level bridges.

Recommended entrypoint:

```lean
import NN.MLTheory.API
```

`API.lean` is the only top level Lean entrypoint in this folder. The subdirectories are
implementation modules grouped by topic; they are public Lean modules, but ordinary users should not
need to import separate umbrellas.

## Folder Map

- `CROWN/`: interval, affine, graph, and certificate oriented bound propagation. This includes the
  reusable soundness layer plus explicit trust boundaries for oracle backed Lyapunov workflows.
- `Generative/`: mathematical semantics for diffusion samplers and latent variable or generative
  objectives.
- `LearningTheory/`: robustness, stability, differential privacy, and a small ridge regression
  bridge that connects real valued theory to executable IEEE32 semantics.
- `Optimization/`: optimizer equations over TorchLean tensors, exact real convergence theorems for
  gradient descent, and the calculus bridge from strong convexity to strong monotonicity.
- `Proofs/`: larger theorem developments for approximation, ReLU constructions, state space
  models, and verification oriented robustness.
- `SelfSupervised/`: finite predictive view SSL algebra, MAE/JEPA instance theorems,
  VICReg/Barlow style geometry guards, masking semantics, real valued view graph alignment energy,
  and local anti-collapse facts.

## Boundaries

- The runtime facing optimizer facts are equations about `Spec.Tensor` programs. The convergence
  theorems are exact `ℝ` results. Applying them to CUDA/Float training requires a separate
  model specific bridge plus floating point error accounting.
- CROWN/LiRPA certificate checkers verify exported artifacts against the formal checker contracts.
  They do not prove that every external producer is sound unless the producer emits a proof object
  or the workflow goes through an explicit oracle/trust boundary.
- Optional operators such as GELU and some trigonometric affine rules are marked experimental when
  the file provides engineering relaxations rather than proved enclosure theorems.
- The `IEEE32Exec` ridge regression bridge is intentionally local: it relates executable binary32
  arithmetic to a semantics that rounds after each primitive, under explicit finiteness
  assumptions. It is not a blanket theorem that all Float/CUDA execution equals real arithmetic.

## Citations and Pointers

- CROWN: Zhang et al., "Efficient Neural Network Robustness Certification with General Activation
  Functions," NeurIPS 2018.
- DeepPoly: Singh et al., "An Abstract Domain for Certifying Neural Networks," POPL 2019.
- auto_LiRPA: Xu et al., "Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond," NeurIPS 2020.
- PINNs: Raissi, Perdikaris, and Karniadakis, "Physics-informed neural networks," JCP 2019.
- MAE: He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.
- JEPA/I-JEPA: Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture," CVPR 2023.
- VICReg: Bardes, Ponce, and LeCun, "VICReg: Variance-Invariance-Covariance Regularization for
  Self-Supervised Learning," ICLR 2022.
- Barlow Twins: Zbontar et al., "Barlow Twins: Self-Supervised Learning via Redundancy Reduction,"
  ICML 2021.
- Alignment/uniformity: Wang and Isola, "Understanding Contrastive Representation Learning through
  Alignment and Uniformity on the Hypersphere," ICML 2020.
- Spectral SSL view: Balestriero and LeCun, "Contrastive and Non-Contrastive Self-Supervised
  Learning Recover Global and Local Spectral Embedding Methods," NeurIPS 2022.
- IEEE floating point: IEEE Std 754-2019; Goldberg (1991), "What Every Computer Scientist Should
  Know About Floating-Point Arithmetic"; Muller et al., *Handbook of Floating-Point Arithmetic*.
