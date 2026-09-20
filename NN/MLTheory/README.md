# `NN/MLTheory`

This directory is TorchLean's ML-theory layer. It is where executable ML objects start getting
turned into mathematical claims: bounds for neural-network verification, optimizer laws, generative
objective identities, self-supervised-learning algebra, approximation theorems, stability facts, and
floating-point bridges.

The recommended import is:

```lean
import NN.MLTheory.API
```

Most users should not need to import individual theory files directly. The subdirectories are still
public Lean modules, but they are organized by proof topic rather than by beginner-facing API.

## Definitions, Checkers, And Theorems

TorchLean separates three kinds of evidence.

1. **Definitions.** A file may define the mathematical object: a CROWN affine form, a Muon update
   equation, a diffusion loss, a robustness margin, or a stability predicate.
2. **Executable checkers.** A file may recompute a bounded condition from finite data: an interval
   bound, a JSON certificate, a residual inequality, or an optimizer update trace.
3. **Theorems.** A file may prove that a checker or update rule implies the stated mathematical
   property under explicit hypotheses.

Those roles are deliberately kept separate. A fast checker gives evidence about the artifact it
recomputed; a theorem gives a mathematical claim under stated hypotheses. CUDA runs and serialized
decimal artifacts enter through named bridges or trust boundaries, so the reader can see exactly
which object a theorem is about.

## Folder Map

| Folder | What it is for |
| --- | --- |
| `CROWN/` | Interval, affine, CROWN/LiRPA-style bound propagation, graph certificates, alpha/beta certificate structures, and Lyapunov certificate workflows. |
| `Generative/` | Mathematical semantics for diffusion, VAE/VQ-VAE/GAN-style objectives, and the small identities used by the generative examples. |
| `LearningTheory/` | Robustness, stability, differential privacy, and ridge-regression bridges between real-valued theory and executable IEEE32-style semantics. |
| `Optimization/` | Optimizer equations, proof layer optimizer interfaces, Muon certificates, projected-gradient material, and exact real convergence theorems. |
| `Proofs/` | Larger theorem developments: approximation, ReLU constructions, state-space and Mamba-style scan facts, and verification-oriented robustness. |
| `SelfSupervised/` | MAE/JEPA/VICReg/Barlow-style finite algebra, view alignment, masking semantics, and anti-collapse conditions. |

## CROWN, LiRPA, And Certificates

The CROWN material is the verification backbone. It contains the objects that a verifier actually
manipulates: interval boxes, affine lower and upper forms, graph payloads, ReLU relaxations,
alpha/beta phase information, and soundness statements for supported graph fragments.

The central distinction is producer versus checker:

- An external tool may propose slopes, beta splits, or JSON bounds.
- TorchLean parses those artifacts through `NN.Verification.*`.
- The theory layer states what accepted local bounds mean over the graph semantics.

That means a certificate checker can carry value before every producer is verified. The trusted
boundary is explicit: Lean checked the artifact it received; the external search procedure is not
silently promoted into a theorem.

## Optimizers

The optimizer theory files package optimizers as pure tensor update rules and prove that the
executable update follows the named spec.

The general pattern is:

1. define a pure update equation over tensor parameters and optimizer state,
2. package it as a shape-polymorphic `TensorOptimizer`,
3. state a one-step `StepSpec`,
4. prove reusable stream facts such as `runSteps_append` and
   `runSteps_eq_optimizer_runSteps`,
5. add optimizer-specific laws only where the rule has real structure.

Current coverage includes ordinary first-order optimizers and newer optimizer-adjacent ideas:

- **SGD, momentum SGD, Adagrad, RMSProp, Adam, AdamW, Adadelta.** These are ordinary optimizer
  update rules. Their public runtime names live under `TorchLean.optim`, and the proof layer shape
  is the generic `TensorOptimizer`/`StepSpec` interface.
- **Muon.** Muon is treated as an optimizer with an explicit orthogonalization backend. The proof
  separates the momentum buffer recurrence, the backend output used as the update direction, and
  the parameter update equation. A backend can provide an exact certificate such as
  $Q^\mathsf{T}Q=I$, or an approximate certificate bounding $Q^\mathsf{T}Q-I$ entrywise.
- **GaLore.** GaLore is treated as projected-gradient machinery. The proof object names the
  projection and also names the optimizer applied after projection, so the update equation remains
  explicit.
- **LoRA.** LoRA is adapter/parameterization structure. Its claims belong with adapter weights,
  parameterization, and model structure, rather than with optimizer state.

For Muon, the main handles are grouped by role:

- `Optimization/OptimizerLaws.lean` gives the generic optimizer interface and reusable
  step/stream laws.
- `Optimization/Muon.lean` and `Optimization/Muon/` give the orthogonalizer contracts, Muon update equations, QR/Gram-Schmidt exact
  certificates, and Newton-Schulz residual-checked certificates.
- `NN/Examples/Optimization/MuonCertificates.lean` shows how a downstream proof consumes the
  packaged exact or approximate certificate.

The Newton-Schulz path is intentionally certificate-shaped. The polynomial iteration is executable,
but the theorem-level claim is made through a residual condition on the produced direction. That
keeps CUDA or other fast backends honest: they should target the same checked exact/approximate
backend record rather than changing Muon's semantics.

The convergence theorems in this directory are exact `ℝ` statements. For strongly convex
gradient descent, `Optimization/StronglyConvexGD.lean` states the contraction with the explicit
`q^k` factor (`dist_sq_iterate_le_of_q_lt_one`) and derives the step-size form
(`dist_sq_iterate_le_of_step_size`). Applying them to executable
Float32 operations, CUDA kernels, or a particular trained model requires a separate refinement
argument with explicit floating-point error accounting.

## Generative And Self-Supervised Theory

The generative files state the algebra that later examples cite:

- diffusion sampler and objective identities,
- VAE and VQ-VAE loss decompositions,
- nearest-code facts for finite codebooks,
- GAN/LSGAN scalar objective packaging and zero-loss cases.

The self-supervised files follow the same style. MAE, JEPA, VICReg, Barlow-style terms, masking, and
view-alignment predicates are stated as finite mathematical objects. The Lean claim names
the objective, mask, view graph, or anti-collapse condition that the example uses. Masked losses
count array occurrences, including duplicate indices. A positive collapse penalty does not alone
exclude a collapsed minimizer or prove representation quality. JEPA target extensionality is a
value-level identity, not a stop-gradient theorem.

## Learning Theory And Floating-Point Bridges

`LearningTheory/` holds robustness, stability, privacy, and ridge-regression examples. Some
statements live over exact real numbers. Others use executable IEEE32-style semantics to make the
finite path explicit.

The real ridge theorem assumes positive regularization and bounded training, replacement, and
test examples. Its explicit bound is `4 * X^2 * Y^2 * (λ + X^2)^2 / (λ^3 * N)`, so it decays
as `1/N` with the other parameters fixed. The separate binary32 bridge relates expression
evaluation to semantics that round after each primitive, under a `FiniteEval` hypothesis. It does
not transfer the real stability bound to floating-point training.

Expected-loss stability predicates use totalized Bochner integrals; finite-expectation readings
require integrability assumptions. The dynamics ISS predicate states the trajectory inequality
without class-K/class-KL conditions, and the real stability margin is a supremum rather than a
proved largest invariant radius. The privacy layer proves budget monotonicity and measurable
post-processing, not a privacy guarantee for a concrete training procedure.

## Adding Theory

Add a file to `NN/MLTheory` when the main object is a mathematical statement or proof interface.
Add it somewhere else when the main object is:

- a runnable training script: `NN/Examples`,
- executable autograd or backend code: `NN/Runtime`,
- a parsed artifact checker or CLI: `NN/Verification`,
- a tensor/layer/model denotation: `NN/Spec`,
- a public user-facing constructor: `NN/API`.

When adding a new theorem family, include a short module docstring that says:

- what object is being defined,
- what is proved,
- which runtime or external producer assumptions remain,
- where the runnable example or checker lives, if there is one.

## Citations And Pointers

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
- IEEE floating point: IEEE Std 754-2019; Goldberg (1991), "What Every Computer Scientist Should
  Know About Floating-Point Arithmetic"; Muller et al., *Handbook of Floating-Point Arithmetic*.
