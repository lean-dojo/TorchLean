/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Generative.Diffusion.Core
public import NN.Spec.Generative.Diffusion.ForwardProcess
public import NN.Spec.Generative.Diffusion.ImageDDIM
public import NN.Spec.Generative.Diffusion.Loss
public import NN.Spec.Generative.Diffusion.PFODE
public import NN.Spec.Generative.Diffusion.ReverseDDIM
public import NN.Spec.Generative.Diffusion.ReverseDDPM
public import NN.Spec.Generative.Diffusion.Schedule

/-!
# Diffusion / flow specs (umbrella)

This is the curated public entrypoint for TorchLean's diffusion / flow spec layer.

It re-exports:

- a discrete VP schedule + forward noising (`qSample`),
- reverse samplers (DDPM, deterministic DDIM, and the clipped image sampler), and
- a continuous-time VP schedule + probability-flow ODE drift (`pfOdeRhs`).

All specs are scalar-polymorphic (`Context α`) so the same definitions can be reused for:

- runtime execution (`Float`, `Float32`, or the configured binary32 type `ExecFloat.Binary 8 23`),
- CPU software execution at a chosen precision (`FloatLib.Floats.ExecFloat.Binary`),
- proofs (`ℝ` or the noncomputable rounded-real model `FloatLib.Floats.Formats.Flocq.NF`), and
- verification backends (interval scalars, etc.).
-/

@[expose] public section
