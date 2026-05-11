/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.BugZoo.ShapeAndBroadcast
public import NN.Examples.BugZoo.StableLoss
public import NN.Examples.BugZoo.IgnoredLabelLoss
public import NN.Examples.BugZoo.AutogradDomain
public import NN.Examples.BugZoo.AttentionMask
public import NN.Examples.BugZoo.CompilerBoundary
public import NN.Examples.BugZoo.FloatBoundary
public import NN.Examples.BugZoo.NormalizationState
public import NN.Examples.BugZoo.BatchInvariance
public import NN.Examples.BugZoo.KVCache
public import NN.Examples.BugZoo.RoPEPosition
public import NN.Examples.BugZoo.TokenizerBoundary
public import NN.Examples.BugZoo.Geometry3DProjection

/-!
# BugZoo Case Studies

Folder-local umbrella for public case studies mapping real neural-network bug classes to TorchLean
contracts.

Each submodule follows the same pattern:
- cite the paper or incident class that motivates the bug;
- state the TorchLean boundary honestly (`prevents`, `detects`, or `requires external conformance`);
- expose a small checked theorem or definition tied to the real spec/proof stack.

This is intentionally not a separate theory fork. The examples re-export the actual TorchLean
semantics and proof theorems so the examples stay connected to the library.

The writing style is deliberately plain: tell the reader what the bug is, state the contract in
ordinary language, then show the small checked theorem.
-/

@[expose] public section
