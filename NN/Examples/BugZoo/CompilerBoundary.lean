/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor

/-!
# BugZoo: compiler and export semantic mismatches

DL compiler bugs are especially dangerous because they can be silent: the optimized graph runs and
returns a tensor, but its semantics no longer match the source model.

NNSmith is the clean citation for this class. It generates valid neural-network graphs, searches for
inputs that avoid floating-point exceptional values, and differentially tests DL compilers. The
authors report 72 new bugs across TVM, TensorRT, ONNXRuntime, and PyTorch, with 58 confirmed and 51
fixed:

- Liu et al., “NNSmith: Generating Diverse and Valid Test Cases for Deep Learning Compilers”,
  ASPLOS 2023.
  https://doi.org/10.1145/3575693.3575707
  https://arxiv.org/abs/2207.13066

FreeFuzz gives the same warning at the framework/API level: mining real usage snippets found
confirmed PyTorch/TensorFlow library bugs, including backend- and mode-specific failures:

- Wei et al., “Free Lunch for Testing: Fuzzing Deep-Learning Libraries from Open Source”,
  ICSE 2022.
  https://arxiv.org/abs/2201.06589

A 2026 PyTorch-compiler study focuses on the same kind of boundary: silent `torch.compile`
correctness bugs where compiled models return incorrect outputs without an exception or warning:

- Li et al., “Demystifying the Silence of Correctness Bugs in PyTorch Compiler”, 2026.
  https://arxiv.org/abs/2604.08720

TorchLean's answer is a semantic boundary. For the supported IR fragment, successful compilation to
the executable graph should be justified by a theorem that executable evaluation agrees with the
denotational source semantics. External compilers and GPU kernels still need their own conformance
evidence; this file spells out the contract shape so that “we tested it once” is not confused with a
semantic guarantee.

The full TorchLean compiler-correctness chapter contains the stronger IR-specific theorem. This
BugZoo card stays lightweight so that importing the examples chapter does not force every heavy
compiler proof to elaborate.
-/

@[expose] public section

namespace NN.Examples.BugZoo.CompilerBoundary

open Spec

/--
The semantic contract at compiler/export/backend boundaries.

`sourceEval` is the reference semantics, `targetEval` is the compiled/exported/backend semantics,
and `preserves` says the target agrees with the source on every input. This compact structure is not a
replacement for the IR-specific compiler proof; it is the reusable shape of the claim that NNSmith-
style and FreeFuzz-style bugs violate.
-/
structure SemanticBoundary
    (Source Target Input Output : Type) where
  /-- Reference/source semantics. -/
  sourceEval : Source → Input → Output
  /-- Target/backend semantics. -/
  targetEval : Target → Input → Output
  /-- Relation saying that `target` was produced from or is meant to implement `source`. -/
  implements : Source → Target → Prop
  /-- Semantic preservation required for an accepted implementation. -/
  preserves :
    ∀ {source : Source} {target : Target},
      implements source target → ∀ input, targetEval target input = sourceEval source input

/-- Once a backend has the semantic-boundary evidence, every accepted implementation agrees. -/
theorem acceptedBackend_matches_source
    {Source Target Input Output : Type}
    (boundary : SemanticBoundary Source Target Input Output)
    {source : Source} {target : Target}
    (h : boundary.implements source target) :
    ∀ input, boundary.targetEval target input = boundary.sourceEval source input :=
  boundary.preserves h

end NN.Examples.BugZoo.CompilerBoundary
