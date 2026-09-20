/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence

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

TorchLean's answer is a semantic boundary. For the supported IR fragment, successful lowering to
the executable typed graph is justified by the theorem below: executable evaluation agrees with the
denotational source semantics. External compilers and GPU kernels still need their own conformance
evidence; this theorem does not silently extend across those boundaries.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.BugZoo.CompilerBoundary

/--
Successful lowering preserves the denotation of every node for every input.

The one fragment hypothesis excludes raw logarithm nodes, whose source semantics rejects
nonpositive inputs while the executable graph totalizes them. The executable graph is a Lean
reference evaluator; native kernels and external compilers remain separate conformance boundaries.
-/
theorem successfulLowering_preservesDenotation
    {α : Type} [Storage α] [Context α]
    (graph : NN.IR.Graph)
    (payload : NN.IR.Payload α)
    (executable : Runtime.Autograd.IRExec.ForwardGraph α)
    (hNoRawLog : Runtime.Autograd.IRExec.NoRawLog graph)
    (hLowered :
      Runtime.Autograd.IRExec.lowerToForwardGraph (α := α) graph payload = .ok executable) :
    ∀ input : Tensor α executable.inShape,
      NN.IR.Graph.denoteAll (α := α) (g := graph) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) executable.inShape input) =
        .ok (Runtime.Autograd.IRExec.ForwardGraph.denoteAll
          (α := α) (e := executable) input) :=
  Runtime.Autograd.IRExec.denoteAll_eq_of_lowerToForwardGraph
    graph payload executable hNoRawLog hLowered

end NN.Examples.BugZoo.CompilerBoundary
