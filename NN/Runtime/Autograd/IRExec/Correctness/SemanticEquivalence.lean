/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceOpCases
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Activations
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Concat
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Constants
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Convolution
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Elementwise
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.LinearAlgebra
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Loss
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Normalization
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Pooling
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Permutation
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Random
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Reductions
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Structural
public import NN.Runtime.Autograd.IRExec.Correctness.Ops.Unary
import NN.Tactic.Except

/-!
# Semantic Equivalence

End-to-end semantic equivalence proof for the IR -> executable SSA graph bridge.

This module proves equivalence between two Lean-level graph interpretations:
- `NN.IR.Graph.denoteAll*` (IR denotational semantics), and
- `Runtime.Autograd.IRExec.lowerToForwardGraph` / `ForwardGraph.eval` (the shape-indexed reference
  evaluator).

It does not identify that reference evaluator with eager tensor execution, native CPU kernels,
CUDA, or LibTorch. Those implementations require their own refinement boundary.

## Main definitions

- `buildFrom_preserves_denotation`: recursive preservation theorem for `buildFrom`.
- `denoteAll_eq_of_lowerToForwardGraph`: end-to-end forward semantic equivalence theorem, under the
  `NoRawLog` side condition.

The recursive proof dispatches to the operation lemmas in `Correctness/Ops`. Its invariant relates
the IR value table to the shape-indexed context for the already-lowered prefix.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace IRExec

open Spec TorchLean
open Proofs.Autograd.Algebra
open NN.IR
open Internal

/--
Recursive semantic preservation lemma for `buildFrom`.

If `buildFrom` successfully lowers the IR tail starting at node `i`, extending an existing
lowered prefix `st` to `st'`, then the IR evaluator `NN.IR.Graph.denoteAllFrom` produces exactly
the same value table as `denoteAllState` for the lowered state, for the named supported fragment.

This theorem is the workhorse behind `denoteAll_eq_of_lowerToForwardGraph`.
-/
private theorem buildFrom_preserves_denotation
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape}
    (i : Nat) (st st' : State α inShape)
    (hNoRawLog : NoRawLog g)
    (h : buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i) st = .ok
      st') :
    ∀ x : Tensor α inShape,
      NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
        (input := Spec.SomeTensor.mk (α := α) inShape x)
        (i := i) (vals := denoteAllState (α := α) inShape st x) =
        .ok (denoteAllState (α := α) inShape st' x) := by
  classical
  intro x
  rcases st with ⟨ss, gd⟩

  by_cases hi : i < g.nodes.size
  ·
    have hBuild := h
    unfold buildFrom at hBuild
    simp [hi] at hBuild
    cases hN : g.getNode i with
    | error msg =>
        have : False := by
          simp [hN] at hBuild
        cases this
    | ok n =>
        simp (config := { failIfUnchanged := false })
          [hN] at hBuild
        let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x
        -- The tail advances the node index while preserving the prefix invariant.
        have tail
            (st1 : State α inShape)
            (hRec :
              buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i + 1) st1
                = .ok st') :
            NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
                (input := input) (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
              .ok (denoteAllState (α := α) inShape st' x) := by
          simpa [input] using
            (buildFrom_preserves_denotation (α := α) (g := g) (payload := payload)
              (inShape := inShape) (i := i + 1) (st := st1) (st' := st') hNoRawLog hRec x)
        cases hk : n.kind with
          | custom name shapes output =>
              simp [hk] at hBuild
          | input =>
              exact buildFrom_denoteAllFrom_input_impossible (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h
          | const s =>
              exact buildFrom_denoteAllFrom_const (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (s := s)
                hN hk hi h tail
          | permute perm =>
              exact buildFrom_denoteAllFrom_permute (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (perm := perm)
                hN hk hi h tail
          | detach =>
              exact buildFrom_denoteAllFrom_detach (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | randUniform seed =>
              exact buildFrom_denoteAllFrom_randUniform (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (seed := seed) hN hk hi h
                tail
          | bernoulliMask seed =>
              exact buildFrom_denoteAllFrom_bernoulliMask (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (seed := seed) hN hk hi h
                tail
          | add =>
              exact buildFrom_denoteAllFrom_add (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | sub =>
              exact buildFrom_denoteAllFrom_sub (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | mulElem =>
              exact buildFrom_denoteAllFrom_mulElem (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | abs =>
              exact buildFrom_denoteAllFrom_abs (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | sqrt =>
              exact buildFrom_denoteAllFrom_sqrt (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | inv =>
              exact buildFrom_denoteAllFrom_inv (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | maxElem =>
              exact buildFrom_denoteAllFrom_maxElem (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | minElem =>
              exact buildFrom_denoteAllFrom_minElem (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h tail
          | maxPool config =>
              exact buildFrom_denoteAllFrom_maxPool (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (config := config)
                hN hk hi h tail
          | avgPool config =>
              exact buildFrom_denoteAllFrom_avgPool (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (config := config)
                hN hk hi h tail
          | broadcastTo s₁ s₂ =>
              exact buildFrom_denoteAllFrom_broadcastTo (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (s₁ := s₁) (s₂ := s₂) hN hk hi h tail
          | reduceSum axis =>
              exact buildFrom_denoteAllFrom_reduceSum (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (axis := axis) hN hk hi h tail
          | reduceMean axis =>
              exact buildFrom_denoteAllFrom_reduceMean (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (axis := axis) hN hk hi h tail
          | sum =>
              exact buildFrom_denoteAllFrom_sum (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | matmul =>
                exact buildFrom_denoteAllFrom_matmul (α := α) (g := g) (payload := payload)
                  (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h
                  tail
          | linear =>
              exact buildFrom_denoteAllFrom_linear (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | conv config =>
              exact buildFrom_denoteAllFrom_conv (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (config := config)
                hN hk hi h tail
          | batchNormEval channelAxis channels =>
              exact buildFrom_denoteAllFrom_batchNormEval (α := α) (g := g)
                (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (channelAxis := channelAxis) (channels := channels) hN hk hi h
                tail
          | relu =>
              exact buildFrom_denoteAllFrom_relu (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | tanh =>
              exact buildFrom_denoteAllFrom_tanh (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | sigmoid =>
              exact buildFrom_denoteAllFrom_sigmoid (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | softplus =>
              exact buildFrom_denoteAllFrom_softplus (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | safeLog =>
              exact buildFrom_denoteAllFrom_safeLog (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | exp =>
              exact buildFrom_denoteAllFrom_exp (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | log =>
              have hImpossible : False := hNoRawLog i n hN hk
              cases hImpossible
          | sin =>
              exact buildFrom_denoteAllFrom_sin (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | cos =>
              exact buildFrom_denoteAllFrom_cos (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi h tail
          | softmax axis =>
              exact buildFrom_denoteAllFrom_softmax (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (axis := axis) hN hk hi h tail
          | hardMaskedSoftmax mask =>
              exact buildFrom_denoteAllFrom_hardMaskedSoftmax
                (α := α) (g := g) (payload := payload) (gd := gd) (i := i)
                (st' := st') (x := x) (n := n) (mask := mask)
                hN hk hi h tail
          | layernorm axis =>
                exact buildFrom_denoteAllFrom_layernorm (α := α) (g := g) (payload := payload)
                  (gd := gd) (i := i) (st' := st') (x := x) (n := n) (axis := axis)
                  hN hk hi h
                  tail
          | reshape inS outS =>
              exact buildFrom_denoteAllFrom_reshape (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (inS := inS) (outS := outS)
                hN hk hi h tail
          | flatten s =>
              exact buildFrom_denoteAllFrom_flatten (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (s := s)
                hN hk hi h tail
          | concat axis =>
              exact buildFrom_denoteAllFrom_concat (α := α) g payload gd i st' x n axis hN hk hi
                h tail
          | transpose axis₁ axis₂ =>
              exact buildFrom_denoteAllFrom_transpose (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (axis₁ := axis₁) (axis₂ := axis₂) hN hk hi h
                tail
          | mseLoss =>
              exact buildFrom_denoteAllFrom_mseLoss (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi h
                tail
  · -- Out-of-bounds: lowering pass is identity and evaluator returns the current table.
    have h0 := h
    unfold buildFrom at h0
    simp [hi] at h0
    cases h0
    unfold NN.IR.Graph.denoteAllFrom
    simp [hi]
    rfl
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) hi

/--
End-to-end semantic equivalence for successful IR lowering.

If `lowerToForwardGraph` returns an executable graph, evaluating that executable graph on any input
matches the denotational semantics of the original IR graph.

One side condition remains. `NoRawLog` excludes raw `.log`: the IR evaluator rejects nonpositive
inputs while the lowered closure applies `Tensor.logSpec` to every input, so the two can only be
compared under a positivity precondition the theorem does not carry.

Under `NoRawLog`, the theorem covers every successfully lowered operation kind, including
`.mseLoss` and `.concat` along any axis. Matmul uses the canonical dimension contract, with vector
promotion and broadcasting of leading dimensions; linear accepts arbitrary leading dimensions.
Custom computations are rejected by this forward-only lowering. The theorem concerns successful
lowering, not equality of the two interpreters' failure results.
-/
theorem denoteAll_eq_of_lowerToForwardGraph
    {α : Type} [TorchLean.Storage α] [Context α]
    (g : NN.IR.Graph) (payload : Payload α) (exec : ForwardGraph α)
    (hNoRawLog : NoRawLog g)
    (h : lowerToForwardGraph (α := α) g payload = .ok exec) :
    ∀ x : Tensor α exec.inShape,
      NN.IR.Graph.denoteAll (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) exec.inShape x) =
        .ok (ForwardGraph.denoteAll (α := α) (e := exec) x) := by
  classical
  unfold lowerToForwardGraph at h
  except_cases hWF : g.checkWellFormed using h with _checked =>
    simp [hWF] at h
    except_cases hN0 : g.getNode 0 using h with n0 =>
      simp (config := { failIfUnchanged := false }) [hN0] at h
      cases hk0 : n0.kind
      case input =>
        simp (config := { failIfUnchanged := false }) [hk0] at h
        except_cases hSt : buildFrom (α := α) (g := g) (payload := payload)
            (inShape := n0.outShape) (i := 1)
            (st := (⟨[], .nil⟩ : State α n0.outShape)) using h with stFinal =>
          simp [hSt] at h
          cases h
          intro x
          have hExec :
              ForwardGraph.denoteAll (α := α)
                  (e := (fun a ↦ { inShape := n0.outShape, ss := a.fst, body := a.snd })
                    stFinal) x
                    =
                denoteAllState (α := α) n0.outShape stFinal x := by
            rfl
          simp [NN.IR.Graph.denoteAll, hWF]
          -- Seed the prefix with the input before applying the tail invariant.
          have h0 :
              NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                  (input := Spec.SomeTensor.mk (α := α) n0.outShape x) (vals := #[]) (i := 0) =
                .ok (Spec.SomeTensor.mk (α := α) n0.outShape x) := by
            simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
              NN.IR.Graph.normalizeNodeOutput, hN0, hk0, NN.IR.Graph.expectShape,
              throw_eq_error]
            rfl
          have hTail :
              NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
                  (input := Spec.SomeTensor.mk (α := α) n0.outShape x)
                  (i := 1) (vals := #[Spec.SomeTensor.mk (α := α) n0.outShape x]) =
                .ok (denoteAllState (α := α) n0.outShape stFinal x) := by
            have hInit :
                denoteAllState (α := α) n0.outShape (st := (⟨[], .nil⟩ : State α n0.outShape)) x
                  =
                  #[Spec.SomeTensor.mk (α := α) n0.outShape x] := by
              simp
            simpa [hInit] using
              (buildFrom_preserves_denotation (α := α) (g := g) (payload := payload)
                (inShape := n0.outShape)
                (i := 1) (st := (⟨[], .nil⟩ : State α n0.outShape)) (st' := stFinal)
                hNoRawLog hSt x)
          have hSize : 0 < g.nodes.size := by
            -- If `g.nodes.size = 0`, `getNode 0` would be out of bounds.
            cases hs : g.nodes.size with
            | zero =>
                have : g.getNode 0 = Except.error s!"IR graph: node id out of bounds: {0}" := by
                  simp [NN.IR.Graph.getNode, hs, throw, throwThe,
                    MonadExceptOf.throw]
                have : False := by
                  simp [this] at hN0
                cases this
            | succ n =>
                simp
          unfold NN.IR.Graph.denoteAllFrom
          rw [dite_eq_left hSize, h0, hExec]
          simpa using hTail
      all_goals
        have : False := by
          simp [hk0, throw_eq_error] at h
        cases this


end IRExec
end Autograd
end Runtime
