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

This module ties the per-op correctness lemmas together into the recursive preservation argument.

## Main definitions

- `buildFrom_preserves_denotation`: recursive preservation theorem for `buildFrom`.
- `denoteAll_eq_of_lowerToForwardGraph`: end-to-end forward semantic equivalence theorem, under the
  `NoRawLog` side condition.

## Implementation notes

- The proof mirrors `buildFrom` branch-by-branch. It is verbose, but this "same shape as code"
  style makes regressions easier to diagnose when new ops are added.
- This is one of the slower proof modules in TorchLean. The theorem recursively walks an IR graph,
  dispatches every supported node kind, and maintains equality between an untyped IR value table and
  a shape-indexed execution context. Even simple operator branches can become expensive once shape
  equality, `Except` success/failure paths, and cast proof irrelevance all appear in the same goal.
- Branch-local work belongs in `Correctness/Ops/*` files, with repeated simplification scripts
  replaced by named lemmas. Each lowering branch should stay small enough that adding a new IR op is
  routine.

## Tags

semantic equivalence, correctness, ir, lowering pass
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
  let vals0 : Array (Spec.SomeTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  -- The runtime context corresponding to the already-lowered prefix.
  let ctx : TorchLean.TensorPack α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)

  by_cases hi : i < g.nodes.size
  · -- Step case.
    have hBuild0 :
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i)
            (st := (⟨ss, gd⟩ : State α inShape)) = .ok st' := h
    have hBuild :
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i)
            (st := (⟨ss, gd⟩ : State α inShape)) = .ok st' := hBuild0
    unfold buildFrom at hBuild
    simp [hi] at hBuild
    cases hN : g.getNode i with
    | error msg =>
        -- `buildFrom` cannot return `.ok` if `getNode` fails.
        have : False := by
          simp [hN] at hBuild
        cases this
    | ok n =>
        -- Reduce the successful `getNode` and eliminate the resulting `do`-binder.
        simp (config := { failIfUnchanged := false })
          [hN] at hBuild
        let input : Spec.SomeTensor α := Spec.SomeTensor.mk (α := α) inShape x
        -- Tail correctness helper: wrap the recursive call so the termination side-goal is solved
        -- immediately at the call site.
        have tail
            (st1 : State α inShape)
            (hRec :
              buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i + 1) st1
                = .ok st') :
            NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
                (input := input) (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
              .ok (denoteAllState (α := α) inShape st' x) := by
          -- The recursive call leaves a termination side-goal (`size - (i+1) < size - i`) which we
          -- discharge from the `hi : i < g.nodes.size` step-case hypothesis.
          simpa [input] using
            (buildFrom_preserves_denotation (α := α) (g := g) (payload := payload)
              (inShape := inShape) (i := i + 1) (st := st1) (st' := st') hNoRawLog hRec x)
        -- Common tail step: unfold `denoteAllFrom` once, rewrite by the `evalAt` step result, then
        -- discharge the remaining tail via the recursive correctness lemma.
        have finish
          {τ : Shape} (nodeData : ForwardNode α ([inShape] ++ ss) τ)
          (st1 : State α inShape)
          (hRec :
            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i + 1) st1 =
              .ok st')
          (hEval :
            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                (input := input) (vals := vals0) (i := i) =
              .ok (Spec.SomeTensor.mk (α := α) τ (nodeData.eval ctx)))
          (hStep :
            denoteAllState (α := α) inShape st1 x =
              vals0.push (Spec.SomeTensor.mk (α := α) τ (nodeData.eval ctx))) :
            NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
                (input := input) (i := i) (vals := vals0) =
              .ok (denoteAllState (α := α) inShape st' x) := by
          have hTail := tail (st1 := st1) hRec
          exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
            (i := i) (x := x) (hi := hi) (τ := τ) (nodeData := nodeData) (st1 := st1) (st' := st')
            (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
        -- Mirror the node step, then recurse.
        cases hk : n.kind with
          | input =>
              exact buildFrom_denoteAllFrom_input_impossible (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0
          | const s =>
              exact buildFrom_denoteAllFrom_const (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (s := s)
                hN hk hi hBuild0 tail
          | permute perm =>
              exact buildFrom_denoteAllFrom_permute (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (perm := perm)
                hN hk hi hBuild0 tail
          | detach =>
              exact buildFrom_denoteAllFrom_detach (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | randUniform seed =>
              exact buildFrom_denoteAllFrom_rand_uniform (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (seed := seed) hN hk hi hBuild0
                (fun st1 hRec => tail (st1 := st1) hRec)
          | bernoulliMask seed =>
              exact buildFrom_denoteAllFrom_bernoulli_mask (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (seed := seed) hN hk hi hBuild0
                (fun st1 hRec => tail (st1 := st1) hRec)
          | add =>
              exact buildFrom_denoteAllFrom_add (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | sub =>
              exact buildFrom_denoteAllFrom_sub (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | mulElem =>
              exact buildFrom_denoteAllFrom_mulElem (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | abs =>
              exact buildFrom_denoteAllFrom_abs (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | sqrt =>
              exact buildFrom_denoteAllFrom_sqrt (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | inv =>
              exact buildFrom_denoteAllFrom_inv (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | maxElem =>
              exact buildFrom_denoteAllFrom_max_elem (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | minElem =>
              exact buildFrom_denoteAllFrom_min_elem (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0 tail
          | maxPool config =>
              exact buildFrom_denoteAllFrom_maxPool (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (config := config)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | avgPool config =>
              exact buildFrom_denoteAllFrom_avgPool (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (config := config)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | broadcastTo s₁ s₂ =>
              exact buildFrom_denoteAllFrom_broadcastTo (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (s₁ := s₁) (s₂ := s₂) hN hk hi hBuild0 tail
          | reduceSum axis =>
              exact buildFrom_denoteAllFrom_reduceSum (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (axis := axis) hN hk hi hBuild0 tail
          | reduceMean axis =>
              exact buildFrom_denoteAllFrom_reduceMean (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (axis := axis) hN hk hi hBuild0 tail
          | sum =>
              exact buildFrom_denoteAllFrom_sum (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
            | matmul =>
                exact buildFrom_denoteAllFrom_matmul (α := α) (g := g) (payload := payload)
                  (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0
                  (fun st1 hRec => tail (st1 := st1) hRec)
          | linear =>
              exact buildFrom_denoteAllFrom_linear (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | conv config =>
              exact buildFrom_denoteAllFrom_conv (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (config := config)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | batchNormEval channelAxis channels =>
              exact buildFrom_denoteAllFrom_batchNormEval (α := α) (g := g)
                (payload := payload) (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (channelAxis := channelAxis) (channels := channels) hN hk hi hBuild0
                (fun st1 hRec => tail (st1 := st1) hRec)
          | relu =>
              exact buildFrom_denoteAllFrom_relu (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | tanh =>
              exact buildFrom_denoteAllFrom_tanh (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | sigmoid =>
              exact buildFrom_denoteAllFrom_sigmoid (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | softplus =>
              exact buildFrom_denoteAllFrom_softplus (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | safeLog =>
              exact buildFrom_denoteAllFrom_safeLog (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | exp =>
              exact buildFrom_denoteAllFrom_exp (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | log =>
              have hImpossible : False := hNoRawLog i n hN hk
              cases hImpossible
          | sin =>
              exact buildFrom_denoteAllFrom_sin (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | cos =>
              exact buildFrom_denoteAllFrom_cos (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | softmax axis =>
              exact buildFrom_denoteAllFrom_softmax (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (axis := axis) hN hk hi hBuild0 tail
          | hardMaskedSoftmax mask =>
              exact buildFrom_denoteAllFrom_hardMaskedSoftmax
                (α := α) (g := g) (payload := payload) (gd := gd) (i := i)
                (st' := st') (x := x) (n := n) (mask := mask)
                hN hk hi hBuild0 tail
            | layernorm axis =>
                exact buildFrom_denoteAllFrom_layernorm (α := α) (g := g) (payload := payload)
                  (gd := gd) (i := i) (st' := st') (x := x) (n := n) (axis := axis)
                  hN hk hi hBuild0
                  (fun st1 hRec => tail (st1 := st1) hRec)
          | reshape inS outS =>
              exact buildFrom_denoteAllFrom_reshape (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (inS := inS) (outS := outS)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | flatten s =>
              exact buildFrom_denoteAllFrom_flatten (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (s := s)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | concat axis =>
              exact buildFrom_denoteAllFrom_concat (α := α) g payload gd i st' x n axis hN hk hi
                hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | transpose axis₁ axis₂ =>
              exact buildFrom_denoteAllFrom_transpose (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (axis₁ := axis₁) (axis₂ := axis₂) hN hk hi hBuild0
                (fun st1 hRec => tail (st1 := st1) hRec)
          | mseLoss =>
              exact buildFrom_denoteAllFrom_mse_loss (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) hN hk hi hBuild0
                (fun st1 hRec => tail (st1 := st1) hRec)
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

Every other operation kind is covered, including `.mseLoss` and `.concat` along any axis. The
lowering accepts exactly the shapes the IR semantics accepts for `.matmul` and `.linear` (any shared
leading shape, so rank at least four matmul and batched linear are covered); shapes rejected by the
lowering never reach this theorem because `lowerToForwardGraph` returns an error for them.
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
      -- Node 0 must be `.input` in the successful lowering path.
      cases hk0 : n0.kind
      case input =>
        simp (config := { failIfUnchanged := false }) [hk0] at h
        except_cases hSt : buildFrom (α := α) (g := g) (payload := payload)
            (inShape := n0.outShape) (i := 1)
            (st := (⟨[], .nil⟩ : State α n0.outShape)) using h with stFinal =>
          simp [hSt] at h
          cases h
          intro x
          -- Rewrite the executable result in terms of `stFinal`.
          have hExec :
              ForwardGraph.denoteAll (α := α)
                  (e := (fun a ↦ { inShape := n0.outShape, ss := a.fst, body := a.snd })
                    stFinal) x
                    =
                denoteAllState (α := α) n0.outShape stFinal x := by
            rfl
          -- Start `denoteAll`: it reduces to `denoteAllFrom 0 #[]` after the well-formedness check.
          simp [NN.IR.Graph.denoteAll, hWF]
          -- Evaluate node 0 (`.input`), then apply the semantic equivalence lemma from `i=1`.
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
          -- Now unfold `denoteAllFrom` at `i=0` and rewrite by `h0`/`hTail`.
          have hSize : 0 < g.nodes.size := by
            -- If `g.nodes.size = 0`, `getNode 0` would be out of bounds.
            cases hs : g.nodes.size with
            | zero =>
                have : g.getNode 0 = Except.error s!"IR graph: node id out of bounds: {0}" := by
                  simp [NN.IR.Graph.getNode, NN.IR.Graph.getNode?, hs, throw, throwThe,
                    MonadExceptOf.throw]
                have : False := by
                  -- `hN0` contradicts the computed out-of-bounds error.
                  simp [this] at hN0
                cases this
            | succ n =>
                simp
          -- With `0 < size`, the `if` guard in `denoteAllFrom` is true at `i=0`.
          unfold NN.IR.Graph.denoteAllFrom
          rw [dite_eq_left hSize, h0, hExec]
          simpa using hTail
      all_goals
        have : False := by
          -- Non-`.input` node0 kinds lower to an error, contradicting success.
          simp [hk0, throw_eq_error] at h
        cases this


end IRExec
end Autograd
end Runtime
