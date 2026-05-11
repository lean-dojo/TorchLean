/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Common
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalenceCommon
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalenceOpCases
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Activations
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Constants
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Elementwise
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.LinearAlgebra
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Normalization
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Pooling
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Permutation
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Random
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Reductions
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Structural
public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops.Unary

/-!
# Semantic Equivalence

End-to-end semantic equivalence proof for the IR -> executable SSA graph bridge.

This module proves semantic equivalence between:
- `NN.IR.Graph.denoteAll*` (IR denotational semantics), and
- `Runtime.Autograd.Compiled.execGraphOfIR` / `GraphData.eval` (compiled runtime).

This module ties the per-op correctness lemmas together into the recursive preservation argument.

## Main definitions

- `buildFrom_preserves_denotation`: recursive preservation theorem for `buildFrom`.
- `execGraphOfIR_semantics_eq`: end-to-end forward semantic equivalence theorem.

## Implementation notes

- The proof mirrors `buildFrom` branch-by-branch. It is verbose, but this "same shape as code"
  style makes regressions easier to diagnose when new ops are added.
- Heartbeat limits are explicit because elaboration cost here is dominated by large dependent
  pattern matches and branch-specific simp normalizations.
- This is one of the slower proof modules in TorchLean. The theorem recursively walks an IR graph,
  dispatches every supported node kind, and maintains equality between an untyped IR value table and
  a typed compiled context. Even simple operator branches can become expensive once shape equality,
  `Except` success/failure paths, and cast proof irrelevance all appear in the same goal.
- Branch-local work belongs in `Correctness/Ops/*` files, with repeated simplification scripts
  replaced by named lemmas. The goal is not to make the theorem clever; the goal is to make each
  compiler branch small enough that adding a new IR op is routine.

## Tags

semantic equivalence, correctness, ir, compiler
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR
open IRExec

set_option maxHeartbeats 12000000 in
/--
Recursive semantic preservation lemma for `buildFrom`.

If `buildFrom` successfully compiles the IR tail starting at node `i`, extending an existing
  compiled
prefix `st` to `st'`, then the IR evaluator `NN.IR.Graph.denoteAllFrom` produces exactly the same
value table as `denoteAllState` for the compiled state.

This theorem is the workhorse behind `execGraphOfIR_semantics_eq`.
-/
private theorem buildFrom_preserves_denotation
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape}
    (i : Nat) (st st' : State α inShape)
    (hNoMSE : NoMSELoss g)
    (h : buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i) st = .ok
      st') :
    ∀ x : Tensor α inShape,
      NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
        (input := NN.IR.DVal.mk (α := α) inShape x)
        (i := i) (vals := denoteAllState (α := α) inShape st x) =
        .ok (denoteAllState (α := α) inShape st' x) := by
  classical
  intro x
  rcases st with ⟨ss, gd⟩
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  -- The runtime context corresponding to the already-compiled prefix.
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()

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
          simpa [hN, Except.instMonad, Except.bind, Except.pure] using hBuild
        cases this
    | ok n =>
        -- Reduce the successful `getNode` and eliminate the resulting `do`-binder.
        simp (config := { failIfUnchanged := false })
          [hN, Except.instMonad, Except.bind, Except.pure] at hBuild
        let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x
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
            (buildFrom_preserves_denotation (α := α) (g := g) (payload := payload) (inShape := inShape)
              (i := i + 1) (st := st1) (st' := st') hNoMSE hRec x)
          all_goals
            simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) hi
        -- Common tail step: unfold `denoteAllFrom` once, rewrite by the `evalAt` step result, then
        -- discharge the remaining tail via the recursive correctness lemma.
        have finish
          {τ : Shape} (nodeData : NodeData α Unit ([inShape] ++ ss) τ)
          (st1 : State α inShape)
          (hRec :
            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape) (i := i + 1) st1 =
              .ok st')
          (hEval :
            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                (input := input) (vals := vals0) (i := i) =
              .ok (NN.IR.DVal.mk (α := α) τ (nodeData.forward ctx ())))
          (hStep :
            denoteAllState (α := α) inShape st1 x =
              vals0.push (NN.IR.DVal.mk (α := α) τ (nodeData.forward ctx ()))) :
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
          | mul_elem =>
              exact buildFrom_denoteAllFrom_mul_elem (α := α) (g := g) (payload := payload)
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
            | maxPool2d kH kW stride =>
                exact buildFrom_denoteAllFrom_max_pool2d (α := α) (g := g) (payload := payload)
                  (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                  (kH := kH) (kW := kW) (stride := stride) hN hk hi hBuild0
                  (fun st1 hRec => tail (st1 := st1) hRec)
            | maxPool2dPad kH kW stride padding =>
                exact buildFrom_denoteAllFrom_max_pool2d_pad (α := α) (g := g) (payload := payload)
                  (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                  (kH := kH) (kW := kW) (stride := stride) (padding := padding) hN hk hi hBuild0
                  (fun st1 hRec => tail (st1 := st1) hRec)
            | avgPool2d kH kW stride =>
                exact buildFrom_denoteAllFrom_avg_pool2d (α := α) (g := g) (payload := payload)
                  (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                  (kH := kH) (kW := kW) (stride := stride) hN hk hi hBuild0
                  (fun st1 hRec => tail (st1 := st1) hRec)
            | avgPool2dPad kH kW stride padding =>
                exact buildFrom_denoteAllFrom_avg_pool2d_pad (α := α) (g := g) (payload := payload)
                  (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                  (kH := kH) (kW := kW) (stride := stride) (padding := padding) hN hk hi hBuild0
                  (fun st1 hRec => tail (st1 := st1) hRec)
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
          | conv2d inC outC kH kW stride padding =>
              exact buildFrom_denoteAllFrom_conv2d (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride)
                (padding := padding)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
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
          | exp =>
              exact buildFrom_denoteAllFrom_exp (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
          | log =>
              exact buildFrom_denoteAllFrom_log (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 tail
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
              exact buildFrom_denoteAllFrom_concat_impossible (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n) (axis := axis)
                hN hk hi hBuild0
          | swap_first_two =>
              exact buildFrom_denoteAllFrom_swap_first_two (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | transpose3dLastTwo =>
              exact buildFrom_denoteAllFrom_transpose3dLastTwo (α := α) (g := g) (payload := payload)
                (gd := gd) (i := i) (st' := st') (x := x) (n := n)
                hN hk hi hBuild0 (fun st1 hRec => tail (st1 := st1) hRec)
          | mseLoss =>
              have : False := (hNoMSE i n hN) hk
              cases this
  · -- Out-of-bounds: compiler is identity and evaluator returns the current table.
    have h0 := h
    unfold buildFrom at h0
    simp [hi] at h0
    cases h0
    unfold NN.IR.Graph.denoteAllFrom
    simp [hi, Except.pure]
termination_by g.nodes.size - i
decreasing_by
  simpa using Nat.sub_succ_lt_self (a := g.nodes.size) (i := i) hi

/--
End-to-end semantic equivalence for successful IR compilation.

If `execGraphOfIR` returns an executable graph, evaluating that executable graph on any input
matches the denotational semantics of the original IR graph.
-/
theorem execGraphOfIR_semantics_eq
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (exec : ExecGraphData α)
    (hNoMSE : NoMSELoss g)
    (h : execGraphOfIR (α := α) g payload = .ok exec) :
    ∀ x : Tensor α exec.inShape,
      NN.IR.Graph.denoteAll (α := α) (g := g) (payload := payload)
          (input := NN.IR.DVal.mk (α := α) exec.inShape x) =
        .ok (ExecGraphData.denoteAll (α := α) (e := exec) x) := by
  classical
  -- Unfold the compiler.
  unfold execGraphOfIR at h
  -- Peel the structural check: failure is impossible if we returned `.ok exec`.
  cases hWF : g.checkWellFormed with
  | error msg =>
      have : False := by
        simpa [hWF] using h
      cases this
  | ok _ =>
      simp [hWF] at h
      -- Get node 0: failure is impossible if we returned `.ok exec`.
      cases hN0 : g.getNode 0 with
      | error msg =>
          have : False := by
            simpa [hN0] using h
          cases this
      | ok n0 =>
          simp (config := { failIfUnchanged := false })
            [hN0, Except.instMonad, Except.bind, Except.pure] at h
          -- Node 0 must be `.input` in the successful compilation path.
          cases hk0 : n0.kind
          case input =>
            -- Reduce `execGraphOfIR` to the `.input` branch.
            simp (config := { failIfUnchanged := false }) [hk0] at h
            -- Extract the successful `buildFrom` tail compilation.
            cases hSt : buildFrom (α := α) (g := g) (payload := payload) (inShape := n0.outShape) (i
              := 1)
                (st := (⟨[], .nil⟩ : State α n0.outShape)) <;>
              simp (config := { failIfUnchanged := false }) [hSt] at h
            · cases h
            · rename_i stFinal
              cases h
              intro x
              -- Rewrite the executable result in terms of `stFinal`.
              have hExec :
                  ExecGraphData.denoteAll (α := α)
                      (e := (fun a ↦ { inShape := n0.outShape, ss := a.fst, g := a.snd }) stFinal) x
                        =
                    denoteAllState (α := α) n0.outShape stFinal x := by
                rfl
              -- Start `denoteAll`: it reduces to `denoteAllFrom 0 #[]` after the well-formedness
              -- check.
              simp [NN.IR.Graph.denoteAll, hWF]
              -- Evaluate node 0 (`.input`), then apply the semantic equivalence lemma from `i=1`.
              have h0 :
                  NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                      (input := NN.IR.DVal.mk (α := α) n0.outShape x) (vals := #[]) (i := 0) =
                    .ok (NN.IR.DVal.mk (α := α) n0.outShape x) := by
                simp [NN.IR.Graph.evalAt, hN0, hk0, NN.IR.Graph.expectShape,
                  NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                  throw_eq_error, Except.instMonad, Except.bind, Except.pure]
              have hTail :
                  NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
                      (input := NN.IR.DVal.mk (α := α) n0.outShape x)
                      (i := 1) (vals := #[NN.IR.DVal.mk (α := α) n0.outShape x]) =
                    .ok (denoteAllState (α := α) n0.outShape stFinal x) := by
                have hInit :
                    denoteAllState (α := α) n0.outShape (st := (⟨[], .nil⟩ : State α n0.outShape)) x
                      =
                      #[NN.IR.DVal.mk (α := α) n0.outShape x] := by
                  simpa using (denoteAllState_nil (α := α) (inShape := n0.outShape) x)
                simpa [hInit] using
                  (buildFrom_preserves_denotation (α := α) (g := g) (payload := payload) (inShape :=
                    n0.outShape)
                      (i := 1) (st := (⟨[], .nil⟩ : State α n0.outShape)) (st' := stFinal)
                      hNoMSE hSt x)
              -- Now unfold `denoteAllFrom` at `i=0` and rewrite by `h0`/`hTail`.
              have hSize : 0 < g.nodes.size := by
                -- If `g.nodes.size = 0`, `getNode 0` would be out of bounds.
                cases hs : g.nodes.size with
                | zero =>
                    have : g.getNode 0 = Except.error s!"IR graph: node id out of bounds: {0}" := by
                      simp [NN.IR.Graph.getNode, NN.IR.Graph.getNode?, hs, throw, throwThe,
                        MonadExceptOf.throw, Except.instMonad, Except.bind, Except.pure]
                    have : False := by
                      -- `hN0` contradicts the computed out-of-bounds error.
                      simpa [this] using hN0
                    cases this
                | succ n =>
                    simpa [hs] using Nat.succ_pos n
              -- With `0 < size`, the `if` guard in `denoteAllFrom` is true at `i=0`.
              unfold NN.IR.Graph.denoteAllFrom
              simp [hSize, h0, hTail, hExec]
          all_goals
            have : False := by
              -- Non-`.input` node0 kinds compile to an error, contradicting success.
              simpa [hk0, throw_eq_error,
                Except.instMonad, Except.bind, Except.pure] using h
            cases this


end Compiled
end Autograd
end Runtime
