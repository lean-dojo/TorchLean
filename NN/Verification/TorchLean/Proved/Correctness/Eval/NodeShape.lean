/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Core

/-!
# Compiled Forward Evaluation: Node Shape Preservation
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

open NN.Verification.TorchLean

    /--
    If node evaluation succeeds under a consistent `shapesOfVals` invariant, the resulting dynamic
      value
    has the expected output shape.

    This is a small “shape preservation” lemma used in the main compiler-correctness proof.
    -/
        theorem evalNode_ok_shape_of_hShapes
            {α : Type} [Context α] [DecidableEq Shape]
            {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
        (node : Node α paramShapes inShape ss out)
      (params : Runtime.Autograd.Torch.TList α paramShapes)
      (vals : Array (DVal α))
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      ∀ {v : DVal α}, evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss)
        (out := out)
            node params vals = Except.ok v → v.1 = out := by
        intro v hv
        classical
        -- We only need the *shape tag* of the produced `DVal`. Split on the `getVal` results
        -- without unfolding its dependent cast, then reduce the `do`-blocks with `simp`.
        cases node
        case const wf t =>
            simp [evalNode] at hv
            cases hv
            simp
        case paramConst wf p =>
            simp [evalNode] at hv
            cases hv
            simp
        case add a b =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (DVal.mk (α := α) out (Tensor.addSpec (α := α) ta tb))
                          : Except String (DVal α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case sub a b =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (DVal.mk (α := α) out (Tensor.subSpec (α := α) ta tb))
                          : Except String (DVal α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case mulElem a b =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (DVal.mk (α := α) out (Tensor.mulSpec (α := α) ta tb))
                          : Except String (DVal α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case relu x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (DVal.mk (α := α) out (Activation.reluSpec (α := α) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case exp x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (DVal.mk (α := α) out (Tensor.expSpec (α := α) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case log x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                -- Domain discipline: `log` is undefined outside the positive domain. In Lean's
                -- logic, `panic!` reduces to the default inhabitant, so `evalNode` returns
                -- `if allSpec (0 < ·) tx then logSpec tx else default`.
                simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hv
                simp
        case inv x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (DVal.mk (α := α) out (Tensor.invSpec (α := α) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case matmul2d m n p a b =>
            cases hta :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim m (.dim n .scalar)) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb :
                    getVal (α := α) (inShape := inShape) (ss := ss)
                      (s := .dim n (.dim p .scalar)) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    simp [evalNode, hta, htb] at hv
                    cases hv
                    simp
        case bmm batch m n p a b =>
            cases hta :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim batch (.dim m (.dim n .scalar))) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb :
                    getVal (α := α) (inShape := inShape) (ss := ss)
                      (s := .dim batch (.dim n (.dim p .scalar))) vals b with
                | error e =>
                    have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    simp [evalNode, hta, htb] at hv
                    cases hv
                    simp
        -- `Node.reshape` has arguments `(inS outS : Shape) (hSize : size inS = size outS) (x : Idx … inS)`.
        -- Here `outS` is forced to be the branch output `out`, so `cases` introduces `(inS, x, hSize)`.
        case reshape inS x hSize =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := inS) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok
                        (DVal.mk (α := α) out
                          (Tensor.reshapeSpec (α := α) (s₁ := inS) (s₂ := out) tx hSize))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case swap_first_two m n rest x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := .dim m (.dim n rest)) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok
                        (DVal.mk (α := α) (.dim n (.dim m rest))
                          (Tensor.swapFirstTwoSpec (α := α) (m := m) (n := n) (s := rest) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case transpose3dLastTwo a b c x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim a (.dim b (.dim c .scalar))) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok
                        (DVal.mk (α := α) (.dim a (.dim c (.dim b .scalar)))
                          (Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case softmaxLast _hRank x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (DVal.mk (α := α) out (Activation.softmaxSpec (α := α) tx))
                      : Except String (DVal α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case layernorm2d seqLen embedDim _hSeq _hEmb x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim seqLen (.dim embedDim .scalar)) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq := hv
                simp [evalNode, hx, Bind.bind, Except.bind] at hEq
                cases hEq
                simp
        case linear inDim outDim w b x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := .dim inDim .scalar) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok xT =>
                have hEq := hv
                simp [evalNode, hx, Bind.bind, Except.bind] at hEq
                cases hEq
                simp
        case conv2d inC outC kH kW stride padding inH inW hIn hKH hKW hStride hHeight hWidth kernel bias x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := .dim inC (.dim inH (.dim inW .scalar))) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (DVal α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok xT =>
                have hEq := hv
                simp [evalNode, hx, Bind.bind, Except.bind] at hEq
                cases hEq
                simp
        case mseLoss yhat target =>
            -- `mseLoss` is statically typed with a shared parent shape `s`, but `evalNode`
            -- Mirrors the IR semantics and shape-checks dynamically.
            --
            -- Under `hShapes`, both parent `DVal`s must have shape `s`, so the dynamic check is
            -- provably always true. We reduce the dependent `if` using that proof, rather than
            -- eliminating an unreduced `Decidable.rec`.
            rename_i s
            let yV : DVal α := vals[yhat.id]!
            let tV : DVal α := vals[target.id]!
            have hSomeY : vals[yhat.id]? = some yV := by
              simpa [yV] using
                val_get?_eq_some_of_hShapes (α := α) (vals := vals) (idx := yhat)
                  (hShapes := hShapes)
            have hSomeT : vals[target.id]? = some tV := by
              simpa [tV] using
                val_get?_eq_some_of_hShapes (α := α) (vals := vals) (idx := target)
                  (hShapes := hShapes)
            have hy : yV.shape = s := by
              simpa [yV] using
                shape_of_vals_of_hShapes (α := α) (inShape := inShape) (ss := ss) (s := s)
                  (vals := vals) (idx := yhat) (hShapes := hShapes)
            have ht : tV.shape = s := by
              simpa [tV] using
                shape_of_vals_of_hShapes (α := α) (inShape := inShape) (ss := ss) (s := s)
                  (vals := vals) (idx := target) (hShapes := hShapes)
            -- After unfolding `DVal.shape`, the `evalNode` condition is stated in terms of `.fst`.
            have hCond : yV.fst = tV.fst := by
              -- Bridge through the `shape` equalities coming from `hShapes`.
              simpa [DVal.shape] using (hy.trans ht.symm)
            have hEq := hv
            simp (config := { zeta := true })
              [evalNode, getDVal?, yV, tV, hSomeY, hSomeT, hCond, Bind.bind, Except.bind,
                Except.pure, Pure.pure] at hEq
            cases hEq
            simp
end Correctness

end NN.Verification.TorchLean.Proved
