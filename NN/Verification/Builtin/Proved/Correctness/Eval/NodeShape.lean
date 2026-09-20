/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Proved.Correctness.Eval.Core

/-!
# Lowered Forward Evaluation: Node Shape Preservation
-/

@[expose] public section

namespace NN.Verification.Builtin.Proved

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

namespace Correctness

open NN.Verification.Builtin
-- Typed context indices come from `NN.Proofs.Autograd.Tape.Util.Idx`, the one place
-- `Idx` and `getIdx` are defined.
open Proofs (Idx getIdx)

    /--
    If node evaluation succeeds under a consistent `shapesOfVals` invariant, the resulting dynamic
      value
    has the expected output shape.

    This is a small “shape preservation” lemma used in the main lowering pass-correctness proof.
    -/
        theorem evalNode_ok_shape_of_hShapes
            {α : Type} [TorchLean.Storage α] [Context α]
            {paramShapes : List Shape} {inShape : Shape} {ss : List Shape} {out : Shape}
        (node : Node α paramShapes inShape ss out)
      (params : TorchLean.TensorPack α paramShapes)
      (vals : Array (Spec.SomeTensor α))
      (hShapes : shapesOfVals (α := α) vals = Ctx inShape ss) :
      ∀ {v : Spec.SomeTensor α},
        evalNode (α := α) (paramShapes := paramShapes) (inShape := inShape) (ss := ss) (out := out)
            node params vals = Except.ok v → v.1 = out := by
        intro v hv
        classical
        -- We only need the *shape tag* of the produced `Spec.SomeTensor`. Split on the `getVal`
        -- results without unfolding its dependent cast, then reduce the `do`-blocks with `simp`.
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
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq :
                        (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (Spec.SomeTensor.mk (α := α) out (Tensor.addSpec (α := α) ta tb))
                          : Except String (Spec.SomeTensor α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case sub a b =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq :
                        (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (Spec.SomeTensor.mk (α := α) out (Tensor.subSpec (α := α) ta tb))
                          : Except String (Spec.SomeTensor α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case mulElem a b =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals b with
                | error e =>
                    have hEq :
                        (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    have hEq :
                        (Except.ok (Spec.SomeTensor.mk (α := α) out (Tensor.mulSpec (α := α) ta tb))
                          : Except String (Spec.SomeTensor α)) =
                          Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                      exact hv
                    cases hEq
                    simp
        case relu x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (Spec.SomeTensor.mk (α := α) out (Activation.reluSpec (α := α) tx))
                      : Except String (Spec.SomeTensor α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case exp x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (Spec.SomeTensor.mk (α := α) out (Tensor.expSpec (α := α) tx))
                      : Except String (Spec.SomeTensor α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case log x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                by_cases hpos :
                    Tensor.allSpec (α := α) (s := out) (fun v => decide (0 < v)) tx = true
                · simp [evalNode, hx, hpos, Bind.bind, Except.bind] at hv
                  cases hv
                  simp
                · simp [evalNode, hx, hpos, Bind.bind, Except.bind] at hv
        case inv x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (Spec.SomeTensor.mk (α := α) out (Tensor.invSpec (α := α) tx))
                      : Except String (Spec.SomeTensor α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case matmul leftShape rightShape a b op =>
            cases hta : getVal (α := α) (inShape := inShape) (ss := ss)
                (s := leftShape) vals a with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hta, Bind.bind, Except.bind] at hv
                cases hEq
            | ok ta =>
                cases htb : getVal (α := α) (inShape := inShape) (ss := ss)
                    (s := rightShape) vals b with
                | error e =>
                    have hEq :
                        (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                      simp [evalNode, hta, htb, Bind.bind, Except.bind] at hv
                    cases hEq
                | ok tb =>
                    simp [evalNode, hta, htb] at hv
                    cases hv
                    simp
        -- `Node.reshape` has arguments
        -- `(inS outS : Shape) (hSize : size inS = size outS) (x : Idx … inS)`.
        -- Here `outS` is forced to be the branch output `out`, so `cases` introduces
        -- `(inS, x, hSize)`.
        case reshape inS x hSize =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := inS) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok
                        (Spec.SomeTensor.mk (α := α) out
                          (Tensor.reshapeSpec (α := α) (source := inS) (target := out) tx hSize))
                      : Except String (Spec.SomeTensor α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case transpose s axis₁ axis₂ x hOut =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := s) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                cases hPerm : OpContracts.transposePerm s.rank axis₁ axis₂ with
                | error e =>
                    simp [evalNode, hx, hPerm, Bind.bind, Except.bind] at hv
                | ok perm =>
                    cases hEval : Graph.permuteSomeTensor (α := α)
                        (Spec.SomeTensor.mk (α := α) s tx) perm with
                    | error e =>
                        simp [evalNode, hx, hPerm, hEval, Bind.bind, Except.bind] at hv
                    | ok y =>
                        cases hShape : Graph.expectShape (α := α) (expected := out) y with
                        | error e =>
                            simp [evalNode, hx, hPerm, hEval, hShape, Bind.bind, Except.bind] at hv
                        | ok ty =>
                            simp [evalNode, hx, hPerm, hEval, hShape, Bind.bind, Except.bind] at hv
                            cases hv
                            simp
        case softmax axis hAxis x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq :
                    (Except.ok (Spec.SomeTensor.mk (α := α) out
                      (@Activation.softmaxSpec α _ _ out axis hAxis tx))
                      : Except String (Spec.SomeTensor α)) =
                      Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                  exact hv
                cases hEq
                simp
        case layerNorm op x =>
            cases hx : getVal (α := α) (inShape := inShape) (ss := ss) (s := out) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok tx =>
                have hEq := hv
                simp [evalNode, hx, Bind.bind, Except.bind] at hEq
                cases hEq
                simp
        case linear inDim outDim w b x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss) (s := .dim inDim .scalar)
                  vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
                  simp [evalNode, hx, Bind.bind, Except.bind] at hv
                cases hEq
            | ok xT =>
                have hEq := hv
                simp [evalNode, hx, Bind.bind, Except.bind] at hEq
                cases hEq
                simp
        case conv inC outC kernelShape stride padding inSpatial hIn hKernel hStride hInfer
            kernel bias x =>
            cases hx :
                getVal (α := α) (inShape := inShape) (ss := ss)
                  (s := Shape.ofList (inC :: Tensor.to inSpatial (List Nat))) vals x with
            | error e =>
                have hEq : (Except.error e : Except String (Spec.SomeTensor α)) = Except.ok v := by
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
            -- Under `hShapes`, both parent `Spec.SomeTensor`s must have shape `s`, so the dynamic
            -- check is provably always true. We reduce the dependent `if` using that proof, rather
            -- than eliminating an unreduced `Decidable.rec`.
            rename_i s
            let yV : Spec.SomeTensor α := packedAt vals yhat hShapes
            let tV : Spec.SomeTensor α := packedAt vals target hShapes
            have hSomeY : vals[yhat.id]? = some yV := by
              simpa [yV] using
                getElem?_eq_some_packedAt (α := α) (vals := vals) (idx := yhat)
                  (hShapes := hShapes)
            have hSomeT : vals[target.id]? = some tV := by
              simpa [tV] using
                getElem?_eq_some_packedAt (α := α) (vals := vals) (idx := target)
                  (hShapes := hShapes)
            have hy : yV.shape = s := by
              simp [yV]
            have ht : tV.shape = s := by
              simp [tV]
            have hEq := hv
            simp (config := { zeta := true })
              [evalNode, getValue?, yV, tV, hSomeY, hSomeT, Bind.bind, Except.bind,
                Except.pure, Pure.pure] at hEq
            cases hEq
            simp
end Correctness

end NN.Verification.Builtin.Proved
