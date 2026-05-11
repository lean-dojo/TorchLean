/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps
public import NN.Spec.Models.Autoencoder
public import NN.Spec.Models.Cnn
public import NN.Spec.Models.Gmm
public import NN.Spec.Models.GradientBoostedTrees
public import NN.Spec.Models.Hmm
public import NN.Spec.Models.LinearRegression
public import NN.Spec.Models.LogisticRegression
public import NN.Spec.Models.Mlp
public import NN.Spec.Models.NaiveBayes
public import NN.Spec.Models.Pca
public import NN.Spec.Models.RandomForest
public import NN.Spec.Models.Resnet
public import NN.Spec.Models.Seq2seq
public import NN.Spec.Models.Svm
public import NN.Spec.Models.Transformer
public import NN.Entrypoint.Tensor
public import NN.Tests.Runtime.Floats.Utils
public import Std

/-!
# ModelsSmoke

 Smoke tests for float-backed model specifications. -/

@[expose] public section

open _root_.Spec
open _root_.Spec.Tensor
open ModSpec
open Models
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace ModelsSmoke

abbrev fillND {α : Type} (value : α) (dims : List Nat) :
    Spec.Tensor α (NN.Tensor.shapeOfDims dims) :=
  NN.Tensor.fillND (α := α) value dims

def assertTrue (msg : String) (b : Bool) : IO Unit := do
  if !b then
    throw <| IO.userError msg

def assertProp (msg : String) (p : Prop) [Decidable p] : IO Unit := do
  if !(decide p) then
    throw <| IO.userError msg

def run : IO Unit := do
  IO.println "models_smoke: begin"
  -- --------------------------------------------------------------------------
  -- MLP (forward composition sanity)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: mlp"
  let mlpIn := 2
  let mlpHid := 3
  let mlpOut := 1
  let w1 : Tensor Float (.dim mlpHid (.dim mlpIn .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (0.1 * Float.ofNat (i.val + j.val +
      1))))
  let b1 : Tensor Float (.dim mlpHid .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (0.01 * Float.ofNat (i.val + 1)))
  let w2 : Tensor Float (.dim mlpOut (.dim mlpHid .scalar)) :=
    Tensor.dim (fun _ => Tensor.dim (fun j => Tensor.scalar (0.2 * Float.ofNat (j.val + 1))))
  let b2 : Tensor Float (.dim mlpOut .scalar) := Tensor.dim (fun _ => Tensor.scalar 0.0)
  let l1 : Spec.LinearSpec Float mlpIn mlpHid := { weights := w1, bias := b1 }
  let l2 : Spec.LinearSpec Float mlpHid mlpOut := { weights := w2, bias := b2 }
  let x_mlp : Tensor Float (.dim mlpIn .scalar) :=
    Tensor.dim (fun i => Tensor.scalar ([0.5, 0.8][i.val]!))
  let y_chain := Examples.mlpForward (α := Float) l1 l2 x_mlp
  let y_manual :=
    let z1 := Spec.linearSpec (α := Float) l1 x_mlp
    let a1 := Activation.reluSpec z1
    Spec.linearSpec (α := Float) l2 a1
  assertApprox "mlp forward[0]" (getAtOrZero y_chain [0]) (getAtOrZero y_manual [0])

  -- --------------------------------------------------------------------------
  -- Autoencoder (forward + backward nontriviality)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: autoencoder"
  let aeIn := 3
  let aeHid := 2
  let ae : Spec.AutoencoderSpec Float aeIn aeHid :=
    { encoder_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.1))
      encoder_bias := Tensor.dim (fun _ => Tensor.scalar 0.01)
      decoder_weights := Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 0.2))
      decoder_bias := Tensor.dim (fun _ => Tensor.scalar 0.0)
      activation_type := "tanh" }
  let x_ae : Tensor Float (.dim aeIn .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val + 1)))
  let y_ae := Spec.autoencoderForwardSpec ae x_ae
  let grad_out : Tensor Float (.dim aeIn .scalar) := Tensor.dim (fun _ => Tensor.scalar 1.0)
  let (dWe, _dbe, _dWd, _dbd, _dX) := Spec.autoencoderBackwardSpec ae x_ae grad_out
  assertProp "autoencoder backward expected nonzero dWe[0,0]" (Float.abs (getAtOrZero dWe [0, 0])
    > 1e-12)
  assertFinite "autoencoder forward[0]" (getAtOrZero y_ae [0])

  -- --------------------------------------------------------------------------
  -- PCA (forward + inverse on fixed components)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: pca"
  let pcaIn := 3
  let pcaOut := 2
  let pca : Spec.PCASpec Float pcaIn pcaOut :=
    { components := Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (0.1 * Float.ofNat
      (i.val + j.val + 1))))
      mean := Tensor.dim (fun _ => Tensor.scalar 0.5)
      explained_variance := Tensor.dim (fun i => Tensor.scalar (0.6 - 0.1 * Float.ofNat i.val)) }
  let x_pca : Tensor Float (.dim pcaIn .scalar) := Tensor.dim (fun _ => Tensor.scalar 1.0)
  let z_pca := Spec.pcaForwardSpec pca x_pca
  let x_pca' := Spec.pcaInverseSpec pca z_pca
  assertFinite "pca forward[0]" (getAtOrZero z_pca [0])
  assertFinite "pca inverse[0]" (getAtOrZero x_pca' [0])

  -- --------------------------------------------------------------------------
  -- CNN (small forward check)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: cnn"
  let inC := 1
  let inH := 6
  let inW := 6
  let outC := 1
  let k := 3
  let stride := 1
  let pad := 1
  let poolK := 2
  let poolS := 2
  have h_inC : inC ≠ 0 := by decide
  have h_outC : outC ≠ 0 := by decide
  have h_k : k ≠ 0 := by decide
  have h_poolK : poolK ≠ 0 := by decide
  have h_poolS : poolS ≠ 0 := by decide
  let x_cnn : Tensor Float (.dim inC (.dim inH (.dim inW .scalar))) :=
    fillND (α := Float) 0.0 [inC, inH, inW]
  let convW1 : Tensor Float (.dim outC (.dim inC (.dim k (.dim k .scalar)))) :=
    fillND (α := Float) 0.0 [outC, inC, k, k]
  let convB1 : Tensor Float (.dim outC .scalar) := fillND (α := Float) 0.0 [outC]
  let convW2 : Tensor Float (.dim outC (.dim outC (.dim k (.dim k .scalar)))) :=
    fillND (α := Float) 0.0 [outC, outC, k, k]
  let convB2 : Tensor Float (.dim outC .scalar) := fillND (α := Float) 0.0 [outC]
  let pool : Spec.MaxPool2DSpec poolK poolK poolS h_poolK h_poolK h_poolS :=
    { kernelHeight := poolK, kernelWidth := poolK, stride := poolS }
  let conv1 : Spec.Conv2DSpec inC outC k k stride pad Float h_inC h_k h_k :=
    { kernel := convW1, bias := convB1 }
  let conv2 : Spec.Conv2DSpec outC outC k k stride pad Float h_outC h_k h_k :=
    { kernel := convW2, bias := convB2 }
  let convOut (h : Nat) := (h + 2 * pad - k) / stride + 1
  let poolOut (h : Nat) := (h - poolK) / poolS + 1
  let h1' := convOut inH
  let h2' := poolOut h1'
  let h3' := convOut h2'
  let h4' := poolOut h3'
  let w1' := convOut inW
  let w2' := poolOut w1'
  let w3' := convOut w2'
  let w4' := poolOut w3'
  let flatShape := Shape.dim outC (Shape.dim h4' (Shape.dim w4' Shape.scalar))
  let flatSize : Nat := flatShape.size
  let linW : Tensor Float (.dim outC (Shape.dim flatSize Shape.scalar)) :=
    fillND (α := Float) 1.0 [outC, flatSize]
  let linB : Tensor Float (.dim outC .scalar) := fillND (α := Float) 0.0 [outC]
  let lin : Spec.LinearSpec Float flatSize outC := { weights := linW, bias := linB }
  let cnn := Models.cnnSpec (α := Float) conv1 conv2 pool pool lin
  let y_cnn := SpecChain.forward (α := Float) cnn x_cnn
  assertFinite "cnn output[0]" (getAtOrZero y_cnn [0])

  -- --------------------------------------------------------------------------
  -- GMM (responsibilities sum to ~1)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: gmm"
  let kG := 2
  let dG := 1
  have h_kG : kG ≠ 0 := by decide
  let gmm : Spec.GMMSpec Float kG dG :=
    { weights := Tensor.dim (fun _ => Tensor.scalar 0.5)
      means := Tensor.dim (fun i => Tensor.dim (fun _ => Tensor.scalar (Float.ofNat i.val)))
      covariances := Tensor.dim (fun _ =>
        Tensor.dim (fun _ => Tensor.dim (fun _ => Tensor.scalar 1.0))) }
  let x_gmm : Tensor Float (.dim dG .scalar) := Tensor.dim (fun _ => Tensor.scalar 0.2)
  let resp := Spec.gmmExpectationSpec gmm x_gmm h_kG
  let respSum :=
    match resp with
    | Tensor.dim f =>
      (List.finRange kG).foldl (fun acc i =>
        match f i with
        | Tensor.scalar v => acc + v) 0.0
  assertApprox "gmm responsibilities sum" respSum 1.0 1e-4

  -- --------------------------------------------------------------------------
  -- HMM (forward prob is positive)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: hmm"
  let nStates := 2
  let nObs := 2
  let hmm : Spec.HMMSpec Float nStates nObs :=
    { init_prob := fillND (α := Float) 0.5 [nStates]
      trans_prob := fillND (α := Float) 0.5 [nStates, nStates]
      emission_prob := fillND (α := Float) 0.5 [nStates, nObs] }
  let obs : Spec.ObservationSeq nObs := [fin0!, fin1!]
  let p_obs := Spec.hmmForwardSpec (α := Float) hmm obs
  assertProp "hmm forward expected positive prob" (p_obs > 0.0)

  -- --------------------------------------------------------------------------
  -- Gradient boosted trees (simple additive ensemble)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: gbt"
  let nTrees := 2
  let maxDepth := 1
  let nFeatures := 2
  let tree1 : Spec.DecisionTreeSpec Float maxDepth := { root := Spec.TreeNode.leaf 1.0 }
  let tree2 : Spec.DecisionTreeSpec Float maxDepth := { root := Spec.TreeNode.leaf 2.0 }
  let gbt : Spec.GradientBoostedTreesSpec Float nTrees maxDepth :=
    { trees := Tensor.dim (fun i => Tensor.scalar (if i.val = 0 then tree1 else tree2))
      learning_rate := 0.1
      initial_prediction := 0.5 }
  let x_gbt : Tensor Float (.dim nFeatures .scalar) := fillND (α := Float) 0.0 [nFeatures]
  let y_gbt := Spec.gradientBoostedTreesForwardSpec gbt x_gbt
  assertApprox "gbt forward" (toScalar y_gbt) (0.5 + 0.1 * (1.0 + 2.0)) 1e-6

  -- --------------------------------------------------------------------------
  -- Linear regression (one train step returns finite loss)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: linear_regression"
  let lrIn := 2
  let batch := 2
  have h_batch : batch ≠ 0 := by decide
  let linReg : Spec.LinearRegressionSpec Float lrIn :=
    { weights := fillND (α := Float) 0.0 [lrIn], bias := Tensor.scalar 0.0 }
  let X_lr : Tensor Float (.dim batch (.dim lrIn .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (i.val + j.val + 1))))
  let y_lr : Tensor Float (.dim batch .scalar) := fillND (α := Float) 1.0 [batch]
  let (loss_lr, _model') := Spec.linearRegressionTrainStepSpec linReg X_lr y_lr 0.1 h_batch
  assertFinite "linear_regression loss" (toScalar loss_lr)

  -- --------------------------------------------------------------------------
  -- Logistic regression (predictProba outputs in [0,1])
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: logistic_regression"
  let p := 2
  let n := 4
  let X_log : Tensor Float (.dim n (.dim p .scalar)) :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        match (i.val, j.val) with
        | (0,0) => Tensor.scalar 0.0
        | (0,1) => Tensor.scalar 0.0
        | (1,0) => Tensor.scalar 0.0
        | (1,1) => Tensor.scalar 1.0
        | (2,0) => Tensor.scalar 1.0
        | (2,1) => Tensor.scalar 0.0
        | (3,0) => Tensor.scalar 1.0
        | (3,1) => Tensor.scalar 1.0
        | _ => Tensor.scalar 0.0))
  let y_log : Tensor Float (.dim n .scalar) :=
    Tensor.dim (fun i =>
      Tensor.scalar (if i.val = 0 then 0.0 else 1.0))
  let logModel := fitLogistic (α := Float) X_log y_log 0.1 2
  let probs := predictProba (α := Float) logModel X_log
  for i in List.range n do
    let v := getAtOrZero probs [i]
    assertIn01 s!"logistic_regression prob[{i}]" v

  -- --------------------------------------------------------------------------
  -- Naive Bayes (predicts expected label)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: naive_bayes"
  let data : List NaiveBayes.Example :=
    [ { features := ["rain", "cold"], label := "stay_in" }
    , { features := ["sunny", "warm"], label := "go_out" }
    , { features := ["sunny"], label := "go_out" }
    ]
  let pred := NaiveBayes.predict data ["sunny", "warm"] Float
  assertProp s!"naive_bayes expected go_out, got {pred}" (pred = "go_out")

  -- --------------------------------------------------------------------------
  -- Random forest (majority vote)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: random_forest"
  let tA : DecisionTree String := DecisionTree.leaf "A"
  let tB : DecisionTree String := DecisionTree.leaf "B"
  let forest : random_forest.RandomForest String := { trees := [tA, tA, tB] }
  let decisionFn : String → Bool := fun _ => true
  let agg : List String → String := fun xs =>
    match random_forest.majorityVote xs with
    | some a => a
    | none => "B"
  let predRF := random_forest.predict forest decisionFn agg
  assertProp s!"random_forest expected A, got {predRF}" (predRF = "A")

  -- --------------------------------------------------------------------------
  -- SVM (small fit terminates and predicts ±1)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: svm"
  let nS := 4
  let pS := 2
  let X_svm : Tensor Float (.dim nS (.dim pS .scalar)) := X_log
  let y_svm : Tensor Float (.dim nS .scalar) :=
    Tensor.dim (fun i => Tensor.scalar (if i.val = 0 then -1.0 else 1.0))
  let svmModel := fitLinearSVM (α := Float) X_svm y_svm 0.01 0.1 5
  let yhat := predict (α := Float) svmModel X_svm
  for i in List.range nS do
    let v := getAtOrZero yhat [i]
    assertTrue s!"svm predict expected ±1, got {v}" ((v == 1.0) || (v == -1.0))

  -- --------------------------------------------------------------------------
  -- Seq2Seq (training forward compiles and inference uses embedding lookup)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: seq2seq"
  let srcV := 5
  let tgtV := 4
  let eDim := 3
  let hDim := 4
  let srcLen := 2
  let tgtLen := 3
  have h_srcV : srcV ≠ 0 := by decide
  have h_tgtV : tgtV ≠ 0 := by decide
  have h_eDim : eDim ≠ 0 := by decide
  have h_hDim : hDim ≠ 0 := by decide
  have h_srcLen : srcLen ≠ 0 := by decide
  have h_tgtLen : tgtLen ≠ 0 := by decide
  let srcEmb : Spec.Seq2SeqEmbeddingSpec Float srcV eDim :=
    { embedding := fillND (α := Float) 0.1 [srcV, eDim]
      dropout_rate := 0.0 }
  let tgtEmb : Spec.Seq2SeqEmbeddingSpec Float tgtV eDim :=
    { embedding := fillND (α := Float) 0.2 [tgtV, eDim]
      dropout_rate := 0.0 }
  let enc : Spec.Seq2SeqRNNEncoderSpec Float eDim hDim :=
    { rnn := { weights := fillND (α := Float) 0.05 [hDim, eDim + hDim]
               bias := fillND (α := Float) 0.01 [hDim] }
      dropout_rate := 0.0 }
  let dec : Spec.Seq2SeqDecoderSpec Float eDim hDim tgtV :=
    { rnn := { weights := fillND (α := Float) 0.07 [hDim, eDim + hDim]
               bias := fillND (α := Float) 0.02 [hDim] }
      attention := none
      output_projection := { weights := fillND (α := Float) 0.03 [tgtV, hDim]
                             bias := fillND (α := Float) 0.0 [tgtV] }
      dropout_rate := 0.0 }
  let s2s : Spec.Seq2SeqSpec Float srcV tgtV eDim hDim :=
    { src_embedding := srcEmb
      tgt_embedding := tgtEmb
      encoder := enc
      decoder := dec }
  let srcTokens : Tensor Nat (.dim srcLen .scalar) := Tensor.dim (fun i => Tensor.scalar (i.val %
    srcV))
  let tgtTokens : Tensor Nat (.dim tgtLen .scalar) := Tensor.dim (fun i => Tensor.scalar (i.val %
    tgtV))
  let logitsTrain := Spec.Seq2SeqSpec.forwardTraining s2s srcTokens tgtTokens h_srcV h_tgtV h_eDim
    h_hDim h_srcLen h_tgtLen
  assertFinite "seq2seq training logits[0,0]" (getAtOrZero logitsTrain [0, 0])
  let (logitsInf, _tokInf) := Spec.Seq2SeqSpec.forwardInference tgtLen s2s srcTokens 0 h_srcV
    h_tgtV h_eDim h_hDim h_srcLen h_tgtLen
  assertFinite "seq2seq inference logits[0,0]" (getAtOrZero logitsInf [0, 0])

  -- --------------------------------------------------------------------------
  -- Transformer (small encoder forward)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: transformer"
  let mha : Spec.MultiHeadAttention Float 1 2 2 :=
    { Wq := identityTensorSpec 2
      Wk := identityTensorSpec 2
      Wv := identityTensorSpec 2
      Wo := identityTensorSpec 2 }
  let ffn : Spec.FeedForward 2 4 Float :=
    { W1 := fillND (α := Float) 0.0 [2, 4]
      W2 := fillND (α := Float) 0.0 [4, 2]
      b1 := fillND (α := Float) 0.0 [4]
      b2 := fillND (α := Float) 0.0 [2] }
  let encLayer : Spec.TransformerEncoderLayer 1 2 4 Float :=
    { mha := mha
      ffn := ffn
      norm1_gamma := fillND (α := Float) 1.0 [2]
      norm1_beta := fillND (α := Float) 0.0 [2]
      norm2_gamma := fillND (α := Float) 1.0 [2]
      norm2_beta := fillND (α := Float) 0.0 [2] }
  let encoder : Spec.TransformerEncoder 1 1 2 4 Float := { layers := [encLayer] }
  let x_tr : Tensor Float (.dim 2 (.dim 2 .scalar)) := identityTensorSpec 2
  have h_seq : (2 : Nat) > 0 := by decide
  have h_embed : (2 : Nat) > 0 := by decide
  let y_tr := Spec.TransformerEncoder.forward encoder x_tr h_seq h_embed
  assertFinite "transformer output[0,0]" (getAtOrZero y_tr [0, 0])

  -- --------------------------------------------------------------------------
  -- ResNet (construction sanity; avoid heavy conv evaluation)
  -- --------------------------------------------------------------------------
  IO.println "models_smoke: resnet"
  let inCh := 1
  let nCls := 2
  let H := 8
  let W := 8
  have h_inCh : inCh ≠ 0 := by decide
  have h_nCls : nCls ≠ 0 := by decide
  have h_H : H ≠ 0 := by decide
  have h_W : W ≠ 0 := by decide
  let resnet := Spec.ResNet18Spec Float inCh nCls h_inCh h_nCls
  let depth := Spec.ResNetSpec.depth (cfg := Spec.resnet18Config) h_inCh h_nCls
    Spec.resnet18Config_wf resnet
  let params := Spec.ResNetSpec.parameterCount (cfg := Spec.resnet18Config) h_inCh h_nCls
    Spec.resnet18Config_wf resnet
  assertProp "resnet expected positive depth" (depth > 0)
  assertProp "resnet expected positive parameterCount" (params > 0)

  IO.println "models_smoke: OK"

end ModelsSmoke
end Floats
end Tests
/-!
Model zoo runtime smoke tests (floats).

This file runs a curated set of small models to ensure they compile and execute in the float
runtime. It is intended to be fast and stable.
-/
