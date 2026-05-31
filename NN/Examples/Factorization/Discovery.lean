/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: the CHD discovery decision layer

These checks corroborate `NN.Proofs.Tensor.Basic.FactorizationsDecision`. Once a kernel is built and its
`noise` level computed (`varNoiseSpec`, proven to lie in `[0,1]`), CHD's *discovery loop* turns those
numbers into graph-structure decisions (`decision.py`, `_GraphDiscoveryMain.py`). We exercise the four
deterministic choices the loop makes, each with a positive check and a negative control:

* **prune the least-activated ancestor** — `argMinFn` returns the index of the smallest activation
  (`min_activation = np.argmin(activations)`); the *most*-activated ancestor is correctly **not** chosen;
* **pick the kernel mode that admits an edge** — `kernelChooserFn` (`MinNoiseKernelChooser`) returns the
  valid kernel (`noise < Z_low`) of least `noise`, or `none` when no kernel is valid;
* **report the pruning iteration of largest `noise` jump** — `modeChooserFn` (`MaxIncrementModeChooser`)
  returns the `argmax` of the increments;
* **stop when every ancestor is pruned** — `allPrunedFn` fires on the all-zero mask and not before.

The final block closes the loop end-to-end: it builds an SPD kernel, eigendecomposes it, and runs a
`find_gamma`-style sweep — feeding the *verified* `varNoiseSpec` at several `γ` straight into `argMinFn`
to select the regularization with least noise. Every decision runs over `Float`, the executable runtime
scalar.
-/

@[expose] public section


namespace NN.Examples.Factorization.Discovery

/-- A length-3 `Float` family `Fin 3 → Float` from three entries. -/
def vec3 (a b c : Float) : Fin 3 → Float := fun i => [a, b, c].getD i.val 0.0
/-- A length-4 `Float` family `Fin 4 → Float` from four entries. -/
def vec4 (a b c d : Float) : Fin 4 → Float := fun i => [a, b, c, d].getD i.val 0.0

/-- Build a length-`n` `Float` vector tensor from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- Encode a chooser verdict as an `Int`: `-1` for `none` ("no ancestor"), else the chosen index. -/
def chooserCode {m : Nat} (o : Option (Fin m)) : Int :=
  match o with
  | none => -1
  | some i => Int.ofNat i.val

/-- Compiled positive assertion that a `Bool` decision is `true`. -/
def assertTrue (name : String) (b : Bool) : IO Unit :=
  if b then IO.println s!"{name}: OK"
  else throw (IO.userError s!"{name}: FAIL (expected true)")

/-- Compiled negative-control assertion that a `Bool` decision is `false` (the property correctly does
*not* hold). -/
def assertFalse (name : String) (b : Bool) : IO Unit :=
  if b then throw (IO.userError s!"{name}: FAIL (expected false)")
  else IO.println s!"{name}: OK (correctly false)"

/-! ## Pruning: `argMinFn` removes the least-activated ancestor -/

/-- Activations of four candidate ancestors; ancestor 1 is the least-activated. -/
def activations : Fin 4 → Float := vec4 0.8 0.2 0.5 0.9

#eval IO.println s!"activations = {(List.finRange 4).map activations}, \
  argMin = {(Spec.argMinFn activations).val}"

-- Positive — the prune step removes the least-activated ancestor (`argMinFn_le`).
#eval assertTrue "prune picks the least-activated ancestor (argmin = 1)"
  ((Spec.argMinFn activations).val == 1)

-- Negative — it does *not* remove the most-activated ancestor (index 3).
#eval assertFalse "prune does not pick the most-activated ancestor"
  ((Spec.argMinFn activations).val == 3)

/-! ## Kernel chooser: least-noise valid kernel, or `none` -/

/-- Three candidate kernels' `noise` levels and `Z_low` lower bounds. Validity is `noise < Z_low`:
kernel 0 invalid (`0.3 ≥ 0.2`), kernel 1 valid (`0.1 < 0.4`), kernel 2 invalid (`0.5 ≥ 0.1`). -/
def noisesA : Fin 3 → Float := vec3 0.3 0.1 0.5
def ZlowsA : Fin 3 → Float := vec3 0.2 0.4 0.1

#eval IO.println s!"kernel chooser (one valid) -> code {chooserCode (Spec.kernelChooserFn noisesA ZlowsA)}"

-- Positive — exactly kernel 1 is valid, so the chooser admits an edge via kernel 1 (`kernelChooserFn_eq_some`).
#eval assertTrue "kernel chooser selects the unique valid kernel (some 1)"
  (chooserCode (Spec.kernelChooserFn noisesA ZlowsA) == 1)

/-- Two valid kernels (0 and 1); the chooser must take the one of *least* noise (kernel 0, `0.05`). -/
def noisesB : Fin 3 → Float := vec3 0.05 0.1 0.5
def ZlowsB : Fin 3 → Float := vec3 0.2 0.4 0.1

-- Positive — among valid kernels the chooser takes least noise (kernel 0 beats kernel 1).
#eval assertTrue "kernel chooser takes least noise among valid (some 0)"
  (chooserCode (Spec.kernelChooserFn noisesB ZlowsB) == 0)

/-- No kernel is valid (`noise ≥ Z_low` everywhere): the chooser reports "no ancestor". -/
def noisesC : Fin 3 → Float := vec3 0.5 0.6 0.7
def ZlowsC : Fin 3 → Float := vec3 0.1 0.2 0.3

-- Negative — no valid kernel ⟹ no edge (`kernelChooserFn_eq_none`); code `-1`.
#eval assertTrue "kernel chooser reports none when no kernel is valid (code -1)"
  (chooserCode (Spec.kernelChooserFn noisesC ZlowsC) == -1)

/-! ## Mode chooser: the iteration of largest `noise` increment -/

/-- The per-iteration `noise` sequence of a pruning run. The big jump `0.08 → 0.9` is between iterations
1 and 2, so `MaxIncrementModeChooser` reports iteration 1 (increment `0.82`). -/
def noiseSeq : Fin 4 → Float := vec4 0.05 0.08 0.9 0.95

#eval IO.println s!"increments = {(List.finRange 4).map (Spec.modeIncrementFn noiseSeq)}, \
  modeChooser = {(Spec.modeChooserFn noiseSeq).val}"

-- Positive — the mode chooser reports the largest-jump iteration (`modeChooserFn_ge`).
#eval assertTrue "mode chooser picks the largest noise-increment iteration (argmax = 1)"
  ((Spec.modeChooserFn noiseSeq).val == 1)

-- Negative — it does *not* report a tiny-increment iteration (iteration 0, increment 0.03).
#eval assertFalse "mode chooser does not pick a tiny-increment iteration"
  ((Spec.modeChooserFn noiseSeq).val == 0)

/-! ## Stopping rule: fire exactly when all ancestors are pruned -/

-- Positive — the loop stops when every ancestor mode is zero (`allPrunedFn_iff`).
#eval assertTrue "stopping rule fires when all ancestors are pruned"
  (Spec.allPrunedFn (vec3 0.0 0.0 0.0))

-- Negative — it does not fire while an ancestor remains active.
#eval assertFalse "stopping rule does not fire while an ancestor remains"
  (Spec.allPrunedFn (vec3 0.0 1.0 0.0))

/-! ## End-to-end: `find_gamma` feeds the verified `noise` into `argMinFn`

A `find_gamma`-style sweep: build an SPD kernel, eigendecompose it, evaluate the verified
`varNoiseSpec` at several `γ`, and let `argMinFn` pick the regularization of least noise — exactly the
discovery layer consuming the verified statistic. More regularization means more noise, so the smallest
`γ` wins (index 0). -/

/-- A `3 × 3` symmetric positive-definite kernel. -/
def K : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[2.0, 0.5, 0.3],
         [0.5, 2.0, 0.4],
         [0.3, 0.4, 2.0]]

/-- Its eigendecomposition `(evals, V)` from the Jacobi solver. -/
def evals : Spec.Tensor Float (.dim 3 .scalar) := (Spec.symEigJacobiSpec K 12).1
def V : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := (Spec.symEigJacobiSpec K 12).2

/-- The data vector `ga`. -/
def ga : Spec.Tensor Float (.dim 3 .scalar) := mkVec [1.0, 2.0, 3.0]

/-- The candidate regularizations, increasing. -/
def gammas : Fin 3 → Float := vec3 0.01 0.1 1.0

/-- The verified `noise` at each candidate `γ` (`find_gamma`'s loss). -/
def noiseAt : Fin 3 → Float := fun i => Spec.varNoiseSpec evals V (gammas i) ga

#eval IO.println s!"find_gamma noises = {(List.finRange 3).map noiseAt}, \
  argMin γ index = {(Spec.argMinFn noiseAt).val}"

-- Positive — every swept noise is a genuine fraction in [0,1] (numeric witness of `varNoiseFn_nonneg`/`_le_one`).
#eval assertTrue "find_gamma noises all lie in [0,1]"
  ((List.finRange 3).all (fun i => 0.0 ≤ noiseAt i && noiseAt i ≤ 1.0))

-- Positive — `find_gamma` (argmin of the verified noise) selects the least-regularized γ (index 0).
#eval assertTrue "find_gamma selects least-noise γ via argMinFn (index 0)"
  ((Spec.argMinFn noiseAt).val == 0)

end NN.Examples.Factorization.Discovery
