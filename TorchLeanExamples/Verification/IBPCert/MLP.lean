/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.IBPCert.ExampleInputs
public import TorchLean.NN.IR.Builder

/-!
# IBP certificate check for a small MLP

A compact end-to-end example of checking an IBP certificate for a small MLP in TorchLean's
graph-based verifier. `model` builds the graph of registered operators with its weights, and the
input box is `IBPCert.naturalBox 3 1.0`. The JSON certificate, checked against the bounds
Lean recomputes, is exported from Python by
`TorchLeanExamples/Verification/IBPCert/export_cert.py`.

Run it through the unified CLI registry:

- `scripts/lake.sh exe verify -- ibp-cert [path]`

Without `path`, the CLI uses the bundled certificate
`TorchLeanExamples/Verification/IBPCert/mlp_cert.json`.
-/

@[expose] public section


namespace TorchLean.Examples.IBPCert

open TorchLean.Tensor

/-- The `3 → 4 → 2` ReLU MLP of the IBP certificate example as a graph of registered operators,
with its fixed Float weights and biases as constants. The graph holds the input, then each linear
layer's weight, bias and `Op.linear`, with a ReLU between the two layers. -/
def model : Except String (IR.Graph × IR.Payload Float) := do
  let hiddenWeight : Tensor Float [4, 3] :=
    Tensor.generate [4, 3] fun
      | [i, j] => Float.ofNat (1 + i + j)
      | _ => 0.0
  let hiddenBias : Tensor Float [4] :=
    Tensor.generate [4] fun
      | [i] => Float.ofNat (i + 1)
      | _ => 0.0
  let outputWeight : Tensor Float [2, 4] :=
    Tensor.generate [2, 4] fun
      | [i, j] => Float.ofNat (2 + i + j)
      | _ => 0.0
  let outputBias : Tensor Float [2] :=
    Tensor.generate [2] fun
      | [i] => Float.ofNat i
      | _ => 0.0
  let (b, x) := ({} : IR.Graph.Builder).input 0 (.value [3])
  let (b, P, w) := b.constant {} ⟨.value [4, 3], hiddenWeight⟩
  let (b, P, c) := b.constant P ⟨.value [4], hiddenBias⟩
  let (b, h) := b.op (Op.linear (lead := []) (inDim := 3) (outDim := 4)) [w, c, x]
  let (b, r) := b.op (Op.relu (s := [4])) h
  let (b, P, w) := b.constant P ⟨.value [2, 4], outputWeight⟩
  let (b, P, c) := b.constant P ⟨.value [2], outputBias⟩
  let (b, y) := b.op (Op.linear (lead := []) (inDim := 4) (outDim := 2)) ([w, c] ++ r)
  pure (← b.build y.toArray, P)

/-- Check whether the output bounds in the certificate JSON at `path` contain the IBP bounds that
Lean recomputes for `model`, and return the checker's report, which rejects the certificate when
they do not. -/
def verifyCert (path : String) : IO Certificate.Report := do
  let (g, P) ← IO.ofExcept model
  -- A radius-1 box around [1, 2, 3] at the graph input.
  Certificate.IBP.check g P [naturalBox 3 1.0] 0 path

end TorchLean.Examples.IBPCert
