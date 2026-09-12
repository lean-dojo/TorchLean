/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.DeepDives.Floats.GraphNumericalCertificate
public meta import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate
import NN.Backend.Profile
import NN.Floats.Interval.IEEEExec32
import NN.IR.Graph
import NN.IR.Payload
import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Contracts
import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Enclosure
import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Certificate
import NN.Spec.Core.Tensor.SomeTensor

/-!
# Numerical certificate regression coverage

Operation transfers, malformed assumptions, unsupported operations, and backend reduction policies.
The example supplies the small graph and complete MLP; these fixtures retain the independent
positive and negative cases without placing the test suite in the tutorial.
-/

@[expose] public section

open Proofs.RuntimeApprox.NumericalCertificate
open Spec TorchLean
open TorchLean.Floats.IEEE754
open NN.Examples.DeepDives.Floats.GraphNumericalCertificate

namespace NN.Tests.Verification.GraphNumericalCertificate

/-- A finite interval attached to an arithmetic node is still an invalid source assumption. -/
def misplacedSourceCheck : Except String GraphNumericalCertificate :=
  generate NN.Backend.BackendProfile.checkedCpu graph <|
    sources.push { nodeId := 2, enclosure := interval 0x00000000 0x3f800000 }

/-- Registries are deterministic maps: registering a second source contract is rejected. -/
def duplicateContractCheck : Except String GraphRangeRegistry := do
  let registry <- defaultRegistry
  registry.register sourceContract

/-- A certificate is bound to the named operation registry used to derive its transfer rows. -/
def registryMismatchCheck : Except String RegistryCheckedCertificate := do
  let raw <- generate NN.Backend.BackendProfile.checkedCpu graph sources
  let registry <- defaultRegistry
  let renamed := { registry with name := "example.incompatible-registry" }
  checkWith renamed NN.Backend.BackendProfile.checkedCpu graph raw

/-! ## Coverage before propagation

Coverage is checked after any architecture has lowered to the common IR. An architecture using only
registered primitives needs no architecture-specific checker. A new primitive is rejected with its
node id and operation name until a local range contract is registered.
-/

/-- The base graph is completely covered by the built-in range registry. -/
def baseCoverage : Except String NumericalCoverageReport := do
  let registry <- defaultRegistry
  requireNumericalCoverage registry graph

/-- Exponential is executable in the graph IR, but it intentionally has no built-in interval
transfer yet. This graph demonstrates that unsupported numerical semantics fail explicitly. -/
def unsupportedGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [] },
      { id := 1, parents := #[0], kind := .exp, outShape := [] }
    ] }

/-- Coverage failure occurs before certificate propagation begins. -/
def unsupportedCoverage : Except String NumericalCoverageReport := do
  let registry <- defaultRegistry
  requireNumericalCoverage registry unsupportedGraph

/-! ## A fixed-order reduction

Reduction order is part of the backend audit because floating-point addition is not associative.
The portable profile advertises the same left fold used by `Tensor.sumSpec`, so the checker can
propagate this reduction directly. Native CUDA's implementation-dependent reduction policy is not
silently treated as the same computation.
-/

def reductionGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input,
        outShape := [3] },
      { id := 1, parents := #[0], kind := .sum, outShape := [] }
    ] }

/-- The input range for the reduction example, covering `[-1, 2]`. -/
def reductionSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xbf800000 0x40000000 }
]

/--
Three concrete entries whose exact sum is representable, so any drift comes from the summation order
rather than from the values themselves.
-/
def reductionInput : Spec.SomeTensor IEEE32Exec := by
  let tensor : Tensor IEEE32Exec [3] :=
    [ IEEE32Exec.ofBits 0x3f800000
    , IEEE32Exec.ofBits 0xbf000000
    , IEEE32Exec.ofBits 0x40000000 ]
  exact { shape := [3], tensor }

/-- Generate the certificate for the portable CPU profile, then replay it at bit level. -/
def reductionReplay : Except String RangeCheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu reductionGraph reductionSources
  executeIEEE32 {} reductionInput certificate

/-! ## Matrix accumulation

The same reduction policy governs matrix multiplication. Each output entry is a fixed-left sum of
products in the portable profile, so the checker combines outward-rounded multiplication with the
existing sum transfer. CUDA profiles advertise an implementation-dependent accumulation and are
not accepted by this particular transfer.
-/

def matmulGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input,
        outShape := [2, 2] },
      { id := 1, parents := #[], kind := .const [2, 2],
        outShape := [2, 2] },
      { id := 2, parents := #[0, 1], kind := .matmul,
        outShape := [2, 2] }
    ] }

/-- Ranges for the input matrix and the constant weight matrix. -/
def matmulSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xbf800000 0x3f800000 },
  { nodeId := 1, enclosure := interval 0x00000000 0x3f800000 }
]

/--
The constant weight, supplied as the payload of node 1: the two-by-two identity in row-major order.
-/
def matmulPayload : NN.IR.Payload IEEE32Exec where
  const? := fun nodeId =>
    if nodeId = 1 then
      some
        { n := 4
          v := [ IEEE32Exec.posOne, IEEE32Exec.posZero
               , IEEE32Exec.posZero, IEEE32Exec.posOne ] }
    else
      none

/-- A concrete two-by-two input, so the replay has actual bits to work with. -/
def matmulInput : Spec.SomeTensor IEEE32Exec := by
  let tensor : Tensor IEEE32Exec [2, 2] :=
    [ [IEEE32Exec.posOne, IEEE32Exec.negOne]
    , [IEEE32Exec.negOne, IEEE32Exec.posOne] ]
  exact { shape := [2, 2], tensor }

/-- Certificate plus bit-level replay for the matrix product. -/
def matmulReplay : Except String RangeCheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu matmulGraph matmulSources
  executeIEEE32 matmulPayload matmulInput certificate

/-- Attempt to use the fixed-left matrix transfer with a CUDA reduction policy. -/
def cudaMatmulCertificate : Except String GraphNumericalCertificate :=
  generate NN.Backend.BackendProfile.checkedCuda matmulGraph matmulSources

/-! ## Domain-sensitive square root

The checker propagates absolute value before checking the square-root domain. Thus an input range
that crosses zero is valid for `abs → sqrt`, while the same range passed directly to `sqrt` is
rejected. The square-root endpoints use TorchLean's proved directed binary32 rounders rather than a
host `libm` call.
-/

def sqrtGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [] },
      { id := 1, parents := #[0], kind := .abs, outShape := [] },
      { id := 2, parents := #[1], kind := .sqrt, outShape := [] }
    ] }

/--
A source range crossing zero, `[-4, 9]`. Legal for `abs` followed by `sqrt`, illegal for `sqrt`
alone, which is the contrast this section is built around.
-/
def sqrtSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xc0800000 0x41100000 }
]

/--
A negative concrete input, `-4`, to show that `abs` really is what makes the domain condition hold.
-/
def sqrtInput : Spec.SomeTensor IEEE32Exec :=
  Spec.SomeTensor.ofTensor (Tensor.full [] (IEEE32Exec.ofBits 0xc0800000))

/-- Certificate plus replay for the `abs` then `sqrt` chain. -/
def sqrtReplay : Except String RangeCheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu sqrtGraph sqrtSources
  executeIEEE32 {} sqrtInput certificate

/-- The same graph with `abs` removed, so `sqrt` receives the sign-crossing range directly. -/
def invalidSqrtGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input, outShape := [] },
      { id := 1, parents := #[0], kind := .sqrt, outShape := [] }
    ] }

/-- A source interval containing negative values does not satisfy the real square-root domain. -/
def invalidSqrtCertificate : Except String GraphNumericalCertificate :=
  generate NN.Backend.BackendProfile.checkedCpu invalidSqrtGraph sqrtSources

/-! ## Layer normalization

LayerNorm combines several domain-sensitive steps. The certificate follows the implementation:
mean, centering, squaring, variance, epsilon stabilization, directed square root, and division. The
portable profile fixes the reduction order used by both means.
-/

def layerNormGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input,
        outShape := [2, 3] },
      { id := 1, parents := #[0], kind := .layernorm 1,
        outShape := [2, 3] }
    ] }

/-- Entry range `[-2, 2]` for the LayerNorm input. -/
def layerNormSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xc0000000 0x40000000 }
]

/-- A two-by-three input for the LayerNorm replay. -/
def layerNormInput : Spec.SomeTensor IEEE32Exec := by
  let tensor : Tensor IEEE32Exec [2, 3] :=
    [ [IEEE32Exec.negOne, IEEE32Exec.posZero, IEEE32Exec.posOne]
    , [IEEE32Exec.ofBits 0x40000000, IEEE32Exec.posOne, IEEE32Exec.posZero] ]
  exact { shape := [2, 3], tensor }

/-- Certificate plus replay for LayerNorm on the portable profile. -/
def layerNormReplay : Except String RangeCheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu layerNormGraph layerNormSources
  executeIEEE32 {} layerNormInput certificate

/-- The same fixed-left LayerNorm transfer is not attributed to an unspecified CUDA reduction. -/
def cudaLayerNormCertificate : Except String GraphNumericalCertificate :=
  generate NN.Backend.BackendProfile.checkedCuda layerNormGraph layerNormSources

/-!
## Stable axis softmax

The real softmax theorem proves that a nonempty row lies in $[0,1]$; the bit-level replay then
checks that the executable implementation stayed finite and respected that range for the concrete
input.
-/

def softmaxGraph : NN.IR.Graph :=
  { nodes := #[
      { id := 0, parents := #[], kind := .input,
        outShape := [3] },
      { id := 1, parents := #[0], kind := .softmax 0,
        outShape := [3] }
    ] }

/-- Logit range `[-2, 2]` for the softmax example. -/
def softmaxSources : Array SourceRange := #[
  { nodeId := 0, enclosure := interval 0xc0000000 0x40000000 }
]

/-- Three concrete logits for the softmax replay. -/
def softmaxInput : Spec.SomeTensor IEEE32Exec := by
  let tensor : Tensor IEEE32Exec [3] :=
    [IEEE32Exec.posOne, IEEE32Exec.posZero, IEEE32Exec.negOne]
  exact { shape := [3], tensor }

/-- Certificate plus replay for the numerically stable softmax. -/
def softmaxReplay : Except String RangeCheckedExecution := do
  let certificate <-
    generateChecked NN.Backend.BackendProfile.checkedCpu softmaxGraph softmaxSources
  executeIEEE32 {} softmaxInput certificate

/-- Executable acceptance report. Positive cases should be `true`; deliberately corrupted,
invalid-domain, or wrong-reduction-policy cases should be `false`. This array exercises range
reconstruction and IEEE replay; it does not construct the separate exact-real enclosure proof. -/
def exampleChecks : Array (String × Bool) :=
  NN.Examples.DeepDives.Floats.GraphNumericalCertificate.exampleChecks ++
  #[ ("misplaced source rejected", !accepted misplacedSourceCheck)
  , ("duplicate contract rejected", !accepted duplicateContractCheck)
  , ("registry mismatch rejected", !accepted registryMismatchCheck)
  , ("base graph coverage", accepted baseCoverage)
  , ("unsupported operation rejected", !accepted unsupportedCoverage)
  , ("fixed-left reduction", accepted reductionReplay)
  , ("portable matmul", accepted matmulReplay)
  , ("CUDA matmul policy rejected", !accepted cudaMatmulCertificate)
  , ("directed sqrt", accepted sqrtReplay)
  , ("negative sqrt domain rejected", !accepted invalidSqrtCertificate)
  , ("portable LayerNorm", accepted layerNormReplay)
  , ("CUDA LayerNorm policy rejected", !accepted cudaLayerNormCertificate)
  , ("stable softmax", accepted softmaxReplay)
  ]

/-- Every positive case must succeed and every invalid artifact must be rejected. -/
def run : IO Unit := do
  for (name, ok) in exampleChecks do
    unless ok do
      throw <| IO.userError s!"numerical certificate regression failed: {name}"

end NN.Tests.Verification.GraphNumericalCertificate
