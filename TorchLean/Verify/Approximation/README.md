# `TorchLean/Verify/Approximation`

The floating-point side of universal approximation sits in `FloatInterval/`, in this directory,
and in `TorchLean/Verify/UniversalApproximation/`, which carries the one-dimensional ReLU hinge
network `hingeFun` through the rounded-real `NF` model and configured binary arithmetic. Only over
ℝ is `hingeFun` identified with `MLP.forward` (`mlpEvalScalar_hinge`). The two-layer
decomposition `relu_mlp_binary_three_term` takes all three error bounds as hypotheses, and
`FloatInterval/` evaluates its own network `TwoLayerMLPExact.Net` (`TwoLayerMLPExact.eval`),
which no theorem relates to `mlpEval`. The real constructions are in `TorchLean/NN` and
`TorchLean/NN/UniversalApproximation`.

- `TorchLean/Verify/UniversalApproximation/Rounded.lean`: rounded hinge interpolation in `NF`.
- `TorchLean/Verify/UniversalApproximation/Binary.lean`: hinge sums with IEEE finite intermediates.
- `TorchLean/Verify/UniversalApproximation/BinaryMLP/Error.lean`: a three-term error decomposition.
- `TorchLean/Verify/UniversalApproximation/BinaryMLP.lean`: real readings of configured binary
  tensors and linear layers.
- `FloatInterval/Semantics.lean`: interval semantics of `Binary32` ReLU networks and its soundness.
- `FloatInterval/ConstantTarget.lean`: the base case, a constant target `[c, c]`.

Read as reals, `hingeFun` at configured values is `hingeFun` in the rounded-real model of
`UniversalApproximation.Rounded` when `format.isIEEE = true`, the intermediates are finite
(`HingeSumFinite`, which `HingeSumBounds` establishes from bounds on the exact terms) and the
output is finite. The `NF` carrier has no upper exponent bound. In `relu_Icc` and
`relu_Icc_binary_of_hinge` the rounding term `hingeFunErrorBound` depends on the input and the
parameters, and no theorem bounds it uniformly. The two-layer decomposition separates real
approximation, parameter quantization and execution.

Import the module of the theorem you use, for example the two-layer MLP decomposition:

```lean
import TorchLean.Verify.UniversalApproximation.BinaryMLP.Error
```

## References

- Cybenko, "Approximation by superpositions of a sigmoidal function," 1989.
- Leshno, Lin, Pinkus, and Schocken, "Multilayer feedforward networks with a nonpolynomial
  activation function can approximate any function," 1993.
- Pinkus, *Approximation Theory of the MLP Model in Neural Networks*, 1999.
- Yarotsky, "Error bounds for approximations with deep ReLU networks," 2017.
