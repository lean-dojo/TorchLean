# Check that polynomial pieces meet at their endpoints

Run the bundled example from the repository root:

```bash
lake exe verify -- spline-cert
```

No Julia installation or trained model is needed. `piecewise_linear_cert.json` describes three
line segments through `(0,0)`, `(1,1)`, `(2,0)`, `(3,1)`. A piece stores coefficients in the local
coordinate `t = x - lo`: on `[1,2]`, coefficients `[1,-1]` mean `1 - t`.

The checker verifies increasing knots, matching piece intervals, and exact agreement at both ends
of every piece. Rational numbers are JSON strings such as `"1/2"`, so these checks do not use a
floating-point tolerance. A changed endpoint or coefficient that breaks an equality is rejected.

This establishes consistency of the represented pieces and endpoints. It does not bound the
interior of a higher-degree polynomial or prove that the spline approximates a neural network.
The implementation and optional binary32 cross-check are documented in
`NN/Verification/Splines/PiecewisePolyCert.lean`; `PiecewiseLinearCLI.lean` supplies the command.
