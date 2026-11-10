# Check that polynomial pieces meet at their endpoints

Run the bundled example from the repository root:

```bash
scripts/lake.sh exe verify -- spline-cert
```

No Julia installation or trained model is needed. `piecewise_linear_cert.json` describes three
line segments through `(0,0)`, `(1,1)`, `(2,0)`, `(3,1)`. A piece stores coefficients in the local
coordinate `t = x - lo`: on `[1,2]`, coefficients `[1,-1]` mean `1 - t`.

The checker verifies increasing knots, matching piece intervals, and exact agreement at both ends
of every piece. Rational numbers are JSON strings such as `"1/2"`, so these checks do not use a
floating-point tolerance. A changed endpoint or coefficient that breaks an equality is rejected.

A pass shows that the represented pieces and endpoints are consistent. It does not bound the
interior of a higher-degree polynomial or show that the spline approximates a neural network.

This directory also holds the code. `PiecewisePolyCert.lean` is the checker, with the optional
binary32 replay of `--arithmetic=ieee`, and `PiecewisePolyCLI.lean` is the command.
`fit_piecewise_linear.jl` prints the same three pieces using Julia Base alone, and
`scripts/lake.sh exe verify -- spline-cert --regen` runs it through `Julia.lean` and verifies
the JSON it prints. `Julia.lean` calls `julia` from `PATH`, or the executable named by
`TORCHLEAN_JULIA` when that variable is set.
