# ODE Verification Artifacts

A corridor gives lower and upper functions around a proposed ODE solution. The checker compares
their values at the initial time, their ordering, and their derivative bounds against the ODE's
right-hand side on time intervals. This folder contains both a passing fixture and candidates that
illustrate why those inequalities can remain unproved.

The reusable verifier lives under `NN/Verification/ODE`. This folder contains small JSON artifacts
for examples and regression checks:

| Input | What it represents | Observed checker outcome |
| --- | --- | --- |
| `sample_ode_cert.json` | The constant zero solution of `u′ = 0`, using a one-layer MLP. | Accepts with its pinned IEEE arithmetic. Start here. |
| `sin_cert.json` | The same zero function through a sine-activation network. | Accepts with `--arithmetic ieee`; native outward rounding cannot close the zero-slack initial bound. |
| `logistic_trivial_cert.json` | The coarse corridor `0 ≤ u ≤ 1` for `u′ = u(1-u)`, starting at `0.1`. | Accepts with `--model=direct --arithmetic ieee`; the native graph path cannot close its zero-slack differential inequality. |
| `logistic_learned_cert.json` | Learned lower/upper candidates for the logistic equation. | Rejects: the propagated initial lower-function interval does not establish the required inequality. |

`zero_mlp.json`, `one_mlp.json`, and `zero_siren.json` encode the constant functions used by the
first three rows. `logistic_lower_learned.json` and `logistic_upper_learned.json` are the learned
candidate weights; having exported weights does not make their corridor certified.

Check a bundled certificate directly with:

```bash
lake exe verify -- ode --cert=NN/Examples/Verification/ODE/sample_ode_cert.json
```

When a certificate declares `settings.arithmetic`, the verifier uses that setting unless the
command supplies `--arithmetic` explicitly.

The additional passing checks are:

```bash
lake exe verify -- ode --arithmetic ieee \
  --cert=NN/Examples/Verification/ODE/sin_cert.json
lake exe verify -- ode --model=direct --arithmetic ieee \
  --cert=NN/Examples/Verification/ODE/logistic_trivial_cert.json
```

These commands retain zero slack. Switching arithmetic/backend changes the computation used to
check the candidate, so cite that choice when reporting the result. A rejection means these bounds
did not establish the required inequality; it is not by itself a counterexample to the ODE claim.
In particular, the learned candidate is useful for seeing where training and verification diverge.

The time partitioner seeds each input box directly from its endpoints. It avoids reconstructing
those endpoints from a rounded center and radius. Failed interval checks subdivide until both halves
pass, the configured depth or width limit is reached, or no representable interior split remains.
The final diagnostic names that stopping condition. Initial-time failures are point checks, so
subdividing later time intervals cannot repair them.

Recheck the curated passing fixture with:

```bash
lake exe verify -- ode --cert=NN/Examples/Verification/ODE/sample_ode_cert.json
```
