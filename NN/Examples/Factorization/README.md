# Check matrix factorizations

Run from the repository root:

```bash
lake exe torchlean factorizations
```

No data files or GPU are needed. Each line should say `OK`; the command fails if a numerical
check disagrees with its expected outcome.

| Example | Input and question | Expected result |
| --- | --- | --- |
| `Cholesky.lean` | Can a symmetric positive-definite matrix be reconstructed as `L Lᵀ`? | Error below `1e-6`. An indefinite matrix produces an invalid factor and is rejected. |
| `QR.lean` | Do square and wide matrices reconstruct as `Q R`, with `Qᵀ Q = I`? | Both errors below `1e-6`. A dependent-column example reconstructs but fails orthonormality. |

The negative controls are intentional: an `OK` rejection means the example detected a case outside
the claimed property. The checks use floating-point arithmetic. They complement the factorization
theorems under `NN/Proofs`; they do not prove the native implementation correct.

`Common.lean` contains the shared error checks, including NaN handling. `Check.lean` joins the
examples into the command above.
