# Robustness of a small digit classifier

Start with a few bundled 8×8 digit images:

```bash
scripts/lake.sh exe verify -- digits --eps=0.02 --max=10
```

The command loads `digits_linear_weights.json` (a 64-input, 10-output linear classifier) and
`digits_test.json`. It reports ordinary classification accuracy and the count whose correct-class
lower bound exceeds every competing class's upper bound over the specified input box. A low
certified count is a result, not a program failure. No retraining or dataset download is needed.

Nominal predictions use host binary64 `Float`. The default bound pass uses native binary32
`Float32`; `--arithmetic ieee` selects the bit-level binary32 reference. Neither bound pass is
by itself a proof about the deployed native classifier.
Read `NN/Verification/Robustness/Digits.lean` for the arithmetic, clipping, and propagation choices.

There is a separate report-format example:

```bash
scripts/lake.sh exe verify -- margin-report
```

It checks `digits_linear_margin_cert.json`: the supplied logit intervals, margin decisions, and
summary counts must agree. It does not load the network or recompute those intervals. Use it to
check an exporter; use `digits` to exercise propagation on the actual bundled weights.

`digits-train-certify` invokes the Python producer to train and export new artifacts. It is an
optional regeneration workflow and requires the producer's Python dependencies.
