# IBP certificate for a small MLP

`ibp-cert` replays a stored interval bound for a 3-4-2 ReLU network. Lean recomputes interval bound
propagation (IBP) on a fixed input box and accepts when the certificate's output intervals contain
its own.

- `MLP.lean`: the network `model`, with hidden weights `1 + i + j` and biases `i + 1`, output
  weights `2 + i + j` and biases `i`, and the `ibp-cert` entry point.
- `ExampleInputs.lean`: `naturalCenter`, the vector `[1, 2, ..., dim]`, and `naturalBox`, the
  `L∞` box of a given radius around it. The certificate uses `naturalBox 3 1.0`.
- `mlp_cert.json`: the certificate, for graph `mlp_graph_workflow_v1` and `node_id` 3 (the output in
  the producer's numbering, node 7 of Lean's graph), with output intervals of about [230, 554] and
  [291, 699].
- `export_cert.py`: the producer of `mlp_cert.json`.

```bash
scripts/lake.sh exe verify -- ibp-cert
scripts/lake.sh exe verify -- ibp-cert path/to/cert.json
python3 TorchLeanExamples/Verification/IBPCert/export_cert.py
```

Without a path, `ibp-cert` reads `mlp_cert.json`. On success it prints
`IBP certificate accepted: serialized bounds contain Lean's host-Float recomputation.` We run the
exporter from
the repository root, since it writes into `TorchLeanExamples/Verification/IBPCert` unless
`--out-dir PATH` names another directory. It needs Python 3.10 or later and imports
`scripts/verification/interval.py`.

## What acceptance means

From the file, the checker reads `result.lo` and `result.hi` and nothing else. Lean takes the input
box from `IBPCert.naturalBox`, and `graph`, `node_id` and `input_box` are producer metadata.
Each decimal endpoint is rounded once into binary64, the lower one down and the upper one up, and
containment is tested on the rounded values. An endpoint one binary64 ulp or more inside Lean's is
rejected, so a tighter bound from another verifier fails and a looser one passes.

Our exporter reproduces Lean's own host-`Float` IBP pass operation for operation, so the
certificate matches Lean's recomputation bit for bit and carries no independent evidence. A bound
computed by auto_LiRPA would generally not pass. What the example exercises is the artifact parser
and the replay predicate in host `Float` arithmetic, and no theorem covers it.
