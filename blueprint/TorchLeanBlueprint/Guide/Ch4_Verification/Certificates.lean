import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Verification Certificates" =>
%%%
tag := "certificates"
%%%

A certificate is a finite artifact that lets Lean check a larger verification claim without relying
on the entire search procedure that produced it.

The producer may be α,β-CROWN, a branch-and-bound verifier, a scientific solver, or a Python
script. The checker asks three questions: what finite object was returned, what local conditions
does Lean validate, and what theorem turns acceptance into a semantic claim?

TorchLean does not vendor the Two-Stage / α,β-CROWN repository. The core Lean build does not
require that Python stack, but generating fresh α,β-CROWN leaf JSON artifacts requires a separate
external producer checkout.

During a branch-and-bound verification run, the Two-Stage tooling can emit a small JSON *leaf
certificate* for each terminal subdomain. TorchLean can parse that JSON and validate several
properties entirely inside Lean: every leaf box lies inside the declared root input region; every
leaf marked as verified satisfies the solver's local prune test
(`∃ i, lb[i] > threshold[i]` in the exported fields); and the document is internally consistent
(dimensions, array lengths, and cross-references line up).

These checks focus on the JSON artifact itself: boxes nest correctly, verified leaves satisfy the
stated prune rule, and the fields fit together. The numeric bound propagation that produced each
`lb` belongs to the external producer unless a separate recompute-and-compare certificate path is
added.

The path is:

- α,β-CROWN performs branch and bound outside Lean;
- the producer exports terminal leaf domains in `abcrown_leaf_cert_v0_1.json`;
- TorchLean parses the JSON;
- Lean checks the structural predicate for each leaf;
- the checker accepts or rejects the artifact.

A few terms will help keep the certificate layers straight:

- `certificate`: an artifact produced outside Lean that Lean can parse and check.
- `leaf certificate`: a per-subdomain proof obligation in a branch-and-bound run.
- `verified` / `pruned`: here, a leaf passes the solver's local check and can be removed
  from the search tree.
- `producer hypothesis`: the part supplied by the external solver when Lean checks only the exported
  artifact rather than recomputing the bound.

The checked predicate is small enough to write informally: a leaf is accepted when its box lies
inside the root box, its dimensions are coherent, and every leaf marked verified has a witness
index whose lower bound is above the corresponding threshold.

The leaf prune test has the form:

$$`\exists i,\qquad lb_i>threshold_i.`

Leaf nesting has the form:

$$`B_\ell\subseteq B_{\mathrm{root}}.`

A stronger branch certificate would also check coverage:

$$`B_{\mathrm{root}}\subseteq\bigcup_\ell B_\ell.`

For a semantic margin property, the target shape is:

$$`\forall x\in B_\ell,\qquad c^\top f(x)\ge threshold.`

The full certificate is accepted when every leaf satisfies this predicate and the top level metadata
is coherent. That is the structural certificate account in this fragment.

# What The Checker Proves Today

This `v0.1` format stays small. Lean checks that every leaf domain sits inside the declared
root input box, that every leaf marked as verified really satisfies the solver's own prune test
(`∃ i, lb[i] > threshold[i]`), and that the JSON is internally consistent enough for downstream
tooling to rely on it.

The current artifact checks `lb_i > threshold_i` for an exported lower bound. A stronger artifact
would also check that `lb_i` is a sound lower bound for the graph on the leaf.

If we later add a recompute path, the stronger predicate would recompute the bound for the network
on the leaf box, compare that recomputed bound with the exported `lb`, and then check the same
margin witness.

That would move more work from the external solver into Lean's checker. The current JSON format was
kept deliberately small so the first boundary is clear.

There are therefore three levels of confidence to keep straight:

- structural checking: the artifact is self-consistent.
- recompute-and-compare checking: Lean can reproduce the arithmetic.
- full soundness: Lean also checks the bound computation that produced the artifact.

Those levels are not a weakness of certificates; they are the reason certificate formats are useful.
They let the project increase the amount checked by Lean without changing the surrounding workflow.

# File Format: `abcrown_leaf_cert_v0_1.json`

Top-level object:

```
{
  "format": "abcrown_leaf_cert_v0_1",
  "input_dim": 2,
  "root": { "lo": [-4.8, -10.8], "hi": [4.8, 10.8] },
  "leaves": [
    {
      "lo": [...],
      "hi": [...],
      "lb": [...],
      "threshold": [...],
      "witness_idx": 0,
      "witness_margin": 0.123
    }
  ]
}
```

Semantics:

- `root` describes the input box being verified.
- In the current exporter, `root` is derived from the componentwise min/max over exported leaves so
  the box remains correct even if the first processed subdomain is not the full root.
- Each `leaf` is a sub-box of `root`.
- `lb` and `threshold` are the spec lower bounds and thresholds produced by α,β-CROWN for that
  leaf at the moment it was pruned or verified.
- A leaf is considered "verified" iff `∃ i, lb[i] > threshold[i]`.
  (This matches how `complete_verifier/input_split/branching_domains.py` filters out verified domains.)
- `witness_idx` and `witness_margin` are a convenience witness for the check above:
  `witness_margin = lb[witness_idx] - threshold[witness_idx]`.

# How To Generate

Set `ABCROWN_CERT_OUT` to a path before running α,β-CROWN. The Two-Stage `run_verify.sh` script
can be locally instrumented to export the JSON artifact once verification finishes. If you want to
use that producer, clone it separately:

```
git clone https://github.com/Verified-Intelligence/Two-Stage_Neural_Controller_Training.git \
  Two-Stage_Neural_Controller_Training
```

Run the external verifier with certificate export enabled, save the JSON artifact, then pass that
file to TorchLean's checker.

# How To Check In Lean

Use the unified `verify` CLI tool `abcrown-leaf` to check a JSON certificate against TorchLean's
semantics.

Example:

```
lake exe verify -- abcrown-leaf
```

The checker also runs on the bundled sample certificate when invoked with the tool's default input.

End-to-end Python plus Lean post-check: *Two-Stage Workflows*. What TorchLean proves vs imports:
*Verification*.

# References

- [α,β-CROWN project](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [β-CROWN / α,β-CROWN paper](https://arxiv.org/abs/2103.06624), covering branch and bound with
  optimized bound propagation.
- [LiRPA on general computational graphs](https://arxiv.org/abs/2002.12920), for automatic
  perturbation analysis.
- [Branch-and-bound for neural network verification](https://arxiv.org/abs/1907.10615), as one
  representative background entry.
