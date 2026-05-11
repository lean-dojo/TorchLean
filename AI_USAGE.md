# AI Usage Disclosure

TorchLean has been developed and formalized primarily by hand over an extended
period of work. The core definitions, architecture, theorem statements, proof
decisions, runtime boundaries, examples, and release choices were made and
reviewed by the maintainers. We used AI assistance only as a limited support
tool for some of the harder proof engineering and debugging work, not as an
oracle and not as a replacement for manual formalization.

## Tools We Used

- OpenAI GPT-5.2 Pro was used selectively as an interactive assistant for a few
  difficult proof and engineering tasks. In particular, it helped with proof
  planning, Lean search, refactoring long proof scripts, debugging stubborn Lean
  goals, and explaining possible ways to organize large correctness arguments.
- GPT-5.2 Pro was also useful while working around CUDA and native runtime
  boundaries: reading error logs, thinking through FFI and memory ownership
  issues, checking documentation language, and helping us separate what Lean
  proves from what the CUDA/cuBLAS runtime must be trusted to do.
- Harmonic was used in a narrower exploratory role for a small amount of
  definition design and mathematical organization, especially before committing
  some concepts to Lean.

## Where AI Helped

AI assistance was concentrated in places where the work was unusually long,
technical, or easy to get lost in:

- Long Lean proofs and proof searches, including parts of the autograd proof
  layer, runtime approximation proofs, CROWN style verification explanations,
  and compiled IR execution correctness work.
- Proof engineering around large goals: finding useful intermediate lemmas,
  suggesting tactic structure, proposing ways to split cases, and helping make
  proof scripts more maintainable.
- CUDA and backend debugging: interpreting build/runtime failures, thinking
  through FFI boundaries, documenting GPU assumptions, and clarifying which
  statements are Lean theorems versus trusted native behavior.
- Documentation cleanup for the guide, trust boundaries, examples, and release
  notes.

## How We Checked The Work

Most of the repository was written and formalized manually. When we used AI
help, we treated it the same way we would treat a colleague's sketch on a
whiteboard: useful for ideas, search, debugging, and wording, but not itself a
source of truth.

For the release:

- Theorem statements and proofs count because Lean accepts the corresponding
  source files.
- Runtime claims count only when the relevant executable checker, test, or
  script accepts them.
- CUDA, Python, Julia, external solvers, native libraries, and finite precision
  behavior remain behind the explicit trust boundaries documented in
  `TRUST_BOUNDARIES.md`.
- Any code, proof idea, or prose suggested by an AI tool was manually reviewed, edited, and
  either checked by the repository tooling or left visible as an assumption,
  limitation, or external trust boundary.
