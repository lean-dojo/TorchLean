---
title: Tools
---

# Tools

We've also been building a couple of projects alongside TorchLean. One helps us
understand where our programs spend their time; the other gives us room to explore
larger experiments and formalize the ideas behind recent model papers.

<div class="tool-directory">
  <article class="tool-card">
    <p class="tool-kind">Profiling</p>
    <h2>LeanProfiler</h2>
    <p>
      When a Lean program gets slower, we want to know where the time went. With
      LeanProfiler, we can name the parts we care about, such as loading data,
      running a forward pass or taking a training step, and measure them as the
      program runs. We can open the timeline in Perfetto and compare the JSON
      reports from two runs.
    </p>
    <p>
      We use it around an existing TorchLean model runner, but it works with other
      compiled Lean programs too. For asynchronous GPU work, we can wait for the
      work to finish before closing a measurement. That measures the time until
      completion; individual kernel timings need a device profiler.
    </p>
    <nav class="tool-links" aria-label="LeanProfiler links">
      <a href="https://github.com/lean-dojo/LeanProfiler">Repository</a>
      <a href="https://github.com/lean-dojo/LeanProfiler#add-it-to-a-project">Getting started</a>
      <a href="https://github.com/lean-dojo/LeanProfiler/tree/main/guide">Guide source</a>
    </nav>
  </article>

  <article class="tool-card">
    <p class="tool-kind">Experiments and proofs</p>
    <h2>TorchLean Verified Examples</h2>
    <p>
      This is where we take a machine-learning question and spend more time with
      it. Can changing the surrounding batch change a model's answer? Can we
      export a checkpoint and replay it in Lean? What does it take to train GPT-2
      from Lean? The examples bring together code, experiments and proofs so we
      can work through those questions.
    </p>
    <p>
      We're also working through the Kimi K3 and Nemotron reports, writing their
      mathematical definitions with TorchLean tensors and proving properties of
      those definitions. I'd love to see more model papers explored this way,
      with the equations, Lean code and explanations together : )
    </p>
    <nav class="tool-links" aria-label="TorchLean Verified Examples links">
      <a href="https://robertboy18.github.io/TorchLean-Verified-Examples/">
        Articles and experiments
      </a>
      <a href="https://github.com/Robertboy18/TorchLean-Verified-Examples">
        Source repository
      </a>
    </nav>
  </article>
</div>
