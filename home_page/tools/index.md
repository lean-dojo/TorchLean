---
title: Tools
---

# Tools

<div class="tool-directory">
  <article class="tool-card">
    <p class="tool-kind">Profiling</p>
    <h2>LeanProfiler</h2>
    <p>
      LeanProfiler records nested <code>IO</code> spans, structured runtime metadata, Lean
      heartbeats, and process resource counters. It writes a Trace Event file for Perfetto and a
      JSON summary that can be compared with an earlier run.
    </p>
    <p>
      The TorchLean example wraps an existing model runner. Device completion hooks can include
      synchronized work in a span, and the report labels that measurement as
      completion latency rather than device-kernel time.
    </p>
    <nav class="tool-links" aria-label="LeanProfiler links">
      <a href="https://github.com/lean-dojo/LeanProfiler">Repository</a>
      <a href="https://github.com/lean-dojo/LeanProfiler#add-it-to-a-project">Getting started</a>
      <a href="https://github.com/lean-dojo/LeanProfiler/tree/main/guide">Guide source</a>
    </nav>
  </article>

  <article class="tool-card">
    <p class="tool-kind">Checked case studies</p>
    <h2>TorchLean Verified Examples</h2>
    <p>
      We keep the larger case studies in TorchLean Verified Examples. They range
      from batch-invariant inference and replayable checkpoints to GPT training and a formal Kimi
      K3 architecture specification. Each project identifies its Lean statements and remaining
      runtime assumptions.
    </p>
    <nav class="tool-links" aria-label="TorchLean Verified Examples links">
      <a href="https://github.com/Robertboy18/TorchLean-Verified-Examples">
        Repository and overview
      </a>
      <a
        href="https://github.com/Robertboy18/TorchLean-Verified-Examples/tree/master/week-01-batch-invariant-inference">
        Batch-invariant inference
      </a>
      <a
        href="https://github.com/Robertboy18/TorchLean-Verified-Examples/tree/master/week-02-verifiable-transformer-checkpoint">
        Verifiable transformer checkpoint
      </a>
      <a
        href="https://github.com/Robertboy18/TorchLean-Verified-Examples/tree/master/week-03-gpt-training">
        GPT training in Lean
      </a>
      <a
        href="https://github.com/Robertboy18/TorchLean-Verified-Examples/tree/master/week-04-kimi-k3-specification">
        Kimi K3 specification
      </a>
    </nav>
  </article>
</div>
