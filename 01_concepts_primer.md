# Concepts primer — read this before the code

This is the minimum you need to *understand* (not just run) the induction-head
project, in the order the ideas build on each other.

## 1. The residual stream

A transformer's residual stream is a running sum, one vector per token position,
that every layer reads from and writes back into:

```
x0 -> +attn_layer0 -> x1 -> +mlp_layer0 -> x2 -> +attn_layer1 -> ...
```

Nothing is overwritten — each component (an attention head, an MLP) reads the
current stream, computes something, and *adds* its output back in. This is why
you can "intervene" on a specific layer's output without breaking everything
downstream: you're editing one summand in a sum.

## 2. What an attention head actually does

Following the formalism of Elhage et al. (2021), each head has two
independent circuits:

- **QK circuit (where to look):** for the current token's query, it scores every
  other position's key, softmaxes, and gets an attention pattern — a probability
  distribution over "which past positions matter to me right now."
- **OV circuit (what to copy):** for each position it attends to, it reads that
  position's value vector and writes a (weighted) copy of it into the current
  position's residual stream.

So a head's behavior = "attend according to QK, then copy-with-weights via OV."
Nothing more exotic than that.

## 3. Previous-token heads and induction heads

**Previous-token head:** a head whose QK circuit just attends position `i` to
position `i-1`. Trivial pattern, easy to spot by eye (a diagonal one-off from
the main diagonal in the attention plot).

**Induction head:** given a sequence like `... A B ... A`, when the model sees
the *second* `A`, an induction head attends back to the token that followed the
*first* `A` — i.e. it attends to `B` — and its OV circuit copies "predict B
next" into the residual stream. This is exactly the mechanism behind "complete
the pattern" / in-context copying, and it's one of the few multi-head circuits
that's been rigorously reverse-engineered (Olsson et al., 2022 — see
References at the end of this document).

The circuit requires **two heads composing across two layers**:
1. A previous-token head in an early layer writes "the token before me" info
   into the residual stream at each position.
2. An induction head in a later layer reads *that* written info via its QK
   circuit (K-composition) to find "where did a token matching my current
   token previously appear, and what came after it," then copies that
   "what came after" via its OV circuit.

This is why induction heads only show up in layer 2+ — they depend on
information a previous-token head put there first.

## 4. Hooks (how TransformerLens lets you look inside)

TransformerLens wraps GPT-2 (and others) so every intermediate tensor —
attention patterns, per-head outputs, MLP activations, residual stream at any
point — has a name (e.g. `blocks.5.attn.hook_pattern`) and you can attach a
Python function to run whenever that tensor is computed. A hook function can:
- **read** the tensor (to measure something, e.g. "how much does head 5 in
  layer 5 attend to the induction position") — used for *finding* heads.
- **overwrite** the tensor before it flows onward (e.g. "replace this head's
  output with the value it had on a different input") — used for *patching*.

## 5. Activation patching — why it's causal evidence, not just correlation

Suppose you notice head L5H5 has a high "induction score" (attends to the
right token a lot on repeated sequences). That's *correlational* — it tells
you the head's attention pattern looks right, but not that the head's output
is what's actually driving the model's correct prediction.

Activation patching gives you the causal test:

1. Run the model on a **clean** input where the induction pattern holds
   (`... A B ... A` and the model correctly predicts `B`).
2. Run the model on a **corrupted** input that breaks the pattern (e.g.
   replace the earlier `B` with a random token `C`, so there's nothing correct
   to induct onto) and record its logits.
3. Run the corrupted input again, but **patch in** — overwrite — one specific
   head's output (at the relevant position) with the value it had during the
   *clean* run, leaving everything else corrupted.
4. Measure how much of the clean-vs-corrupt logit difference gets restored.

If patching in just that one head's output recovers most of the correct
prediction, you've shown that head's output is *sufficient* to cause the
correct behavior — not merely correlated with it. If patching a random
non-induction head recovers ~nothing, that's your control, ruling out "maybe
patching anything helps."

That's the whole logical structure of the project:
**find candidates by correlation (induction score) → confirm with causal
intervention (activation patching) → compare against a control head.**
The control comparison is what makes the claim falsifiable: without it,
recovering performance after any intervention would be indistinguishable
from recovering performance specifically because the *correct* head was
patched.

---

## References

- Olsson, C., Elhage, N., Nanda, N., et al. (2022). "In-context Learning
  and Induction Heads." Transformer Circuits Thread.
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Elhage, N., Nanda, N., Olsson, C., et al. (2021). "A Mathematical
  Framework for Transformer Circuits." Transformer Circuits Thread.
  https://transformer-circuits.pub/2021/framework/index.html
