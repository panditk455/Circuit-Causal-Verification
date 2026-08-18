"""
Observational localization of induction heads in GPT-2 small.

Induction heads are a proposed mechanism for in-context pattern completion:
given a sequence of the form [A][B] ... [A], the model attends from the
second occurrence of A back to the token that followed its first
occurrence (B), and copies "predict B next" forward (Olsson et al., 2022).
Olsson et al. identify these heads behaviorally via an "induction score" on
synthetic repeated random-token sequences -- attention paid from a token to
the token following its previous occurrence -- and report a small number of
heads in GPT-2-scale models with near-ceiling scores.

This script reproduces that measurement independently on gpt2-small:
  1. Construct sequences of the form [BOS, A, A] where A is a sequence of
     independently sampled token ids, so the second occurrence of A cannot
     be predicted from memorized bigram statistics -- prediction requires
     genuine in-context retrieval.
  2. For every attention head, measure the induction score: mean attention
     weight from each position in the second copy to the position that
     followed the matching token in the first copy.
  3. Rank heads by score and render a layer x head heatmap.

The result is reported in Table 1 of 04_findings.md. This measurement is
purely observational: a high induction score establishes that a head's
attention pattern is consistent with the induction mechanism, not that the
head's output is causally responsible for the model's prediction. The
causal claim is established separately in 03_activation_patching.py.

Reference:
  Olsson, C., Elhage, N., Nanda, N., et al. (2022). "In-context Learning
  and Induction Heads." Transformer Circuits Thread.
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
"""

import torch as t
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

t.manual_seed(0)
device = "cpu"

print("Loading gpt2-small...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
print(f"gpt2-small: {n_layers} layers x {n_heads} heads")

induction_score_store = t.zeros((n_layers, n_heads))


def induction_score_hook(pattern, hook):
    """pattern: [batch, head_index, dest_pos, src_pos] attention probabilities
    for one layer. Store, per head, the average attention paid from each
    position in the second copy back to the token following its first
    occurrence."""
    seq_len = (pattern.shape[-1] - 1) // 2
    offset = seq_len - 1
    dest_positions = t.arange(seq_len + 1, 2 * seq_len + 1)
    src_positions = dest_positions - offset
    scores = pattern[:, :, dest_positions, src_positions]  # [batch, head, n_positions]
    avg_scores = scores.mean(dim=(0, 2))
    induction_score_store[hook.layer(), :] = avg_scores


def generate_repeated_tokens(model, seq_len, batch=10):
    bos = t.full((batch, 1), model.tokenizer.bos_token_id, dtype=t.int64)
    rand_tokens = t.randint(1, model.cfg.d_vocab, (batch, seq_len))
    return t.cat([bos, rand_tokens, rand_tokens], dim=1)


seq_len = 50
batch = 10
tokens = generate_repeated_tokens(model, seq_len, batch=batch)

pattern_hook_names = [f"blocks.{l}.attn.hook_pattern" for l in range(n_layers)]
with t.no_grad():
    model.run_with_hooks(
        tokens,
        return_type=None,
        fwd_hooks=[(name, induction_score_hook) for name in pattern_hook_names],
    )

print("\nInduction score per head (rows=layer, cols=head):")
print(induction_score_store.round(decimals=2))

flat = induction_score_store.flatten()
top_k = 5
top_vals, top_idx = flat.topk(top_k)
print(f"\nTop {top_k} candidate induction heads:")
top_heads = []
for val, idx in zip(top_vals, top_idx):
    layer = idx.item() // n_heads
    head = idx.item() % n_heads
    top_heads.append((layer, head, val.item()))
    print(f"  L{layer}H{head}: score={val.item():.3f}")

# Sanity check / control: also report the worst-scoring head, we'll use it
# as a control in the patching experiment.
worst_val, worst_idx = flat.topk(1, largest=False)
worst_layer, worst_head = worst_idx.item() // n_heads, worst_idx.item() % n_heads
print(f"\nWorst-scoring head (control for patching): L{worst_layer}H{worst_head}: score={worst_val.item():.3f}")

with open("induction_heads_result.txt", "w") as f:
    f.write("Top candidate induction heads (layer, head, induction_score):\n")
    for layer, head, val in top_heads:
        f.write(f"L{layer}H{head}\t{val:.4f}\n")
    f.write(f"\nControl (lowest-scoring) head:\nL{worst_layer}H{worst_head}\t{worst_val.item():.4f}\n")

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(induction_score_store.detach().numpy(), cmap="viridis", aspect="auto")
ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title("Induction score per head, gpt2-small\n(attention paid to token after prior occurrence, on random repeated sequences)")
fig.colorbar(im, ax=ax, label="induction score")
fig.tight_layout()
fig.savefig("induction_score_heatmap.png", dpi=150)
print("\nSaved induction_score_heatmap.png and induction_heads_result.txt")
