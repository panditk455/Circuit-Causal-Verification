"""
Cross-distribution stability test of the induction circuit identified in
02_find_induction_heads.py.

Olsson et al. (2022) characterize induction heads via their behavior on
synthetic repeated random-token sequences, but do not by themselves
establish whether the same head set is responsible for pattern completion
across qualitatively different input distributions, or whether GPT-2 small
exhibits circuit-level multiplicity -- recruiting a different, partially
redundant set of heads depending on what kind of content is being repeated.
This script tests that directly: if induction is a single mechanism, the
top-scoring heads by induction score should be approximately invariant to
input distribution; if distributions recruit different heads, the "one
circuit" characterization is incomplete.

Three distributions are evaluated with the identical measurement method
used in 02_find_induction_heads.py:
  1. random  -- uniformly random token ids (the distribution used to
                originally localize the circuit)
  2. natural -- a short original English passage, chunked and repeated
  3. digits  -- sequences of single-digit numbers, repeated

Results are reported in Table 3 (§3.2) of 04_findings.md.

Reference:
  Olsson, C., Elhage, N., Nanda, N., et al. (2022). "In-context Learning
  and Induction Heads." Transformer Circuits Thread.
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
"""

import torch as t
from transformer_lens import HookedTransformer

t.manual_seed(3)
device = "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads


def induction_score_for_tokens(tokens, seq_len):
    """tokens: [batch, 1 + 2*seq_len] with a BOS token prepended."""
    store = t.zeros((n_layers, n_heads))
    offset = seq_len - 1
    dest_positions = t.arange(seq_len + 1, 2 * seq_len + 1)
    src_positions = dest_positions - offset

    def score_hook(pattern, hook):
        scores = pattern[:, :, dest_positions, src_positions]
        store[hook.layer(), :] = scores.mean(dim=(0, 2))

    names = [f"blocks.{l}.attn.hook_pattern" for l in range(n_layers)]
    with t.no_grad():
        model.run_with_hooks(tokens, return_type=None, fwd_hooks=[(n, score_hook) for n in names])
    return store


def top_k_heads(store, k=5):
    flat = store.flatten()
    vals, idxs = flat.topk(k)
    return [(idx.item() // n_heads, idx.item() % n_heads, val.item()) for val, idx in zip(vals, idxs)]


bos_id = model.tokenizer.bos_token_id
batch = 10
seq_len = 30
d_vocab = model.cfg.d_vocab

# --- Distribution 1: random tokens ---
bos = t.full((batch, 1), bos_id, dtype=t.int64)
A_random = t.randint(1, d_vocab, (batch, seq_len))
tokens_random = t.cat([bos, A_random, A_random], dim=1)

# --- Distribution 2: natural language, self-written paragraph, chunked ---
paragraph = (
    "The workshop was quiet in the early morning before the first apprentice arrived. "
    "Tools hung on the wall in careful rows, each one worn smooth from years of steady use. "
    "A kettle sat warming on the small stove near the window, its lid rattling gently as "
    "steam began to rise. Outside, the street was still empty, waiting for the town to wake."
)
full_tokens = model.to_tokens(paragraph, prepend_bos=False)[0]
n_chunks = full_tokens.shape[0] // seq_len
chunks = [full_tokens[i * seq_len:(i + 1) * seq_len] for i in range(n_chunks)]
A_natural = t.stack([chunks[i % len(chunks)] for i in range(batch)])
tokens_natural = t.cat([bos, A_natural, A_natural], dim=1)

# --- Distribution 3: digit sequences ---
digit_token_ids = [model.to_tokens(f" {d}", prepend_bos=False)[0, 0].item() for d in range(10)]
digit_token_ids = t.tensor(digit_token_ids)
A_digits = digit_token_ids[t.randint(0, 10, (batch, seq_len))]
tokens_digits = t.cat([bos, A_digits, A_digits], dim=1)

distributions = {
    "random": tokens_random,
    "natural": tokens_natural,
    "digits": tokens_digits,
}

all_top5 = {}
print("Top-5 induction heads by input distribution:\n")
for name, tokens in distributions.items():
    store = induction_score_for_tokens(tokens, seq_len)
    top5 = top_k_heads(store, k=5)
    all_top5[name] = set((l, h) for l, h, v in top5)
    print(f"{name}:")
    for l, h, v in top5:
        print(f"  L{l}H{h}: {v:.3f}")
    print()

random_set = all_top5["random"]
print("Overlap with 'random' distribution's top-5 set:")
lines = []
for name in ["natural", "digits"]:
    overlap = random_set & all_top5[name]
    line = f"  {name}: {len(overlap)}/5 heads shared -> {sorted(overlap)}"
    print(line)
    lines.append(line)

with open("cross_distribution_result.txt", "w") as f:
    for name, heads in all_top5.items():
        f.write(f"{name}: {sorted(heads)}\n")
    f.write("\nOverlap with random:\n")
    for line in lines:
        f.write(line + "\n")

print("\nSaved cross_distribution_result.txt")
