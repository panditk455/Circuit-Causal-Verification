"""
Falsification test of the exact-token-matching claim for induction heads.

Sections 02/03 establish that L5H5, L6H9, and L7H10 attend from the current
token back to whatever followed a matching earlier token, and that this
attention is causally linked to the model's prediction. The mechanistic
definition of an induction head is a stronger claim than "the head looks
backward for repeats": it specifically requires attention to be driven by
exact token identity, not by a coarser positional or distributional
heuristic that happens to produce similar attention patterns on synthetic
repeated sequences.

Variengien (2023) argues that the behavioral definition of induction heads
(high induction score on repeated random tokens) and the mechanistic
definition (exact content-based QK matching) are frequently conflated in
the literature, and that satisfying the former does not by itself establish
the latter. This script runs a direct test of that distinction: take a
standard A-A repeat and corrupt exactly one token in the second copy so it
no longer matches its first-copy counterpart. Under genuine exact-token
matching, attention at that corrupted position should collapse toward
baseline while remaining high at every neighboring, still-matching
position. A failure to collapse cleanly is evidence the head's behavior is
more context-dependent than the exact-matching characterization implies.

Reference:
  Variengien, A. (2023). "Some Common Confusion About Induction Heads."
  LessWrong.
  https://www.lesswrong.com/posts/nJqftacoQGKurJ6fv/some-common-confusion-about-induction-heads
"""

import torch as t
from transformer_lens import HookedTransformer

t.manual_seed(2)
device = "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

top_heads = [(5, 5), (6, 9), (7, 10)]  # from 02_find_induction_heads.py

seq_len = 50
batch = 20
d_vocab = model.cfg.d_vocab

bos = t.full((batch, 1), model.tokenizer.bos_token_id, dtype=t.int64)
A = t.randint(1, d_vocab, (batch, seq_len))
B = A.clone()

# Corrupt exactly one token per sequence in the second copy, at a random
# interior position, to a token guaranteed different from the original.
corrupt_idx = t.randint(10, seq_len - 10, (batch,))
for b in range(batch):
    i = corrupt_idx[b].item()
    new_tok = t.randint(1, d_vocab, (1,)).item()
    while new_tok == A[b, i].item():
        new_tok = t.randint(1, d_vocab, (1,)).item()
    B[b, i] = new_tok

tokens = t.cat([bos, A, B], dim=1)

offset = seq_len - 1
dest_positions = t.arange(seq_len + 1, 2 * seq_len + 1)  # positions in 2nd copy
src_positions = dest_positions - offset

with t.no_grad():
    _, cache = model.run_with_cache(tokens)

print("Testing exact-match specificity of induction heads")
print("(does attention collapse right at the corrupted position, and only there?)\n")

for layer, head in top_heads:
    pattern = cache["pattern", layer][:, head, :, :]  # [batch, dest, src]
    scores = pattern[t.arange(batch).unsqueeze(1), dest_positions.unsqueeze(0), src_positions.unsqueeze(0)]
    # scores: [batch, n_positions] -- one score per (batch, second-copy position)

    corrupted_scores = []
    baseline_scores = []
    for b in range(batch):
        i = corrupt_idx[b].item()
        pos_index = i - 1  # index into dest_positions/scores for this position
        corrupted_scores.append(scores[b, pos_index].item())
        other = t.cat([scores[b, :pos_index], scores[b, pos_index + 1:]])
        baseline_scores.append(other.mean().item())

    corrupted_mean = sum(corrupted_scores) / len(corrupted_scores)
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    print(f"L{layer}H{head}: attention AT corrupted position = {corrupted_mean:.3f}   "
          f"attention at unaffected positions = {baseline_mean:.3f}   "
          f"drop = {baseline_mean - corrupted_mean:.3f}")

with open("robustness_check_result.txt", "w") as f:
    f.write("Exact-match specificity test (see 05_robustness_check.py for method)\n\n")
    for layer, head in top_heads:
        pattern = cache["pattern", layer][:, head, :, :]
        scores = pattern[t.arange(batch).unsqueeze(1), dest_positions.unsqueeze(0), src_positions.unsqueeze(0)]
        corrupted_scores = []
        baseline_scores = []
        for b in range(batch):
            i = corrupt_idx[b].item()
            pos_index = i - 1
            corrupted_scores.append(scores[b, pos_index].item())
            other = t.cat([scores[b, :pos_index], scores[b, pos_index + 1:]])
            baseline_scores.append(other.mean().item())
        corrupted_mean = sum(corrupted_scores) / len(corrupted_scores)
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        f.write(f"L{layer}H{head}: at_corrupted={corrupted_mean:.4f} baseline={baseline_mean:.4f} drop={baseline_mean - corrupted_mean:.4f}\n")

print("\nSaved robustness_check_result.txt")
