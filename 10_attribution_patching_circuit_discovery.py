"""
Automated, hypothesis-free circuit discovery via gradient-based attribution.

Sections 02/03/07/08 hand-select which heads to test: induction scores from
02_find_induction_heads.py identify L5H5, L6H9, L7H10, L5H1, and L7H2 as
candidates, and 03_activation_patching.py confirms causal relevance by
patching each one individually -- an approach that scales as one forward
pass per head if applied exhaustively across all 144 heads. This script
tests whether an automated method can recover the same circuit without a
hand-picked candidate list, by ranking all 144 heads by predicted causal
importance from a single backward pass, then validating that ranking
against the causal patching results already established.

Method: attribution patching (Nanda, 2023), the node-level linear
approximation underlying Edge Attribution Patching (Syed, Rager & Conmy,
2023). It is a first-order Taylor approximation to activation patching:

    predicted_effect(head) ~= grad(metric, corrupted_z[head]) . (clean_z[head] - corrupted_z[head])

Rather than one forward pass per head (the exhaustive approach used
explicitly in 03), this requires exactly one corrupted forward pass with
gradients enabled and one backward pass to obtain gradients at every
hook_z site simultaneously, plus one clean forward pass for activation
values -- O(1) passes rather than O(n_heads) to rank every head in the
model.

Limitations:
  - This is a linear approximation. Attention computed via softmax is
    nonlinear, so heads whose true effect is dominated by second-order
    (curvature) terms can be misranked -- a documented failure mode of
    attribution patching (Syed, Rager & Conmy, 2023), not specific to this
    implementation. This is the motivation for the validation step below,
    which spends real forward passes confirming the top-K predicted heads
    against ground-truth activation patching rather than treating the
    gradient ranking as sufficient on its own.
  - This is node-level attribution (per hook_z site), not the full edge-
    level attribution graph over every sender-to-receiver pair in the
    computational graph. Full edge attribution would recover, for
    instance, the specific K-composition dependency between an upstream
    and downstream head established by hand in 07/08. Ranking 144 nodes in
    a single pass is the tractable node-level slice of the same method
    (Conmy et al., 2023, ACDC, addresses the analogous full-graph search
    problem at greater computational cost); the complete attribution-graph
    extension is left as future work.

References:
  Nanda, N. (2023). "Attribution Patching: Activation Patching At
  Industrial Scale."
  https://www.neelnanda.io/mechanistic-interpretability/attribution-patching

  Syed, A., Rager, C., Conmy, A. (2023). "Attribution Patching Outperforms
  Automated Circuit Discovery." arXiv:2310.10348.
  https://arxiv.org/abs/2310.10348

  Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., Garriga-Alonso,
  A. (2023). "Towards Automated Circuit Discovery for Mechanistic
  Interpretability." arXiv:2304.14997. https://arxiv.org/abs/2304.14997
"""

import functools

import torch as t
from transformer_lens import HookedTransformer, utils

t.manual_seed(3)
device = "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
seq_len = 50
batch = 8
d_vocab = model.cfg.d_vocab

bos = t.full((batch, 1), model.tokenizer.bos_token_id, dtype=t.int64)
A = t.randint(1, d_vocab, (batch, seq_len))
B = t.randint(1, d_vocab, (batch, seq_len))
clean_tokens = t.cat([bos, A, A], dim=1)
corrupted_tokens = t.cat([bos, A, B], dim=1)

dest_positions = t.arange(seq_len + 1, 2 * seq_len)
correct_token_ids = clean_tokens[:, dest_positions + 1]


def get_correct_logit(logits, token_ids, positions):
    sel = logits[:, positions, :]
    return sel.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


# ---------- Pass 1: clean activations (no grad needed) ----------
with t.no_grad():
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_val = get_correct_logit(clean_logits, correct_token_ids, dest_positions).mean().item()

# ---------- Pass 2: corrupted forward + backward, capturing hook_z and its gradient at every layer ----------
z_store = {}


def capture_hook(z, hook):
    z.retain_grad()
    z_store[hook.layer()] = z
    return z


names = [(utils.get_act_name("z", l), capture_hook) for l in range(n_layers)]
corrupted_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=names)
corrupt_val = get_correct_logit(corrupted_logits, correct_token_ids, dest_positions).mean().item()
gap = clean_val - corrupt_val

metric = get_correct_logit(corrupted_logits, correct_token_ids, dest_positions).sum()
metric.backward()

print(f"Clean logit on correct token (upper bound):     {clean_val:.3f}")
print(f"Corrupted logit on correct token (lower bound): {corrupt_val:.3f}")
print(f"Gap to recover: {gap:.3f}\n")

# ---------- Attribution score per head, all 144 in one shot ----------
attribution = t.zeros((n_layers, n_heads))
for layer in range(n_layers):
    clean_z = clean_cache[utils.get_act_name("z", layer)][:, dest_positions, :, :]
    corrupted_z = z_store[layer][:, dest_positions, :, :]
    grad = z_store[layer].grad[:, dest_positions, :, :]
    diff = clean_z - corrupted_z
    # sum over d_head, mean over batch & position -> one score per head
    attribution[layer, :] = (diff * grad).sum(dim=-1).mean(dim=(0, 1))

flat = attribution.flatten()
top_val, top_idx = flat.abs().topk(10)
predicted_ranking = [(idx.item() // n_heads, idx.item() % n_heads, attribution.flatten()[idx].item())
                      for idx in top_idx]

print("Top 10 heads by ONE-SHOT gradient-based attribution (no per-head forward pass):")
for layer, head, score in predicted_ranking:
    print(f"  L{layer}H{head}: attribution score = {score:+.3f}")

known_circuit = {(5, 5), (6, 9), (7, 10), (5, 1), (7, 2)}
predicted_set = {(l, h) for l, h, _ in predicted_ranking}
overlap = known_circuit & predicted_set
print(f"\nKnown circuit from 02/03 (observational + causal patching): {sorted(known_circuit)}")
print(f"Overlap with top-10 attribution-predicted heads: {sorted(overlap)} ({len(overlap)}/{len(known_circuit)})")

# ---------- Validate: actually patch the top-10 predicted heads and see if the linear approximation was right ----------
print("\nValidating gradient predictions against REAL activation patching (ground truth):")


def patch_head_z(z, hook, head_index, clean_cache):
    z[:, :, head_index, :] = clean_cache[hook.name][:, :, head_index, :]
    return z


validation_rows = []
for layer, head, predicted_score in predicted_ranking:
    hook_name = utils.get_act_name("z", layer)
    with t.no_grad():
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, functools.partial(patch_head_z, head_index=head, clean_cache=clean_cache))],
        )
    patched_val = get_correct_logit(patched_logits, correct_token_ids, dest_positions).mean().item()
    real_recovery = (patched_val - corrupt_val) / gap
    validation_rows.append((layer, head, predicted_score, real_recovery))
    print(f"  L{layer}H{head}: predicted attribution={predicted_score:+.3f}  actual patch recovery={real_recovery:.1%}")

# Rank correlation between the one-shot predicted |score| ordering and the
# actual measured recovery ordering, restricted to these top-10 candidates
# -- checks whether the linear approximation's RANKING is trustworthy even
# where its magnitudes aren't calibrated to recovery %.
pred_order = sorted(range(len(validation_rows)), key=lambda i: -abs(validation_rows[i][2]))
real_order = sorted(range(len(validation_rows)), key=lambda i: -validation_rows[i][3])
pred_ranks = [0] * len(validation_rows)
real_ranks = [0] * len(validation_rows)
for rank, i in enumerate(pred_order):
    pred_ranks[i] = rank
for rank, i in enumerate(real_order):
    real_ranks[i] = rank
n = len(validation_rows)
d2 = sum((pred_ranks[i] - real_ranks[i]) ** 2 for i in range(n))
spearman = 1 - (6 * d2) / (n * (n ** 2 - 1)) if n > 1 else float("nan")
print(f"\nSpearman rank correlation (predicted attribution vs. actual patch recovery, top-10 candidates): {spearman:.3f}")

# Negative control: control head from 02 (lowest induction score), should
# have near-zero attribution AND near-zero real recovery.
control_layer, control_head = 4, 11
control_score = attribution[control_layer, control_head].item()
hook_name = utils.get_act_name("z", control_layer)
with t.no_grad():
    control_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, functools.partial(patch_head_z, head_index=control_head, clean_cache=clean_cache))],
    )
control_val = get_correct_logit(control_logits, correct_token_ids, dest_positions).mean().item()
control_recovery = (control_val - corrupt_val) / gap
print(f"\nNegative control L{control_layer}H{control_head}: predicted attribution={control_score:+.3f}  "
      f"actual patch recovery={control_recovery:.1%}")

# L5H5 was the single highest-scoring head observationally (02) and a
# confirmed causal contributor (03) -- but does the one-shot gradient
# ranking even place it near the top? Check explicitly rather than let it
# quietly fall out of the top-10 print above.
full_rank_order = flat.abs().argsort(descending=True)
l5h5_flat_idx = 5 * n_heads + 5
l5h5_rank = (full_rank_order == l5h5_flat_idx).nonzero().item()
print(f"\nL5H5 (top observational induction head from 02): attribution score={attribution[5, 5].item():+.3f}, "
      f"rank #{l5h5_rank + 1} of 144 by one-shot attribution "
      f"({'in' if l5h5_rank < 10 else 'NOT in'} the top 10 predicted heads).")
print("This is the linear approximation's known weak spot: attribution patching can underrank heads whose "
      "true effect runs through a saturated (near-binary) softmax attention pattern, since the local gradient "
      "there is small even though the actual causal effect (measured by real patching) is large.")

with open("attribution_patching_result.txt", "a") as f:
    f.write(f"\nL5H5: attribution={attribution[5, 5].item():.4f} rank={l5h5_rank + 1}/144 "
            f"in_top10={l5h5_rank < 10}\n")

with open("attribution_patching_result.txt", "w") as f:
    f.write(f"clean={clean_val:.4f} corrupted={corrupt_val:.4f} gap={gap:.4f}\n\n")
    f.write(f"known_circuit={sorted(known_circuit)}\n")
    f.write(f"top10_predicted={[(l, h) for l, h, _ in predicted_ranking]}\n")
    f.write(f"overlap={sorted(overlap)} ({len(overlap)}/{len(known_circuit)})\n\n")
    for layer, head, predicted_score, real_recovery in validation_rows:
        f.write(f"L{layer}H{head}: predicted_attribution={predicted_score:.4f} actual_recovery={real_recovery:.4f}\n")
    f.write(f"\nspearman_rank_correlation={spearman:.4f}\n")
    f.write(f"control L{control_layer}H{control_head}: predicted_attribution={control_score:.4f} "
            f"actual_recovery={control_recovery:.4f}\n")

print("\nSaved attribution_patching_result.txt")
