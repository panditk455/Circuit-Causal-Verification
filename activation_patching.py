"""
Causal verification of the induction heads localized in
02_find_induction_heads.py (L5H5, L6H9, L7H10 were the top candidates by
induction score) via activation patching, following the causal-intervention
methodology used to establish circuit-level claims in GPT-2 small (Wang et
al., 2022).

An attention pattern consistent with induction is correlational evidence:
it does not establish that a head's output is used by the rest of the
network to produce the correct prediction. Activation patching provides
the causal test by constructing a minimal-pair intervention:

  clean_tokens     = [BOS, A, A]   (repeated content: induction is possible)
  corrupted_tokens = [BOS, A, B]   (no repetition: nothing to induct on)

For each induction-destination position, the correct continuation is the
token known (from the clean sequence) to follow the earlier occurrence.
The model's logit on that token is measured under three conditions:
  1. clean run                                        -> upper bound
  2. corrupted run                                     -> lower bound
  3. corrupted run with one head's output (hook_z) overwritten by its value
     from the clean run's cache -> fraction of the gap closed by that head
     alone

  recovery = (patched - corrupted) / (clean - corrupted)

A recovery near 1.0 indicates the patched head's output is causally
sufficient to reproduce the induction behavior; near 0.0 indicates no
causal contribution. Each candidate head is tested individually and in
combination, alongside the lowest-scoring head from 02 as a negative
control -- the control result is what distinguishes a causal claim from an
intervention that "recovers" performance regardless of which head is
patched.

Results are reported in Table 2 of 04_findings.md.

Reference:
  Wang, K., Variengien, A., Conmy, A., Shlegeris, B., Steinhardt, J. (2022).
  "Interpretability in the Wild: a Circuit for Indirect Object
  Identification in GPT-2 small." arXiv:2211.00593.
  https://arxiv.org/abs/2211.00593
"""

import functools

import torch as t
from transformer_lens import HookedTransformer, utils

t.manual_seed(1)
device = "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

n_heads = model.cfg.n_heads
seq_len = 50
batch = 8
d_vocab = model.cfg.d_vocab

bos = t.full((batch, 1), model.tokenizer.bos_token_id, dtype=t.int64)
A = t.randint(1, d_vocab, (batch, seq_len))
B = t.randint(1, d_vocab, (batch, seq_len))

clean_tokens = t.cat([bos, A, A], dim=1)
corrupted_tokens = t.cat([bos, A, B], dim=1)

# Same destination positions as the induction-score measurement: positions in
# the second copy, 0-indexed into the full sequence. Stop one short of the
# final position since it has no "next token" to check against.
dest_positions = t.arange(seq_len + 1, 2 * seq_len)
correct_token_ids = clean_tokens[:, dest_positions + 1]  # [batch, n_positions]


def get_correct_logit(logits, token_ids, positions):
    sel = logits[:, positions, :]  # [batch, n_positions, d_vocab]
    return sel.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # [batch, n_positions]


def patch_head_z(z, hook, head_index, clean_cache):
    z[:, :, head_index, :] = clean_cache[hook.name][:, :, head_index, :]
    return z


with t.no_grad():
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

    clean_val = get_correct_logit(clean_logits, correct_token_ids, dest_positions).mean().item()
    corrupt_val = get_correct_logit(corrupted_logits, correct_token_ids, dest_positions).mean().item()

    print(f"Clean logit on correct token (upper bound):     {clean_val:.3f}")
    print(f"Corrupted logit on correct token (lower bound): {corrupt_val:.3f}")
    print(f"Gap to recover: {clean_val - corrupt_val:.3f}\n")

    candidates = [
        ("L5H5 (top induction head)", 5, 5),
        ("L6H9 (2nd induction head)", 6, 9),
        ("L7H10 (3rd induction head)", 7, 10),
        ("L4H11 (control, lowest induction score)", 4, 11),
    ]

    results = []
    for label, layer, head in candidates:
        hook_name = utils.get_act_name("z", layer)
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, functools.partial(patch_head_z, head_index=head, clean_cache=clean_cache))],
        )
        patched_val = get_correct_logit(patched_logits, correct_token_ids, dest_positions).mean().item()
        recovery = (patched_val - corrupt_val) / (clean_val - corrupt_val)
        results.append((label, layer, head, patched_val, recovery))
        print(f"{label}: patched logit={patched_val:.3f}  recovery={recovery:.1%}")

    # Also patch all three top heads simultaneously to show the combined circuit.
    top_heads = [(5, 5), (6, 9), (7, 10)]
    fwd_hooks = [
        (utils.get_act_name("z", layer), functools.partial(patch_head_z, head_index=head, clean_cache=clean_cache))
        for layer, head in top_heads
    ]
    combined_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=fwd_hooks)
    combined_val = get_correct_logit(combined_logits, correct_token_ids, dest_positions).mean().item()
    combined_recovery = (combined_val - corrupt_val) / (clean_val - corrupt_val)
    print(f"\nAll 3 top heads patched together: patched logit={combined_val:.3f}  recovery={combined_recovery:.1%}")

    # Extend to the next two induction-scoring heads to see if recovery grows.
    top5_heads = [(5, 5), (6, 9), (7, 10), (5, 1), (7, 2)]
    fwd_hooks_5 = [
        (utils.get_act_name("z", layer), functools.partial(patch_head_z, head_index=head, clean_cache=clean_cache))
        for layer, head in top5_heads
    ]
    top5_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=fwd_hooks_5)
    top5_val = get_correct_logit(top5_logits, correct_token_ids, dest_positions).mean().item()
    top5_recovery = (top5_val - corrupt_val) / (clean_val - corrupt_val)
    print(f"Top 5 heads patched together: patched logit={top5_val:.3f}  recovery={top5_recovery:.1%}")

    # Sanity check on the method itself: patch the WHOLE residual stream right
    # after layer 7 (resid_post) instead of individual heads. If this doesn't
    # recover ~100%, something about the setup (not the heads) is off.
    resid_hook_name = utils.get_act_name("resid_post", 7)

    def patch_resid(resid, hook, clean_cache):
        return clean_cache[hook.name]

    resid_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(resid_hook_name, functools.partial(patch_resid, clean_cache=clean_cache))],
    )
    resid_val = get_correct_logit(resid_logits, correct_token_ids, dest_positions).mean().item()
    resid_recovery = (resid_val - corrupt_val) / (clean_val - corrupt_val)
    print(f"[sanity check] Full resid_post patch after layer 7: recovery={resid_recovery:.1%}")

with open("activation_patching_result.txt", "w") as f:
    f.write(f"Clean logit (upper bound): {clean_val:.4f}\n")
    f.write(f"Corrupted logit (lower bound): {corrupt_val:.4f}\n\n")
    for label, layer, head, patched_val, recovery in results:
        f.write(f"{label}: patched={patched_val:.4f} recovery={recovery:.4f}\n")
    f.write(f"\nAll 3 top heads combined: patched={combined_val:.4f} recovery={combined_recovery:.4f}\n")
    f.write(f"Top 5 heads combined: patched={top5_val:.4f} recovery={top5_recovery:.4f}\n")
    f.write(f"[sanity check] full resid_post patch after layer 7: recovery={resid_recovery:.4f}\n")

print("\nSaved activation_patching_result.txt")
