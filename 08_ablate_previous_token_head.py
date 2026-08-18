"""
Targeted ablation test of upstream K-composition, redesigned after a
confound identified in 07_path_patching_prev_token_head.py.

The cross-run patch in 07 produced a null result (0.1% recovery). Diagnosis:
substituting the corrupted sequence's second half changes the token
identity at the receiver's own query position, and an induction head's
query vector is derived from its current token. No upstream patch can
repair a query built from the wrong token identity -- that confound is
orthogonal to the K-composition dependency the experiment intended to
isolate (Elhage et al., 2021), and the null result reflects the confound
rather than the absence of the dependency.

This script tests the same hypothesis without the query-corruption
confound: query token identity is held correct throughout by operating
entirely within the clean run, with no cross-run activation swap. Instead,
the candidate previous-token head's contribution is removed via targeted
mean-ablation. If the induction head's attention pattern degrades under
this intervention, that demonstrates the K-composition dependency without
the confound present in 07.

Method note: mean ablation (replacing a head's output at each position with
the batch mean at that position) is used in place of zero ablation, to
avoid moving the intervened activation further off-distribution than
necessary.

Reference:
  Elhage, N., Nanda, N., Olsson, C., et al. (2021). "A Mathematical
  Framework for Transformer Circuits." Transformer Circuits Thread.
  https://transformer-circuits.pub/2021/framework/index.html
"""

import torch as t
from transformer_lens import HookedTransformer

t.manual_seed(4)
device = "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

batch = 10
seq_len = 50
d_vocab = model.cfg.d_vocab

bos = t.full((batch, 1), model.tokenizer.bos_token_id, dtype=t.int64)
A = t.randint(1, d_vocab, (batch, seq_len))
clean_tokens = t.cat([bos, A, A], dim=1)

pt_layer, pt_head = 4, 11       # previous-token head, found in 07
control_layer, control_head = 0, 5  # lowest prev-token score, found in 07

receiver_layer, receiver_head = 5, 5  # top induction head, from 02

offset = seq_len - 1
dest_positions = t.arange(seq_len + 1, 2 * seq_len + 1)
src_positions = dest_positions - offset


def get_receiver_score(tokens, extra_hooks=None):
    store = {}

    def capture(pattern, hook):
        scores = pattern[:, receiver_head, dest_positions, src_positions]
        store["score"] = scores.mean().item()

    hooks = list(extra_hooks) if extra_hooks else []
    hooks.append((f"blocks.{receiver_layer}.attn.hook_pattern", capture))
    with t.no_grad():
        model.run_with_hooks(tokens, return_type=None, fwd_hooks=hooks)
    return store["score"]


def make_mean_ablate_hook(head):
    def ablate_hook(z, hook):
        z[:, :, head, :] = z[:, :, head, :].mean(dim=0, keepdim=True)
        return z
    return ablate_hook


baseline_score = get_receiver_score(clean_tokens)
pt_ablated_score = get_receiver_score(
    clean_tokens, extra_hooks=[(f"blocks.{pt_layer}.attn.hook_z", make_mean_ablate_hook(pt_head))]
)
control_ablated_score = get_receiver_score(
    clean_tokens, extra_hooks=[(f"blocks.{control_layer}.attn.hook_z", make_mean_ablate_hook(control_head))]
)

print(f"Receiver head L{receiver_layer}H{receiver_head}'s induction-attention score:")
print(f"  normal clean run (no ablation):                    {baseline_score:.3f}")
print(f"  prev-token head L{pt_layer}H{pt_head} mean-ablated:            {pt_ablated_score:.3f}")
print(f"  control head L{control_layer}H{control_head} mean-ablated:              {control_ablated_score:.3f}")

pt_drop = baseline_score - pt_ablated_score
control_drop = baseline_score - control_ablated_score
print(f"\nDrop from ablating candidate previous-token head: {pt_drop:.3f}")
print(f"Drop from ablating control head:                  {control_drop:.3f}")

with open("prev_token_ablation_result.txt", "w") as f:
    f.write(f"baseline={baseline_score:.4f}\n")
    f.write(f"pt_ablated (L{pt_layer}H{pt_head})={pt_ablated_score:.4f} drop={pt_drop:.4f}\n")
    f.write(f"control_ablated (L{control_layer}H{control_head})={control_ablated_score:.4f} drop={control_drop:.4f}\n")

print("\nSaved prev_token_ablation_result.txt")
