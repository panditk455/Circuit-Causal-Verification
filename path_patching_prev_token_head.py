"""
Verification of upstream K-composition into the induction circuit.

The mathematical framework of Elhage et al. (2021) formalizes how heads in
different layers can compose through the residual stream: a downstream
head's query-key (QK) circuit can read a quantity an upstream head wrote in
an earlier layer, a dependency termed K-composition. Under this framework,
an induction head's QK circuit requires a previous-token head's output to
already be written into the residual stream, so that the induction head can
determine "what token came immediately before each position" when matching
the current token against earlier occurrences. Sections 02/03/05/06
establish the induction heads' own behavior and causal role in isolation;
this script tests the predicted upstream dependency.

Method, following the path patching technique of Wang et al. (2022):
  1. Identify the previous-token head: the head with highest mean attention
     from position i to position i-1 on ordinary (non-repeated) context.
  2. Path-patch its output into a known induction head. Using the corrupted
     sequence (no repetition, so the induction head has no valid target),
     overwrite only the previous-token head's output (hook_z) with its
     value from the clean run, and measure whether the induction head's
     attention pattern shifts toward its clean-run behavior.

Limitation: this patches the sender head's full output, which can in
principle influence the receiver's query and value inputs in addition to
its key input. A strictly isolated path patch would freeze the receiver's
query and value at their corrupted values and allow only the key input to
vary, requiring a three-run freezing procedure not implemented here. The
present intervention is a node-level patch with a downstream-attention-
pattern readout, not an isolated K-only path patch; the distinction is
material and is treated as an open limitation rather than elided.

References:
  Elhage, N., Nanda, N., Olsson, C., et al. (2021). "A Mathematical
  Framework for Transformer Circuits." Transformer Circuits Thread.
  https://transformer-circuits.pub/2021/framework/index.html

  Wang, K., Variengien, A., Conmy, A., Shlegeris, B., Steinhardt, J. (2022).
  "Interpretability in the Wild: a Circuit for Indirect Object
  Identification in GPT-2 small." arXiv:2211.00593.
  https://arxiv.org/abs/2211.00593
"""

import functools

import torch as t
from transformer_lens import HookedTransformer, utils

t.manual_seed(4)
device = "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
batch = 10
seq_len = 50
d_vocab = model.cfg.d_vocab

bos = t.full((batch, 1), model.tokenizer.bos_token_id, dtype=t.int64)
A = t.randint(1, d_vocab, (batch, seq_len))
B = t.randint(1, d_vocab, (batch, seq_len))
clean_tokens = t.cat([bos, A, A], dim=1)
corrupted_tokens = t.cat([bos, A, B], dim=1)

# ---------- Step 1: find the previous-token head ----------
prev_token_score = t.zeros((n_layers, n_heads))


def prev_token_hook(pattern, hook):
    seq = pattern.shape[-1]
    dest = t.arange(1, seq)
    src = dest - 1
    scores = pattern[:, :, dest, src]
    prev_token_score[hook.layer(), :] = scores.mean(dim=(0, 2))


names = [f"blocks.{l}.attn.hook_pattern" for l in range(n_layers)]
with t.no_grad():
    model.run_with_hooks(clean_tokens, return_type=None, fwd_hooks=[(n, prev_token_hook) for n in names])

flat = prev_token_score.flatten()
best_val, best_idx = flat.topk(1)
pt_layer, pt_head = best_idx.item() // n_heads, best_idx.item() % n_heads
worst_val, worst_idx = flat.topk(1, largest=False)
worst_layer, worst_head = worst_idx.item() // n_heads, worst_idx.item() % n_heads

print(f"Top previous-token head: L{pt_layer}H{pt_head}  score={best_val.item():.3f}")
print(f"Control (lowest-scoring) head: L{worst_layer}H{worst_head}  score={worst_val.item():.3f}\n")

# ---------- Step 2: path-patch into the induction head ----------
receiver_layer, receiver_head = 5, 5  # top induction head from 02_find_induction_heads.py

offset = seq_len - 1
dest_positions = t.arange(seq_len + 1, 2 * seq_len + 1)
src_positions = dest_positions - offset


def run_and_get_receiver_score(tokens, extra_hooks=None):
    store = {}

    def capture_hook(pattern, hook):
        scores = pattern[:, receiver_head, dest_positions, src_positions]
        store["score"] = scores.mean().item()

    hooks = list(extra_hooks) if extra_hooks else []
    hooks.append((f"blocks.{receiver_layer}.attn.hook_pattern", capture_hook))
    with t.no_grad():
        model.run_with_hooks(tokens, return_type=None, fwd_hooks=hooks)
    return store["score"]


with t.no_grad():
    _, clean_cache = model.run_with_cache(clean_tokens)

clean_score = run_and_get_receiver_score(clean_tokens)
corrupted_score = run_and_get_receiver_score(corrupted_tokens)


def make_patch_hook(layer, head, cache):
    def patch(z, hook):
        z[:, :, head, :] = cache[hook.name][:, :, head, :]
        return z
    return patch


patch_name = utils.get_act_name("z", pt_layer)
sender_patched_score = run_and_get_receiver_score(
    corrupted_tokens, extra_hooks=[(patch_name, make_patch_hook(pt_layer, pt_head, clean_cache))]
)

control_patch_name = utils.get_act_name("z", worst_layer)
control_patched_score = run_and_get_receiver_score(
    corrupted_tokens, extra_hooks=[(control_patch_name, make_patch_hook(worst_layer, worst_head, clean_cache))]
)

print(f"Receiver head L{receiver_layer}H{receiver_head}'s induction-attention score:")
print(f"  clean run (upper bound):                              {clean_score:.3f}")
print(f"  corrupted run (lower bound):                          {corrupted_score:.3f}")
print(f"  corrupted + patched candidate sender L{pt_layer}H{pt_head}:            {sender_patched_score:.3f}")
print(f"  corrupted + patched control head L{worst_layer}H{worst_head} (lowest prev-tok score): {control_patched_score:.3f}")

gap = clean_score - corrupted_score
sender_recovery = (sender_patched_score - corrupted_score) / gap
control_recovery = (control_patched_score - corrupted_score) / gap
print(f"\nRecovery from patching candidate sender: {sender_recovery:.1%}")
print(f"Recovery from patching control head:     {control_recovery:.1%}")

with open("path_patching_result.txt", "w") as f:
    f.write(f"Top previous-token head: L{pt_layer}H{pt_head} score={best_val.item():.4f}\n")
    f.write(f"Control head: L{worst_layer}H{worst_head} score={worst_val.item():.4f}\n\n")
    f.write(f"Receiver L{receiver_layer}H{receiver_head} induction-attention score:\n")
    f.write(f"clean={clean_score:.4f} corrupted={corrupted_score:.4f}\n")
    f.write(f"sender_patched={sender_patched_score:.4f} recovery={sender_recovery:.4f}\n")
    f.write(f"control_patched={control_patched_score:.4f} recovery={control_recovery:.4f}\n")

print("\nSaved path_patching_result.txt")
