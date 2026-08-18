"""
Feature-level decomposition of the induction circuit via sparse
autoencoders.

Sections 02/03/07/08 establish that L5H5, L6H9, L7H10, L5H1, and L7H2 form
a distributed induction circuit at the level of whole attention heads. A
head's hook_z output is nonetheless a 64-dimensional vector that may itself
be polysemantic, encoding multiple unrelated computations in superposition
(Elhage et al., 2021; Bricken et al., 2023). This script tests whether the
induction-driving signal within a head's output is concentrated in a small
number of monosemantic directions, or is distributed across many, using
sparse autoencoders (SAEs) trained directly on gpt2-small's attention-head
outputs -- Kissane et al.'s (2024) hook_z SAE release, obtained via SAELens
(Bloom, 2024) -- one SAE per layer, each mapping the concatenated 12-head x
64-dim hook_z (768 dims) to a ~49k-dimensional sparse feature space and
back.

Method, using the identical clean/corrupted setup and recovery metric as
03_activation_patching.py so results are directly comparable:
  1. Encode clean and corrupted hook_z activations through each layer's SAE.
  2. Identify which features are most active at the induction-destination
     positions.
  3. Decompose each candidate feature's decoder direction into per-head
     norms, to test whether the most active feature is in fact attributable
     to the head already identified as causally relevant (e.g. head 5 at
     layer 5).
  4. Feature-level patching: rather than patching a head's full hook_z
     output, patch only one (or a small set of) SAE feature coefficient(s)
     from the clean run into the corrupted run, reconstruct, and preserve
     the corrupted run's own reconstruction error term -- an error-
     preserving patch that isolates the causal contribution of the
     targeted feature(s) from the SAE's baseline reconstruction loss,
     following standard practice in SAE-based circuit analysis.
  5. Compare single- and multi-feature recovery against whole-head recovery
     (5.5-8.3%, from 03_activation_patching.py) and against a ceiling
     condition -- the SAE's full reconstruction of the head (all ~49k
     features, no error term) -- which separates "the wrong features were
     selected" from "this SAE's dictionary cannot represent the relevant
     direction on this input."
  6. Negative control: patch a feature with near-zero activation at the
     induction positions; expected recovery is approximately zero.

Note on interpretive limits: SAE reconstruction is lossy, dead and
low-frequency features are a known property of trained SAEs, and decoder-
direction norms support only a hypothesis about head attribution, not a
proof of feature semantics. Reconstruction error magnitude is reported
alongside all recovery figures for this reason.

References:
  Elhage, N., Nanda, N., Olsson, C., et al. (2021). "A Mathematical
  Framework for Transformer Circuits." Transformer Circuits Thread.
  https://transformer-circuits.pub/2021/framework/index.html

  Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing
  Language Models With Dictionary Learning." Transformer Circuits Thread.
  https://transformer-circuits.pub/2023/monosemantic-features/index.html

  Kissane, C., Krzyzanowski, R., et al. (2024). "Interpreting Attention
  Layer Outputs with Sparse Autoencoders." arXiv:2406.17759.
  https://arxiv.org/abs/2406.17759

  Bloom, J. (2024). "Open Source Sparse Autoencoders for All Residual
  Stream Layers of GPT2-Small." SAELens / LessWrong.
  https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD
"""

import functools

import torch as t
from sae_lens import SAE
from transformer_lens import HookedTransformer, utils

t.manual_seed(9)
device = "cpu"

model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()

n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
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


with t.no_grad():
    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits = model(corrupted_tokens)

clean_val = get_correct_logit(clean_logits, correct_token_ids, dest_positions).mean().item()
corrupt_val = get_correct_logit(corrupted_logits, correct_token_ids, dest_positions).mean().item()
gap = clean_val - corrupt_val

print(f"Clean logit on correct token (upper bound):     {clean_val:.3f}")
print(f"Corrupted logit on correct token (lower bound): {corrupt_val:.3f}")
print(f"Gap to recover: {gap:.3f}\n")

# Known induction heads from 02/03, grouped by the layer whose hook_z SAE
# we need to load. (layer, head, prior single-head recovery % from 03)
KNOWN_HEADS = {
    5: [(5, "5.8%")],
    6: [(9, "7.9%")],
    7: [(10, "5.5%"), (2, None)],
}


def feature_patch_hook(z, hook, sae, feature_indices, clean_z_full):
    """Patch only `feature_indices` of the SAE code from clean into the
    corrupted run's hook_z, keep the corrupted run's own reconstruction
    error for everything else, then decode back to hook_z-space."""
    corrupted_feats = sae.encode(z)
    clean_feats = sae.encode(clean_z_full)
    recon_corrupted = sae.decode(corrupted_feats)
    error_corrupted = z - recon_corrupted

    patched_feats = corrupted_feats.clone()
    patched_feats[..., feature_indices] = clean_feats[..., feature_indices]
    recon_patched = sae.decode(patched_feats)
    return recon_patched + error_corrupted


def full_head_patch_hook(z, hook, head_index, clean_z_full):
    z[:, :, head_index, :] = clean_z_full[:, :, head_index, :]
    return z


def full_reconstruction_patch_hook(z, hook, sae, head_index, clean_z_full):
    """Swap in the SAE's full reconstruction (ALL ~49k features, decoded
    back) of the clean head -- not the true clean activation, and not a
    hand-picked subset of features. This bounds how much of the head's
    causal effect is representable by ANY combination of this SAE's
    dictionary at all, independent of which features we chose to patch."""
    clean_recon_full = sae.decode(sae.encode(clean_z_full))
    z[:, :, head_index, :] = clean_recon_full[:, :, head_index, :]
    return z


report_lines = []

for layer, head_entries in KNOWN_HEADS.items():
    hook_name = utils.get_act_name("z", layer)
    print(f"{'=' * 60}\nLayer {layer} hook_z SAE\n{'=' * 60}")
    report_lines.append(f"\n=== Layer {layer} ===")

    sae, _, _ = SAE.from_pretrained("gpt2-small-hook-z-kk", f"blocks.{layer}.hook_z", device=device)
    sae.eval()

    clean_z = clean_cache[hook_name]

    with t.no_grad():
        clean_feats = sae.encode(clean_z)
        recon_clean = sae.decode(clean_feats)
        recon_error = (clean_z - recon_clean).norm(dim=-1).mean().item()
        clean_norm = clean_z.norm(dim=-1).mean().item()
    print(f"SAE reconstruction error / activation norm at layer {layer}: {recon_error:.3f} / {clean_norm:.3f}"
          f" ({recon_error / clean_norm:.1%})")

    # Which features fire most at the induction-destination positions,
    # LAYER-WIDE (no restriction to a specific head)?
    dest_feats = clean_feats[:, dest_positions, :]  # [batch, n_pos, d_sae]
    mean_activation = dest_feats.mean(dim=(0, 1))  # [d_sae]
    top_val, top_idx = mean_activation.topk(3)
    alive_indices = (mean_activation > 1e-4).nonzero().squeeze(-1)
    alive_values = mean_activation[alive_indices]
    control_val, control_local_idx = alive_values.min(dim=0)
    control_idx = alive_indices[control_local_idx].item()

    print(f"Top 3 firing features at induction positions (layer-wide, ignoring head identity): "
          f"{[(i.item(), f'{v.item():.3f}') for v, i in zip(top_val, top_idx)]}")
    print(f"Near-silent control feature: {control_idx} (mean activation {control_val.item():.5f})")

    # Control comparison: test whether the loudest layer-wide feature is
    # attributable to the known induction head. Empirically it typically is
    # not -- the loudest feature at a position is often a generic
    # positional or token-identity feature shared across many heads, rather
    # than the one performing the induction copy. This mirrors, at the
    # feature level, the head-level finding in section 2 of 04_findings.md:
    # activation magnitude is not a causal-importance proxy.
    naive_top_feature = top_idx[0].item()
    naive_decoder_dir = sae.W_dec[naive_top_feature].reshape(n_heads, d_head)
    naive_head_norms = naive_decoder_dir.norm(dim=-1)
    naive_head_share = naive_head_norms / naive_head_norms.sum()
    naive_best_head = naive_head_share.argmax().item()
    print(f"Loudest feature {naive_top_feature}'s decoder direction is {naive_head_share[naive_best_head]:.1%} "
          f"attributable to head {naive_best_head} (heads we actually care about: "
          f"{[h for h, _ in head_entries]}) -- loud is not the same as causally relevant.")

    report_lines.append(f"reconstruction_error/norm = {recon_error:.4f}/{clean_norm:.4f} "
                         f"({recon_error / clean_norm:.4f})")
    report_lines.append(f"naive_loudest_feature={naive_top_feature} best_head_by_decoder_norm={naive_best_head} "
                         f"share={naive_head_share[naive_best_head].item():.4f} "
                         f"(check against target heads {[h for h, _ in head_entries]})")

    # Per-feature decoder-norm decomposition, once for the whole SAE, so we
    # can restrict the search to features that actually live inside a
    # SPECIFIC known head rather than whichever feature is loudest overall.
    all_decoder_dirs = sae.W_dec.reshape(-1, n_heads, d_head)  # [d_sae, n_heads, d_head]
    all_head_norms = all_decoder_dirs.norm(dim=-1)  # [d_sae, n_heads]
    all_head_share = all_head_norms / all_head_norms.sum(dim=-1, keepdim=True)  # [d_sae, n_heads]

    for head, prior_recovery in head_entries:
        head_share_for_head = all_head_share[:, head]  # [d_sae]
        for threshold in (0.5, 0.3, 0.1, 0.0):
            candidate_mask = head_share_for_head > threshold
            n_candidates = candidate_mask.sum().item()
            if n_candidates >= 10:
                break
        candidate_indices = candidate_mask.nonzero().squeeze(-1)
        candidate_activations = mean_activation[candidate_indices]
        ranked_local = candidate_activations.topk(min(10, n_candidates)).indices
        ranked_candidates = candidate_indices[ranked_local]
        top_feature = ranked_candidates[0].item()
        top10_head_restricted = ranked_candidates.tolist()
        print(f"\n  Head L{layer}H{head}: {n_candidates}/{sae.cfg.d_sae} features are >{threshold:.0%} "
              f"decoder-attributable to this head. Most-active such feature: {top_feature} "
              f"(share={head_share_for_head[top_feature].item():.1%}, "
              f"activation={mean_activation[top_feature].item():.3f})")
        # (a) full raw-head patch, for direct comparison
        full_hooks = [(hook_name, functools.partial(full_head_patch_hook, head_index=head, clean_z_full=clean_z))]
        with t.no_grad():
            full_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=full_hooks)
        full_val = get_correct_logit(full_logits, correct_token_ids, dest_positions).mean().item()
        full_recovery = (full_val - corrupt_val) / gap

        # (b) single top feature patched
        single_hooks = [(hook_name, functools.partial(
            feature_patch_hook, sae=sae, feature_indices=[top_feature], clean_z_full=clean_z))]
        with t.no_grad():
            single_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=single_hooks)
        single_val = get_correct_logit(single_logits, correct_token_ids, dest_positions).mean().item()
        single_recovery = (single_val - corrupt_val) / gap

        # (c) top-10 head-restricted features patched together
        top10_hooks = [(hook_name, functools.partial(
            feature_patch_hook, sae=sae, feature_indices=top10_head_restricted, clean_z_full=clean_z))]
        with t.no_grad():
            top10_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=top10_hooks)
        top10_val = get_correct_logit(top10_logits, correct_token_ids, dest_positions).mean().item()
        top10_recovery = (top10_val - corrupt_val) / gap

        # (d) negative control: patch the layer-wide near-silent feature
        control_hooks = [(hook_name, functools.partial(
            feature_patch_hook, sae=sae, feature_indices=[control_idx], clean_z_full=clean_z))]
        with t.no_grad():
            control_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=control_hooks)
        control_val = get_correct_logit(control_logits, correct_token_ids, dest_positions).mean().item()
        control_recovery = (control_val - corrupt_val) / gap

        # (e) ceiling check: swap in the SAE's FULL reconstruction (all
        # ~49k features, no error term) of the clean head. This tells us
        # whether low single/top-10 recovery is "wrong features picked" or
        # "this SAE's dictionary can't represent the relevant direction on
        # this out-of-distribution synthetic input at all."
        full_recon_hooks = [(hook_name, functools.partial(
            full_reconstruction_patch_hook, sae=sae, head_index=head, clean_z_full=clean_z))]
        with t.no_grad():
            full_recon_logits = model.run_with_hooks(corrupted_tokens, fwd_hooks=full_recon_hooks)
        full_recon_val = get_correct_logit(full_recon_logits, correct_token_ids, dest_positions).mean().item()
        full_recon_recovery = (full_recon_val - corrupt_val) / gap

        print(f"  (prior full-head recovery from 03: {prior_recovery})")
        print(f"    full head patched (true activation, this script):  recovery={full_recovery:.1%}")
        print(f"    full SAE reconstruction of head patched (ceiling): recovery={full_recon_recovery:.1%}")
        print(f"    single head-restricted top feature {top_feature} patched:  recovery={single_recovery:.1%}")
        print(f"    top-10 head-restricted features patched:       recovery={top10_recovery:.1%}")
        print(f"    near-silent (layer-wide) control feature patched: recovery={control_recovery:.1%}")

        report_lines.append(f"L{layer}H{head}: n_candidates={n_candidates} full_head_recovery={full_recovery:.4f} "
                             f"full_sae_reconstruction_recovery={full_recon_recovery:.4f} "
                             f"single_feature_recovery={single_recovery:.4f} "
                             f"top10_features_recovery={top10_recovery:.4f} "
                             f"control_feature_recovery={control_recovery:.4f}")

with open("sae_feature_analysis_result.txt", "w") as f:
    f.write(f"clean={clean_val:.4f} corrupted={corrupt_val:.4f} gap={gap:.4f}\n")
    f.write("\n".join(report_lines) + "\n")

print("\nSaved sae_feature_analysis_result.txt")
