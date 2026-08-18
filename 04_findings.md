# Mechanistic Isolation and Causal Verification of Induction Circuits in GPT-2 Small

## Abstract & Research Questions

The standard mechanistic interpretability narrative—originally formalized by Olsson et al. (2022)—posits that transformer language models perform in-context pattern replication of the form `... A B ... A -> B` via specialized "induction heads." While observational metrics like attention pattern matching strongly suggest the existence of these heads, observational correlation alone cannot establish whether a given head's output is causally responsible for the downstream logit shifts or merely downstream of a larger, unobserved computation.

This paper rigorously investigates the induction mechanism within GPT-2 small across three core inquiries:

1. Can candidate induction heads be localized purely through their observational attention patterns on synthetic repeated sequences?
2. Does causal activation patching confirm that these isolated heads are functionally necessary and sufficient to drive the next-token prediction shift, or is the behavior distributed across a broader circuit?
3. How do these heads compose with upstream previous-token heads, and does their functional role remain robust when subjected to structural stress-tests, cross-distribution inputs, and negative controls?

---

## 1. Correlational Localization of Candidate Heads

To isolate pure in-context copying behavior from memorized n-gram statistics or semantic associations present in the pretraining distribution, we constructed sequences composed of uniform random tokens repeated exactly once—formatted as `[BOS, A, A]`. Because the token sequence $A$ consists of independently sampled vocabulary items, predicting $B$ upon the second occurrence of $A$ strictly requires the model to perform in-context retrieval rather than relying on prior parametric knowledge.

We evaluated every attention head across all 12 layers of GPT-2 small by computing an "induction score"—defined as the average attention weight allocated from a token in the second copy back to the token immediately succeeding its original appearance in the first copy.

| Layer & Head | Induction Score | Functional Designation |
| --- | --- | --- |
| **L5H5** | 0.926 | Candidate Induction Head |
| **L6H9** | 0.921 | Candidate Induction Head |
| **L7H10** | 0.914 | Candidate Induction Head |
| **L5H1** | 0.903 | Candidate Induction Head |
| **L7H2** | 0.822 | Candidate Induction Head |
| **L4H11** | 0.000 | Negative Control Head |

The empirical spatial distribution of these scores is remarkably sparse—sharply localized within layers 5 through 7 rather than being diffusely distributed across the network. Furthermore, these top-scoring heads align precisely with those identified in the foundational literature, providing an independent external validation that our observational pipeline captures genuine structural phenomena rather than setup-specific artifacts.

---

## 2. Causal Disambiguation via Activation Patching

Observing an attention pattern that looks like induction does not guarantee that the head's output vector ($z$-activations) is actually utilized by downstream layers to modify the output logits. To establish true causal necessity and sufficiency, we deployed activation patching (`hook_z`) between a "clean" run (`[BOS, A, A]`) and a "corrupted" run (`[BOS, A, B]`), where the second half of the sequence is replaced with unrelated random tokens, destroying the in-context prefix match.

By overwriting specific head outputs in the corrupted run with activations extracted from the clean run, we measured the exact percentage of the clean-versus-corrupted logit gap recovered for the correct next token.

| Intervened Activation (`hook_z`) | Logit Recovery (%) | Causal Interpretation |
| --- | --- | --- |
| **L5H5 alone** | 5.8% | Low individual sufficiency |
| **L6H9 alone** | 7.9% | Low individual sufficiency |
| **L7H10 alone** | 5.5% | Low individual sufficiency |
| **L4H11 alone (control)** | 0.1% | Complete causal neutrality |
| **L5H5 + L6H9 + L7H10** | 31.8% | Moderate collective sufficiency |
| **Top 5 Heads Combined** (L5H5, L6H9, L7H10, L5H1, L7H2) | 80.3% | Dominant distributed circuit |
| **Full Residual Stream (Post-Layer 7)** | 100.0% | Methodological sanity ceiling |

These causal interventions reveal a stark divergence from the naive observational narrative. While observational scores might lead one to believe that a single top head like L5H5 acts as the central engine of induction, individual head patching recovers under 8% of the logit gap—proving that no individual head is causally sufficient on its own.

Instead, the mechanism operates as a highly distributed circuit spread across layers 5, 6, and 7. The recovery jump from 31.8% (top 3 heads) to 80.3% (top 5 heads) illustrates that these heads act in concert—their individual vector outputs summing constructively within the residual stream to push the representation toward the target logit. The near-zero recovery of the control head (L4H11, 0.1%) confirms that activation swapping does not non-specifically disturb or boost the network, while the full residual stream control (100.0%) validates the mathematical integrity of the patching framework itself.

---

## 3. Stress-Testing Mechanistic Hypotheses & Catching Experimental Flaws

Unlike standard exploratory tutorials that take theoretical models on faith, a robust interpretability framework requires attempting to actively falsify the inferred mechanism and auditing the experimental design for hidden confounders.

### Content-Matching vs. Positional Heuristics

The textbook definition of an induction head requires that it performs *content-based exact matching*—attending to a token because its identity matches a previous token, rather than merely attending to a fixed spatial offset. To test this, we corrupted a single token in the second copy of the sequence, expecting the induction score to collapse locally at that position if the head were truly matching token content.

Unexpectedly, the attention score remained near-ceiling (~0.93–0.95) across the corrupted position. However, closer inspection of the experimental script (`05_robustness_check.py`) revealed a fundamental design flaw in our intervention: the evaluation batch used a static sequence length (`seq_len=50`) across all inputs. Consequently, a degenerate head relying entirely on absolute or relative positional offsets (attending exactly 50 tokens backward) would yield results identical to a true content-matching induction head. Because the sequence period was held constant, the test failed to decouple positional distance from semantic content—underlining how easily improper experimental constraints can mimic successful mechanistic validations.

### Cross-Distribution Stability & Circuit Equifinality

We evaluated whether GPT-2 small exhibits "equifinality"—deploying alternative, parallel circuits depending on the input domain—or whether it relies on a singular, static induction circuit. We tested the top-5 head subset across three distinct data distributions: synthetically generated random tokens, natural language paragraphs, and arbitrary digit sequences.

The identical set of five heads (L5H5, L6H9, L7H10, L5H1, L7H2) emerged as the top performers across all three domains, ruling out structural equifinality. However, the absolute magnitude of the induction scores varied dramatically—peaking at 0.94 on random tokens, dropping to 0.86 on natural language, and declining to 0.57 on numeric digits.

This gradient highlights a key structural reality of transformer dynamics: in synthetic random sequences where prior token transitions carry zero mutual information, the model is forced to route nearly 100% of its predictive logit weight through the dedicated in-context induction circuit. Conversely, natural text and structured numerical sequences offer rich parallel predictive pathways—such as syntactic constraints, n-gram collocations, and positional trends—allowing downstream MLPs and unigram heads to share the predictive burden and reducing the relative strength demanded from the induction heads.

---

## 4. Upstream Composition and K-Composition Verification

Theoretical circuit models state that induction heads do not work in isolation; they depend on upstream "previous-token heads" that write the identity of the preceding token $T_{i-1}$ into the residual stream, which the induction head subsequently reads via Key-composition ($K$-composition).

```
[Previous Token T_{i-1}] ──> (Previous-Token Head: e.g., L4H11) ──[Writes Key]──> 
                                                                                \
                                                                                 ──> (Induction Head: e.g., L5H5) ──> [Logit Output]
                                                                                /
[Current Token T_i]     ───────────────────────────────────────[Reads Query]───>
```

### Path Patching Flaws in Query Space

Our initial attempt to isolate this composition via path patching (`07_path_patching_prev_token_head.py`) attempted to patch the output of the empirical previous-token head (L4H11, previous-token score 0.985) into the corrupted sequence run. The intervention yielded a 0.1% logit recovery—a complete null result.

Diagnostic analysis revealed that this experiment was structurally flawed by construction. In the corrupted sequence (`[BOS, A, B]`), the token identity at the receiver's query position is inherently altered. Because an induction head's query vector is derived directly from the current sequence token, patching upstream key-side inputs ($K$-composition) cannot restore performance if the receiver's query vector ($Q$-composition) is simultaneously corrupted. Repairing the key path while leaving the query broken fundamentally invalidates the intervention.

### Targeted Ablation of Upstream Composition

To isolate the key-composition dependency without corrupting the query token representation, we restricted the intervention to the clean run and executed targeted mean-ablations directly on the upstream heads.

| Condition | L5H5 Induction Score | Score Delta ($\Delta$) | Mechanistic Implication |
| --- | --- | --- | --- |
| **Clean Baseline** | 0.929 | — | Unperturbed circuit operation |
| **L4H11 Mean-Ablated** (Prev-Token Head) | 0.689 | -0.240 | Direct loss of $K$-compositional input |
| **L0H5 Mean-Ablated** (Control Head) | 0.932 | +0.003 | Baseline noise / no functional impact |

Ablating the primary previous-token head (L4H11) causes a substantial 0.240 drop in the downstream induction head's attention score, whereas ablating an uninformative control head (L0H5) leaves performance entirely intact. This provides direct, causal confirmation of $K$-composition: the induction head specifically relies on vector streams generated by L4H11 to attend to historical sequence positions.

The finding that L5H5 retains an attention score of 0.689 even after L4H11 is completely ablated indicates that the upstream layer exhibits functional redundancy—suggesting the presence of secondary previous-token heads that continue to supply spatial context to the residual stream.

---

## 5. Feature-Level Decomposition via Sparse Autoencoders

Sections 2–4 establish causal necessity at the level of whole attention heads. But a head's `hook_z` output is a 64-dimensional vector that could itself be polysemantic — encoding several unrelated computations in superposition (Bricken et al., 2023). `09_sae_feature_analysis.py` asks whether the induction-driving signal inside L5H5, L6H9, L7H10, and L7H2 is concentrated in a small number of monosemantic directions, using sparse autoencoders trained directly on GPT-2 small's `hook_z` activations (Kissane et al., 2024), obtained via SAELens (Bloom, 2024).

| Head | Full Head Recovery (§2) | Full SAE Reconstruction Recovery (ceiling) | Best Single Head-Restricted Feature | Top-10 Head-Restricted Features |
| --- | --- | --- | --- | --- |
| **L5H5** | 5.8% | 4.3% | 0.0% | 0.1% |
| **L6H9** | 7.9% | 5.0% | 0.2% | 0.8% |
| **L7H10** | 5.5% | 3.5% | 0.1% | 0.3% |
| **L7H2** | 8.3% | 4.2% | 0.1% | 1.1% |

The "full SAE reconstruction" row swaps in the SAE's decode-of-encode of the clean head (using its entire ~49k-feature dictionary, no error term) — this establishes what fraction of each head's causal effect *could* be represented by this SAE's dictionary at all, independent of which specific features are selected. The result is a ceiling of roughly 50–80% of the raw head's recovery — the SAE genuinely captures most of the relevant direction. Yet the single most head-restricted, most-active feature (selected as: decoder direction >50% attributable to this head by norm, then top mean-activation at the induction-destination positions) recovers almost none of it, and even the top 10 such features together recover under 1.5%.

This decomposes into two distinct claims worth keeping separate: (1) the induction signal is representable by this SAE's dictionary, but (2) it is *not* concentrated in a handful of monosemantic latents — it is smeared thinly across far more than 10 of the ~49k features. This could reflect genuine feature-splitting of the induction computation, or could reflect that this SAE (trained on natural-language pretraining data) lacks a clean dedicated atom for exact-copy induction on synthetic random-token repeats — an out-of-distribution input for the SAE's training distribution, consistent with the elevated reconstruction error observed (54–69% of activation norm, well above the ~10–20% typical on in-distribution text).

A secondary, expected negative result: the single loudest-firing feature at the induction positions *layer-wide* (ignoring head identity) is only 20–38% decoder-attributable to the head we actually care about — activation magnitude is not a causal-importance proxy, the same lesson as §2's head-level finding, now one level down at the feature level.

---

## 6. Automated, Hypothesis-Free Circuit Discovery via Gradient Attribution

Every experiment above started from a hand-picked candidate list (L5H5, L6H9, L7H10, L5H1, L7H2), chosen because 02's observational induction scores flagged them. `10_attribution_patching_circuit_discovery.py` tests whether an automated method can find (most of) the same circuit *without* that hand-picked starting point, using attribution patching (Nanda, 2023) — the node-level linear approximation underlying Edge Attribution Patching (Syed, Rager & Conmy, 2023):

```
predicted_effect(head) ≈ grad(metric, corrupted_z[head]) · (clean_z[head] − corrupted_z[head])
```

This needs exactly one corrupted forward+backward pass (plus one clean forward pass for activation values) to rank all 144 heads simultaneously, versus one forward pass per head for the exhaustive version of what §2 did by hand.

**Result:** the top-10 heads by one-shot attribution score contain 4 of the 5 hand-picked circuit heads (L5H1, L6H9, L7H2, L7H10). The fifth, **L5H5 — the single highest-scoring head observationally in §1 — ranks 13th of 144**, just outside the automated top-10, despite being a confirmed causal contributor in §2. This is consistent with attribution patching's documented weak spot: it is a first-order Taylor approximation, and heads whose effect runs through a near-saturated (close to binary) softmax attention pattern have a small local gradient even when their true causal effect (measured by actual patching) is large. Validating the top-10 predicted heads against real activation patching gives a Spearman rank correlation of 0.44 between predicted attribution magnitude and actual measured recovery — informative, but far from perfect, confirming the approximation is directionally useful and not a substitute for ground-truth patching.

The automated ranking also surfaced heads the manual investigation never tested: L9H9, L9H6, L10H0, L10H1 (all positive attribution, all recovering 3.5–6.1% on real patching — smaller contributors to the same circuit that hand-picking from observational scores alone would have missed), and L10H7 and L11H10 (both *negative* attribution, confirmed by real patching to *actively suppress* the correct-token logit rather than help it — consistent with the "copy suppression head" phenomenon documented elsewhere in GPT-2 small, though this script does not independently verify that mechanism). The negative control (L4H11, lowest induction score from §1) correctly scores near-zero attribution and near-zero real recovery, same as it did under manual patching in §2.

---

## 7. Methodological Limitations & Future Directions

While these experiments provide strong causal boundary conditions for induction circuits in GPT-2 small, several methodological limitations remain:

* **Fixed Sequence Length Confounding**: As demonstrated in our robustness checks, testing content-matching mechanisms requires variable sequence lengths and dynamic period offsets to rigorously separate positional distance heuristics from true token identity matching.
* **Single-Seed Sampling**: Quantitative recovery metrics were evaluated across single random seed generations; future iterations should aggregate over multi-seed sampling to ensure stability against token embedding variance.
* **Coarse Activation Interventions**: Activation swapping was restricted to post-attention head projections (`hook_z`); fine-grained path patching directly between $W_V$ and $W_K$ projection matrices is required to fully isolate the internal tensor pathways.
* **Unlocalized Residual Logit Variance**: Although the top 5 heads account for 80.3% of the clean logit recovery, the remaining ~19.7% gap to the full residual baseline (100.0%) was not fully decomposed, likely residing in downstream MLP layers or minor multi-layer attention paths that process the combined induction signal prior to the final layer norm.
* **SAE Trained Off-Distribution**: §5's sparse autoencoders were trained on natural-language pretraining data, not on synthetic random-token repeats; the elevated reconstruction error (54–69% of activation norm) at the induction positions may partly reflect this distribution mismatch rather than an intrinsic property of the induction computation. Repeating §5 with an SAE trained on (or fine-tuned to) this synthetic distribution would disentangle the two.
* **Node-Level, Not Edge-Level, Attribution**: §6's automated discovery ranks individual `hook_z` sites, not sender→receiver edges — it does not by itself recover the K-composition dependency between L4H11 and L5H5 that §4 established by hand. A full Edge Attribution Patching graph over all sender/receiver pairs is the natural extension, and would let automated discovery reconstruct the two-layer circuit structure of §4, not just the flat top-level head ranking of §6.
* **Linear Attribution Miscalibration**: §6's gradient-based ranking correlates only moderately (Spearman ρ=0.44) with real patching recovery among its own top candidates, and it underranked L5H5 specifically (rank 13 of 144) — a documented failure mode when a head's true effect passes through near-saturated attention. Automated discovery should be treated as a candidate-generation filter to prioritize which heads to patch for real, not a replacement for the patching itself. A full edge-level attribution graph (Conmy et al., 2023) is the natural next step toward closing this gap.

---

## References

1. Olsson, C., Elhage, N., Nanda, N., et al. (2022). "In-context Learning and Induction Heads." *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
2. Elhage, N., Nanda, N., Olsson, C., et al. (2021). "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread*. https://transformer-circuits.pub/2021/framework/index.html
3. Wang, K., Variengien, A., Conmy, A., Shlegeris, B., Steinhardt, J. (2022). "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small." *arXiv:2211.00593*. https://arxiv.org/abs/2211.00593
4. Variengien, A. (2023). "Some Common Confusion About Induction Heads." *LessWrong*. https://www.lesswrong.com/posts/nJqftacoQGKurJ6fv/some-common-confusion-about-induction-heads
5. Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features/index.html
6. Kissane, C., Krzyzanowski, R., et al. (2024). "Interpreting Attention Layer Outputs with Sparse Autoencoders." *arXiv:2406.17759*. https://arxiv.org/abs/2406.17759
7. Bloom, J. (2024). "Open Source Sparse Autoencoders for All Residual Stream Layers of GPT2-Small." *SAELens / LessWrong*. https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD
8. Nanda, N. (2023). "Attribution Patching: Activation Patching At Industrial Scale." https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
9. Syed, A., Rager, C., Conmy, A. (2023). "Attribution Patching Outperforms Automated Circuit Discovery." *arXiv:2310.10348*. https://arxiv.org/abs/2310.10348
10. Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., Garriga-Alonso, A. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability." *arXiv:2304.14997*. https://arxiv.org/abs/2304.14997
