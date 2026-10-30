# Phase 6 — Abstractive Legal Summarization

**Task.** Generate an **abstractive summary** of a long judgment.

**Data.** 7,030 train documents with reference summaries (2 have an empty summary and are dropped), 100 test documents. Median document ≈3.1k words (≈7.5k tokens), median summary ≈640 words. No dev split is given: held-out train documents are used.

**Metric.** BERTScore F1 (primary)

## Shared pipeline

A judgment and its summary are much longer than any Phase-1 context, so summarization is done **segment by segment**:

1. **Segments.** The document is cut at sentence ends into segments of about `SEGMENT_TOKENS` tokens (1,024; 500 for Arch-4 so that the
   segment fits its 512-token bidirectional prefix).
2. **Targets.** Every reference-summary sentence is assigned to the segment that shares most of its words and word pairs
   (unigram recall + 2 × bigram recall) → every segment gets its own small target part (possibly empty).
3. **Training.** Decoder models learn `<eos> segment "\n\nSummary:\n" → target part <eos>` (loss on the target only);
   the encoder-decoder reads the segment and writes the target part.
4. **Generation.** Batched greedy decoding with a KV cache (prompts left-padded, own positions per row; checked against plain forward passes),
   no repeated 4-grams of non-space tokens, a per-segment token budget, and an **EOS penalty** chosen on dev.
5. **Assembly.** Parts are joined in document order, near-duplicate sentences are removed (bigram overlap > 0.7), and the summary is trimmed to
   `LENGTH_SCALE` × the expected length (log-linear fit of summary length vs. document length on train).
6. **Evaluation.** Own ROUGE-1/2/L implementation on held-out train documents. The leaderboard's BERTScore needs a pretrained model and is not
   computed locally; the notebooks save `dev_predictions.json` + `dev_references.json` for the official evaluator.


## Submissions

| # | Submission | Backbone | Method | Codabench BERTScore F1 | Codabench ROUGE-L F1 | Codabench BLEU-4 | Own dev / validation |
|---|---|---|---|---|---|---|---|
| 01 | [01_arch1_lora](01_arch1_lora/) | Arch-1 | Arch-1 + LoRA | — | — | — | held-out dev: ROUGE-1 0.5019 · ROUGE-2 0.2452 · ROUGE-L 0.2545 (675 words; EOS penalty 2.0, length × 1.25) |
| 02 | [02_arch1_lorma](02_arch1_lorma/) | Arch-1 | Arch-1 + LoRMA | — | — | — | held-out dev: ROUGE-1 0.5026 · ROUGE-2 0.2507 · ROUGE-L 0.2521 (711 words; EOS penalty 2.0, length × 1.25) |
| 03 | [03_arch2_houlsby_adapters](03_arch2_houlsby_adapters/) | Arch-2 | Arch-2 + Houlsby adapters | — | — | — | held-out dev: ROUGE-1 0.3976 · ROUGE-2 0.1293 · ROUGE-L 0.2208 (513 words; EOS penalty 4.0, length × 0.8) |
| 04 | [04_arch4_adalora](04_arch4_adalora/) | Arch-4 | Arch-4 + AdaLoRA | — | — | — | held-out dev: ROUGE-1 0.3848 · ROUGE-2 0.1185 · ROUGE-L 0.2194 (521 words; EOS penalty 4.0, length × 0.8) |
| 05 | [05_kd_arch4_to_arch3](05_kd_arch4_to_arch3/) | Arch-3 | KD Arch-4 → Arch-3 | — | — | — | held-out dev (student): ROUGE-1 0.4385 · ROUGE-2 0.1718 · ROUGE-L 0.2357 (520 words; EOS penalty 4.0, length × 0.8) |

Best: **02_arch1_lorma** — Architecture-1 decoder + LoRMA, segment-wise summarization (chosen by held-out dev ROUGE until the Codabench scores are added).

## Observations

* Arch-1 with LoRMA or LoRA is clearly best on dev (ROUGE-1 ≈ 0.50, ROUGE-2 ≈ 0.25) and beats the lead baseline (first N words: 0.486 / 0.226 / 0.231).
* Every model ends segments too early without an EOS penalty (Arch-2/3/4 produce 0 words at penalty 0): the penalty search on dev is essential.
* Arch-2 and Arch-4 repeat themselves more (12–17% repeated 4-grams vs. 4.3% in the references).
* In the KD run the Arch-3 student ends with a much lower dev loss than its Arch-4 teacher (1.18 vs 1.80), so the gain comes mostly from the reference tokens.

Every submission folder holds the notebook (with its training log), `submission.zip` as uploaded to Codabench, and a README
with the method, the exact configuration and a map of the notebook cells.
