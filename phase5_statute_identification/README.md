# Phase 5 — Legal Statute Identification (LSI)

**Task.** **Multi-label** classification: assign every case all applicable statute ids out of 100.

**Data.** Train and dev documents with statute lists; test documents without labels. The text contains placeholders (`<ENTITY>`, `<SECTION>`, `<ACT>`).

**Metric.** Macro-F1 (primary) · Micro-F1 · subset accuracy

## Shared pipeline

1. **Input.** Placeholders are shortened (`<ENTITY>` → `X`, `<SECTION>` → `section`, `<ACT>` → `Act`); up to 8 chunks of 512 tokens
   (the whole document if it fits in 4,096 tokens, otherwise 8 chunks spread over its start, middle and end).
2. **Token vectors.** Every chunk goes through the Phase-1 backbone; token vector = learned **mix of all layers** (embeddings + every block),
   standardized. Whitespace tokens stay in the input (as in pretraining) but are **not** pooled or attended — they are about half of all tokens.
3. **Classifier (v2 head)** = two parts that are added:
   * linear path: mean of the document's token vectors → 100 scores; it **starts as a logistic regression** fitted on the frozen features;
   * **label attention**: every statute has its own attention over all tokens of the document (`a_s = softmax(q_s · tanh(K z))`), starts at 0.
4. **Loss.** Binary cross-entropy with a positive weight per statute (√(negatives / positives), capped at 10) for the rare statutes.
5. **Thresholds.** One threshold per statute, tuned on dev, turns the 100 probabilities into a set of statute ids.
6. **Protocol.** Trained on train only; dev picks the best epoch and the thresholds.


## Submissions

| # | Submission | Backbone | Method | Codabench Macro-F1 | Codabench Micro-F1 | Codabench Subset acc. | Own dev / validation |
|---|---|---|---|---|---|---|---|
| 01 | [01_arch1_lorma](01_arch1_lorma/) | Arch-1 | Arch-1 + LoRMA | 0.2861 | 0.378 | 0.009 | dev macro-F1 0.3158, micro-F1 0.3990, subset acc. 0.0103 (best epoch 4) |
| 02 | [02_arch1_qlora](02_arch1_qlora/) | Arch-1 | Arch-1 + QLoRA | **0.2919** | 0.38 | 0.01 | dev macro-F1 0.3188, micro-F1 0.4016, subset acc. 0.0120 (best epoch 4) |
| 03 | [03_arch2_full_ft](03_arch2_full_ft/) | Arch-2 | Arch-2 + full fine-tuning | 0.2858 | 0.375 | 0.01 | dev macro-F1 0.3124, micro-F1 0.3957, subset acc. 0.0097 (best epoch 10) |
| 04 | [04_arch3_houlsby_adapters](04_arch3_houlsby_adapters/) | Arch-3 | Arch-3 + Houlsby adapters | 0.2838 | 0.375 | 0.01 | dev macro-F1 0.3144, micro-F1 0.3918, subset acc. 0.0070 (best epoch 4) |
| 05 | [05_arch3_adalora](05_arch3_adalora/) | Arch-3 | Arch-3 + AdaLoRA | 0.2817 | 0.368 | 0.0113 | dev macro-F1 0.3105, micro-F1 0.3905, subset acc. 0.0100 (best epoch 4) |
| 06 | [06_arch4_partial_ft](06_arch4_partial_ft/) | Arch-4 | Arch-4 + partial fine-tuning | 0.2431 | 0.331 | 0.004 | dev macro-F1 0.2733, micro-F1 0.3456 (best epoch 3) |
| 07 | [07_kd_arch4_to_arch2](07_kd_arch4_to_arch2/) | Arch-2 | KD Arch-4 → Arch-2 | 0.2463 | 0.331 | 0.005 | dev macro-F1 (student) 0.2757, micro-F1 0.3444 (best epoch 4) |

Best: **02_arch1_qlora** — Architecture-1 decoder in 4-bit NF4 + LoRA (QLoRA) + layer mix + label attention, Codabench Macro-F1 0.2919.

## Observations

* All Arch-1/2/3 adaptations land within 0.28–0.29 Codabench macro-F1; QLoRA on Arch-1 is best (0.2919), so 4-bit quantization of the frozen base costs nothing here.
* Architecture-4 based systems are clearly weaker (0.243–0.246), including the distilled student of an Arch-4 teacher.
* Dev macro-F1 (≈0.31–0.32) is higher than Codabench (≈0.28–0.29): 100 per-statute thresholds tuned on dev overfit somewhat.
* A TF-IDF word-count baseline reaches dev macro-F1 0.394 (printed in the notebook header): statute ids are signalled by specific words, which the pooled LM features capture only partly. Subset accuracy stays around 1%.

Every submission folder holds the notebook (with its training log), `submission.zip` as uploaded to Codabench, and a README
with the method, the exact configuration and a map of the notebook cells.
