# Phase 4 — Court Judgment Prediction (CJPE)

**Task.** Predict the **binary outcome** of a court case (appeal accepted / rejected) from the case text.

**Data.** Train and dev cases with labels (≈37.4k train cases), test cases without labels.

**Metric.** Macro-F1

## Shared pipeline

1. **What the model reads.** A judgment is far longer than the context, and the outcome is stated near the end, so every case is reduced
   to its **last 1,536 tokens** (`DOC_PART = "end"`), cut from the end into 3 chunks of 512 tokens.
2. **Chunk vectors.** Every chunk goes through the Phase-1 backbone separately; token vectors (the last layer, or a learned mix of all
   layers in the later notebooks) are standardized; chunk vector = [attention-pooled ; mean] of its tokens (the first 4 tokens of a chunk
   are skipped — they behave abnormally in LMs).
3. **Case head.** LayerNorm → BiLSTM over the chunks in order → attention pooling (learns which chunks matter) → 2 classes.
   Class-balanced cross-entropy; epoch 1 trains the head only.
4. **Fine-tuning technique** (the point of this phase): BitFit, LoRMA, gradual unfreezing, knowledge distillation — see the table.
5. **Decision threshold.** P(label = 1) is thresholded at the value that maximises dev macro-F1 (shown as `@t` in the logs).
6. **Protocol.** Trained on train only; dev picks the best epoch and the threshold. A frozen-layer probe and a TF-IDF reference
   (dev AUC 0.708, macro-F1 0.653) are printed for comparison.


## Submissions

| # | Submission | Backbone | Method | Codabench Macro-F1 | Codabench Precision | Codabench Recall | Own dev / validation |
|---|---|---|---|---|---|---|---|
| 01 | [01_arch1_bitfit](01_arch1_bitfit/) | Arch-1 | Arch-1 + BitFit | 0.6206 | 0.66 | 0.5157 | dev macro-F1 0.6371 at threshold 0.39, AUC 0.6835 (best epoch 5) |
| 02 | [02_arch1_freeze_unfreeze](02_arch1_freeze_unfreeze/) | Arch-1 | Arch-1 + gradual unfreezing | 0.6047 | 0.625 | 0.538 | dev macro-F1 0.6473 at threshold 0.22, AUC 0.6785 (best epoch 3: top 6/12 blocks trained) |
| 03 | [03_arch1_lorma](03_arch1_lorma/) | Arch-1 | Arch-1 + LoRMA | 0.6556 | 0.698 | 0.564 | dev macro-F1 0.6532 at threshold 0.48, AUC 0.7190 (best epoch 5) |
| 04 | [04_arch2_bitfit](04_arch2_bitfit/) | Arch-2 | Arch-2 + BitFit | 0.5592 | 0.575 | 0.4855 | dev macro-F1 0.5729 at threshold 0.36, AUC 0.5858 (best epoch 2) |
| 05 | [05_arch2_freeze_unfreeze](05_arch2_freeze_unfreeze/) | Arch-2 | Arch-2 + gradual unfreezing | 0.5479 | 0.557 | 0.494 | dev macro-F1 0.5562 at threshold 0.43, AUC 0.5732 (best epoch 1 = head only) |
| 06 | [06_arch2_lorma](06_arch2_lorma/) | Arch-2 | Arch-2 + LoRMA | 0.5898 | 0.588 | 0.61 | dev macro-F1 0.5920 at threshold 0.29, AUC 0.6297 (best epoch 2) |
| 07 | [07_arch3_bitfit](07_arch3_bitfit/) | Arch-3 | Arch-3 + BitFit | 0.5974 | 0.629 | 0.501 | dev macro-F1 0.6188 at threshold 0.20, AUC 0.6595 (best epoch 3) |
| 08 | [08_arch3_freeze_unfreeze](08_arch3_freeze_unfreeze/) | Arch-3 | Arch-3 + gradual unfreezing | 0.6451 | 0.66 | 0.59 | dev macro-F1 0.6670 at threshold 0.28, AUC 0.7278 (best epoch 6, all 16 blocks) |
| 09 | [09_arch3_lorma](09_arch3_lorma/) | Arch-3 | Arch-3 + LoRMA | **0.6896** | 0.71 | 0.645 | dev macro-F1 0.6811 at threshold 0.33, AUC 0.7462 (best epoch 6) |
| 10 | [10_arch4_bitfit](10_arch4_bitfit/) | Arch-4 | Arch-4 + BitFit | 0.527 | 0.527 | 0.57 | dev macro-F1 0.5219 at threshold 0.43, AUC 0.5220 (best epoch 2) |
| 11 | [11_kd_arch4_to_arch2](11_kd_arch4_to_arch2/) | Arch-2 | KD Arch-4 → Arch-2 | 0.5937 | 0.6 | 0.5656 | dev macro-F1: teacher 0.6304 (AUC 0.6568), student 0.6395 (AUC 0.6711, best epoch 3) |

Best: **09_arch3_lorma** — Architecture-3 E-LegalFormer (frozen) + LoRMA, Codabench Macro-F1 0.6896.

## Observations

* Arch-3 + LoRMA is the best Phase-4 system (0.6896), followed by Arch-1 + LoRMA (0.6556) and Arch-3 gradual unfreezing (0.6451).
* LoRMA beats BitFit and gradual unfreezing on Arch-1 and Arch-3; on Arch-2 every technique stays below 0.59.
* Arch-4 + BitFit barely learns (dev AUC 0.52).
* Distillation compresses well: the Arch-2 student keeps 101% of the Arch-4 teacher's dev macro-F1 with 33% of its weights and is 4× faster at prediction.
* Only the best runs beat the TF-IDF reference: the outcome is carried mostly by which words appear near the end of the judgment.

Every submission folder holds the notebook (with its training log), `submission.zip` as uploaded to Codabench, and a README
with the method, the exact configuration and a map of the notebook cells.
