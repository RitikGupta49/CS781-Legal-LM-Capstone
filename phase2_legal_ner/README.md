# Phase 2 — Legal Named Entity Recognition (L-NER)

**Task.** Find legal named entities (appellant, respondent, judge, court, statute, precedent, case number, date, witness, counsel, authority, …) as exact **character spans** with labels.

**Data.** Two annotated folds (`fold1`, `fold2`) for training and the label list; `fold3` (35 documents) is the test set.

**Metric.** Strict Macro-F1 (exact span + label)

## Shared pipeline

1. **Character-safe units.** Text is split into words / numbers / punctuation with exact character offsets, and every unit gets a
   BIO tag (12 entity types → 25 tags). An offset self-test turns the gold spans into tags and back: strict macro-F1 ceiling **0.9938**.
2. **Windows.** Documents are longer than the context, so they are cut into 512-token windows with overlap (stride 384 or 256);
   overlapping predictions are averaged.
3. **Backbone.** One of the Phase-1 models, mostly frozen, with a PEFT method (LoRMA, LoRA) or full / partial fine-tuning.
4. **Head** — two families:
   * word features + BiLSTM + CRF (submissions 01, 02, 04): first sub-token vector of each word + 32-d word-shape embedding + 6 word flags
     → BiLSTM (256 per direction) → linear-chain CRF with invalid BIO transitions blocked;
   * linear token classifier + constrained Viterbi (submission 03).
5. **Protocol.** Stage A: train on fold 1, validate on fold 2 (score for the report, best epoch). Stage B: retrain on fold 1 + fold 2
   for that many epochs, predict fold 3.
6. **Output.** `prediction.json` = for every test document the list of `[start, end, label]` spans, zipped at the root of `submission.zip`.


## Submissions

| # | Submission | Backbone | Method | Codabench Strict Macro-F1 | Codabench Strict Macro-P | Codabench Strict Macro-R | Own dev / validation |
|---|---|---|---|---|---|---|---|
| 01 | [01_arch1_lorma](01_arch1_lorma/) | Arch-1 | Arch-1 + LoRMA | **0.5535** | 0.6067 | 0.5303 | fold-2 strict macro-F1 0.4953 (P 0.5634, R 0.4775), best epoch 8 of stage A |
| 02 | [02_arch2_full_ft](02_arch2_full_ft/) | Arch-2 | Arch-2 + full fine-tuning | 0.4246 | 0.4839 | 0.3919 | fold-2 strict macro-F1 0.3317 (P 0.3795, R 0.3055), best epoch 8 of stage A |
| 03 | [03_arch3_lora](03_arch3_lora/) | Arch-3 | Arch-3 + LoRA | 0.4912 | 0.5297 | 0.4814 | validation strict macro-F1 0.4722 (micro-F1 0.6067), best epoch 10 |
| 04 | [04_kd_arch4_to_arch2](04_kd_arch4_to_arch2/) | Arch-2 | KD Arch-4 → Arch-2 | 0.4445 | 0.4917 | 0.436 | fold-2 strict macro-F1: teacher 0.3618, student 0.4309 |

Best: **01_arch1_lorma** — Architecture-1 decoder (frozen) + LoRMA + word features + BiLSTM + CRF, Codabench Strict Macro-F1 0.5535.

## Observations

* The frozen Arch-1 decoder with LoRMA and the BiLSTM-CRF head is clearly best (0.5535). Causal attention was kept, the BiLSTM provides the right-hand context.
* Codabench scores are higher than our fold-2 validation scores (e.g. 0.5535 vs 0.4953) because the stage-B model also learns from fold 2.
* Distillation helped the small Arch-2 encoder: the KD student (0.4445) beats Arch-2 full fine-tuning (0.4246) and even its Arch-4 teacher on validation. Distilling only on words the teacher labels correctly kept its errors out of the student.
* Most remaining errors are span-boundary errors with the right label, not wrong labels.

Every submission folder holds the notebook (with its training log), `submission.zip` as uploaded to Codabench, and a README
with the method, the exact configuration and a map of the notebook cells.
