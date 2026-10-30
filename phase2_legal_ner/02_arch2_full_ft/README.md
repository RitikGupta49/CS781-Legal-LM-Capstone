# Phase 2 · Submission 02 — Architecture-2 encoder, full fine-tuning + word features + BiLSTM + CRF

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | full fine-tuning of the Architecture-2 encoder (all 6 layers + embeddings) |
| Codabench | Strict Macro-F1 **0.4246** · Strict Macro-P 0.4839 · Strict Macro-R 0.3919  (rank 4 of 4 in this phase) |
| Own dev / validation | fold-2 strict macro-F1 0.3317 (P 0.3795, R 0.3055), best epoch 8 of stage A |
| Notebook | `p2_arch2_full_ft.ipynb` (original name: `p2-archi-2-full-fine-tuning.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p2-archi-2-full-FT-submission-969987.zip`), contains `prediction.json` at the root |


## Method

* Same data pipeline, word features, BiLSTM and CRF as submission 01.
* Backbone: the encoder half of the Phase-1 encoder-decoder (bidirectional, 6 layers); the decoder is not used.
* All encoder weights train (full fine-tuning), 15 epochs, batch 8 windows of 512 tokens (stride 384).
* Stage A (fold 1 → fold 2) picks the epoch count, stage B retrains on both folds.

The data handling, head and evaluation shared by all Phase-2 notebooks are described in the [phase README](../README.md).

## Notes

* Weakest Phase-2 run: the 6-layer encoder trained on 1,024-token contexts gives weaker token features than the 12-layer decoder.

## Configuration (CELL 2 of the notebook)

```python
TRAIN_FOLDS = ["fold1"]          # stage A: train
VAL_FOLDS = ["fold2"]            # stage A: validate (score for the report)
FINAL_TRAIN_ON_ALL_FOLDS = True
WINDOW = 512
STRIDE = 384
SHAPE_DIM = 32           # word-shape embedding ("Xxxx", "dd.dd.dddd", ...)
LSTM_HIDDEN = 256        # per direction
HEAD_DROPOUT = 0.3
EPOCHS = 15
BATCH_SIZE = 8                   # windows per step. If CUDA out-of-memory: 4
EVAL_BATCH_SIZE = 16
LR_ENCODER = 1e-4                # pretrained encoder
LR_HEAD = 2e-3                   # word features + BiLSTM + CRF (new)
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0
PATIENCE = 4                     # stage A early stopping (epochs without val improvement)
TRAIN_EVAL_DOCS = 8              # training docs scored each epoch (under/overfitting check)
DEBUG_MODE = False
DEBUG_DOCS = 12
USE_AMP = DEVICE.type == "cuda"
USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16
```

## Notebook cells

1. CELL 1 — IMPORTS + REPRODUCIBILITY + DEVICE
2. CELL 2 — CONFIGURATION
3. CELL 3 — LOAD TOKENIZER + TOKENIZE WITH CHARACTER OFFSETS
4. CELL 4 — DOWNLOAD PHASE-2 DATA  (needs the Kaggle secret "HF")
5. CELL 5 — LOAD TEXTS, LABELS AND GOLD ENTITIES
6. CELL 6 — TAGS (BIO), WORD FEATURES, SPANS, STRICT F1
7. CELL 7 — TOKENIZE ALL DOCUMENTS (tokens -> words) + WINDOWS
8. CELL 8 — OFFSET SELF-TEST  (run before training!)
9. CELL 9 — BATCHES
10. CELL 10 — MODEL
11. CELL 11 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-2 CHECKPOINT
12. CELL 12 — OPTIMIZER + LR SCHEDULE
13. CELL 13 — PREDICTION + EVALUATION
14. CELL 14 — TRAINING FUNCTION + STAGE A (train fold1, validate fold2)
15. CELL 15 — STAGE A REPORT (per label) + SAVE RESULTS
16. CELL 16 — STAGE B (fold1 + fold2) + TEST PREDICTIONS -> submission.zip
