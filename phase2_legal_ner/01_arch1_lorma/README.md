# Phase 2 · Submission 01 — Architecture-1 decoder (frozen) + LoRMA + word features + BiLSTM + CRF

| | |
|---|---|
| Phase-1 backbone | Architecture-1 decoder-only (110M) |
| Adaptation | LoRMA, rank 16, α 32, on qkv / out_proj / gate / up / down of all 12 blocks |
| Codabench | Strict Macro-F1 **0.5535** · Strict Macro-P 0.6067 · Strict Macro-R 0.5303  (rank 1 of 4 in this phase) |
| Own dev / validation | fold-2 strict macro-F1 0.4953 (P 0.5634, R 0.4775), best epoch 8 of stage A |
| Notebook | `p2_arch1_lorma.ipynb` (original name: `p2-arch1-lorma.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p2-archi1-lorma-submission-973871.zip`), contains `prediction.json` at the root |
| Best of the phase | ✅ this folder is also in the Hello IITK `code/phase2/` folder |

## Method

* Text is split into words with exact character offsets; every word is labelled with a BIO tag (12 entity types → 25 tags).
* Tokens → frozen Architecture-1 decoder with **LoRMA** on every linear layer (`W' = (I + α/r·BA)W`, B = 0 at the start so the model starts as the Phase-1 model) → vector of the first sub-token of every word.
* Word vector + word-shape embedding (32-d, e.g. `Xxxx`, `dd.dd.dddd`) + 6 word flags → BiLSTM (256 per direction) over the words → linear-chain CRF; invalid BIO transitions are fixed to −10,000.
* Causal attention as in pretraining (the BiLSTM supplies the right-hand context); documents in 512-token windows with stride 384.
* Stage A: train fold 1, validate fold 2 (report score, best epoch). Stage B: retrain on fold 1 + fold 2 for the best number of epochs, predict fold 3 (test).
* LR 1e-3 (LoRMA) / 2e-3 (head), 6% warm-up, early stopping (patience 6).

The data handling, head and evaluation shared by all Phase-2 notebooks are described in the [phase README](../README.md).

## Notes

* Best Phase-2 submission. An offset self-test (gold spans → tokens → spans) gives a ceiling of strict macro-F1 0.9938, so the tokenization loses almost nothing.

## Model size (printed by the notebook)

```
Phase-1 decoder (frozen): 109,534,464
LoRMA (trainable)       : 3,047,424  (2.78% of the decoder)
Head (trainable)        : 2,196,476
Trainable total         : 5,243,900  (4.57% of 114,778,364)
```

## Configuration (CELL 2 of the notebook)

```python
TRAIN_FOLDS = ["fold1"]          # stage A: train
VAL_FOLDS = ["fold2"]            # stage A: validate (score for the report)
FINAL_TRAIN_ON_ALL_FOLDS = True
WINDOW = 512
STRIDE = 384
LORMA_TARGETS = ["qkv", "out_proj", "gate", "up", "down"]   # linear layers of every block
LORMA_RANK = 16
LORMA_ALPHA = 32
LORMA_DROPOUT = 0.05
ATTENTION = "causal"
SHAPE_DIM = 32
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.3
EPOCHS = 30
BATCH_SIZE = 8                   # windows per step. If CUDA out-of-memory: 4
EVAL_BATCH_SIZE = 16
LR_LORMA = 1e-3                  # LoRMA matrices A, B
LR_HEAD = 2e-3                   # word features + BiLSTM + CRF
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0
PATIENCE = 6
TRAIN_EVAL_DOCS = 8
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
11. CELL 11 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-1 CHECKPOINT
12. CELL 12 — OPTIMIZER + LR SCHEDULE  (only LoRMA + head are trained)
13. CELL 13 — PREDICTION + EVALUATION
14. CELL 14 — TRAINING FUNCTION + STAGE A (train fold1, validate fold2)
15. CELL 15 — STAGE A REPORT (per label) + SAVE RESULTS
16. CELL 16 — STAGE B (fold1 + fold2) + TEST PREDICTIONS -> submission.zip
17. CELL 17 — SAVE FOLD-2 PREDICTIONS + GOLD FOR THE OFFICIAL EVALUATOR
