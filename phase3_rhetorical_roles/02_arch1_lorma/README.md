# Phase 3 · Submission 02 — Architecture-1 decoder (frozen) + LoRMA + sentence BiLSTM + CRF

| | |
|---|---|
| Phase-1 backbone | Architecture-1 decoder-only (110M) |
| Adaptation | LoRMA, rank 16, α 32, on qkv / out_proj / gate / up / down |
| Codabench | Macro-F1 **0.5266** · Macro-P 0.5735 · Macro-R 0.5141  (rank 1 of 7 in this phase) |
| Own dev / validation | dev macro-F1: CL 0.4778, IT 0.5524, mean 0.5151 (best epoch 13) |
| Notebook | `p3_arch1_lorma.ipynb` (original name: `p3-archi-1-lorma.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p3-archi-1-lorma-submission.zip`), contains `prediction.json` at the root |
| Best of the phase | ✅ this folder is also in the Hello IITK `code/phase3/` folder |

## Method

* Shared Phase-3 pipeline with **LoRMA** (multiplicative low-rank adaptation) on every linear layer of the frozen decoder.
* LR 5e-4 (LoRMA) / 2e-3 (head); 1 head-only epoch first; 15 epochs, patience 5.

The data handling, head and evaluation shared by all Phase-3 notebooks are described in the [phase README](../README.md).

## Notes

* Best Phase-3 submission: LoRMA beats LoRA by 0.09 macro-F1 on Codabench with the same pipeline.

## Model size (printed by the notebook)

```
Phase-1 decoder (frozen) : 109,534,464
LoRMA (trainable)        : 3,047,424  (2.78% of the decoder)
Head (trainable)         : 2,113,744
Trainable total          : 5,161,168  (4.50% of 114,695,632)
```

## Configuration (CELL 2 of the notebook)

```python
CONTEXT = "document"
RUN_NAME = f"p3_arch1_lorma_{CONTEXT}"
MAX_SENT_TOKENS = 256        # longer sentences are cut (CELL 6 prints how many)
WINDOW = 1024                # "document": consecutive sentences are packed into windows of 1024 tokens
WINDOWS_PER_CHUNK = 8        # "document": windows per training step (8 x 1024 tokens, ~7 GB). If CUDA out-of-memory: 4
SENTENCES_PER_CHUNK = 32     # "sentence": sentences per training step
LORMA_TARGETS = ["qkv", "out_proj", "gate", "up", "down"]    # every linear layer of every block
LORMA_RANK = 16
LORMA_ALPHA = 32
LORMA_DROPOUT = 0.05
LR_LORMA = 5e-4
LSTM_HIDDEN = 256
MLP_HIDDEN = 256
HEAD_DROPOUT = 0.3
LR_HEAD = 2e-3
CLASS_BALANCING = True       # rare roles get a larger loss weight (macro-F1 counts every role equally)
CLASS_WEIGHT_MAX = 5.0       # cap, so a role with 3 sentences does not dominate
CE_WEIGHT = 0.5              # "document": loss = CRF + CE_WEIGHT * class-weighted cross-entropy
MIN_LABEL_COUNT = 10
EPOCHS = 15
PATIENCE = 5                 # stop if the dev score does not improve for 5 epochs
HEAD_ONLY_EPOCHS = 1
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0
TRAIN_EVAL_DOCS = 10         # a few train documents are scored every epoch (to see under-fitting)
FINAL_TRAIN_ON_DEV = True
DEBUG_MODE = False
DEBUG_DOCS = 6               # per file
USE_AMP = DEVICE.type == "cuda"
USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16
```

## Notebook cells

1. CELL 1 — IMPORTS + REPRODUCIBILITY + DEVICE
2. CELL 2 — CONFIGURATION
3. CELL 3 — LOAD THE PHASE-1 TOKENIZER
4. CELL 4 — PHASE-3 DATA FILES  (download needs the Kaggle secret "HF")
5. CELL 5 — LOAD DOCUMENTS + LABELS + STATISTICS
6. CELL 6 — TOKENIZE + WINDOWS + CHUNKS
7. CELL 7 — SCORES + VITERBI
8. CELL 8 — MODEL
9. CELL 9 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-1 CHECKPOINT
10. CELL 10 — OPTIMIZER + LR SCHEDULE  (only LoRMA + head are trained)
11. CELL 11 — PREDICTION + EVALUATION
12. CELL 12 — TRAINING FUNCTION + STAGE A (train on train, pick the best epoch on dev)
13. CELL 13 — DEV REPORT: per domain, per label, confusions + save results
14. CELL 15 — PREDICT TEST + WRITE prediction.json + submission.zip
