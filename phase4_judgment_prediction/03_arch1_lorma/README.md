# Phase 4 · Submission 03 — Architecture-1 decoder (frozen) + LoRMA

| | |
|---|---|
| Phase-1 backbone | Architecture-1 decoder-only (110M) |
| Adaptation | LoRMA, rank 16, α 32, on qkv / out_proj / gate / up / down |
| Codabench | Macro-F1 **0.6556** · Precision 0.698 · Recall 0.564  (rank 2 of 11 in this phase) |
| Own dev / validation | dev macro-F1 0.6532 at threshold 0.48, AUC 0.7190 (best epoch 5) |
| Notebook | `p4_arch1_lorma.ipynb` (original name: `p4-archi1-lorma.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p4-archi1-lorma-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-4 pipeline with **LoRMA** on every linear layer (LR 1e-4); epoch 1 head only.

The data handling, head and evaluation shared by all Phase-4 notebooks are described in the [phase README](../README.md).

## Notes

* Best Architecture-1 run in Phase 4.

## Model size (printed by the notebook)

```
Phase-1 decoder (frozen) : 109,534,464
LoRMA (trainable)        : 3,047,424  (2.78% of the decoder)
Head (trainable)         : 3,842,564
Trainable total          : 6,889,988  (5.92% of 116,424,452)
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p4_arch1_lorma"
DOC_PART = "end"
DOC_TOKENS = 1536            # tokens per document the model reads
CHUNK_LEN = 512              # every chunk goes through the decoder separately
LORMA_TARGETS = ["qkv", "out_proj", "gate", "up", "down"]    # every linear layer of every block
LORMA_RANK = 16
LORMA_ALPHA = 32
LORMA_DROPOUT = 0.05
LR_LORMA = 1e-4             # small: LoRMA changes the features the head reads
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
CLASS_BALANCING = True       # loss weights = inverse class frequency (macro-F1 counts both classes)
EPOCHS = 5
EPOCH_DOCS = 8000            # documents per epoch (all train documents are used before any repeats)
BATCH_DOCS = 4               # documents per forward pass (4 x 3 chunks x 512 tokens). If CUDA out-of-memory: 2
GRAD_ACCUM = 4               # -> 16 documents per weight update
PATIENCE = 2                 # stop if dev macro-F1 does not improve for 2 epochs
HEAD_ONLY_EPOCHS = 1         # epoch 1 trains only the head (LoRMA waits). Its dev score =
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
EVAL_CHUNKS = 48             # chunks per forward pass when predicting
STATS_DOCS = 1000
PROBE_DOCS = 3000
RUN_SIGNAL_CHECK = True
SIGNAL_TRAIN_DOCS = 15000
DEBUG_MODE = False
DEBUG_DOCS = 64              # per split
USE_AMP = DEVICE.type == "cuda"
USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16
MAX_CHUNKS = DOC_TOKENS // CHUNK_LEN
```

## Notebook cells

1. CELL 1 — IMPORTS + REPRODUCIBILITY + DEVICE
2. CELL 2 — CONFIGURATION
3. CELL 3 — LOAD THE PHASE-1 TOKENIZER + FAST ENCODING
4. CELL 4 — PHASE-4 DATA FILES  (download needs the Kaggle secret "HF")
5. CELL 5 — LOAD CASES + STATISTICS
6. CELL 6 — TOKENIZE (only the part the model reads) + CHUNKS
7. CELL 7 — METRICS + DECISION THRESHOLD
8. CELL 8 — WHERE IS THE SIGNAL?  (data analysis only, NOT a submission)
9. CELL 9 — MODEL
10. CELL 10 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-1 CHECKPOINT
11. CELL 11 — PREDICTION + EVALUATION
12. CELL 12 — PHASE-1 FEATURES: TOKEN STATISTICS + LINEAR PROBE (before training)
13. CELL 13 — OPTIMIZER + TRAINING FUNCTION
14. CELL 14 — TRAIN (TRAIN data only; dev picks the best epoch)
15. CELL 15 — DEV REPORT
16. CELL 16 — PREDICT TEST + WRITE prediction.json + submission.zip
