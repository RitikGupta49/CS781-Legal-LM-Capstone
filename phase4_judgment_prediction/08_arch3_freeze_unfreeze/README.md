# Phase 4 · Submission 08 — Architecture-3 E-LegalFormer + gradual unfreezing

| | |
|---|---|
| Phase-1 backbone | Architecture-3 E-LegalFormer, GQA (125M) |
| Adaptation | gradual unfreezing of the 16 blocks (discriminative LRs) |
| Codabench | Macro-F1 **0.6451** · Precision 0.66 · Recall 0.59  (rank 3 of 11 in this phase) |
| Own dev / validation | dev macro-F1 0.6670 at threshold 0.28, AUC 0.7278 (best epoch 6, all 16 blocks) |
| Notebook | `p4_arch3_freeze_unfreeze.ipynb` (original name: `p4-archi3-freeze-unfreeze.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p4-archi3-freezeUnfreeze-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-4 pipeline with the gradual-unfreezing schedule on the E-LegalFormer.

The data handling, head and evaluation shared by all Phase-4 notebooks are described in the [phase README](../README.md).

## Notes

* Second-best Phase-4 submission.

## Model size (printed by the notebook)

```
Head (trainable)         : 3,842,581
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p4_arch3_unfreeze"
DOC_PART = "end"
DOC_TOKENS = 1536            # tokens per document the model reads
CHUNK_LEN = 512              # every chunk goes through the E-LegalFormer separately
UNFREEZE_FRACTIONS = [0.0, 0.25, 0.5, 1.0]   # per epoch: share of the blocks (from the top) that train;
LR_TOP_BLOCK = 3e-5          # learning rate of the top block (full weights, as in normal fine-tuning)
LAYER_LR_DECAY = 0.8         # every block gets 0.8 x the learning rate of the block above it
UNFREEZE_WARMUP = 200        # updates: a block that was just unfrozen warms its learning rate up from 0
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
CLASS_BALANCING = True       # loss weights = inverse class frequency (macro-F1 counts both classes)
EPOCHS = 6                   # epoch = EPOCH_DOCS cases; epoch 1 = head only, then gradual unfreezing
EPOCH_DOCS = 8000             # cases per epoch, then dev is checked (all train cases are used before any repeats)
BATCH_DOCS = 4               # documents per forward pass (4 x 3 chunks x 512 tokens). If CUDA out-of-memory: 2
GRAD_ACCUM = 4               # -> 16 documents per weight update
PATIENCE = 3                 # stop if dev macro-F1 does not improve for 3 epochs (checked once every block is unfrozen)
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
EVAL_CHUNKS = 48             # chunks per forward pass when predicting
STATS_DOCS = 1000
PROBE_DOCS = 3000
RUN_SIGNAL_CHECK = False
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
10. CELL 10 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-3 CHECKPOINT
11. CELL 11 — PREDICTION + EVALUATION
12. CELL 12 — PHASE-1 FEATURES: WHICH LAYER KNOWS THE OUTCOME? + TOKEN STATISTICS (before training)
13. CELL 13 — OPTIMIZER + TRAINING FUNCTION (gradual unfreezing)
14. CELL 14 — TRAIN (TRAIN data only; dev picks the best epoch)
15. CELL 15 — DEV REPORT
16. CELL 16 — PREDICT TEST + WRITE prediction.json + submission.zip
