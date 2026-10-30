# Phase 4 · Submission 06 — Architecture-2 encoder (frozen) + LoRMA

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | LoRMA, rank 16, α 32, on the attention and FFN projections of the encoder (1.33M weights) |
| Codabench | Macro-F1 **0.5898** · Precision 0.588 · Recall 0.61  (rank 8 of 11 in this phase) |
| Own dev / validation | dev macro-F1 0.5920 at threshold 0.29, AUC 0.6297 (best epoch 2) |
| Notebook | `p4_arch2_lorma.ipynb` (original name: `p4-archi2-lorma.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p4-archi2-lorma-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-4 pipeline on the Architecture-2 encoder with **LoRMA** (LR 2e-4: 1e-4 hardly moved, 5e-4 from the start destroyed the features).

The data handling, head and evaluation shared by all Phase-4 notebooks are described in the [phase README](../README.md).

## Model size (printed by the notebook)

```
Phase-1 model (frozen)   : 67,894,272
LoRMA (trainable)        : 1,327,104  (1.955% of the Phase-1 model, 72 tensors)
Head (trainable)         : 3,842,571
Trainable total          : 5,169,675  (7.08% of 73,063,947)
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p4_arch2_lorma_v3"
DOC_PART = "end"
DOC_TOKENS = 1536            # tokens per document the model reads
CHUNK_LEN = 512              # every chunk goes through the encoder separately
LORMA_TARGETS = ["self_attention.q_proj", "self_attention.k_proj", "self_attention.v_proj",
                 "self_attention.out_proj", "feed_forward.fc1", "feed_forward.fc2"]
LORMA_RANK = 16
LORMA_ALPHA = 32
LORMA_DROPOUT = 0.05
LR_PEFT = 2e-4              # v1: 1e-4 (hardly moved), v2: 5e-4 from the start (destroyed the features)
BACKBONE_DROPOUT = 0.0       # Phase-1 dropout inside the encoder is switched off (Architectures 1 and 3
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
CLASS_BALANCING = True       # loss weights = inverse class frequency (macro-F1 counts both classes)
EPOCHS = 8                   # epoch = EPOCH_DOCS cases; 8 x 12,000 = ~2.6 passes over train
EPOCH_DOCS = 12000           # cases per epoch, then dev is checked (all train cases are used before any repeats)
BATCH_DOCS = 8               # documents per forward pass (8 x 3 chunks x 512 tokens). If CUDA out-of-memory: 4
GRAD_ACCUM = 2               # -> 16 documents per weight update
PATIENCE = 3                 # stop if dev macro-F1 does not improve for 3 epochs
HEAD_ONLY_EPOCHS = 1         # epoch 1 trains only the head + layer mix (LoRMA waits, the encoder runs
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
10. CELL 10 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-2 CHECKPOINT
11. CELL 11 — PREDICTION + EVALUATION
12. CELL 12 — PHASE-1 FEATURES: WHICH LAYER KNOWS THE OUTCOME? + TOKEN STATISTICS (before training)
13. CELL 13 — OPTIMIZER + TRAINING FUNCTION
14. CELL 14 — TRAIN (TRAIN data only; dev picks the best epoch)
15. CELL 15 — DEV REPORT
16. CELL 16 — PREDICT TEST + WRITE prediction.json + submission.zip
