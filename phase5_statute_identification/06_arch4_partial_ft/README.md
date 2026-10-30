# Phase 5 · Submission 06 — Architecture-4 hybrid prefix-LM + partial fine-tuning + layer mix + label attention

| | |
|---|---|
| Phase-1 backbone | Architecture-4 hybrid prefix-LM (205M) |
| Adaptation | top 4 of 16 blocks train with layer-wise LRs (3e-5 at the top, ×0.8 per block) |
| Codabench | Macro-F1 **0.2431** · Micro-F1 0.331 · Subset acc. 0.004  (rank 7 of 7 in this phase) |
| Own dev / validation | dev macro-F1 0.2733, micro-F1 0.3456 (best epoch 3) |
| Notebook | `p5_arch4_partial_ft.ipynb` (original name: `p5-archi-4-partialfinetuning.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p5-archi-4-partialfinetuning-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-5 pipeline; epoch 1 trains the head only, then the top 4 blocks of the 205M model; Phase-1 dropout off.

The data handling, head and evaluation shared by all Phase-5 notebooks are described in the [phase README](../README.md).

## Notes

* Architecture-4 gives the weakest statute features in this phase.

## Model size (printed by the notebook)

```
head (trainable)          : 376,181
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p5_arch4_partial_ft"
CHUNK_LEN = 512              # tokens per chunk (every chunk goes through the Phase-1 model separately)
MAX_CHUNKS = 8               # up to 8 x 512 = 4096 tokens per document (covers ~75% of documents fully)
PLACEHOLDERS = {"<ENTITY>": "X", "<SECTION>": "section", "<ACT>": "Act"}
POOL_SKIP_SPACES = True      # whitespace tokens are not pooled / attended (they stay in the input)
PARTIAL_TOP_BLOCKS = 4
LR_TOP_BLOCK = 3e-5
LAYER_LR_DECAY = 0.8
BACKBONE_DROPOUT = 0.0       # Phase-1 dropout inside the model is switched off
PREFIX_LENGTH = None         # bidirectional prefix of every chunk. None = the Phase-1 prefix_length
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LABEL_ATTENTION_DIM = 256    # size of the attention keys of the label attention
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
POS_WEIGHT_POWER = 0.5
POS_WEIGHT_MAX = 10.0
EPOCHS = 3
EPOCH_DOCS = 8000           # documents per epoch (all train documents are used before any repeats)
BATCH_DOCS = 2               # documents per forward pass (each up to 8 chunks). If CUDA out-of-memory: 1
GRAD_ACCUM = 8               # -> 16 documents per weight update
HEAD_ONLY_EPOCHS = 1
PATIENCE = 2                 # stop if dev macro-F1 does not improve for 2 epochs
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
EVAL_CHUNKS = 24             # chunks per forward pass when predicting
DEV_EVAL_DOCS = 2000         # dev documents scored after every epoch (the full dev set at the end, CELL 14)
STATS_DOCS = 800
PROBE_DOCS = 3000
DEBUG_MODE = False
DEBUG_DOCS = 200             # per split
USE_AMP = DEVICE.type == "cuda"
USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16
```

## Notebook cells

1. CELL 1 — IMPORTS + REPRODUCIBILITY + DEVICE
2. CELL 2 — CONFIGURATION
3. CELL 3 — LOAD THE PHASE-1 TOKENIZER + FAST ENCODING
4. CELL 4 — PHASE-5 DATA FILES  (download needs the Kaggle secret "HF")
5. CELL 5 — LOAD DOCUMENTS + STATUTES
6. CELL 6 — TOKENIZE + CHUNKS
7. CELL 7 — METRICS + THRESHOLDS (multi-label)
8. CELL 8 — MODEL
9. CELL 9 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-4 CHECKPOINT
10. CELL 10 — PREDICTION + EVALUATION
11. CELL 11 — BEFORE TRAINING: PROBE EVERY LAYER + START OF THE CLASSIFIER
12. CELL 12 — OPTIMIZER + TRAINING FUNCTION
13. CELL 13 — TRAIN (TRAIN data only; dev picks the best epoch)
14. CELL 14 — DEV REPORT (all dev documents)
15. CELL 15 — PREDICT TEST + WRITE prediction.json + submission.zip
