# Phase 5 · Submission 04 — Architecture-3 E-LegalFormer (frozen) + Houlsby adapters + layer mix + label attention

| | |
|---|---|
| Phase-1 backbone | Architecture-3 E-LegalFormer, GQA (125M) |
| Adaptation | Houlsby bottleneck adapters (dim 64) after attention.o_proj and ffn.down_proj of every block |
| Codabench | Macro-F1 **0.2838** · Micro-F1 0.375 · Subset acc. 0.01  (rank 4 of 7 in this phase) |
| Own dev / validation | dev macro-F1 0.3144, micro-F1 0.3918, subset acc. 0.0070 (best epoch 4) |
| Notebook | `p5_arch3_houlsby_adapters.ipynb` (original name: `p5-archi-3-adaptors.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p5-archi-3-adaptors-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-5 pipeline with **Houlsby adapters** (Houlsby et al., 2019): `out = y + up(GELU(down(y)))`, up-projection initialised at 0 so the model starts unchanged (LR 5e-4).

The data handling, head and evaluation shared by all Phase-5 notebooks are described in the [phase README](../README.md).

## Model size (printed by the notebook)

```
head (trainable)          : 376,181
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p5_arch3_adapters"
CHUNK_LEN = 512              # tokens per chunk (every chunk goes through the Phase-1 model separately)
MAX_CHUNKS = 8               # up to 8 x 512 = 4096 tokens per document (covers ~75% of documents fully)
PLACEHOLDERS = {"<ENTITY>": "X", "<SECTION>": "section", "<ACT>": "Act"}
POOL_SKIP_SPACES = True      # whitespace tokens are not pooled / attended (they stay in the input)
ADAPTER_TARGETS = ["attention.o_proj", "ffn.down_proj"]    # in every block
ADAPTER_DIM = 64
ADAPTER_DROPOUT = 0.05
LR_PEFT = 5e-4
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LABEL_ATTENTION_DIM = 256    # size of the attention keys of the label attention
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
POS_WEIGHT_POWER = 0.5
POS_WEIGHT_MAX = 10.0
EPOCHS = 4
EPOCH_DOCS = 10000           # documents per epoch (all train documents are used before any repeats)
BATCH_DOCS = 2               # documents per forward pass (each up to 8 chunks). If CUDA out-of-memory: 1
GRAD_ACCUM = 8               # -> 16 documents per weight update
HEAD_ONLY_EPOCHS = 1
PATIENCE = 2                 # stop if dev macro-F1 does not improve for 2 epochs
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
EVAL_CHUNKS = 48             # chunks per forward pass when predicting
DEV_EVAL_DOCS = 3000         # dev documents scored after every epoch (the full dev set at the end, CELL 14)
STATS_DOCS = 800
PROBE_DOCS = 4000
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
9. CELL 9 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-3 CHECKPOINT
10. CELL 10 — PREDICTION + EVALUATION
11. CELL 11 — BEFORE TRAINING: PROBE EVERY LAYER + START OF THE CLASSIFIER
12. CELL 12 — OPTIMIZER + TRAINING FUNCTION
13. CELL 13 — TRAIN (TRAIN data only; dev picks the best epoch)
14. CELL 14 — DEV REPORT (all dev documents)
15. CELL 15 — PREDICT TEST + WRITE prediction.json + submission.zip
