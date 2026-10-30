# Phase 5 · Submission 02 — Architecture-1 decoder in 4-bit NF4 + LoRA (QLoRA) + layer mix + label attention

| | |
|---|---|
| Phase-1 backbone | Architecture-1 decoder-only (110M) |
| Adaptation | QLoRA: frozen linear weights quantized to 4-bit NF4 (block size 64) + LoRA rank 16, α 32 |
| Codabench | Macro-F1 **0.2919** · Micro-F1 0.38 · Subset acc. 0.01  (rank 1 of 7 in this phase) |
| Own dev / validation | dev macro-F1 0.3188, micro-F1 0.4016, subset acc. 0.0120 (best epoch 4) |
| Notebook | `p5_arch1_qlora.ipynb` (original name: `p5-archi1-qlora.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p5-archi1-qlora-submission.zip`), contains `prediction.json` at the root |
| Best of the phase | ✅ this folder is also in the Hello IITK `code/phase5/` folder |

## Method

* Own NF4 implementation (Dettmers et al., 2023): the 84.9M linear weights of the decoder are stored in 4 bits per weight with one absmax scale per block of 64 (42 MB instead of 170 MB in 16-bit), dequantized on the fly.
* Round-trip check: relative weight error 9.1%; the model output changes by 17% (relative) after quantization — LoRA learns to compensate.
* LoRA (rank 16) on the same five linear layers, shared Phase-5 head.

The data handling, head and evaluation shared by all Phase-5 notebooks are described in the [phase README](../README.md).

## Notes

* Best Phase-5 submission (Codabench macro-F1 0.2919): quantizing the frozen base did not cost accuracy.

## Model size (printed by the notebook)

```
head (trainable)          : 376,177
4-bit Phase-1 weights     : 84,934,656 (42 MB instead of 170 MB in 16-bit)
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p5_arch1_qlora"
CHUNK_LEN = 512              # tokens per chunk (every chunk goes through the Phase-1 model separately)
MAX_CHUNKS = 8               # up to 8 x 512 = 4096 tokens per document (covers ~75% of documents fully)
PLACEHOLDERS = {"<ENTITY>": "X", "<SECTION>": "section", "<ACT>": "Act"}
POOL_SKIP_SPACES = True      # whitespace tokens are not pooled / attended (they stay in the input)
QLORA_TARGETS = ["attn.qkv", "attn.out_proj", "mlp.gate", "mlp.up", "mlp.down"]    # every linear layer of every block
QUANT_BLOCK = 64
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR_PEFT = 2e-4
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
9. CELL 9 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-1 CHECKPOINT
10. CELL 10 — PREDICTION + EVALUATION
11. CELL 11 — BEFORE TRAINING: PROBE EVERY LAYER + START OF THE CLASSIFIER
12. CELL 12 — OPTIMIZER + TRAINING FUNCTION
13. CELL 13 — TRAIN (TRAIN data only; dev picks the best epoch)
14. CELL 14 — DEV REPORT (all dev documents)
15. CELL 15 — PREDICT TEST + WRITE prediction.json + submission.zip
