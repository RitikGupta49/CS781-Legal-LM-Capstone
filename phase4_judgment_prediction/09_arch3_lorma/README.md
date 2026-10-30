# Phase 4 · Submission 09 — Architecture-3 E-LegalFormer (frozen) + LoRMA

| | |
|---|---|
| Phase-1 backbone | Architecture-3 E-LegalFormer, GQA (125M) |
| Adaptation | LoRMA, rank 16, α 32, on q/k/v/o and the SwiGLU projections (3.54M weights, 2.8%) |
| Codabench | Macro-F1 **0.6896** · Precision 0.71 · Recall 0.645  (rank 1 of 11 in this phase) |
| Own dev / validation | dev macro-F1 0.6811 at threshold 0.33, AUC 0.7462 (best epoch 6) |
| Notebook | `p4_arch3_lorma.ipynb` (original name: `p4-archi3-lorma.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p4-archi3-lorma-submission.zip`), contains `prediction.json` at the root |
| Best of the phase | ✅ this folder is also in the Hello IITK `code/phase4/` folder |

## Method

* Shared Phase-4 pipeline with **LoRMA** on every linear layer of the E-LegalFormer (LR 1e-4), 10 epochs of 8,000 cases, patience 2.

The data handling, head and evaluation shared by all Phase-4 notebooks are described in the [phase README](../README.md).

## Notes

* Best Phase-4 submission; also above the TF-IDF reference printed in the notebooks (dev macro-F1 0.653, AUC 0.708).

## Model size (printed by the notebook)

```
Phase-1 model (frozen)   : 125,267,712
LoRMA (trainable)        : 3,538,944  (2.825% of the Phase-1 model, 224 tensors)
Head (trainable)         : 3,842,564
Trainable total          : 7,381,508  (5.56% of 132,649,220)
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p4_arch3_lorma"
DOC_PART = "end"
DOC_TOKENS = 1536            # tokens per document the model reads
CHUNK_LEN = 512              # every chunk goes through the E-LegalFormer separately
LORMA_TARGETS = ["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj",
                 "ffn.gate_proj", "ffn.up_proj", "ffn.down_proj"]
LORMA_RANK = 16
LORMA_ALPHA = 32
LORMA_DROPOUT = 0.05
LR_PEFT = 1e-4              # small: LoRMA changes the features the head reads
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
CLASS_BALANCING = True       # loss weights = inverse class frequency (macro-F1 counts both classes)
EPOCHS = 10
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
12. CELL 12 — PHASE-1 FEATURES: TOKEN STATISTICS + LINEAR PROBE (before training)
13. CELL 13 — OPTIMIZER + TRAINING FUNCTION
14. CELL 14 — TRAIN (TRAIN data only; dev picks the best epoch)
15. CELL 15 — DEV REPORT
16. CELL 16 — PREDICT TEST + WRITE prediction.json + submission.zip
