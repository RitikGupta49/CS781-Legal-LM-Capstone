# Phase 4 · Submission 10 — Architecture-4 hybrid prefix-LM + BitFit

| | |
|---|---|
| Phase-1 backbone | Architecture-4 hybrid prefix-LM (205M) |
| Adaptation | BitFit: new biases on both attention branches, the fusion layers and the FFN (200,704 weights, 0.098%) |
| Codabench | Macro-F1 **0.527** · Precision 0.527 · Recall 0.57  (rank 11 of 11 in this phase) |
| Own dev / validation | dev macro-F1 0.5219 at threshold 0.43, AUC 0.5220 (best epoch 2) |
| Notebook | `p4_arch4_bitfit.ipynb` (original name: `p4-archi4-bitfit.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p4-archi4-bitfit-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-4 pipeline on Architecture-4 with **BitFit**; the bidirectional prefix keeps the Phase-1 prefix length.

The data handling, head and evaluation shared by all Phase-4 notebooks are described in the [phase README](../README.md).

## Notes

* Dev AUC stayed near 0.52: the head hardly learned on top of this backbone (the notebook prints a warning for this case).

## Model size (printed by the notebook)

```
Phase-1 model (frozen)   : 205,496,064
BitFit (trainable)       : 200,704  (0.098% of the Phase-1 model, 208 tensors)
Head (trainable)         : 3,842,564
Trainable total          : 4,043,268  (1.93% of 209,539,332)
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p4_arch4_bitfit"
DOC_PART = "end"
DOC_TOKENS = 1536            # tokens per document the model reads
CHUNK_LEN = 512              # every chunk goes through the hybrid prefix-LM separately
BITFIT_TARGETS = ["encoder_attention.q_proj", "encoder_attention.k_proj",
                  "encoder_attention.v_proj", "encoder_attention.out_proj",
                  "causal_attention.q_proj", "causal_attention.k_proj", "causal_attention.v_proj",
                  "causal_attention.out_proj", "hybrid_fusion.gate_proj", "hybrid_fusion.out_proj",
                  "ffn.gate_proj", "ffn.up_proj", "ffn.down_proj"]
LR_PEFT = 1e-4              # small: the biases shift the features the head reads
BACKBONE_DROPOUT = 0.0       # Phase-1 dropout inside the model is switched off (Architectures 1 and 3
PREFIX_LENGTH = None         # bidirectional prefix of every chunk. None = the Phase-1 prefix_length
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
CLASS_BALANCING = True       # loss weights = inverse class frequency (macro-F1 counts both classes)
EPOCHS = 10
EPOCH_DOCS = 6000            # documents per epoch (all train documents are used before any repeats)
BATCH_DOCS = 2               # documents per forward pass (2 x 3 chunks x 512 tokens). If CUDA out-of-memory: 1
GRAD_ACCUM = 8               # -> 16 documents per weight update
PATIENCE = 2                 # stop if dev macro-F1 does not improve for 2 epochs
HEAD_ONLY_EPOCHS = 1         # epoch 1 trains only the head (BitFit waits). Its dev score =
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
EVAL_CHUNKS = 24             # chunks per forward pass when predicting
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
10. CELL 10 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-4 CHECKPOINT
11. CELL 11 — PREDICTION + EVALUATION
12. CELL 12 — PHASE-1 FEATURES: TOKEN STATISTICS + LINEAR PROBE (before training)
13. CELL 13 — OPTIMIZER + TRAINING FUNCTION
14. CELL 14 — TRAIN (TRAIN data only; dev picks the best epoch)
15. CELL 15 — DEV REPORT
16. CELL 16 — PREDICT TEST + WRITE prediction.json + submission.zip
