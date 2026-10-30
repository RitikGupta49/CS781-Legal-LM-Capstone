# Phase 3 · Submission 01 — Architecture-1 decoder (frozen) + LoRA + sentence BiLSTM + CRF

| | |
|---|---|
| Phase-1 backbone | Architecture-1 decoder-only (110M) |
| Adaptation | LoRA, rank 16, α 32, on qkv / out_proj / gate / up / down |
| Codabench | Macro-F1 **0.4362** · Macro-P 0.4862 · Macro-R 0.45  (rank 2 of 7 in this phase) |
| Own dev / validation | dev macro-F1: CL 0.4145, IT 0.5796, mean 0.4970 (best epoch 5) |
| Notebook | `p3_arch1_lora.ipynb` (original name: `p3-archi-1-lora.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p3-archi-1-lora-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-3 pipeline (see the phase README) with **LoRA** adapters on every linear layer of the frozen decoder.
* Context = document: consecutive sentences packed into 1,024-token windows, so every sentence sees the sentences before it.

The data handling, head and evaluation shared by all Phase-3 notebooks are described in the [phase README](../README.md).

## Notes

* Same pipeline as submission 02 — the only difference is LoRA vs LoRMA.

## Model size (printed by the notebook)

```
Phase-1 decoder (frozen) : 109,534,464
LoRA (trainable)         : 2,506,752  (2.29% of the decoder)
Head (trainable)         : 2,113,744
Trainable total          : 4,620,496  (4.05% of 114,154,960)
```

## Configuration (CELL 2 of the notebook)

```python
CONTEXT = "document"
RUN_NAME = f"p3_arch1_lora_{CONTEXT}"
MAX_SENT_TOKENS = 256        # longer sentences are cut (CELL 6 prints how many)
WINDOW = 1024                # "document": consecutive sentences are packed into windows of 1024 tokens
WINDOWS_PER_CHUNK = 8        # "document": windows per training step (8 x 1024 tokens, ~7 GB). If CUDA out-of-memory: 4
SENTENCES_PER_CHUNK = 32     # "sentence": sentences per training step
LORA_TARGETS = ["qkv", "out_proj", "gate", "up", "down"]     # every linear layer of every block
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR_LORA = 5e-4
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
10. CELL 10 — OPTIMIZER + LR SCHEDULE  (only LoRA + head are trained)
11. CELL 11 — PREDICTION + EVALUATION
12. CELL 12 — TRAINING FUNCTION + STAGE A (train on train, pick the best epoch on dev)
13. CELL 13 — DEV REPORT: per domain, per label, confusions + save results
14. CELL 15 — PREDICT TEST + WRITE prediction.json + submission.zip
