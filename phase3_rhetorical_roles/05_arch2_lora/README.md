# Phase 3 · Submission 05 — Architecture-2 encoder (frozen) + LoRA + sentence BiLSTM + CRF

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | LoRA, rank 16, α 32, on q/k/v/out_proj/fc1/fc2 of all encoder layers |
| Codabench | Macro-F1 **0.345** · Macro-P 0.402 · Macro-R 0.3458  (rank 6 of 7 in this phase) |
| Own dev / validation | dev macro-F1: CL 0.3196, IT 0.3721, mean 0.3459 (best epoch 11) |
| Notebook | `p3_arch2_lora.ipynb` (original name: `p3-archi-2-lora.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p3-archi-2-lora-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-3 pipeline on the bidirectional Architecture-2 encoder with **LoRA** (1.33M adapter weights).
* LR 5e-4 (same as the Architecture-1 LoRA notebook) so the two backbones compare directly.

The data handling, head and evaluation shared by all Phase-3 notebooks are described in the [phase README](../README.md).

## Model size (printed by the notebook)

```
LoRA (trainable)         : 1,327,104  (1.955% of the encoder)
Head (trainable)         : 2,113,744
Trainable total          : 3,440,848  (4.82% of 71,335,120)
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p3_arch2_lora"
MAX_SENT_TOKENS = 256        # longer sentences are cut
WINDOW = 1024                # consecutive sentences are packed into windows of 1024 tokens
WINDOWS_PER_CHUNK = 8        # windows per training step (8 x 1024 tokens). If CUDA out-of-memory: 4
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]   # every linear layer of every encoder block
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LR_ADAPTER = 5e-4            # same as the Architecture-1 LoRA notebook
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.3
LR_HEAD = 2e-3
CLASS_BALANCING = True       # rare roles get a larger loss weight (macro-F1 counts every role equally)
CLASS_WEIGHT_MAX = 5.0       # cap, so a role with 3 sentences does not dominate
CE_WEIGHT = 0.5              # loss = CRF + CE_WEIGHT * class-weighted cross-entropy
MIN_LABEL_COUNT = 10
EPOCHS = 15
PATIENCE = 5                 # stop if the dev score does not improve for 5 epochs
HEAD_ONLY_EPOCHS = 1         # epoch 1 trains only the head (the Phase-1 encoder is used as it is)
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0
TRAIN_EVAL_DOCS = 10         # a few train documents are scored every epoch (to see under-fitting)
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
9. CELL 9 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-2 CHECKPOINT
10. CELL 10 — PREDICTION + EVALUATION
11. CELL 11 — LOSS + OPTIMIZER + TRAINING FUNCTION
12. CELL 12 — TRAIN (TRAIN data only, best epoch picked on dev)
13. CELL 13 — DEV REPORT: per domain, per label, confusions + save results
14. CELL 14 — PREDICT TEST + WRITE prediction.json + submission.zip
