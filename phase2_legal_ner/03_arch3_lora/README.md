# Phase 2 · Submission 03 — Architecture-3 E-LegalFormer (frozen) + LoRA, bidirectional attention, linear token head

| | |
|---|---|
| Phase-1 backbone | Architecture-3 E-LegalFormer, GQA (125M) |
| Adaptation | LoRA, rank 16, α 32, on q/k/v/o and the SwiGLU projections |
| Codabench | Strict Macro-F1 **0.4912** · Strict Macro-P 0.5297 · Strict Macro-R 0.4814  (rank 2 of 4 in this phase) |
| Own dev / validation | validation strict macro-F1 0.4722 (micro-F1 0.6067), best epoch 10 |
| Notebook | `p2_arch3_lora.ipynb` (original name: `p2-archi-3-lora.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p2-archi-3-lora-submission.zip`), contains `prediction.json` at the root |


## Method

* Text is split into word / number / punctuation units with exact character offsets; each unit takes the prediction of its last sub-token.
* Frozen E-LegalFormer + **LoRA** (rank 16) on every attention and SwiGLU projection; the causal mask is removed during fine-tuning so every token sees left and right context (`USE_BIDIRECTIONAL = True`).
* Linear token-classification head over 25 BIO tags; constrained Viterbi decoding (no `I-` without its `B-`).
* 512-token sliding windows, stride 256 for training and prediction; overlapping predictions are averaged.
* 7 training documents held out for validation, 12 epochs with early stopping (patience 3).

The data handling, head and evaluation shared by all Phase-2 notebooks are described in the [phase README](../README.md).

## Notes

* Error analysis on validation: most errors are boundary errors with the right label (194) rather than label errors.

## Configuration (CELL 2 of the notebook)

```python
EXPERIMENT_NAME = "E3_LoRA_Arch3_ELegalFormer"
DEBUG = False                # True -> 6 docs, 1 epoch (pipeline check only)
USE_BIDIRECTIONAL = True     # remove causal mask during fine-tuning
WINDOW_TOKENS = 512          # tokens per window (must be <= 2048)
TRAIN_STRIDE = 256           # window step during training
PREDICT_STRIDE = 256         # window step during prediction (overlaps are averaged)
NUM_VAL_DOCS = 7
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]
HEAD_DROPOUT = 0.1
EPOCHS = 12
PATIENCE = 3                 # early stop after this many epochs without val F1 gain
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 1
LORA_LR = 5e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0
PREDICT_BATCH_SIZE = 16
```

## Notebook cells

1. CELL 1 — IMPORTS AND SETTINGS
2. CELL 2 — PATHS
3. CELL 3 — EXPERIMENT CONFIG
4. CELL 4 — LOAD PHASE-2 DATA
5. CELL 5 — DATA CHECKS
6. CELL 6 — LOAD PHASE-1 TOKENIZER
7. CELL 7 — UNITS AND CHARACTER-SAFE TOKENIZATION
8. CELL 8 — BIO LABELS, SPAN CONVERSION, VITERBI
9. CELL 9 — STRICT SCORER (same rule as Codabench)
10. CELL 10 — PIPELINE SANITY CHECK (no model)
11. CELL 11 — TRAIN / VALIDATION SPLIT AND ENCODING
12. CELL 12 — SLIDING WINDOWS AND DATALOADER
13. CELL 13 — ARCH-3 E-LEGALFORMER (encoder mode for NER)
14. CELL 14 — LOAD PHASE-1 CHECKPOINT
15. CELL 15 — CHECK: DOES THE LOADED MODEL STILL KNOW LEGAL TEXT?
16. CELL 16 — LoRA
17. CELL 17 — NER MODEL (backbone + token-classification head)
18. CELL 18 — OPTIMIZER AND SCHEDULE
19. CELL 19 — PREDICTION (document -> character spans)
20. CELL 20 — TRAINING
21. CELL 21 — TRAINING CURVES
22. CELL 22 — BEST MODEL: VALIDATION REPORT + ERROR ANALYSIS
23. CELL 23 — PREDICT TEST SET
24. CELL 24 — VALIDATE AND WRITE prediction.json + ZIP
25. CELL 25 — SAVE EXPERIMENT SUMMARY (for the report)
