# Phase 4 · Submission 11 — Knowledge distillation: Architecture-4 teacher → Architecture-2 encoder student

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | head + token embeddings of each Phase-1 model (partial fine-tuning); transformer blocks frozen |
| Codabench | Macro-F1 **0.5937** · Precision 0.6 · Recall 0.5656  (rank 7 of 11 in this phase) |
| Own dev / validation | dev macro-F1: teacher 0.6304 (AUC 0.6568), student 0.6395 (AUC 0.6711, best epoch 3) |
| Notebook | `p4_kd_arch4_to_arch2.ipynb` (original name: `p4-kd-archi4-2.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p4-kd-4-2-submission.zip`), contains `prediction.json` at the root |


## Method

* Teacher = Architecture-4 (205.5M), student = Architecture-2 encoder (67.9M backbone weights, 3.0× smaller); both read the last 1,536 tokens and use the same head (mix of all layers → token attention → BiLSTM over chunks).
* Both start as a logistic regression on the frozen features, then train the head + token embeddings (LR 1e-4).
* Teacher probabilities for all 37,387 train cases; student loss = 0.5 · CE(gold) + 0.5 · T²·KL(teacher ‖ student), T = 1.
* The student's predictions are submitted.

The data handling, head and evaluation shared by all Phase-4 notebooks are described in the [phase README](../README.md).

## Notes

* The student keeps 101% of the teacher's dev macro-F1 with 33% of its weights; they agree on 70.4% of dev cases.

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p4_kd_arch4_to_arch2_v3"
DOC_PART = "end"             # "end" = the last DOC_TOKENS tokens, "start", or "start_end"
DOC_TOKENS = 1536            # tokens per document
CHUNK_LEN = 512              # every chunk goes through the model separately
KD_ALPHA = 0.5               # weight of the teacher term (0 = no distillation)
KD_TEMPERATURE = 1.0         # 2 classes and soft teacher probabilities -> T = 1
TEACHER_SCORED_DOCS = None   # train cases the teacher scores (= the student's train cases).
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
CLASS_BALANCING = True       # loss weights = inverse class frequency (macro-F1 counts both classes)
TRAIN_EMBEDDINGS = True      # False = head only (the Phase-1 model stays completely frozen)
LR_EMBEDDINGS = 1e-4         # token embeddings (AdamW, no weight decay); 3e-4 made the loss jump at the start
BACKBONE_DROPOUT = 0.0       # Phase-1 dropout inside both models is switched off
PREFIX_LENGTH = None         # Architecture-4 bidirectional prefix: None = the Phase-1 prefix_length
TEACHER_EPOCHS = 3           # epoch = EPOCH_DOCS cases, then dev is checked
TEACHER_EPOCH_DOCS = 8000
TEACHER_BATCH_DOCS = 8       # documents per forward pass (the blocks run without gradients). If CUDA out-of-memory: 4
TEACHER_GRAD_ACCUM = 2       # -> 16 documents per weight update
STUDENT_EPOCHS = 4
STUDENT_EPOCH_DOCS = 12000
STUDENT_BATCH_DOCS = 8       # If CUDA out-of-memory: 4
STUDENT_GRAD_ACCUM = 2       # -> 16 documents per weight update
PATIENCE = 2                 # stop if dev macro-F1 does not improve for 2 epochs
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
EVAL_CHUNKS = 24             # chunks per forward pass when predicting
STATS_DOCS = 1000
PROBE_DOCS = 5000
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
8. CELL 8 — MODELS
9. CELL 9 — LOAD BOTH PHASE-1 CHECKPOINTS + MODEL BUILDERS
10. CELL 10 — PREDICTION + EVALUATION (any model: teacher or student)
11. CELL 11 — PREPARE A MODEL BEFORE TRAINING (teacher and student)
12. CELL 12 — OPTIMIZER + TRAINING FUNCTION (teacher and student)
13. CELL 13 — STEP 1: THE TEACHER (Architecture-4), trained on TRAIN with the gold labels
14. CELL 14 — STEP 2: TEACHER PROBABILITIES
15. CELL 15 — STEP 3: THE STUDENT (Architecture-2 encoder) WITH KNOWLEDGE DISTILLATION
16. CELL 16 — DEV REPORT (the STUDENT) + TEACHER vs STUDENT
17. CELL 17 — PREDICT TEST WITH THE STUDENT + WRITE prediction.json + submission.zip
