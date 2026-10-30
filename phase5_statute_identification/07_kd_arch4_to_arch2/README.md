# Phase 5 · Submission 07 — Knowledge distillation: Architecture-4 teacher → Architecture-2 encoder student (multi-label)

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | teacher: frozen Architecture-4 + classifier; student: classifier + all 6 encoder layers (layer-wise LRs) |
| Codabench | Macro-F1 **0.2463** · Micro-F1 0.331 · Subset acc. 0.005  (rank 6 of 7 in this phase) |
| Own dev / validation | dev macro-F1 (student) 0.2757, micro-F1 0.3444 (best epoch 4) |
| Notebook | `p5_kd_arch4_to_arch2.ipynb` (original name: `p5-kd-4-2.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p5-kd-4-2-submission.zip`), contains `prediction.json` at the root |


## Method

* Teacher = frozen Architecture-4 (205.5M) + the shared classifier, trained on 16,000 train documents.
* The teacher scores **other** train documents it did not learn from (honest soft labels) for all 100 statutes.
* Student = Architecture-2 encoder (~68M weights used) + the same classifier; loss = 0.5 · BCE(gold) + 0.5 · T²·BCE(student, teacher probabilities), T = 1.
* The student's predictions are submitted.

The data handling, head and evaluation shared by all Phase-5 notebooks are described in the [phase README](../README.md).

## Notes

* The student keeps 100% of the teacher's dev macro-F1 with 33% of its weights, but the teacher itself is weak (Architecture-4 features).

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p5_kd_arch4_to_arch2"
CHUNK_LEN = 512
MAX_CHUNKS = 8
PLACEHOLDERS = {"<ENTITY>": "X", "<SECTION>": "section", "<ACT>": "Act"}
POOL_SKIP_SPACES = True      # whitespace tokens are not pooled / attended (they stay in the input)
KD_ALPHA = 0.5               # weight of the teacher term (0 = no distillation)
KD_TEMPERATURE = 1.0         # the teacher's probabilities are already soft
TEACHER_TRAIN_DOCS = 16000
STUDENT_TRAIN_DOCS = 16000   # ~40 min for the teacher to score them
POOL_SKIP_TOKENS = 4
LABEL_ATTENTION_DIM = 256
HEAD_DROPOUT = 0.1
LR_HEAD = 1e-3
POS_WEIGHT_POWER = 0.5
POS_WEIGHT_MAX = 10.0
STUDENT_TOP_BLOCKS = 6       # Architecture-2 encoder layers that train (6 = all; the embeddings stay frozen)
LR_TOP_BLOCK = 3e-5          # top encoder layer; every layer below gets LAYER_LR_DECAY x the rate above it
LAYER_LR_DECAY = 0.8
BACKBONE_DROPOUT = 0.0       # Phase-1 dropout inside both models is switched off
PREFIX_LENGTH = None         # Architecture-4 bidirectional prefix: None = the Phase-1 prefix_length
TEACHER_EPOCHS = 2           # classifier only (the teacher's Phase-1 model stays frozen)
TEACHER_EPOCH_DOCS = 8000
TEACHER_BATCH_DOCS = 2
TEACHER_GRAD_ACCUM = 8       # -> 16 documents per weight update
STUDENT_EPOCHS = 4           # epoch 1 = classifier only, then classifier + encoder layers
STUDENT_EPOCH_DOCS = 8000
STUDENT_BATCH_DOCS = 2       # If CUDA out-of-memory: 1
STUDENT_GRAD_ACCUM = 8
STUDENT_HEAD_ONLY_EPOCHS = 1
PATIENCE = 2
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
EVAL_CHUNKS = 24             # chunks per forward pass when predicting
DEV_EVAL_DOCS = 2000         # dev documents scored after every epoch (the full dev set at the end, CELL 16)
STATS_DOCS = 500
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
8. CELL 8 — MODELS
9. CELL 9 — LOAD BOTH PHASE-1 CHECKPOINTS + MODEL BUILDERS
10. CELL 10 — PREDICTION + EVALUATION
11. CELL 11 — PREPARE A MODEL BEFORE TRAINING (teacher and student)
12. CELL 12 — OPTIMIZER + TRAINING FUNCTION (teacher and student)
13. CELL 13 — STEP 1: THE TEACHER (Architecture-4), trained on its TRAIN documents with the gold statutes
14. CELL 14 — STEP 2: TEACHER PROBABILITIES
15. CELL 15 — STEP 3: THE STUDENT (Architecture-2 encoder) WITH KNOWLEDGE DISTILLATION
16. CELL 16 — DEV REPORT (the STUDENT, all dev documents) + TEACHER vs STUDENT
17. CELL 17 — PREDICT TEST WITH THE STUDENT + WRITE prediction.json + submission.zip
