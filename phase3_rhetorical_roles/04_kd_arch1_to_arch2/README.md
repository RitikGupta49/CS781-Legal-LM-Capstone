# Phase 3 · Submission 04 — Knowledge distillation: Architecture-1 + LoRMA teacher → Architecture-2 encoder student

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | teacher: LoRMA on the frozen decoder; student: all Architecture-2 encoder weights |
| Codabench | Macro-F1 **0.36** · Macro-P 0.425 · Macro-R 0.3592  (rank 5 of 7 in this phase) |
| Own dev / validation | dev mean macro-F1: teacher 0.4816, student 0.3602 |
| Notebook | `p3_kd_arch1_to_arch2.ipynb` (original name: `p3-archi-1-2-kd.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p3-archi-1-2-KD-submission.zip`), contains `prediction.json` at the root |


## Method

* Step 1: teacher = Architecture-1 + LoRMA (the 02 setup), trained on train.
* Step 2: teacher probabilities for every train sentence (it is right on 82.3% of them).
* Step 3: student = Architecture-2 encoder (70.0M, 61% of the teacher) + the same head; loss = 0.5 · gold loss + 0.5 · T²·KL(teacher ‖ student) at T = 2, only on sentences the teacher gets right.
* The student's predictions are submitted.

The data handling, head and evaluation shared by all Phase-3 notebooks are described in the [phase README](../README.md).

## Notes

* The student keeps 75% of the teacher's dev macro-F1 with 61% of its parameters and 31% of its prediction time.

## Model size (printed by the notebook)

```
Teacher: 114,695,632 parameters, 5,161,168 trainable (LoRMA + head)
Student: 70,008,016 parameters, all trainable -> 61% of the teacher's size
```

## Configuration (CELL 2 of the notebook)

```python
KD_ALPHA = 0.5               # loss = (1 - KD_ALPHA) * gold loss + KD_ALPHA * distillation loss
KD_TEMPERATURE = 2.0         # softens the teacher's probabilities
KD_ONLY_WHERE_TEACHER_CORRECT = True   # distil only sentences the teacher labels correctly
RUN_NAME = "p3_kd_arch1lorma_to_arch2" if KD_ALPHA > 0 else "p3_arch2_no_kd"
MAX_SENT_TOKENS = 256        # longer sentences are cut
WINDOW = 1024                # consecutive sentences are packed into windows of 1024 tokens
TEACHER_WINDOWS_PER_CHUNK = 8    # windows per teacher training step. If CUDA out-of-memory: 4
STUDENT_WINDOWS_PER_CHUNK = 4    # windows per student training step (all weights train). If out-of-memory: 2
LORMA_TARGETS = ["qkv", "out_proj", "gate", "up", "down"]    # every linear layer of every block
LORMA_RANK = 16
LORMA_ALPHA = 32
LORMA_DROPOUT = 0.05
LR_LORMA = 5e-4
TEACHER_EPOCHS = 15
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.3
CLASS_BALANCING = True       # rare roles get a larger loss weight (macro-F1 counts every role equally)
CLASS_WEIGHT_MAX = 5.0       # cap, so a role with 3 sentences does not dominate
CE_WEIGHT = 0.5              # gold loss = CRF + CE_WEIGHT * class-weighted cross-entropy
MIN_LABEL_COUNT = 10
LR_STUDENT_ENCODER = 1e-4    # same as the Phase-2 student
WEIGHT_DECAY = 0.01          # student encoder only
STUDENT_EPOCHS = 15
PATIENCE = 5                 # stop if the dev score does not improve for 5 epochs
HEAD_ONLY_EPOCHS = 1         # epoch 1 trains only the head (backbone unchanged)
LR_HEAD = 2e-3
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
8. CELL 8 — MODELS
9. CELL 9 — PREDICTION + EVALUATION (any model: teacher or student)
10. CELL 10 — LOSSES + OPTIMIZER + TRAINING FUNCTION (used for teacher AND student)
11. CELL 11 — STEP 1: TRAIN THE TEACHER
12. CELL 12 — STEP 2: TEACHER SCORES FOR EVERY TRAIN SENTENCE
13. CELL 13 — STUDENT: Architecture-2 encoder (all weights trained) + head
14. CELL 14 — STEP 3: TRAIN THE STUDENT WITH DISTILLATION (TRAIN data only)
15. CELL 15 — DEV REPORT: teacher vs student, per label, confusions + save results
16. CELL 16 — PREDICT TEST WITH THE STUDENT + WRITE prediction.json + submission.zip
