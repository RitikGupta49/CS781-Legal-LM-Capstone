# Phase 2 · Submission 04 — Knowledge distillation: Architecture-4 teacher (partial FT) → Architecture-2 encoder student

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | teacher: top 8 of 16 Architecture-4 blocks fine-tuned (layer-wise LR decay 0.85); student: full Architecture-2 encoder |
| Codabench | Strict Macro-F1 **0.4445** · Strict Macro-P 0.4917 · Strict Macro-R 0.436  (rank 3 of 4 in this phase) |
| Own dev / validation | fold-2 strict macro-F1: teacher 0.3618, student 0.4309 |
| Notebook | `p2_kd_arch4_to_arch2.ipynb` (original name: `p2-archiu4-2-KD.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p2-archiu4-2-KD-975420.zip`), contains `prediction.json` at the root |


## Method

* Teacher (E4): Architecture-4 hybrid prefix-LM, top 8 blocks trainable with layer-wise learning rates (3e-4 at the top, ×0.85 per block below), same word features + BiLSTM + CRF head; 20 epochs.
* Student (E5): Architecture-2 encoder (70.1M parameters, 3.0× smaller), all weights trainable, same head.
* Distillation (Hinton et al., 2015): student loss = 0.5 · CRF loss on gold tags + 0.5 · T² · KL(teacher ‖ student) at T = 2, **only on words the teacher labels correctly** (its mistakes are not copied).
* Stage A picks epochs on fold 2; stage B retrains teacher and student on fold 1 + fold 2. The submitted file is the student's (3,897 test entities).

The data handling, head and evaluation shared by all Phase-2 notebooks are described in the [phase README](../README.md).

## Notes

* The student beats its teacher on validation (0.431 vs 0.362) with 34% of its parameters and 32% of its prediction time, and beats Architecture-2 full fine-tuning on Codabench (0.4445 vs 0.4246).

## Model size (printed by the notebook)

```
Teacher (E4): 207,692,508 parameters, 91,868,892 trainable (44.2%)
Student (E5): 70,090,716 parameters, all trainable -> 3.0x smaller than the teacher
```

## Configuration (CELL 2 of the notebook)

```python
TRAIN_FOLDS = ["fold1"]          # stage A: train
VAL_FOLDS = ["fold2"]            # stage A: validate (scores for the report)
FINAL_TRAIN_ON_ALL_FOLDS = True
WINDOW = 512
STRIDE = 384
SHAPE_DIM = 32
LSTM_HIDDEN = 256
HEAD_DROPOUT = 0.3
TEACHER_TRAIN_TOP_BLOCKS = 8     # of 16 blocks (4 was too few: the teacher underfitted)
TEACHER_EPOCHS = 20
TEACHER_PATIENCE = 5
TEACHER_BATCH_SIZE = 8           # if CUDA out-of-memory: 4
LR_TEACHER_BLOCKS = 3e-4
LR_TEACHER_HEAD = 2e-3
TEACHER_LAYER_LR_DECAY = 0.85    # layer-wise LR: top block LR_TEACHER_BLOCKS, each lower block x 0.85
KD_ALPHA = 0.5
KD_TEMPERATURE = 2.0
KD_ONLY_WHERE_TEACHER_CORRECT = True
STUDENT_EPOCHS = 30
STUDENT_PATIENCE = 6
STUDENT_BATCH_SIZE = 8
LR_STUDENT_ENCODER = 1e-4
LR_STUDENT_HEAD = 2e-3
EVAL_BATCH_SIZE = 16
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
TRAIN_EVAL_DOCS = 8
DOC_PROPAGATION = True
DEBUG_MODE = False
DEBUG_DOCS = 12
USE_AMP = DEVICE.type == "cuda"
USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16
```

## Notebook cells

1. CELL 1 — IMPORTS + REPRODUCIBILITY + DEVICE
2. CELL 2 — CONFIGURATION
3. CELL 3 — LOAD TOKENIZER + TOKENIZE WITH CHARACTER OFFSETS
4. CELL 4 — DOWNLOAD PHASE-2 DATA  (needs the Kaggle secret "HF")
5. CELL 5 — LOAD TEXTS, LABELS AND GOLD ENTITIES
6. CELL 6 — TAGS (BIO), WORD FEATURES, SPANS, STRICT F1
7. CELL 7 — TOKENIZE ALL DOCUMENTS (tokens -> words) + WINDOWS
8. CELL 8 — OFFSET SELF-TEST  (run before training!)
9. CELL 9 — BATCHES
10. CELL 10 — MODELS
11. CELL 11 — LOAD BOTH PHASE-1 CHECKPOINTS + MODEL BUILDERS
12. CELL 12 — PREDICTION + EVALUATION
13. CELL 13 — TRAINING, TEACHER TARGETS, SUBMISSION HELPERS
14. CELL 14 — E4 STAGE A: TEACHER (Architecture-4, partial fine-tuning)
15. CELL 15 — E5 STAGE A: STUDENT (Architecture-2) WITH KNOWLEDGE DISTILLATION
16. CELL 16 — TEACHER vs STUDENT (for the report) + SAVE RESULTS
17. CELL 17 — STAGE B TEACHER (fold1 + fold2) -> E4 SUBMISSION
18. CELL 18 — STAGE B STUDENT (fold1 + fold2, distilled) -> E5 SUBMISSION
