# Phase 6 · Submission 05 — Knowledge distillation: Architecture-4 + LoRA teacher → Architecture-3 + LoRA student (token-level)

| | |
|---|---|
| Phase-1 backbone | Architecture-3 E-LegalFormer, GQA (125M) |
| Adaptation | teacher: LoRA on Architecture-4 (6.29M); student: LoRA on Architecture-3 (3.47M) |
| Codabench | not yet on the results page — add the Codabench score here |
| Own dev / validation | held-out dev (student): ROUGE-1 0.4385 · ROUGE-2 0.1718 · ROUGE-L 0.2357 (520 words; EOS penalty 4.0, length × 0.8) |
| Notebook | `p6_kd_arch4_to_arch3.ipynb` (original name: `p6-kd-arch4-to-arch3.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p6-archi-4-3-KDsubmission.zip`), contains `prediction.json` at the root |


## Method

* Step 1: teacher = Architecture-4 + LoRA trained 1 epoch on the reference summary parts (dev loss 2.010 → 1.797).
* Step 2: student = Architecture-3 + LoRA; loss = 0.5 · CE(reference tokens) + 0.5 · T²·KL(teacher ‖ student) over the same target token positions, T = 1 (the teacher runs alongside without gradients).
* Both models use the same tokenizer and the same segment / target, so the distributions line up token by token.

The data handling, head and evaluation shared by all Phase-6 notebooks are described in the [phase README](../README.md).

## Notes

* Student dev loss 1.181 vs teacher 1.797: the student ends up far better than its teacher, so most of the gain comes from the reference tokens.

## Model size (printed by the notebook)

```
STEP 1 | teacher LoRA weights: 6,291,456
STEP 2 | student LoRA weights: 3,473,408 | KD alpha 0.5, T 1.0
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p6_kd_arch4_to_arch3"
SEGMENT_TOKENS = 500         # document tokens per segment
TARGET_TOKENS = 384          # longest target (summary part) per segment in training; longer ones are cut
MAX_SEGMENTS = 48            # summarized segments per document (longer documents: 48 spread over it)
EMPTY_KEEP = 0.3             # share of segments with an EMPTY target kept for training (the model learns to skip)
SEPARATOR = "\n\nSummary:\n"
KD_ALPHA = 0.5               # weight of the teacher term (0 = no distillation)
KD_TEMPERATURE = 1.0
TEACHER_TARGETS = ["encoder_attention.q_proj", "encoder_attention.k_proj", "encoder_attention.v_proj",
                   "encoder_attention.out_proj", "causal_attention.q_proj", "causal_attention.k_proj",
                   "causal_attention.v_proj", "causal_attention.out_proj", "hybrid_fusion.gate_proj",
                   "hybrid_fusion.out_proj", "ffn.gate_proj", "ffn.up_proj", "ffn.down_proj"]
STUDENT_TARGETS = ["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.o_proj",
                   "ffn.gate_proj", "ffn.up_proj", "ffn.down_proj"]
PEFT_RANK = 16
PEFT_ALPHA = 32
PEFT_DROPOUT = 0.05
LR_PEFT = 2e-4
TEACHER_EPOCHS = 1           # teacher: reference summaries only
STUDENT_EPOCHS = 2           # student: reference summaries + teacher (the teacher runs alongside, no gradients)
EPOCH_EXAMPLES = 16000       # (segment, target) examples per epoch (all used before any repeats)
BATCH_SIZE = 4               # sequences per forward pass (each <= ~1.7k tokens). If CUDA out-of-memory: 2
GRAD_ACCUM = 4               # -> 16 sequences per weight update
WARMUP_RATIO = 0.03
GRAD_CLIP = 1.0
DEV_DOCS = 300               # train documents held out: dev loss after every epoch (picks the best epoch)
DEV_LOSS_EXAMPLES = 1000     # dev examples for the dev loss
GEN_DEV_DOCS = 60            # held-out documents summarized at the end: pick the length + ROUGE report
GEN_BATCH = 24               # segments decoded together. If CUDA out-of-memory: 12
MAX_NEW_TOKENS = 384         # per segment (hard limit)
PART_BUDGET = 1.5            # a segment may generate up to 1.5 x its share of the expected summary
NO_REPEAT_NGRAM = 4          # a sequence of 4 non-space tokens (~3 words) is never generated twice in a part
EOS_PENALTIES = [0.0, 2.0, 4.0]   # subtracted from the <eos> score: > 0 = longer parts (greedy decoding tends to stop
LENGTH_SCALES = [0.6, 0.8, 1.0, 1.25, None]     # x the expected summary length (None = no trimming), picked on dev
DEBUG_MODE = False
DEBUG_DOCS = 120
USE_AMP = DEVICE.type == "cuda"
USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16
```

## Notebook cells

1. CELL 1 — IMPORTS + REPRODUCIBILITY + DEVICE
2. CELL 2 — CONFIGURATION
3. CELL 3 — LOAD THE PHASE-1 TOKENIZER + FAST ENCODING
4. CELL 4 — PHASE-6 DATA FILES  (download needs the Kaggle secret "HF")
5. CELL 5 — LOAD DOCUMENTS + SUMMARIES, HOLD OUT DEV DOCUMENTS, EXPECTED SUMMARY LENGTH
6. CELL 6 — SENTENCES -> SEGMENTS -> WHICH SUMMARY SENTENCES BELONG TO WHICH SEGMENT
7. CELL 7 — ROUGE-1 / ROUGE-2 / ROUGE-L F1 (lower-case words; LCS with a fast bit-parallel algorithm)
8. CELL 8 — MODELS: TEACHER (Architecture-4) + STUDENT (Architecture-3), LoRA, distillation loss,
9. CELL 9 — LOAD BOTH PHASE-1 CHECKPOINTS + BUILDERS
10. CELL 10 — SUMMARIZE DOCUMENTS
11. CELL 11 — TRAINING FUNCTION (teacher or student)
12. CELL 12 — STEP 1: TEACHER (Architecture-4 + LoRA) on the reference summaries
13. CELL 13 — DEV REPORT (the STUDENT, held-out train documents) + TEACHER vs STUDENT
14. CELL 14 — SUMMARIZE THE TEST DOCUMENTS WITH THE STUDENT + WRITE prediction.json + submission.zip
