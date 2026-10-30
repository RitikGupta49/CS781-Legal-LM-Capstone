# Phase 6 · Submission 01 — Architecture-1 decoder + LoRA, segment-wise summarization

| | |
|---|---|
| Phase-1 backbone | Architecture-1 decoder-only (110M) |
| Adaptation | LoRA, rank 16, α 32, on qkv / out_proj / gate / up / down (2.51M weights, 2.29%) |
| Codabench | BERTScore F1 **0.8385** · ROUGE-1 0.524 · ROUGE-L 0.255  (rank 2 of 5 in this phase) |
| Own dev / validation | held-out dev: ROUGE-1 0.5019 · ROUGE-2 0.2452 · ROUGE-L 0.2545 (675 words; EOS penalty 2.0, length × 1.25) |
| Notebook | `p6_arch1_lora.ipynb` (original name: `p6-archi1-lora.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p6-archi1-lora-submission.zip`), contains `prediction.json` at the root |


## Method

* Shared Phase-6 pipeline: 1,024-token segments, target ≤ 640 tokens per segment, prompt `<eos> segment "\n\nSummary:\n"`.
* 2 epochs × 12,000 (segment, target) examples; dev loss 1.134 (no training) → 0.790 → 0.776.

The data handling, head and evaluation shared by all Phase-6 notebooks are described in the [phase README](../README.md).

## Notes

* Without an EOS penalty the model stops very early (ROUGE-1 0.10, 51 words); the penalty search on dev fixes it.

## Model size (printed by the notebook)

```
Phase-1 decoder (frozen): 109,534,464 | LoRA (trainable): 2,506,752 (2.29%)
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p6_arch1_lora_v2"
SEGMENT_TOKENS = 1024        # document tokens per segment
TARGET_TOKENS = 640          # longest target (summary part) per segment in training; longer ones are cut
MAX_SEGMENTS = 32            # summarized segments per document (longer documents: 32 spread over it)
EMPTY_KEEP = 0.3             # share of segments with an EMPTY target kept for training (the model learns to skip)
SEPARATOR = "\n\nSummary:\n"
PEFT_TARGETS = ["attn.qkv", "attn.out_proj", "mlp.gate", "mlp.up", "mlp.down"]    # every linear layer of every block
PEFT_RANK = 16
PEFT_ALPHA = 32
PEFT_DROPOUT = 0.05
LR_PEFT = 2e-4
EPOCHS = 2                   # run 1 (LoRA): dev loss 1.13 -> 0.80 after epoch 1, small gains after that
EPOCH_EXAMPLES = 12000       # (segment, target) examples per epoch, ~20 min on a T4 (all used before any repeats)
BATCH_SIZE = 4               # sequences per forward pass (each <= ~1.7k tokens). If CUDA out-of-memory: 2
GRAD_ACCUM = 4               # -> 16 sequences per weight update
WARMUP_RATIO = 0.03
GRAD_CLIP = 1.0
DEV_DOCS = 300               # train documents held out: dev loss after every epoch (picks the best epoch)
DEV_LOSS_EXAMPLES = 1000     # dev examples for the dev loss
GEN_DEV_DOCS = 60            # held-out documents summarized at the end: pick the length + ROUGE report
GEN_BATCH = 24               # segments decoded together. If CUDA out-of-memory: 12
MAX_NEW_TOKENS = 512         # per segment (hard limit)
PART_BUDGET = 1.5            # a segment may generate up to 1.5 x its share of the expected summary
NO_REPEAT_NGRAM = 4          # a sequence of 4 non-space tokens (~3 words) is never generated twice in a part
EOS_PENALTIES = [0.0, 2.0]   # subtracted from the <eos> score: > 0 = longer parts (greedy decoding tends to stop
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
8. CELL 8 — MODEL: Architecture-1 decoder LM (+ LoRA) with a KV cache for batched generation
9. CELL 9 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-1 CHECKPOINT (+ LoRA)
10. CELL 10 — SUMMARIZE DOCUMENTS
11. CELL 11 — TRAINING FUNCTION
12. CELL 12 — TRAIN (LoRA; train documents only)
13. CELL 13 — DEV REPORT (held-out train documents)
14. CELL 14 — SUMMARIZE THE TEST DOCUMENTS + WRITE prediction.json + submission.zip
