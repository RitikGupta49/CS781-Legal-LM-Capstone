# Phase 6 · Submission 03 — Architecture-2 encoder-decoder + Houlsby adapters, segment-wise summarization

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | Houlsby adapters (dim 64) after self_attention.out_proj and feed_forward.fc2 in every encoder and decoder layer (2.38M weights, 1.59%) |
| Codabench | BERTScore F1 **0.8162** · ROUGE-1 0.416 · ROUGE-L 0.223  (rank 4 of 5 in this phase) |
| Own dev / validation | held-out dev: ROUGE-1 0.3976 · ROUGE-2 0.1293 · ROUGE-L 0.2208 (513 words; EOS penalty 4.0, length × 0.8) |
| Notebook | `p6_arch2_houlsby_adapters.ipynb` (original name: `p6-arch2-houlsby-adapters.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p6_arch2_houlsby_adapters-submission.zip`), contains `prediction.json` at the root |


## Method

* The encoder reads the segment (≤ 1,024 tokens), the decoder writes the segment's summary part (≤ 510 tokens) — the same 1,024 → 512 shape as Phase-1 pretraining.
* 2 epochs × 16,000 examples; dev loss 1.786 → 1.618 → 1.610.

The data handling, head and evaluation shared by all Phase-6 notebooks are described in the [phase README](../README.md).

## Notes

* Generates nothing without an EOS penalty; repeated 4-grams 12.2% (reference 4.3%).

## Model size (printed by the notebook)

```
Phase-1 model (frozen): 149,578,752 | Houlsby adapters (trainable): 2,379,264 (1.59%)
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p6_arch2_houlsby_adapters"
SEGMENT_TOKENS = 1024        # document tokens per segment
TARGET_TOKENS = 510          # longest target (summary part) per segment in training; longer ones are cut
MAX_SEGMENTS = 32            # summarized segments per document (longer documents: 32 spread over it)
EMPTY_KEEP = 0.3             # share of segments with an EMPTY target kept for training (the model learns to skip)
SEPARATOR = "\n\nSummary:\n"           # (not used by the encoder-decoder)
PEFT_TARGETS = ["self_attention.out_proj", "feed_forward.fc2"]    # in every encoder and decoder layer
ADAPTER_DIM = 64
PEFT_DROPOUT = 0.05
LR_PEFT = 5e-4
EPOCHS = 2                   # Arch-1 run (LoRA): most of the gain in epoch 1
EPOCH_EXAMPLES = 16000       # (segment, target) examples per epoch (all used before any repeats)
BATCH_SIZE = 4               # sequences per forward pass (each <= ~1.7k tokens). If CUDA out-of-memory: 2
GRAD_ACCUM = 4               # -> 16 sequences per weight update
WARMUP_RATIO = 0.03
GRAD_CLIP = 1.0
DEV_DOCS = 300               # train documents held out: dev loss after every epoch (picks the best epoch)
DEV_LOSS_EXAMPLES = 1000     # dev examples for the dev loss
GEN_DEV_DOCS = 60            # held-out documents summarized at the end: pick the length + ROUGE report
GEN_BATCH = 32               # segments decoded together. If CUDA out-of-memory: 16
MAX_NEW_TOKENS = 511         # per segment (hard limit)
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
8. CELL 8 — MODEL: Architecture-2 encoder-decoder (+ Houlsby adapters), batched generation with a KV cache
9. CELL 9 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-2 CHECKPOINT (+ Houlsby adapters)
10. CELL 10 — SUMMARIZE DOCUMENTS
11. CELL 11 — TRAINING FUNCTION
12. CELL 12 — TRAIN (Houlsby adapters; train documents only)
13. CELL 13 — DEV REPORT (held-out train documents)
14. CELL 14 — SUMMARIZE THE TEST DOCUMENTS + WRITE prediction.json + submission.zip
