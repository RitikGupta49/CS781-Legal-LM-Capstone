# Phase 1 · Architecture-3 — E-LegalFormer (grouped-query attention)

| | |
|---|---|
| Model | 125.3M parameters |
| Codabench (1,000 test prompts) | BERTScore F1 **0.8196** · ROUGE-L F1 0.1630 · BLEU-4 0.0192 |
| Own validation | best validation loss 1.2228 (perplexity 3.40) |
| Notebook | `p1_arch3_elegalformer.ipynb` (original name: `proj-cap-p-1-archi-3.ipynb`) |
| Inputs | outputs of `tokenizer/01_bpe_tokenizer.ipynb` and `tokenizer/02_tokenize_corpus.ipynb` |

## Architecture and training

* 16 decoder blocks, d_model 768, grouped-query attention with 12 query heads sharing 4 key/value heads, RoPE, SwiGLU FFN 2048, RMSNorm, tied LM head, context 2,048.
* Data: 200M-token budget split 15/35/25/25 over the four sources, 98 : 2 train/validation split, memory-mapped token files.
* Training: AdamW (wd 0.1 on matrices only), LR 3e-4 → 3e-5 with 3% warm-up, micro-batch 1 × 8 accumulation × 2,048 tokens, up to 8,000 optimizer steps, mixed precision, auto-resume from the newest checkpoint.
* Generation: KV cache (4 KV heads cached), temperature 0.7, top-k 50 / top-p 0.95, repetition penalty 1.1, resumable over the 1,000 test prompts with retries for short outputs.

## Configuration (CELL 2 of the notebook)

```python
DEBUG = False
SEED = 42
VOCAB_SIZE = 32004
BLOCK_SIZE = 2048
TOKEN_DTYPE = np.uint16
FULL_TOTAL_TOKENS = 200_000_000
DEBUG_TOTAL_TOKENS = 2_000_000
TOTAL_TOKEN_BUDGET = (
    DEBUG_TOTAL_TOKENS
    if DEBUG
    else FULL_TOTAL_TOKENS
)
SOURCE_RATIOS = {
    "acts": 15,
    "high_courts": 35,
    "supreme_court": 25,
    "tribunals": 25,
}
TRAIN_RATIO = 0.98
TEST_RATIO = 0.02
SOURCE_FILES = {
    "acts":
        TOKENIZED_DIR / "acts_100m.bin",

    "high_courts":
        TOKENIZED_DIR / "high_courts_100m.bin",

    "supreme_court":
    ...
RUN_NAME = (
    "debug"
    if DEBUG
    else "full"
)
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()
BATCH_SIZE = 1
MICRO_BATCH_SIZE = 1
TRAIN_NUM_WORKERS = 2
VAL_NUM_WORKERS = 2
```

## Notebook cells

1. MODEL 3 — DATASET CONFIGURATION
2. VERIFY SOURCE FILES
3. Merges the replica into the existing directory, overwriting matching files
4. MEMORY-MAPPED TOKEN SOURCE
5. LOAD ALL FOUR SOURCES
6. SOURCE TOKEN BUDGET
7. 98 : 2 SPLIT
8. CREATE MERGED DATASETS
9. CAUSAL LM DATASET
10. ARCHITECTURE 3
11. MODEL 3 — TRAINING CONFIG
12. CONFIGURATION VALIDATION
13. DATASETS
14. RMS NORMALIZATION
15. ROPE
16. GROUPED-QUERY ATTENTION
17. SWIGLU FEED-FORWARD NETWORK
18. EFFICIENT TRANSFORMER BLOCK
19. E-LEGALFORMER
20. CREATE MODEL 3
21. PARAMETER COUNT
22. FORWARD PASS TEST
23. MODEL 3 — TRAINING CONFIGURATION
24. OPTIMIZER PARAMETER GROUPS
25. TRAINING STEP CALCULATION
26. LEARNING RATE SCHEDULER
27. AUTOMATIC MIXED PRECISION
28. LANGUAGE MODEL LOSS
29. SINGLE TRAINING STEP
30. VALIDATION
31. PERPLEXITY
32. CHECKPOINT
33. LOAD CHECKPOINT
34. TRAINING HISTORY
35. MODEL 3 — AUTO RESUME / START FROM SCRATCH
36. MODEL 3 — RESUMABLE TRAINING LOOP
37. SAVE FINAL MODEL
38. TRAINING CURVES
39. LOAD TRAINED BPE TOKENIZER
40. MODEL 3 GENERATION — KV CACHE  (FIXED)
41. LOAD TRAINED MODEL 3
42. SINGLE PROMPT TEST GENERATION
43. Hugging Face dataset
44. RESUMABLE TEST GENERATION — SETUP
45. RESUMABLE FULL TEST GENERATION  (FIXED)
46. VALIDATE GENERATED PREDICTIONS  (FIXED)
47. CREATE FINAL prediction.json + submission.zip  (FIXED)
