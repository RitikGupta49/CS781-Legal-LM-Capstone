# Phase 1 · Architecture-4 — Hybrid prefix-LM (bidirectional prefix + causal branch, gated fusion)

| | |
|---|---|
| Model | 205.5M parameters |
| Codabench (1,000 test prompts) | BERTScore F1 **0.8283** · ROUGE-L F1 0.1945 · BLEU-4 0.0355 |
| Own validation | best validation loss 1.2631 (perplexity 3.54) |
| Notebook | `p1_arch4_hybrid_prefix_lm.ipynb` (original name: `proj-cap-p-1-archi-4 (1).ipynb`) |
| Inputs | outputs of `tokenizer/01_bpe_tokenizer.ipynb` and `tokenizer/02_tokenize_corpus.ipynb` |

## Architecture and training

* 16 hybrid blocks, d_model 768, 12 heads, SwiGLU FFN 2048, RMSNorm, learned positions, tied LM head, context 2,048.
* Every block has two attention branches: an encoder branch with bidirectional attention over the first 512 tokens (the prefix) and a causal branch over the whole sequence; a learned sigmoid gate mixes them (`GatedHybridFusion`). A leakage test checks that no token sees the future outside the prefix.
* Training: AdamW (β 0.9/0.95, wd 0.1), LR 3e-4 → 3e-5 cosine with 200 warm-up steps, batch 1 × 4 accumulation × 2,048 tokens, up to 30,000 steps, checkpoints every 250 steps.
* Generation: batched KV-cache decoding (checked against the slow generator), temperature 0.8, top-k 50 / top-p 0.95, repetition penalty 1.15 over 256 tokens.

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
9. CELL 8 — CAUSAL LM DATASET
10. ARCHITECTURE 4 — HYBRID PREFIX ENCODER LANGUAGE MODEL
11. CELL 10 — RMSNorm
12. CELL 11 — SDPA BIDIRECTIONAL PREFIX ENCODER ATTENTION
13. CELL 12 — SDPA CAUSAL SELF-ATTENTION
14. CELL 13 — GATED HYBRID FUSION
15. CELL 14 — SwiGLU FEED-FORWARD NETWORK
16. CELL 15 — HYBRID TRANSFORMER BLOCK
17. CELL 16 — COMPLETE HYBRID LANGUAGE MODEL
18. CELL 17 — MODEL INSTANTIATION + PARAMETER COUNT
19. CELL 18 — FORWARD PASS + SHAPE VERIFICATION
20. CELL 19 — CAUSAL / FUTURE-TOKEN LEAKAGE TEST
21. CELL 20 — SINGLE TRAINING STEP
22. CELL 21 — CREATE DATALOADERS FROM EXISTING DATASET
23. CELL 22 — ONE REAL LEGAL TRAINING STEP
24. CELL 23 — TRAINING CONFIG + CHECKPOINT SYSTEM
25. CELL 24 — VALIDATION + TRAINING UTILITIES
26. CELL 25 — CHECKPOINT-BASED FULL TRAINING
27. CELL 26 — LOAD BEST MODEL FOR EVALUATION
28. Hugging Face dataset
29. LOAD TRAINED BPE TOKENIZER
30. TEST GENERATION — SETTINGS
31. CELL 34 — VERIFY GENERATION ENVIRONMENT
32. CELL 35 — HYBRIDLM AUTOREGRESSIVE GENERATION
33. CELL 35b — FAST GENERATION: KV CACHE + BATCHED DECODING
34. CELL 35c — CHECK FAST GENERATOR AGAINST THE SLOW ONE
35. CELL 36 — PREVIEW GENERATION ON 10 TEST EXAMPLES
36. CELL 37 — RESUMABLE GENERATION STATE
37. CELL 38 — FULL RESUMABLE TEST GENERATION (FAST, BATCHED)
38. CREATE FINAL prediction.json
39. CELL 39 — VERIFY FINAL prediction.json
