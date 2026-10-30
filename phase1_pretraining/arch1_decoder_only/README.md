# Phase 1 · Architecture-1 — Llama-style decoder-only LM

| | |
|---|---|
| Model | 109.5M parameters (tied embeddings) |
| Codabench (1,000 test prompts) | BERTScore F1 **0.8238** · ROUGE-L F1 0.1679 · BLEU-4 0.0256 |
| Own validation | validation loss 1.0066 (perplexity 2.74) after 6,000 steps |
| Notebook | `p1_arch1_decoder_only.ipynb` (original name: `decoder-only-archi-1.ipynb`) |
| Inputs | outputs of `tokenizer/01_bpe_tokenizer.ipynb` and `tokenizer/02_tokenize_corpus.ipynb` |

## Architecture and training

* 12 decoder blocks, d_model 768, 12 heads, SwiGLU FFN 2048, RMSNorm (pre-norm), RoPE (θ = 10,000), QK-norm, tied LM head, context 2,048.
* Data: `uint16` token files of the four sources mixed 15% acts / 35% high courts / 25% supreme court / 25% tribunals; 1% of the blocks of every source held out for validation.
* Training: AdamW (β 0.9/0.95, weight decay 0.1), LR 6e-4 → 6e-5 cosine, 300 warm-up steps, batch 4 × 8 accumulation × 2,048 tokens, 6,000 steps (≈393M tokens), gradient clip 1.0, fp16 + GradScaler on a T4, time limit so the run saves before Kaggle's 12 h cut-off.
* Generation: KV cache, top-k 50 / top-p 0.95, temperature 0.8, repetition penalty 1.1, special tokens banned; target length 128 words.

## Configuration (CELL 2 of the notebook)

```python
RESUME_FROM = None
SOURCES = ["acts", "high_courts", "supreme_court", "tribunals"]
MIXING_RATIOS = {
    "acts": 0.15,
    "high_courts": 0.35,
    "supreme_court": 0.25,
    "tribunals": 0.25,
}
BLOCK_SIZE = 2048     # context length
VAL_RATIO = 0.01      # 1% of blocks of every source are kept for validation
MODEL_CONFIG = {
    "vocab_size": None,        # filled in from the tokenizer (32004)
    "block_size": BLOCK_SIZE,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "d_ff": 2048,              # SwiGLU hidden size (≈ 8/3 × d_model)
    "rope_theta": 10000.0,
    ...
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 8
MAX_TRAIN_STEPS = 6000
LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WARMUP_STEPS = 300
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
LOG_INTERVAL = 10
EVAL_INTERVAL = 250
MAX_VAL_BATCHES = 50
TIME_LIMIT_HOURS = 10.75    # stop + save before Kaggle's 12h limit
USE_COMPILE = False         # True = ~20-40% faster, but compiling takes a few minutes
NUM_WORKERS = 2
DEBUG_MODE = False
GEN_TARGET_WORDS = 128
GEN_MAX_NEW_TOKENS = 768
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 50
GEN_TOP_P = 0.95
GEN_REPETITION_PENALTY = 1.1
USE_AMP = DEVICE.type == "cuda"
USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16
TOKENS_PER_STEP = BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE
```

## Notebook cells

1. CELL 1 — IMPORTS + REPRODUCIBILITY + DEVICE
2. CELL 2 — CONFIGURATION
3. CELL 3 — LOAD TOKENIZER (same tokenizer.py as the tokenizer notebook)
4. CELL 4 — LOAD TOKENIZED DATA (.bin files, uint16)
5. CELL 5 — TRAIN / VALIDATION SPLIT (by blocks of BLOCK_SIZE)
6. CELL 6 — MIXED TRAINING ORDER + VALIDATION INDEX
7. CELL 7 — DATASET + VALIDATION DATALOADER
8. CELL 8 — RMSNORM + ROTARY POSITION EMBEDDING (RoPE)
9. CELL 9 — CAUSAL SELF-ATTENTION (with RoPE, QK-norm, KV cache)
10. CELL 10 — SWIGLU FEED-FORWARD + TRANSFORMER BLOCK
11. CELL 11 — DECODER-ONLY LANGUAGE MODEL
12. CELL 12 — CREATE + VERIFY MODEL
13. CELL 13 — OPTIMIZER + LR SCHEDULE + GRAD SCALER
14. CELL 14 — VALIDATION + CHECKPOINT FUNCTIONS
15. CELL 15 — RESUME FROM CHECKPOINT (if available)
16. CELL 16 — TRAINING LOOP
17. CELL 17 — LOAD BEST CHECKPOINT
18. CELL 18 — GENERATION (with KV cache)
19. CELL 19 — MANUAL GENERATION TEST
20. CELL 20 — DOWNLOAD PHASE-1 TEST DATA
21. CELL 21 — GENERATE TEST PREDICTIONS
22. CELL 22 — VALIDATE + SAVE prediction.json + SUBMISSION ZIP
