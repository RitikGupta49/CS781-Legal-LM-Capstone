# Phase 1 · Architecture-2 — Transformer encoder-decoder (1,024 tokens → next 512)

| | |
|---|---|
| Model | 149.6M parameters |
| Codabench (1,000 test prompts) | BERTScore F1 **0.8289** · ROUGE-L F1 0.1858 · BLEU-4 0.0329 |
| Own validation | validation loss 1.4996 (perplexity 4.48) on the 512 target tokens after 34,000 steps |
| Notebook | `p1_arch2_encoder_decoder.ipynb` (original name: `proj-cap-p-1-archi-2.ipynb`) |
| Inputs | outputs of `tokenizer/01_bpe_tokenizer.ipynb` and `tokenizer/02_tokenize_corpus.ipynb` |

## Architecture and training

* 6 encoder + 6 decoder layers, d_model 768, 12 heads, GELU FFN 3072, pre-LayerNorm, learned absolute positions; decoder embedding tied to the LM head.
* Objective: the encoder reads 1,024 tokens of a document, the decoder predicts the next 512 tokens (teacher forcing, loss on the target only).
* Training: AdamW (β 0.9/0.95, wd 0.1), LR 3e-4 → 3e-5, effective batch 16 windows of 1,024 + 512 tokens, 34,000 steps (token budget 800M), dropout 0.1, fp16 + GradScaler, resumable checkpoints.
* Generation: top-k 50 / top-p 0.95, temperature 0.8, repetition penalty 1.15 over the last 64 tokens, no repeated 12-token n-grams.

## Configuration (CELL 2 of the notebook)

```python
DEBUG_MODE = False
SOURCES = ["acts", "high_courts", "supreme_court", "tribunals"]
TOKEN_FILE_PATTERN = "{source}_100m.bin"     # written by updated-tokenize-data
TOKEN_DTYPE = np.uint16                      # that notebook saved uint16
VOCAB_SIZE = 32004
MIXING_RATIOS = {
    "acts": 0.15,
    "high_courts": 0.35,
    "supreme_court": 0.25,
    "tribunals": 0.25,
}
VAL_FRACTION = 0.02
MAX_TRAIN_TOKENS = 800_000_000
TRAIN_TOKEN_BUDGET = {
    source: int(MAX_TRAIN_TOKENS * MIXING_RATIOS[source])
    for source in SOURCES
}
DROPOUT = 0.1
ENCODER_LAYERS = N_LAYERS
DECODER_LAYERS = N_LAYERS
WINDOW_LEN = SRC_LEN + TGT_LEN
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRAD_ACCUM_STEPS
LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
EPSILON = 1e-8
GRAD_CLIP = 1.0
USE_GRADIENT_CHECKPOINTING = False
TIME_LIMIT_HOURS = 11.0
RESUME_TRAINING = False
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_SCALER = AMP_DTYPE == torch.float16
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 50
GEN_TOP_P = 0.95
GEN_TARGET_WORDS = 128
```

## Notebook cells

1. PHASE 1 — ARCHITECTURE 2 (ENCODER-DECODER)
2. CELL 2 — LOAD THE SAVED V3 BPE TOKENIZER
3. CELL 3 — LOAD TOKENIZED CORPUS (uint16) INTO RAM
4. CELL 4 — TRAIN / VALIDATION SPLIT PER SOURCE
5. CELL 5 — BATCH SAMPLER (no DataLoader needed)
6. CELL 6 — MULTI-HEAD ATTENTION (unchanged)
7. CELL 7 — POSITIONAL EMBEDDING + FEED-FORWARD (unchanged)
8. CELL 8 — ENCODER BLOCK + DECODER BLOCK (Pre-LN, unchanged)
9. CELL 9 — FULL ENCODER + FULL DECODER (unchanged)
10. CELL 10 — ENCODER-DECODER LM + WEIGHT INIT
11. CELL 11 — OPTIMIZER + LR SCHEDULE + GRAD SCALER
12. CELL 12 — VALIDATION + CHECKPOINT UTILITIES
13. CELL 13 — ONE-STEP MEMORY / SPEED CHECK
14. CELL 14 — TRAINING LOOP
15. CELL 15 — LOAD BEST CHECKPOINT + VERIFY
16. CELL 16 — GENERATION (top-k / top-p + anti-repetition)
17. CELL 17 — GENERATION TEST
