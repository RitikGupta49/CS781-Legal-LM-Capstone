# Phase 7 · Submission 03 — Architecture-2 encoder + full fine-tuning retriever, fused with the learned re-ranker

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | full fine-tuning of the Architecture-2 encoder (all 67.9M weights; the decoder is not used), LR 2e-5 + retriever head (0.59M) |
| Codabench | Micro-F1@7 **0.4696** · Micro-P@7 0.4792 · Micro-R@7 0.4603  (rank 4 of 6 in this phase) |
| Own dev / validation | dev micro-F1@7: fused 0.4747 (model weight 0.5) · model alone 0.2668 · re-ranker 0.4725 · BM25 0.4647 |
| Notebook | `p7_arch2_full_ft.ipynb` (original name: `p7-archi-2-ft.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p7-archi-2-FT-submission.zip`), contains `prediction.json` at the root |


## Method

* The bidirectional encoder reads every citation context / passage (≤ 512 tokens); its token vectors feed the same retriever head.
* Contrastive training (InfoNCE, τ 0.05) on the 801 train queries with a reachable relevant case: 8 queries per update, each with 1 relevant candidate + 3 BM25 hard negatives (from the BM25 top 30) + all other candidates of the update; a candidate shows its 4 best BM25 passages for that query; GradCache keeps the whole update in memory; the best epoch is the one with the best model-alone dev F1@7.
* 6 epochs × 101 updates; 29 min on a T4, peak GPU 5.2 GB.
* Model alone on dev by epoch: 0.2023 → 0.1979 → 0.2324 → 0.2435 → 0.2412 → **0.2457** → 0.2435 (epoch 5 kept).
* Test ranking = 0.5 · z(model) + 0.5 · z(re-ranker), the weight picked on dev.

The data handling, head and evaluation shared by all Phase-7 notebooks are described in the [phase README](../README.md).

## Notes

* The fastest retriever: 264 chunks/s on dev, 3.5× Architecture-1, because only the 6-layer encoder runs.

## Model size (printed by the notebook)

```
Phase-1 model             : 67,894,272 (6 blocks)
trained in it (full fine-tuning): 67,894,272  (100.00%)
head (trainable)          : 590,599
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p7_arch2_full_ft"
CITATION_WINDOW = 1          # sentences kept before / after a <CITATION> sentence
MAX_CONTEXTS = 48            # citation contexts per query (more: evenly spread)
PASSAGE_WORDS = 200          # words per candidate passage (~ 480 tokens with the Phase-1 tokenizer)
MAX_CHUNKS = 16              # passages per candidate for ranking (more: evenly spread over the case)
CHUNK_LEN = 512              # tokens per chunk (a longer passage / context is cut)
DATE_FILTER = True           # never rank a candidate decided after the query
POOL_SKIP_SPACES = True      # whitespace tokens stay in the input but are not pooled
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LR_BACKBONE = 2e-5           # every encoder weight moves: small learning rate
BACKBONE_DROPOUT = 0.0       # 0: both GradCache passes must compute the same vectors
TEMPERATURE = 0.05           # cosine / TEMPERATURE in the contrastive loss
LR_HEAD = 5e-4               # layer mix + projection
REUSE_TRAINED = False        # True: load retriever.pt of an earlier run of THIS notebook (attach that
EPOCHS = 6
BATCH_QUERIES = 8            # queries per weight update
HARD_NEGATIVES = 3           # BM25 hard negatives per query (+ the other candidates of the batch)
HARD_POOL = 30               # hard negatives are drawn from the BM25 top 30 (relevant / filtered cases removed)
TRAIN_CONTEXTS = 8           # citation contexts per query in a training step (random subset)
TRAIN_CHUNKS = 4             # passages per candidate in a training step: its best BM25 passages for that query
EPOCH_EVAL_CHUNKS = 8        # passages per candidate for the dev check after every epoch (faster than MAX_CHUNKS)
EMBED_GROUP_CHUNKS = 16      # chunks per forward pass WITH gradients (more = a little faster, more memory)
EVAL_CHUNKS = 64             # chunks per forward pass without gradients
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
PATIENCE = 3                 # stop if the model's dev F1@7 does not improve for 3 epochs
STATS_DOCS = 300             # train candidates for the token statistics (standardization) before training
TOP_K = 7                                                       # the leaderboard reads the first 7 candidates
FUSION_WEIGHTS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]   # weight of the MODEL score (rest = re-ranker), picked on dev
RERANK_TOP = 100             # the re-ranker scores the BM25 top 100 of every query (the rest is never predicted)
RERANK_C = 1.0               # inverse regularization strength of the logistic regression (0.01 - 10 all give the same F1)
BM25_K1 = 1.2
BM25_B = 0.75
DEBUG_MODE = False
DEBUG_QUERIES = 40           # per split
DEBUG_EXTRA_CANDIDATES = 300 # random candidates kept in every pool (+ the relevant ones)
USE_AMP = DEVICE.type == "cuda"
USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16
```

## Notebook cells

1. CELL 1 — IMPORTS + REPRODUCIBILITY + DEVICE
2. CELL 2 — CONFIGURATION
3. CELL 3 — LOAD THE PHASE-1 TOKENIZER + FAST ENCODING
4. CELL 4 — PHASE-7 DATA FILES  (download needs the Kaggle secret "HF")
5. CELL 5 — LOAD QUERIES + CANDIDATE POOLS + YEARS (date filter) + STATISTICS
6. CELL 6 — QUERY CITATION CONTEXTS + CANDIDATE PASSAGES (text)
7. CELL 7 — BM25 SCORES (own implementation) WITH THE SAME STRUCTURE AS THE MODEL
8. CELL 8 — TOKENIZE: one chunk per citation context and per candidate passage
9. CELL 9 — RANKING + MICRO-F1@7 (same rule as the leaderboard)
10. CELL 10 — LEARNED RE-RANKER (logistic regression, fitted on TRAIN queries only)
11. CELL 11 — MODEL
12. CELL 12 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-2 CHECKPOINT
13. CELL 13 — MODEL SCORES + EVALUATION ON A SPLIT
14. CELL 14 — CONTRASTIVE TRAINING FUNCTION (GradCache: big batches with little memory)
15. CELL 15 — BEFORE TRAINING: TOKEN STATISTICS + GRADCACHE CHECK
16. CELL 16 — TRAIN (TRAIN queries only; the model's own dev F1@7 picks the best epoch)
17. CELL 17 — DEV REPORT (all passages) + FUSION WEIGHT FOR THE TEST RANKING
18. CELL 18 — RANK THE TEST POOL + WRITE prediction.json + submission.zip
