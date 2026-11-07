# Phase 7 · Submission 05 — Architecture-4 hybrid prefix-LM + partial (layer-wise) fine-tuning retriever, fused with the learned re-ranker

| | |
|---|---|
| Phase-1 backbone | Architecture-4 hybrid prefix-LM (205M) |
| Adaptation | partial fine-tuning: all weights of the top 4 of 16 blocks (44.8M, 21.8%), LR 3e-5 for the top block × 0.8 per block below + retriever head (0.59M) |
| Codabench | Micro-F1@7 **0.4684** · Micro-P@7 0.478 · Micro-R@7 0.4592  (rank 5 of 6 in this phase) |
| Own dev / validation | dev micro-F1@7: fused 0.4703 (model weight 0.1) · model alone 0.0523 · re-ranker 0.4725 · BM25 0.4647 |
| Notebook | `p7_arch4_partial_ft.ipynb` (original name: `p7-acrhi-4-partial-ft.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p7-acrhi-4-partial-FT-submission.zip`), contains `prediction.json` at the root |


## Method

* Every chunk (≤ 512 tokens) lies entirely in Architecture-4's bidirectional 512-token prefix.
* Contrastive training (InfoNCE, τ 0.05) on the 801 train queries with a reachable relevant case: 8 queries per update, each with 1 relevant candidate + 3 BM25 hard negatives (from the BM25 top 30) + all other candidates of the update; a candidate shows its 4 best BM25 passages for that query; GradCache keeps the whole update in memory; the best epoch is the one with the best model-alone dev F1@7.
* 4 epochs × 101 updates; 62 min on a T4, peak GPU 9.1 GB.
* Model alone on dev by epoch: 0.0334 → 0.0222 → 0.0289 → 0.0434 → **0.0478** (epoch 4 kept).
* Test ranking = 0.1 · z(model) + 0.9 · z(re-ranker), the weight picked on dev.

The data handling, head and evaluation shared by all Phase-7 notebooks are described in the [phase README](../README.md).

## Notes

* Architecture-4's pooled vectors barely rank anything (dev 0.05 alone): its representations transfer worst to retrieval, as they did for statutes in Phase 5. The submission is essentially the re-ranker.

## Model size (printed by the notebook)

```
Phase-1 model             : 205,496,064 (16 blocks)
trained in it (partial (layer-wise) fine-tuning): 44,835,840  (21.82%)
head (trainable)          : 590,609
```

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p7_arch4_partial_ft"
CITATION_WINDOW = 1          # sentences kept before / after a <CITATION> sentence
MAX_CONTEXTS = 48            # citation contexts per query (more: evenly spread)
PASSAGE_WORDS = 200          # words per candidate passage (~ 480 tokens with the Phase-1 tokenizer)
MAX_CHUNKS = 16              # passages per candidate for ranking (more: evenly spread over the case)
CHUNK_LEN = 512              # tokens per chunk (a longer passage / context is cut)
DATE_FILTER = True           # never rank a candidate decided after the query
POOL_SKIP_SPACES = True      # whitespace tokens stay in the input but are not pooled
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
PARTIAL_TOP_BLOCKS = 4
LR_TOP_BLOCK = 3e-5
LAYER_LR_DECAY = 0.8
BACKBONE_DROPOUT = 0.0       # 0: both GradCache passes must compute the same vectors
PREFIX_LENGTH = None         # bidirectional prefix of every chunk. None = the Phase-1 prefix (512 = the whole chunk)
TEMPERATURE = 0.05           # cosine / TEMPERATURE in the contrastive loss
LR_HEAD = 5e-4               # layer mix + projection
REUSE_TRAINED = False        # True: load retriever.pt of an earlier run of THIS notebook (attach that
EPOCHS = 4
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
12. CELL 12 — BUILD THE MODEL FROM THE PHASE-1 ARCHITECTURE-4 CHECKPOINT
13. CELL 13 — MODEL SCORES + EVALUATION ON A SPLIT
14. CELL 14 — CONTRASTIVE TRAINING FUNCTION (GradCache: big batches with little memory)
15. CELL 15 — BEFORE TRAINING: TOKEN STATISTICS + GRADCACHE CHECK
16. CELL 16 — TRAIN (TRAIN queries only; the model's own dev F1@7 picks the best epoch)
17. CELL 17 — DEV REPORT (all passages) + FUSION WEIGHT FOR THE TEST RANKING
18. CELL 18 — RANK THE TEST POOL + WRITE prediction.json + submission.zip
