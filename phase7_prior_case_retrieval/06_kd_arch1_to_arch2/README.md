# Phase 7 · Submission 06 — Knowledge distillation: Architecture-1 + LoRMA retriever (teacher) → Architecture-2 encoder retriever (student)

| | |
|---|---|
| Phase-1 backbone | Architecture-2 encoder-decoder (150M) |
| Adaptation | teacher: LoRMA on Architecture-1 (3.05M trained, 112.6M backbone); student: Architecture-2 encoder, full fine-tuning (67.9M) |
| Codabench | Micro-F1@7 **0.4702** · Micro-P@7 0.4798 · Micro-R@7 0.4609  (rank 2 of 6 in this phase) |
| Own dev / validation | dev micro-F1@7 (student): fused 0.4725 (model weight 0.2) · model alone 0.2579 (teacher 0.3669) · re-ranker 0.4725 |
| Notebook | `p7_kd_arch1_to_arch2.ipynb` (original name: `p7-kd-arch1-to-arch2.ipynb`) |
| Submission file | `submission.zip` (Codabench upload; original name `p7_kd_arch1_to_arch2-submission.zip`), contains `prediction.json` at the root |


## Method

* Step 1: teacher = the Architecture-1 + LoRMA retriever, trained in this notebook for 3 epochs (48 min; model alone on dev 0.3669).
* Step 2: student = Architecture-2 encoder retriever. Loss = 0.5 · InfoNCE(relevant candidate) + 0.5 · KL(teacher scores ‖ student scores) over the candidates of the update + 1.0 · (1 − cos(student chunk vector, teacher chunk vector)).
* The teacher runs next to the student without gradients and encodes exactly the same chunks, so the two score matrices line up.
* Student: 5 epochs × 101 updates (47 min); model alone on dev 0.2023 → 0.2112 → 0.2435 → 0.2457 → **0.2524** → 0.2457 (epoch 4 kept).
* Test ranking = 0.2 · z(student) + 0.8 · z(re-ranker), the weight picked on dev.

The data handling, head and evaluation shared by all Phase-7 notebooks are described in the [phase README](../README.md).

## Notes

* Student = 60% of the teacher's backbone weights, encodes 3.5× as fast (245 vs 71 chunks/s) and keeps 70% of its model-alone dev F1@7 (0.2579 vs 0.3669).
* Distillation did not beat plain contrastive training of the same student (submission 03: 0.2668 alone); on Codabench the two differ by one correct prediction.

## Configuration (CELL 2 of the notebook)

```python
RUN_NAME = "p7_kd_arch1_to_arch2"
TEACHER_RUN = "p7_arch1_lorma"
CITATION_WINDOW = 1          # sentences kept before / after a <CITATION> sentence
MAX_CONTEXTS = 48            # citation contexts per query (more: evenly spread)
PASSAGE_WORDS = 200          # words per candidate passage (~ 480 tokens with the Phase-1 tokenizer)
MAX_CHUNKS = 16              # passages per candidate for ranking (more: evenly spread over the case)
CHUNK_LEN = 512              # tokens per chunk (a longer passage / context is cut)
DATE_FILTER = True           # never rank a candidate decided after the query
POOL_SKIP_SPACES = True      # whitespace tokens stay in the input but are not pooled
POOL_SKIP_TOKENS = 4         # the first tokens of a chunk are not pooled (they behave abnormally in LMs)
LORMA_TARGETS = ["attn.qkv", "attn.out_proj", "mlp.gate", "mlp.up", "mlp.down"]   # every linear layer of every block
LORMA_RANK = 16
LORMA_ALPHA = 32
LORMA_DROPOUT = 0.0          # 0: both GradCache passes must compute the same vectors
LR_PEFT = 2e-4
TEACHER_EPOCHS = 3           # only if the teacher has to be trained here
LR_BACKBONE = 2e-5
BACKBONE_DROPOUT = 0.0
KD_ALPHA = 0.5               # weight of the teacher-score term (the gold term gets 1 - KD_ALPHA)
KD_COSINE_WEIGHT = 1.0       # weight of the case-vector term
TEMPERATURE = 0.05           # cosine / TEMPERATURE in the contrastive loss
LR_HEAD = 5e-4               # layer mix + projection
REUSE_TRAINED = False        # True: load retriever.pt of an earlier run of THIS notebook (attach that
EPOCHS = 5
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
11. CELL 11 — MODELS
12. CELL 12 — LOAD BOTH PHASE-1 CHECKPOINTS + MODEL BUILDERS
13. CELL 13 — MODEL SCORES + EVALUATION ON A SPLIT
14. CELL 14 — CONTRASTIVE TRAINING FUNCTION (GradCache: big batches with little memory)
15. CELL 15 — BEFORE TRAINING: TOKEN STATISTICS + GRADCACHE CHECK
16. CELL 16 — STEP 1: THE TEACHER (Architecture-1 + LoRMA retriever)
17. CELL 17 — STEP 2: THE STUDENT (Architecture-2 encoder) WITH DISTILLATION (TRAIN queries only)
18. CELL 18 — DEV REPORT (all passages) + FUSION WEIGHT FOR THE TEST RANKING
19. CELL 19 — RANK THE TEST POOL + WRITE prediction.json + submission.zip
