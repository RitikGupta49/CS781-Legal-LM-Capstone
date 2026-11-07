# Phase 7 — Prior Case Retrieval (PCR)

**Task.** Given a query judgment, rank the candidate pool of its split and return the **7 prior cases it most likely cites** (exactly 7 unique candidate ids per query, best first, never the query itself).

**Data.** Train 827 queries / pool 4,320 candidates; dev 118 / 1,023; test 237 / 1,727. 6.4 (train) and 8.3 (dev) relevant cases per query on average, max 92. Median case ≈4.5k words (≈11k tokens). Citations inside the text are masked as `<CITATION>`, names and organisations as `<NAME>` / `<ORG>`. Because many queries have fewer than 7 relevant cases, the best possible micro-F1@7 is 0.639 (train) / 0.618 (dev).

**Metric.** Micro-F1@7 (primary) · micro-precision@7 · micro-recall@7 (exactly 7 candidates per query)

## Shared pipeline

All six notebooks share the same retrieval pipeline (built from what the data showed; dev micro-F1@7 of plain BM25 in brackets):

1. **Queries = citation contexts.** Every sentence with a `<CITATION>` plus one sentence before and after is a separate small query
   (≈6–7 per case, at most 48). Whole query vs whole candidate [0.157] → contexts vs passages [0.466 with the date filter].
2. **Candidates = passages** of 200 words (at most 16, spread over the case); a candidate's score is its best passage.
3. **Date filter.** A judgment only cites older cases: candidates decided after the query (year from `DATE OF JUDGMENT`, else
   the latest year in the text) are never ranked [0.157 → 0.203 for whole cases]. It removes ≈43% of every pool and loses 7% of the relevant pairs.
4. **BM25** (own implementation, idf over the split's passages): per context, z-scored over the pool, then the best context of the query.
5. **Learned re-ranker** (CELL 10). Logistic regression on 9 signals of the BM25 top 100 — best and 2nd-best context, share of contexts
   with the candidate in their top 3, gap to the query's 7th best, BM25 of the whole case, years between the cases, Supreme Court
   judgment, candidate length, number of contexts — fitted on **train queries only**. Dev 0.4647 → **0.4725**; 5-fold on train
   queries 0.390 → 0.404 (better in every fold).
6. **Retriever on the Phase-1 model.** Each context / passage (≤ 512 tokens) → backbone → learned softmax mix of the LayerNorm-ed
   states of all layers → standardization (fixed train statistics) → mean over the non-whitespace tokens → projection (identity at the
   start) → unit vector. score(context, candidate) = best cosine over the candidate's passages, z-scored over the pool; query score =
   best context. Epoch 0 = the Phase-1 model without training.
7. **Training.** InfoNCE (τ 0.05), 8 queries per update × (1 relevant + 3 BM25 hard negatives) + all other candidates of the update;
   a candidate shows its 4 best BM25 passages for that query; GradCache (exact large-batch gradient with little memory, checked
   against a direct backward pass before training). The best epoch is chosen by the model's own dev F1@7.
8. **Fusion.** Test ranking = w · z(model) + (1 − w) · z(re-ranker) over the re-ranker's top 100, w ∈ {0.1 … 1.0} picked on dev.
9. **Evaluation.** Micro-F1@7 implemented exactly as the leaderboard (recall counts every relevant case, also those the date filter
   removes); the last cell checks 7 unique ids from the right pool for every test query.


## Submissions

| # | Submission | Backbone | Method | Codabench Micro-F1@7 | Codabench Micro-P@7 | Codabench Micro-R@7 | Own dev / validation |
|---|---|---|---|---|---|---|---|
| 01 | [01_arch1_lorma](01_arch1_lorma/) | Arch-1 | Arch-1 + LoRMA | 0.466 | 0.4756 | 0.4569 | dev micro-F1@7: fused 0.4769 (model weight 0.4) · model alone 0.3702 · re-ranker 0.4725 · BM25 0.4647 |
| 02 | [02_arch1_qlora](02_arch1_qlora/) | Arch-1 | Arch-1 + QLoRA | **0.4725** | 0.4822 | 0.4632 | dev micro-F1@7: fused 0.4736 (model weight 0.1) · model alone 0.3624 · re-ranker 0.4725 · BM25 0.4647 |
| 03 | [03_arch2_full_ft](03_arch2_full_ft/) | Arch-2 | Arch-2 + full fine-tuning | 0.4696 | 0.4792 | 0.4603 | dev micro-F1@7: fused 0.4747 (model weight 0.5) · model alone 0.2668 · re-ranker 0.4725 · BM25 0.4647 |
| 04 | [04_arch3_ia3](04_arch3_ia3/) | Arch-3 | Arch-3 + IA3 | 0.4702 | 0.4798 | 0.4609 | dev micro-F1@7: fused 0.4725 (model weight 0.2) · model alone 0.3002 · re-ranker 0.4725 · BM25 0.4647 |
| 05 | [05_arch4_partial_ft](05_arch4_partial_ft/) | Arch-4 | Arch-4 + partial fine-tuning | 0.4684 | 0.478 | 0.4592 | dev micro-F1@7: fused 0.4703 (model weight 0.1) · model alone 0.0523 · re-ranker 0.4725 · BM25 0.4647 |
| 06 | [06_kd_arch1_to_arch2](06_kd_arch1_to_arch2/) | Arch-2 | KD Arch-1 → Arch-2 | 0.4702 | 0.4798 | 0.4609 | dev micro-F1@7 (student): fused 0.4725 (model weight 0.2) · model alone 0.2579 (teacher 0.3669) · re-ranker 0.4725 |

Best: **02_arch1_qlora** — Architecture-1 decoder in 4-bit NF4 + LoRA (QLoRA) retriever, fused with the learned re-ranker, Codabench Micro-F1@7 0.4725.

## Observations

* All six submissions land within 11 correct predictions of each other on Codabench (789–800 of 1,659; micro-F1@7 0.4660–0.4725): the shared BM25 + re-ranker pipeline carries most of the score, the Phase-1 retriever moves it by a few hits.
* The model alone ranks the backbones clearly (dev micro-F1@7): Arch-1 + LoRMA 0.370 > Arch-1 + QLoRA 0.362 > Arch-3 + IA3 0.300 > Arch-2 full FT 0.267 > KD student 0.258 > Arch-4 0.052. The decoder-only Arch-1 gives the best case representations, Arch-4 transfers worst (as for statutes in Phase 5).
* Training helps every backbone: Arch-1 goes from 0.284 (Phase-1 model, no training) to 0.350, Arch-3 + IA3 from 0.232 to 0.288 with 0.03% of the weights trained.
* Lifting the dev score above the re-ranker does not reliably carry over to test: Arch-1 + LoRMA had the best dev score (0.4769 vs 0.4725, model weight 0.4) and the lowest Codabench score (0.4660), while QLoRA with the smallest model weight (0.1) is best on test (0.4725). 118 dev queries are too few to pick a fusion weight precisely.
* On this data a dense retriever overlaps strongly with BM25: a bag-of-words retriever that scores 0.40 alone adds only +0.004 when fused with BM25 (checked during development), and training the model on BM25's residual made it worse.
* QLoRA keeps 98% of the LoRMA retriever's model-alone F1 with 4-bit base weights and a lower peak memory (12.8 vs 15.0 GB).
* The KD student (Arch-2 encoder, 60% of the teacher's weights, 3.5× faster) keeps 70% of the teacher's model-alone F1 but does not beat the same encoder trained without distillation (0.258 vs 0.267).
* The case headers are not masked: a candidate's `CITATOR INFO` lists the later cases that cite it, which would reveal ≈10% of the relevant dev pairs directly. This was deliberately **not** used — the task asks for retrieval from the legal content.

Every submission folder holds the notebook (with its training log), `submission.zip` as uploaded to Codabench, and a README
with the method, the exact configuration and a map of the notebook cells.
