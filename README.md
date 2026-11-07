# CS-781 Capstone — Legal Language Modelling Challenge

**Ritik Gupta** · CS-781, IIT Kanpur · individual submission · Codabench

A self-hosted legal NLP stack built **from scratch**: our own BPE tokenizer and four compact legal language models
(110M–205M parameters, no pretrained weights), trained on the official legal corpus and then adapted to six downstream
legal tasks — named-entity recognition, rhetorical roles, judgment prediction, statute identification, summarization and
prior case retrieval. Every downstream system starts from one of the Phase-1 checkpoints and is adapted with a
fine-tuning or parameter-efficient method implemented in plain PyTorch.

## Results overview

| Phase | Task | Primary metric | Submissions | Best submission | Best Codabench score |
|---|---|---|---|---|---|
| 1 | [Legal LM pretraining](phase1_pretraining/) | BERTScore F1 | 4 | [Architecture-2 (Transformer encoder-decoder)](phase1_pretraining/arch2_encoder_decoder/) | 0.8289 |
| 2 | [Legal NER](phase2_legal_ner/) | Strict Macro-F1 | 4 | [Arch-1 + LoRMA](phase2_legal_ner/01_arch1_lorma/) | 0.5535 |
| 3 | [Rhetorical roles](phase3_rhetorical_roles/) | Macro-F1 | 7 | [Arch-1 + LoRMA](phase3_rhetorical_roles/02_arch1_lorma/) | 0.5266 |
| 4 | [Judgment prediction](phase4_judgment_prediction/) | Macro-F1 | 11 | [Arch-3 + LoRMA](phase4_judgment_prediction/09_arch3_lorma/) | 0.6896 |
| 5 | [Statute identification](phase5_statute_identification/) | Macro-F1 | 7 | [Arch-1 + QLoRA](phase5_statute_identification/02_arch1_qlora/) | 0.2919 |
| 6 | [Summarization](phase6_summarization/) | BERTScore F1 | 5 | [Arch-1 + LoRMA](phase6_summarization/02_arch1_lorma/) | 0.8404 |
| 7 | [Prior case retrieval](phase7_prior_case_retrieval/) | Micro-F1@7 | 6 | [Arch-1 + QLoRA](phase7_prior_case_retrieval/02_arch1_qlora/) | 0.4725 |

Each phase README has the full results table (all metrics, own dev scores) and the analysis; each submission folder has its
own README with the method, the exact configuration and a map of the notebook cells.

## The four Phase-1 models

| | Arch-1 | Arch-2 | Arch-3 | Arch-4 |
|---|---|---|---|---|
| Type | decoder-only (Llama-style) | encoder-decoder | GQA decoder (E-LegalFormer) | hybrid prefix-LM |
| Parameters | 109.5M | 149.6M | 125.3M | 205.5M |
| Layers · d_model | 12 · 768 | 6 + 6 · 768 | 16 · 768 | 16 · 768 |
| Attention | causal, RoPE, QK-norm | bidirectional encoder + causal decoder | causal, 12 q / 4 kv heads, RoPE | bidirectional 512-token prefix + causal branch, gated fusion |
| FFN | SwiGLU 2048 | GELU 3072 | SwiGLU 2048 | SwiGLU 2048 |
| Context | 2,048 | 1,024 → 512 | 2,048 | 2,048 |
| Phase-1 BERTScore | 0.8238 | **0.8289** | 0.8196 | 0.8283 |

Details: [phase1_pretraining/README.md](phase1_pretraining/README.md). Tokenizer: own BPE, 32,004 tokens, NFKC, byte-exact decoding
([phase1_pretraining/tokenizer](phase1_pretraining/tokenizer/)).

## Methods used per phase

| Method | P2 | P3 | P4 | P5 | P6 | P7 |
|---|---|---|---|---|---|---|
| LoRA | A3 | A1, A2 |  |  | A1, KD |  |
| LoRMA | A1 | A1 | A1, A2, A3 | A1 | A1 | A1 |
| IA3 |  | A1, A2, A3 |  |  |  | A3 |
| BitFit |  |  | A1, A2, A3, A4 |  |  |  |
| Gradual unfreezing |  |  | A1, A2, A3 |  |  |  |
| Full fine-tuning | A2 |  |  | A2 |  | A2 |
| Partial fine-tuning | A4 (teacher) |  | KD | A4 |  | A4 |
| QLoRA (4-bit NF4) |  |  |  | A1 |  | A1 |
| Houlsby adapters |  |  |  | A3 | A2 |  |
| AdaLoRA |  |  |  | A3 | A4 |  |
| Knowledge distillation | A4→A2 | A1→A2 | A4→A2 | A4→A2 | A4→A3 | A1→A2 |

Formulas, initialisation and settings for every method: [docs/methods.md](docs/methods.md).

## How the downstream systems are built (common design)

* **Long documents** — every phase has documents longer than the context: windows with overlap (NER), packed sentence windows
  (roles), the last 1,536 tokens in 3 chunks (judgment), up to 8 chunks spread over the document (statutes), sentence-aligned
  segments with their own summary targets (summarization), citation contexts against 200-word passages (retrieval).
* **Heads on top of the frozen / adapted backbone** — BiLSTM + CRF over words (NER) or sentences (roles), BiLSTM + attention
  pooling over chunks (judgment), layer mix + label attention starting from a logistic regression (statutes), the LM head itself (summarization),
  a layer-mix retriever head trained contrastively and fused with a learned BM25 re-ranker (retrieval).
* **Model selection on dev only** — test labels are never used; the dev set picks epochs, thresholds and decoding settings.
* **Every submission is a notebook** with numbered `CELL N —` cells: CELL 2 holds all settings, the last cell checks the
  prediction format (every test id, nothing extra) and writes `submission.zip` with `prediction.json` at its root.

## Repository layout

```
CS781-Legal-LM-Capstone/
├── README.md                          ← this file
├── docs/
│   ├── methods.md                     ← every fine-tuning / PEFT / distillation method, with formulas
│   └── reproducing.md                 ← how to run any notebook on Kaggle
├── phase1_pretraining/
│   ├── tokenizer/                     ← BPE tokenizer + corpus tokenization + tokenizer.py
│   ├── arch1_decoder_only/            ← one folder per Phase-1 model: notebook + README
│   ├── arch2_encoder_decoder/
│   ├── arch3_elegalformer/
│   └── arch4_hybrid_prefix_lm/
├── phase2_legal_ner/                  ← one folder per submission:
│   ├── README.md                      ←   phase pipeline, results table, observations
│   └── 01_arch1_lorma/                ←   notebook + submission.zip + README
│       ...
├── phase3_rhetorical_roles/
├── phase4_judgment_prediction/
├── phase5_statute_identification/
├── phase6_summarization/
├── phase7_prior_case_retrieval/
├── requirements.txt
└── .gitignore
```

## All submissions

<details><summary><b>Phase 2</b> — Legal NER (4 submissions, Codabench Strict Macro-F1)</summary>

| # | Method | Submission | Strict Macro-F1 |
|---|---|---|---|
| 01 | [Arch-1 + LoRMA](phase2_legal_ner/01_arch1_lorma/) | Architecture-1 decoder (frozen) + LoRMA + word features + BiLSTM + CRF | 0.5535 |
| 02 | [Arch-2 + full fine-tuning](phase2_legal_ner/02_arch2_full_ft/) | Architecture-2 encoder, full fine-tuning + word features + BiLSTM + CRF | 0.4246 |
| 03 | [Arch-3 + LoRA](phase2_legal_ner/03_arch3_lora/) | Architecture-3 E-LegalFormer (frozen) + LoRA, bidirectional attention, linear token head | 0.4912 |
| 04 | [KD Arch-4 → Arch-2](phase2_legal_ner/04_kd_arch4_to_arch2/) | Knowledge distillation: Architecture-4 teacher (partial FT) → Architecture-2 encoder student | 0.4445 |

</details>

<details><summary><b>Phase 3</b> — Rhetorical roles (7 submissions, Codabench Macro-F1)</summary>

| # | Method | Submission | Macro-F1 |
|---|---|---|---|
| 01 | [Arch-1 + LoRA](phase3_rhetorical_roles/01_arch1_lora/) | Architecture-1 decoder (frozen) + LoRA + sentence BiLSTM + CRF | 0.4362 |
| 02 | [Arch-1 + LoRMA](phase3_rhetorical_roles/02_arch1_lorma/) | Architecture-1 decoder (frozen) + LoRMA + sentence BiLSTM + CRF | 0.5266 |
| 03 | [Arch-1 + IA3](phase3_rhetorical_roles/03_arch1_ia3/) | Architecture-1 decoder (frozen) + IA3 + sentence BiLSTM + CRF | 0.37 |
| 04 | [KD Arch-1 → Arch-2](phase3_rhetorical_roles/04_kd_arch1_to_arch2/) | Knowledge distillation: Architecture-1 + LoRMA teacher → Architecture-2 encoder student | 0.36 |
| 05 | [Arch-2 + LoRA](phase3_rhetorical_roles/05_arch2_lora/) | Architecture-2 encoder (frozen) + LoRA + sentence BiLSTM + CRF | 0.345 |
| 06 | [Arch-2 + IA3](phase3_rhetorical_roles/06_arch2_ia3/) | Architecture-2 encoder (frozen) + IA3 + sentence BiLSTM + CRF | 0.2789 |
| 07 | [Arch-3 + IA3](phase3_rhetorical_roles/07_arch3_ia3/) | Architecture-3 E-LegalFormer (frozen) + IA3 + sentence BiLSTM + CRF | 0.421 |

</details>

<details><summary><b>Phase 4</b> — Judgment prediction (11 submissions, Codabench Macro-F1)</summary>

| # | Method | Submission | Macro-F1 |
|---|---|---|---|
| 01 | [Arch-1 + BitFit](phase4_judgment_prediction/01_arch1_bitfit/) | Architecture-1 decoder + BitFit | 0.6206 |
| 02 | [Arch-1 + gradual unfreezing](phase4_judgment_prediction/02_arch1_freeze_unfreeze/) | Architecture-1 decoder + gradual unfreezing | 0.6047 |
| 03 | [Arch-1 + LoRMA](phase4_judgment_prediction/03_arch1_lorma/) | Architecture-1 decoder (frozen) + LoRMA | 0.6556 |
| 04 | [Arch-2 + BitFit](phase4_judgment_prediction/04_arch2_bitfit/) | Architecture-2 encoder + BitFit | 0.5592 |
| 05 | [Arch-2 + gradual unfreezing](phase4_judgment_prediction/05_arch2_freeze_unfreeze/) | Architecture-2 encoder + gradual unfreezing | 0.5479 |
| 06 | [Arch-2 + LoRMA](phase4_judgment_prediction/06_arch2_lorma/) | Architecture-2 encoder (frozen) + LoRMA | 0.5898 |
| 07 | [Arch-3 + BitFit](phase4_judgment_prediction/07_arch3_bitfit/) | Architecture-3 E-LegalFormer + BitFit | 0.5974 |
| 08 | [Arch-3 + gradual unfreezing](phase4_judgment_prediction/08_arch3_freeze_unfreeze/) | Architecture-3 E-LegalFormer + gradual unfreezing | 0.6451 |
| 09 | [Arch-3 + LoRMA](phase4_judgment_prediction/09_arch3_lorma/) | Architecture-3 E-LegalFormer (frozen) + LoRMA | 0.6896 |
| 10 | [Arch-4 + BitFit](phase4_judgment_prediction/10_arch4_bitfit/) | Architecture-4 hybrid prefix-LM + BitFit | 0.527 |
| 11 | [KD Arch-4 → Arch-2](phase4_judgment_prediction/11_kd_arch4_to_arch2/) | Knowledge distillation: Architecture-4 teacher → Architecture-2 encoder student | 0.5937 |

</details>

<details><summary><b>Phase 5</b> — Statute identification (7 submissions, Codabench Macro-F1)</summary>

| # | Method | Submission | Macro-F1 |
|---|---|---|---|
| 01 | [Arch-1 + LoRMA](phase5_statute_identification/01_arch1_lorma/) | Architecture-1 decoder (frozen) + LoRMA + layer mix + label attention | 0.2861 |
| 02 | [Arch-1 + QLoRA](phase5_statute_identification/02_arch1_qlora/) | Architecture-1 decoder in 4-bit NF4 + LoRA (QLoRA) + layer mix + label attention | 0.2919 |
| 03 | [Arch-2 + full fine-tuning](phase5_statute_identification/03_arch2_full_ft/) | Architecture-2 encoder, full fine-tuning + layer mix + label attention | 0.2858 |
| 04 | [Arch-3 + Houlsby adapters](phase5_statute_identification/04_arch3_houlsby_adapters/) | Architecture-3 E-LegalFormer (frozen) + Houlsby adapters + layer mix + label attention | 0.2838 |
| 05 | [Arch-3 + AdaLoRA](phase5_statute_identification/05_arch3_adalora/) | Architecture-3 E-LegalFormer (frozen) + AdaLoRA + layer mix + label attention | 0.2817 |
| 06 | [Arch-4 + partial fine-tuning](phase5_statute_identification/06_arch4_partial_ft/) | Architecture-4 hybrid prefix-LM + partial fine-tuning + layer mix + label attention | 0.2431 |
| 07 | [KD Arch-4 → Arch-2](phase5_statute_identification/07_kd_arch4_to_arch2/) | Knowledge distillation: Architecture-4 teacher → Architecture-2 encoder student (multi-label) | 0.2463 |

</details>

<details><summary><b>Phase 6</b> — Summarization (5 submissions, Codabench BERTScore F1)</summary>

| # | Method | Submission | BERTScore F1 |
|---|---|---|---|
| 01 | [Arch-1 + LoRA](phase6_summarization/01_arch1_lora/) | Architecture-1 decoder + LoRA, segment-wise summarization | 0.8385 |
| 02 | [Arch-1 + LoRMA](phase6_summarization/02_arch1_lorma/) | Architecture-1 decoder + LoRMA, segment-wise summarization | 0.8404 |
| 03 | [Arch-2 + Houlsby adapters](phase6_summarization/03_arch2_houlsby_adapters/) | Architecture-2 encoder-decoder + Houlsby adapters, segment-wise summarization | 0.8162 |
| 04 | [Arch-4 + AdaLoRA](phase6_summarization/04_arch4_adalora/) | Architecture-4 hybrid prefix-LM + AdaLoRA, segment-wise summarization | 0.8068 |
| 05 | [KD Arch-4 → Arch-3](phase6_summarization/05_kd_arch4_to_arch3/) | Knowledge distillation: Architecture-4 + LoRA teacher → Architecture-3 + LoRA student (token-level) | 0.8233 |

</details>

<details><summary><b>Phase 7</b> — Prior case retrieval (6 submissions, Codabench Micro-F1@7)</summary>

| # | Method | Submission | Micro-F1@7 |
|---|---|---|---|
| 01 | [Arch-1 + LoRMA](phase7_prior_case_retrieval/01_arch1_lorma/) | Architecture-1 decoder + LoRMA retriever, fused with the learned re-ranker | 0.466 |
| 02 | [Arch-1 + QLoRA](phase7_prior_case_retrieval/02_arch1_qlora/) | Architecture-1 decoder in 4-bit NF4 + LoRA (QLoRA) retriever, fused with the learned re-ranker | 0.4725 |
| 03 | [Arch-2 + full fine-tuning](phase7_prior_case_retrieval/03_arch2_full_ft/) | Architecture-2 encoder + full fine-tuning retriever, fused with the learned re-ranker | 0.4696 |
| 04 | [Arch-3 + IA3](phase7_prior_case_retrieval/04_arch3_ia3/) | Architecture-3 E-LegalFormer + IA3 retriever, fused with the learned re-ranker | 0.4702 |
| 05 | [Arch-4 + partial fine-tuning](phase7_prior_case_retrieval/05_arch4_partial_ft/) | Architecture-4 hybrid prefix-LM + partial (layer-wise) fine-tuning retriever, fused with the learned re-ranker | 0.4684 |
| 06 | [KD Arch-1 → Arch-2](phase7_prior_case_retrieval/06_kd_arch1_to_arch2/) | Knowledge distillation: Architecture-1 + LoRMA retriever (teacher) → Architecture-2 encoder retriever (student) | 0.4702 |

</details>

## Reproducing

See [docs/reproducing.md](docs/reproducing.md): add the Hugging Face token as the Kaggle secret `HF`, attach the tokenizer and
Phase-1 outputs as notebook inputs, *Run All*. Model checkpoints and the competition data are not part of this repository.

## Competition rules followed

* Phase-1 models trained from scratch (50M–300M parameters); no pretrained language model, external LLM or API is used for
  training, inference or evaluation.
* Only the official competition data; the Hugging Face dataset is downloaded at run time and not redistributed here.
* Test labels are private and never used; model selection uses train / dev only.
* No access tokens in the code (Kaggle secrets); notebook outputs that printed raw dataset text were removed before publishing.
* Phase 7: the unmasked `CITATOR INFO` lists in the case headers (later cases citing a judgment) are not used for matching.

## References

1. Joshi et al. (2024). *IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning.* arXiv:2407.05399.
2. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
3. Bihany, Patel & Modi (2025). *LoRMA: Low-Rank Multiplicative Adaptation for LLMs.*
4. Liu et al. (2022). *Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning* (IA3).
5. Ben Zaken, Goldberg & Ravfogel (2022). *BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.*
6. Howard & Ruder (2018). *Universal Language Model Fine-tuning for Text Classification* (gradual unfreezing).
7. Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.*
8. Houlsby et al. (2019). *Parameter-Efficient Transfer Learning for NLP.*
9. Zhang et al. (2023). *AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.*
10. Hinton, Vinyals & Dean (2015). *Distilling the Knowledge in a Neural Network.*
