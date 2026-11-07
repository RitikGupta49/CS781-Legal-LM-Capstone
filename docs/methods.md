# Fine-tuning and adaptation methods

Every Phase 2–7 notebook starts from a Phase-1 checkpoint (no external pretrained model) and adapts it
with one of the methods below. All of them are implemented in the notebooks themselves (plain PyTorch, no PEFT library).
"Starts unchanged" means the adapted model is exactly the Phase-1 model before the first update (the Phase 5–7 notebooks assert this before training).

| Method | Update | Starts unchanged by | Used in |
|---|---|---|---|
| **LoRA** (Hu et al., 2021) | `W'x = Wx + (α/r)·B(Ax)` | `B = 0` | P2 Arch-3 · P3 Arch-1/2 · P6 Arch-1, KD |
| **LoRMA** (Bihany, Patel & Modi, 2025) | `W'x = (I + (α/r)·BA)·Wx = h + (α/r)·B(A h)`, `h = Wx` | `B = 0` | P2 Arch-1 · P3 Arch-1 (+KD teacher) · P4 Arch-1/2/3 · P5 Arch-1 · P6 Arch-1 · P7 Arch-1 (+KD teacher) |
| **IA3** (Liu et al., 2022) | keys `·l_k`, values `·l_v`, FFN hidden `·l_ff` | vectors = 1 | P3 Arch-1/2/3 · P7 Arch-3 |
| **BitFit** (Ben Zaken et al., 2022) | only bias vectors train (new zero biases where a layer had none) | biases = 0 | P4 Arch-1/2/3/4 |
| **Gradual unfreezing** (Howard & Ruder, 2018) | head → top ¼ → top ½ → all blocks, layer-wise LRs (×0.8 per block) | head-only first epoch | P4 Arch-1/2/3 |
| **Full fine-tuning** | every backbone weight trains (small LR) | — | P2 Arch-2 · P5 Arch-2 · P7 Arch-2 (+KD student) |
| **Partial fine-tuning** | top k blocks (or the token embeddings) train with layer-wise LRs | — | P2 Arch-4 teacher · P4 KD · P5 Arch-4 · P7 Arch-4 |
| **QLoRA** (Dettmers et al., 2023) | frozen weights in 4-bit NF4 (block 64, absmax scale) + LoRA | `B = 0` | P5 Arch-1 · P7 Arch-1 |
| **Houlsby adapters** (Houlsby et al., 2019) | `out = y + up(GELU(down(y)))`, bottleneck 64 | `up = 0` | P5 Arch-3 · P6 Arch-2 |
| **AdaLoRA** (Zhang et al., 2023) | `ΔW = P·diag(E ⊙ mask)·Q`, importance-based rank pruning (12 → 8 on average), orthogonality penalty | `E = 0` | P5 Arch-3 · P6 Arch-4 |
| **Knowledge distillation** (Hinton et al., 2015) | student loss = (1 − α)·gold loss + α·T²·KL(teacher ‖ student) | — | P2 4→2 · P3 1→2 · P4 4→2 · P5 4→2 · P6 4→3 · P7 1→2 |

## Knowledge distillation variants

| Phase | Teacher → student | What is matched | Detail |
|---|---|---|---|
| 2 | Arch-4 (top 8 blocks FT) → Arch-2 encoder | per-word tag distributions | T = 2, α = 0.5, only words the teacher labels correctly |
| 3 | Arch-1 + LoRMA → Arch-2 encoder | per-sentence role distributions | T = 2, α = 0.5, only sentences the teacher labels correctly (82.3%) |
| 4 | Arch-4 → Arch-2 encoder | P(outcome) per case | T = 1, α = 0.5, teacher scores all train cases |
| 5 | Arch-4 (frozen) → Arch-2 encoder | 100 statute probabilities | T = 1, α = 0.5, BCE with soft labels on documents the teacher did not train on |
| 6 | Arch-4 + LoRA → Arch-3 + LoRA | next-token distributions on the target tokens | T = 1, α = 0.5, both share the tokenizer, so positions line up |
| 7 | Arch-1 + LoRMA retriever → Arch-2 encoder retriever | the score distribution over the candidates of an update + the chunk vectors | T = 1, α = 0.5, + 1.0 · (1 − cosine) between student and teacher chunk vectors; the teacher encodes the same chunks |

## Rank / budget settings

* LoRA, LoRMA, QLoRA: rank 16, α 32, dropout 0.05 (0 in Phase 7, where GradCache needs two identical forward passes), on every linear layer of every block (attention + FFN).
* AdaLoRA: initial rank 12, target average rank 8, α 24, β 0.85 importance smoothing, warm-up 200 updates, cut every 10 updates,
  ranks fixed for the last 400–500 updates, orthogonality weight 0.5.
* Houlsby adapters: bottleneck 64 after the attention output projection and the FFN down projection.

## Phase 7 retrieval components (not adaptation methods)

* **Contrastive objective** — InfoNCE over the candidates of an update: `loss = −log softmax(s(q, c⁺)/τ)`, τ = 0.05, where
  s = best context × best passage cosine; the other relevant cases of a query are masked out of its denominator.
* **GradCache** (Gao et al., 2021) — pass 1 encodes every chunk without gradients and gets ∂loss/∂vector; pass 2 re-encodes the chunks in
  small groups with gradients and back-propagates that vector gradient: the gradient of the full batch with the memory of one group.
* **Learned re-ranker** — logistic regression (scikit-learn) on 9 BM25-based signals of the BM25 top 100, fitted on train queries only;
  the Phase-1 retriever is fused with its score (weight picked on dev).
