# Phase 1 — Legal Language Model Pretraining

**Task.** Train compact legal language models **from scratch** on the official legal corpus (acts, high courts, supreme court, tribunals) and continue 1,000 test prompts. Every later phase starts from one or more of these four checkpoints.

**Data.** Official Phase-1 corpus (four JSON sources), tokenized once with our own BPE tokenizer and stored as `uint16` token files (`<source>_100m.bin`). Test: 1,000 prompts to continue.

**Metric.** BERTScore F1 (primary) · ROUGE-L F1 · BLEU-4


### Tokenizer (`tokenizer/`)

| | |
|---|---|
| Type | own byte-exact **BPE**, trained from scratch on the legal corpus (`01_bpe_tokenizer.ipynb`) |
| Vocabulary | 32,004 = 4 special tokens + 32,000 pieces (initial characters + BPE merges) |
| Special tokens | `<pad>` = 0, `<unk>` = 1, `<bos>` = 2, `<eos>` = 3 |
| Normalization | Unicode NFKC, CRLF → LF, NUL removed; newlines and spaces are kept |
| Pre-tokenization | regex: newline runs · optional space/tab + word (with apostrophes) · optional space + one symbol · runs of spaces |
| Training | word-frequency statistics (≤ 100M words per source, min frequency 2) → incremental BPE merges |
| Decoding | `"".join(tokens)` — byte-exact round trip (checked in the notebook) |
| Package | `tokenizer.py` + `vocab.json` + merges + config, loaded by every later notebook with `load_tokenizer()` |

`02_tokenize_corpus.ipynb` streams the four JSON sources, tokenizes every document and writes one
`uint16` token file per source with `<eos>` between documents (the same `<eos>` is the summary
separator in Phase 6). `tokenizer.py` in this folder is the standalone module that the tokenizer
notebook writes out.

Note: the pre-tokenizer keeps every whitespace run as its own piece, so roughly every second token is a
space (2.4 tokens per word). Later phases account for this (e.g. Phase 5 does not pool whitespace tokens).

### The four Phase-1 models

| | Arch-1 | Arch-2 | Arch-3 | Arch-4 |
|---|---|---|---|---|
| Type | decoder-only (Llama-style) | encoder-decoder | decoder, grouped-query attention ("E-LegalFormer") | hybrid prefix-LM |
| Parameters | 109.5M | 149.6M | 125.3M | 205.5M |
| Layers | 12 | 6 enc + 6 dec | 16 | 16 |
| d_model / heads | 768 / 12 | 768 / 12 | 768 / 12 q, 4 kv | 768 / 12 |
| FFN | SwiGLU 2048 | GELU 3072 | SwiGLU 2048 | SwiGLU 2048 |
| Norm | RMSNorm + QK-norm | pre-LayerNorm | RMSNorm | RMSNorm |
| Positions | RoPE | learned | RoPE | learned |
| Context | 2,048 | 1,024 source → 512 target | 2,048 | 2,048 (first 512 bidirectional) |
| LM head | tied | tied to decoder embedding | tied | tied |
| Best validation loss | 1.007 | 1.500 (target tokens) | 1.223 | 1.263 |

Validation losses come from each notebook's own held-out split and objective, so they are not directly comparable
(Arch-2 predicts 512 tokens from a 1,024-token context; the others are next-token losses over 2,048-token blocks).


## Results (Codabench, 1,000 test prompts)

| Model | Type | BERTScore F1 | ROUGE-L F1 | BLEU-4 | Own validation |
|---|---|---|---|---|---|
| [Architecture-1](arch1_decoder_only/) | Llama-style decoder-only LM | 0.8238 | 0.1679 | 0.0256 | validation loss 1.0066 (perplexity 2.74) after 6,000 steps |
| [Architecture-2](arch2_encoder_decoder/) | Transformer encoder-decoder (1,024 tokens → next 512) | **0.8289** | 0.1858 | 0.0329 | validation loss 1.4996 (perplexity 4.48) on the 512 target tokens after 34,000 steps |
| [Architecture-3](arch3_elegalformer/) | E-LegalFormer (grouped-query attention) | 0.8196 | 0.1630 | 0.0192 | best validation loss 1.2228 (perplexity 3.40) |
| [Architecture-4](arch4_hybrid_prefix_lm/) | Hybrid prefix-LM (bidirectional prefix + causal branch, gated fusion) | 0.8283 | 0.1945 | 0.0355 | best validation loss 1.2631 (perplexity 3.54) |

## Observations

* Arch-2 (encoder-decoder) has the best BERTScore (0.8289); Arch-4 (hybrid prefix-LM) has the best ROUGE-L (0.1945) and BLEU-4 (0.0355). The four models are within 0.01 BERTScore of each other.
* Arch-3 (GQA) scores lowest on all three metrics; it also saw the fewest training tokens (up to 8,000 steps of 8 × 2,048 tokens).
* Although Arch-2 wins Phase 1, the decoder-only Arch-1 gives the best features for most downstream phases (2, 3, 5, 6) and Arch-3 for Phase 4.

## Folder

| Folder | Content |
|---|---|
| `tokenizer/` | BPE tokenizer training, corpus tokenization, standalone `tokenizer.py` |
| `arch1_decoder_only/` | Architecture-1 pretraining + test generation |
| `arch2_encoder_decoder/` | Architecture-2 pretraining + test generation |
| `arch3_elegalformer/` | Architecture-3 pretraining + test generation |
| `arch4_hybrid_prefix_lm/` | Architecture-4 pretraining + test generation |
