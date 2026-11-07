# Reproducing a run (Kaggle)

All notebooks were run on Kaggle (GPU T4 / P100, ≤ 12 h per session). Each one is self-contained: CELL 1 imports, CELL 2 holds
every setting, the last cell writes `prediction.json` and `submission.zip`.

## 1. One-time setup

1. Accept the dataset terms of `Exploration-Lab/CS-781-Capstone` on Hugging Face and create a read token.
2. In Kaggle: *Add-ons → Secrets* → add the token under the name **`HF`** (the notebooks read it with
   `UserSecretsClient().get_secret("HF")`; no token is ever written in a notebook).
3. Enable *Internet* and a GPU accelerator for the notebook.

## 2. Phase 1 (once)

| Order | Notebook | Kaggle inputs | Produces |
|---|---|---|---|
| 1 | `phase1_pretraining/tokenizer/01_bpe_tokenizer.ipynb` | — | `legal_tokenizer_v3/tokenizer/final_tokenizer/` |
| 2 | `phase1_pretraining/tokenizer/02_tokenize_corpus.ipynb` | output of 1 | `<source>_100m.bin` token files |
| 3 | `phase1_pretraining/arch*/…ipynb` | outputs of 1 and 2 | best checkpoint (`best_model.pt`, `best_encoder_decoder.pt`, …) |

The Phase-1 runs are resumable: attach the notebook's previous output and it continues from `last.pt`.

## 3. Phases 2–7

1. Attach as Kaggle inputs: the tokenizer notebook output and the output of the Phase-1 notebook(s) the run starts from
   (both, for distillation notebooks).
2. Check the paths at the top of CELL 2 (`TOKENIZER_DIR`, `CHECKPOINT_PATH`). If a checkpoint is not found there, the notebooks
   search `/kaggle/input` for a matching file.
3. Optional: set `DEBUG_MODE = True` in CELL 2 for a few-minute run on a handful of documents (pipeline check).
4. *Run All*. The notebook prints the dev report and writes `/kaggle/working/<RUN_NAME>/submission.zip`
   (with `prediction.json` at the root, as Codabench requires).

## Phase 7 notes

* Run `p7_arch1_lorma` before the distillation notebook and attach its output: the KD notebook loads `retriever.pt` as the teacher
  (otherwise it trains the teacher itself first).
* `REUSE_TRAINED = True` in CELL 2 loads the `retriever.pt` of an earlier run of the same notebook instead of training
  (useful to re-rank with a changed fusion or re-ranker without retraining).

## What is not in this repository

* model checkpoints (`*.pt`) — too large and not needed to read the code;
* the competition data — gated on Hugging Face, downloaded by every notebook at run time;
* Kaggle secrets / tokens.
