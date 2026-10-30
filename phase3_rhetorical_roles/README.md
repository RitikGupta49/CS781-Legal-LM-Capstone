# Phase 3 — Rhetorical Role Prediction (RR)

**Task.** Label **every sentence** of a judgment with its rhetorical role (Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, PrecedentReliedUpon, PrecedentNotReliedUpon, RatioOfTheDecision, RulingByLowerCourt, RulingByPresentCourt, None).

**Data.** Two domains — Corporate Law (CL) and Information Technology (IT) — with train and dev files each (dev: 5 CL documents / 1,396 sentences, 5 IT documents / 746 sentences). Test: `CL_test.json` + `IT_test.json` (10 documents, 2,197 sentences).

**Metric.** Macro-F1

## Shared pipeline

1. **Sentences → windows.** Every sentence is tokenized (cut at 256 tokens). With `CONTEXT = "document"` consecutive sentences are packed
   into 1,024-token windows, so each sentence is read together with the sentences before it.
2. **Sentence vectors.** Phase-1 backbone (frozen + PEFT) → mean of the token vectors of each sentence, plus its relative position in the
   document and a domain flag.
3. **Document-level head.** BiLSTM over the sentence vectors of the whole document → linear-chain **CRF** over the roles.
   Loss = CRF + 0.5 × class-weighted cross-entropy (weights = inverse frequency, capped at 5) because macro-F1 counts every role equally.
4. **Constraints.** Roles that never occur in a domain's training data are never predicted for that domain; every sentence gets exactly one label.
5. **Protocol.** Trained on the train files only; the dev files pick the best epoch (and, for IA3, CRF vs. best-role decoding).
6. **Output.** `prediction.json` maps every test document id to its ordered list of labels (same length as its sentence list).


## Submissions

| # | Submission | Backbone | Method | Codabench Macro-F1 | Codabench Macro-P | Codabench Macro-R | Own dev / validation |
|---|---|---|---|---|---|---|---|
| 01 | [01_arch1_lora](01_arch1_lora/) | Arch-1 | Arch-1 + LoRA | 0.4362 | 0.4862 | 0.45 | dev macro-F1: CL 0.4145, IT 0.5796, mean 0.4970 (best epoch 5) |
| 02 | [02_arch1_lorma](02_arch1_lorma/) | Arch-1 | Arch-1 + LoRMA | **0.5266** | 0.5735 | 0.5141 | dev macro-F1: CL 0.4778, IT 0.5524, mean 0.5151 (best epoch 13) |
| 03 | [03_arch1_ia3](03_arch1_ia3/) | Arch-1 | Arch-1 + IA3 | 0.37 | 0.3892 | 0.3872 | dev macro-F1 0.4449 (best epoch 12) |
| 04 | [04_kd_arch1_to_arch2](04_kd_arch1_to_arch2/) | Arch-2 | KD Arch-1 → Arch-2 | 0.36 | 0.425 | 0.3592 | dev mean macro-F1: teacher 0.4816, student 0.3602 |
| 05 | [05_arch2_lora](05_arch2_lora/) | Arch-2 | Arch-2 + LoRA | 0.345 | 0.402 | 0.3458 | dev macro-F1: CL 0.3196, IT 0.3721, mean 0.3459 (best epoch 11) |
| 06 | [06_arch2_ia3](06_arch2_ia3/) | Arch-2 | Arch-2 + IA3 | 0.2789 | 0.3675 | 0.285 | dev macro-F1: CL 0.3203, IT 0.3917, mean 0.3560 (best epoch 13) |
| 07 | [07_arch3_ia3](07_arch3_ia3/) | Arch-3 | Arch-3 + IA3 | 0.421 | 0.437 | 0.437 | dev macro-F1: CL 0.4002, IT 0.4558, all 0.4777 (best epoch 9) |

Best: **02_arch1_lorma** — Architecture-1 decoder (frozen) + LoRMA + sentence BiLSTM + CRF, Codabench Macro-F1 0.5266.

## Observations

* Arch-1 + LoRMA is the best Phase-3 system (0.5266); the same pipeline with LoRA reaches 0.4362, so LoRMA's multiplicative update helps here.
* IA3 trains only ~30–40k backbone weights: it works on Arch-3 (0.421) but is weak on Arch-1 (0.37) and Arch-2 (0.279).
* The Arch-2 encoder gives the weakest sentence features (0.279–0.36), also as a distilled student.
* IT documents are consistently easier than CL documents on dev (e.g. 0.55 vs 0.48 macro-F1 for LoRMA).

Every submission folder holds the notebook (with its training log), `submission.zip` as uploaded to Codabench, and a README
with the method, the exact configuration and a map of the notebook cells.
