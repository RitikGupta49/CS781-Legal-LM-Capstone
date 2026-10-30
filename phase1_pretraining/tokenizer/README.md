# Legal BPE tokenizer (Phase 1)

| File | What it does |
|---|---|
| `01_bpe_tokenizer.ipynb` | downloads the Phase-1 corpus, normalizes + pre-tokenizes it, collects piece frequencies and trains the BPE merges (incremental pair-count updates), then writes the tokenizer package (`vocab.json`, merges, `tokenizer_config.json`, `tokenizer.py`, manifest with file hashes) |
| `02_tokenize_corpus.ipynb` | streams the four JSON sources, tokenizes every document with the final tokenizer and writes one `uint16` token file per source (`<source>_100m.bin`, `<eos>` between documents) + metadata |
| `tokenizer.py` | the standalone module the notebook writes (normalization, pre-tokenization regex, BPE encode/decode, `load_tokenizer`) |

Vocabulary 32,004 · special tokens `<pad>`=0 `<unk>`=1 `<bos>`=2 `<eos>`=3 · NFKC normalization · byte-exact `decode(encode(text)) == normalize(text)`.

### Notebook cells — `01_bpe_tokenizer.ipynb`
1. PHASE 1 — LEGAL BPE TOKENIZER V2
2. Hugging Face dataset
3. NORMALIZATION
4. PRE-TOKENIZATION — LEGAL CORPUS V3
5. Read ONLY the first JSON object
6. TOKENIZER CONFIGURATION
7. STREAMING LEGAL DOCUMENT READER
8. INSPECT DATA
9. PIECE FREQUENCIES
10. COLLECT TOKENIZER TRAINING STATISTICS
11. INITIAL VOCABULARY
12. VALIDATE INITIAL VOCABULARY
13. PREPARE BPE TRAINING PIECES
14. INITIAL BPE SEQUENCES
15. BPE PAIR COUNTING
16. TEST PAIR COUNTING
17. TEST FREQUENCY WEIGHTING
18. BUILD BPE SEQUENCES
19. INITIAL BPE PAIR STATISTICS
20. APPLY ONE BPE MERGE
21. TEST MERGE OPERATION
22. BPE TRAINING
23. SMALL BPE TEST
24. FINAL CHECKS BEFORE REAL BPE TRAINING
25. OPTIMIZED INCREMENTAL BPE — INITIALIZE STATE
26. INCREMENTAL BPE — STATE UPDATE
27. INCREMENTAL BPE — SELECT BEST VALID PAIR
28. TEST INCREMENTAL BPE STATE
29. BUILD REAL INCREMENTAL BPE STATE
30. INCREMENTAL BPE TRAINER
31. START REAL BPE TRAINING
32. V3 PRE-TOKENIZATION CHECKS
33. FINAL TOKENIZER PACKAGE

### Notebook cells — `02_tokenize_corpus.ipynb`
1. CELL 1 — IMPORTS + CONFIGURATION
2. CELL 2 — LOAD FINAL BPE TOKENIZER
3. CELL 3 — DOWNLOAD TRAINING FILES
4. CELL 4 — NORMALIZATION
5. CELL 5 — WORD-LIKE PIECE DEFINITION
6. CELL 6 — STREAM JSON DOCUMENTS
7. CELL 7 — TOKENIZE ONE DOCUMENT
8. CELL 8 — STREAMING TOKEN WRITER
9. CELL 9 — TOKENIZER PRETOKENIZER ACCESS
10. CELL 10 — SMALL SANITY TEST
11. CELL 11 — TOKENIZE ALL FOUR SOURCES
12. CELL 12 — SAVE TOKENIZATION METADATA
13. CELL 13 — VERIFY TOKENIZED DATA
14. CELL 14 — FINAL DATASET INVENTORY
