"""Standalone legal BPE tokenizer (written out by 01_bpe_tokenizer.ipynb, loaded by every later notebook).

Usage:
    from tokenizer import load_tokenizer
    tokenizer, config = load_tokenizer("path/to/final_tokenizer")
"""

import json
import re
import unicodedata


# ============================================================
# PRETOKENIZATION
# ============================================================

PRETOKEN_PATTERN = re.compile(
    r"""
    \n+
    |[ \t]*[\w]+(?:['’][\w]+)*
    |[ \t]*[^\w\s]
    |[ \t]+
    """,
    re.VERBOSE | re.UNICODE,
)


def normalize_text(text: str) -> str:

    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    text = text.replace(
        "\x00",
        "",
    )

    return text


def pretokenize(text: str):

    text = normalize_text(text)

    return PRETOKEN_PATTERN.findall(
        text
    )


# ============================================================
# BPE TOKENIZER
# ============================================================

class BPETokenizer:

    def __init__(
        self,
        vocab,
        merges,
        special_tokens=None,
    ):

        self.vocab = {
            str(token): int(idx)
            for token, idx in vocab.items()
        }

        self.id_to_token = {
            idx: token
            for token, idx in self.vocab.items()
        }

        self.merges = [
            tuple(pair)
            for pair in merges
        ]

        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(
                self.merges
            )
        }

        self.vocab_size = len(
            self.vocab
        )

        if special_tokens is None:

            special_tokens = {
                "pad_token": "<pad>",
                "unk_token": "<unk>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            }

        self.special_tokens = (
            special_tokens
        )

        self.pad_token = (
            special_tokens["pad_token"]
        )

        self.unk_token = (
            special_tokens["unk_token"]
        )

        self.bos_token = (
            special_tokens["bos_token"]
        )

        self.eos_token = (
            special_tokens["eos_token"]
        )

        self.pad_id = self.vocab.get(
            self.pad_token
        )

        self.unk_id = self.vocab.get(
            self.unk_token
        )

        self.bos_id = self.vocab.get(
            self.bos_token
        )

        self.eos_id = self.vocab.get(
            self.eos_token
        )

    # ========================================================
    # APPLY BPE
    # ========================================================

    def _apply_bpe(
        self,
        symbols,
    ):

        symbols = list(symbols)

        while len(symbols) >= 2:

            best_pair = None

            best_rank = float("inf")

            # Find the highest-priority merge
            # currently applicable.

            for i in range(
                len(symbols) - 1
            ):

                pair = (
                    symbols[i],
                    symbols[i + 1],
                )

                rank = self.merge_ranks.get(
                    pair
                )

                if (
                    rank is not None
                    and rank < best_rank
                ):

                    best_rank = rank

                    best_pair = pair

            if best_pair is None:
                break

            merged = []

            i = 0

            while i < len(symbols):

                if (
                    i < len(symbols) - 1
                    and symbols[i]
                    == best_pair[0]
                    and symbols[i + 1]
                    == best_pair[1]
                ):

                    merged.append(
                        symbols[i]
                        + symbols[i + 1]
                    )

                    i += 2

                else:

                    merged.append(
                        symbols[i]
                    )

                    i += 1

            symbols = merged

        return symbols

    # ========================================================
    # ENCODE
    # ========================================================

    def encode(
        self,
        text,
    ):

        if not text:
            return []

        text = normalize_text(
            text
        )

        ids = []

        pieces = pretokenize(
            text
        )

        for piece in pieces:

            symbols = list(
                piece
            )

            symbols = self._apply_bpe(
                symbols
            )

            for token in symbols:

                token_id = self.vocab.get(
                    token
                )

                if token_id is not None:

                    ids.append(
                        token_id
                    )

                else:

                    # Character fallback.
                    for ch in token:

                        ch_id = self.vocab.get(
                            ch
                        )

                        if ch_id is None:

                            ids.append(
                                self.unk_id
                            )

                        else:

                            ids.append(
                                ch_id
                            )

        return ids

    # ========================================================
    # DECODE
    # ========================================================

    def decode(
        self,
        ids,
    ):

        tokens = []

        for idx in ids:

            idx = int(idx)

            tokens.append(
                self.id_to_token.get(
                    idx,
                    self.unk_token,
                )
            )

        return "".join(
            tokens
        )

    # ========================================================
    # VOCAB SIZE
    # ========================================================

    def get_vocab_size(self):

        return self.vocab_size


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer(
    tokenizer_dir,
):

    tokenizer_dir = str(
        tokenizer_dir
    )

    vocab_path = (
        f"{tokenizer_dir}/vocab.json"
    )

    merges_path = (
        f"{tokenizer_dir}/merges.txt"
    )

    config_path = (
        f"{tokenizer_dir}/tokenizer_config.json"
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    with open(
        vocab_path,
        "r",
        encoding="utf-8",
    ) as f:

        vocab = json.load(f)

    # --------------------------------------------------------
    # JSONL merges
    # --------------------------------------------------------

    merges = []

    with open(
        merges_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.rstrip(
                "\n"
            )

            if not line:
                continue

            pair = json.loads(
                line
            )

            if (
                not isinstance(pair, list)
                or len(pair) != 2
            ):

                raise ValueError(
                    "Invalid merge entry: "
                    f"{pair}"
                )

            merges.append(
                (
                    pair[0],
                    pair[1],
                )
            )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(f)

    # --------------------------------------------------------
    # Construct tokenizer
    # --------------------------------------------------------

    tokenizer = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=config[
            "special_tokens"
        ],
    )

    return tokenizer, config
