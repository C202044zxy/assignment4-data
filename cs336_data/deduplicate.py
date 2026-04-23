import os
import hashlib
from collections import Counter
from nltk.tokenize import word_tokenize


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    counts: Counter[str] = Counter()
    for input_file in input_files:
        with open(input_file, "rb") as f:
            text = f.read().decode("utf-8")
        for line in text.splitlines():
            counts[hashlib.sha256(line.encode("utf-8")).hexdigest()] += 1

    os.makedirs(output_directory, exist_ok=True)
    for input_file in input_files:
        with open(input_file, "rb") as f:
            text = f.read().decode("utf-8")

        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_directory, file_name)
        with open(output_file, "wb") as f:
            for line in text.splitlines():
                hash_value = hashlib.sha256(line.encode("utf-8")).hexdigest()
                if counts[hash_value] == 1:
                    f.write((line + "\n").encode("utf-8"))


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    for input_file in input_files:
        pass