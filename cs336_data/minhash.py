import os
import shutil
from nltk.tokenize import word_tokenize
import mmh3
from itertools import combinations


def _ngrams(tokens: list[str], n: int) -> list[tuple[str]]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def minhash_signature(ngrams: list[tuple[str]], k: int, seeds: list[int]):
    sig = [float('inf')] * k
    for ng in ngrams:
        b = " ".join(ng).encode("utf-8")
        for i in range(k):
            h = mmh3.hash(b, seeds[i])
            if h < sig[i]:
                sig[i] = h
    return sig


def matches(sig_x: list[int], sig_y: list[int], jaccard_threshold: float):
    assert len(sig_x) == len(sig_y)
    agreements = sum(1 for x, y in zip(sig_x, sig_y) if x == y)
    estimated_jaccard = agreements / len(sig_x)
    return estimated_jaccard >= jaccard_threshold


def dfs(x: int, vis: dict, edges: dict):
    vis.add(x)
    for y in edges.get(x, []):
        if y in vis:
            continue
        dfs(y, vis, edges)


def minhash_deduplication(
    input_files: list[os.PathLike],
    k: int,
    b: int,
    n: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    assert k % b == 0
    r = k // b  # rows per band
    # generate minhash signatures
    seeds = list(range(k))
    sigs: list[list[float]] = []
    for input_file in input_files:
        with open(input_file, "rb") as f:
            text = f.read().decode("utf-8")
        tokens = word_tokenize(text)
        ngrams = _ngrams(tokens, n)
        sigs.append(minhash_signature(ngrams, k, seeds))

    # LSH - find candidate pairs
    grps: dict[tuple[int, int], list[int]] = {}
    for i in range(len(sigs)):
        sig = sigs[i]
        for band_idx in range(b):
            band = sig[band_idx * r : (band_idx + 1) * r]
            key = (band_idx, mmh3.hash(str(band).encode("utf-8")))
            grps.setdefault(key, []).append(i)

    # cluster files
    edges: dict[int, list[int]] = {}
    checked: set[tuple[int, int]] = set()
    for grp in grps.values():
        if len(grp) < 2:
            continue
        for x, y in combinations(sorted(set(grp)), 2):
            if (x, y) in checked:
                continue
            checked.add((x, y))
            if matches(sigs[x], sigs[y], jaccard_threshold):
                edges.setdefault(x, []).append(y)
                edges.setdefault(y, []).append(x)
    
    os.makedirs(output_directory, exist_ok=True)
    vis: set[int] = set()
    for i, input_file in enumerate(input_files):
        if i in vis:
            continue
        dfs(i, vis, edges)
        shutil.copy(input_file, output_directory)
