from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from resiliparse.extract.html2text import extract_plain_text
from fastwarc.warc import ArchiveIterator, WarcRecordType
from pathlib import Path
from typing import Any
import fasttext
import regex as re
from nltk.tokenize import word_tokenize


proj_root = Path(__file__).parent.parent
warc_path = proj_root / "data/CC/example.warc.gz"
model_path = proj_root / "data/classifier"
lang_model_path = model_path / "lid.176.bin"
nsfw_model_path = model_path / "jigsaw_fasttext_bigrams_nsfw_final.bin"
hatespeech_model_path = model_path / "jigsaw_fasttext_bigrams_hatespeech_final.bin"


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    coding = detect_encoding(html_bytes)
    html = bytes_to_str(html_bytes, coding)
    text = extract_plain_text(html)
    return text


def parse_example_warc(file_path: str):
    with open(file_path, "rb") as stream:
        cnt = 0
        for record in ArchiveIterator(stream):
            if record.record_type != WarcRecordType.response:
                continue
            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)
            if cnt >= 3:
                break
            cnt += 1


_lang_model = None
_nsfw_model = None
_hatespeech_model = None


def _get_model(name: str):
    match name:
        case "lang":
            global _lang_model
            if _lang_model is None:
                _lang_model = fasttext.load_model(str(lang_model_path))
            return _lang_model

        case "nsfw":
            global _nsfw_model
            if _nsfw_model is None:
                _nsfw_model = fasttext.load_model(str(nsfw_model_path))
            return _nsfw_model

        case "hatespeech":
            global _hatespeech_model
            if _hatespeech_model is None:
                _hatespeech_model = fasttext.load_model(str(hatespeech_model_path))
            return _hatespeech_model

    return None


def identify(text: str, type: str) -> tuple[str, float]:
    model = _get_model(type)
    cleaned = text.replace("\n", " ")
    labels, probs = model.predict(cleaned)
    lang = labels[0].replace("__label__", "")
    score = float(probs[0])
    return lang, score


def mask_emails(text: str) -> tuple[str, int]:
    mask = '|||EMAIL_ADDRESS|||'
    pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    new_text, num = re.subn(pattern, mask, text)
    return (new_text, num)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    mask = '|||PHONE_NUMBER|||'
    pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    new_text, num = re.subn(pattern, mask, text)
    return (new_text, num)


def mask_ips(text: str) -> tuple[str, int]:
    mask = '|||IP_ADDRESS|||'
    pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    new_text, num = re.subn(pattern, mask, text)
    return (new_text, num)


def gopher_quality_filter(text: str) -> bool:
    tokens = word_tokenize(text)
    n = len(tokens)
    if n < 50 or n > 100000:
        return False

    mean_len = sum(len(token) for token in tokens) / n
    if mean_len < 3 or mean_len > 10:
        return False
    
    lines = text.splitlines()
    ellipsis_lines = sum(
        1 for line in lines
        if line.rstrip().endswith("...")
    )
    if ellipsis_lines / len(lines) > 0.3:
        return False

    alpha_words = sum(1 for token in tokens if any(c.isalpha() for c in token))
    if alpha_words / n < 0.8:
        return False
    
    return True