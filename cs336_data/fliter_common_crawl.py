from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from resiliparse.extract.html2text import extract_plain_text
from fastwarc.warc import ArchiveIterator, WarcRecordType
from pathlib import Path
from typing import Any
import fasttext


proj_root = Path(__file__).parent.parent
warc_path = proj_root / "data/CC/example.warc.gz"
model_path = proj_root / "data/classifier/lid.176.bin"


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


def _get_lang_model():
    global _lang_model
    if _lang_model is None:
        _lang_model = fasttext.load_model(str(model_path))
    return _lang_model


def identify_language(text: str) -> tuple[str, float]:
    model = _get_lang_model()
    cleaned = text.replace("\n", " ")
    labels, probs = model.predict(cleaned)
    print(labels, probs)
    lang = labels[0].replace("__label__", "")
    score = float(probs[0])
    return lang, score


if __name__ == "__main__":
    parse_example_warc(warc_path)