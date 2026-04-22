from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from resiliparse.extract.html2text import extract_plain_text
from fastwarc.warc import ArchiveIterator, WarcRecordType
from pathlib import Path


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


if __name__ == "__main__":
    proj_root = Path(__file__).parent.parent
    file_path = proj_root / "data/CC/example.warc.gz"
    parse_example_warc(file_path)