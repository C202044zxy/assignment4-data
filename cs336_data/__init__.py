import importlib.metadata
from .fliter_common_crawl import (
    extract_text_from_html_bytes,
    identify_language,
    mask_emails,
    mask_phone_numbers,
    mask_ips,
)

__version__ = importlib.metadata.version("cs336-data")

__all__ = [
    "extract_text_from_html_bytes",
    "identify_language",
    "mask_emails",
    "mask_phone_numbers",
    "mask_ips",
]