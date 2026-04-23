import importlib.metadata
from .fliter_common_crawl import (
    extract_text_from_html_bytes,
    identify,
    mask_emails,
    mask_phone_numbers,
    mask_ips,
    gopher_quality_filter,
)
from .deduplicate import(
    exact_line_deduplication,
)

__version__ = importlib.metadata.version("cs336-data")

__all__ = [
    "extract_text_from_html_bytes",
    "identify",
    "mask_emails",
    "mask_phone_numbers",
    "mask_ips",
    "gopher_quality_filter",
    "exact_line_deduplication",
]