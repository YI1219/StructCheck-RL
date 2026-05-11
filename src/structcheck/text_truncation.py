"""Tokenizer truncation helpers."""

from contextlib import contextmanager
from typing import Any


@contextmanager
def truncation_side_left(tokenizer: Any):
    old = getattr(tokenizer, "truncation_side", "right")
    try:
        tokenizer.truncation_side = "left"
        yield
    finally:
        tokenizer.truncation_side = old
