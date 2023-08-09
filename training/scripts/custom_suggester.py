# -*- coding: utf-8 -*-
import re
from typing import Iterable, List, Optional, cast

import spacy
from spacy.pipeline.spancat import Suggester
from spacy.tokens import Doc, Span
from thinc.api import Ops, get_current_ops
from thinc.types import Ints1d, Ragged


def get_spans(doc: Doc, start: int, end: int) -> List[Span]:
    spans = []
    for length in range(1, end - start):
        for i in range(start, end - length):
            spans.append((doc[i : i + length].start, doc[i : i + length].end))
    return spans


@spacy.registry.misc("custom_suggester")
def build_custom_suggester() -> Suggester:
    """Custom suggerter"""
    return custom_suggester


def custom_suggester(docs: Iterable[Doc], *, ops: Optional[Ops] = None) -> Ragged:
    pattern = """["'«„].+?["'»“]"""
    if ops is None:
        ops = get_current_ops()
    spans = []
    lengths = []

    for doc in docs:
        cache = []
        length = 0

        for match in re.finditer(pattern, doc.text):
            span = doc.char_span(match.start(), match.end())
            if span:
                if (span.start, span.end) not in cache:
                    spans.append((span.start, span.end))
                    cache.append((span.start, span.end))
                    length += 1
            else:
                print(
                    f"Suggester warning: span is None, match: ({match.start()}, {match.end()}), span: {doc.text[match.start():match.end()]}, doc: {doc.text}"
                )

        start = 0
        for s, e in cache:
            sp = get_spans(doc, start, s)
            length += len(sp)
            spans += sp
            start = e
        sp = get_spans(doc, start, len(doc))
        length += len(sp)
        spans += sp
        start = e

        lengths.append(length)

    lengths_array = cast(Ints1d, ops.asarray(lengths, dtype="i"))
    if len(spans) > 0:
        output = Ragged(ops.asarray(spans, dtype="i"), lengths_array)
    else:
        output = Ragged(ops.xp.zeros((0, 0), dtype="i"), lengths_array)

    return output
