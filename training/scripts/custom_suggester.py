import re

import spacy
from spacy.tokens import Doc, Span
from spacy.pipeline.spancat import Suggester

from typing import List, Optional, Iterable, cast
from thinc.api import get_current_ops, Ops
from thinc.types import Ragged, Ints1d

def get_spans(doc:Doc, start:int, end:int) -> List[Span]:
    spans = []
    for l in range(1, end - start):
        for i in range(start, end-l):
            spans.append((doc[i:i+l].start, doc[i:i+l].end))
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
            if (span.start, span.end) not in cache:
                spans.append((span.start, span.end))
                cache.append((span.start, span.end))
                length += 1
        
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