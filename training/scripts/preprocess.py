# -*- coding: utf-8 -*-
import json
from pathlib import Path

import spacy
import typer
from spacy.tokens import DocBin, Span
from wasabi import Printer

msg = Printer()


def main(
    json_loc: Path,
    train_file: Path,
    dev_file: Path,
    eval_split: float,
    span_key: str,
):
    """Parse the annotations into a training and development set for Spancat."""

    docs = []
    nlp = spacy.blank("en")
    total_span_count = {}
    max_span_length = 0

    msg.info(f"Processing {json_loc.name}")
    # Load dataset
    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            if example["answer"] == "accept":
                doc = nlp(example["text"])
                spans = []

                if "spans" in example:
                    if example["spans"] is None:
                        example["spans"] = []
                    for span in example["spans"]:  # type: ignore
                        spans.append(
                            Span(
                                doc,
                                span["token_start"],
                                span["token_end"] + 1,
                                span["label"],
                            )
                        )

                        if span["label"] not in total_span_count:
                            total_span_count[span["label"]] = 0

                        total_span_count[span["label"]] += 1

                        span_length = (span["token_end"] + 1) - span["token_start"]
                        if span_length > max_span_length:
                            max_span_length = span_length

                doc.spans[span_key] = spans
                docs.append(doc)

    # Split
    train = []
    dev = []

    split = int(len(docs) * eval_split)
    train = docs[split:]
    dev = docs[:split]

    # Save to disk
    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(train_file)

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(dev_file)

    # Info
    msg.info(f"Examples: {len(docs)}")
    for key in total_span_count:
        msg.info(f"{key}: {total_span_count[key]}")
    msg.info(f"Max span lenght: {max_span_length}")
    msg.good(f"Processing {json_loc.name} done")


if __name__ == "__main__":
    typer.run(main)
