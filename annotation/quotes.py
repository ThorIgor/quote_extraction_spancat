import numpy as np

from typing import List, Optional, Union, Iterable

import spacy
from spacy.language import Language
from spacy.training import Example
from spacy.tokens.doc import SetEntsDefault

import copy
import re
from collections import Counter

from prodigy.components.preprocess import add_tokens, make_raw_doc
from prodigy.components.sorters import prefer_low_scores
from prodigy.components.loaders import get_stream
from prodigy.core import recipe, Controller
from prodigy.util import set_hashes, log, split_string, get_labels, color
from prodigy.util import BINARY_ATTR
from prodigy.util import INPUT_HASH_ATTR, msg
from prodigy.types import StreamType, RecipeSettingsType

from cleanlab.multiannotator import get_active_learning_scores

def add_author(stream:StreamType, author:str):
    for eg in stream:
        eg['meta']['author'] = author
        yield eg 

def chunks(text:str, max:int):
    if len(text) < max:
        return [text]
    return [text[i:i+max] for i in range(0, len(text), max)]

def get_cleanlab_scores(nlp:Language, stream:StreamType, batch_size:int):
    while True:
        msg.info(f"Computing clenalab  scores in batch. Size: {batch_size}. Could take some time")
        batch = []
        for i, s in enumerate(stream):
            if i > batch_size:
                break
            batch.append(s)
        if batch:
            docs = list(nlp.pipe([s['text'] for s in batch]))
            # computing confidence socres
            inds, scores = nlp.get_pipe("spancat").predict(docs)    
            
            # computing cleanlab scores
            _, scores = get_active_learning_scores(pred_probs_unlabeled=scores)
            
            # mean socres in each document for spancat
            count = 0
            new_scores = []
            for l in inds.lengths:
                if l > 0:
                    new_scores.append(np.mean(scores[count:count+l]))
                else:
                    new_scores.append(1)
                count+=l
            
            # mean socres in each document for span_finder
            if "span_finder" in nlp.pipe_names:
                finder_scores = nlp.get_pipe("span_finder").predict(docs)
                _, finder_scores = get_active_learning_scores(pred_probs_unlabeled=finder_scores)

                count = 0
                new_finder_scores = []
                for doc in docs:
                    l = len(doc)
                    if l > 0:
                        new_finder_scores.append(np.mean(finder_scores[count:count+l]))
                    else:
                        new_finder_scores.append(1)
                    count+=l
            
            for i, el in enumerate(batch):
                score = new_scores[i]
                if "span_finder" in nlp.pipe_names:
                    score = min(new_finder_scores[i], score)
                if 'meta' not in el:
                    el['meta'] = {}
                el['meta']['score'] = score
                yield (score, el)
        else:
            break


@recipe(
    "quotes.manual",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline for tokenization or blank:lang (e.g. blank:en)", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    exclude=("Dataset name or comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    author=("Name of annotator", "option", "a", str),
    #unsegmented=("Don't get only parts with quotes", "flag", "U", bool),
    # fmt: on
)
def manual(
        dataset: str,
        spacy_model: str,
        source: Union[str, Iterable[dict]],
        loader: Optional[str] = None,
        label: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        author: Optional[str] = None,
        #unsegmented: bool = False,
) -> RecipeSettingsType:
    """
    Mark spans by token. Requires only a tokenizer, and doesn't do any active learning. 
    The recipe will present all examples in order, so even examples without matches are shown. 
    """
    log("RECIPE: Starting recipe quotes.manual", locals())
    blocks = [{"view_id": "spans_manual"},
              {"view_id": "text_input",
               "field_rows": 3,
               "field_id": "feedback",
               "field_label": "Optional feedback",
               "field_placeholder": "Type here..."}]
    nlp = spacy.load(spacy_model)
    labels = label  # comma-separated list or path to text file
    if not labels:
        labels = nlp.pipe_labels.get("ner", [])
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    stream = get_stream(
        source, loader=loader, rehash=True, dedup=True, input_key="text", is_binary=False
    )
    if author:
        stream = add_author(stream, author)
    # if not unsegmented:
    #     stream = get_parts_with_quotes(stream, nlp.lang)
    # Add "tokens" key to the tasks, either with words or characters
    stream = add_tokens(nlp, stream)

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "exclude": exclude,
        "on_exit": print_results,
        "before_db": None,
        "config": {
            "labels": labels,
            "exclude_by": "input",
            "auto_count_stream": True,
            "blocks": blocks,
            "buttons": ["accept", "ignore", "undo"],
        },
    }


@recipe(
    "quotes.teach",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline with an entity recognizer", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    exclude=("Dataset name or comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    batch_size=("Batch size for cleanlab socres computing", "option", "b", int),
    author=("Name of annotator", "option", "a", str),
    #unsegmented=("Don't get only parts with quotes", "flag", "U", bool),
    # fmt: on
)
def teach(
        dataset: str,
        spacy_model: str,
        source: Union[str, Iterable[dict]],
        loader: Optional[str] = None,
        label: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        batch_size: int = 64,
        author: Optional[str] = None,
        #unsegmented: bool = False,
) -> RecipeSettingsType:
    """
    Collect the best possible training data for a spancat model. 
    Prodigy will decide which questions to ask next based on cleanlab score.
    """
    log("RECIPE: Starting recipe quotes.teach", locals())
    blocks = [{"view_id": "spans"},
              {"view_id": "text_input",
               "field_rows": 3,
               "field_id": "feedback",
               "field_label": "Optional feedback",
               "field_placeholder": "Type here..."}]
    
    nlp = spacy.load(spacy_model)
    labels = nlp.get_pipe("spancat").labels
    spans_key = nlp.get_pipe("spancat").cfg['spans_key']
    log(f"RECIPE: Creating EntityRecognizer using model {spacy_model}")
    if not label:
        label = labels
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")

    stream = get_stream(
        source, loader=loader, rehash=True, dedup=True, input_key="text"
    )
    if author:
        stream = add_author(stream, author)
    # if not unsegmented:
    #     stream = get_parts_with_quotes(stream, nlp.lang)
    stream = prefer_low_scores(get_cleanlab_scores(nlp, stream, batch_size))

    def make_tasks(nlp: Language, stream: StreamType) -> StreamType:
        """Add a 'spans' key to each example, with predicted entities."""
        texts = ((eg["text"], eg) for eg in stream)
        for doc, eg in nlp.pipe(texts, as_tuples=True, batch_size=10):
            task = copy.deepcopy(eg)
            spans = []
            for ent in doc.spans[spans_key]:
                if labels and ent.label_ not in labels:
                    continue
                spans.append(
                    {
                        "token_start": ent.start,
                        "token_end": ent.end - 1,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "text": ent.text,
                        "label": ent.label_,
                        "source": spacy_model,
                        "input_hash": eg[INPUT_HASH_ATTR],
                    }
                )
            task["spans"] = spans
            task[BINARY_ATTR] = False
            task = set_hashes(task)
            yield task
    
    stream = make_tasks(nlp, stream)

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": (eg for eg in stream),
        "exclude": exclude,
        "on_exit": print_results,
        "config": {
            "lang": nlp.lang,
            "label": ", ".join(label) if label is not None else "all",
            "blocks": blocks,
            "buttons": ["accept", "reject"],
        },
    }

@recipe(
    "quotes.correct",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    spacy_model=("Loadable spaCy pipeline with an entity recognizer", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    exclude=("Dataset name or comma-separated list of dataset IDs whose annotations to exclude", "option", "e", split_string),
    batch_size=("Batch size for cleanlab socres computing", "option", "b", int),
    author=("Name of annotator", "option", "a", str),
    #unsegmented=("Don't get only parts with quotes", "flag", "U", bool),
    update=("Whether to update the model during annotation", "flag", "UP", bool),
    # fmt: on
)
def correct(
        dataset: str,
        spacy_model: str,
        source: Union[str, Iterable[dict]],
        loader: Optional[str] = None,
        label: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        batch_size: int = 64,
        author: Optional[str] = None,
        #unsegmented: bool = False,
        update: bool = False,
) -> RecipeSettingsType:
    """
    Create gold data for spancat by correcting a model's suggestions.
    Prodigy will decide which questions to ask next based on cleanlab score
    """
    log("RECIPE: Starting recipe quotes.correct", locals())
    blocks = [{"view_id": "spans_manual"},
              {"view_id": "text_input",
               "field_rows": 3,
               "field_id": "feedback",
               "field_label": "Optional feedback",
               "field_placeholder": "Type here..."}]
    nlp = spacy.load(spacy_model)
    labels = nlp.get_pipe("spancat").labels
    spans_key = nlp.get_pipe("spancat").cfg['spans_key']
    if not label:
        label = labels
        if not labels:
            msg.fail("No --label argument set and no labels found in model", exits=1)
        msg.text(f"Using {len(labels)} labels from model: {', '.join(labels)}")
    # Check if we're annotating all labels present in the model or a subset
    no_missing = len(set(label).intersection(set(labels))) == len(labels)
    log(f"RECIPE: Annotating with {len(labels)} labels", labels)
    stream = get_stream(
        source, loader=loader, rehash=True, dedup=True, input_key="text"
    )
    if author:
        stream = add_author(stream, author)
    # if not unsegmented:
    #     stream = get_parts_with_quotes(stream, nlp.lang)
    stream = prefer_low_scores(get_cleanlab_scores(nlp, stream, batch_size))
    stream.first_n = batch_size
    stream = add_tokens(nlp, stream)

    def make_tasks(nlp: Language, stream: StreamType) -> StreamType:
        """Add a 'spans' key to each example, with predicted entities."""
        texts = ((eg["text"], eg) for eg in stream)
        for doc, eg in nlp.pipe(texts, as_tuples=True, batch_size=10):
            task = copy.deepcopy(eg)
            spans = []
            for ent in doc.spans[spans_key]:
                if labels and ent.label_ not in labels:
                    continue
                spans.append(
                    {
                        "token_start": ent.start,
                        "token_end": ent.end - 1,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "text": ent.text,
                        "label": ent.label_,
                        "source": spacy_model,
                        "input_hash": eg[INPUT_HASH_ATTR],
                    }
                )
            task["spans"] = spans
            task[BINARY_ATTR] = False
            task = set_hashes(task)
            yield task

    def make_update(answers: Iterable[dict]) -> None:
        log(f"RECIPE: Updating model with {len(answers)} answers")
        examples = []
        for eg in answers:
            if eg["answer"] == "accept":
                doc = make_raw_doc(nlp, eg)
                ref = make_raw_doc(nlp, eg)
                spans = [
                    doc.char_span(span["start"], span["end"], label=span["label"])
                    for span in eg.get("spans", [])
                ]
                value = SetEntsDefault.outside if no_missing else SetEntsDefault.missing
                ref.set_ents(spans, default=value)
                ref.spans[spans_key] = spans
                if "span_finder" in nlp.pipe_names:
                    doc.spans['span_candidates'] = spans
                examples.append(Example(doc, ref))
        nlp.update(examples)
    stream = make_tasks(nlp, stream)

    return {
        "view_id": "blocks",
        "dataset": dataset,
        "stream": stream,
        "update": make_update if update else None,
        "exclude": exclude,
        "config": {
            "labels": label,
            "on_exit": print_results,
            "exclude_by": "input",
            "auto_count_stream": not update,
            "blocks": blocks,
            "buttons": ["accept", "ignore", "undo"],
        },
    }

def print_results(ctrl: Controller) -> None:
    examples = ctrl.db.get_dataset(ctrl.session_id)
    if examples:
        counts = Counter()
        for eg in examples:
            counts[eg["answer"]] += 1
        for key in ["accept", "reject", "ignore"]:
            if key in counts:
                msg.row([key.title(), color(round(counts[key]), key)], widths=10)

# Code that not used but could be helpfull in the future

def get_parts_with_quotes(stream:StreamType, lang:str)->StreamType:
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    pattern = """["'«„].+?["'»“]"""
    for doc in stream:
        sents = []
        for chunk in chunks(doc['text'], 1000000):
            sents += list(nlp(chunk).sents)

        # Search for candidate results
        results = []
        for quote in re.finditer(pattern, doc['text']):
            result = [0, 0]
            for i, sent in enumerate(sents):
                if sent.start_char < quote.start(0):
                    result[0] = sents[i-1 if i != 0 else i].start_char
                if sent.end_char > quote.end(0):
                    result[1] = sents[i+1 if i != len(sents)-1 else i].end_char
                    results.append(result)
                    break

        # Unite intersected results
        last_result = None
        new_results = []
        for result in results:
            if last_result is None:
                last_result = result
            if last_result[1] > result[0]:
                last_result = [last_result[0], max(last_result[1], result[1])]
            else:
                new_results.append(last_result)
                last_result = result
        if last_result is not None:
            new_results.append(last_result)

        for result in new_results:
            eg = set_hashes({"text": doc['text'][result[0]:result[1]], "meta":doc['meta']})
            yield eg