# Annotation

This folder contains the custom scripts to annotate quotes to create training data. Expand the dropdowns below for usage notes.

We used three entity types for quote extraction:

1. `Source`
2. `Cue`
3. `Content`

## Requirements

Developed with Python v3.9.13. All requirements are listed in the `requirements.txt` file.

Please note that you will need a copy of [Prodigy](https://prodi.gy/) (v1.11.5) which is proprietary software.

## `quotes.manual`

Example:`python -m prodigy quotes.manual <dataset> blank:en <input-data> -l Source,Content,Cue -F quotes.py`

<details>
<summary>Expand for details</summary>
    
```
usage: prodigy quotes.manual [-h] [-lo None] [-l None] [-e None] [-U] dataset spacy_model source

    Mark spans by token. Requires only a tokenizer, and doesn't do any active learning. 
    The recipe will present all examples in order, so even examples without matches are shown.


positional arguments:
  dataset               Dataset to save annotations to
  spacy_model           Loadable spaCy pipeline for tokenization or blank:lang (e.g. blank:en)
  source                Data to annotate (file path or '-' to read from standard input)

optional arguments:
  -h, --help            show this help message and exit
  -lo None, --loader None
                        Loader (guessed from file extension if not set)
  -l None, --label None
                        Comma-separated label(s) to annotate or text file with one label per line
  -e None, --exclude None
                        Comma-separated list of dataset IDs whose annotations to exclude
  -U, --unsegmented     Don't get only parts with quotes
```
</details>

## `quotes.correct`

Example:`python -m prodigy quotes.correct <dataset> <model> <input-data> -l Source,Content,Cue -b 256 -F quotes.py`

<details>
<summary>Expand for details</summary>
    
```
usage: prodigy quotes.correct [-h] [-lo None] [-l None] [-e None] [-b 64] [-U] [-UP] dataset spacy_model source

    Create gold data for spancat by correcting a model's suggestions.
    Prodigy will decide which questions to ask next based on cleanlab score


positional arguments:
  dataset               Dataset to save annotations to
  spacy_model           Loadable spaCy pipeline with an entity recognizer
  source                Data to annotate (file path or '-' to read from standard input)

optional arguments:
  -h, --help            show this help message and exit
  -lo None, --loader None
                        Loader (guessed from file extension if not set)
  -l None, --label None
                        Comma-separated label(s) to annotate or text file with one label per line
  -e None, --exclude None
                        Comma-separated list of dataset IDs whose annotations to exclude
  -b 64, --batch-size 64
                        Batch size for cleanlab socres computing
  -U, --unsegmented     Don't get only parts with quotes
  -UP, --update         Whether to update the model during annotation
```
</details>

## `quotes.teach`

Example:`python -m prodigy quotes.teach <dataset> <model> <input-data> -l Source,Content,Cue -F quotes.py`

<details>
<summary>Expand for details</summary>
    
```
usage: prodigy quotes.teach [-h] [-lo None] [-l None] [-e None] [-b 64] [-U] dataset spacy_model source

    Collect the best possible training data for a spancat model.
    Prodigy will decide which questions to ask next based on cleanlab score.


positional arguments:
  dataset               Dataset to save annotations to
  spacy_model           Loadable spaCy pipeline with an entity recognizer
  source                Data to annotate (file path or '-' to read from standard input)

optional arguments:
  -h, --help            show this help message and exit
  -lo None, --loader None
                        Loader (guessed from file extension if not set)
  -l None, --label None
                        Comma-separated label(s) to annotate or text file with one label per line
  -e None, --exclude None
                        Comma-separated list of dataset IDs whose annotations to exclude
  -b 64, --batch-size 64
                        Batch size for cleanlab socres computing
  -U, --unsegmented     Don't get only parts with quotes
```
</details>
