# Spancat training

Change configuration is possible in project.yml

<details>
<summary>Suggesters</summary>

Suggester is a function that gives the categorizer spans to classify.

There are config files for three types of them: `base_config` - `ngram_suggester`, `config_cs` - `custom_suggester`, `config_sf` - `span_finder`

`ngram_suggester` - gives ngrams of predefined size

`custom_suggester` - gives text in quotes and every other ngram

`span_finder` - a trainable layer that learns to provide the best spans

Change `config` variable in project.yml to define suggester
```
...
vars:
  #base_config - use ngram_suggester
  #config_cs - use custom_suggester
  #config_sf - use span_finder
  config: "base_config" # <--- here
...
```

</details>

Preprocess dataset: `python -m spacy project run preprocess`

Train: `python -m spacy project run spancat`