# vocabs/

This folder contains two kinds of files:

## i) Input files (external sources)

These files are used as inputs by the vocabulary generation script and were obtained from external sources:

- **`core-wordnet.txt`** — A compact lexicon of core English words (nouns, adjectives, verbs) provided by Princeton University as part of the WordNet project.
- **`mscoco.txt`** — A list of frequent words extracted from image captions in the [MS-COCO dataset](https://cocodataset.org). Used as a candidate pool for concept words.
- **`prompts.txt`** — A set of prompt templates (e.g., `a photo of a {}.`) used to obtain concept vectors via CLIP's text encoder, following the approach in explainable AI literature such as [Ravi Mangal's work](https://arxiv.org/abs/2403.19837).

## ii) Generated vocabularies

These files are produced by running `src/build_vocab.py` and represent the concept vocabularies used in this project:

- **`MCS_<threshold>_NA_<dataset>.txt`** — Vocabulary generated for a given dataset (RIVAL10 or EuroSAT) with a maximum cosine similarity threshold of `<threshold>`. Each word in the list is sufficiently dissimilar from the class labels and from other concepts in the vocabulary.

The corresponding CLIP text embedding tensors are stored in `intermediate_results/` (gitignored).
