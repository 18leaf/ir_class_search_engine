# Simple Indexer And Retriever
Lindsey Ferguson
Jiho Noh
CS4422 - Information Retrieval
---
## Overview
This project involves implenting a simple indexer and document search engine that implements the following high-level features.

1. Read document files (i.e., the crawled IR book webpages) and parse the content of the files. (If you weren't able to complete the previous assignment, you can use the provided dataset.)
2. Index the documents by building an inverted index.
3. Given a user's input query, retrieve relevant documents ordered by the [BM25+](https://dl.acm.org/doi/pdf/10.1145/2063576.2063584) scores

## This Document
The purpose of this document is to outline the implementation + planning that went into developing the retrieval system. This is an entry point into understanding the decisions made with building this repository. 

## Vocabulary
Punctuation and Digits will be removed, I will be lemmatizing words (reducing to base words = running - run) using NLTK.
I will maintain a tok2idx and idx2token mapping
## Index - Postings
Postings will contain term locations in the document and a total of occurences in that document. term -> (doc_id, t_freq, (position in doc))

---
## Architecture
This summarizes the data flow/pipeline of this system, as well as some higher level implementation details crucial for understanding this approach.
### Data Pipeline
1. **Files saved to some directory, representing the domain of a scraped site.**
    - Each file has key-value line seperated: URI, Access Time, Title, and all raw content below Text
2. **Files must be read at the document level and the text is **PREPROCESSED** for all documents.**
    - I will be removing punctuation, digits,
    - lowercasing all words,
    - Expanding Contractions (shouldn't -> should not, etc.)
    - remove stopwords (from NLTK stopwords english)
    - applying lemmatization (potentially maintain a mapping of their pre/post-fix) ie running -> (run, pfix_id)
    - Use root word lemma + consider usig some context vector?? 
    - *NOTE* Upon each token found in doc, that token position (after pre-processing steps) will be recorded and tok -> document_positions will be recorded for each document


### Dependencies
NLTK, Numpy,
[BTrees](https://btrees.readthedocs.io/en/latest/index.html)
