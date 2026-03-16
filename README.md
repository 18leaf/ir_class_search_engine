# Simple Indexer and Retriever Report

**Lindsey Ferguson**  
**Jiho Noh**  
**CS 4422 – Information Retrieval**

## Implementation
This project implements a basic indexing and retrieval pipeline over a collection of crawled IR-related webpages. The system is divided into four main parts:

- **`dataloader.py`** parses the raw files and creates `Document` objects containing the URI, title, and text content.
- **`indexer.py`** preprocesses the documents, builds the vocabulary, computes document lengths, and constructs the inverted index.
- **`search_agents.py`** processes user queries and ranks documents using either **BM25+** or **TF-IDF**.
- **`main.py`** runs the pipeline by loading documents, building the index, and executing search queries.

Both documents and queries use the same preprocessing pipeline:
- lowercase conversion
- removal of punctuation and digits
- stopword removal with NLTK
- lemmatization with WordNetLemmatizer

The final postings structure stores terms as:
- `term -> [(doc_id, term_frequency)]`

This allows efficient lookup of documents containing a query term and supports both ranking methods.

## Design Choices
A major design choice was to keep the system simple and readable rather than heavily optimized. The postings list stores term strings directly instead of only numeric IDs, which makes debugging easier. The same preprocessing steps are applied to both documents and queries to ensure matching is consistent.

For retrieval, the system supports both **BM25+** and **TF-IDF**:
- **BM25+** was chosen as the main ranking model because it handles document length normalization better and reduces unfair penalization of longer documents.
- **TF-IDF** was included as a baseline model for comparison.

BM25+ uses document frequency, term frequency, document length, average document length, and the parameters `k1`, `b`, and `delta` to compute relevance. TF-IDF uses log-scaled term frequency and inverse document frequency to produce a weighted term-overlap score.

## Retrieval Results
The system was tested with several example queries to compare BM25+ and TF-IDF.

### Query 1: `information retrieval ranking model`
This query returned documents focused on retrieval models, ranking, and core IR concepts. BM25+ generally placed the most topically centered documents first, while TF-IDF returned similar documents but sometimes in a different order.

### Query 2: `inverted index`
This query produced highly relevant results related to indexing structures and postings lists. Since the query is short and specific, both BM25+ and TF-IDF performed similarly, although BM25+ still handled document length more effectively.

### Query 3: `probabilistic model bm25`
This query strongly favored documents discussing probabilistic retrieval and BM25-style ranking. BM25+ produced the most intuitive ranking because the query terms are highly specific to the model.

In general, BM25+ gave more stable rankings across different document lengths, while TF-IDF worked well as a simpler comparison baseline.

## Challenges and Solutions
One challenge was making sure that the query pipeline matched the document preprocessing pipeline exactly. If queries and documents are normalized differently, relevant terms may fail to match. This was addressed by reusing the same token-cleaning logic for both.

Another challenge was aligning retrieval code with the actual structure of the postings lists. The postings were stored as tuples `(doc_id, term_frequency)`, so the scoring code had to be updated to unpack tuples directly instead of assuming posting objects with named fields.

A final challenge was balancing implementation simplicity with retrieval quality. Instead of adding phrase search or positional indexing, the project focused on getting the core inverted index and ranking models working correctly first.

## Conclusion
This project successfully implemented a simple document retrieval system with inverted indexing and ranked search using both BM25+ and TF-IDF. The design emphasizes clarity and correctness, and the retrieval results show that BM25+ is a strong primary model while TF-IDF provides a useful baseline for comparison.
