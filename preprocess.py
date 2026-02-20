from collections import Counter
from dataloader import Document
import numpy as np
import re

from nltk.stem import WordNetLemmatizer as wnl
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# Stores Corpus = All Documents, Raw Documents
# avg doc length
# Vocab
# DocId2FileName
# Postings (Inverted Index)
class Indexer:
    corpus: list[Document]
    # doc_id -> [tok] (list_size) = cleaned words in document
    processed_docs: dict[int, list[str]]
    avgdl: float
    term2tok: dict[str, int] # also = vocab

    def __init__(self, documents: list[Document]):
        # sort by 
        documents.sort(key=lambda x: x.doc_id)

        self.doc_term_list = {}
        self.avgdl = 0.0
        self.term2tok = {}
        self.corpus = documents

    def preprocess(self):
        total = len(self.corpus)
        attempts = 0
        for doc in self.corpus:
            # preprocess doc
            doc_id = doc.doc_id
            raw_text = doc.raw_content
            raw_tokens = Indexer._raw_tokens(raw_text)
            processed_toks = Indexer._nltk_processing(raw_tokens)
            # insert into doc_term_list
            self.doc_term_list[doc_id] = processed_toks
            attempts += 1

        print(f"Processed: {attempts}\nOut of: {total}")
    
    def _raw_tokens(raw_text: str) -> list[str]:
        # lowercase+remove
        raw_text = raw_text.lower()
        text = re.sub(r"[^a-z\s]+", " ", raw_text)
        # strip extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    # remove stopwords, lemmatize word
    def _nltk_processing(raw_toks: list[str]) -> list[str]:
        # remove stopwords
        processed_toks = []
        for word in raw_toks:
            # lemmatizer eith lemmatizers or returns orgiinl
            if word in STOPWORDS:
                continue
            word = wnl().lemmatize(word)
            processed_toks.append(word)

        return processed_toks
     
# parse Text from them, save uri

    # for each text, apply preprocessing
        # remove non alphabetical symbols (punct, digi)
        # expand contractions (shouldn't - should not)
        # remove stop words
        # lemmatization
        # Maintain Global Vocab list + count (using B-TreeMap)
        # Build Posting Dict -> Doc_id -> [(token, pos), (tokenN, posN)]
