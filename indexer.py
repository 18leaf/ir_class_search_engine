from __future__ import annotations # type annotations for loading/saving from files
from collections import Counter
from dataloader import Document
import numpy as np
import re
import pickle


from nltk.stem import WordNetLemmatizer as wnl
import nltk
from nltk.corpus import stopwords

# download stopwords and lemmatize wordnet if not alread exists
nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(stopwords.words('english'))

class Indexer:
    corpus: list[Document]
    # doc_id -> [processed words in their position) = cleaned words in document
    processed_docs: dict[int, list[str]]
    vocab: np.ndarray  # make this an array of all words + their frequencies
    # after vocab
    doc_term_matrix: np.ndarray | None  # numpy, where each row = a document, each column = idx of preproc token, each cell has occurences in that document row

    avgdl: float
    # this is basically the postings
    doc_term_list: dict[int, list[str]]
    term_to_id: dict[str, int]
    id_to_term: dict[int, str]
    is_indexed: bool
    postings: Postings | None

    def __init__(self, documents: list[Document]):
        # sort by doc id for deterministic ordering
        documents.sort(key=lambda x: x.doc_id)
        self.doc_term_list: dict[int, list[str]] = {}
        self.processed_docs: dict[int, list[str]] = self.doc_term_list

        self.term_to_id: dict[str, int] = {}
        self.id_to_term: dict[int, str] = {}

        self.vocab: np.ndarray = np.array([], dtype=object)
        self.doc_term_matrix: np.ndarray | None = None

        self.avgdl = 0.0
        self.corpus = documents
        self.is_indexed = False
        self.postings = None

    def construct_postings(self) -> Postings | None:
        if not self.is_indexed:
            print("Not Ready Yet... Not preprocessed")
            return None

        # create mapping of id to uri 
        doc_id_to_uri = {doc.doc_id: doc.uri for doc in self.corpus}
        
        self.postings = Postings.from_doc_term_list(
            doc_term_list=self.doc_term_list,
            term_to_id=self.term_to_id,
            id_to_term=self.id_to_term,
            vocab=self.vocab,
            avgdl=self.avgdl,
            num_docs=len(self.corpus),
            doc_id_to_uri=doc_id_to_uri
        )

        return self.postings


    def preprocess(self):
        total = len(self.corpus)
        attempts = 0
        # initialize a mutable vocab list using collection counter
        # at each iteration, expand the counts from that doc that was preprocessed
        vocab_counter = Counter()
        doc_lengths = []
        for doc in self.corpus:
            # preprocess doc
            doc_id = doc.doc_id
            raw_text = doc.raw_content
            raw_tokens = Indexer._raw_tokens(raw_text)
            processed_toks = Indexer._nltk_processing(raw_tokens)
            # insert into doc_term_list
            self.doc_term_list[doc_id] = processed_toks
            vocab_counter.update(processed_toks)
            doc_lengths.append(len(processed_toks))
            attempts += 1

        # create a dual mapping of term_id -> term, term -> term_id
        # optionally, prefer lower memory size items for higher frequency terms in the future, ignore for now
        sorted_terms = [
            term for term, _ in sorted(
                vocab_counter.items(),
                key=lambda item: (-item[1], item[0])
            )
        ]
        self.term_to_id = {term: idx for idx, term in enumerate(sorted_terms)}
        self.id_to_term = {idx: term for term, idx in self.term_to_id.items()}
    
        # construct vocab, such that higher freq terms have lower ids (not doing anything with it)
        # but can use encodings for postings later
        self.vocab = np.array(
            [(term, vocab_counter[term]) for term in sorted_terms],
            dtype=object
        )

        # construct doc-term matrix ... essentially the postings matrix (term pos is lost, only freq)
        num_docs = len(self.corpus)
        num_terms = len(sorted_terms)

        self.doc_term_matrix = np.zeros((num_docs, num_terms), dtype=np.int32)

        for row_idx, doc in enumerate(self.corpus):
            doc_id = doc.doc_id
            term_counts = Counter(self.doc_term_list[doc_id])

            for term, count in term_counts.items():
                col_idx = self.term_to_id[term]
                self.doc_term_matrix[row_idx, col_idx] = count

        # acg document length
        self.avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

        self.is_indexed = True
        print(f"Processed: {attempts}\nOut of: {total}")

    @staticmethod
    def _raw_tokens(raw_text: str) -> list[str]:
        # lowercase+remove
        raw_text = raw_text.lower()
        text = re.sub(r"[^a-z\s]+", " ", raw_text)
        # strip extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    # remove stopwords, lemmatize word
    @staticmethod
    def _nltk_processing(raw_toks: list[str]) -> list[str]:
        # remove stopwords
        processed_toks = []
        # initialize once
        lemmatizer = wnl()
        for word in raw_toks:
            # lemmatizer eith lemmatizers or returns orgiinl
            if word in STOPWORDS:
                continue
            word = lemmatizer.lemmatize(word)
            processed_toks.append(word)

        return processed_toks

class Postings:
    # mapping of vocabulary->ids, ids->vocabulary
    # uses a HashMap -> Do not have to worry about expanding terms (static input data, no resizing cost)
    # only stores term (actual term not id) -> sorted_list of tuples by doc_id (doc_id, freq)
    term_to_id: dict[str, int]
    id_to_term: dict[int, str]
    postings: dict[str, list[tuple[int, int]]]
    vocab: np.ndarray
    avgdl: float
    num_docs: int
    # support BM25 discrete doc-length. not only avg
    doc_lengths: dict[int, int]
    doc_id_to_uri: dict[int, str]

    def __init__(
        self,
        postings: dict[str, list[tuple[int, int]]],
        term_to_id: dict[str, int],
        id_to_term: dict[int, str],
        vocab: np.ndarray,
        avgdl: float,
        num_docs: int,
        doc_lengths: dict[int, int],
        doc_id_to_uri: dict[int, str],
    ):
        self.postings = postings
        self.term_to_id = term_to_id
        self.id_to_term = id_to_term
        self.vocab = vocab
        self.avgdl = avgdl
        self.num_docs = num_docs
        self.doc_lengths = doc_lengths
        self.doc_id_to_uri = doc_id_to_uri

    @classmethod
    def from_doc_term_list(
        cls,
        doc_term_list: dict[int, list[str]],
        term_to_id: dict[str, int],
        id_to_term: dict[int, str],
        vocab: np.ndarray,
        avgdl: float,
        num_docs: int,
        doc_id_to_uri: dict[int, str]
    ):
        # initialize empty postings list, remember tuple is doc_id, freq
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_lengths: dict[int, int] = {}

        # doc_term_list should already be sorted
        for doc_id in sorted(doc_term_list.keys()):
            # for raw tokens/counting doc lengt @ doc level
            tokens = doc_term_list[doc_id]
            term_counts = Counter(doc_term_list[doc_id])
            doc_lengths[doc_id] = len(tokens)
            

            for term, freq in term_counts.items():
                if term not in postings:
                    postings[term] = []
                postings[term].append((doc_id, freq))

        return cls(
            postings=postings,
            term_to_id=term_to_id,
            id_to_term=id_to_term,
            vocab=vocab,
            avgdl=avgdl,
            num_docs=num_docs,
            doc_lengths=doc_lengths,
            doc_id_to_uri = doc_id_to_uri
        )

    
    def get_uri(self, doc_id: int) -> str | None:
        # helper to get url from matches
        return self.doc_id_to_uri[doc_id]
        
    def save_postings(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_postings(cls, path: str) -> "Postings":
        with open(path, "rb") as f:
            return pickle.load(f)
