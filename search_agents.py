from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Literal

from dataloader import Document
from indexer import Indexer, Postings


@dataclass(slots=True)
class SearchResult:
    doc_id: int
    score: float
    title: str
    uri: str
    snippet: str


class SearchAgent:
    def __init__(
        self,
        indexer: Indexer | None = None,
        postings: Postings | None = None,
        postings_path: str | Path | None = None,
        *,
        k1: float = 1.2,
        b: float = 0.75,
        delta: float = 1.0,
    ):
        if indexer is None and postings is None and postings_path is None:
            raise ValueError("Provide indexer, postings, or postings_path")

        self.indexer = indexer
        self.k1 = k1
        self.b = b
        self.delta = delta

        if postings is not None:
            self.postings = postings
        elif postings_path is not None:
            self.postings = Postings.load_postings(str(postings_path))
        else:
            if not indexer.is_indexed:
                indexer.preprocess()
            self.postings = (
                indexer.postings
                if indexer.postings is not None
                else indexer.construct_postings()
            )

        if self.postings is None:
            raise ValueError("Could not initialize postings")

        self._doc_lookup: dict[int, Document] = {}
        if self.indexer is not None:
            self._doc_lookup = {doc.doc_id: doc for doc in self.indexer.corpus}

    def _clean_query(self, query: str) -> list[str]:
        raw_tokens = Indexer._raw_tokens(query)
        return Indexer._nltk_processing(raw_tokens)

    def _document_frequency(self, term: str) -> int:
        return len(self.postings.postings.get(term, []))

    def _idf_bm25_plus(self, term: str) -> float:
        df = self._document_frequency(term)
        if df == 0:
            return 0.0
        return log((self.postings.num_docs + 1) / df)

    def _idf_tfidf(self, term: str) -> float:
        df = self._document_frequency(term)
        return log((self.postings.num_docs + 1) / (df + 1)) + 1.0

    def _score_bm25_plus(self, query_terms: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        avgdl = self.postings.avgdl if self.postings.avgdl > 0 else 1.0
        query_term_counts = Counter(query_terms)

        for term, qtf in query_term_counts.items():
            postings_list = self.postings.postings.get(term)
            if not postings_list:
                continue

            idf = self._idf_bm25_plus(term)

            for doc_id, tf in postings_list:
                doc_len = self.postings.doc_lengths.get(doc_id, 0)
                norm = self.k1 * (1 - self.b + self.b * (doc_len / avgdl))
                denom = tf + norm
                tf_component = ((self.k1 + 1) * tf) / denom if denom != 0 else 0.0

                scores[doc_id] += qtf * idf * (tf_component + self.delta)

        return dict(scores)

    def _score_tfidf(self, query_terms: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        query_term_counts = Counter(query_terms)

        for term, qtf in query_term_counts.items():
            postings_list = self.postings.postings.get(term)
            if not postings_list:
                continue

            idf = self._idf_tfidf(term)
            query_weight = (1.0 + log(qtf)) * idf if qtf > 0 else 0.0

            for doc_id, tf in postings_list:
                doc_weight = (1.0 + log(tf)) * idf if tf > 0 else 0.0
                scores[doc_id] += query_weight * doc_weight

        return dict(scores)

    def _make_snippet(
        self,
        doc_id: int,
        query_terms: list[str],
        max_chars: int = 180,
    ) -> str:
        doc = self._doc_lookup.get(doc_id)
        if doc is None:
            return ""

        content = " ".join(doc.raw_content.split())
        lowered = content.lower()

        hit_pos = -1
        hit_term = ""
        for term in query_terms:
            pos = lowered.find(term.lower())
            if pos != -1 and (hit_pos == -1 or pos < hit_pos):
                hit_pos = pos
                hit_term = term

        if hit_pos == -1:
            return content[:max_chars] + ("..." if len(content) > max_chars else "")

        half_window = max_chars // 2
        start = max(0, hit_pos - half_window)
        end = min(len(content), hit_pos + len(hit_term) + half_window)

        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet += "..."
        return snippet

    def _to_results(
        self,
        scores: dict[int, float],
        query_terms: list[str],
        top_k: int,
    ) -> list[SearchResult]:
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        results: list[SearchResult] = []

        for doc_id, score in ranked:
            doc = self._doc_lookup.get(doc_id)
            title = doc.title if doc is not None else f"Document {doc_id}"
            uri = doc.uri if doc is not None else self.postings.get_uri(doc_id)
            snippet = self._make_snippet(doc_id, query_terms)

            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=score,
                    title=title,
                    uri=uri,
                    snippet=snippet,
                )
            )

        return results

    def query(
        self,
        query_str: str,
        *,
        model: Literal["bm25+", "tfidf"] = "bm25+",
        top_k: int = 10,
        display: bool = True,
    ) -> list[SearchResult]:
        query_terms = self._clean_query(query_str)

        if not query_terms:
            if display:
                print("Query is empty after preprocessing.")
            return []

        if model == "bm25+":
            scores = self._score_bm25_plus(query_terms)
        elif model == "tfidf":
            scores = self._score_tfidf(query_terms)
        else:
            raise ValueError("model must be 'bm25+' or 'tfidf'")

        results = self._to_results(scores, query_terms, top_k)

        if display:
            self.display_results(results, query_str=query_str, model=model)

        return results

    def compare_models(
        self,
        query_str: str,
        *,
        top_k: int = 10,
    ) -> dict[str, list[SearchResult]]:
        bm25_results = self.query(query_str, model="bm25+", top_k=top_k, display=False)
        tfidf_results = self.query(query_str, model="tfidf", top_k=top_k, display=False)

        print(f"\n=== BM25+ results for: {query_str!r} ===")
        self.display_results(bm25_results, query_str=query_str, model="bm25+")

        print(f"\n=== TF-IDF results for: {query_str!r} ===")
        self.display_results(tfidf_results, query_str=query_str, model="tfidf")

        return {"bm25+": bm25_results, "tfidf": tfidf_results}

    def display_results(
        self,
        results: list[SearchResult],
        *,
        query_str: str | None = None,
        model: str | None = None,
    ) -> None:
        if query_str is not None and model is not None:
            print(f"\nQuery: {query_str}")
            print(f"Model: {model}")

        if not results:
            print("No results found.")
            return

        for rank, result in enumerate(results, start=1):
            print(f"{rank}. [{result.score:.4f}] {result.title}")
            print(f"   URL: {result.uri}")
            if result.snippet:
                print(f"   Snippet: {result.snippet}")
            print()
