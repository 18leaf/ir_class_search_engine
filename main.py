from pathlib import Path

from dataloader import DataLoader
from indexer import Indexer
from search_agents import SearchAgent


def main():
    dl = DataLoader(Path("./data"))
    dl.start()

    indexer = Indexer(dl.get_documents())
    indexer.preprocess()
    indexer.construct_postings()

    agent = SearchAgent(indexer=indexer)

    agent.compare_models("bm25")


if __name__ == "__main__":
    main()
