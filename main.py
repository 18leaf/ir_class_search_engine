from dataloader import DataLoader
from pathlib import Path


def main():
    print("Hello from a2!")
    dl = DataLoader(Path("./data"))
    # begin loading data into Documents
    dl.start()


if __name__ == "__main__":
    main()
