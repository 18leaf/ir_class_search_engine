from pathlib import Path

class Document:
    uri: str
    path: Path
    title: str
    raw_content: str


    # takes raw file and parses uri, title and content
    def _parse_file(path: Path) -> tuple[str, str, str]:
        text = path.read_text(encoding="utf-8", errors="ignore")

        it = iter(text.splitlines())
        
        url = next(it).split(":", 1)[1].strip()
        # ignore
        _access_time = next(it).split(":", 1)[1].strip()
        title = next(it).split(":", 1)[1].strip()

        raw_content = "\n".join(it).strip()
        return (url, title, raw_content)

    # init document from path string
    def __init__(self, raw_file_path: Path):
        url, title, raw_content = Document._parse_file(raw_file_path)
        self.uri = url
        self.title = title
        self.raw_content = raw_content
        self.path = raw_file_path

    def __hash__(self):
        return hash(self.uri)

class DataLoader:
    # Path to document
    docs_to_load: list[Path]
    raw_documents: list[Document]
    
    def __init__(self, directory: Path):
        # get all direct children (that are files) in directory
        self.docs_to_load = [c for c in directory.iterdir() if c.is_file()]
        # empty list
        self.raw_documents = []

    def start(self):
        for file in self.docs_to_load:
            # create Document instance and add to raw_documents
            file_document = Document(file)
            # remove file from docs to lad
            self.docs_to_load.remove(file)
            self.raw_documents.append(file_document)
