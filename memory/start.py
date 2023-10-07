from pathlib import Path

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore


embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en")
VECTOR_DB_PATH = "./chroma_db"


def index_data(embedding_function=embedding_function) -> VectorStore:
    # load the document and split it into chunks
    loader = PyPDFLoader("~/Downloads/andrew-ng-machine-learning-yearning-1.pdf")
    documents = loader.load()

    # split it into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1_000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # create the open-source embedding function

    # load it into Chroma
    return Chroma.from_documents(
        docs, embedding_function, persist_directory=VECTOR_DB_PATH
    )


def get_search_index() -> VectorStore:
    if Path(VECTOR_DB_PATH).exists():
        print("Found vector db. Loading...")
        return Chroma(
            persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function
        )
    else:
        print("Could not find existing vector db. Indexing new file(s)...")
        return index_data()


def main():
    search_index = get_search_index()

    # query it
    while True:
        query = input("Input: ")
        docs = search_index.similarity_search(query)
        # print results
        for doc in docs:
            print(["".join(doc.page_content).replace("\n", "")])


if __name__ == "__main__":
    main()
