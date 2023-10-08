from pathlib import Path

import dotenv
from langchain import hub
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore


dotenv.load_dotenv()
embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en")
VECTOR_DB_PATH = "./chroma_db"
MODEL_PATH = "/Users/rlm/Desktop/Code/llama.cpp/models/llama-2-13b-chat.ggufv3.q4_0.bin"


def get_search_index() -> VectorStore:
    if Path(VECTOR_DB_PATH).exists():
        print("Found vector db. Loading...")
        return Chroma(
            persist_directory=VECTOR_DB_PATH, embedding_function=embedding_function
        )
    else:
        print("Could not find existing vector db. Indexing new file(s)...")
        return index_data()


def index_data(embedding_function=embedding_function) -> VectorStore:
    # load the document and split it into chunks
    loader = PyPDFLoader("~/Downloads/andrew-ng-machine-learning-yearning-1.pdf")
    documents = loader.load()

    # split it into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)

    # load it into Chroma
    return Chroma.from_documents(
        chunks, embedding_function, persist_directory=VECTOR_DB_PATH
    )


def main():
    search_index = get_search_index()
    retriever = search_index.as_retriever()

    rag_prompt = hub.pull("rlm/rag-prompt")
    llm = HuggingFacePipeline.from_model_id(
        model_id="bigscience/bloom-1b7",
        task="text-generation",
        model_kwargs={"temperature": 0, "max_length": 64},
    )
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm
    )

    while True:
        rag_chain.invoke("What is Task Decomposition?")


if __name__ == "__main__":
    main()
