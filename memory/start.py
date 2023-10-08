from pathlib import Path

import dotenv
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, VectorStore


dotenv.load_dotenv()
embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en")
VECTOR_DB_PATH = "./chroma_db"
MODEL_PATH = "/Users/rlm/Desktop/Code/llama.cpp/models/llama-2-13b-chat.ggufv3.q4_0.bin"


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
    from langchain.chains import LLMChain
    from langchain.llms import HuggingFaceHub

    from langchain.prompts import PromptTemplate

    question = "Who won the FIFA World Cup in the year 1994? "
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    repo_id = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 264}
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    print(llm_chain.run(question))

    # # query it
    # while True:
    #     query = input("Input: ")
    #     docs = search_index.similarity_search(query)
    #     # print results
    #     for doc in docs:
    #         print(["".join(doc.page_content).replace("\n", "")])


if __name__ == "__main__":
    main()
