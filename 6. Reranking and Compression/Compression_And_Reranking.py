from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

# ChromaDb -> Hybrid Search -> Cohere Reranking -> Compression Retriever -> Output

load_dotenv()

loader = PyPDFLoader(r"GenAI\0. Documents\RAG for NLP.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size =1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embeddings = MistralAIEmbeddings(model="mistral-embed")

vecStore = Chroma.from_documents(
    documents= chunks,
    embedding= embeddings
)

sem_retrvr = vecStore.as_retriever(search_kwargs = {"k":10})

bm25 = BM25Retriever.from_documents(documents = chunks, k = 10)

hybrid = EnsembleRetriever(
    retrievers= [sem_retrvr, bm25],
    weights= [0.5, 0.5]
)

base_compressor = CohereRerank(
    model = "rerank-english-v3.0",
    top_n = 3
)

compression_retriever = ContextualCompressionRetriever(
    base_retriever = hybrid,
    base_compressor = base_compressor
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","The user will provide you with Context and a Question so answer the question only from the given context. If you can't find the anser say 'I couldn't find the answer.'"),
        ("human","Question -> {query} \n\nContext -> {context}")
    ]
)

parser = StrOutputParser()

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context" : compression_retriever | format_docs,
        "query" : RunnablePassthrough()
    }
    | prompt
    | model
    | parser
)

test_queries = [
    "What are the core conclusions of the RAG architecture paper?",
    "what is RAG Token Model?",
    "What is abstract Question Answering?"
]

result = []
for i, query in enumerate(test_queries):

    result.append(chain.invoke(query))
    print(f"Query{i} : {query} \n{result[i]}")
