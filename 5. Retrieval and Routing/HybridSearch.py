from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader(file_path=r"0. Documents\RAG for NLP.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 250
)

chunks = splitter.split_documents(documents= docs)

# for i,chunk in enumerate(chunks[2:6]):
#     print(f"Chunk {i+2} \n{chunk.page_content}")

mistral_model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)

embeddings = MistralAIEmbeddings(model="mistral-embed")
vecStore  = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

sem_retriever = vecStore.as_retriever(search_kwargs= {"k":3})
bm25_retriever = BM25Retriever.from_documents(documents= chunks,k= 3)
hybid = EnsembleRetriever(
    retrievers=[bm25_retriever,sem_retriever],
    weights=[0.5,0.5]
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an excellent Document retriever so from the context provided to you answer the user's query if you can't find answer say 'I don't know'"),
        ("human","Question : {query} \n\n Context : {context}")
    ]
)

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context":hybid | format_docs,
        "query" : RunnablePassthrough() 
    }
    | prompt
    | mistral_model
    | parser
)

result = chain.invoke("what is RAG token model")
print(result)



