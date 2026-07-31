from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings
from langchain_core.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

loader = TextLoader(r"Documents\router_manual.txt")
docs = loader.load()
print(f"number of docs before splitting : {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap = 100
)

chunks = splitter.split_documents(documents=docs)
print(f"number of docs after splitting : {len(chunks)}")

# for chunk in chunks:
#     print(f"New Page \n {chunk.page_content} \n")

# Initialize hosted API embeddings 
embedd_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

vectorStore = Chroma.from_documents(
    embedding=embedd_model,
    documents=chunks
)

retriever = vectorStore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":2 }
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI Assistant that answers a query from the the Document provided to you"),
        ("human","""From the provided context give me answer for this query below \n
         
         Context -> "{context}"\n

         Query -> "{query}"
         """)
    ]
)

query = "Tell me how To set up your OmniConnect X1 Routeruc"

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.3,
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs, 
        "query": RunnablePassthrough()
    } | prompt | model | parser
)

response = rag_chain.invoke(query)
print("\n--- Response ---")
print(response)