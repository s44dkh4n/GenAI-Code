from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpointEmbeddings,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

loader = DirectoryLoader(
    path = r"Documents\CompanyPolicy",
    loader_cls = TextLoader,
    glob = "*.txt"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 900,
    chunk_overlap = 180
)

chunks = splitter.split_documents(documents= docs)

print(len(chunks))
# for chunk in chunks:
#     print("New chunk\n",chunk.page_content,"\n")

embedd_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

vectorStore = FAISS.from_documents(
    embedding = embedd_model,
    documents=chunks
)

retriever = vectorStore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":2}
)

query = "tell me about PTO."

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.3
)
model = ChatHuggingFace(llm = llm)

prompt = ChatPromptTemplate.from_messages(
    [
      ("system", "From the context provided to you answer any query u are asked"),  
      ("human", "Answer this query \n\n Query -> {query} \n\n Context -> {context}"),  
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context" : retriever | format_docs,
     "query" : RunnablePassthrough()
    }
    | prompt
    | model
    | parser
)

result = chain.invoke(query)
print(result)