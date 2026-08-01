from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from dotenv import load_dotenv 



# Load environment variables
load_dotenv()

files_path = ["Documents\LangChain.pdf","Documents/Deep Learning.pdf","Documents/Machine Learning.pdf"]

# Load the PDF resume
docs = []
for path in files_path:
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size =1200,
    chunk_overlap=200 
)

splits = text_splitter.split_documents(docs)

embed_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embed_model,
    persist_directory="VectorStoreCollections"
)

retriever = vector_store.as_retriever(search_kwargs = {"k":4})
query = "Tell me about FEATURE VECTORS & EMBEDDINGS?"

parser = StrOutputParser()

# Initialize the HuggingFaceEndpoint with conversational task
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    temperature=0.3,
)
# Wrap it in ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# Prompt
prompt = ChatPromptTemplate.from_template(
    """
You are an AI Tutor.

Answer the question ONLY from the provided context.

If the answer is not present in the context, say:
"I could not find that information in the provided documents."

Context:
{context}

Question:
{question}
"""
)

# Helper Function to join all the documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context":retriever | format_docs,
        "question" : RunnablePassthrough()
    }
    | prompt | chat_model | parser
)

result = chain.invoke(query)
print(result)
