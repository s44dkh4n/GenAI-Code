from dotenv import load_dotenv # Loads environment variables from a .env file (e.g., API keys)

# 1. Document Loaders (Data Ingestion)
from unstructured.partition.pdf import partition_pdf 
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    TextLoader, 
)

# 2. Text Splitters (Chunking Strategies)
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter, 
)

# 3. Embedding Models (Vector Generators)
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from langchain_mistralai import MistralAIEmbeddings 

# 4. Language Models (LLMs for Generation)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_mistralai import ChatMistralAI 
from langchain_ollama import ChatOllama

# 5. Storage Layer (Vector Databases & Key-Value Stores)
from langchain_chroma import Chroma 
from langchain_community.vectorstores import FAISS 
from langchain_core.stores import InMemoryByteStore 
from langchain_core.vectorstores import InMemoryVectorStore 

# 6. Retrievers & Re-rankers (Search Optimization Layer)
from langchain_classic.retrievers import (
    ContextualCompressionRetriever, 
    EnsembleRetriever, 
    ParentDocumentRetriever, 
)
from langchain_classic.retrievers.document_compressors import (
    CohereRerank, 
    EmbeddingsFilter,
    )
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.document_compressors import flashrank_rerank
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever 
from langchain_community.retrievers import BM25Retriever 

# 7. LCEL Orchestration Layer (Chains, Prompts, & Parsing)
from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough 

# Initialize environment variables
load_dotenv()

load_dotenv()
import os

# loader = TextLoader(r"Documents\text.txt")
# docs = loader.load()

# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Llama-3.1-8B-Instruct",
#     temperature=0.5
# )
# model = ChatHuggingFace(llm = llm)

# embeddings = HuggingFaceEndpointEmbeddings(
#     model="sentence-transformers/all-MiniLM-L6-v2",
#     task="feature-extraction"
# )

# mistral_model = ChatMistralAI(
#     model="mistral-small-latest",
#     temperature=0.3
# )

# embeddings = MistralAIEmbeddings(model="mistral-embed")

# messages = ChatPromptTemplate.from_messages(
#     [
#         ("system", "you are an Document summarizer"),
#         ("human", "{data}")
#     ]
# )

# prompt = messages.format_messages(data= docs[0].page_content)

# result = hf_model.invoke(prompt)
# print(result.content)

 
print("\n\n","="*60)

# Multimodal RAG Ingestion
# 2. Path to the PDF file you want to parse
pdf_file_path = r"0. Documents\attention.pdf"

# 3. Initialize the loader with API settings
loader = partition_pdf(
    file_path=pdf_file_path,
    partition_via_api=True,   
    strategy="hi_res",        # Required for image extraction
    chunking_strategy="by_title",
    # Pass extra arguments to the underlying Unstructured API
    unstructured_kwargs={
        "extract_image_block_types": ["Image"],  # Extract both images and tables
        "extract_image_block_to_payload": True,           # Base64 encodes the images into element metadata
        "infer_table_structure": True
    }
)


print("Sending PDF to Unstructured Serverless API for parsing...")

# 4. Load and parse the document
docs = loader.load()

print(f"Successfully parsed PDF! Generated {len(docs)} document chunks.\n")

print(docs[20].to_json())

# 5. Inspect the extracted content
# for index, doc in enumerate(docs[:25]):  # Look at the first 20 chunks
#     print(f"--- Chunk {index + 1} ---")
#     print(f"Content Preview:\n{doc.page_content}")
#     print("\n" + "="*40 + "\n")

