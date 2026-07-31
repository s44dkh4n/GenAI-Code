from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()

mistral_model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.1,
    max_tokens= 700
)

embeddings = MistralAIEmbeddings(model="mistral-embed")

class ExtractedInsights(BaseModel):
    summary: str = Field(description="Brief summary answering the query using the contextualized chunks.")
    key_findings: List[str] = Field(description="Core key findings extracted from the retrieved contextual chunks.")
    context_relevance_score: float = Field(description="Score between 0.0 and 1.0 indicating how well prepended context resolved query entities.")
    sources_used: List[str] = Field(description="List of document sections or contexts referenced.")

# Step 1: Load and Chunk Document
def process_document(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

# Step 2: Contextual Retrieval Transformation (Context Generation + Prepending)
def apply_contextual_retrieval(chunks: List[Document]) -> List[Document]:
    # Extract full document text efficiently
    full_doc_text = "\n\n".join([chunk.page_content for chunk in chunks])
    
    # Use ChatPromptTemplate.from_messages for structured prompt construction
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert document analyzer. Given a document and a snippet from it, answer with a brief 1-2 sentence context explaining where this snippet fits and what it covers."),
        ("human", "Document overview:\n{document}\n\nChunk snippet:\n{chunk}\n\nProvide short context to prepend:")
    ])
    
    # Free Mistral model for context generation
    # Low temperature prevents contextual hallucinations
    context_llm = ChatMistralAI(
        model="mistral-small",
        temperature=0.0
    )
    
    context_chain = context_prompt | context_llm
    
    contextualized_chunks = []
    
    # Truncate full doc overview to avoid context window explosion
    doc_summary = full_doc_text[:3000]
    
    for chunk in chunks:
        # Generate specific context header for current chunk
        generated_context = context_chain.invoke({
            "document": doc_summary,
            "chunk": chunk.page_content
        }).content
        
        # Create new document with prepended context
        # Prepending enriches both dense embeddings and BM25 keywords
        new_content = f"Context: {generated_context}\n\nContent: {chunk.page_content}"
        
        contextualized_doc = Document(
            page_content=new_content,
            metadata=chunk.metadata
        )
        contextualized_chunks.append(contextualized_doc)
        
    return contextualized_chunks

# Step 3: Build Contextual Hybrid Retriever
def build_contextual_retriever(contextualized_chunks: List[Document]):
    embedding_model = MistralAIEmbeddings(
        model="mistral-embed"
    )
    
    # Chroma indexes contextually prepended chunks
    vectorstore = Chroma.from_documents(
        documents=contextualized_chunks,
        embedding=embedding_model
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # BM25 captures keywords generated in prepended headers
    sparse_retriever = BM25Retriever.from_documents(contextualized_chunks)
    sparse_retriever.k = 10
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

# Step 4: Add Cohere Reranking Layer
def setup_reranked_pipeline(ensemble_retriever):
    compressor = CohereRerank(
        model="rerank-english-v3.0",
        top_n=4
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )
    return compression_retriever

# Step 5: Execution with Structured Output
def execute_rag_pipeline(retriever, query: str) -> ExtractedInsights:
    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise data extraction assistant. Answer the user query using ONLY the provided contextualized document snippets."),
        ("human", "Retrieved Context:\n{context}\n\nQuery: {query}")
    ])
    
    llm = ChatMistralAI(
        model="mistral-small",
        temperature=0.0
    )
    
    structured_llm = llm.with_structured_output(ExtractedInsights)
    rag_chain = rag_prompt | structured_llm
    
    response = rag_chain.invoke({
        "context": context_text,
        "query": query
    })
    
    return response

if __name__ == "__main__":

    file = r"0. Documents\Simple Doc 10 pages.pdf"

    test_queries = [
        "What specific vector metrics does it use to solve chunk isolation?",
        "Which module handles BM25 sparse search and what function does it serve in hybrid setups?",
        "Compare the purpose of Cross-Encoder reranking in Module 7 with Self-RAG reflection tokens in Module 8."
    ]

    chunks = process_document (file)
    context_chunks = apply_contextual_retrieval(chunks= chunks)
    ensemble = build_contextual_retriever(context_chunks)
    retriever = setup_reranked_pipeline(ensemble_retriever= ensemble)

    for i,query in enumerate(test_queries):
        print(f"Query {i+1} : {query}")
        response = execute_rag_pipeline(query= query, retriever= retriever)
        print(f"{response.page_content}\n")