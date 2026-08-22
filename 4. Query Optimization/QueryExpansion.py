import os
from typing import List
from pydantic import BaseModel, Field

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document

# 1. Initialize Models
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)
embeddings = MistralAIEmbeddings(model="mistral-embed")

# 2. Define Output Schema for Query Expansion
class ExpandedQueries(BaseModel):
    queries: List[str] = Field(
        description="List of 3 to 5 alternative formulations of the user's input query."
    )

parser = PydanticOutputParser(pydantic_object=ExpandedQueries)

# 3. Create Query Expansion Prompt
expansion_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI assistant specialized in information retrieval. "
     "Your task is to generate 3 to 5 distinct rephrasings or related versions of the user's input query. "
     "Cover different synonyms, technical terms, and perspectives to maximize search recall.\n"
     "{format_instructions}"),
    ("human", "{user_query}")
]).partial(format_instructions=parser.get_format_instructions())

# Chain to generate alternative queries
query_expansion_chain = expansion_prompt | llm | parser

# 4. Helper Function: Multi-Query Retrieval and Deduplication
def retrieve_with_query_expansion(user_query: str, retriever, top_k: int = 3) -> List[Document]:
    
    # Step A: Expand the original query
    expanded_result = query_expansion_chain.invoke({"user_query": user_query})
    all_queries = [user_query] + expanded_result.queries
    
    print(f"Original Query: {user_query}")
    print("Generated Variants:")
    for q in expanded_result.queries:
        print(f" - {q}")
    
    # Step B: Execute retrieval for all queries
    retrieved_docs = []
    seen_doc_ids = set()
    
    for q in all_queries:
        docs = retriever.invoke(q)
        for doc in docs:
            # Deduplicate based on document content or metadata identifier
            doc_identifier = doc.metadata.get("id", doc.page_content)
            if doc_identifier not in seen_doc_ids:
                seen_doc_ids.add(doc_identifier)
                retrieved_docs.append(doc)
                
    return retrieved_docs

if __name__ == "__main__":
    # Create a dummy vector store with sample documents
    sample_documents = [
        Document(page_content="HNSW index parameters control latency and recall trade-offs in vector databases.", metadata={"id": 1}),
        Document(page_content="Vector search latency can be reduced by lowering M and efConstruction parameters during index creation.", metadata={"id": 2}),
        Document(page_content="BM25 and semantic search combined via Reciprocal Rank Fusion yield optimal hybrid retrieval results.", metadata={"id": 3}),
    ]

    vectorstore = Chroma.from_documents(sample_documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Run retrieval with Query Expansion
    query = "How to fix slow vector similarity search?"
    results = retrieve_with_query_expansion(query, retriever)

    print(f"\nRetrieved {len(results)} unique documents:")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.page_content}")