import os
from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatHuggingFace

# Define Structured Output Schema 
class SubQueries(BaseModel):
    queries: List[str] = Field(
        description="A list of 2-4 standalone search queries broken down from the complex input question."
    )

# Initialize Decomposer Chain 
def build_decomposer_chain(llm):
    # PydanticOutputParser enforces structured schema validation
    parser = PydanticOutputParser(pydantic_object=SubQueries)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert search query planner. Break down complex queries into simple standalone sub-queries.\n{format_instructions}"),
        ("human", "{question}")
    ])
    
    # Partial prompt binding for format instructions ensures clean schema injection
    prompt_bound = prompt.partial(format_instructions=parser.get_format_instructions())
    
    return prompt_bound | llm | parser

# Parallel Retrieval Execution 
def retrieve_sub_query_contexts(sub_queries: List[str], vectorstore: Chroma, k: int = 2) -> List[Document]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    aggregated_docs = []
    seen_contents = set() # Track unique text chunks to prevent token duplication

    for query in sub_queries:
        docs = retriever.invoke(query)
        for doc in docs:
            # Deduplicate documents based on content hash/text
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                aggregated_docs.append(doc)
                
    return aggregated_docs

# Synthesis Chain 
def build_synthesis_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question comprehensively using only the provided background context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    return prompt | llm

# Full Pipeline Runner 
def run_subquery_rag(question: str, vectorstore: Chroma, llm):
    # Hyperparameter Settings:
    # temperature=0.0: Forces deterministic sub-query parsing and strict factual synthesis
    # top_p=1.0: Kept standard when temperature is zero
    # max_tokens=512: Cap response length to prevent runaway costs/latency
    
    decomposer = build_decomposer_chain(llm)
    synthesizer = build_synthesis_chain(llm)
    
    # Step 1: Decompose query
    parsed_queries: SubQueries = decomposer.invoke({"question": question})
    print(f"Generated Sub-Queries: {parsed_queries.queries}")
    
    # Step 2: Retrieve deduplicated contexts
    retrieved_docs = retrieve_sub_query_contexts(parsed_queries.queries, vectorstore)
    
    # Format context blocks for the synthesizer prompt
    context_str = "\n\n".join([f"--- Chunk ---\n{doc.page_content}" for doc in retrieved_docs])
    
    # Step 3: Synthesize final answer
    response = synthesizer.invoke({
        "context": context_str,
        "question": question
    })
    
    return response.content