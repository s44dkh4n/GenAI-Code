import os
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# 1. Initialize Model
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.0)
embeddings = MistralAIEmbeddings(model="mistral-embed")

# 2. Define Query Rewriter Prompt
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert search engine query optimizer. "
     "Your task is to analyze the user's input and rewrite it into a clear, concise, "
     "and keyword-rich query optimized for semantic vector store retrieval. "
     "Do not answer the question. Return ONLY the rewritten query text."),
    ("human", "{user_query}")
])

# Chain: Prompt -> LLM -> String
rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# 3. Execution Function
def rewrite_and_retrieve(user_query: str, retriever):
    # Step A: Rewrite the Query
    rewritten_query = rewrite_chain.invoke({"user_query": user_query})
    
    print(f"Original Query:  '{user_query}'")
    print(f"Rewritten Query: '{rewritten_query}'\n")
    
    # Step B: Perform Retrieval with the Rewritten Query
    retrieved_docs = retriever.invoke(rewritten_query)
    return retrieved_docs

if __name__ == "__main__":
    # Create sample vector store
    docs = [
        Document(page_content="To create a collection in Chroma DB, call vectorstore = Chroma(collection_name='my_coll')."),
        Document(page_content="Chroma DB requires an embedding function to index text chunks."),
    ]

    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # Run pre-retrieval query rewriting
    raw_query = "Hey can you tell me how to make a new dataset bucket in chroma?"
    results = rewrite_and_retrieve(raw_query, retriever)

    print(f"Retrieved Result: {results[0].page_content}")