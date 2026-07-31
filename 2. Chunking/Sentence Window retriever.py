import os
from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.0
)

embeddings = MistralAIEmbeddings(model="mistral-embed")

# Sample corpus
raw_text = """
The Apollo program was the third United States human spaceflight program carried out by NASA. 
It accomplished landing the first humans on the Moon in 1969. 
Commander Neil Armstrong and Lunar Module Pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. 
Armstrong became the first person to walk on the Moon six hours and 39 minutes later on July 21. 
Aldrin joined him 19 minutes later, and they spent two hours and a quarter together exploring the site. 
They collected 47.5 pounds of lunar material to bring back to Earth.
"""

# Step 1: Split raw text into sentences
sentence_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1,
    chunk_overlap=0,
    separators=["\n\n", "\n", ".", "!", "?"]
)

raw_sentences = [doc.page_content.strip() for doc in sentence_splitter.create_documents([raw_text]) if doc.page_content.strip()]

# Step 2: Build sentence window documents with metadata
window_size = 1  # 1 sentence before and 1 sentence after
sentence_docs = []

for idx, sentence in enumerate(raw_sentences):
    # Determine the context window bounds
    start_idx = max(0, idx - window_size)
    end_idx = min(len(raw_sentences), idx + window_size + 1)
    
    # Construct the surrounding window context
    window_context = " ".join(raw_sentences[start_idx:end_idx])
    
    # Store individual sentence as page_content and window in metadata
    sentence_docs.append(
        Document(
            page_content=sentence,
            metadata={
                "sentence_id": idx,
                "window_context": window_context
            }
        )
    )

# Step 3: Store in VectorDB (Embeddings built on small sentences)
vectorstore = Chroma.from_documents(
    documents=sentence_docs,
    embedding=embeddings
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Step 4: Custom Window Retriever Function
def sentence_window_retriever(query: str) -> str:
    # Retrieve top-k sentence matches
    matched_docs = base_retriever.invoke(query)
    
    # Extract surrounding window contexts from metadata
    extracted_windows = [doc.metadata["window_context"] for doc in matched_docs]
    
    # Combine and deduplicate retrieved context windows
    unique_windows = list(dict.fromkeys(extracted_windows))
    return "\n---\n".join(unique_windows)

# Step 5: Construct LCEL Chain (2-Step RAG)
prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context provided below.
    
Context:
{context}

Question: {question}
Answer:"""
)

rag_chain = (
    {
        "context": sentence_window_retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

# Step 6: Query
response = rag_chain.invoke("How much lunar material did they collect and who landed on the moon?")
print(response)