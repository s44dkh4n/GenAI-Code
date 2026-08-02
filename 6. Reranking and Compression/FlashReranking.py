from langchain_community.document_compressors import FlashrankRerank
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever

# 1. Create a dummy corpus and setup standard VectorStore
docs = [
    Document(page_content="Downtown 2-bedroom luxury condo for $4200. Includes in-unit laundry and garage parking. Pets welcome."),
    Document(page_content="Cozy 2-bedroom apartment downtown for $2300. Features in-unit washer/dryer, cats allowed."),
    Document(page_content="Affordable 2-bedroom downtown for $2100. Shared laundry room in basement. Strictly no pets allowed."),
    Document(page_content="Renovated loft downtown for $2450. Dog-friendly building with in-unit laundry facilities."),
]

# Free open-source embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# 2. Setup Base Retriever (High Recall: retrieve top 10 candidates)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 3. Initialize FlashRerank Document Compressor
# top_n: Number of documents to keep after reranking
# model: Default is "ms-marco-TinyBERT-L-2-v2" (~4MB CPU model)
compressor = FlashrankRerank(
    model="ms-marco-TinyBERT-L-2-v2",
    top_n=2, # Return only the top 2 reranked chunks
)

# 4. Wrap the base retriever with the Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# 5. Execute retrieval query directly
query = "Which apartments allow pets and have in-unit laundry under 2500?"
compressed_docs = compression_retriever.invoke(query)

# Inspect Reranked Output
for i, doc in enumerate(compressed_docs, 1):
    # FlashRerank adds a relevance score inside metadata
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"Rank {i} | Score: {score}")
    print(f"Content: {doc.page_content}\n")