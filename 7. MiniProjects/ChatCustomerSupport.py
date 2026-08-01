from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEndpointEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

# Loader Initialization
loader = DirectoryLoader(
    path=r"D:\Course\Projects\Documents",
    glob="Documents_*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

# Adding metadata so they can be Classififed Betterly 
for doc in docs:
    source_path = os.path.basename(doc.metadata["source"]).lower()

    if "billing" in source_path or "refund" in source_path or "subscription" in source_path:
        doc.metadata["category"] = "billing"
    
    elif "api_setup" in source_path or "password_reset" in source_path or "troubleshooting" in source_path:
        doc.metadata["category"] = "technical"
    
    elif "returns" in source_path or "shipping" in source_path:
        doc.metadata["category"] = "shipping"

# Splitter Instance
splitter = RecursiveCharacterTextSplitter(
    chunk_size= 800,
    chunk_overlap= 150
)

# Splitted Documents
split_docs = splitter.split_documents(docs)
print("Number of Documents After Splitting:",len(split_docs),"\n")

technical_docs = []
billing_docs = []
shipping_docs = []

for doc in split_docs:
    category = doc.metadata.get("category")
    if category == "billing":
        billing_docs.append(doc)
    
    elif category == "technical":
        technical_docs.append(doc)
    
    elif category == "shipping":
        shipping_docs.append(doc)

# Embedding Model
embed = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

# Technical VectorStore
technical_vectorStore = Chroma.from_documents(
    documents= technical_docs,
    persist_directory=r"VectorStoreCollections\ChatCustomerSupport\TechnicalVecStore",
    collection_name="Technical",
    embedding=embed
)

# Billing VectorStore
billing_vectorStore = Chroma.from_documents(
    documents= billing_docs,
    persist_directory=r"VectorStoreCollections\ChatCustomerSupport\BillingVecStore",
    collection_name="Billing",
    embedding=embed
)

# Shipping vectorStore
shipping_vectorStore = Chroma.from_documents(
    documents= shipping_docs,
    persist_directory=r"VectorStoreCollections\ChatCustomerSupport\ShippingVecStore",
    collection_name="Shipping",
    embedding=embed
)

# Retrievers
ship_retriever = shipping_vectorStore.as_retriever(search_type="similarity",search_kwargs={"k":3})
bill_retriever = billing_vectorStore.as_retriever(search_type="similarity",search_kwargs={"k":3})
tech_retriever = technical_vectorStore.as_retriever(search_type="similarity",search_kwargs={"k":3})

# Prompt Template
prompt= ChatPromptTemplate.from_template(
    """
    You are an AI customer support assistant.
Your job is to answer the user's question using ONLY the provided context.
If the answer is not available in the context, say:
'I could not find that information in the support documents.'

Rules:
1. Be clear and professional.
2. Do not make up policies, prices, or procedures.
3. If steps are needed, present them in numbered form.
4. If the context includes policy conditions, mention them clearly.
5. Keep the answer focused on the user's question.

Question -> {question}\n\n
Context -> {context} \n\n
Answer -> 
"""
)

# Chat Model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.3
)
model = ChatHuggingFace(llm= llm)

def queryRouter(query):
    query = query.lower()

    technical_keywords = [
        "password", "login", "reset", "api", "bug", "issue",
        "error", "technical", "troubleshoot", "setup", "install", "failed"
    ]

    billing_keywords = [
        "bill", "billing", "refund", "subscription", "plan",
        "price", "payment", "charged", "invoice", "cancel"
    ]

    shipping_keywords = [
        "shipping", "delivery", "return", "returns", "order",
        "package", "track", "dispatch", "courier"
    ]

    if any(word in query for word in technical_keywords):
        return "technical", tech_retriever

    if any(word in query for word in billing_keywords):
        return "billing", bill_retriever

    if any(word in query for word in shipping_keywords):
        return "shipping", ship_retriever

    # This is added so that if Query doesn't Matched any type we will use Billing Retriever as Default
    return "billing", bill_retriever


# Helper to format docs
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

parser = StrOutputParser()

# Main Chatbot Function
def support_chatbot(query: str):
    category, retriever = queryRouter(query)
    retrieved_docs = retriever.invoke(query)
    context = format_docs(retrieved_docs)

    chain = prompt | model | parser
    answer = chain.invoke({
        "question": query,
        "context": context
    })

    return {
        "category": category,
        "answer": answer,
        "retrieved_docs": retrieved_docs
    }

while True:
        user_query = input("You: ")

        if user_query.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        result = support_chatbot(user_query)

        print(f"\nDetected Category: {result['category']}")
        print(f"Chatbot: {result['answer']}\n")