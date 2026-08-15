from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# 1. Document Loading
loader = TextLoader(file_path=r"0. Documents\EncodersPractice.txt")
docs = loader.load()
print("Loaded docs...")

# 2. Text Splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
chunks = splitter.split_documents(documents=docs)
print("Docs split...")

# 3. Vectorization and Storage
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:4b", 
    dimensions=1536,
    base_url="http://localhost:11434"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. LLM Initialization
model = ChatOllama(
    model="qwen2.5:7b",
    temperature=0.3,
    base_url="http://localhost:11434"
)

# 5. Pipeline Prompt
prompt = ChatPromptTemplate.from_template(
    template="You are an expert document answerer. Answer only from the provided context if its not present in context you say 'I do not know'\n\n Question -> {question} \n\n Context -> {context}"
)

# Document Formatting Function
def format_cxt(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# FIXED: Build the chain OUTSIDE the loop for speed and efficiency
chain = (
    {
        "context": retriever | format_cxt,
        "question": RunnablePassthrough()
    } 
    | prompt
    | model
    | parser
)

print("RAG System Ready! Type 'exit' to quit.\n")

# 6. Chat Loop
while True:
    query = input("You -> ")

    if query.strip().lower() == "exit":
        print("Goodbye!")
        break

    # Skip empty inputs to prevent accidental API errors
    if not query.strip():
        continue

    # Execution is much lighter now that the chain structure is compiled
    result = chain.invoke(query)
    print(f"AI -> {result}\n")
