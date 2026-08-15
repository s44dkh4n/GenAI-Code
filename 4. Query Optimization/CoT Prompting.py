from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

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
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. LLM Initialization
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct", 
    temperature=0.1, 
    max_new_tokens=1024, 
    timeout=30
)
model = ChatHuggingFace(llm= llm)

class CoTRAG(BaseModel):
    reasoning : list[str] = Field(description="Step-by-step evaluation of retrieved facts against the question.")
    result : str = Field(description="Concise final response based strictly on the reasoning steps.")

parser = OutputFixingParser.from_llm(
    llm=model,
    parser=PydanticOutputParser(pydantic_object=CoTRAG)
)

# 5. Pipeline Prompt
prompt = ChatPromptTemplate.from_messages([
   ("system","""You are an expert AI assistant that uses Chain-of-Thought reasoning.
               "Analyze the provided context, list your reasoning step-by-step, and answer the question.
               {format_instructions}"""),
    ("human","Question:\n {question} \n\n Context:\n {context}")
]).partial(format_instructions= parser.get_format_instructions())

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
    print(f"\nAI Reasoning -> {result.reasoning}")
    print(f"AI Result -> {result.result}\n")