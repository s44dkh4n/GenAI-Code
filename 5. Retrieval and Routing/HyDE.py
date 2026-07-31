from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Temperature 0.1 for consistent hypothetical generation and grounded synthesis
mistral_model = ChatMistralAI(model="mistral-small-latest", temperature=0.1)

embeddings = MistralAIEmbeddings(model="mistral-embed")
parser = StrOutputParser()


def get_vector_store(path: str, persist_dir: str = "./chroma_db"):
    # Load document payload
    loader = PyPDFLoader(path)
    docs = loader.load()

    # Recursive chunking with overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)
    chunks = splitter.split_documents(docs)

    # Initialize vector store with persistent storage
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_dir
    )
    return vector_store


def hyde_retrieval(query: str, vector_store: Chroma):
    # Prompt 1: Generate hypothetical document
    hyde_prompt = ChatPromptTemplate.from_template(
        "Write a detailed hypothetical passage that answers the following question.\n"
        "Question: {question}"
    )
    hyde_chain = hyde_prompt | mistral_model | parser
    hypothetical_doc = hyde_chain.invoke({"question": query})

    # Retrieve relevant document chunks using the hypothetical passage
    top_k_results = vector_store.similarity_search(hypothetical_doc, k=3)

    # Format context chunks into a single readable string
    context_str = "\n\n".join([doc.page_content for doc in top_k_results])

    # Prompt 2: Grounded synthesis using retrieved context
    qa_prompt = ChatPromptTemplate.from_template(
        "Answer the user query strictly based on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}"
    )
    final_chain = qa_prompt | mistral_model | parser
    final_response = final_chain.invoke(
        {"context": context_str, "query": query}
    )

    return {
        "Actual Answer": final_response,
        "Relevant Chunks": top_k_results,
        "Hypothetical Answer": hypothetical_doc,
    }


if __name__ == "__main__":
    pdf_path = (
        r"D:\Course\Projects\0. Documents\langchain_enterprise_guide.pdf"
    )

    vector_db = get_vector_store(pdf_path)
    result = hyde_retrieval(
        query="Advanced Retrieval Pipeline Components of RAG",
        vector_store=vector_db,
    )

    print("Hypothetical Document:\n", result["Hypothetical Answer"])
    print("\nFinal Answer:\n", result["Actual Answer"])