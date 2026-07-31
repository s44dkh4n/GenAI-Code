from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

# 1. Create a dummy dataset
# Note how Document 1 contains some completely irrelevant gossip in the middle of a recipe.
documents = [
    Document(
        page_content="""
        By the way, did you hear that the local library is hosting a book sale next Tuesday? They have some rare first editions.
        To bake the perfect chocolate chip cookies:
        Preheat your oven to 375°F (190°C).
        Cream together the butter, white sugar, and brown sugar until smooth.
        Stir in the chocolate chips and spoon onto a baking sheet.
        """,
        metadata={"source": "recipe_book.txt"}
    ),
    Document(
        page_content="""
        The standard process to fix a leaking faucet:
        Turn off the main water supply valve under the sink.
        Remove the faucet handle using an Allen wrench or screwdriver.
        Replace the worn-out O-ring or cartridge inside the stem.
        """,
        metadata={"source": "plumbing_guide.txt"}
    )
]

# 2. Setup standard vector database and retriever
vectorstore = Chroma.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Setup the Contextual Compressor
# The EmbeddingsFilter splits documents and only keeps chunks that match the query 
# above our defined similarity threshold (e.g., 0.75).
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings, 
    similarity_threshold=0.75
)

# Wrap our base retriever with the compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter, 
    base_retriever=base_retriever
)

# 4. Test the retrieval
query = "What temperature should I set the oven to for baking cookies?"

print("--- STANDARD RETRIEVAL ---")
# Returns the entire recipe document, including the irrelevant gossip about the library.
standard_docs = base_retriever.invoke(query)
for i, doc in enumerate(standard_docs):
    print(f"Doc {i+1}:\n{doc.page_content.strip()}\n")

print("--- CONTEXTUAL COMPRESSION RETRIEVAL ---")
# Only returns the portion of the document directly relevant to baking temperatures.
compressed_docs = compression_retriever.invoke(query)
for i, doc in enumerate(compressed_docs):
    print(f"Doc {i+1}:\n{doc.page_content.strip()}\n")