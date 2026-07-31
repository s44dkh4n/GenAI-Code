from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)
embeddings = MistralAIEmbeddings(model="mistral-embed")

docs = [
    Document(
        page_content="""
        To bake the perfect chocolate chip cookies:
        Preheat your oven to 375°F (190°C). Cream together the butter, white sugar, and brown sugar until smooth. By the way, did you hear that the local library is hosting a book sale next Tuesday? They have some rare first editions. Stir in the chocolate chips and spoon onto a baking sheet.
        """,
        metadata={"source": "recipe_book.txt"}
    ),
    Document(
        page_content="""
        The standard process to fix a leaking faucet:
        1. Turn off the main water supply valve under the sink.
        2. Remove the faucet handle using an Allen wrench or screwdriver.
        3. Replace the worn-out O-ring or cartridge inside the stem.
        """,
        metadata={"source": "plumbing_guide.txt"}
    )
]

vecStore = Chroma.from_documents(
    embedding=embeddings,
    documents=docs
)

retriever = vecStore.as_retriever(search_kwargs={"k":2})
base_comp = LLMChainExtractor.from_llm(llm=model)

compressor = ContextualCompressionRetriever(
    base_compressor= base_comp,
    base_retriever= retriever
)

query = "What temperature should I set the oven to for baking cookies?"

print("--- STANDARD RETRIEVAL ---")
# Returns the entire recipe document, including the irrelevant gossip about the library.
standard_docs = compressor.invoke(query)
for i, doc in enumerate(standard_docs):
    print(f"Doc {i+1}:\n{doc.page_content.strip()}\n")

print("--- CONTEXTUAL COMPRESSION RETRIEVAL ---")
# Only returns the portion of the document directly relevant to baking temperatures.
compressed_docs = compressor.invoke(query)
for i, doc in enumerate(compressed_docs):
    print(f"Doc {i+1}:\n{doc.page_content.strip()}\n")