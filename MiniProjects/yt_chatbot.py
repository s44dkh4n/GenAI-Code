from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint,HuggingFaceEndpointEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.documents import Document
import youtube_transcript_api
from youtube_transcript_api._api import YouTubeTranscriptApi as APIClient
from dotenv import load_dotenv 

# Load environment variables
load_dotenv()

# Full link: https://youtu.be/Gfr50f6ZBvo?si=zXrUCPLrx1StY5Gr
video_id = "Gfr50f6ZBvo"

try:
    # Directly using the internal API client
    client = APIClient()

    # The method in the actual engine is 'list'
    transcript_list = client.list(video_id)

    # Get the English transcript
    transcript = transcript_list.find_transcript(['en'])
    transcript_data = transcript.fetch()

    # Handling both dictionary and object formats
    try:
        transcript_text = " ".join(word["text"] for word in transcript_data)
    except (TypeError, KeyError):
        # Fallback for when elements are objects (FetchedTranscriptSnippet)
        transcript_text = " ".join(word.text for word in transcript_data)

    print("Transcript successfully retrieved.")

except Exception as e:
    print(f"Direct retrieval failed: {e}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

split_doc = splitter.create_documents([transcript_text])

embed_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

vector_store = Chroma.from_documents(
    documents=split_doc,
    embedding=embed_model,
    collection_name="yt_TRNSRPT",
    persist_directory="VectorStoreCollections"
)

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs = {"k":4})
query = "Was Nuclear fusion discussed in this video? If yes then what was discussed."

prompt = ChatPromptTemplate.from_template(
    """
You are an Expert Youtube Transcript Reader and Knowledge Saver.

Answer the question ONLY from the provided context.

If the answer is not present in the context, say:
"I could not find that information in the provided documents."

transcript_text:
{text}

Query:
{query}
"""
)

parser = StrOutputParser()

# Helper Function to join all the documents
def format_docs(docus):
    return "\n\n".join(doc.page_content for doc in docus)

# Initialize the HuggingFaceEndpoint with conversational task
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    temperature=0.3,
)
chat_model = ChatHuggingFace(llm=llm)

chain = (
    {
        "text" : retriever | format_docs,
        "query": RunnablePassthrough()
    }

    | prompt | chat_model | parser
)

result = chain.invoke(query)
print(result)