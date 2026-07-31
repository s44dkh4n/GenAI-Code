from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embedd_model = MistralAIEmbeddings(
    model="mistral-embed"
)

# This fuction splits text SEMANTICALLY by Default but Can be used to split RECURSIVE by changing to useSemantic= False
def Smart_Chunker(text, useSemantic = True, chars_size = 900, overlap = 180):

    if useSemantic:

        sem_chunker = SemanticChunker(
            embeddings = embedd_model,  
            breakpoint_threshold_type = "percentile", 
            breakpoint_threshold_amount = 90
        )

        if isinstance(text, str):
            sem_chunks = sem_chunker.split_text(text)

        elif isinstance(text, list) and all(isinstance(item, Document) for item in text):
            sem_chunks = sem_chunker.split_documents(text)
        
        elif isinstance(text, list) and all(isinstance(item, str) for item in text):
            combined_text = "\n\n".join(text)
            sem_chunks = sem_chunker.split_text(text)
        
        return sem_chunks
    
    else:

        rec_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chars_size,
            chunk_overlap = overlap 
        )

        if isinstance(text, str):
            rec_chunks = rec_splitter.split_text(text)

        elif isinstance(text, list) and all(isinstance(item, Document) for item in text):
            rec_chunks = rec_splitter.split_documents(text)
        
        elif isinstance(text, list) and all(isinstance(item, str) for item in text):
            combined_text = "\n\n".join(text)
            rec_chunks = rec_splitter.split_text(text)
        
        return rec_chunks