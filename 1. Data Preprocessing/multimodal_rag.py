import json
from typing import List

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

def partition_document(file_path: str):
    # Extract elements from PDF using unstructured
    print(f"📄 Partitioning document: {file_path}")
    
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True
    )
    
    print(f"✅ Extracted {len(elements)} elements")
    return elements

def create_chunks_by_title(elements):
    # Create intelligent chunks using title-based strategy
    print("🔨 Creating smart chunks...")
    
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

def separate_content_types(chunk):
    # Analyze what types of content are in a chunk
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

# OpenRouter acts as an OpenAI-compatible gateway for open-source models
def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    try:

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is missing from environment variables.")
        # Configures OpenRouter base URL to access hosted open-source vision models
        vision_llm = ChatOpenAI(
            model="qwen/qwen-2.5-vl-7b-instruct",
            openai_api_key="YOUR_OPENROUTER_API_KEY", # Insert your free/cheap OpenRouter API key
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.2, # inline comment: low temperature minimizes hallucination
            max_tokens=512 # inline comment: token limit per response payload
        )

        prompt_text = f"CONTENT TO ANALYZE:\nTEXT:\n{text}\n"
        if tables:
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n"

        prompt_text += "\nProvide a concise, searchable summary of the content above:"

        message_content = [{"type": "text", "text": prompt_text}]

        for image_base64 in images:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        response = vision_llm.invoke([HumanMessage(content=message_content)])
        return response.content

    except Exception as e:
        print(f"❌ Summary failed: {e}")
        return text[:300]
    
def summarise_chunks(chunks):
    # Process all chunks with AI Summaries
    print("🧠 Processing chunks with AI Summaries...")
    
    langchain_documents = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        current_chunk = i + 1
        print(f"   Processing chunk {current_chunk}/{total_chunks}")
        
        content_data = separate_content_types(chunk)
        
        if content_data['tables'] or content_data['images']:
            print(f"     → Creating AI summary for mixed content...")
            try:
                enhanced_content = create_ai_enhanced_summary(
                    content_data['text'],
                    content_data['tables'], 
                    content_data['images']
                )
            except Exception as e:
                print(f"     ❌ AI summary failed: {e}")
                enhanced_content = content_data['text']
        else:
            enhanced_content = content_data['text']
        
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images']
                })
            }
        )
        
        langchain_documents.append(doc)
    
    print(f"✅ Processed {len(langchain_documents)} chunks")
    return langchain_documents

def create_vector_store(documents, persist_directory="dbv1/chroma_db"):

    print("🔮 Creating embeddings and storing in ChromaDB...")
        
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5" # Free open-source state-of-the-art embedding model
    )
    
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"✅ Vector store created and saved to {persist_directory}")
    return vectorstore

def generate_final_answer(chunks, query):

    try:
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-VL-72B-Instruct", # Large multimodal open-source model
            temperature=0.1, # low temperature ensures high factual accuracy
            max_new_tokens=1024, # token limit for comprehensive response output
            timeout=60.0 # handle larger model request overhead safely
        )
        vision_llm = ChatHuggingFace(llm=llm)
        
        prompt_text = f"""Based on the following documents, please answer this question: {query}\n\nCONTENT TO ANALYZE:\n"""
        
        for i, chunk in enumerate(chunks):
            prompt_text += f"--- Document {i+1} ---\n"
            
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                
                raw_text = original_data.get("raw_text", "")
                if raw_text:
                    prompt_text += f"TEXT:\n{raw_text}\n\n"
                
                tables_html = original_data.get("tables_html", [])
                if tables_html:
                    prompt_text += "TABLES:\n"
                    for j, table in enumerate(tables_html):
                        prompt_text += f"Table {j+1}:\n{table}\n\n"
            
            prompt_text += "\n"
        
        prompt_text += """Please provide a clear answer. If documents lack info, say "I don't have enough information."\n\nANSWER:"""

        message_content = [{"type": "text", "text": prompt_text}]
        
        for chunk in chunks:
            if "original_content" in chunk.metadata:
                original_data = json.loads(chunk.metadata["original_content"])
                images_base64 = original_data.get("images_base64", [])
                
                for image_base64 in images_base64:
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    })
        
        message = HumanMessage(content=message_content)
        response = vision_llm.invoke([message])
        
        return response.content
        
    except Exception as e:
        print(f"❌ Answer generation failed: {e}")
        return "Sorry, I encountered an error while generating the answer."

def run_complete_ingestion_pipeline(pdf_path: str):
    print("🚀 Starting RAG Ingestion Pipeline")
    print("=" * 50)
    
    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised_chunks = summarise_chunks(chunks)
    db = create_vector_store(summarised_chunks, persist_directory="dbv2/chroma_db")
    
    print("🎉 Pipeline completed successfully!")
    return db

# Execution Flow
db = run_complete_ingestion_pipeline(r"D:\Course\Projects\0. Documents\Attention.pdf")
query = "How many attention heads does the Transformer use, and what is the dimension of each head?"

retriever = db.as_retriever(search_kwargs={"k": 3})
retrieved_chunks = retriever.invoke(query)

final_answer = generate_final_answer(retrieved_chunks, query)
print(final_answer)