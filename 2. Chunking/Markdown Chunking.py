from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path= r"0. Documents\simple markkdown.txt")
docs = loader.load()

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on= [
        ("#","Header 1"),
        ("##","Header 2"),
        ("###","Header 3")
    ],
    strip_headers= True
)

markdown_chunks = splitter.split_text(docs[0].page_content)

for i,chunk in enumerate(markdown_chunks):
    print(f"Chunk {i+1} : {chunk.page_content} \nMetaData : {chunk.metadata}")
    print("\n\n")

# Sometimes The MarkdownHeaderTextSplitter divides data in such big chunks that can't be processed 
# by an embedding Model so always use RecursiveCharacterTextSplitter along with MarkdownHeaderTextSplitter

textsplitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

chunks = textsplitter.split_documents(markdown_chunks)

for i,chunk in enumerate(chunks):
    print(f"Chunk {i+1} : {chunk.page_content} \nMetaData : {chunk.metadata}")
    print("\n\n")