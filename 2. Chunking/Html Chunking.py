from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader(r"0. Documents\simple html.html")
docs = loader.load()

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"), 
        ("h2", "Header 2"),
        ("h3","Header 3"),
        ],
        return_each_element= False # If True, every element (headers, paragraphs, etc.)  is returned as a separate Document.
)

chunks = splitter.split_text(docs[0].page_content)

for i,chunk in enumerate(chunks):
    print(f"Chunk {i+1} : {chunk.page_content} \nMetaData : {chunk.metadata}")
    print("=="*50)