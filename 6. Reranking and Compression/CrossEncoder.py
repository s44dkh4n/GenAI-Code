from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

parser = StrOutputParser()

loader = TextLoader(r"0. Documents\EncodersPractice.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 400,
    chunk_overlap = 50
)
chunks = splitter.split_documents(docs)

embeddings = MistralAIEmbeddings(model="mistral-embed")

vec_store = Chroma.from_documents(
    embedding = embeddings,
    documents= chunks
)

retriever = vec_store.as_retriever(search_kwargs = {"k":15})

query = "How to reduce model memory footprint using quantization?"

top_15_retrieved = retriever.invoke(query) 

# for i, answer in enumerate(top_15_retrieved):
#     print(f"Answer: {i+1} \n{answer}\n")

query_answer_pairs = [(query, doc.page_content) for doc in top_15_retrieved]

model_name = "BAAI/bge-reranker-base"
cross_encoder = CrossEncoder(model_name_or_path= model_name)

encoder_scores = cross_encoder.predict(
    query_answer_pairs,
    batch_size= 15,
    show_progress_bar= False
)

reranked_docs = sorted(
    zip(top_15_retrieved, encoder_scores),
    key= lambda x: x[1],    # For each tuple, sort using the second element of zip function(i.e the score).
    reverse= True
)

top_5 = [doc for doc,_ in reranked_docs[:5]]
for top in top_5:
    print(top.page_content)

# context = "\n\n".join(
#         doc.page_content
#         for doc in top_5
#     )

# print(reranked_docs)
