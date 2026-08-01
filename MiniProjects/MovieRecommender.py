from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Model Selection & Creation
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.4,
    task="text-generation",
    max_new_tokens=250
)
model = ChatHuggingFace(llm = llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
    You are an expert movie recommendation assistant.

    User Mood: {mood}
    Preferred Genre: {genre}

    Recommend exactly 2 movies.

    For each movie provide:
    1. Movie Name
    2. Release Year
    3. Two short reasons why it matches the user's mood and genre

    Response Format:

    Movie 1:
    Name:
    Release Year:
    Reasons:
    - 
    - 

    Movie 2:
    Name:
    Release Year:
    Reasons:
    - 
    - 

    Do not add introductions or conclusions.
""",
    input_variables=['mood','genre']
)

chain = prompt | model | parser

result = chain.invoke({'mood':"sad", "genre": "romantic"})
print(result)