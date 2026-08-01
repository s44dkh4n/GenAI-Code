from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

# Model Selection & Creation
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.4,
    task="text-generation",
    max_new_tokens=250
)
model = ChatHuggingFace(llm = llm)

class OutPut(BaseModel):

    skills : list[str] = Field(description="Extract all the skills from the resume")
    strength: list[str] = Field(description="Extract all the strength of the person from the resume")
    weakness: list[str] = Field(description="Extract all the weaknesses of the person from the resume")
    recommended_roles: list[str] = Field(description="Recommend some job roles the person based on the skills and strength and weaknesses you from the resume")

parser = PydanticOutputParser(pydantic_object=OutPut)

prompt = PromptTemplate(
    template= " Extract the details from the resume below. \n {format_instructions}\n resume -> {resume} ",
    input_variables=["resume"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke("""Muhammad Khan
Peshawar, Pakistan | khan@email.com | linkedin.com/in/mkhan | github.com/mkhan

EDUCATION
Bachelor of Science in Computer Science — Expected 2026
University of Engineering and Technology (UET), Peshawar
CGPA: 3.7/4.0

TECHNICAL SKILLS

Languages: Python, C++, SQL
ML/AI: NumPy, Pandas, Scikit-learn, LangChain, MLFlow, NLTK
Tools & Platforms: Git, GitHub, VS Code, Jupyter Notebook, uv
Other: NLP, Machine Learning, Data Preprocessing, REST APIs


PROJECTS
Cyberbullying Detection System (NLP & Machine Learning)

Built a text classification pipeline to detect cyberbullying in social media content using NLP techniques and machine learning models
Applied preprocessing, feature extraction (TF-IDF, word embeddings), and model evaluation with Scikit-learn
Achieved high classification accuracy through hyperparameter tuning and cross-validation


EXPERIENCE & LEADERSHIP
Coordinator — Computer Cell Society, UET Peshawar (2023 – Present)

Coordinated campus tech events, workshops, and seminars for students
Managed volunteer teams and handled event logistics and outreach activities
Fostered student engagement in computing and technology initiatives


CERTIFICATIONS & LEARNING

Machine Learning with Python (self-directed)
LangChain for LLM Applications


INTERESTS
Applied Machine Learning · Natural Language Processing · AI Engineering · Open Source""")

print(result)
