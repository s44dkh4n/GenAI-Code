from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Load the PDF resume
loader = PyPDFLoader(file_path="Documents/M.Saad.Resume.pdf")
docs = loader.load()

# Initialize the HuggingFaceEndpoint with conversational task
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation", 
    temperature=0.3,
)
# Wrap it in ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# Define your Pydantic schema
class ResumeSummary(BaseModel):
    skills: list[str] = Field(description="Extract the skills from the Resume but discard the Soft skills") 
    missingSkills: list[str] = Field(description="Identify 3-5 skills that naturally complement the existing skills and improve employability") 
    jobRoles: list[str] = Field(description="Recommend Job Role on basis of skills")
    summary: str = Field(description="Generate a short summary of the Resume (between 200 to 320 characters). Mention things like User Name, past work experiences, and their strongest technical field.")
    resumeScore : int = Field(description="Resume score from 1 to 100")

# Initialize parser
parser = PydanticOutputParser(pydantic_object=ResumeSummary)

# Rewrite prompt to use ChatPromptTemplate (Required for conversational models)
Prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Expert Resume Analyzer.\nYour Task is to extract the following from the Resume in this format:\n{format_instructions}"),
    ("user", "Resume -> {resume}")
]).partial(format_instructions=parser.get_format_instructions())

# Rebuild the chain using chat_model instead of llm
chain = Prompt | chat_model | parser

# Invoke the chain
result = chain.invoke({"resume": docs[0].page_content})

print(result)