from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=500
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=["skills", "interests", "cgpa"],
    
    template="""
You are an expert AI Career Advisor.

Student Information:
Skills: {skills}
Interests: {interests}
CGPA: {cgpa}

Based on the above information, provide:

1. Best career paths
2. Important missing skills
3. Step-by-step learning roadmap

Keep the response clear and structured.
"""
)

parser = StrOutputParser()

chain = prompt | model | parser

skills = input("Enter your skills: ")
interests = input("Enter your interests: ")
cgpa = input("Enter your CGPA: ")


result = chain.invoke({
    "skills": skills,
    "interests": interests,
    "cgpa": cgpa
})


print("\n===== Career Advice =====\n")
print(result)