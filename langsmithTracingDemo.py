from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith.run_trees import RunTree
from langsmith import traceable 
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "RAG"

parser = StrOutputParser()

@traceable(name="Chain Tracing Demo ")
def tracingDemo():

    mistral_model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        "Explain {topic} in one sentence"
    )

    chain = prompt | mistral_model | parser
    result = chain.invoke("AI")
    print(f"Result -> {result}")

@traceable(name= "Summary tracing demo")
def demo_run():
    mistral_model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("Give short summary of -> {text}")

    chain = prompt | mistral_model | parser
    result = chain.invoke("AI is Future of Coding")
    print("Summary Result:\n",result)

@traceable(name="Greeting Tracing demo")
def greeting(username, req_type):

    mistral_model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
    )

    result = mistral_model.invoke(f"hello from {username}")

    print("\n",result.content)

if __name__ == "__main__":
    
    tracingDemo()
    demo_run()
    greeting("user123","greeting")
