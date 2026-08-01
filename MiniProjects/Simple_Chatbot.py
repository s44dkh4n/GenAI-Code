from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage
from dotenv import load_dotenv

load_dotenv()

messages = [
    SystemMessage(content="You are a helpful AI Assistant")
]

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature="0.9"
)

model = ChatHuggingFace(llm = llm)

while True:
    user_input = input("You: ")
    messages.append(HumanMessage(content=user_input))
    if user_input.lower() == "exit":
        break

    result = model.invoke(user_input)
    messages.append(AIMessage(content=result.content))
    print("AI:",result.content)


print("="*100)
print(messages)