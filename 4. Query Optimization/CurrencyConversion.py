import json
from typing import Annotated
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import requests

load_dotenv()

API_KEY = "2d379e82e7a553c9b0fe7ceb"

# Tool Definitions
@tool
def get_current_rate(your_currency: str, exchange_currency: str) -> str:
    """Get the current exchange rate between your_currency and exchange_currency."""
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{your_currency}/{exchange_currency}"
    response = requests.get(url=url)
    data = response.json()
    
    # Return only key info to save tokens and prevent context clutter
    if data.get("result") == "success":
        return json.dumps({
            "base_code": data.get("base_code"),
            "target_code": data.get("target_code"),
            "conversion_rate": data.get("conversion_rate")
        })
    return json.dumps({"error": "Failed to fetch exchange rate"})

@tool
def conversion(currency: float, exchange_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert any amount of currency using a given exchange rate."""
    return currency * exchange_rate

tools = [get_current_rate, conversion]
tools_by_name = {t.name: t for t in tools}

# Model Setup - Increased max_new_tokens to give room for generation
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1,
    max_new_tokens=512,
    timeout=30,
)
model = ChatHuggingFace(llm=llm)
model_with_tools = model.bind_tools(tools=tools)

base_currency = input("Enter Base Currency (e.g., 100 INR): ")
target_currency = input("Enter Target Currency (e.g., PKR): ")

messages = [
    SystemMessage("You are a helpful assistant for currency conversion. First use get_current_rate to fetch the exchange rate, then summarize the result clearly for the user."),
    HumanMessage(f"Convert {base_currency} into {target_currency}")
]

# Step 1: Initial invocation using tool-bound model
ai_response = model_with_tools.invoke(messages)
messages.append(ai_response)

# Step 2: Handle tool calls
if ai_response.tool_calls:
    for call in ai_response.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"].copy()
        
        if tool_name == "get_current_rate":
            raw_result = get_current_rate.invoke(tool_args)
            
            tool_message = ToolMessage(
                content=raw_result,
                tool_call_id=call["id"]
            )
            messages.append(tool_message)

    # Step 3: Invoke the UNBOUND model to force final text generation
    final_message = model.invoke(messages)
    messages.append(final_message)
    
    print(f"\nFinal Result:\n{final_message.content}\n")

print("Complete Chat History:")
for msg in messages:
    print(f"- {type(msg).__name__}: {msg.content}")