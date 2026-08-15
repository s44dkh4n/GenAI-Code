from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

# Tool Creation
@tool
def add(a: int, b: int) -> int:
    """Return the sum of two numbers"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Return the difference of two numbers"""
    return a - b

tools = [add, subtract]
tools_by_name = {t.name: t for t in tools}

print(f"Tools available: \n{list(tools_by_name.keys())}\n")

# Initializing Chat Model and Binding it with Tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1,
    max_new_tokens=256,
    timeout=30,
)
model = ChatHuggingFace(llm=llm)
model_with_tools = model.bind_tools(tools=tools)

messages = [
    SystemMessage("You are an assistant who helps me with Arithmetic Operations."),
    HumanMessage("What is the sum of 5 and 7? Also tell me what is the difference of 599 and 435."),
]

# Step 1: Get initial response from model
ai_response = model_with_tools.invoke(messages)
messages.append(ai_response)

# Step 2: Dynamically process all requested tool calls
if ai_response.tool_calls:
    for tool_call in ai_response.tool_calls:
        tool_name = tool_call["name"]
        selected_tool = tools_by_name.get(tool_name)
        
        if selected_tool:
            # Execute the tool using parsed arguments
            output = selected_tool.invoke(tool_call["args"])
            
            # Append execution result back to history as a ToolMessage
            tool_message = ToolMessage(
                content=str(output),
                tool_call_id=tool_call["id"]
            )
            messages.append(tool_message)

    # Step 3: Get final response from model after tools have executed
    final_result = model_with_tools.invoke(messages)
    messages.append(final_result)
    print(f"Final Result:\n{final_result.content}\n")

print(f"Complete Chat History:")
for msg in messages:
    print(f"- {type(msg).__name__}: {msg.content}")