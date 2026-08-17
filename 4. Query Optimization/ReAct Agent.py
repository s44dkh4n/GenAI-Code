from dotenv import load_dotenv
import json
import re

from langchain_mistralai import ChatMistralAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

# Import the dedicated tool class alongside the wrapper
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.tools import OpenWeatherMapQueryRun

load_dotenv()

react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (e.g., 28, 20)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_prompt_template)

# Initialize the tool directly
weather_wrapper = OpenWeatherMapAPIWrapper()
weather = OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)


@tool
def product(tool_input: str) -> float:
    """Returns the product of two numbers. Input should be a comma-separated pair like '28, 20' or JSON like '{{"a": 28, "b": 20}}'."""
    cleaned_input = str(tool_input).replace('\nObserv', '').replace('\nObservation:', '')
    cleaned_input = cleaned_input.strip('\'" \t\n')

    try:
        data = json.loads(cleaned_input)
        if isinstance(data, dict):
            return float(data.get("a", 0)) * float(data.get("b", 0))
    except Exception:
        pass

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", cleaned_input)
    if len(numbers) >= 2:
        return float(numbers[0]) * float(numbers[1])
    
    raise ValueError(f"Could not parse two numbers from input: '{tool_input}'")


tools = [weather, product]

model = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.3
)

agent = create_react_agent(
    llm=model,
    prompt=prompt,
    tools=tools
)

agent_exe = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

response = agent_exe.invoke({
    "input": "What is today weather in Nowshera, KPK? multiply the temperature value by 20"
})

print("\nFinal Result:")
print(response["output"])