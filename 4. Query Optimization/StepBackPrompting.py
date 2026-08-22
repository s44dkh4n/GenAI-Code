from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Template 1: Generate the Step-Back Question
step_back_prompt = ChatPromptTemplate.from_template(
    "You are an expert at high-level abstraction. Given this query, step back "
    "and ask a broader question about the underlying principles.\n\n"
    "Query: {user_query}\n"
    "Step-Back Query:"
)
generate_step_back = step_back_prompt | llm | StrOutputParser()

# Template 2: Synthesis
response_prompt = ChatPromptTemplate.from_template(
    "Answer the user's question using the context and high-level concept.\n\n"
    "Concept Question: {step_back_query}\n"
    "Retrieved Context:\n{context}\n\n"
    "Original Question: {user_query}\n"
    "Answer:"
)

# Chain assembly
chain = (
    {
        "user_query": RunnablePassthrough(),
        "step_back_query": generate_step_back,
    }
    | {
        "context": lambda x: retriever.invoke(x["user_query"]) + retriever.invoke(x["step_back_query"]),
        "user_query": lambda x: x["user_query"],
        "step_back_query": lambda x: x["step_back_query"]
    }
    | response_prompt
    | llm
)