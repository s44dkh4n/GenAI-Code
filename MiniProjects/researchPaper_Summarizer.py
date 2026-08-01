from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Loading the Environment Variable
load_dotenv()

# Selecting the ChatModel from Hugging Face Available Models
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7 
)

model = ChatHuggingFace(llm = llm)

# Streamlit Code to Create a Dropdown Menu Like GUI
st.header("Research Paper Summarizer")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"])

# Detailed Prompt for the Model 
prompt = PromptTemplate(

    template=
    """
    Please summarize the research paper titled "{paper_name}" with the following specifications
    Explanation style "{explain_type}"
    Explanation length "{explain_length}"

    1. Mathematical Details: (if the the required style is mathematical)
    Include relevant mathematical equations if present in the paper.
    Explain the mathematical concepts using simple, intuitive code snippets where applicable.
                
    2. Analogies:
    Use relatable analogies to simplify complex ideas.
    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing

    Ensure the summary is clear, accurate and alinged with the provided length and style
    """,

    input_variables=["paper_name","explain_type","explain_length"]  
)

# When use click "Summarize" Button it will show results
if st.button("Summarize"):
    chain = prompt | model
    with st.spinner("Analyzing and summarizing paper..."):
        result = chain.invoke({
        "paper_name": paper_input,
        "explain_type":style_input,
        "explain_length":length_input
    })
        st.write(result.content)
