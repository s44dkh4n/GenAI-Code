from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.3,
    max_new_tokens=500
)

model = ChatHuggingFace(llm=llm)

class LeadAnalysis(BaseModel):

    company_name: str = Field(
        description="Name of the company"
    )

    industry: str = Field(
        description="Industry of the company"
    )

    pain_points: list[str] = Field(
        description="Possible business pain points"
    )

    growth_signals: list[str] = Field(
        description="Indicators that the company is growing"
    )

    qualification: Literal[
        "High Fit",
        "Medium Fit",
        "Low Fit"
    ] = Field(
        description="Lead qualification score"
    )

    personalization_points: list[str] = Field(
        description="Specific details useful for personalization"
    )

analysis_parser = PydanticOutputParser(
    pydantic_object=LeadAnalysis
)

email_parser = StrOutputParser()

analysis_prompt = PromptTemplate(
    template="""
You are an expert AI Sales Development Representative.

Analyze the provided lead information carefully.

Tasks:
- identify company name
- identify industry
- identify business pain points
- identify growth indicators
- determine qualification score
- extract personalization opportunities

Rules:
- do not hallucinate
- use only the provided information
- think like an experienced SDR

{format_instructions}

Lead Data:
{lead_data}
""",

    input_variables=["lead_data"],

    partial_variables={
        "format_instructions":
        analysis_parser.get_format_instructions()
    }
)

email_prompt = PromptTemplate(
    template="""
You are an expert B2B cold email copywriter.

Generate a highly personalized outreach email.

Lead Analysis:
{analysis}

Product Description:
{product_description}

Rules:
- keep email under 150 words
- sound human-written
- avoid generic sales language
- mention personalization points naturally
- focus on business value
- include a natural CTA
- avoid buzzwords
- do not exaggerate

Return only the email.
""",

    input_variables=[
        "analysis",
        "product_description"
    ]
)

analysis_chain = (
    analysis_prompt
    | model
    | analysis_parser
)

email_chain = (
    email_prompt
    | model
    | email_parser
)

lead_data = """
TechFlow is a SaaS startup helping sales teams automate CRM workflows.

The company recently expanded operations into Europe and is aggressively hiring SDRs and account executives.

Their LinkedIn posts frequently discuss sales productivity, pipeline management, and lead generation challenges.

The company currently uses multiple disconnected tools for outreach and CRM management.
"""

product_description = """
An AI-powered sales automation platform that automates lead enrichment,
personalized outreach, and CRM workflow management.
"""

try:

    analysis_result = analysis_chain.invoke({
        "lead_data": lead_data
    })

    print("\n========================")
    print("LEAD ANALYSIS")
    print("========================\n")

    print(analysis_result)
    
# Step 2 -> Generate Email
    email_result = email_chain.invoke({

        "analysis": analysis_result,

        "product_description":
        product_description
    })

    print("\n========================")
    print("PERSONALIZED EMAIL")
    print("========================\n")

    print(email_result)

except Exception as e:

    print("\nParsing / Chain Error:\n")
    print(e)

