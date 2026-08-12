import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Schema for lead analysis output
class LeadAnalysis(BaseModel):
    company_name: str = Field(description="Name of the company")
    industry: str = Field(description="Industry of the company")
    pain_points: list[str] = Field(description="Possible business pain points")
    growth_signals: list[str] = Field(description="Indicators that the company is growing")
    qualification: Literal["High Fit", "Medium Fit", "Low Fit"] = Field(description="Lead qualification score")
    personalization_points: list[str] = Field(description="Specific details useful for personalization")


class ColdEmailEngine:
    def __init__(self):
        # HuggingFace LLM configuration
        # temperature: Low values lower randomness for structured responses
        # max_new_tokens: Token generation output limit
        self.llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation",
            temperature=0.3,
            max_new_tokens=500
        )
        self.model = ChatHuggingFace(llm=self.llm)
        self.vector_store = None

        self.analysis_parser = PydanticOutputParser(pydantic_object=LeadAnalysis)
        self.email_parser = StrOutputParser()

        self.analysis_prompt = PromptTemplate(
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
            partial_variables={"format_instructions": self.analysis_parser.get_format_instructions()}
        )

        self.email_prompt = PromptTemplate(
            template="""
You are an expert B2B cold email copywriter.

Generate a highly personalized outreach email.

Lead Analysis:
{analysis}

Product / Context Description:
{product_description}

Rules:
1. Candidate Perspective: NEVER position the email as a software vendor selling a product (e.g., "our e-commerce engine can help you"). Always write from the perspective of an engineer demonstrating relevant project experience.
2. Directness: Skip all generic fluff, compliments, and greetings like "I hope this email finds you well" or "I was intrigued by your growth."
3. Technical Relevance: Directly link 2-3 specific technical features or technologies from the candidate's projects to key requirements in the JD.
4. Tone & Style: Direct, concise, candid, and professional. Use single hyphens (-) for bullet points. No emojis.
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
            input_variables=["analysis", "product_description"]
        )

        self.analysis_chain = self.analysis_prompt | self.model | self.analysis_parser
        self.email_chain = self.email_prompt | self.model | self.email_parser

    # Extract clean text from job link
    def scrape_url(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=" ")
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return " ".join(chunk for chunk in chunks if chunk)

    # Ingest document and build Vector store
    def create_vector_store(self, file_path: str):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            texts = [doc.page_content for doc in docs]
        elif file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            texts = df.astype(str).agg(" ".join, axis=1).tolist()
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
            texts = df.astype(str).agg(" ".join, axis=1).tolist()
        else:
            raise ValueError("Unsupported file format")

        self.vector_store = Chroma.from_texts(texts, embeddings)

    # Query local VectorDB context
    def get_context(self, query: str) -> str:
        if not self.vector_store:
            return ""
        docs = self.vector_store.similarity_search(query, k=4)
        return "\n".join([doc.page_content for doc in docs])

    # Core generation chain execution
    def generate_email(self, lead_data: str) -> str:
        analysis_result = self.analysis_chain.invoke({"lead_data": lead_data})
        context = self.get_context(lead_data)
        product_desc = context if context else "Standard AI automation and software services."

        # Potential failure mode: API rate limits or timeout errors on long context payloads
        email_result = self.email_chain.invoke({
            "analysis": analysis_result,
            "product_description": product_desc
        })
        return email_result

    # Dispatch email using environment variables or basic SMTP settings
    def send_email(self, recipient_email: str, subject: str, body: str):
        sender_email = os.getenv("SENDER_EMAIL")
        sender_password = os.getenv("SENDER_PASSWORD")
        
        if not sender_email or not sender_password:
            raise ValueError("SMTP credentials missing in .env file (SENDER_EMAIL, SENDER_PASSWORD).")

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()