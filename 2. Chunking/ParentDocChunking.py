from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

docs = [Document(page_content= """Macroeconomics examines the behavior and decision-making of an economy as a whole, focusing heavily on indicators like inflation. Inflation represents the rate at which the general level of prices for goods and services rises, subsequently eroding consumer purchasing power. Economists generally categorize inflation into two primary drivers: demand-pull inflation, which occurs when aggregate demand outpaces aggregate supply, and cost-push inflation, driven by sharp increases in the costs of raw materials and wages.
Central banks, such as the Federal Reserve, use monetary policy tools to manage inflation and maintain economic stability. The primary tool is the manipulation of the benchmark interest rate, often referred to as the federal funds rate. When inflation runs above target limits, central banks execute contractionary monetary policy by raising interest rates. This increase makes borrowing more expensive for consumers and businesses, effectively cooling economic activity and slowing price growth.
Conversely, during economic downturns or deflationary periods, central banks pivot to expansionary monetary policy. By lowering interest rates, they incentivize borrowing, spending, and corporate investment to stimulate growth. When traditional interest rate cuts reach the zero lower bound, central banks may deploy unconventional tools like Quantitative Easing (QE). QE involves buying long-term government bonds to inject liquidity directly into the financial system.""", metadata = {"TITLE": "Macroeconomic Theory: Inflation Dynamics and Central Bank Monetary Policy"}), 
Document(page_content="""Quantum computing departs from classical computing by utilizing quantum mechanics to process complex informational states. The fundamental unit of quantum information is the qubit, which can exist in a state of superposition. Superposition allows qubits to represent both a 0 and a 1 simultaneously, exponentially increasing computational capacity. To exploit this state, quantum algorithms use quantum gates to manipulate probabilities and perform parallel calculations that would take classical supercomputers millennia to solve.
A major obstacle in scaling quantum hardware is quantum decoherence, where environmental noise destroys fragile quantum states. Things like temperature fluctuations and electromagnetic interference cause qubits to lose their quantum properties, resulting in calculation errors. To combat this, researchers are developing Fault-Tolerant Quantum Computing frameworks. These architectures isolate processors in dilution refrigerators, cooling the system to temperatures colder than deep space.
To achieve reliable computation, systems implement Quantum Error Correction (QEC) codes, such as the surface code. QEC works by entangling multiple physical qubits to form a single, highly stable logical qubit. Because information is distributed across the entangled network, the system can detect and correct individual physical qubit flips without measuring—and thus destroying—the underlying quantum state. This redundancy is essential for running complex algorithms like Shor's algorithm.""", metadata = {"TITLE ":"Fundamentals of Quantum Computing Architecture and Error Correction"}),

Document(page_content="""Enterprise cybersecurity requires a defense-in-depth architecture that spans multiple operational layers. The perimeter defense layer relies heavily on Next-Generation Firewalls (NGFW) and Automated Threat Detection systems. These platforms use state-of-the-art heuristic analysis to inspect incoming packet headers and payloads. By identifying anomalous traffic patterns that deviate from established network baselines, they can block malicious actors before they penetrate internal subnets.
Within the internal infrastructure, Identity and Access Management (IAM) systems enforce the principle of least privilege. IAM architectures utilize Role-Based Access Control (RBAC) and Multi-Factor Authentication (MFA) to ensure that users only access resources critical to their specific organizational roles. Additionally, privileged access management tools monitor administrative sessions in real-time, logging every command executed to prevent insider threats and credential exploitation.
When a breach occurs, the incident response protocol dictates an immediate containment strategy. Security Operations Center (SOC) analysts isolate affected endpoints from the broader corporate network using automated Endpoint Detection and Response (EDR) agents. Once containment is achieved, forensic investigators analyze system memory dumps and event logs to determine the initial entry vector. This digital forensics process helps organizations patch vulnerabilities and update their threat intelligence feeds to prevent future compromises.""", metadata = {"TITLE": "Enterprise Cybersecurity Architecture and Incident Response Framework"})
]

test_queries = ["What temperature environments are required to prevent quantum decoherence?", "How does the surface code protect logical qubits from flipping?", "How do Next-Generation Firewalls use heuristic analysis to isolate anomalous traffic?"]

small_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 30
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1100,
    chunk_overlap = 220
)

vec_store = Chroma(
    embedding_function= embeddings
)

store = InMemoryStore()

retriever = ParentDocumentRetriever(
    child_splitter= small_splitter,
    parent_splitter= None,
    docstore= store,
    vectorstore= vec_store
)

retriever.add_documents(docs)

for i,query in enumerate(test_queries):
    print(f"Query {i+1} : {query} \n Answer: {retriever.invoke(query)[0].page_content}\n {'=='*50}\n")