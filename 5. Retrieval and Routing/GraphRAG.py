from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from dotenv import load_dotenv
import os

load_dotenv()

url = os.getenv("NEO4J_URI")
usr_name = os.getenv("NEO4J_USERNAME")
usr_pwd = os.getenv("NEO4J_PASSWORD")
db = os.getenv("NEO4J_DATABASE")

model =  ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.1
)

# text = """Elon Reeve Musk (born June 28, 1971) is a businessman and investor known for his key roles in space company SpaceX and automotive company Tesla, Inc. Other involvements include ownership of X Corp., formerly Twitter, and his role in the founding of The Boring Company, xAI, Neuralink and OpenAI.
# He is one of the wealthiest people in the world; as of July 2024, Forbes estimates his net worth to be
# US$221 billion.Musk was born in Pretoria to Maye and engineer Errol Musk, and briefly attended
# the University of Pretoria before immigrating to Canada at age 18, acquiring citizenship through
# his Canadian-born mother. Two years later, he matriculated at Queen's University at Kingston in Canada.
# Musk later transferred to the University of Pennsylvania and received bachelor's degrees in economics
# and physics. He moved to California in 1995 to attend Stanford University, but dropped out after
# two days and, with his brother Kimbal, co-founded online city guide software company Zip2."""

# docs = Document(page_content=text)

# graph_tansformer = LLMGraphTransformer(llm= model)

# graph_docs = graph_tansformer.convert_to_graph_documents(documents= [docs])
# print(f"\nNodes extracted: {len(graph_docs[0].nodes)}")
# print(f"\nRelationships extracted: {len(graph_docs[0].relationships)}")

### Instance of Neo4j
n4j_graph = Neo4jGraph(url= url,username= usr_name, password= usr_pwd, database=db)

### Neo$j Query To Load a CSV files along with some defined nodes and relationships
movie_query="""
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv' as row

MERGE(m:Movie{id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') |
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') |
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') |
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

n4j_graph.query(movie_query)
n4j_graph.refresh_schema()

print("Graph Created")

graphQAchain = GraphCypherQAChain.from_llm(llm = model, graph = n4j_graph, verbose=True,allow_dangerous_requests= True)

response = graphQAchain.invoke({"query":"who directed the movies Toy Story and who are the actors?"})

print(response)