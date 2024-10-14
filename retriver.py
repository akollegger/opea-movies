from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain.vectorstores.neo4j_vector import Neo4jVector
from llama_index.llms.ollama import Ollama
from ollama_functions import OllamaFunctions
from langchain.chat_models import ChatOpenAI
import os
from typing import List
import warnings
warnings.filterwarnings("ignore")


########################################################################
#   initialize neo4j
########################################################################
print(f' ########################neo4j insitialize ')
from langchain_community.graphs import Neo4jGraph
import neo4j
import os

NEO4J_URL = "neo4j://localhost:7687"
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "intel123"
NEO4J_DATABASE = "neo4j"
os.environ["NEO4J_URL"] = NEO4J_URL
os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
os.environ["NEO4J_DATABASE"] = NEO4J_DATABASE

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
print(graph.schema)

##graph.query("""
##  CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS  // Create a vector index named 'movie_tagline_embeddings' if it doesn't already exist
##  FOR (m:Movie) ON (m.taglineEmbedding)                       // Index the 'taglineEmbedding' property of Movie nodes
##  OPTIONS { indexConfig: {                                    // Set options for the index
##    `vector.dimensions`: 384,                                 // Specify the dimensionality of the vector space (384 dimensions)
##    `vector.similarity_function`: 'cosine'                    // Specify the similarity function to be cosine similarity
##  }}"""
##)
##
##graph.query("""
##  SHOW VECTOR INDEXES     // Retrieves information about all vector indexes in the database
##  """
##)
##
##cypher_query = """ MATCH (n) RETURN count(n) """
##graph.query(cypher_query)
##
########################################################################
#   initialize neo4j
########################################################################

q_one = "What was the cast of the Casino?"
q_two = "What are the most common genres for movies released in 1995?"
q_three = "What are the similar movies to the ones that Tom Hanks acted in?"

llm = ChatOllama(model="mistral", temperature=0)

chain = GraphCypherQAChain.from_llm(
#    llm, graph=graph, verbose=True,
    llm, graph=graph, 
    allow_dangerous_requests=True,
    )

print('################################################################################')
print(f'##################### graphcypherqachain    #################################')
print('################################################################################')

response = chain.invoke({"query": q_one})
print(response)
print("\nLLM response:", response["result"])

response = chain.invoke({"query": q_two})
print(response)
print("\nLLM response:", response["result"])

response = chain.invoke({"query": q_three})
print(response)
print("\nLLM response:", response["result"])


print('################################################################################')
print(f'##################### cypherchain    ##########################################')
print('################################################################################')

from sentence_transformers import SentenceTransformer, util
#model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
model = SentenceTransformer('all-MiniLM-L6-v2')

from typing import List

def embed_text(text:str)->List:
    """
    Embeds the given text using the specified model.
    Parameters:
        text (str): The text to be embedded.
    Returns:
        List: A list containing the embedding of the text.
    """
    response = model.encode(text)
    return response

print(f'stage1')

###############################################################################
question = "What movies are about Adventure?"
question_embedding = embed_text(question)

result = graph.query("""
    with $question_embedding as question_embedding
    CALL db.index.vector.queryNodes(
        'movie_tagline_embeddings', 
        $top_k, 
        question_embedding
        ) YIELD node AS movie, score
    RETURN movie.title, movie.tagline, score
    """,
    params={
        "question_embedding": question_embedding,
        "top_k": 5
    })

print(f'stage2')

prompt = f"# Question:\n{question}\n\n# Graph DB search results:\n{result}"
messages = [
    {"role": "system", "content": str(
        "You will be given the user question along with the search result of that question over a Neo4j graph database. summarize and provide the user a proper answer."
    )},
    {"role": "user", "content": prompt}
]

print(f'stage3')
#llm = ChatOllama(model="mistral", temperature=0)
response = client.chat.completions.create(
    model=llm,
    messages=messages
)

print(f'stage4')
print(response.choices[0].message.content)

print(' ############################################## THE END ##########################################')
