#!pip install --upgrade pip
#!pip install py2neo
#!pip install langchain
#!pip install langchain_community
#!pip install langchain_experimental
#!pip install pyprojroot
#!pip install neo4j

from langchain_community.graphs import Neo4jGraph
import pandas as pd
# from neo4j.debug import watch
#from pyprojroot import here
# watch("neo4j")

import time 

#########################################################################
#       Neo4j instanced 
#########################################################################
print(f' ########################  neo4j initializing ######### ')
import os
import neo4j
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

embed_dim=384
embedding_dimension=384

from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
#########################################################################
#       Cleanup and start from scratch
#########################################################################
# Delete everything in a database
cypher = """
MATCH (n) DETACH DELETE n
"""
graph.query(cypher)

print(graph.schema)

# Match all nodes in the graph
cypher = """
  MATCH (n)
  RETURN count (n)
  """
result = graph.query(cypher)

graph.refresh_schema()
print(graph.schema)

#########################################################################
#   Import graph
#########################################################################

from langchain_community.graphs import Neo4jGraph
import pandas as pd

# Import a test movie database csv file only 20 rows for now 
df = pd.read_csv("https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv", nrows=20)
print(df.shape)
print(df.columns)


# the movie database has only movie details.  Will add 
# Description 
# Location movies were made 
# Similar movies 
# Generate some taglines using chatgot:

taglines = ["The adventure life of toys takes off!",
"Roll the dice and unleash the excitement!",
"Still Yelling. Still Fighting. Still Ready for Love.",
"Friends are the people who let you be yourself... and never let you forget it.",
"Just When His World Is Back To Normal... He's In For The Surprise Of His Life!",
"A Los Angeles crime saga",
"You are cordially invited to the most surprising merger of the year.",
"The Original Bad Boys.",
"Terror goes into overtime.",
"No limits. No fears. No substitutes.",
"Why can't the most powerful man in the world have the one thing he wants most?",
"Give blood...a whole new meaning.",
"Part Dog. Part Wolf. All Hero.",
"He had greatness within his grasp.",
"The Course Has Been Set. There Is No Turning Back. Prepare Your Weapons. Summon Your Courage. Discover the Adventure of a Lifetime!",
"No one stays at the top forever.",
"Lose your heart and come to your senses.",
"Twelve outrageous guests. Four scandalous requests. And one lone bellhop, in his first day on the job, who's in for the wildest New year's Eve of his life.",
"New animals. New adventures. Same hair.",
"Get on, or GET OUT THE WAY!"]

location = ["United States", "United States", "United States", "United States", "United States",
           "United States", "United States", "United States", "United States", "United Kingdom",
           "United States", "United States", "United States", "United States", "Malta",
           "United States", "United Kingdom", "United States", "United States", "United States"]

similar_movie = ["Finding Nemo", "Jumanji: Welcome to the Jungle", "The Bucket List", "The Best Man Holiday", "Cheaper by the Dozen",
                 "The Departed", "Notting Hill", "The Adventures of Huck Finn", "Die Hard", "Mission Impossible",
                 "Dave", "Dead and Loving It: Young Frankenstein", "Spirit: Stallion of the Cimarron", "JFK", "Pirates of the Caribbean: The Curse of the Black Pearl",
                 "Goodfellas", "Pride and Prejudice", "Pulp Fiction", "The Mask", "Speed"
                 ]

# Add this to df
df["similar_movie"] = similar_movie
df["tagline"]       = taglines
df["location"]      = location
df.to_csv('/home/saraghava/moviebot/sample_data/movie.csv')
print(f' ########################33 stage3 ')

import pathlib


movie_csv_path = pathlib.Path('/home/saraghava/moviebot/sample_data/movie.csv')
print(pd.read_csv(movie_csv_path).columns)
print("Data shape:", pd.read_csv(movie_csv_path).shape)

print(f' ########################33 stage5 ')



######################################################################################################
##the above pd frame needs to be copied into the correct path for neo4j to import
##### ORiginal query local pointer did not work

graph.query("""
LOAD CSV WITH HEADERS FROM  'file:///opt/neo4j/import/movie.csv'   // Load CSV data from a file specified by $movie_directory
AS row                                                      // Each row in the CSV will be represented as 'row'

MERGE (m:Movie {id:row.movieId})                            // Merge a Movie node with the id from the row
SET m.released = date(row.released),                        // Set the 'released' property of the Movie node to the date from the row
    m.title = row.title,                                    // Set the 'title' property of the Movie node to the title from the row
    m.tagline = row.tagline,                                // Set the 'tagline' property of the Movie node to the tagline from the row
    m.imdbRating = toFloat(row.imdbRating)                  // Convert the 'imdbRating' from string to float and set it as the property

FOREACH (director in split(row.director, '|') |             // For each director in the list of directors from the row (split by '|')
    MERGE (p:Person {name:trim(director)})                  // Merge a Person node with the director's name from the row, trimming any extra spaces
    MERGE (p)-[:DIRECTED]->(m))                             // Create a DIRECTED relationship from the director to the Movie

FOREACH (actor in split(row.actors, '|') |                  // For each actor in the list of actors from the row (split by '|')
    MERGE (p:Person {name:trim(actor)})                     // Merge a Person node with the actor's name from the row, trimming any extra spaces
    MERGE (p)-[:ACTED_IN]->(m))                             // Create an ACTED_IN relationship from the actor to the Movie

FOREACH (genre in split(row.genres, '|') |                  // For each genre in the list of genres from the row (split by '|')
    MERGE (g:Genre {name:trim(genre)})                      // Merge a Genre node with the genre's name from the row, trimming any extra spaces
    MERGE (m)-[:IN_GENRE]->(g))                             // Create an IN_GENRE relationship from the Movie to the Genre

MERGE (l:Location {name:trim(row.location)})
MERGE (m)-[:WAS_TAKEN_IN]->(l)

MERGE (s:SimilarMovie {name:trim(row.similar_movie)})
MERGE (m)-[:IS_SIMILAR_TO]->(s)
""",
params={"movie_directory": str( movie_csv_path )}   )         # Pass the parameter movie_directory which contains the path to the CSV file
print(f' #############   ALL INSERTED TO NEO4J   ###########33 stage3 ')

#########################################################
# Embeddings 
#########################################################
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

######################################################################
# For all taglines generate semantic meaning 
######################################################################
print(f' creating embeddings for all taglines')
print(f'{df["tagline"]}')
embedding_list = [embed_text(i)  for i in df["tagline"]]
df["taglineEmbedding"] = embedding_list 

print("Number of vectors:", len(embedding_list))
print("Embedding dimension:", len(embedding_list[0]))
#print(f'{embedding_list[0][:5]}')
#print(f'{embedding_list[19][:5]}')
#print(f'{df.head(5)}')
############################################

from langchain.vectorstores import Neo4jVector

# Delete the existing vector index
#Neo4jVector.delete_index("taglineEmbedding")
# Create a new vector index with the correct dimension
#Neo4jVector.create_new_index("taglineEmbedding", dimension=384)
#vec_size = graph.retrieve_existing_index()
#print(f'{vec_size}')
########################################
## Create vector index
########################################
graph.query("""
  CREATE VECTOR INDEX movie_tagline_embeddings IF NOT EXISTS      // Create a vector index named 'movie_tagline_embeddings' if it doesn't already exist  
  FOR (m:Movie) ON (m.taglineEmbedding)                           // Index the 'taglineEmbedding' property of Movie nodes 
  OPTIONS { indexConfig: {                                        // Set options for the index
    `vector.dimensions`: 384,                                    // Specify the dimensionality of the vector space (384 dimensions)
    `vector.similarity_function`: 'cosine'                        // Specify the similarity function to be cosine similarity
  }}"""
)

time.sleep(5)

graph.query("""
  SHOW VECTOR INDEXES     // Retrieves information about all vector indexes in the database
  """
)

print(f'### End Show vector index')

#####################################################################
#    Query and write to the neo4j database 
#####################################################################
for index, row in df.iterrows():
    movie_id = row['movieId']
    embedding = row['taglineEmbedding']
    graph.query(f"MATCH (m:Movie {{id: '{movie_id}'}}) SET m.taglineEmbedding = '{embedding}'")

graph.refresh_schema()
print(graph.schema)
##########################################################################################
