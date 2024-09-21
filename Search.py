import numpy as np
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from transformers import AutoTokenizer, AutoModel
import json
import torch

# Initialize Nebula Graph connection
def init_nebula_graph():
    config = Config()
    config.max_connection_pool_size = 10
    connection_pool = ConnectionPool()
    assert connection_pool.init([('127.0.0.1', 9669)], config)
    
    session = connection_pool.get_session('username', 'password')
    return session

def graph_db_connection(session):
       # Initialize Nebula Graph
    session = init_nebula_graph()
    graph_space_name = input("Enter the graph space name on which you want to perform query\n")
    #connection with graph space
    result = session.execute(f'USE {graph_space_name}')
    print(f"Result of USE query: {result}")
    if result is None:
        raise Exception(f"Failed to execute 'USE {graph_space_name}'.")
    elif not result.is_succeeded():
        raise Exception(f"Failed to execute 'USE {graph_space_name}': {result.error_msg()}")
    else:
        print("Successful connection to the graph")
    
    print("\nNote: To perform search your graph should be having embeddings stored under the tag's(criminal) vertices\n")

# Load Mistral 7B model for embeddings generation
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Mistral-7B")
    model = AutoModel.from_pretrained("Mistral-7B")
    return tokenizer, model

# Generate embeddings for the input query
def generate_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Calculate cosine similarity between two vecrtors
def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity_value = dot_product / (norm_vector1 * norm_vector2)
    return similarity_value

# Query the graph to retrieve vertices and their embeddings for the specific tag criminal
def retrieve_vertices_and_embeddings(session):
    query = 'MATCH (v) WHERE v.criminal.embeddings IS NOT NULL RETURN id(v), v.criminal.name, v.action.description, v.action.embeddings'
    result_set = session.execute(query)
    
    vertices = []
    for record in result_set.rows():
        vertex_id = record.values[0].get_sVal()
        name = record.values[1].get_sVal()
        description = record.values[2].get_sVal()
        embeddings = json.loads(record.values[3].get_sVal())  # Assuming embeddings are stored as JSON arrays
        
        vertices.append({
            'id': vertex_id,
            'name': name,
            'description': description,
            'embeddings': embeddings
        })
    return vertices

# Get all connected nodes for a given vertex
def get_connected_nodes(session, vertex_id):
    query = f'MATCH (v)-[e]-(n) WHERE id(v) == "{vertex_id}" RETURN e, n' #nGQL query
    result_set = session.execute(query)
    
    connections = []
    for record in result_set.rows():
        edge = record.values[0].get_sVal()
        connected_vertex = record.values[1].get_sVal()
        connections.append({
            'edge': edge,
            'connected_vertex': connected_vertex
        })
    return connections

# Main function to perform semantic search
def semantic_search(session, tokenizer, model, query):
    # Generate embedding for user query
    query_embedding = generate_embeddings(query, tokenizer, model)

    # Retrieve all vertices with embeddings
    vertices = retrieve_vertices_and_embeddings(session)

    # Calculate cosine similarity between query embedding and vertex embeddings
    similarities = []
    for vertex in vertices:
        sim = cosine_similarity(query_embedding, vertex['embeddings'])
        similarities.append((vertex, sim))
    
    # Sort vertices by similarity
    similarities.sort(key=lambda x: x[1], reverse=True) #Here lamda function will perform sorting based on the cosine similarity values in desecding order
    
    # Return the most similar vertex and all connected nodes
    top_vertex = similarities[0][0]
    print(f"Most similar vertex: {top_vertex['name']} (ID: {top_vertex['id']})")
    print(f"Description: {top_vertex['description']}")
    
    # Retrieve connected nodes
    connections = get_connected_nodes(session, top_vertex['id'])
    print("\nLinks that might give you clue of the case\n")
    print("\nConnected nodes/ connected chains from your target:")
    for connection in connections:
        print(f"Edge: {connection['edge']}, Connected Vertex: {connection['connected_vertex']}")

# Main function
def main():
    # Initialize Nebula Graph
    session = init_nebula_graph()
    
    # Load Mistral 7B model
    tokenizer, model = load_model()

    #taking user query
    while True:
        user_query = input("Enter your search query (type 'quit' or 'exit' or 'stop' to stop): ").lower()
        
        # Check if user wants to quit
        if user_query in ['quit', 'exit','stop']:
            print("Exiting the program...")
            break
    
    # Otherwise, continue with the execution
    print(f"You searched for: {user_query}")

    

    
    # Perform semantic search
    semantic_search(session, tokenizer, model, user_query)
    
    # Close the session
    session.release()

if __name__ == "__main__":
    main()
