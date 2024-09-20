import os
import json
import time
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from transformers import AutoTokenizer, AutoModel

# Initialize Nebula Graph connection
def init_nebula_graph():
    config = Config()
    config.max_connection_pool_size = 10
    connection_pool = ConnectionPool()
    print("Starting the conection pool")
    assert connection_pool.init([('127.0.0.1', 9669)], config)
    print("Initializing the host configuration")
    
    session = connection_pool.get_session('username', 'password') #Change the username and password accordingly
    return session

# Create the necessary tags and edges in Nebula Graph to label the tags on the vertices.
# Note a vertex can be labeled with two tags
def create_tags_and_edges(session):
    session.execute('CREATE TAG IF NOT EXISTS location()')
    session.execute('CREATE TAG IF NOT EXISTS crime()')
    session.execute('CREATE TAG IF NOT EXISTS criminal(name string, description string, embeddings string)')
    session.execute('CREATE TAG IF NOT EXISTS contact(job string, description string)')

    session.execute('CREATE EDGE IF NOT EXISTS happend()')
    session.execute('CREATE EDGE IF NOT EXISTS responsible()')
    session.execute('CREATE EDGE IF NOT EXISTS in_contact_with(alive bool)')
    time.sleep(10)
    print("Buffer time to create Tags and Edges\n")

# Insert location and crime vertices along with happend edge between location and crime vertex.
def insert_location_crime(session, criminal_list):
    for location, crimes in criminal_list.items():
        # Insert location vertex
        session.execute(f'INSERT VERTEX location() VALUES "{location}":()')
        
        # Insert crime vertices and edges
        for crime in crimes:
            crime_vertex = crime.split('_')[0]  # Extract the first part
            session.execute(f'INSERT VERTEX crime() VALUES "{crime_vertex}":()')
            session.execute(f'INSERT EDGE happend() VALUES "{location}" -> "{crime_vertex}":()')

# Load the Mistral 7B model for embeddings
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Mistral-7B")
    model = AutoModel.from_pretrained("Mistral-7B")
    return tokenizer, model

# Generate embeddings for text using Mistral 7B
def generate_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    embeddings = model(**inputs).last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings.tolist()  # Convert to list for storing in Nebula

# Insert criminal vertices, connect them to crimes, and create in_contact_with vertices
def insert_criminal_and_contact(session, criminal_list, Background_Data_Folder, tokenizer, model):
    for location, crimes in criminal_list.items():
        for crime in crimes:
            # Open the corresponding Background_Data folder JSON file
            json_file = os.path.join(Background_Data_Folder, f"{crime}.json")
            if os.path.exists(json_file):
                with open(json_file) as f:
                    Background_Data_data = json.load(f)
                    # Read name, description from the Background_Data file
                    part_data = Background_Data_data.get("part", [])[0]
                    name = part_data.get("crimeName_criminal")
                    description = part_data.get("description")
                    
                    # Generate embeddings for the description
                    embeddings = generate_embeddings(description, tokenizer, model)
                    
                    # Insert criminal vertex
                    criminal_vertex = crime  # Using the crime and criminal name
                    session.execute(
                        f'INSERT VERTEX criminal(name, description, embeddings) '
                        f'VALUES "{criminal_vertex}":("{name}", "{description}", "{embeddings}")'
                    )
                    
                    # Insert responsible edge between tags crime vertices and criminal vertices
                    crime_vertex = crime.split('_')[0]
                    session.execute(f'INSERT EDGE responsible() VALUES "{crime_vertex}" -> "{criminal_vertex}":()')

                    # Insert contact vertices and in_contact_with edges between tag criminal vertices and tag contact verticies
                    bg_support = part_data["background"]["contacts"]
                    alive = part_data["background"].get("alive", [])
                    
                    for bg_support_name, bg_support_info in bg_support.items():
                        bg_support_job = bg_support_info.get("job", "")
                        bg_support_desc = bg_support_info.get("description", "")
                        
                        # Insert contact vertices along with the data inside it.
                        session.execute(
                            f'INSERT VERTEX contact(job, description) VALUES "{bg_support_name}":("{bg_support_job}", "{bg_support_desc}")'
                        )
                        
                        # Create in_contact_with edge between criminal and contact
                        is_alive = bg_support_name in alive
                        session.execute(f'INSERT EDGE in_contact_with(alive) VALUES "{criminal_vertex}" -> "{bg_support_name}":({is_alive})')

# Main function
def main():
    # Initialize Nebula Graph
    session = init_nebula_graph()
    graph_space_name = input("Enter the graph space name")
    #connection with graph space
    result = session.execute(f'USE {graph_space_name}')
    print(f"Result of USE query: {result}")
    if result is None:
        raise Exception(f"Failed to execute 'USE {graph_space_name}'.")
    elif not result.is_succeeded():
        raise Exception(f"Failed to execute 'USE {graph_space_name}': {result.error_msg()}")
    else:
        print("Successful connection to the graph")


    # Creating required tags and edges for the graph database 
    create_tags_and_edges(session)
    
    # Read criminal_list.json file
    with open('criminal_list.json') as f:
        criminal_list = json.load(f)
    print("\nCriminal_list file is loaded\n")

    # Insert vertices for the tags location and crime 
    insert_location_crime(session, criminal_list)
    print("\nAll the vertices under tags location and crime is executed\n")
    
    # Load Mistral 7B model from local machine.
    tokenizer, model = load_model()
    print("Mistral Model is loaded")

    # Insert vertices for the tags criminal and contact
    Background_Data_Folder = 'Background_Data'  # Folder containing Background_Data JSON files
    insert_criminal_and_contact(session, criminal_list, Background_Data_Folder, tokenizer, model)
    print("\nAll the vertices under tags criminal and contact is executed\n")

    # Close session
    session.release()
    print("\nSession is released....\n")

if __name__ == "__main__":
    main()
    print("\nCriminal data graph is created and ready for manual or automated query\n")
    
