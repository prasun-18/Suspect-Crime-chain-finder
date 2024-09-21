**Note** Still updating the read_me file 

---

# Suspect-Crime-chain-finder
- This `Linux/Ubuntu-based` AI tool uses Nebula Graph DB to find suspect and criminal relationship chains. Raw data in `JSON` format is used and automated insertion is performed using Python for graph db. Here I am using `Mistral-7B-v2` LLM to generates embeddings for graph data, and also user queries are converted into embeddings and processed through cosine similarity to deliver the most likely results efficiently.


- **Illustration of the graph based on the dummy data base.**
![Graph](D:/Prasun/Git Projects/Suspect-Crime-chain-finder/Graph database.png)


- Here for our project we are using [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/main) model locally, or you can make end point connections.

---
## Mistral (LLM)
- **Mistral 7B** is a type of **LLM (Large Language Model)**, specifically one that has 7 billion parameters, developed by Mistral AI. It’s designed to be lightweight yet highly efficient in generating and understanding natural language, despite having fewer parameters compared to some larger models. 


- Check out the different [LLMs-Large Language Models](https://huggingface.co/)


let's break down the key characteristics of **Mistral/LLMs**:
 
### 1. **Architecture and Size**
   - **Mistral 7B** is smaller in size compared to models like **GPT-3 (175B parameters)** or **LLaMA 2 (70B parameters)**, yet it is still powerful enough to handle complex language tasks. 
   - LLMs like **Mistral 7B** are optimized for performance without sacrificing too much accuracy, making them faster and more cost-effective for real-time applications.

### 2. **Performance** 
   - Despite having fewer parameters (7 billion), **Mistral 7B** can perform a wide range of tasks including text generation, summarization, question answering, and more. This is a testament to the optimization strategies used to get the most out of its parameter set.
   - Other models, such as **GPT-4**, which has an even larger number of parameters, typically offer even higher accuracy and understanding of nuanced prompts due to their increased capacity.

### 3. **Training**  
   - LLMs like **Mistral 7B** are pre-trained on massive datasets to capture linguistic patterns and knowledge. They can then be fine-tuned on specific datasets to adapt them to specialized tasks.
   - Similar models include **Claude 2** by Anthropic, which is focused on aligning AI with human intentions, and **PaLM 2** by Google, optimized for multilingual tasks and reasoning.

### Other Examples of LLMs:
   - **GPT-4** (by OpenAI): One of the most well-known large language models with billions of parameters, offering state-of-the-art performance in natural language understanding and generation.
   - **LLaMA 2** (by Meta): A family of models ranging in size from 7B to 70B parameters, optimized for open research and various practical applications.
   - **Claude 2** (by Anthropic): A model designed with safety in mind, focusing on AI alignment and ethical interaction.
   - **PaLM 2** (by Google DeepMind): A multilingual language model with strengths in reasoning and multi-lingual understanding.

In summary, **Mistral 7B** serves as a great example of an LLM, where the focus is on balancing model size and efficiency while still delivering high performance across a range of language tasks. LLMs come in various sizes, such as **GPT-4** and **PaLM 2**, with larger models offering more accuracy but often at the cost of computational resources.

---

## Nebula Graph :- A Graph DB

**Nebula Graph** is an open-source, distributed, and scalable graph database designed to efficiently handle large-scale graphs with billions of vertices (nodes) and trillions of edges. It is well-suited for scenarios where relationships between entities are highly complex, such as social networks, recommendation systems, knowledge graphs, and network monitoring.

- [Installation of the nebula graph](https://docs.nebula-graph.io/3.3.0/4.deployment-and-installation/2.compile-and-install-nebula-graph/1.install-nebula-graph-by-compiling-the-source-code/), checkout for nebula graph setup.

### Key Features of Nebula Graph:

#### 1. High Scalability:
Nebula Graph is designed to handle extremely large datasets. It can scale horizontally, meaning you can increase capacity by adding more machines to the cluster, making it suitable for big data applications.

#### 2. High Performance:
Its architecture is optimized for fast data retrieval and complex queries. Nebula Graph supports millisecond-level query responses, even for complex traversals across large graphs.

#### 3. Distributed Architecture:
The database is distributed across multiple machines, which means it can offer fault tolerance and high availability. If one node fails, the system can still function properly.

#### 4. Flexible Schema:
Nebula Graph allows both schema and schema-less data storage. This flexibility allows you to define structures for your data, but it can also handle dynamic or unstructured datasets.

#### 5. Powerful Query Language (nGQL):
Nebula Graph comes with its own query language, called **nGQL**, which is similar to SQL and designed for efficient querying of graph data. It allows you to perform complex operations like traversals, pattern matching, and graph analytics.

#### 6. Rich Ecosystem:
Nebula Graph offers various tools for data visualization, integration with other big data platforms (like Apache Spark, Apache Flink), and client libraries in multiple programming languages (Python, Java, Go, etc.).

#### 7. Strong Consistency:
Nebula Graph guarantees strong consistency of data across the distributed system by using the Raft consensus algorithm, which ensures that data replication and synchronization are reliable and consistent.

#### 8. Graph Algorithms Support:
It supports common graph algorithms like shortest path, PageRank, and depth-first search (DFS), making it useful for various analytical and recommendation-based applications.

### Use Cases of Nebula Graph:

- **Social Networks**: Managing and analyzing relationships between people, posts, interactions, and recommendations.
- **Recommendation Engines**: Leveraging graph algorithms to offer personalized product or content recommendations.
- **Knowledge Graphs**: Organizing and querying vast amounts of structured and unstructured data with relationships.
- **Network Monitoring**: Visualizing and analyzing the structure and traffic patterns of large computer or communication networks.
- **Fraud Detection**: Detecting anomalies and suspicious connections within large financial or transaction datasets.

### Comparison with Other Graph Databases:

- **Nebula Graph vs Neo4j**: While Neo4j is a leading graph database, Nebula Graph is more focused on distributed systems and is better suited for large-scale deployments where horizontal scaling is needed.
  
- **Nebula Graph vs ArangoDB**: ArangoDB is a multi-model database, whereas Nebula Graph is specifically optimized for graph processing, making it more efficient for complex relationship data and traversals.

---
## How to convert words or letters into **Vector Embeddings**?

Calculating the **embedding for a word** involves representing that word as a vector of numbers, which encodes its meaning in such a way that similar words have similar vectors. Word embeddings are typically learned from large text corpora using machine learning models like **Word2Vec**, **GloVe**, or **BERT**.

Here’s a simplified step-by-step explanation of how word embeddings can be calculated using a method like **Word2Vec**:

### Steps to Calculate Word Embeddings

#### 1. **Training Data Preparation**
First, we need a large corpus of text (for example, Wikipedia articles, books, etc.).  
The corpus is preprocessed, which involves:
- Lowercasing all words.
- Removing stopwords (like "the," "is").
- Tokenizing sentences into words.

**Example Sentence**:  
"The cat sat on the mat."

After preprocessing, it becomes:  
**Preprocessed Sentence**:  
"cat sat mat"

#### 2. **Context Windows**
In Word2Vec (Skip-gram model), the goal is to predict the surrounding words (context) of a given word.  
A context window is a fixed number of words around a target word.

**Example**: For the word **"sat"**, with a window size of 2, the context words are **["cat", "mat"]**.

#### 3. **Training the Model (Word2Vec)**
The Word2Vec model learns embeddings by predicting context words from a target word.  
It creates a neural network that learns to associate a word with its surrounding context.

**Example**:  
For the word **"sat"**, the model tries to predict **"cat"** and **"mat"** as its context.  
As the model trains, it updates the word vectors (embeddings) for each word in the vocabulary based on how often they appear near each other.

#### 4. **Vector Representation**
After training, each word in the vocabulary is represented by a fixed-length vector.  
The word embedding captures semantic similarities. Words that frequently occur in similar contexts will have similar embeddings.

**Example**:  
Suppose the final word embeddings are 3-dimensional (in practice, they are usually much higher, like 100 or 300 dimensions). The vectors might look something like this:

```plaintext
"cat" = [0.4, 0.7, 0.5]
"sat" = [0.2, 0.8, 0.6]
"mat" = [0.3, 0.6, 0.5]
```

### 5. Using Word Embeddings
Once you have the word embeddings, you can use them for various tasks like:

- Finding similar words using **cosine similarity**.
- Performing clustering or classification tasks.
- Feeding them into other machine learning models (e.g., for sentiment analysis or translation).

**Example**:  
To find words similar to **"cat"**, you can calculate the cosine similarity between the embedding of **"cat"** and all other word embeddings. Words like **"dog"** or **"kitten"** might have vectors with high similarity to **"cat"**.

### Simple Example of Word Embedding Process
Let’s go through a tiny, easy-to-understand example:

- **Corpus**: `"cat sat mat"`
- **Vocabulary**: `["cat", "sat", "mat"]`

Suppose we train a Word2Vec model using this tiny corpus. The model learns the following embeddings:

- **"cat"** = [0.1, 0.3, 0.2]
- **"sat"** = [0.4, 0.5, 0.6]
- **"mat"** = [0.1, 0.3, 0.4]

### Summary of Process:
1. **Input data**: Preprocess a large corpus to prepare words and their contexts.
2. **Context Window**: Select a target word and its surrounding context words.
3. **Training**: Use a method like Word2Vec to train a model to predict context words, which updates word vectors.
4. **Output**: After training, each word has a vector (embedding) that represents its meaning based on its usage in the corpus.



---
## What is Cosine Similarity and how it is calculated? 
**Cosine Similarity** is a metric used to measure how similar two vectors are, by calculating the cosine of the angle between them. It is often used in text analysis, information retrieval, and machine learning for comparing documents, sentences, or any high-dimensional data. The value of cosine similarity ranges between `-1` (exactly opposite) and `1` (exactly the same), where `0` means the vectors are orthogonal (i.e., no similarity).

### Formula
The formula for cosine similarity between two vectors **A** and **B** is:

**`Cosine Similarity = (A · B) / (||A|| * ||B||)`**


Where:
- `A · B` is the **dot product** of vectors A and B.
- `||A||` and `||B||` are the **magnitudes** (or lengths) of the vectors.

## Steps for Calculation
Let's walk through the steps with a simple example.

### Example:
Given two vectors:  
**A = [1, 2, 3]**  
**B = [4, 5, 6]**

### Step 1: Compute the Dot Product (A · B)
The dot product is calculated as:

**`A · B = (1 * 4) + (2 * 5) + (3 * 6) = 4 + 10 + 18 = 32`**


### Step 2: Compute the Magnitude of Vector A (||A||)
The magnitude of a vector is calculated as the square root of the sum of the squares of its components:

**`||A|| = sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14) ≈ 3.74`**


### Step 3: Compute the Magnitude of Vector B (||B||)
Similarly, for vector B:

**`||B|| = sqrt(4^2 + 5^2 + 6^2) = sqrt(16 + 25 + 36) = sqrt(77) ≈ 8.77`**


### Step 4: Compute Cosine Similarity
Now, substitute the dot product and magnitudes into the cosine similarity formula:

**`Cosine Similarity = (A · B) / (||A|| * ||B||) = 32 / (3.74 * 8.77) = 32 / 32.82 ≈ 0.975`**



So, the cosine similarity between vectors **A** and **B** is approximately **0.975**, which indicates a high similarity.

### Key Points:
- **Dot Product** measures the similarity between the directions of the vectors.
- **Magnitude** normalizes the vectors, preventing their lengths from affecting the similarity score.
- **Cosine similarity** focuses on the angle between the vectors, ignoring their magnitude, making it useful for text similarity (e.g., document comparison).


#### Thanks for reading till the end!!
`XD`
