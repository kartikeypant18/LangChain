from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Example documents
documents = [
    "Hello, world!",
    "The capital of India is New Delhi.",
    "LangChain makes working with LLMs easier.",
    "Python is great for AI applications."
]

# Get embeddings for documents
doc_embeddings = embeddings.embed_documents(documents)

# Query
query = "What is the capital city of India?"
query_embedding = embeddings.embed_query(query)

# --- Compute cosine similarity ---
# sklearn expects 2D arrays, so reshape query
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find most similar document
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Print result
print("Most similar document:")
print(documents[index])
print("Similarity score:", score)
