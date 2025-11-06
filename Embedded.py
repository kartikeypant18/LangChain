from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
# Initialize the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# List of sentences/documents
texts = [
    "Hello, world!",
    "The capital of India is New Delhi.",
    "LangChain makes working with LLMs easier."
]

# Get embeddings for all of them
vectors = embeddings.embed_documents(texts)

# Print all embeddings as string
print(str(vectors))
