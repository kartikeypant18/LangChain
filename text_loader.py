from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Initialize Gemini model
model = init_chat_model("google_genai:gemini-2.5-flash-lite")

# Define prompt template
prompt = PromptTemplate(
    template="Summarize the following text:\n\n{{text}}",
    input_variables=["text"]
)

# Define parser to extract string output
parser = StrOutputParser()

# Load text file
loader = PyPDFLoader("tac.pdf")
docs = loader.load()

# Combine components into a single chain (Prompt → Model → Output)
chain = prompt | model | parser

# Run the chain with the text content
result = chain.invoke({"text": docs[0].page_content})

print(f"Total pages: {len(docs)}\n")

# Loop through and print content of each page
for i, doc in enumerate(docs):
    print(f"----- Page {i + 1} -----")
    print(doc.page_content)
    print("\n")