#task was to create a RAG system that can answer questions based on the content of a single PDF document.
#approach involves loading a PDF file, splitting it into chunks, creating a vector store, and then using a retriever to fetch relevant chunks based on a user's question. Finally, it generates an answer using a language model.

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()

#using embedded model of gemini and chat model of gemini
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
#usinfg pdf loader to load the pdf file
loader = PyPDFLoader("tac.pdf")
docs = loader.load()

# Splitting the documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
print(f"✅ Total chunks created: {len(chunks)}")

# Creating a Chroma vector store from the document chunks
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Automatically persisted
)

#using retriever to get relevant documents from the vector store
retriever = db.as_retriever(search_kwargs={"k": 4})

# Defining the prompt template for question answering
prompt = PromptTemplate(
    template=(
        "You are a helpful assistant. "
        "Answer the question based only on the provided context.\n\n"
        "Context:\n{text}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["text", "question"]
)
#string output parser to get the final answer as string
parser = StrOutputParser()


question = "Summarize the key terms and conditions mentioned in this PDF."


docs = retriever.invoke(question)
context = "\n\n".join([d.page_content for d in docs])

#creating the final chain to get the answer
chain = prompt | llm | parser
answer = chain.invoke({"text": context, "question": question})


print("\nAnswer:\n")
print(answer)
