import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
parser = StrOutputParser()


folder_path = "./docs"
all_docs = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(folder_path, file_name))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_name
        all_docs.extend(docs)

print(f"✅ Loaded {len(all_docs)} pages from {len(os.listdir(folder_path))} documents")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(all_docs)
print(f"✅ Total chunks created: {len(chunks)}")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db_multi"
)

retriever = db.as_retriever(search_kwargs={"k": 4})

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

question = "What are the security measures?"
docs = retriever.invoke(question)
context = "\n\n".join([d.page_content for d in docs])

sources = set([d.metadata["source"] for d in docs])
print("📄 Answer retrieved from:", sources)

chain = prompt | llm | parser
answer = chain.invoke({"text": context, "question": question})

print("\n💬 Answer:\n", answer)
