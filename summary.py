from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

prompt = PromptTemplate(
    template=(
        "Summarize the following PDF content into key points and a concise paragraph:\n\n"
        "{text}"
    ),
    input_variables=["text"]
)

parser = StrOutputParser()

loader = PyPDFLoader("tac.pdf")
docs = loader.load()

full_text = "\n\n".join([doc.page_content for doc in docs])

chain = prompt | model | parser

result = chain.invoke({"text": full_text})

print("----- Full PDF Summary -----\n")
print(result)
