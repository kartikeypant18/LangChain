from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("google_genai:gemini-2.5-flash-lite")

result = model.invoke("what is the capital of india")
print(result.content)