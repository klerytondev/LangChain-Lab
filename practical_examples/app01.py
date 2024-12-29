from langchain_openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = model.invoke(
    input='Quem foi Alan Truing?',
    temperature=1,
    max_tokens=100,

)
print(response)