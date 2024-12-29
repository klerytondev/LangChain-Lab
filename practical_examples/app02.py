from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = ChatOpenAI(
    model='gpt-3.5-turbo',
)

messages = [
    {
        "role": "system",
        "content": "Você é um assitente que fornece informações sobre figuras publicas e eventos históricos."
    },
    {
        "role": "user",
        "content": "Quem foi Alan Turing?"
    }
]
response = response.invoke(messages)
# print(response)
print(response.content)
