from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

load_dotenv()
model = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

set_llm_cache(InMemoryCache())

prompt = "Me diga quem foi Allan Turing"

response01 = model.invoke(prompt)
response02 = model.invoke(prompt)

print(f'response01' + response01)
print(f'response02' + response02)

