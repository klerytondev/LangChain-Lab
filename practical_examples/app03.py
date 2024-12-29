from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
from langchain_community.cache import InMemoryCache, SQLiteCache
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_llm_cache

def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, _, client = initial_parameters()

# set_llm_cache(InMemoryCache())
set_llm_cache(
    SQLiteCache(
        database_path="langchain_cache.db"
    )
)

prompt = "Me diga quem foi Albert Einstein"

response01 = client.invoke(prompt)
response02 = client.invoke(prompt)

print(f'response01' + response01)
print(f'response02' + response02)

