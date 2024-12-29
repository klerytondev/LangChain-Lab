from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, parser, client = initial_parameters()

set_llm_cache(
    SQLiteCache(
        database_path="langchain_cache.db"
    )
)

template='''
    Me fale sobre o carro {carro}.
    '''

prompt_template = PromptTemplate.from_template(
    template=template
    )

runnable_sequence = prompt_template | model | parser

response = runnable_sequence.invoke({'carro': 'Fusca'})

print(response)
