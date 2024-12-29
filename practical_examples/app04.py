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

model, _, client = initial_parameters()

# set_llm_cache(InMemoryCache())
set_llm_cache(
    SQLiteCache(
        database_path="langchain_cache.db"
    )
)

template = '''
    Traduza o texto do {language} para o {language2}:
    {content}
'''

prompt_template = PromptTemplate.from_template(
    template=template
    )

prompt = prompt_template.format(
    language='inglês',
    language2='português',
    content='Who was Alan Turing?'
)

response01 = client.invoke(prompt)

print(f'response01' + response01)
# print(f'response02' + response02)

