from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
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

chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="Você deve responder baseado em dados geograficos de regiões do Brasil."),
        HumanMessagePromptTemplate.from_template('Por favor me fale sobre a regiã0 {region} do Brasil.'),
        AIMessage(content="Claro, vou começar a responder, com base em dados geograficos sobre a região {region}."),
        HumanMessage(content='Certifique-se de incluir dados sobre a população, clima e vegetação.'),
        AIMessage(content='Aqui estão as respostas baseadas em dados geograficos sobre a região {region}.'),
    ]
)
prompt = chat_template.format_messages(region='Nordeste')
response01 = client.invoke(prompt)
print(f'response01' + response01)
