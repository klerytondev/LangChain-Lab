from langchain_openai import ChatOpenAI, OpenAI
import os
from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader

def initial_parameters() -> tuple:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return model, parser, client

model, parser, client = initial_parameters()

loader = TextLoader('data/base_conhecimento.txt')
documents = loader.load()

loader = PyPDFLoader('data/base_conhecimento.pdf')
documents = loader.load()

loader = CSVLoader('data/base_conhecimento.csv')
documents = loader.load()

prompt_base_conhecimento = PromptTemplate(
    input_variables=['contexto', 'pergunta'],
    template='''Use o seguinte contexto para responder à pergunta. 
    Responda apenas com base nas informações fornecidas.
    Não utilize informações externas ao contexto:
    Contexto: {contexto}
    Pergunta: {pergunta}'''
)

chain = prompt_base_conhecimento | model | StrOutputParser()

response = chain.invoke(
    {
        'contexto': '\n'.join(doc.page_content for doc in documents),
        'pergunta': 'Quantas marchas posseu um fiat marea?',
    }
)

print(response)


