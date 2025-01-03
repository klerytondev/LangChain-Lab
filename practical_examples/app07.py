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

classification_chain = (
    PromptTemplate.from_template(
        ''''
        Classifique a pergunta do usuário em um dos seguintes setores:
        - Financeiro
        - Suporte Técnico
        - Outras Informações

        Pergunta: {pergunta}
        '''
    )
    | model
    | parser
)

financial_chain = (
    PromptTemplate.from_template(
        ''''
        Você é um especialista financeiro.
        Sempre responda às perguntas começando com "Bem-vindo ao Setor Financeiro".
        Responda à pergunta do usuário:
        Pergunta: {pergunta}
        '''
    )
    | model
    | parser
)

tech_support_chain = (
    PromptTemplate.from_template(
        """
        Você é um especialista em suporte técnico.
        Sempre responda às perguntas começando com "Bem-vindo ao Suporte Técnico".
        Responda à pergunta do usuário:
        Pergunta: {pergunta}
        """
    )
    | model
    | parser 
)

other_info_chain = (
    PromptTemplate.from_template(
        """
        Você é um assistente de informações gerais.
        Sempre responda às perguntas começando com "Bem-vindo ao setor de Central de Informações".
        Responda à pergunta do usuário:
        Pergunta: {pergunta}
        """
    )
    | model
    | parser
)

def route(classification):
    classification = classification.lower()
    if 'financeiro' in classification:
        return financial_chain
    elif 'técnico' in classification:
        return tech_support_chain
    else:
        return other_info_chain


pergunta = input('Qual a sua pergunta?')

classification = classification_chain.invoke(
    {'pergunta': pergunta}
)

response_chain = route(classification=classification)

response = response_chain.invoke(
    {'pergunta': pergunta}
)
print(response)

