# https://habr.com/ru/articles/729664/

# --------------------------------------------
# Добавляем силу ChatGPT
# Мы убедились, что наш векторный поиск вытаскивает релевантные документы.
# Теперь наша задача передать их на вход LLM для получения более развернутого и человекоподобного ответа.
# В LangChain это можно сделать несколькими способами:
# 1 Построить цепочку load_qa_chain из ответов нашего ретривера.
# 2 Обратиться напрямую к векторной БД RetrievalQA.
# 3 Получить конкретный контекст и передать в цепочке LLM LLMChain.
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# создаем цепочку
qa_chain = RetrievalQA.from_chain_type(
llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
chain_type='stuff',
retriever=db.as_retriever()
)

query = 'не знаю как прикрепить сотрудника'

qa_chain.run(query)


# ИИ тут явно переврал, поэтому так:
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# создаем шаблон для промта
prompt_template = """Используй контекст для ответа на вопрос, пользуясь следующими правилами:

Не изменяй текст, который находится в кавычках.
В конце обязательно добавь ссылку на полный документ
{answer}
url: {url}
"""

PROMPT = PromptTemplate(
template=prompt_template, input_variables=['answer', 'url']
)

# цепочка с кастомным промтом
chain = LLMChain(
  llm=OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=500),
  prompt=PROMPT
)

relevants = db.similarity_search('не знаю как прикрепить сотрудника')
doc = relevants[0].dict()['metadata']

chain.run(doc)

# А что Alpaca?
from langchain import HuggingFaceHub

#подключение по API huggingface
alpaca_chain = LLMChain(
  prompt=PROMPT,
  llm=HuggingFaceHub(repo_id='IlyaGusev/fred_t5_ru_turbo_alpaca',
    huggingfacehub_api_token=YOUR_API_KEY,
    model_kwargs={'temperature':0, 'max_length':128}
    )
)

alpaca_chain.run(doc)
