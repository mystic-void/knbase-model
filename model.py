# https://habr.com/ru/articles/729664/
import ollama # для ембеддинга и для модели

response = ollama.chat(model='llama3.1', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response['message']['content'])
# ollama.embeddings(model='nomic-embed-text', prompt='The sky is blue because of rayleigh scattering')

# пример доков
# Сгенерируй набор из 10 примеров документов, которые далее будут добавлены в векторное хранилище. В документах содержится информаци о запросах в службу поддержки компании. Каждый документ должен иметь следующую структуру:
# 1. id: int (уникальный идентификатор документа)
# 2. question: str(вопрос пользователя)
# 3. answer: str(ответ службы поддержки)
# 4. url: str (ссылка на источник информации, предпочтительно страницу confluence)
# Обрати внимание, что тексты документов должны быть разнообразными и содержательными. Результат должен быть реализован в виде python кода
documents = [
  {"id": 1, "question": "Как восстановить пароль?", "answer": "Для восстановления пароля перейдите по ссылке 'Забыли пароль?' на странице входа. Введите свой адрес электронной почты, и мы вышлем вам инструкции по восстановлению пароля.", "url": "https://example.com/confluence/recover-password"},
  {"id": 2, "question": "Как связаться со службой поддержки?", "answer": "Вы можете связаться со службой поддержки, написав нам на электронную почту support@example.com или позвонив по телефону +1 (123) 456-7890.", "url": "https://example.com/confluence/contact-support"},
  {"id": 3, "question": "Как настроить двухфакторную аутентификацию?", "answer": "Для настройки двухфакторной аутентификации перейдите в раздел 'Настройки безопасности' вашего аккаунта и следуйте инструкциям.", "url": "https://example.com/confluence/2fa-setup"},
]


# --------------------------------------------
# Индексируем документы в FAISS
import pandas as pd # если нет импорта пандаса, то остальной код не имеет смысла

from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings # вместо этого оллама будет
from langchain.vectorstores import FAISS

# создаем из наших документов датафрейм
df = pd.DataFrame(documents)
df.head()


# грузим фрейм в лоадер, выделив колонку для векторизации (здесь может быть место для дискуссий)
# что за датафреймлоадер
loader = DataFrameLoader(df, page_content_column='question')
documents = loader.load()

# создаем сплиттер документов, чтобы уложиться в лимит по токенам, в нашем случае это не очень полезный шаг
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# задаем векторайзер
embeddings = OpenAIEmbeddings(openai_api_key=YOUR_API_KEY)

# создаем хранилище
db = FAISS.from_documents(texts, embeddings)
db.as_retriever()

# также можно сохранить хранилище локально
db.save_local('faiss_index')

# тестируем ретривер
db.similarity_search_with_score('не знаю как прикрепить сотрудника')


# --------------------------------------------
# Добавляем силу ChatGPT
# Мы убедились, что наш векторный поиск вытаскивает релевантные документы. Теперь наша задача передать их на вход LLM для получения более развернутого и человекоподобного ответа. В LangChain это можно сделать несколькими способами:
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
prompt=PROMPT)

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

# Использовать модели по API может быть не очень оптимально. В LangChain вы можете припарковать их на собственный хост через функционал SelfHostedPipeline и SelfHostedHuggingFaceLLM.