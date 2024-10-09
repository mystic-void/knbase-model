import os

# Для ембеддинга и модели
from llm import LLM

# Для индексации документов в FAISS (векторная БД)
import pandas as pd
from pandas import DataFrame

from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# пример документов
documents1 = [
  {"id": 1, "question": "Как восстановить пароль?", "answer": "Для восстановления пароля перейдите по ссылке 'Забыли пароль?' на странице входа. Введите свой адрес электронной почты, и мы вышлем вам инструкции по восстановлению пароля.", "url": "https://example.com/confluence/recover-password"},
]
documents2 = [
  {"id": 2, "question": "Как связаться со службой поддержки?", "answer": "Вы можете связаться со службой поддержки, написав нам на электронную почту support@example.com или позвонив по телефону +1 (123) 456-7890.", "url": "https://example.com/confluence/contact-support"},
  {"id": 3, "question": "Как настроить двухфакторную аутентификацию?", "answer": "Для настройки двухфакторной аутентификации перейдите в раздел 'Настройки безопасности' вашего аккаунта и следуйте инструкциям.", "url": "https://example.com/confluence/2fa-setup"},
]

# todo создать интерфейс для загрузки документов (парсер например)

class VectorStore():
  """ Векторная база данных """
  
  def __init__(self, embedding_model, path=None, name="VectorStore") -> None:
    self.embedding_model = embedding_model
    self.is_loaded = False
    self.name = name
    self.load_storage(path)

  def load_storage(self, path):
      if path is not None and os.path.exists(path):
        self.store = FAISS.load_local(self.name, self.embedding_model)
        self.is_loaded = True
    
  def add(self, docs: DataFrame, index_col: str):
    loader = DataFrameLoader(docs, page_content_column=index_col)
    documents = loader.load()
    
    # Делим документ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = self.embedding_model # задаем векторайзер

    # Создаем хранилище, если оно не загружено
    if not self.is_loaded:
      self.store = FAISS.from_documents(texts, embeddings)
    else:
      self.store.add_documents(texts)

    # Обновляем
    self.store.save_local(self.name)
    # Подготовка к запросам
    self.store.as_retriever()
    

if __name__ == "__main__":
  df1 = pd.DataFrame(documents1)
  df2 = pd.DataFrame(documents2)
  
  model = LLM(model="llama3.2:3b", host="127.0.0.1", port=11434)
  db = VectorStore(embedding_model=model.get_embeding_model())
  
  
  # Добавление 1 части вопросов и поиск
  db.add(docs=df1, index_col="question")
  data = db.store.similarity_search_with_score('не знаю как прикрепить сотрудника')
  print(data)
  
  # Добавление 2 части вопросов и поиск
  db.add(docs=df2, index_col="question")
  
  # тестируем ретривер k=1 самый вероятный
  data = db.store.similarity_search_with_score('не знаю как прикрепить сотрудника')
  print(data)
