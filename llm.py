from typing import List, Dict
from langchain_community.embeddings import OllamaEmbeddings
import requests
import json


class LLM():
    """ Класс-обертка для запуска локальных моделей """
    # Модели: llama3.2:3b или llama3.2:1b и mxbai-embed-large
    # todo добавить бд и индексацию (возможно)
    
    def __init__(self, host: str, port: int|str, model: str, stream: bool=False) -> None:
        self.model = model
        self.host = host
        self.port = port
        self.stream = stream
    
    def generate(self, promt: str):
        """ Ответ одним сообщением (ответ) """
        json_data={"model": self.model, "prompt": promt, "stream": self.stream}
        data = self.send(json_data, path="generate")
        return json.loads(data.text)["response"]
    
    def chat(self, messages: List[Dict]):
        """ Ответ на основе множества сообщений (чат) 
        - messages: пример сообщения [{"role": "user", "content": "question"}]
        """
        json_data={"model": self.model, "messages": messages, "stream": self.stream}
        data = self.send(json_data, path="chat")
        return json.loads(data.text)["message"]["content"]
    
    def get_embeding_model(self, model: str="mxbai-embed-large"):
        """ Ембеддинги текстов """
        return OllamaEmbeddings(model=model, base_url=f"http://{self.host}:{self.port}")
    
    def send(self, json, path):
        """ Метод отправки данных в модель """
        return requests.post(f"http://{self.host}:{self.port}/api/{path}", json=json)


if __name__ == "__main__":
    model = LLM(model="llama3.2:3b", host="127.0.0.1", port=11434)
    
    generate_ans = model.generate("Hello")
    print(f"{generate_ans}\n")

    messages = [{"role": "user", "content": "Remember a 25"}, {"role": "user", "content": "What is number you remember?"}]
    chat_ans = model.chat(messages)
    print(f"{chat_ans}\n")
    
    embed_ans = model.embed("Hello")
    print(f"Hello -> {embed_ans}\n")
    