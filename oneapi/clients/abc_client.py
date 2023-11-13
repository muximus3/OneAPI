from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Self

class AbstractMethod(BaseModel):
    api_key: str
    api_base: str
    api_type: str

class AbstractClient(ABC):
    LIST_MSG_TEMP = """{% for message in messages %}{% if loop.first %}{% if message['role'] == 'user' %}{% if loop.length != 1 %}{{ '<s>Human:\n' + message['content'] }}{% else %}{{ '<s>Human:\n' + message['content'] + '\n\nAssistant:\n' }}{% endif %}{% elif message['role'] == 'system' %}{{ '<s>System:\n' + message['content'] }}{% endif %}{% elif message['role'] == 'user' %}{% if loop.last %}{{ '\n\nHuman:\n' + message['content'] + '\n\nAssistant:\n'}}{% else %}{{ '\n\nHuman:\n' + message['content']}}{% endif %}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant:\n' + message['content'] }}{% endif %}{% endfor %}"""

    def __init__(self, method: AbstractMethod) -> None:
        self.method = method
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: dict = None, config_file: str = "") -> Self:
        raise NotImplementedError


    @abstractmethod
    def format_prompt(self, prompt: str|list[str]|list[dict], system: str = ""):
        raise NotImplementedError

    @abstractmethod
    def chat(self, prompt: str | list[str] | list[dict], system: str = "", **kwargs):
        raise NotImplementedError
        
    @abstractmethod
    async def achat(self, prompt: str | list[str] | list[dict], system: str = "", max_new_tokens: int = 1024, **kwargs):
        raise NotImplementedError
    
    
    @abstractmethod
    def count_tokens(self, texts: List[str], model: str = "") -> int:
        raise NotImplementedError