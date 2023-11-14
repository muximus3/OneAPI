from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Self

class AbstractConfig(BaseModel):
    api_key: str
    api_base: str
    api_type: str

class AbstractClient(ABC):

    def __init__(self, config: AbstractConfig) -> None:
        self.config = config
    
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