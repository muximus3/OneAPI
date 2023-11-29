from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, List, Self, Optional

class AbstractConfig(BaseModel):
    api_key: Optional[str] = ""
    api_base: Optional[str] = ""
    api_type: Optional[str] = ""

class AbstractClient(ABC):

    def __init__(self, config: AbstractConfig) -> None:
        self.config = config
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.chat(*args, **kwds)
    
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
    async def achat(self, prompt: str | list[str] | list[dict], system: str = "", max_tokens: int = 1024, **kwargs):
        raise NotImplementedError
    
    
    @abstractmethod
    def count_tokens(self, texts: List[str], model: str = "") -> int:
        raise NotImplementedError