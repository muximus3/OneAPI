from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, List, Optional, Union
import requests
import numpy as np
from tqdm import tqdm


class AbstractConfig(BaseModel):
    api_key: Optional[str] = ""
    api_base: Optional[str] = ""
    api_type: Optional[str] = ""


class AbstractClient(ABC):
    DEFAULT_LIST_MSG_TEMP_SYSTEM_USER_ASSISTANT = """{% for message in prompt %}{% if loop.first %}{% if message['role'] == 'user' %}{% if loop.length != 1 %}{{ '<s>Human:\n' + message['content'] }}{% else %}{{ '<s>Human:\n' + message['content'] + '\n\nAssistant:\n' }}{% endif %}{% elif message['role'] == 'system' %}{{ '<s>System:\n' + message['content'] }}{% endif %}{% elif message['role'] == 'user' %}{% if loop.last %}{{ '\n\nHuman:\n' + message['content'] + '\n\nAssistant:\n'}}{% else %}{{ '\n\nHuman:\n' + message['content']}}{% endif %}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant:\n' + message['content'] }}{% endif %}{% endfor %}"""
    DEFAULT_STR_TEMP_SYSTEM_USER_ASSISTANT = """{% if system != '' %}{{'<s>System:\n'+system+'\n\nHuman\n'+prompt+'\n\nAssistant:\n'}}{% else %}{{'<s>Human:\n'+prompt+'\n\nAssistant:\n'}}{% endif %}"""

    def __init__(self, config: AbstractConfig) -> None:
        self.config = config

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.chat(*args, **kwds)

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict = None, config_file: str = ""):
        raise NotImplementedError

    @abstractmethod
    def format_prompt(
        self, prompt: Union[str, list[str], list[dict]], system: str = ""
    ):
        raise NotImplementedError

    @abstractmethod
    def chat(
        self, prompt: Union[str, list[str], list[dict]], system: str = "", **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    async def achat(
        self,
        prompt: Union[str, list[str], list[dict]],
        system: str = "",
        max_tokens: int = 1024,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def count_tokens(self, texts: List[str], model: str = "") -> int:
        raise NotImplementedError

    def get_embeddings(
        self, texts: List[str], max_sequence_length=512, max_batch_size=32
    ) -> List[List[float]]:
        def get_batch_embeddings(batch_texts):
            if max_sequence_length > 0:
                batch_texts = [s[:max_sequence_length].replace("\n", " ") for s in batch_texts]
            res = requests.post(
                self.config.api_base,
                json={"inputs": batch_texts},
                headers={"Content-Type": "application/json"},
            )
            if res.status_code != 200:
                raise Exception(f"Error when calling embedding api: {res.text}")
            return res.json()

        if len(texts) <= max_batch_size:
            return get_batch_embeddings(texts)
        embeddings = []
        for i in tqdm(range(0, len(texts), max_batch_size), desc="Embedding"):
            batch_text = texts[i : i + max_batch_size]
            embeddings.append(get_batch_embeddings(batch_text))
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
        return embeddings

    def get_embedding(self, text: str, max_sequence_length=512) -> List[float]:
        return self.get_embeddings([text], max_sequence_length)[0]
