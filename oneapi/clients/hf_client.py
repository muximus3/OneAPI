from huggingface_hub import InferenceClient, AsyncInferenceClient
from typing import Optional, List, Self
from pydantic import BaseModel
import os
import json
import sys
from urllib.parse import urlparse
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../.."))
from oneapi.utils import compile_jinja_template
from oneapi.clients.abc_client import AbstractConfig, AbstractClient

class HuggingFaceConfig(AbstractConfig):
    api_key: str = ""
    api_base: str
    api_type: str = "huggingface"
    chat_template : str = "{{system + prompt}}"

class HuggingFaceDecodingArguments(BaseModel):
    prompt: str
    stream: bool = False
    do_sample: bool = False
    max_new_tokens: int = 1024
    best_of: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    seed: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

class HuggingfaceClient(AbstractClient):

    def __init__(self, config : HuggingFaceConfig) -> None:
        super().__init__(config)
        self.config = config
        self.huggingface_client = None
        self.async_huggingface_client = None
    
    @classmethod
    def from_config(cls, config: dict=None, config_file: str="") -> Self:
        if isinstance(config_file, str) and os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        if not config:
            raise ValueError("config is empty, pass a config file or a config dict")
        # remove path from url
        url_base = config.get("api_base", "")
        if url_base:
            parse_result = urlparse(url_base)
            if parse_result.scheme and parse_result.netloc:
                config["api_base"] = f"{parse_result.scheme}://{parse_result.netloc}"
        if not config.get('chat_template'):
            config.pop('chat_template')
        return cls(HuggingFaceConfig(**config))

    def format_prompt(self, prompt: str|list[str]|list[dict], system: str = "") -> str:
        template = compile_jinja_template(self.config.chat_template)
        return template.render(prompt=prompt, system=system)

    def chat_stream(self, resp):
        for data in resp:
            if data.details and data.details.finish_reason:
                break
            yield data.token.text

    def chat(self, prompt: str | list[str] | list[dict], system: str = "",  max_new_tokens: int = 1024, **kwargs):
        # OpenAI use 'stop'
        if 'stop' in kwargs and kwargs['stop']:
            kwargs['stop_sequences'] = kwargs.pop('stop')
        args = HuggingFaceDecodingArguments(prompt=self.format_prompt(prompt=prompt, system=system), max_new_tokens=max_new_tokens, **kwargs)
        if "verbose" in kwargs and kwargs["verbose"]:
            print(f"reqeusts args = {json.dumps(args.model_dump(), indent=4, ensure_ascii=False)}")
        if self.huggingface_client is None:
            self.huggingface_client = InferenceClient(self.config.api_base)
        resp = self.huggingface_client.text_generation(**args.model_dump(), details=True)
        if args.stream:
            return self.chat_stream(resp)
        else:
            return resp.generated_text
    
    async def achat(self, prompt: str | list[str] | list[dict], system: str = "",  max_new_tokens: int = 1024, **kwargs):
        # OpenAI use 'stop'
        if 'stop' in kwargs and kwargs['stop']:
            kwargs['stop_sequences'] = kwargs.pop('stop')
        args = HuggingFaceDecodingArguments(prompt=self.format_prompt(prompt=prompt, system=system), max_new_tokens=max_new_tokens, **kwargs)
        if "verbose" in kwargs and kwargs["verbose"]:
            print(f"reqeusts args = {json.dumps(args.model_dump(), indent=4, ensure_ascii=False)}")
        if self.async_huggingface_client is None:
            self.async_huggingface_client = AsyncInferenceClient(self.config.api_base)
        resp = await self.async_huggingface_client.text_generation(**args.model_dump(), details=True)
        if args.stream:
            full_comp = ""
            async for data in resp:
                if data.details and data.details.finish_reason:
                    break
                full_comp += data.token.text
            return full_comp
        else:
            return resp.generated_text

    def count_tokens(self, texts: List[str], model: str = "") -> int:
        raise NotImplementedError