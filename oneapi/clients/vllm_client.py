from requests.exceptions import HTTPError
import json
from oneapi.clients.abc_client import AbstractConfig, AbstractClient
from oneapi.utils import compile_jinja_template
import requests
import aiohttp
from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Self
from urllib.parse import urljoin
import os
import sys
sys.path.append(os.path.normpath(
    f"{os.path.dirname(os.path.abspath(__file__))}/../.."))


class VLLMConfig(AbstractConfig):
    api_key: str = ""
    api_base: str
    api_type: str = "vllm"
    chat_template: str = "{{system + prompt}}"


class VLLMDecodingArguments(BaseModel):
    prompt: str
    stream: bool = False
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: Union[bool, str] = False
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False
    max_tokens: int = 16
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    skip_special_tokens: bool = True


class VLLMClient(AbstractClient):

    def __init__(self, 
                 config: AbstractConfig,
                 headers: Optional[Dict[str, str]] = None,
                 cookies: Optional[Dict[str, str]] = None,
                 timeout: int = 20
                 ) -> None:
        super().__init__(config)
        self.config = config
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: dict = None, config_file: str = "") -> Self:
        if isinstance(config_file, str) and os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        if not config:
            raise ValueError(
                "config is empty, pass a config file or a config dict")
        # add path to url
        if 'generate' not in config['api_base']:
            config['api_base'] = urljoin(config['api_base'], 'generate')
        if not config.get('chat_template'):
            config.pop('chat_template')
        return cls(VLLMConfig(**config))

    def format_prompt(self, prompt: str | list[str] | list[dict], system: str = "") -> str:
        template = compile_jinja_template(self.config.chat_template)
        return template.render(prompt=prompt, system=system)

    def chat_stream(self, response):
        for chunk in response.iter_lines(chunk_size=8192,
                                         decode_unicode=False,
                                         delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"][0]
                yield output

    def chat(self, prompt: str | list[str] | list[dict], system: str = "", max_new_tokens: int = 1024, **kwargs):
        args = VLLMDecodingArguments(prompt=self.format_prompt(
            prompt=prompt, system=system), max_tokens=max_new_tokens, **kwargs)
        if "verbose" in kwargs and kwargs["verbose"]:
            print(
                f"reqeusts args = {json.dumps(args.model_dump(), indent=4, ensure_ascii=False)}")
        response = requests.post(self.config.api_base,
                                 json=args.model_dump(), stream=args.stream)
        if args.stream:
            return self.chat_stream(response)
        else:
            return response.json()["text"][0]

    async def achat(self, prompt: str | list[str] | list[dict], system: str = "", max_new_tokens: int = 1024, **kwargs):
        args = VLLMDecodingArguments(prompt=self.format_prompt(
            prompt=prompt, system=system), max_tokens=max_new_tokens, **kwargs)
        if "verbose" in kwargs and kwargs["verbose"]:
            print(
                f"reqeusts args = {json.dumps(args.model_dump(), indent=4, ensure_ascii=False)}")
        async with aiohttp.ClientSession(headers=self.headers, cookies=self.cookies, timeout=aiohttp.ClientTimeout(self.timeout)) as session:
            async with session.post(self.config.api_base, json=args.model_dump()) as response:
                if args.stream:
                    texts = ""
                    async for line in response.content.iter_chunks():
                        data = json.loads(
                            line[0].decode('utf-8').rstrip('\x00'))
                        texts = data['text'][0]
                    return texts
                else:
                    return (await response.json())['text'][0]

    def count_tokens(self, texts: List[str], model: str = "") -> int:
        raise NotImplementedError
