# -*- coding: utf-8 -*-
import logging
import asyncio
import aiohttp
import time
from typing import List
from tqdm import tqdm
from oneapi.utils import load_json
from  oneapi import clients
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a"
)
class OneAPITool():
    LIST_MSG_TEMP_SYSTEM_USER_ASSISTANT = """{% for message in prompt %}{% if loop.first %}{% if message['role'] == 'user' %}{% if loop.length != 1 %}{{ '<s>Human:\n' + message['content'] }}{% else %}{{ '<s>Human:\n' + message['content'] + '\n\nAssistant:\n' }}{% endif %}{% elif message['role'] == 'system' %}{{ '<s>System:\n' + message['content'] }}{% endif %}{% elif message['role'] == 'user' %}{% if loop.last %}{{ '\n\nHuman:\n' + message['content'] + '\n\nAssistant:\n'}}{% else %}{{ '\n\nHuman:\n' + message['content']}}{% endif %}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant:\n' + message['content'] }}{% endif %}{% endfor %}"""
    STR_TEMP_SYSTEM_USER_ASSISTANT = """{% if system != '' %}{{'<s>System:\n'+system+'\n\nHuman\n'+prompt+'\n\nAssistant:\n'}}{% else %}{{'<s>Human:\n'+prompt+'\n\nAssistant:\n'}}{% endif %}"""

    def __init__(self, client: clients.AbstractClient) -> None:
        self.client = client

    @classmethod
    def from_config_file(cls, config_file):
        config = load_json(config_file)
        api_type = config.get("api_type")
        client_cls = clients.clients_register.get(api_type)
        client = client_cls.from_config(config)
        return cls(client)

    @classmethod
    def from_config(cls, api_key, api_base, api_type, api_version="2023-07-01-preview", chat_template="", **kwargs):
        client_cls = clients.clients_register.get(api_type)
        client = client_cls.from_config(dict(api_key=api_key, api_base=api_base, api_type=api_type, api_version=api_version, chat_template=chat_template) | kwargs)
        return cls(client)

    def format_prompt(self, prompt: str | list[str] | list[dict], system: str = ""):
        return self.client.format_prompt(prompt, system)

    def chat(self, prompt: str | list[str] | list[dict], system: str = "", **kwargs):
        response = self.client.chat(prompt , system, **kwargs)
        return response

    async def achat(self, prompt: str | list[str] | list[dict], system: str = "", **kwargs):
        response = await self.client.achat(prompt, system, **kwargs)
        return response

    def get_embeddings(self, texts: List[str], model="text-embedding-ada-002") -> List[List[float]]:
        return self.client.get_embeddings(texts, model)

    def get_embedding(self, text: str, model="text-embedding-ada-002") -> List[float]:
        return self.client.get_embedding(text, model)

    def count_tokens(self, texts: List[str], model: str = 'gpt-4') -> int:
        assert isinstance(
            texts, list), f"Input texts must be a list of strings. Got {type(texts)} instead."
        return self.client.count_tokens(texts, model)


async def bound_fetch(sem, pbar, tool: OneAPITool, prompt: str, model: str, **kwargs):
    async with sem:
        try:
            res = await tool.achat(prompt=prompt, model=model, **kwargs)
            pbar.update(1)
            return res
        except Exception as e:
            logger.error(f"Error when calling {model} api: {e}")
            return None


async def batch_chat(api_configs, texts, engines=None, request_interval=1, process_num=1, **kwargs):
    if isinstance(api_configs[0], str):
        tools = [OneAPITool.from_config_file(
            config_file) for config_file in api_configs]
    else:
        tools = [OneAPITool.from_config(api_key=config.get('api_key'), api_base=config.get('api_base'), api_type=config.get(
            'api_type'), api_version=config.get('api_version'), chat_template=config.get('chat_template')) for config in api_configs]
    process_num = max(len(api_configs), process_num)
    engines = engines if engines is not None else [''] * process_num
    if engines is not None and len(engines) == 1:
        engines = engines * process_num
    if engines is not None and len(engines) > 1:
        assert len(
            engines) == process_num, f'Number of engines must be equal to number of api config files when specific multiple engines, but got {len(engines)} engines and {process_num} api config files.'

    sem = asyncio.Semaphore(process_num)
    pbar = tqdm(total=len(texts))
    tasks = [asyncio.ensure_future(bound_fetch(sem, pbar, tools[i % process_num], prompt=prompt,
                                   model=engines[i % process_num], **kwargs))for i, prompt in enumerate(texts)]
    task_batches = [tasks[i:i+process_num]
                    for i in range(0, len(tasks), process_num)]
    results = []
    async with aiohttp.ClientSession() as session:
        for batch in task_batches:
            batch_result = await asyncio.gather(*batch)
            results.extend(batch_result)
            time.sleep(request_interval)
    return results
