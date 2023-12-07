# -*- coding: utf-8 -*-
import logging
import asyncio
import time
from typing import Any, List, Union
from tqdm import tqdm
from oneapi.utils import load_json
from oneapi import clients
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a",
)


class OneAPITool:
    def __init__(self, client: clients.AbstractClient) -> None:
        self.client = client

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.client(*args, **kwds)

    @classmethod
    def from_config(
        cls,
        config_file: str = "",
        api_key: str = "",
        api_base: str = "",
        api_type: str = "",
        api_version: str = "2023-07-01-preview",
        chat_template: str = "",
        **kwargs,
    ):
        if config_file and os.path.isfile(config_file):
            config = load_json(config_file)
            api_type = config.get("api_type")
            client_cls = clients.clients_register.get(api_type)
        else:
            client_cls = clients.clients_register.get(api_type)
            if not client_cls:
                raise ValueError(
                    f"Unknown api type: {api_type}, please choose from {clients.clients_register.keys()}"
                )
            config = dict(
                api_key=api_key,
                api_base=api_base,
                api_type=api_type,
                api_version=api_version,
                chat_template=chat_template,
            )
            config = {k: v for k, v in config.items() if v}
        client = client_cls.from_config(config | kwargs)
        return cls(client)

    def format_prompt(
        self, prompt: Union[str, list[str], list[dict]], system: str = ""
    ):
        return self.client.format_prompt(prompt, system)

    def chat(
        self,
        prompt: Union[str, list[str], list[dict]],
        system: str = "",
        max_tokens: int = 512,
        stop: List[str] = None,
        **kwargs,
    ):
        # dont use empty value to overwrite default values
        # kwargs = {k: v for k, v in kwargs.items() if v}
        if stop:
            kwargs["stop"] = stop
        response = self.client.chat(prompt, system, max_tokens=max_tokens, **kwargs)
        return response

    async def achat(
        self, prompt: Union[str, list[str], list[dict]], system: str = "", **kwargs
    ):
        # dont overwrite default values
        # kwargs = {k: v for k, v in kwargs.items() if v}
        response = await self.client.achat(prompt, system, **kwargs)
        return response

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.client.get_embeddings(texts)

    def get_embedding(self, text: str) -> List[float]:
        return self.client.get_embedding(text)

    def count_tokens(self, texts: List[str], model: str = "gpt-4") -> int:
        assert isinstance(
            texts, list
        ), f"Input texts must be a list of strings. Got {type(texts)} instead."
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


async def batch_chat(
    api_configs, texts, engines=None, request_interval=0.1, max_process_num=1, **kwargs
):
    if isinstance(api_configs[0], str):
        tools = [OneAPITool.from_config(config_file) for config_file in api_configs]
    else:
        tools = [OneAPITool.from_config(**config) for config in api_configs]
    max_process_num = max(len(api_configs), max_process_num)
    engines = engines if engines is not None else [""] * max_process_num
    if engines is not None and len(engines) == 1:
        engines = engines * max_process_num
    if engines is not None and len(engines) > 1:
        assert (
            len(engines) == max_process_num
        ), f"Number of engines must be equal to number of api config files when specific multiple engines, but got {len(engines)} engines and {max_process_num} api config files."

    sem = asyncio.Semaphore(max_process_num)
    pbar = tqdm(total=len(texts), desc=f"Batch requests")
    tasks = [
        asyncio.ensure_future(
            bound_fetch(
                sem,
                pbar,
                tools[i % max_process_num],
                prompt=prompt,
                model=engines[i % max_process_num],
                **kwargs,
            )
        )
        for i, prompt in enumerate(texts)
    ]
    task_batches = [
        tasks[i : i + max_process_num] for i in range(0, len(tasks), max_process_num)
    ]
    results = []
    for batch in task_batches:
        batch_result = await asyncio.gather(*batch)
        results.extend(batch_result)
        time.sleep(request_interval)
    return results
