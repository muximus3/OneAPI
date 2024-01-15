# -*- coding: utf-8 -*-
import logging
import asyncio
import time
from typing import Any, List, Union
from tqdm import tqdm
from oneapi.utils import load_json
from oneapi import clients
import os
import traceback

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
        if kwargs:
            client = client_cls.from_config(config | kwargs)
        else:
            client = client_cls.from_config(config)
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
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if stop:
            kwargs["stop"] = stop
        response = self.client.chat(prompt, system, max_tokens=max_tokens, **kwargs)
        return response

    async def achat(
        self,
        prompt: Union[str, list[str], list[dict]],
        system: str = "",
        max_tokens: int = 512,
        **kwargs,
    ):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        response = await self.client.achat(
            prompt, system, max_tokens=max_tokens, **kwargs
        )
        return response

    def get_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self.client.get_embeddings(texts, **kwargs)

    def get_embedding(self, text: str, **kwargs) -> List[float]:
        return self.client.get_embedding(text, **kwargs)

    def count_tokens(self, texts: List[str], model: str = "gpt-4") -> int:
        assert isinstance(
            texts, list
        ), f"Input texts must be a list of strings. Got {type(texts)} instead."
        return self.client.count_tokens(texts, model)


async def bound_fetch(sem, pbar, tool: OneAPITool, prompt: str, model: str, max_retries=2, **kwargs):
    async with sem:
        while max_retries > 0:
            max_retries -= 1
            try:
                res = await tool.achat(prompt=prompt, model=model, **kwargs)
                pbar.update(1)
                return res
            except (Exception, asyncio.CancelledError) as e:
                traceback.print_stack()
                logger.error(f"Error when calling {model} api: {e}")
                logger.info(f"Retrying {model} api...")
                continue
        return None


async def batch_chat(
    api_configs,
    prompts,
    models=None,
    request_interval=0.1,
    min_process_num=1,
    max_process_num=10,
    **kwargs,
):
    if not isinstance(api_configs, list):
        api_configs = [api_configs]
    tools = [
        OneAPITool.from_config(config_file)
        if isinstance(config_file, str)
        else OneAPITool.from_config(**config_file)
        for config_file in api_configs
    ]
    min_process_num = max(len(api_configs), min_process_num)
    if len(tools) < min_process_num:
        tools = tools * min_process_num
    specific_model = kwargs.pop("model", None)
    if models is None:
        models = [specific_model] * min_process_num
    else:
        if "model" in kwargs:
            logger.warning(
                f"Both models and model are specified, model will be ignored."
            )
        if len(models) == 1:
            models = models * min_process_num
        else:
            assert (
                len(models) == min_process_num
            ), f"Number of models must be equal to number of api config files when specific multiple models, but got {len(models)} models and {min_process_num} api config files."

    sem = asyncio.Semaphore(max_process_num)
    pbar = tqdm(total=len(prompts), desc=f"Batch requests")
    async with asyncio.TaskGroup() as g:
        tasks = [
            g.create_task(
                bound_fetch(
                    sem,
                    pbar,
                    tools[i % min_process_num],
                    prompt=prompt,
                    model=models[i % min_process_num],
                    **kwargs,
                )
            )
            for i, prompt in enumerate(prompts)
        ]
        task_batches = [
            tasks[i : i + min_process_num]
            for i in range(0, len(tasks), min_process_num)
        ]
        results = []
        for batch in task_batches:
            batch_result = await asyncio.gather(*batch)
            results.extend(batch_result)
            await asyncio.sleep(request_interval)
    pbar.close()
    return results
