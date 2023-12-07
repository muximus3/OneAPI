# -*- coding: utf-8 -*-
import json
from typing import List
import openai
import anthropic
from pydantic import BaseModel
from abc import ABC, abstractmethod
import sys
import os
from typing import Callable, Optional, Sequence, List, Union
import tiktoken
import asyncio
import transformers
import logging
from typing import Self

sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from oneapi import OneAPITool, register_client, AbstractClient, AbstractConfig


class MockClient(AbstractClient):
    def __init__(self, method: AbstractConfig) -> None:
        super().__init__(method)
        self.method = method
        self.client = None
        self.aclient = None

    @classmethod
    def from_config(cls, config: dict = None, config_file: str = "") -> Self:
        return cls(AbstractConfig(**config))

    def format_prompt(
        self, prompt: Union[str, list[str], list[dict]], system: str = ""
    ):
        pass

    def chat(
        self, prompt: Union[str, list[str], list[dict]], system: str = "", **kwargs
    ):
        return prompt

    def achat(
        self,
        prompt: Union[str, list[str], list[dict]],
        system: str = "",
        max_tokens: int = 1024,
        **kwargs,
    ):
        pass

    def count_tokens(self, texts: List[str], model: str = "") -> int:
        pass


register_client("mock", MockClient)

tool = OneAPITool.from_config("../ant/config/openapi_azure_config_xiaoduo_dev5.json")

str_msg = "太平洋有多大"
list_msg = ["太平洋有多大", "太平洋有多大"]
list_msg_dict = [{"role": "user", "content": "太平洋有多大"}]
system_msg = "太平洋有多大"
for msg in [str_msg, list_msg, list_msg_dict]:
    print(tool.chat(msg, system=system_msg, verbose=True))
