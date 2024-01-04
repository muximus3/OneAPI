from typing import Any, List, Optional, Sequence, Union
from pydantic import BaseModel
import tiktoken
import json
import os
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from oneapi.clients.abc_client import AbstractConfig, AbstractClient
import numpy as np
from tqdm import tqdm
from rich import print


class OpenAIConfig(AbstractConfig):
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    api_type: str = "openai"
    api_version: str = ""


class AzureConfig(AbstractConfig):
    api_key: str
    api_base: str  # Your Azure OpenAI resource's endpoint value.
    api_type: str = "azure"
    # Official API version, usually there is no need to modify this field.
    api_version: str = "2023-07-01-preview"


class OpenAIDecodingArguments(BaseModel):
    messages: List[dict]
    model: str = "gpt-4"
    max_tokens: int = 2048
    temperature: float = 1
    tools: Optional[list] = None
    seed: Optional[int] = None
    tool_choice: Optional[Union[str, object]] = None
    response_format: dict = None
    top_p: float = 1
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0
    frequency_penalty: float = 0
    user: Optional[str] = ""


class AzureDecodingArguments(BaseModel):
    messages: List[dict]
    # The deployment name you chose when you deployed the ChatGPT or GPT-4 model
    model: str = "gpt-4"
    max_tokens: int = 2048
    tools: Optional[list] = None
    temperature: float = 1
    seed: Optional[int] = None
    tool_choice: Optional[Union[str, object]] = None
    response_format: dict = None
    top_p: float = 1
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0
    frequency_penalty: float = 0
    user: Optional[str] = ""


class OpenAIClient(AbstractClient):
    def __init__(self, config: AbstractConfig) -> None:
        super().__init__(config)
        self.config = config
        self.encoder = None
        if isinstance(self.config, OpenAIConfig):
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)
            self.aclient = AsyncOpenAI(api_key=config.api_key, base_url=config.api_base)
        elif isinstance(self.config, AzureConfig):
            self.client = AzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.api_base,
                api_version=config.api_version,
            )
            self.aclient = AsyncAzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.api_base,
                api_version=config.api_version,
            )

    @classmethod
    def from_config(cls, config: dict = None, config_file: str = ""):
        if isinstance(config_file, str) and os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        if not config:
            raise ValueError("config is empty, pass a config file or a config dict")
        if config["api_type"] == "azure":
            config = AzureConfig(**config)
        else:
            config = OpenAIConfig(**config)
        return cls(config)

    def format_prompt(
        self, prompt: Union[str, list[str], list[dict]], system: str = ""
    ) -> List[dict]:
        msgs = [] if not system else [dict(role="system", content=system)]
        if isinstance(prompt, str):
            msgs.append(dict(role="user", content=prompt))
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            msgs.extend(
                [
                    dict(role="user", content=p)
                    if i % 2 == 0
                    else dict(role="assistant", content=p)
                    for i, p in enumerate(prompt)
                ]
            )
        elif isinstance(prompt, list) and isinstance(prompt[0], dict):
            msgs.extend(prompt)
        else:
            raise AssertionError(
                f"Prompt must be a string, list of strings, list of dict. Got {type(prompt)} instead."
            )
        return msgs

    def chat_stream(self, resp):
        for chunk in resp:
            if not chunk.id or len(chunk.choices) == 0:
                continue
            chunk_choice = chunk.choices[0]
            chunk_message = chunk_choice.delta  # extract the message
            finish_reason = chunk_choice.finish_reason
            if finish_reason == "stop" or finish_reason == "length":
                break
            if chunk_message.content is not None:
                yield chunk_message.content

    def chat(
        self,
        prompt: Union[str, list[str], list[dict]],
        system: str = "",
        max_tokens: int = 1024,
        **kwargs,
    ):
        """
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions
        https://platform.openai.com/docs/api-reference/chat
        """
        if isinstance(self.config, AzureConfig):
            args = AzureDecodingArguments(
                messages=self.format_prompt(prompt=prompt, system=system),
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            args = OpenAIDecodingArguments(
                messages=self.format_prompt(prompt=prompt, system=system),
                max_tokens=max_tokens,
                **kwargs,
            )
        if "verbose" in kwargs and kwargs["verbose"]:
            print(
                f"reqeusts args = {json.dumps(args.model_dump(exclude_none=True), indent=4, ensure_ascii=False)}"
            )
        data = args.model_dump(exclude_none=True)
        completion = self.client.chat.completions.create(**data)
        if data.get("stream", False):
            return self.chat_stream(completion)
        else:
            response_message = completion.choices[0].message
            return response_message.content

    async def achat(
        self,
        prompt: Union[str, list[str], list[dict]],
        system: str = "",
        max_tokens: int = 1024,
        **kwargs,
    ):
        if isinstance(self.config, AzureConfig):
            args = AzureDecodingArguments(
                messages=self.format_prompt(prompt=prompt, system=system),
                max_tokens=max_tokens,
                **kwargs,
            )
        else:
            args = OpenAIDecodingArguments(
                messages=self.format_prompt(prompt=prompt, system=system),
                max_tokens=max_tokens,
                **kwargs,
            )
        if "verbose" in kwargs and kwargs["verbose"]:
            print(
                f"reqeusts args = {json.dumps(args.model_dump(exclude_none=True), indent=4, ensure_ascii=False)}"
            )
        data = args.model_dump(exclude_none=True)
        completion = await self.aclient.chat.completions.create(**data)
        if data.get("stream", False):
            response_message = ""
            # iterate through the stream of events
            async for chunk in completion:
                if not chunk.id or len(chunk.choices) == 0:
                    continue
                chunk_choice = chunk.choices[0]
                chunk_message = chunk_choice.delta  # extract the message
                finish_reason = chunk_choice.finish_reason
                if finish_reason == "stop" or finish_reason == "length":
                    break
                if chunk_message.content is not None:
                    response_message += chunk_message.content
            return response_message
        else:
            response_message = completion.choices[0].message
            return response_message.content

    def get_embeddings(
        self, texts: List[str], model="text-embedding-ada-002", max_batch_size=512, **kwargs
    ) -> List[List[float]]:
        def get_batch_embeddings(batch_texts):
            assert (
                len(batch_texts) <= max_batch_size
            ), "The batch size should not be larger than 2048."
            # replace newlines, which can negatively affect performance.
            batch_texts = [s.replace("\n", " ") for s in batch_texts]

            data = self.client.embeddings.create(input=batch_texts, model=model).data
            # maintain the same order as input.
            data = sorted(data, key=lambda x: x.index)
            return [d.embedding for d in data]

        if len(texts) <= max_batch_size:
            return get_batch_embeddings(texts)
        embeddings = []
        for i in tqdm(range(0, len(texts), max_batch_size), desc="Embedding"):
            batch_text = texts[i : i + max_batch_size]
            embeddings.append(get_batch_embeddings(batch_text))
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
        return embeddings

    def get_embedding(self, text: str, model="text-embedding-ada-002", **kwargs) -> List[float]:
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=model).data[0].embedding
        )

    def count_tokens(self, texts: List[str], model: str = "gpt-4") -> int:
        """
        Encoding name	OpenAI models
        cl100k_base	    gpt-4, gpt-3.5-turbo, text-embedding-ada-002
        p50k_base	    Codex models, text-davinci-002, text-davinci-003
        r50k_base (or gpt2)	GPT-3 models like davinci
        Args:
            texts (List[str]): [description]
            encoding_name (str, optional): Defaults to 'cl100k_base'.
        Returns:
            int: [description]
        """
        if self.encoder is None:
            self.encoder = tiktoken.encoding_for_model(model)
        list_of_tokens = self.encoder.encode_batch(texts)
        return sum([len(tokens) for tokens in list_of_tokens])
