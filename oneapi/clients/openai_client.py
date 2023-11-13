from typing import List, Optional, Sequence
from pydantic import BaseModel
import tiktoken
import json
import os
from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI
from oneapi.clients.abc_client import AbstractMethod, AbstractClient

class AzureMethod(AbstractMethod):
    api_key: str
    api_base: str # Your Azure OpenAI resource's endpoint value.
    api_type: str = "azure"
    api_version : str = "2023-07-01-preview" # Official API version, usually there is no need to modify this field.
    # deployment_name = "gpt-35-turbo" # The deployment name you chose when you deployed the ChatGPT or GPT-4 model, used for raw http requests
    # method_chat = f"/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}" # used for raw http requests
    method_list_models :str = "" 
    method_model_info :str = ""
    method_commpletions :str = ""
    chat_template: str = ""

class OpenAIMethod(AbstractMethod):
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    api_type: str = "openai"
    api_version: str = ""
    method_list_models :str = "models"
    method_model_info :str = "models"
    method_chat : str = "/chat/completions"
    method_commpletions : str = "completions"
    chat_template: str = ""

    

class OpenAIDecodingArguments(BaseModel):
    messages: List[dict] 
    model: str = "gpt-4-turbo"
    max_tokens: int = 2048
    temperature: float = 1
    tools: Optional[list] = None
    tool_choice: Optional[str|object] = None
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
    model: str = "gpt-35-turbo" # The deployment name you chose when you deployed the ChatGPT or GPT-4 model
    max_tokens: int = 2048
    temperature: float = 1
    top_p: float = 1
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0
    frequency_penalty: float = 0
    user: Optional[str] = ""



    
class OpenAIClient(AbstractClient):


    @classmethod
    def from_config(cls, config: dict = None, config_file: str = ""):
        if isinstance(config_file, str) and os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        if not config:
            raise ValueError("config is empty, pass a config file or a config dict")
        if config['api_type'] == "azure":
            method = AzureMethod(**config)
        else:
            method = OpenAIMethod(**config)
        return cls(method)

    def __init__(self, method : AbstractMethod) -> None:
        super().__init__(method)
        self.method = method
        self.encoder = None
        if isinstance(method, OpenAIMethod):
            self.client = OpenAI(api_key=method.api_key, base_url=method.api_base)
            self.aclient = AsyncOpenAI(api_key=method.api_key, base_url=method.api_base)
        elif isinstance(method, AzureMethod):
            self.client = AzureOpenAI(api_key=method.api_key, azure_endpoint=method.api_base, api_version=method.api_version)
            self.aclient = AsyncAzureOpenAI(api_key=method.api_key, azure_endpoint=method.api_base, api_version=method.api_version)

    def format_prompt(self, prompt: str|list[str]|list[dict], system: str = "") -> List[dict]:
        msgs = [] if not system else [dict(role="system", content=system)]
        if isinstance(prompt, str):
            msgs.append(dict(role="user", content=prompt))
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            msgs.extend([dict(role="user", content=p) if i%2 == 0 else dict(role='assistant', content=p) for i, p in enumerate(prompt)])
        elif isinstance(prompt, list) and isinstance(prompt[0], dict):
            msgs.extend(prompt)
        else:
            raise AssertionError(f"Prompt must be a string, list of strings. Got {type(prompt)} instead.")
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
    
    def chat(self, prompt: str | list[str] | list[dict], system: str = "", max_new_tokens: int = 1024, **kwargs):
        """
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions
        https://platform.openai.com/docs/api-reference/chat
        """
        if isinstance(self.method, AzureMethod):
            args = AzureDecodingArguments(messages=self.format_prompt(prompt=prompt, system=system), max_tokens=max_new_tokens, **kwargs)
        else:
            args = OpenAIDecodingArguments(messages=self.format_prompt(prompt=prompt, system=system), max_tokens=max_new_tokens, **kwargs)
        if "verbose" in kwargs and kwargs["verbose"]:
            print(f"reqeusts args = {json.dumps(args.model_dump(), indent=4, ensure_ascii=False)}")
        data = args.model_dump()
        completion = self.client.chat.completions.create(**data)
        if data.get("stream", False):
            return self.chat_stream(completion)
        else:
            response_message = completion.choices[0].message
            return response_message.content

    async def achat(self, prompt: str | list[str] | list[dict], system: str = "", max_new_tokens: int = 1024, **kwargs):
        if isinstance(self.method, AzureMethod):
            args = AzureDecodingArguments(messages=self.format_prompt(prompt=prompt, system=system), max_tokens=max_new_tokens, **kwargs)
        else:
            args = OpenAIDecodingArguments(messages=self.format_prompt(prompt=prompt, system=system), max_tokens=max_new_tokens, **kwargs)
        if "verbose" in kwargs and kwargs["verbose"]:
            print(f"reqeusts args = {json.dumps(args.model_dump(), indent=4, ensure_ascii=False)}")
        data = args.model_dump()
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


    def get_embeddings(self, texts: List[str], model="text-embedding-ada-002") -> List[List[float]]:
        assert len(texts) <= 2048, "The batch size should not be larger than 2048."
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]

        data = self.client.embeddings.create(input=texts, model=model).data
        data = sorted(data, key=lambda x: x.index)  # maintain the same order as input.
        return [d.embedding for d in data]


    def get_embedding(self, text: str, model="text-embedding-ada-002") -> List[float]:
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def count_tokens(self, texts: List[str], model: str = 'gpt-4') -> int:
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