# -*- coding: utf-8 -*-
import json
from typing import Optional, Sequence, List
import openai
import anthropic
from pydantic import BaseModel
from abc import ABC, ABCMeta, abstractmethod


CLAUDE_TEMPLATE = "\n\nHuman: {prompt}\n\nAssistant: "

class ChatGPTMessage(BaseModel):
    role: str
    content: str
    
class AbstrctMethod(BaseModel):
    api_key: str
    api_base: str
    api_type: str
    method_list_models :str 
    method_model_info :str
    method_chat :str
    method_commpletions :str

class ClaudeMethod(AbstrctMethod):
    api_key: str
    api_base: str = "https://api.anthropic.com",
    api_type: str = "claude"
    method_list_models = ""
    method_model_info = ""
    method_chat = "/v1/complete"
    method_commpletions = ""

class AzureMethod(AbstrctMethod):
    api_key: str
    api_base: str # Your Azure OpenAI resource's endpoint value.
    api_type: str = "azure"
    api_version = "2023-03-15-preview" # Official API version, usually there is no need to modify this field.
    deployment_name = "gpt-35-turbo" # The deployment name you chose when you deployed the ChatGPT or GPT-4 model, used for raw http requests
    method_chat = f"/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}" # used for raw http requests
    method_list_models :str = "" 
    method_model_info :str = ""
    method_commpletions :str = ""

class OpenAIMethod(AbstrctMethod):
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    api_type: str = "open_ai"
    method_list_models = "models"
    method_model_info = "models"
    method_chat = "/chat/completions"
    method_commpletions = "completions"

class OpenAIDecodingArguments(BaseModel):
    messages: List[ChatGPTMessage] 
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 2048
    temperature: float = 1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 1.0
    frequency_penalty: float = 1.0
    user: Optional[str] = ""

class AzureDecodingArguments(BaseModel):
    messages: List[ChatGPTMessage] 
    engine: str = "gpt-35-turbo" # The deployment name you chose when you deployed the ChatGPT or GPT-4 model
    max_tokens: int = 2048
    temperature: float = 1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 1.0
    frequency_penalty: float = 1.0


class ClaudeDecodingArguments(BaseModel):
    prompt: str
    model: str = "claude-instant-v1"
    max_tokens_to_sample: int = 2048
    temperature: float = 1
    top_p: float = -1
    top_k: int = -1
    stream: bool = True
    stop_sequences: Optional[Sequence[str]] = [anthropic.HUMAN_PROMPT]


class AbstractAPITool(ABC):

    def simple_chat(args):
        raise NotImplementedError
        

class OpenAITool(AbstractAPITool):

    def __init__(self,method : AbstrctMethod) -> None:
        self.method = method
        if isinstance(self.method, AzureMethod):
            openai.api_key = self.method.api_key
            openai.api_base = self.method.api_base
            openai.api_type = self.method.api_type
            openai.api_version = self.method.api_version
        else:
            openai.api_key = self.method.api_key
            openai.api_base = self.method.api_base

    def simple_chat(self, args: OpenAIDecodingArguments):
        """
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions
        https://platform.openai.com/docs/api-reference/chat
        """

        data = args.dict()
        completion = openai.ChatCompletion.create(**data)
        if data.get("stream", False):
            # create variables to collect the stream of chunks
            collected_chunks = []
            collected_messages = []
            # iterate through the stream of events
            for chunk in completion:
                collected_chunks.append(chunk)  # save the event response
                chunk_choice = chunk["choices"][0]
                chunk_message = chunk_choice["delta"]  # extract the message
                finish_reason = chunk_choice["finish_reason"]
                collected_messages.append(chunk_message)  # save the message
            full_reply_content = "".join([m.get("content", "") for m in collected_messages])
            return full_reply_content
        else:
            return completion.choices[0].message.get("content", "")


    def get_embeddings(self, list_of_text: List[str], engine="text-embedding-ada-002") -> List[List[float]]:
        assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."
        openai.api_key = self.method.api_key
        # replace newlines, which can negatively affect performance.
        list_of_text = [text.replace("\n", " ") for text in list_of_text]

        data = openai.Embedding.create(input=list_of_text, engine=engine).data
        data = sorted(data, key=lambda x: x["index"])  # maintain the same order as input.
        return [d["embedding"] for d in data]


    def get_embedding(self, text: str, engine="text-embedding-ada-002") -> List[float]:
        openai.api_key = self.method.api_key
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")

        return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]




class ClaudeAITool(AbstractAPITool):
    """
    https://console.anthropic.com/claude
    https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/basic_stream.py
    https://console.anthropic.com/docs/api/reference
    """
    def __init__(self,method : AbstrctMethod) -> None:
        self.method = method
        self.client = anthropic.Client(method.api_key)
    
    async def simple_chat(self, args: ClaudeDecodingArguments):
        if args.stream:
            resp = await self.client.acompletion_stream(**args.dict())
            async for data in resp:
                if data["stop_reason"] == "stop_sequence" or data["stop_reason"] == "max_tokens":
                    return data["completion"]
        else:
            resp = await self.client.acompletion(**args.dict())
            return resp
        
    def simple_chat(self, args: ClaudeDecodingArguments):
        if args.stream:
            resp = self.client.completion_stream(**args.dict())
            for  data in resp:
                if data["stop_reason"] == "stop_sequence" or data["stop_reason"] == "max_tokens":
                    return data["completion"]
        else:
            resp = self.client.completion(**args.dict())
            return resp

            
            

class OneAPITool():
    def __init__(self, tool: AbstractAPITool) -> None:
        self.tool = tool

    @classmethod
    def from_config_file(cls, config_file):
        config = cls.load_json(config_file)
        api_type = config.get("api_type")

        if api_type == "claude":
            return cls(ClaudeAITool(ClaudeMethod(api_key=config["api_key"], api_base=config["api"])))
        elif api_type == "azure":
            return cls(OpenAITool(AzureMethod(api_key=config["api_key"], api_base=config["api"])))
        elif api_type == "open_ai":
            return cls(OpenAITool(OpenAIMethod(api_key=config["api_key"], api_base=config["api"])))
        else:
            raise AssertionError(f"Couldn\'t find API type in config file: {config_file}. Please specify \"api_type\" as \"claude\", \"azure\", or \"open_ai\".")

    @classmethod
    def from_config(cls, api_key, api, api_type):
        if api_type == "claude":
            return cls(ClaudeAITool(ClaudeMethod(api_key=api_key, api_base=api)))
        elif api_type == "azure":
            return cls(OpenAITool(AzureMethod(api_key=api_key, api_base=api)))
        elif api_type == "open_ai":
            return cls(OpenAITool(OpenAIMethod(api_key=api_key, api_base=api)))
        else:
            raise AssertionError(f"Couldn\'t find API type: {api_type}. Please specify \"api_type\" as \"claude\", \"azure\", or \"open_ai\".")


    @staticmethod
    def load_json(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def simple_chat(self, prompt, model="", temperature=1, max_new_tokens=2048, stream=True, **kwargs):
        if isinstance(self.tool, OpenAITool):
            msgs = [ChatGPTMessage(role="user", content=prompt)]
            if isinstance(self.tool.method, AzureMethod):
                args = AzureDecodingArguments(messages=msgs, engine=model if model else "gpt-35-turbo", temperature=temperature, max_tokens=max_new_tokens, stream=stream, **kwargs)
            elif isinstance(self.tool.method, OpenAIMethod):
                args = OpenAIDecodingArguments(messages=msgs, model=model if model else "gpt-3.5-turbo", temperature=temperature, max_tokens=max_new_tokens, stream=stream, **kwargs)
        elif isinstance(self.tool.method, ClaudeMethod):
            args = ClaudeDecodingArguments(prompt=f"\n\nHuman: {prompt}\n\nAssistant:", model=model if model else "claude-v1.3", temperature=temperature, max_tokens_to_sample=max_new_tokens, stream=stream, **kwargs)
        else:
            raise AssertionError(f"Not supported api type: {type(self.tool)}")

        response = self.tool.simple_chat(args)
        return response