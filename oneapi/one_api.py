# -*- coding: utf-8 -*-
import json
from typing import Optional, Sequence, List
import openai
import anthropic
from pydantic import BaseModel
from abc import ABC, abstractmethod
import sys
import os
from typing import Callable, Optional, Sequence, List
import tiktoken
import logging
from openai.openai_object import OpenAIObject
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from oneapi.utils import generate_function_description
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filemode="a"
)
CLAUDE_TEMPLATE = "\n\nHuman: {prompt}\n\nAssistant:"
    
class AbstrctMethod(BaseModel):
    api_key: str
    api_base: str
    api_type: str
    api_version: str
    method_list_models :str 
    method_model_info :str
    method_chat :str
    method_commpletions :str

class ClaudeMethod(AbstrctMethod):
    api_key: str
    api_base: str = "https://api.anthropic.com",
    api_type: str = "claude"
    api_version: str = None
    method_list_models = ""
    method_model_info = ""
    method_chat = "/v1/complete"
    method_commpletions = ""

class AzureMethod(AbstrctMethod):
    api_key: str
    api_base: str # Your Azure OpenAI resource's endpoint value.
    api_type: str = "azure"
    api_version = "2023-07-01-preview" # Official API version, usually there is no need to modify this field.
    deployment_name = "gpt-35-turbo" # The deployment name you chose when you deployed the ChatGPT or GPT-4 model, used for raw http requests
    method_chat = f"/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}" # used for raw http requests
    method_list_models :str = "" 
    method_model_info :str = ""
    method_commpletions :str = ""

class OpenAIMethod(AbstrctMethod):
    api_key: str
    api_base: str = "https://api.openai.com/v1"
    api_type: str = "open_ai"
    api_version: str = None
    method_list_models = "models"
    method_model_info = "models"
    method_chat = "/chat/completions"
    method_commpletions = "completions"

class OpenAIDecodingArguments(BaseModel):
    messages: List[dict] 
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 2048
    temperature: float = 1
    functions: Optional[list] = None
    # Controls how the model responds to function calls. 
    # "none" means the model does not call a function, and responds to the end-user. 
    # "auto" means the model can pick between an end-user or calling a function. Specifying a particular function via {"name":\ "my_function"} forces the model to call that function. 
    # "none" is the default when no functions are present. "auto" is the default if functions are present.
    function_call : Optional[str|dict] = None
    top_p: float = 1
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0
    frequency_penalty: float = 0
    user: Optional[str] = ""

class AzureDecodingArguments(BaseModel):
    messages: List[dict] 
    engine: str = "gpt-35-turbo" # The deployment name you chose when you deployed the ChatGPT or GPT-4 model
    max_tokens: int = 2048
    temperature: float = 1
    top_p: float = 1
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0
    frequency_penalty: float = 0
    user: Optional[str] = ""


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

    @abstractmethod
    def simple_chat(args):
        raise NotImplementedError
        

class OpenAITool(AbstractAPITool):

    def __init__(self,method : AbstrctMethod) -> None:
        self.method = method
        self.encoder = None


    def reset_environment(self):
        """Reset the environment variables to the default values. When using asyncio to all multiple apis, this function should be called before each api call.
        """
        openai.api_key = self.method.api_key
        openai.api_base = self.method.api_base
        openai.api_type = self.method.api_type
        openai.api_version = self.method.api_version

    def simple_chat(self, args: OpenAIDecodingArguments):
        """
        https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt?pivots=programming-language-chat-completions
        https://platform.openai.com/docs/api-reference/chat
        """
        self.reset_environment()
        data = args.dict()
        is_function_call = data.get("functions", None) is not None
        if is_function_call:
            data["stream"] = False  
        else:
            if isinstance(args, OpenAIDecodingArguments):
                data.pop("functions")
                data.pop("function_call")
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
            response_message = completion.choices[0].message
            if is_function_call:
                if response_message.get("function_call") is not None:
                    return response_message
            return response_message.get("content", "")

    async def asimple_chat(self, args: OpenAIDecodingArguments):
        self.reset_environment()
        data = args.dict()
        is_function_call = data.get("functions", None) is not None
        if is_function_call:
            data["stream"] = False  
        else:
            if isinstance(args, OpenAIDecodingArguments):
                data.pop("functions")
                data.pop("function_call")
        completion = await openai.ChatCompletion.acreate(**data)
        if data.get("stream", False):
            # create variables to collect the stream of chunks
            collected_chunks = []
            collected_messages = []
            # iterate through the stream of events
            async for chunk in completion:
                collected_chunks.append(chunk)  # save the event response
                chunk_choice = chunk["choices"][0]
                chunk_message = chunk_choice["delta"]  # extract the message
                finish_reason = chunk_choice["finish_reason"]
                collected_messages.append(chunk_message)  # save the message
            full_reply_content = "".join([m.get("content", "") for m in collected_messages])
            return full_reply_content
        else:
            response_message = completion.choices[0].message
            if is_function_call:
                if response_message.get("function_call") is not None:
                    return response_message
            return response_message.get("content", "")
         


    def get_embeddings(self, texts: List[str], engine="text-embedding-ada-002") -> List[List[float]]:
        self.reset_environment()
        assert len(texts) <= 2048, "The batch size should not be larger than 2048."
        # replace newlines, which can negatively affect performance.
        texts = [text.replace("\n", " ") for text in texts]

        data = openai.Embedding.create(input=texts, engine=engine).data
        data = sorted(data, key=lambda x: x["index"])  # maintain the same order as input.
        return [d["embedding"] for d in data]


    def get_embedding(self, text: str, engine="text-embedding-ada-002") -> List[float]:
        self.reset_environment()
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], engine=engine)["data"][0]["embedding"]

    def count_tokens(self, texts: List[str], encoding_name: str = 'cl100k_base') -> int:
        self.reset_environment()
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
            self.encoder = tiktoken.get_encoding(encoding_name)
        list_of_tokens = self.encoder.encode_batch(texts)
        return sum([len(tokens) for tokens in list_of_tokens])



class ClaudeAITool(AbstractAPITool):
    """
    https://console.anthropic.com/claude
    https://github.com/anthropics/anthropic-sdk-python/blob/main/examples/basic_stream.py
    https://console.anthropic.com/docs/api/reference
    """
    def __init__(self,method : AbstrctMethod) -> None:
        self.method = method
        self.client = anthropic.Client(method.api_key)
    
    async def asimple_chat(self, args: ClaudeDecodingArguments):
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
            return cls(ClaudeAITool(ClaudeMethod(api_key=config["api_key"], api_base=config["api_base"])))
        elif api_type == "azure":
            api_version = config.get("api_version")
            if api_version is None:
                api_version = "2023-07-01-preview"
            return cls(OpenAITool(AzureMethod(api_key=config["api_key"], api_base=config["api_base"], api_version=api_version)))
        elif api_type == "open_ai":
            return cls(OpenAITool(OpenAIMethod(api_key=config["api_key"], api_base=config["api_base"])))
        else:
            raise AssertionError(f"Couldn\'t find API type in config file: {config_file}. Please specify \"api_type\" as \"claude\", \"azure\", or \"open_ai\".")

    @classmethod
    def from_config(cls, api_key, api_base, api_type, api_version="2023-07-01-preview"):
        if api_type == "claude":
            return cls(ClaudeAITool(ClaudeMethod(api_key=api_key, api_base=api_base)))
        elif api_type == "azure":
            return cls(OpenAITool(AzureMethod(api_key=api_key, api_base=api_base, api_version=api_version)))
        elif api_type == "open_ai":
            return cls(OpenAITool(OpenAIMethod(api_key=api_key, api_base=api_base)))
        else:
            raise AssertionError(f"Couldn\'t find API type: {api_type}. Please specify \"api_type\" as \"claude\", \"azure\", or \"open_ai\".")


    @staticmethod
    def load_json(file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def simple_chat(self, prompt: str|list|dict, system:str="", functions:List[Callable]=None, function_call:Optional[str|dict]=None, model:str="", temperature:int=1, max_new_tokens:int=2048, stream:bool=True, **kwargs):
        if isinstance(self.tool, OpenAITool):
            msgs = [] if not system else [dict(role="system", content=system)]
            if isinstance(prompt, str):
                msgs.append(dict(role="user", content=prompt))
            elif isinstance(prompt, list):
                msgs.extend(prompt)
            elif isinstance(prompt, dict):
                msgs.append(prompt)
            else:
                raise AssertionError(f"Prompt must be a string, list of strings, or ChatGPTMessage. Got {type(prompt)} instead.")
            if isinstance(self.tool.method, AzureMethod):
                args = AzureDecodingArguments(messages=msgs, engine=model if model else "gpt-35-turbo", temperature=temperature, max_tokens=max_new_tokens, stream=stream, **kwargs)
            elif isinstance(self.tool.method, OpenAIMethod):
                if functions is not None and isinstance(functions, list):
                    functions = [generate_function_description(func) for func in functions]
                if function_call is None and functions is not None:
                    function_call = "auto"
                if function_call is not None and functions is None:
                    function_call = None
                args = OpenAIDecodingArguments(messages=msgs, functions=functions, function_call=function_call, model=model if model else "gpt-3.5-turbo-0613", temperature=temperature, max_tokens=max_new_tokens, stream=stream, **kwargs)
        elif isinstance(self.tool, ClaudeAITool):
            args = ClaudeDecodingArguments(prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}", model=model if model else "claude-v1.3-100k", temperature=temperature, max_tokens_to_sample=max_new_tokens, stream=stream, **kwargs)
        else:
            raise AssertionError(f"Not supported api type: {type(self.tool)}")

        response = self.tool.simple_chat(args)
        return response

    async def asimple_chat(self, prompt: str|list|dict, system:str="", functions:List[Callable]=None, function_call:Optional[str|dict]=None, model:str="", temperature:int=1, max_new_tokens:int=2048, stream:bool=False, **kwargs):
        if isinstance(self.tool, OpenAITool):
            msgs = [] if not system else [dict(role="system", content=system)]
            if isinstance(prompt, str):
                msgs.append(dict(role="user", content=prompt))
            elif isinstance(prompt, list):
                msgs.extend(prompt)
            elif isinstance(prompt, dict):
                msgs.append(prompt)
            else:
                raise AssertionError(f"Prompt must be a string, list of strings, or ChatGPTMessage. Got {type(prompt)} instead.")
            if isinstance(self.tool.method, AzureMethod):
                args = AzureDecodingArguments(messages=msgs, engine=model if model else "gpt-35-turbo", temperature=temperature, max_tokens=max_new_tokens, stream=stream, **kwargs)
            elif isinstance(self.tool.method, OpenAIMethod):
                if functions is not None and isinstance(functions, list):
                    functions = [generate_function_description(func) for func in functions]
                if function_call is None and functions is not None:
                    function_call = "auto"
                if function_call is not None and functions is None:
                    function_call = None
                args = OpenAIDecodingArguments(messages=msgs, functions=functions, function_call=function_call, model=model if model else "gpt-3.5-turbo-0613", temperature=temperature, max_tokens=max_new_tokens, stream=stream, **kwargs)
        elif isinstance(self.tool, ClaudeAITool):
            args = ClaudeDecodingArguments(prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}", model=model if model else "claude-2", temperature=temperature, max_tokens_to_sample=max_new_tokens, stream=stream, **kwargs)
        else:
            raise AssertionError(f"Not supported api type: {type(self.tool)}")

        response = await self.tool.asimple_chat(args)
        return response

    def function_chat(self, prompt: str|list|dict, system:str="", functions:List[Callable]=None, function_call:Optional[str|dict]=None, model:str="", temperature:int=1, max_new_tokens:int=2048, stream:bool=True, **kwargs):
        """A full chain of function calling.
        Step1: Call the model with functions and user prompt.
        Step2: Use the model response to call your API.
        Step3: Send the API response back to the model to summarize.

        Args:
            prompt (str | list | dict): User input.
            system (str, optional): System message for ChatGPT. Defaults to "".
            functions (List[Callable], optional): A list of functions for model to decide with function to use. Defaults to None.
            function_call (Optional[str | dict], optional): Controls how the model responds to function calls. "none" means the model does not call a function, and responds to the end-user. "auto" means the model can pick between an end-user or calling a function. Specifying a particular function via {"name":\ "my_function"} forces the model to call that function. "none" is the default when no functions are present. "auto" is the default if functions are present.. Defaults to None.
            model (str, optional): Model name. Defaults to GPT-3.5-turbo/Claude-v1.3-100k. 
            temperature (int, optional): What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 1.
            max_new_tokens (int, optional): Defaults to 2048.
            stream (bool, optional): Defaults to True.

        Raises:
            AssertionError: When no function response found. Usually because of prompt injection.
        """
        assert len(functions) > 0, "No functions found."
        if isinstance(self.tool, OpenAITool) and isinstance(self.tool.method, OpenAIMethod):
            msgs = [] if not system else [dict(role="system", content=system)]
            if isinstance(prompt, str):
                msgs.append(dict(role="user", content=prompt))
            elif isinstance(prompt, list):
                msgs.extend(prompt)
            elif isinstance(prompt, dict):
                msgs.append(prompt)
            else:
                raise AssertionError(f"Prompt must be a string, list of strings, or ChatGPTMessage. Got {type(prompt)} instead.")
            function_response = self.simple_chat(prompt, system, functions, function_call, model, temperature, max_new_tokens, stream, **kwargs)
            if not isinstance(function_response, dict) or not function_response.get("function_call"):
                raise AssertionError(f"Function call not found in response: {function_response}")
            function_response_detail = function_response.get("function_call")
            logger.debug(f"Function calling step1, function_response_detail: {function_response_detail}")
            arguments = json.loads(function_response_detail["arguments"])
            function_name = function_response_detail["name"]
            # force to use the function name in function_call
            if isinstance(function_call, dict) and "name" in function_call:
                function_name = function_call["name"] 
            func_name_vs_func = {func.__name__: func for func in functions}
            func = func_name_vs_func.get(function_name)
            if not func:
                raise AssertionError(f"Chosen function {function_name} not found in functions: {functions}")
            api_response = func(**arguments)
            logger.debug(f"Function calling step2, calling fun: {function_name}, api_response: {api_response}")
            msgs.append(function_response)
            msgs.append({"role": "function", "name": function_name, "content": json.dumps(api_response)})
            final_response = self.simple_chat(msgs, model=model, temperature=temperature, max_new_tokens=max_new_tokens, stream=stream, **kwargs)
            logger.debug(f"Function calling step3, Model summarize, final_response: {final_response}")
            return final_response
        else:
            raise AssertionError(f"Function chat currently only support api type: {type(self.tool)}")

        



    def get_embeddings(self, texts: List[str], engine="text-embedding-ada-002") -> List[List[float]]:
        if isinstance(self.tool, OpenAITool):
            return self.tool.get_embeddings(texts, engine)  
        else:
            raise AssertionError(f"Not supported api type for embeddings: {type(self.tool)}")

    def get_embedding(self, text: str, engine="text-embedding-ada-002") -> List[float]:
        if isinstance(self.tool, OpenAITool):
            return self.tool.get_embedding(text, engine)
        else:
            raise AssertionError(f"Not supported api type for embeddings: {type(self.tool)}")
    
    def count_tokens(self, texts: List[str], encoding_name: str = 'cl100k_base') -> int:
        if isinstance(self.tool, OpenAITool):
            return self.tool.count_tokens(texts, encoding_name)
        elif isinstance(self.tool, ClaudeAITool):
            return sum([anthropic.count_tokens(text) for text in texts])
        else:
            raise AssertionError(f"Not supported api type for token counting: {type(self.tool)}")