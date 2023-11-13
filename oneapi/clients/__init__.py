from oneapi.clients.abc_client import AbstractClient, AbstractMethod
from oneapi.clients.claude_client import ClaudeClient, ClaudeMethod, ClaudeDecodingArguments
from oneapi.clients.hf_client import HuggingfaceClient, HuggingFaceMethod, HuggingFaceDecodingArguments
from oneapi.clients.vllm_client import VLLMClient, VLLMMethod, VLLMDecodingArguments
from oneapi.clients.openai_client import OpenAIClient, OpenAIMethod, OpenAIDecodingArguments, AzureMethod, AzureDecodingArguments

clients_register = {
    "claude": ClaudeClient,
    "openai": OpenAIClient,
    "open_ai": OpenAIClient,
    "azure": OpenAIClient,
    "huggingface": HuggingfaceClient,
    "vllm": VLLMClient
}