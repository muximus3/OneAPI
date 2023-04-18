# OneAPI
Use only one line of code to call multiple model APIs similar to ChatGPT. Currently supported: Azure OpenAI Resource endpoint API, OpenAI Official API, and Anthropic Claude series model API.
## Installation
```sh
pip install -U one-api-tool
```
## Usage
### 1. (Recommended method) Set your key information in the local configuration file.

OpenAI config:
```json
{
    "api_key": "YOUR_API_KEY",
    "api": "https://api.openai.com/v1",
    "api_type": "open_ai"
}
```
Azure OpenAI config:
```json
{
    "api_key": "YOUR_API_KEY",
    "api": "Change to your Azure OpenAI resource's endpoint value.",
    "api_type": "azure"
}
```
Antropic config:
```json
{
    "api_key": "YOUR_API_KEY",
    "api": "https://api.anthropic.com",
    "api_type": "claude"
}
```
`api_key` OpenAI API key is availalle on the [website](https://platform.openai.com/account/api-keys), Clade API key here [website](https://console.anthropic.com/account/keys)

`api` The base api used to send requests, you may also specify to a proxy url like: "https://your_proxy_domain/v1". For Azure APIs, you can find relevant information on the Azure resource dashboard, the form of API is usually:  https://{your origization}.openai.azure.com/

`api_type` Currently supported: "open_ai", "azure" or "claude".

Initialize the `OneAPITool` object from a local configuration file.:
```python
from oneapi import OneAPITool
res = OneAPITool.from_config_file("your_config_file.json").simple_chat("Hello AI!")
print(res)
```
### 2. (Not recommended) Writing the configuration directly into the code
```python
from oneapi import OneAPITool
res = OneAPITool.from_config(api_key, api, api_type).simple_chat("Hello AI!")
print(res)
```