# OneAPI
Use a single line of code to call multiple model APIs similar to ChatGPT. Currently supported: Azure OpenAI Resource endpoint API, OpenAI Official API, and Anthropic Claude series model API.

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
    "api": "Replace with your Azure OpenAI resource's endpoint value.",
    "api_type": "azure"
}
```
Anthropic config:
```json
{
    "api_key": "YOUR_API_KEY",
    "api": "https://api.anthropic.com",
    "api_type": "claude"
}
```
`api_key`: Obtain your OpenAI API key from the [OpenAI website](https://platform.openai.com/account/api-keys) and your Claude API key from the [Anthropic website](https://console.anthropic.com/account/keys).

`api`: The base API used to send requests. You may also specify a proxy URL like: "https://your_proxy_domain/v1". For Azure APIs, you can find relevant information on the Azure resource dashboard. The API format is usually: `https://{your_organization}.openai.azure.com/`.

`api_type`: Currently supported values are "open_ai", "azure", or "claude".

Initialize the `OneAPITool` object from a local configuration file:
```python
from oneapi import OneAPITool
res = OneAPITool.from_config_file("your_config_file.json").simple_chat("Hello AI!")
print(res)
```

### 2. (Not recommended) Write the configuration directly into the code
```python
from oneapi import OneAPITool
res = OneAPITool.from_config(api_key, api, api_type).simple_chat("Hello AI!")
print(res)
```

### 3. Using with command line
```sh
open-api --config_file CHANGE_TO_YOUR_CONFIG_PATH \
--model gpt-3.5-turbo \
--prompt "1+1=?" 
```
<details><summary>Click to see output detail</summary>

-------------------- prompt detail 🚀  --------------------

1+1=?

-------------------- prompt end --------------------

-------------------- gpt-35-turbo response ⭐️ --------------------

2

-------------------- response end --------------------

</details>

#### Arguments Definitions:

`--config_file` string ${\color{orange}\text{Required}}$ <br>A local configuration file containing API key information.

`--prompt` string ${\color{orange}\text{Required}}$ <br>
The question that would be predicted by LLMs, e.g., A math question would be like: "1+1=?".

`--model` string ${\color{grey}\text{Optional}}$  Defaults to GPT-3.5-turbo or Claude-v1.3 depends on `api_type`<br> Which model to perform evaluation.

`--temperature` number ${\color{grey}\text{Optional}}$ Defaults to 1 <br>What sampling temperature to use.  Higher values like 1 will make the output more random, while lower values like 0.1 will make it more focused and deterministic.

`--max_new_tokens` integer ${\color{grey}\text{Optional}}$ Defaults to 2048 <br>
The maximum number of tokens to generate in the chat completion.
The total length of input tokens and generated tokens is limited by the model's context length.