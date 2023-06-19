# OneAPI
Easily access multiple ChatGPT or similar APIs with just one line of code/command.

Save a significant amount of ☕️ time by avoiding the need to read multiple API documents and test them individually.

The currently supported APIs include:
 - [x] OpenAI Official API.
 - [x] Microsoft Azure OpenAI Resource endpoint API.
 - [x] Anthropic Claude series model API.

## Installation
```sh
pip install -U one-api-tool
```

## Usage
### 1. Using python.

OpenAI config:
```json
{
    "api_key": "YOUR_API_KEY",
    "api_base": "https://api.openai.com/v1",
    "api_type": "open_ai"
}
```
Azure OpenAI config:
```json
{
    "api_key": "YOUR_API_KEY",
    "api_base": "Replace with your Azure OpenAI resource's endpoint value.",
    "api_type": "azure"
}
```
Anthropic config:
```json
{
    "api_key": "YOUR_API_KEY",
    "api_base": "https://api.anthropic.com",
    "api_type": "claude"
}
```
`api_key`: Obtain OpenAI API key from the [OpenAI website](https://platform.openai.com/account/api-keys) and Claude API key from the [Anthropic website](https://console.anthropic.com/account/keys).

`api_base`: This is the base API that is used to send requests. You can also specify a proxy URL, such as "https://your_proxy_domain/v1". For example, you can use Cloudflare workers to proxy the OpenAI site.

If you are using Azure APIs, you can find relevant information on the Azure resource dashboard. The API format typically follows this pattern: `https://{your_organization}.openai.azure.com/`.

`api_type`: Currently supported values are "open_ai", "azure", or "claude".

Here is  simple example:
```python
from oneapi import OneAPITool
# Two ways to initialize the OneAPITool object  
# tool = OneAPITool.from_config(api_key, api_base, api_type)
tool = OneAPITool.from_config_file("your_config_file.json")
# Say hello to ChatGPT/Claude/GPT-4
res = tool.simple_chat("Hello AI!")
print(res)
# Get embeddings of some sentences for further usage, e.g., clustering
embeddings = tool.get_embeddings(["Hello AI!", "Hello world!"])
print(len(embeddings)))
# Count the number of tokens
print(tool.count_tokens(["Hello AI!", "Hello world!"]))
```
**Note: Currently, `count_tokens` and `get_embeddings` only support OpenAI or Microsoft Azure API.**
### 2. Using command line
```sh
open-api --config_file CHANGE_TO_YOUR_CONFIG_PATH \
--model gpt-3.5-turbo \
--prompt "1+1=?" 
```
<details open><summary>Output detail</summary>

```text
-------------------- prompt detail 🚀  --------------------

1+1=?

-------------------- prompt end --------------------

-------------------- gpt-3.5-turbo response ⭐️ --------------------

2

-------------------- response end --------------------
```

</details>

#### Arguments detail:

`--config_file` string ${\color{orange}\text{Required}}$ <br>A local configuration file containing API key information.

`--prompt` string ${\color{orange}\text{Required}}$ <br>
The question that would be predicted by LLMs, e.g., A math question would be like: "1+1=?".

`--model` string ${\color{grey}\text{Optional}}$  Defaults to GPT-3.5-turbo or Claude-v1.3 depends on `api_type`<br> Which model to use, e.g., gpt-4.

`--temperature` number ${\color{grey}\text{Optional}}$ Defaults to 1 <br>What sampling temperature to use.  Higher values like 0.9 will make the output more random, while lower values like 0.1 will make it more focused and deterministic. 

`--max_new_tokens` integer ${\color{grey}\text{Optional}}$ Defaults to 2048 <br>
The maximum number of tokens to generate in the chat completion.
The total length of input tokens and generated tokens is limited by the model's context length.

## ToDo
- [ ] Batch requests.
- [ ] Token number counting.
- [ ] Custom token budget.