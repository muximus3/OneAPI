# OneAPI
Easily access multiple ChatGPT or similar APIs with just one line of code/command.

Save a significant amount of ☕️ time by avoiding the need to read multiple API documents and test them individually.

The currently supported APIs include:
 - [x] OpenAI Official API.
    - [x] ChatGPT: GPT-3.5-turbo/GPT-4.
    - [x] Token number counting.
    - [x] Embedding generation.
    - [x] Function calling.
 - [x] Microsoft Azure OpenAI Resource endpoint API.
    - [x] ChatGPT: GPT-3.5-turbo/GPT-4.
    - [x] Token number counting.
    - [x] Embedding generation.
 - [x] Anthropic Claude series model API.
    - [x] Claude-v1.3-100k, etc.
    - [x] Token number counting.

## Installation

Requirements Python >=3.10

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
    "api_type": "azure",
    "api_version": "2023-03-15-preview" 
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

`api_version`: This field is optional. Azure provides several versions of APIs, such as "2023-03-15-preview". However, the OpenAI SDK always has a default value set for this field. Therefore, you should only specify a specific value if you want to use that particular version of APIs.

#### Chat example:
#### There are three acceptable types of inputs for function `simple_chat()`: 
- list of dicts
- string
- list of string
```python
from oneapi import OneAPITool
import asyncio
# Two ways to initialize the OneAPITool object  
# tool = OneAPITool.from_config(api_key, api_base, api_type)
tool = OneAPITool.from_config_file("your_config_file.json")

# There are three acceptable types of inputs.
conversations_sharegpt_style = [{"from": "human", "value": "hello"}, {"from": "assistant", "value": "Hello, how can i assistant you today?"}, {"from": "human", "value": "I want to know the weather of tomorrow"}]
conversations_openai_style = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hello, how can i assistant you today?"}, {"role": "user", "content": "I want to know the weather of tomorrow"}]
conversation_with_system_msg = [{"role": "system", "content": "Now you are a weather forecast assistant."},{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hello, how can i assistant you today?"}, {"role": "user", "content": "I want to know the weather of tomorrow"}]
string_message = "Hello AI!"
list_of_string_messages = ["Hello AI!", "Hello, how can i assistant you today?", "I want to know the weather of tomorrow"]

for msg in [conversations_sharegpt_style, conversations_openai_style, conversation_with_system_msg, string_message, list_of_string_messages]:
    res = tool.simple_chat(msg)
    print(res)

# Pass system message independently
res = tool.simple_chat("I want to know the weather of tomorrow", system="Now you are a weather forecast assistant.")
print(res)

# Async chat 
res = asyncio.run(tool.asimple_chat("How\'s the weather today?", model="gpt-4", stream=False))
print(res)

# Get embeddings of some sentences for further usage, e.g., clustering
embeddings = tool.get_embeddings(["Hello AI!", "Hello world!"])
print(len(embeddings))

# Count the number of tokens
print(tool.count_tokens(["Hello AI!", "Hello world!"]))
```
**Note: Currently, `get_embeddings` only support OpenAI or Microsoft Azure API.**
### Batch request with asyncio
```python
from oneapi.one_api import batch_chat
import asyncio

claude_config = "anthropic_config.json"
openai_config = "openapi_config.json"
azure_config = "openapi_azure_config.json"
# The coccurent number of requests would be 3, which is the same as the length of the configs list.
configs = [claude_config, openai_config, azure_config]
prompts = ["How\'s the weather today?", "How\'s the weather today?", "How\'s the weather today?"]
res = asyncio.run(batch_chat(configs, prompts, stream=False))
print(res)

```
#### Simple function calling example:

```python
from oneapi import OneAPITool

def get_whether_of_city(city: str, date: str) -> dict:
    """Get the weather of a city at a date

    Args:
        city (str): City name
        date (str): Date of the weather

    Returns:
        Dict: Weather information
    """
    return {"city": city, "date": date, "weather": "sunny", "temperature": 30, "air_condition": "good"}

# tool = OneAPITool.from_config(api_key, api_base, api_type)
tool = OneAPITool.from_config_file("your_config_file.json")
res = tool.function_chat("What's the weather like in New York on July 10th?", functions=[get_whether_of_city])
print(res)
```

<details open> <summary>Output detail</summary>

```text

On July 10th, 2022, the weather in New York is expected to be sunny. The temperature will be around 30 degrees Celsius (86 degrees Fahrenheit). The air quality is expected to be good.
```

</details>

#### Custom function calling example:
```python
from oneapi import OneAPITool
import json

def get_whether_of_city(city: str, date: str) -> dict:
    """Get the weather of a city at a date

    Args:
        city (str): City name
        date (str): Date of the weather

    Returns:
        Dict: Weather information
    """
    return {"city": city, "date": date, "weather": "sunny", "temperature": 30, "air_condition": "good"}

# tool = OneAPITool.from_config(api_key, api_base, api_type)
tool = OneAPITool.from_config_file("your_config_file.json")
msgs = [{"role": "user", "content": "What's the weather like in New York on July 10th?"}]
function_response = tool.simple_chat(msgs, model="gpt-3.5-turbo-0613", functions=[get_whether_of_city])
print(f"Function response:\n{function_response}")
function_call = function_response["function_call"]
arguments = json.loads(function_call["arguments"])
wether_info = get_whether_of_city(**arguments)
print(f"Wether_info:\n{wether_info}")
msgs.append(function_response)
msgs.append({"role": "function", "name": function_call["name"], "content": json.dumps(wether_info)})
second_res = api.simple_chat(msgs, model="gpt-3.5-turbo-0613")
print(f"Second response:\n{second_res}")

```
<details open> <summary>Output detail</summary>

Function response:
```json
{
  "role": "assistant",
  "content": null,
  "function_call": {
    "name": "get_whether_of_city",
    "arguments": "{\n  \"city\": \"New York\",\n  \"date\": \"2022-07-10\"\n}"
  }
}
```
Wether_info: 
```json
{"city": "New York", "date": "July 10th", "weather": "sunny", "temperature": 30, "air_condition": "good"}
```

Second response:

```text
On July 10th, 2022, the weather in New York is expected to be sunny with a temperature of 30 degrees Celsius. The air quality is also good.
```
</details>


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

`--system` string ${\color{grey}\text{Optional}}$  Defaults to null <br> System message to instruct chatGPT, e.g., You are a helpful assistant.

`--model` string ${\color{grey}\text{Optional}}$  Defaults to GPT-3.5-turbo or Claude-v1.3 depends on `api_type`<br> Which model to use, e.g., gpt-4.

`--temperature` int ${\color{grey}\text{Optional}}$ Defaults to 1 <br>What sampling temperature to use.  Higher values like 0.9 will make the output more random, while lower values like 0.1 will make it more focused and deterministic. 

`--max_new_tokens` int ${\color{grey}\text{Optional}}$ Defaults to 2048 <br>
The maximum number of tokens to generate in the chat completion.
The total length of input tokens and generated tokens is limited by the model's context length.

`--save_to_file` bool ${\color{grey}\text{Optional}}$ Defaults to True <br>
Save the prompt and response to local file at directory "~/.cache/history_cache_{date_of_month}" with the format style of shareGPT.

## ToDo
- [x] Batch requests.
- [x] OpenAI function_call.
- [x] Token number counting.
- [x] Async requests.
- [ ] Custom token budget.