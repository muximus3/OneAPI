# OneAPI

The LLM calling tool is designed for researchers to interact with the language model. It can be accessed through a user interface (UI), code, or command.

You can engage in multi-turn conversations with ChatGPT or other LLMs APIs and automatically save them in a **training-specific** data format.

**Step 1: Installation (requires Python environment and python >= 3.11):** `pip install one-api-tool`

**Step 2: Start the command:** `one-api`

**Step 3: Select the API type and set the key or other information following the guide.**

![Alt text](/assets/select.png)

**Step 4: Initiate the chat dialogue and start a conversation:**

![Alt text](/assets/chat.png)

**Step 5: Edit the conversation without restart:**
- `: + clear` to clear the conversation history
- `: + d_clear` to clear the conversation history and system prompt
- `: + undo` to remove the latest message
- `: + save` to save the current session history (the program also saves automatically upon exit)
- `: + load` to load the last conversation from cache file. Specify a numerical index to load a specific conversation, e.g., `:load -2`.
- `: + system` to set the system prompt

![Alt text](/assets/save.png)



#### The currently supported APIs include:
 - [x] OpenAI Official API.
    - [x] ChatGPT: GPT-3.5-turbo/GPT-4.
    - [x] Token number counting.
    - [x] Embedding generation.
 - [x] Microsoft Azure OpenAI Resource endpoint API.
    - [x] ChatGPT: GPT-3.5-turbo/GPT-4.
    - [x] Token number counting.
    - [x] Embedding generation.
 - [x] Anthropic Claude series model API.
    - [x] Claude-v1.3-100k, etc.
    - [x] Token number counting.
- [x] Huggingface LLMs.
    - [x] Huggingface_hub
    - [x] Local deployed Inference Endpoint, e.g., [Text Generation Inference](https://github.com/huggingface/text-generation-inference).

- [x] [vLLM](https://github.com/vllm-project/vllm) deployed Inference Endpoint.

    Note: If you deploy vLLM with OpenAI-Compatible API as follows, you should set your api config as `api_type="openai"` and `api_base="http://ip:port/v1"` and `api_key="EMPTY`.
    
    Deploy vLLM with OpenAI-Compatible API:
    ```python 
    python -m vllm.entrypoints.openai.api_server \
    --model /path_to_the_model/facebook/opt-125m \
    --chat-template ./examples/template_chatml.jinja
    ```
    
## Installation

Requirements Python >=3.9

```sh
pip install -U one-api-tool
```

## Usage
### 1. With python.

OpenAI config:
```json
{
    "api_key": "YOUR_API_KEY",
    "api_base": "https://api.openai.com/v1",
    "api_type": "openai"
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
    "api_type": "anthropic"
}
```
Huggingface config:

```json
{
    "api_key": "",
    "api_base": "http://ip:port",
    "api_type": "huggingface",
    "chat_template": "your_jinja2_template"
}
```
VLLM config:

```json
{
    "api_key": "",
    "api_base": "http://ip:port/generate",
    "api_type": "vllm",
    "chat_template": "your_jinja2_template"
}
```

`api_key`: Obtain OpenAI API key from the [OpenAI website](https://platform.openai.com/account/api-keys) and Claude API key from the [Anthropic website](https://console.anthropic.com/account/keys).

`api_base`: This is the base API that is used to send requests. You can also specify a proxy URL, such as "https://your_proxy_domain/v1". For example, you can use Cloudflare workers to proxy the OpenAI site.

If you are using Azure APIs, you can find relevant information on the Azure resource dashboard. The API format typically follows this pattern: `https://{your_organization}.openai.azure.com/`.

`api_type`: Currently supported values are "open_ai", "azure", or "anthropic".

`api_version`: This field is optional. Azure provides several versions of APIs, such as "2023-03-15-preview". However, the OpenAI SDK always has a default value set for this field. Therefore, you should only specify a specific value if you want to use that particular version of APIs.

`chat_template`: This field is optional. When using a local endpoint server, you can pass a Jinja2 template specifically designed for that model, **mathing the template during training**. The template render function takes `prompt` and `system` as parameters: `template.render(prompt=prompt, system=system)`. The default template differs based on the type of prompt. For string input, it is as follows:
 ```python
 DEFAULT_STR_TEMP_SYSTEM_USER_ASSISTANT = """{% if system != '' %}{{'<s>System:\n'+system+'\n\nHuman\n'+prompt+'\n\nAssistant:\n'}}{% else %}{{'<s>Human:\n'+prompt+'\n\nAssistant:\n'}}{% endif %}"""
 ```
For a list of dictionary messages in OpenAI format, the default template is:
 ```python  
 DEFAULT_LIST_MSG_TEMP_SYSTEM_USER_ASSISTANT = """{% for message in prompt %}{% if loop.first %}{% if message['role'] == 'user' %}{% if loop.length != 1 %}{{ '<s>Human:\n' + message['content'] }}{% else %}{{ '<s>Human:\n' + message['content'] + '\n\nAssistant:\n' }}{% endif %}{% elif message['role'] == 'system' %}{{ '<s>System:\n' + message['content'] }}{% endif %}{% elif message['role'] == 'user' %}{% if loop.last %}{{ '\n\nHuman:\n' + message['content'] + '\n\nAssistant:\n'}}{% else %}{{ '\n\nHuman:\n' + message['content']}}{% endif %}{% elif message['role'] == 'assistant' %}{{ '\n\nAssistant:\n' + message['content'] }}{% endif %}{% endfor %}"""
 ```

#### Chat example:
#### There are three acceptable types of inputs for function `chat()`: 
- list of dicts
- string
- list of string
```python
from oneapi import OneLLM
import asyncio
# Two ways to initialize the OneAPITool object  
# llm = OneAPITool.from_config(api_key=api_key, api_base=api_base, api_type=api_type)
llm = OneLLM.from_config("your_config_file.json")

# There are three acceptable types of inputs.
conversations_openai_style = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hello, how can i assistant you today?"}, {"role": "user", "content": "I want to know the weather of tomorrow"}]
conversation_with_system_msg = [{"role": "system", "content": "Now you are a weather forecast assistant."},{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hello, how can i assistant you today?"}, {"role": "user", "content": "I want to know the weather of tomorrow"}]
string_message = "Hello AI!"
list_of_string_messages = ["Hello AI!", "Hello, how can i assistant you today?", "I want to know the weather of tomorrow"]

for msg in [conversations_sharegpt_style, conversations_openai_style, conversation_with_system_msg, string_message, list_of_string_messages]:
    res = llm(msg)
    print(res)

# Pass system message independently
res = llm("Hello AI!", system="Now you are a helpful assistant.")
print(res)

#Set `vebose=True` to print the detail of args passing to LLMs
res = llm("Hello AI!", verbose=True) 

# Async chat 
res = asyncio.run(llm.achat("How\'s the weather today?", model="gpt-4", stream=False))
print(res)

# Get embeddings of some sentences for further usage, e.g., clustering
embeddings = llm.get_embeddings(["Hello AI!", "Hello world!"])
print(len(embeddings))

# Count the number of tokens
print(llm.count_tokens(["Hello AI!", "Hello world!"]))
```
**Note: Currently, the `get_embeddings` function only supports OpenAI, Microsoft Azure API, and locally deployed Inference Endpoint using [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) (Which is really fast).**

#### Batch request with asyncio, `python >= 3.11` required.
```python
from oneapi import batch_chat
import asyncio
import time

configs = [
    {"api_key": "EMPTY", "api_base": "http://0.0.0.0:8000/v1", "api_type": "openai"},
    {"api_key": "EMPTY", "api_base": "http://0.0.0.0:8000", "api_type": "vllm"},
    {"api_key": "", "api_base": "http://0.0.0.0:8999", "api_type": "huggingface"},
    {"api_key": "", "api_base": "http://0.0.0.0:8998", "api_type": "huggingface"},
    "azure_openai_config.json",
    "anthropic_config.json",
    "openapi_config.json",
]
prompts = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of Belgium?",
    "What is the capital of Netherlands?",
    "What is the capital of Luxembourg?",
    "What is the capital of Denmark?",
    "What is the capital of Sweden?",
    "What is the capital of Norway?",
    "What is the capital of Finland?",
    "Tell me a joke about Trump.",
    "Tell me a joke about Biden.",
    "Tell me a joke about Obama.",
    "Tell me a joke about Bush.",
    "Tell me a joke about Clinton.",
    "Tell me a joke about Reagan.",
    "Tell me a joke about Carter.",
    "Tell me a joke about Soviet Union.",
    [{"role": "user", "content": "Where is Mars?"}]
]
tic = time.time()
models = ["path_to_vllm_deployed_model", "path_to_vllm_deployed_model", "", "", "gpt-4", "claude-v2.1", "gpt-3.5-turbo"]
res = asyncio.run(
    batch_chat(configs, prompts, models, request_interval=0.01, min_process_num=200)
)
print(len(res))
print("Time: ", time.time() - tic)
```


### 2. Using command line
#### Interactive
```sh
one-api
```

#### Non-interactive

```sh
open-api --config_file CHANGE_TO_YOUR_CONFIG_PATH \
--model gpt-3.5-turbo \
--prompt "1+1=?" 
```

<details open><summary>Output detail</summary>

```text
-------------------- prompt detail 🚀  --------------------

1+1=?

-------------------- gpt-3.5-turbo response ⭐️ --------------------

2

```

</details>

#### Arguments detail:

`--config_file` string ${\color{orange}\text{Required}}$ <br>A local configuration file containing API key information.

`--prompt` string ${\color{orange}\text{Required}}$ <br>
The question that would be predicted by LLMs, e.g., A math question would be like: "1+1=?".

`--system` string ${\color{grey}\text{Optional}}$  Defaults to null <br> System message to instruct chatGPT, e.g., You are a helpful assistant.

`--model` string ${\color{grey}\text{Optional}}$  Defaults to GPT-3.5-turbo or Claude-v1.3 depends on `api_type`<br> Which model to use, e.g., gpt-4.

`--temperature` int ${\color{grey}\text{Optional}}$ Defaults to 1 <br>What sampling temperature to use.  Higher values like 0.9 will make the output more random, while lower values like 0.1 will make it more focused and deterministic. 

`--max_tokens` int ${\color{grey}\text{Optional}}$ Defaults to 2048 <br>
The maximum number of tokens to generate in the chat completion.
The total length of input tokens and generated tokens is limited by the model's context length.

`--save_to_file` bool ${\color{grey}\text{Optional}}$ Defaults to True <br>
Save the prompt and response to local file at directory "~/.cache/history_cache_{date_of_month}" with the format style of shareGPT.

## ToDo
- [x] Batch requests.
- [x] Token number counting.
- [x] Async requests.
- [x] Custom LLMs.
- [ ] Custom token budget.
- [ ] Using tools.

## Architecture

![img](/assets/architecture.png)