import sys
import os
import argparse
from pathlib import Path
from packaging import version
from rich import print as rprint
from rich.markdown import Markdown
from rich.rule import Rule
import datetime
import time
import json
import requests
import pkg_resources
import inquirer
from dotenv import load_dotenv, find_dotenv
sys.path.append(
    os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi import OneAPITool, ChatAgent
import oneapi
local_env_path = f'{oneapi.__path__[0]}/.env'
load_dotenv(local_env_path)

def check_for_update():
    # Fetch the latest version from the PyPI API
    response = requests.get(f'https://pypi.org/pypi/one-api-tool/json')
    latest_version = response.json()['info']['version']

    # Get the current version using pkg_resources
    current_version = pkg_resources.get_distribution("one-api-tool").version

    return version.parse(latest_version) > version.parse(current_version)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description="one-api [<args>]")
    parser.add_argument("-c",
                        "--config_file",
                        type=str,
                        default="",
                        help="config file path",
                        required=False)
    parser.add_argument("-p",
                        "--prompt",
                        type=str,
                        default="hello",
                        help="question",
                        required=False)
    parser.add_argument("-s",
                        "--system",
                        type=str,
                        default="",
                        help="question",
                        required=False)
    parser.add_argument("-m",
                        "--model",
                        type=str,
                        default="",
                        help="evaluate model name, e.g., gpt-35-turbo, gpt-4",
                        required=False)
    parser.add_argument("-stf",
                        "--save_to_file",
                        type=str2bool,
                        default=True,
                        help="save to file or not,  if true, save to ~/.cache/history_cache_month.jsonl",
                        required=False)
    parser.add_argument("-te",
                        "--temperature",
                        type=float,
                        default=0.1,
                        help="0-1, higher temperature more random result",
                        required=False)
    parser.add_argument("-mnt",
                        "--max_new_tokens",
                        type=int,
                        default=2048,
                        help="max output token length",
                        required=False)
    parser.add_argument('--version',
                        action='store_true',
                        help='display current Open Interpreter version')
    args = parser.parse_args()
    if args.version:
        print("One API", pkg_resources.get_distribution("open-interpreter").version)
        if check_for_update():
            print("A new version is available. Please run 'pip install --upgrade one-api-tool'.")
        return
    if args.prompt and args.config_file:
        tool = OneAPITool.from_config_file(args.config_file)
        response = tool.simple_chat(
            prompt=args.prompt,
            system=args.system,
            model=args.model,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        prompt = f'{args.system}\n\n{args.prompt}' if args.system else args.prompt
        print(
            f"\n{'-'*20} prompt detail 🚀  {'-'*20}\n\n{prompt}\n\n{'-'*20} prompt end {'-'*20}"
        )
        print(
            f"{'-'*20} {args.model} response ⭐️ {'-'*20}\n\n{response}\n\n{'-'*20} response end {'-'*20}\n\n"
        )
        if args.save_to_file:
            if not args.model:
                api_type = OneAPITool.load_json(args.config_file)['api_type']
                if api_type in ["open_ai", "azure"]:
                    model = "gpt-35-turbo" 
                else:
                    model = "claude-2"
            else:
                model = args.model 
            month = datetime.datetime.now().strftime("%Y%m")
            cace_dir = Path.home()/".cache"
            if not cace_dir.exists():
                cace_dir.mkdir(parents=True)
            cace_file_dir = cace_dir/f"history_cache_{month}.jsonl"
            with open(cace_file_dir, "a+" if cace_file_dir.is_file() else "w+") as f:
                format_data = {'id': str(time.time()), 'system_prompt': args.system, 'model': model, 'dataset_name': 'human', 'conversations': [{'from': 'user', 'value': args.prompt}, {'from': 'assistant', 'value': response.strip()}]}
                f.write(json.dumps(format_data, ensure_ascii=False) + "\n")
            print(f'Save model response to file success! DIR: {cace_file_dir}')
    else:
        rprint(Markdown(f"\nWelcome to **One API**.\n"))
        questions = [inquirer.List('api_type', message="Please select the API type you want to use", choices=['open_ai', 'azure', 'claude'])]
        answers = inquirer.prompt(questions)
        api_type = answers['api_type']
        if api_type == "open_ai":
            key_word = "OPENAI"
            default_url = "https://api.openai.com/v1"
            default_model = 'gpt-4'
        elif api_type == "azure":
            key_word = "AZURE"
            default_url = ""
            default_model = 'gpt-4'
        elif api_type == "claude":
            key_word = "CLAUDE"
            default_url = "https://api.anthropic.com"
            default_model = 'claude-2'
        else:
            pass

        if f'{key_word}_API_KEY' in os.environ:
            api_key = os.environ.get(f"{key_word}_API_KEY")
            api_base = os.environ.get(f"{key_word}_API_BASE")
            model = os.environ.get(f"{key_word}_MODEL")
            if api_base.strip() == "":
                api_base = default_url
            if model.strip() == "":
                model = default_model
        else: 
            questions = [inquirer.Text("api_key", message=f"{key_word.title()} API key"),
                inquirer.Text("api_base", message=f"{key_word.title()} API base URL." + (f" Enter to use the default {key_word.title()} URL." if api_type != 'azure' else "")),
                inquirer.Text("model", message=f"Model/Engine, Enter to use the default {key_word.title()} model: {default_model}"), 
                inquirer.Confirm("set_to_envs", message=f"Save API setting to the local environment path? \"{local_env_path}\"", default=True)
                ]
            answers = inquirer.prompt(questions)
            api_key = answers['api_key']
            api_base = answers.get('api_base')
            model = answers.get('model')
            set_to_envs = answers['set_to_envs']
            if api_base.strip() == "":
                api_base = default_url
            if model.strip() == "":
                model = default_model
            if set_to_envs:
                with open(local_env_path, 'a+') as f:
                    f.write(f"{key_word}_API_KEY={api_key}\n")
                    f.write(f"{key_word}_API_BASE={api_base}\n")
                    f.write(f"{key_word}_MODEL={model}\n")
        agent = ChatAgent(llm=OneAPITool.from_config(api_key=api_key, api_base=api_base, api_type=api_type))
        agent.model = model
        rprint(Rule(style='white', end="\n\n"))
        rprint(Markdown(f"Using **Model:** `{model}`"), end="\n\n")
        rprint(Markdown(f"Set `system prompt`, press enter to skip: "))
        system_message = input("> ")
        rprint(Rule(style='white', title="Start chat", end="\n\n"))

        agent.system_message = system_message
        agent.chat()
if __name__ == "__main__":
    main()