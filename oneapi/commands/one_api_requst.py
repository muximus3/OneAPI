import sys
import os
import argparse
from pathlib import Path
import datetime
import time
import json
sys.path.append(
    os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi import OneAPITool


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
                        help="config file path",
                        required=True)
    parser.add_argument("-p",
                        "--prompt",
                        type=str,
                        help="question",
                        required=True)
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
    args = parser.parse_args()
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
            format_data = {'id': str(time.time()), 'system_prompt': args.system, 'model': model, 'conversations': [{'from': 'user', 'value': args.prompt}, {'from': 'assistant', 'value': response.strip()}]}
            f.write(json.dumps(format_data, ensure_ascii=False) + "\n")
        print(f'Save model response to file success! DIR: {cace_file_dir}')


if __name__ == "__main__":
    main()