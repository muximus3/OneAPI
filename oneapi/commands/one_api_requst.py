import sys
import os
import argparse
sys.path.append(
    os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi import OneAPITool

def main():
    parser = argparse.ArgumentParser(description="one-api [<args>]")
    parser.add_argument("-c", "--config_file", type=str, help="config file path", required=True)
    parser.add_argument("-p", "--prompt", type=str, help="question", required=True)
    parser.add_argument("-m", "--model",type=str, default="", help="evaluate model name, e.g., gpt-35-turbo, gpt-4", required=False)
    parser.add_argument("-t", "--temperature",type=float, default=1., help="0-1", required=False)
    parser.add_argument("-mnt", "--max_new_tokens",type=int, default=2048, help="", required=False)
    args = parser.parse_args()
    tool = OneAPITool.from_config_file(args.config_file)
    result = tool.simple_chat(
        prompt=args.prompt, 
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        )
    print(f'{args.model} response:\n{result}')


if __name__ == "__main__":
    main()