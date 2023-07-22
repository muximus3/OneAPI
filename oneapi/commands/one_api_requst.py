import sys
import os
import argparse

sys.path.append(
    os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi import OneAPITool


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
    print(
        f"\n{'-'*20} prompt detail 🚀  {'-'*20}\n\n{args.prompt}\n\n{'-'*20} prompt end {'-'*20}"
    )
    print(
        f"{'-'*20} {args.model} response ⭐️ {'-'*20}\n\n{response}\n\n{'-'*20} response end {'-'*20}\n\n"
    )


if __name__ == "__main__":
    main()