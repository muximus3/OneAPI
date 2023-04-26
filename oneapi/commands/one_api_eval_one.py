import sys
import os
import argparse
from typing import List
sys.path.append(
    os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi.one_ai_eval import eval_one
from oneapi.one_api import OneAPITool

def main():
    parser = argparse.ArgumentParser(description="one-eval-line <command> [<args>]")
    parser.add_argument("--config_file", type=str, help="config file path", required=True)
    parser.add_argument("--prompt", type=str, help="question", required=True)
    parser.add_argument("--answers", help="answer list", nargs='+',required=True)
    parser.add_argument("--target",type=str, default="", help="standard answer", required=False)
    parser.add_argument("--model",type=str, default="", help="evaluate model name, e.g., gpt-35-turbo, gpt-4", required=False)
    parser.add_argument("--detail",type=bool, default=True, help="print every prompt and response detail", required=False)
    args = parser.parse_args()
    tool = OneAPITool.from_config_file(args.config_file)
    result = eval_one(
        prompt=args.prompt, 
        answers=args.answers, 
        target=args.target, 
        model=args.model,
        api_tool=tool,
        detail=args.detail  
        )
 
    print('\nSCORE:')
    print(result)


if __name__ == "__main__":
    main()
