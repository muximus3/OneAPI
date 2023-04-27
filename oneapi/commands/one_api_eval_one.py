import sys
import os
import argparse
from typing import List
sys.path.append(
    os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi.one_ai_eval import eval_one
from oneapi.one_api import OneAPITool
from oneapi.prompt_template import EvalPrompt
from oneapi import prompt_template

def main():
    parser = argparse.ArgumentParser(description="one-eval-line <command> [<args>]")
    parser.add_argument("-c", "--config_file", type=str, help="config file path", required=True)
    parser.add_argument("-tp", "--template_path",type=str, default=None, help="eval prompt template path", required=False)
    parser.add_argument("-d", "--verbose",type=bool, default=True, help="print every prompt and response detail", required=False)
    parser.add_argument("-m", "--model",type=str, default="", help="evaluate model name, e.g., gpt-35-turbo, gpt-4", required=False)
    parser.add_argument("-p", "--prompt", type=str, help="question", required=True)
    parser.add_argument("-a", "--answers", nargs='+', help="candidate answers",required=True)
    parser.add_argument("-t", "--target",type=str, default="", help="standard answer", required=False)
    parser.add_argument("-tt", "--temperature",type=float, default=0.1, help="0-1, higher temperature more random result", required=False)
    parser.add_argument("-mnt", "--max_new_tokens",type=int, default=2048, help="max output token length", required=False)

    args = parser.parse_args()
    tool = OneAPITool.from_config_file(args.config_file)
    if not args.template_path:
        template_path =  os.path.join(os.path.dirname(prompt_template.__file__), 'eval_prompt_template.json')
    else:
        template_path = args.template_path
    eval_prompter = prompt_template.EvalPrompt.from_config(template_path, verbose=args.verbose)
    score, raw_response = eval_one(
        eval_prompter=eval_prompter,
        question=args.prompt, 
        candidate_answers=args.answers, 
        target=args.target,
        api_tool=tool,
        model=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        )
 
    print(f'\nSCORE:\n{score}')


if __name__ == "__main__":
    main()
