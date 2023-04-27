import os
import sys
import argparse
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi.one_ai_eval import eval_one_file, EvalConfig
from oneapi import prompt_template
def main():
    parser = argparse.ArgumentParser(description="one-eval-file [<args>]")
    parser.add_argument("-c", "--config_file", type=str, help="config file path", required=True)
    parser.add_argument("-tp", "--template_path",type=str, default=None, help="eval prompt template path", required=False)
    parser.add_argument("-ep", "--eval_data_path",type=str, help="", required=True)
    parser.add_argument("-op", "--output_path",type=str, default="", help="", required=False)
    parser.add_argument("-m", "--model",type=str, default="", help="eval_model_name", required=False)
    parser.add_argument("-ec", "--eval_categories", default=None, nargs="+",help="only evaluate chosen categories", required=False)
    parser.add_argument("-sn", "--sample_num", type=int, default=0, help="", required=False)
    parser.add_argument("-i", "--interval",type=int, default=1, help="request interval, gpt-4 need longer interval, e.g.,10s", required=False)
    parser.add_argument("-r", "--retry",type=bool, default=True, help="", required=False)
    parser.add_argument("-d", "--verbose",type=bool, default=True, help="print every prompt and response detail", required=False)
    parser.add_argument("-tt", "--temperature",type=float, default=0.1, help="0-1, higher temperature more random result", required=False)
    parser.add_argument("-mnt", "--max_new_tokens",type=int, default=2048, help="max output token length", required=False)

    args = parser.parse_args()
    if not args.template_path:
        template_path =  os.path.join(os.path.dirname(prompt_template.__file__), 'eval_prompt_template.json')
    else:
        template_path = args.template_path
    eval_prompter = prompt_template.EvalPrompt.from_config(template_path, verbose=args.verbose)
    eval_one_file(
        EvalConfig(
        eval_prompter=eval_prompter,
        api_config_file=args.config_file, 
        eval_data_path=args.eval_data_path, 
        output_path=args.output_path, 
        model=args.model,
        eval_categories=args.eval_categories,
        sample_num=args.sample_num, 
        request_interval=args.interval,
        retry=args.retry,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        ))
    

if __name__ == "__main__":
    main()