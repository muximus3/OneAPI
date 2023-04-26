import os
import sys
import argparse
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi.one_ai_eval import eval_one_file

def main():
    
    parser = argparse.ArgumentParser(description="one-eval-file [<args>]")
    parser.add_argument("--config_file", type=str, help="config file path", required=True)
    parser.add_argument("--eval_data_path",type=str, help="", required=True)
    parser.add_argument("--output_path",type=str, default="", help="", required=False)
    parser.add_argument("--model",type=str, default="", help="eval_model_name", required=False)
    parser.add_argument("--eval_categories", default=None, nargs="+",help="", required=False)
    parser.add_argument("--sample_num", type=int, default=0, help="", required=False)
    parser.add_argument("--interval",type=int, default=1, help="request interval, gpt-4 need longer interval, e.g.,10s", required=False)
    parser.add_argument("--retry",type=bool, default=True, help="", required=False)
    parser.add_argument("--detail",type=bool, default=True, help="print every prompt and response detail", required=False)

    args = parser.parse_args()
    eval_one_file(
        config_file=args.config_file, 
        eval_data_path=args.eval_data_path, 
        output_path=args.output_path, 
        model=args.model,
        eval_categories=args.eval_categories,
        sample_num=args.sample_num, 
        request_interval=args.interval,
        retry=args.retry,
        detail=args.detail
        )
    

if __name__ == "__main__":
    main()