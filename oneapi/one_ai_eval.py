# -*- coding: utf-8 -*-
import os
import pandas as pd
import traceback
from typing import Union, List
import random
from tqdm import tqdm
from oneapi.one_api import OneAPITool
import time
import sys

sys.path.append(
    os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from utils import extract_all_json, generate_letters, df_reader, df_saver

# eval_prompt_header = "You are a helpful and precise assistant for checking the quality of the answer."
eval_prompt_solve_first_instruction = """Please solve the [Question] independently to obtain the [Correct answer], and then evaluate and comment each [Candidate answer] based on the [Correct answer]. Finally, output all [Candidate answers] scores (0-1) in a summary format of {"number": "score"}, e.g, {"A": 0.2, "B": 0.8}."""
eval_prompt_with_target_instruction = """Please evaluate and comment each [Candidate answer] based on the [Correct answer]. Then output all [Candidate answer] scores (0-1) in a summary format of {"number": "score"}, e.g, {"A": 0.2, "B": 0.8}."""
eval_prompt_with_target_template = "[Question]: {q}\n[Correct answer]: {target}\n[Candidate answer]:\n{options}\n[System]:\n{instruction}"
eval_prompt_without_target_template = "[Question]: {q}\n[Candidate answer]:\n{options}\n[System]:\n{instruction}"


def eval_one(prompt: str,
             answers: List[str],
             target: Union[str, None],
             api_tool: OneAPITool,
             model: str,
             detail=False) -> Union[List[float], None]:
    answers_option_number = generate_letters(len(answers))
    format_option_data = '\n'.join([
        f'{answers_option_number[i]}. {answers[i]}'
        for i in range(len(answers))
    ])
    target = '' if target and len(target) > 1024 else target
    if target:
        eval_prompt = eval_prompt_with_target_template.format(
            q=prompt,
            target=target,
            options=format_option_data,
            instruction=eval_prompt_with_target_instruction)
    else:
        eval_prompt = eval_prompt_without_target_template.format(
            q=prompt,
            options=format_option_data,
            instruction=eval_prompt_solve_first_instruction)
    result = ''
    try:
        result = api_tool.simple_chat(eval_prompt, model=model)
        result_json = extract_all_json(result)
        if detail:
            print(
                f"{'-'*20}prompt detail{'-'*20}\n{eval_prompt}\n{'-'*20}prompt end{'-'*20}"
            )
            print(
                f"{'-'*20}'{model} response detail'{'-'*20}\n{result}\n{'-'*20}{model} response end{'-'*20}"
            )
        return [
            float(result_json.get(answers_option_number[i], 0))
            for i in range(len(answers))
        ]
    except Exception as e:
        print(f'error, request result:{result}, exception:{e}')
        traceback.print_exc()
        return None


def eval_one_group(data_group: pd.DataFrame, api_tool: OneAPITool, model: str,
                   detail: bool) -> Union[pd.DataFrame, None]:
    group = data_group.reset_index()
    prompt = group['prompt'].at[0]
    answers = group['output']
    if 'target' in data_group.keys():
        target = group['target'].at[0]
    else:
        target = ''
    scores =  eval_one(prompt, answers, target, api_tool, model, detail)
    if scores is not None:
        for i in range(len(scores)):
                group.at[i, 'score'] = scores[i]
        return group
    else:
        return None



def eval_one_file(config_file,
                  eval_data_path: str,
                  output_path: str = '',
                  model='',
                  eval_categories: List[str] = None,
                  sample_num=0,
                  request_interval=1,
                  retry=True,
                  detail=False):

    eval_data = df_reader(eval_data_path)
    assert 'output' in eval_data.keys(), f'format error, eval data must include column:\"prompt\", \"output\"'
    eval_data = eval_data.fillna('')
    if 'instruction' in eval_data.keys():
        eval_data['prompt'] = eval_data['instruction'].str.cat(eval_data['input'], sep=' ')
    # filter specific categories
    if eval_categories is not None and len( eval_categories) > 0 and 'category' in eval_data.keys():
        eval_data = eval_data[eval_data['category'].isin(eval_categories)]

    grouped = eval_data.groupby(by=['prompt'])
    if sample_num > 0:
        sample_keys = random.sample(grouped.groups.keys(), sample_num)
    else:
        sample_keys = grouped.groups.keys()

    tool = OneAPITool.from_config_file(config_file=config_file)
    eval_results = []
    failed_results = []
    for key in tqdm(sample_keys):
        group = grouped.get_group(key)
        result = eval_one_group(group, tool, model, detail)
        if result is not None:
            eval_results.append(result)
        else:
            failed_results.append(group)
        time.sleep(request_interval)
    if len(failed_results) > 0 and retry:
        for group in failed_results.copy():
            result = eval_one_group(group, tool, model, detail)
            if result is not None:
                eval_results.append(result)
                failed_results.remove(group)
            time.sleep(request_interval)
    eval_results.extend(failed_results)
    eval_results_df = pd.concat(eval_results)
    # log score results
    if 'model' in eval_data.keys():
        score_models = eval_results_df.groupby('model')['score'].sum()
        print(f'{"-"*20}SCORE BY MODEL{"-"*20}\n{score_models.to_markdown()}')
        if 'category' in eval_data.keys():
            score_category = eval_results_df.groupby([
                'model', 'category'
            ])['score'].sum().reset_index().sort_values(by='model')
            print(
                f'{"-"*20}SCORE BY CATEGORY{"-"*20}\n{score_category.to_markdown(index=False)}'
            )
    print(f'Eval model: {model}')
    print(f'Fail requests: {len(failed_results)}/{len(sample_keys)}')
    # save results
    if output_path:
        print(f'Saving score reuslt: {output_path}')
        df_saver(eval_results_df, output_path)
        print(f'Saving success.')
