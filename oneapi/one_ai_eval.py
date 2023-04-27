# -*- coding: utf-8 -*-
import os
import pandas as pd
import traceback
from typing import Union, List
import random
from tqdm import tqdm
from oneapi.one_api import OneAPITool
from typing import Optional, Tuple
import time
import sys
from dataclasses import dataclass
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from oneapi.utils import df_reader, df_saver
from oneapi.prompt_template import Prompter


def eval_one(eval_prompter: Prompter,
             question: str,
             candidate_answers: List[str],
             target: Union[str, None],
             api_tool: OneAPITool,
             model: str,
             temperature=0.1,
             max_new_tokens=2048) -> Tuple[Union[List[float], None], str]:
    raw_response = ''
    eval_prompt = eval_prompter.generate_prompt(question, target,
                                                candidate_answers)
    try:
        raw_response = api_tool.simple_chat(eval_prompt,
                                      model=model,
                                      temperature=temperature,
                                      max_new_tokens=max_new_tokens)
        scores = eval_prompter.extract_result_from_response(raw_response)
        return scores, raw_response
    except Exception as e:
        print(f'error, request result:{raw_response}, exception:{e}')
        traceback.print_exc()
        return None, raw_response


def eval_one_group(
    api_tool: OneAPITool,
    eval_prompter: Prompter,
    data_group: pd.DataFrame,
    model: str,
    temperature=0.1,
    max_new_tokens=2048,
) -> Union[pd.DataFrame, None]:
    group = data_group.reset_index()
    question = group['question'].unique()[0]
    candidate_answers = group['output']
    if 'target' in data_group.keys():
        target = group['target'].unique()[0]
    else:
        target = ''
    scores, raw_response = eval_one(eval_prompter=eval_prompter,
                      question=question,
                      candidate_answers=candidate_answers,
                      target=target,
                      api_tool=api_tool,
                      model=model,
                      temperature=temperature,
                      max_new_tokens=max_new_tokens)
    if scores is not None and len(scores) == len(candidate_answers):
        group.at[0, 'raw_response'] = raw_response
        for i in range(len(scores)):
            group.at[i, 'score'] = scores[i]
        return group
    else:
        return None



def prepare_eval_data(eval_data: pd.DataFrame, eval_categories: Optional[List[str]] = None, sample_num: int=0) -> List[pd.DataFrame]:
    if 'score' in eval_data.keys():
        eval_data['score'] = 0.
    eval_data = eval_data.fillna('')
    if len({'instruction', 'input', 'output'} - set(eval_data.keys())) == 0:
        eval_data['question'] = eval_data['instruction'].str.cat(
            eval_data['input'], sep=' ')
    elif len({'prompt', 'output'} - set(eval_data.keys())) == 0:
        eval_data['quesion'] = eval_data['prompt'].copy()
    elif len({'question', 'output'} - set(eval_data.keys())) == 0:
        pass
    else:
        raise KeyError(
            f'Eval data columns must be either: ["instruction", "input", "output"] or ["prompt", "output"] or ["question", "output"]'
        )
    # filter specific categories
    if eval_categories is not None and len(
            eval_categories) > 0 and 'category' in eval_data.keys():
        eval_data = eval_data[eval_data['category'].isin(eval_categories)]
    grouped = eval_data.groupby(by=['question'])
    if sample_num > 0:
        sample_keys = random.sample(grouped.groups.keys(), sample_num)
    else:
        sample_keys = grouped.groups.keys()
    return [grouped.get_group(key) for key in sample_keys]

def log_score_results(eval_results_df: pd.DataFrame):
    if 'model' in eval_results_df.keys():
        print( eval_results_df.groupby('model')['score'])
        score_models = eval_results_df.groupby('model')['score'].apply(lambda x: f'{x.sum():.1f}/{len(x)}')
        print(f'{"-"*20} SCORE BY MODEL {"-"*20}\n{score_models.to_markdown()}')
        if 'category' in eval_results_df.keys():
            score_category = eval_results_df.groupby([
                'model', 'category'
            ])['score'].apply(lambda x: f'{x.sum():.1f}/{len(x)}').reset_index().sort_values(by='model')
            print(
                f'\n{"-"*20} SCORE BY CATEGORY {"-"*20}\n{score_category.to_markdown(index=False)}'
            )

def save_results(eval_results_df: pd.DataFrame, output_path: str):
    print(f'Saving score result: {output_path}')
    df_saver(eval_results_df, output_path)
    print(f'Saved successfully.')

@dataclass
class EvalConfig:
    api_config_file: str
    eval_prompter: Prompter
    eval_data_path: str
    output_path: str = ''
    model: str = ''
    eval_categories: Optional[List[str]] = None
    sample_num: int = 0
    request_interval: int = 1
    retry: bool = True
    verbose: bool = False
    temperature: float = 0.1
    max_new_tokens: int = 2048

def eval_one_file(
    eval_config: EvalConfig
):
    # Preparing data
    eval_data = df_reader(eval_config.eval_data_path)
    eval_groups = prepare_eval_data(eval_data, eval_config.eval_categories, eval_config.sample_num)

    # Init api tool and prompter
    tool = OneAPITool.from_config_file(config_file=eval_config.api_config_file)

    eval_results = []
    failed_results = []
    for group in tqdm(eval_groups):
        result = eval_one_group(tool, eval_config.eval_prompter,  group,  eval_config.model, eval_config.temperature, eval_config.max_new_tokens)
        if result is not None:
            eval_results.append(result)
        else:
            failed_results.append(group)
        time.sleep(eval_config.request_interval)

    # Retry failed requests
    if len(failed_results) > 0 and eval_config.retry:
        for group in tqdm(failed_results.copy(), desc='RETRY'):
            result = eval_one_group(tool, eval_config.eval_prompter,  group,  eval_config.model, eval_config.temperature, eval_config.max_new_tokens)
            if result is not None:
                eval_results.append(result)
                failed_results = [df for df in failed_results if not df.equals(group)]
            time.sleep(eval_config.request_interval)

    eval_results.extend(failed_results)
    eval_results_df = pd.concat(eval_results)
    log_score_results(eval_results_df=eval_results_df)
    # Log score results
    print(f'Eval model: {eval_config.model}')
    print(f'Failed requests: {len(failed_results)}/{len(eval_groups)}')
    # Save results
    if eval_config.output_path:
        save_results(eval_results_df=eval_results_df, output_path=eval_config.output_path)
