import os
import sys
from typing import List
import abc
import json
sys.path.append(
    os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/../../"))
from oneapi.utils.data_utils import load_json, generate_letters, extract_last_json, extract_numbers


class Prompter(abc.ABC):
    @abc.abstractmethod
    def generate_prompt(**kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def extract_result_from_response(response):
        raise NotImplementedError

class EvalPrompt(Prompter):
    __slots__ = ("eval_with_target_instruction",
                 "eval_without_target_instruction",
                 "eval_with_target_template", "eval_without_target_template",
                 "score_output_type", "response_score_position", "_verbose")

    @classmethod
    def from_config(cls,
                    prompt_template_file: str = "",
                    verbose: bool = False):
        if not os.path.exists(prompt_template_file):
            raise ValueError(f'File not exists:{prompt_template_file}')
        template_data = load_json(prompt_template_file)
        eval_with_target_template = template_data['eval_with_target_template']
        eval_without_target_template = template_data[
            'eval_without_target_template']
        eval_with_target_instruction = template_data.get(
            'eval_with_target_instruction', '')
        eval_without_target_instruction = template_data.get(
            'eval_without_target_instruction', '')
        score_output_type = template_data.get('score_output_type', 'json')
        response_score_position = template_data.get('response_score_position',
                                                    'tail')

        instruction_header = template_data.get('header', '')
        if verbose:
            print(f'Using prompt templat:\n {json.dumps(template_data, sort_keys=True, indent=2)}')
        return cls(eval_with_target_template, eval_without_target_template,
                   eval_with_target_instruction,
                   eval_without_target_instruction, score_output_type,
                   response_score_position, instruction_header, verbose)

    def __init__(self,
                 eval_with_target_template: str,
                 eval_without_target_template: str,
                 eval_with_target_instruction: str,
                 eval_without_target_instruction: str,
                 score_output_type: str,
                 response_score_position: str,
                 instruction_header: str,
                 verbose: bool = False) -> None:
        self._verbose = verbose
        self.eval_with_target_template = eval_with_target_template
        self.eval_without_target_template = eval_without_target_template
        self.eval_with_target_instruction = eval_with_target_instruction
        self.eval_without_target_instruction = eval_without_target_instruction
        self.score_output_type = score_output_type
        self.response_score_position = response_score_position
        if instruction_header and eval_with_target_instruction and eval_without_target_instruction:
            self.eval_with_target_instruction = f'{instruction_header} {eval_with_target_instruction}'
            self.eval_without_target_instruction = f'{instruction_header} {eval_without_target_instruction}'

    def generate_prompt(self, question: str, target: str,
                        candidate_answers: List[str]) -> str:
        """
        Generate a prompt based on the given question, target, and candidate answers.
        """
        candidate_answer_numbers = generate_letters(len(candidate_answers))
        format_option_data = '\n'.join([
            f'{candidate_answer_numbers[i]}. {candidate_answers[i]}'
            for i in range(len(candidate_answers))
        ])
        if target:
            if self.eval_with_target_instruction:
                eval_prompt = self.eval_with_target_template.format(
                    q=question,
                    target=target,
                    options=format_option_data,
                    instruction=self.eval_with_target_instruction)
            else:
                eval_prompt = self.eval_with_target_template.format(
                    q=question, target=target, options=format_option_data)
        else:
            if self.eval_without_target_instruction:
                eval_prompt = self.eval_without_target_template.format(
                    q=question,
                    options=format_option_data,
                    instruction=self.eval_without_target_instruction)
            else:
                eval_prompt = self.eval_without_target_template.format(
                    q=question, options=format_option_data)
        if self._verbose:
            print(
                f"\n{'-'*20} prompt detail 🚀  {'-'*20}\n\n{eval_prompt}\n\n{'-'*20} prompt end {'-'*20}"
            )
        return eval_prompt

    def extract_result_from_response(self, response: str) -> List[float]:
        """
        Extract the result (scores) from the given model response.
        """
        if self._verbose:
            print(
                f"{'-'*20} response detail ⭐️ {'-'*20}\n\n{response}\n\n{'-'*20} response end {'-'*20}\n"
            )
        if self.score_output_type == 'json':
            result_json = extract_last_json(response.strip())
            scores = list(map(lambda x: float(x), result_json.values()))
        elif self.score_output_type == 'list':
            score_positon = 0 if self.response_score_position == 'head' else -1
            scores = extract_numbers(response.split('\n')[score_positon])
        return scores