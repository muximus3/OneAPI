# -*- coding: utf-8 -*-
import json
from typing import Optional, Sequence, List
import openai
import anthropic
from pydantic import BaseModel
from abc import ABC, abstractmethod
import sys
import os
from typing import Callable, Optional, Sequence, List
import tiktoken
import asyncio
import transformers
import logging
from openai.openai_object import OpenAIObject
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from oneapi.one_api import batch_chat, OneAPITool, compile_jinja_template, render_jinja_template

def print_special_token(tokenizer_hf: transformers.PreTrainedTokenizer):
    print(f"""tokenizer:\n 
          vocab_size:{len(tokenizer_hf)},
          eos:{tokenizer_hf.eos_token},{tokenizer_hf.eos_token_id},
          bos:{tokenizer_hf.bos_token},{tokenizer_hf.bos_token_id},
          pad:{tokenizer_hf.pad_token},{tokenizer_hf.pad_token_id},
          unk:{tokenizer_hf.unk_token},{tokenizer_hf.unk_token_id},
          mask:{tokenizer_hf.mask_token},{tokenizer_hf.mask_token_id},
          cls:{tokenizer_hf.cls_token},{tokenizer_hf.cls_token_id},
          sep:{tokenizer_hf.sep_token},{tokenizer_hf.sep_token_id},
          all_special:{tokenizer_hf.all_special_tokens},{tokenizer_hf.all_special_ids},
          """)



if __name__ == "__main__":
   claude_config = '../ant/config/anthropic_config_personal.json'
   openai_config = '../ant/config/openapi_official_chenghao.json'
   azure_config = '../ant/config/openapi_azure_config_xiaoduo_dev5.json'
   config_file = openai_config
   tool = OneAPITool.from_config(api_key="", api_base="http://10.0.0.135:8080", api_type="huggingface")

   msgs =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "让我们来角色扮演。"},
           ]
#    print(prompt)
   res = tool.chat(msgs, stream=False)
   print('================res:')
   print(res)
#    print(tool.chat(msgs))
#    ep = [msg[-1], {"from": "assistant", "value":res}]
#    print(json.dumps(ep, ensure_ascii=False))

