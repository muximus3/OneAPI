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
   config_files = ['../ant/config/huggingface_config_sft_devc.json', '../ant/config/huggingface_config_sft_gpu01.json', '../ant/config/huggingface_config_sft_devb.json', '../ant/config/huggingface_config_sft_devc.json', '../ant/config/huggingface_config_sft_gpu01.json', '../ant/config/huggingface_config_sft_devb.json','../ant/config/huggingface_config_sft_devc.json', '../ant/config/huggingface_config_sft_gpu01.json', '../ant/config/huggingface_config_sft_devb.json', '../ant/config/huggingface_config_sft_devc.json', '../ant/config/huggingface_config_sft_gpu01.json', '../ant/config/huggingface_config_sft_devb.json']   
   tool = OneAPITool.from_config(api_key="", api_base="http://10.0.0.135:8090", api_type="huggingface")

   msgs =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "让我们来角色扮演。"}]
   msgs1 =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "1+1=?"}]
   msgs2 =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "2+2=?"}]
   msgs3 =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "3+2=?"}]
   msgs4 =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "3+4=?"}]
   msgs5 =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "3+12=?"}]
   msgs6 =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "13+2=?"}]
   msgs7 =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "1-1=?"}]
   msgs8 =  [{"role": "system", "content": "这是一个好奇心很重的问题少年在向你提问。"},
           {"role": "user", "content": "100-1=?"}]
   print(asyncio.run(batch_chat(config_files * 1, [msgs, msgs1, msgs2, msgs3, msgs4, msgs5, msgs6, msgs7, msgs8])))

