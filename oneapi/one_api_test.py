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
from oneapi.one_api import batch_chat, OneAPITool

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
   tool = OneAPITool.from_config_file(config_file)
   {"system_prompt": "", "conversations": [{"from": "human", "value": "这是一个基于提示的简短故事：\n\n外星飞船突破大气层，以炽热的火光划过天际。其系统正失效，飞船失去了控制。当其向地面坠落时，外星人拼命试图启动紧急降落程序，但为时已晚。飞船轰隆一声坠毁在地球上，留下一个巨大的坑洞。\n\n外星人从残骸中走出来，晕头转向。它艰难地走到了地面上，试图确定自己的位置。它遇到的第一件事是一个小小的人类村庄，它谨慎地走近。\n\n其中一个村民看到了外星人，走向它。外星人用它的大而多面的眼睛看着这个人类，并说：“你好，基于碳的生命形式。”\n\n这个人类惊讶地凝视着，不知道如何对待这个奇怪的生物。外星人继续说：“我和平而来。我的名字是X'arzor，我在你们的星球上坠毁了。你能帮助我吗？”\n\n人类犹豫了一会儿，仍在试图理解发生了什么。但最终，他点了点头，并提议将外星人带到他的村庄，帮助他重新站稳脚跟。\n\n在接下来的几天里，X'arzor 对人类和它居住的星球有了较多了解。他对地球上的生命的多样性感到惊讶，对人类文化的复杂性感到着迷。他甚至开始与村民们建立起了联系，感觉已经找到了一个新家。\n\n但最终，X'arzor 知道他必须离开。他的同伴肯定在寻找他，他有责任返回。当他向村民们告别时，他转向最先迎接他的人类，说：“谢谢你，基于碳的生命形式。你对我显示了善意和热情，我永远不会忘记。我希望我们未来还能相遇。”\n\n带着最后的点头，X'arzor 登上了他的飞船，消失在星空中，离开了人类。但他心中明白，他将永远携带着地球的一部分，与村民们建立的联系永远不会真正消失。"}, {"from": "assistant", "value": "似乎你误会了我的原话。我并无法为创建/生成故事提示提供C#代码。我的意思是，如果你能提供具体的代码问题或你想要实现的具体功能，我可以根据你需要的帮助为你提供相应的C#代码。对于你提出的关于创作一个外星人故事的请求，我可以提供一些修改和扩展的建议，但这并不涉及C#代码。"}], "id": "8KDaUGk", "dataset_name": "wizard", "model": "gpt-4-0613"}

   msg =  [{"from": "system", "value": "这是一个好奇心很重的问题少年在向你提问。"},
           {"from": "human", "value": "让我们来角色扮演。"},
           ]
   
   print(tool._preprocess_claude_prompt(msg))
   res = tool.chat(msg, stream=True)
   print(res)
   for i in res:
        print(i)
#    ep = [msg[-1], {"from": "assistant", "value":res}]
#    print(json.dumps(ep, ensure_ascii=False))

