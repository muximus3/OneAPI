# -*- coding: utf-8 -*-
import os
import sys
import inspect
from docstring_parser import parse
import enum
import re
from pathlib import Path
import fnmatch
import json
try:
    from jinja2.exceptions import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment
except ImportError:
    raise ImportError("comile requires jinja2 to be installed.")

sys.path.append(os.path.normpath(f'{os.path.dirname(os.path.abspath(__file__))}/..'))


def compile_jinja_template(chat_template):
    def raise_exception(message):
        raise TemplateError(message)
    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(chat_template)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def map_conversations_format(conversations):
    if len(conversations) > 0 and 'from' and 'value' in conversations[0]:
        conversations = [{'role': item['from'], 'content': item['value']} for item in conversations]
    return conversations

def correct_zh_punctuation(text):
    # 中文字符的Unicode范围
    chinese_pattern = "[\u4e00-\u9fa5]"
    
    # 英文标点符号
    english_punctuations = ",.!?;:"
    
    # 中文标点符号
    chinese_punctuations = "，。！？；："
    
    # 找到所有中文字符后面跟着英文标点的位置
    for m in re.finditer(chinese_pattern + "[" + english_punctuations + "]", text):
        pos = m.start() + 1
        # 将英文标点替换为对应的中文标点
        text = text[:pos] + chinese_punctuations[english_punctuations.index(text[pos])] + text[pos+1:]
    
    return text

def find_files_unrecu(directory, pattern):
    """ finds all files matching the pattern."""
    if not Path(directory).is_dir():
        raise AssertionError('NOT A VALID DIRECTORY!!')
    files = []
    for filename in fnmatch.filter(os.listdir(directory), pattern):
        files.append(os.path.join(os.path.abspath(directory), filename))
    return files

def load_jsonl(data_path: str, obj_item: bool=True):
    if obj_item: 
        data = []
        for i, l in enumerate(open(data_path, "r")):
            try:
                data.append(json.loads(l)) 
            except json.decoder.JSONDecodeError as e:
                print(f'load file:{data_path}, line {i} error: {l}')
                continue
        return data
    return [l for l in open(data_path, "r")] 

def python_type_to_json_type(python_type):
    if python_type in [str, 'str']:
        return 'string'
    elif python_type in [int, 'int']:
        return 'integer'
    elif python_type in [float, 'float']:
        return 'number'
    elif python_type in [bool, 'bool']:
        return 'boolean'
    elif python_type in [list, 'list', 'List']:
        return 'array'
    elif python_type in [dict, 'dict']:
        return 'object'
    else:
        if 'typing.List' in str(python_type) or 'typing.Iterable' in str(python_type):
            return 'array'
        return 'string'

def generate_function_description(func) -> dict:
    # 获取函数的元数据
    func_name = func.__name__
    func_doc = inspect.getdoc(func)
    sig = inspect.signature(func)
    func_args = sig.parameters
    return_type = sig.return_annotation

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    function_description = {
        "name": func_name,
        "description": "",
        "parameters": parameters,
    }
    

    for arg_name, arg in func_args.items():
        if arg_name == "self":
            continue
        default_value = arg.default if arg.default != inspect.Parameter.empty else None
        type_annotation = arg.annotation if arg.annotation != inspect.Parameter.empty else None
        param_description = {}
        # if default_value is not None:
            # param_description["default"] = default_value
        if type_annotation is not None:
            json_schema_type_name = python_type_to_json_type(type_annotation)
            param_description["type"] = json_schema_type_name
            if json_schema_type_name == 'array':
                if '__args__' in dir(type_annotation) and len(type_annotation.__args__) > 0:
                    param_description["items"] = {
                        "type": python_type_to_json_type(type_annotation.__args__[0])
                    }
                else:
                    param_description["items"] = {
                        "type": "string"
                    }
            if isinstance(type_annotation, enum.EnumMeta):
                param_description["enum"] = [e.name for e in type_annotation]
        else:
            if default_value is not None:
                param_description["type"] = python_type_to_json_type(type(default_value))
        
        # Add parameter description to function description
        parameters["properties"][arg_name] = param_description
        if arg.default == inspect.Parameter.empty:
            parameters["required"].append(arg_name)

    if func_doc is not None:
        parsed_doc = parse(func_doc)
        func_desc = parsed_doc.short_description
        params = parsed_doc.params
        if func_desc is not None:
            function_description["description"] = func_desc
        for param in params:
            param_name = param.arg_name
            param_desc = param.description
            param_type = param.type_name
            if param_name in parameters["properties"]:
                if param_desc is not None:
                    parameters["properties"][param_name]["description"] = param_desc
                # if param_type is not None:
                #     parameters["properties"][param_name]["type"] = python_type_to_json_type(param_type)

    # Parse to json and return
    return function_description

    