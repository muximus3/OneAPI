# -*- coding: utf-8 -*-
import os
import sys
import inspect
from docstring_parser import parse
import enum
sys.path.append(os.path.normpath(f'{os.path.dirname(os.path.abspath(__file__))}/..'))

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