import os
import numpy as np
import ast
import jsonschema
from jsonschema import validate
from utils.json_validate import schema
import time
import requests
import json
from arguments import get_args
from urllib.parse import urlsplit, urlunsplit


gpt_name = [
            'text-davinci-003',
            'gpt-3.5-turbo-0125',
            'gpt-4o',
            'gpt-4o-mini'
        ]           

args = get_args()

api_key = os.environ.get("VLN_API_KEY") or args.api
if not api_key or api_key == "xx-xxxx":
    api_key = (
        os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or api_key
    )

api_base = (
    os.environ.get("VLN_API_BASE")
    or os.environ.get("DEEPSEEK_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://api.openai.com/v1"
)

text_model = os.environ.get("VLN_MODEL") or "gpt-4o"

def _env_bool(name, default=True):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

supports_temperature = _env_bool("VLN_MODEL_SUPPORTS_TEMPERATURE", True)

def _model_options(model_name):
    raw = os.environ.get("VLN_MODEL_OPTIONS")
    if raw:
        try:
            options = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid VLN_MODEL_OPTIONS: {exc}") from exc
        if not isinstance(options, dict):
            raise ValueError("VLN_MODEL_OPTIONS must be a JSON object")
        return options
    if model_name.startswith("deepseek") or "deepseek" in api_base:
        return {"thinking": {"type": "disabled"}}
    return {}

import openai
from openai.error import OpenAIError
openai.api_key = api_key
openai.api_base = api_base


def trace_configuration():
    """Non-secret effective request settings, for the structured run manifest."""
    def redact(value):
        if isinstance(value, dict):
            return {key: ('<redacted>' if key.lower() in {
                'api', 'api_key', 'authorization', 'password', 'secret', 'token',
                'access_token', 'api_token'} else redact(item))
                    for key, item in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    vision_model = os.environ.get("VLN_MODEL") or gpt_name[args.gpt_type]
    configurations = {}
    for role, model, limit, temperature in (
            ('parsing', text_model, 512, 0.5),
            ('generator', vision_model, 100, 1.1),
            ('discriminator', vision_model, 100, 1.1)):
        options = _model_options(model).copy()
        options.setdefault('max_tokens', limit)
        if supports_temperature:
            options.setdefault('temperature', temperature)
        else:
            options.pop('temperature', None)
        options['model'] = model
        options.pop('messages', None)
        options.pop('response_format', None)
        if role != 'discriminator':
            options['response_format'] = {'type': 'json_object'}
        configurations[role] = redact(options)
    # The parsing request forces 512 tokens only before applying model options,
    # just as chat_with_gpt does. No authorization headers or keys are included.
    url = urlsplit(api_base)
    safe_base = urlunsplit((url.scheme, url.netloc.rsplit('@', 1)[-1], url.path, '', ''))
    return {'api_base': safe_base, 'supports_temperature': supports_temperature,
            'requests': configurations}



def chat_with_gpt(chat_history):
    retries = 10    
    while retries > 0:  
        try: 
            params = {
                "model": text_model,
                "response_format": {"type": "json_object"},
                "messages": chat_history,
                "max_tokens": 512,
            }
            params.update(_model_options(text_model))
            params["model"] = text_model
            params["messages"] = chat_history
            params["response_format"] = {"type": "json_object"}
            if supports_temperature:
                params.setdefault("temperature", 0.5)
            else:
                params.pop("temperature", None)
            response = openai.ChatCompletion.create(**params)

            response_message = response.choices[0].message['content']
            # print(response_message)
            print(text_model + " response: ")
            try:
                ground_json = ast.literal_eval(response_message)
                break
                
            except (SyntaxError, ValueError) as e:
                print(e)
        except OpenAIError as e:
            if e:
                print(e)
                print('Timeout error, retrying...')    
                retries -= 1
                time.sleep(5)
            else:
                raise e
            
    print(response_message)
    return response_message

def chat_with_gpt4v(chat_history, gpt_type = args.gpt_type, json_mode = True):
    
    model_name = os.environ.get("VLN_MODEL") or gpt_name[gpt_type]
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    retries = 10    
    while retries > 0:  
        try: 
            payload = {
                "model": model_name,
                "messages": chat_history,
                "max_tokens": 100,
            }
            payload.update(_model_options(model_name))
            payload["model"] = model_name
            payload["messages"] = chat_history
            payload.setdefault("max_tokens", 100)
            if supports_temperature:
                payload.setdefault("temperature", 1.1)
            else:
                payload.pop("temperature", None)
            if json_mode:
                payload["response_format"] = { "type": "json_object" }
            else:
                payload.pop("response_format", None)
            
            response = requests.post(api_base.rstrip("/") + "/chat/completions", headers=headers, json=payload)

            response_message = response.json()["choices"][0]["message"]["content"]
            # print(response_message)
            print(model_name + " response: ")
            if not json_mode:
                if response_message and response_message.strip():
                    break
                retries -= 1
                time.sleep(5)
                continue
            try:
                ground_json = ast.literal_eval(response_message)
                break
                
            except (SyntaxError, ValueError) as e:
                print(e)
        except OpenAIError as e:
            if e:
                print(e)
                print('Timeout error, retrying...')    
                retries -= 1
                time.sleep(5)
            else:
                raise e
            
    print(response_message)
    return response_message

def chat_with_llama3(user_prompt, chat_history):

    chat_history.append({"role": "user", "content": user_prompt})

    retries = 10    
    while retries > 0:  
        
        url = 'http://localhost:11434/api/chat'
        data = {
            "model": "llama3",
            "messages": chat_history,
            "stream": False,
            "format": "json",
            "options": {
                "seed": 101,
                "temperature": 0
            }
        }
        
        response = requests.post(url, json=data)
        decoded_string = response.content.decode('utf-8')
        response_message = json.loads(decoded_string)["message"]["content"]
        ground_json = ast.literal_eval(response_message)
                
        try: 
            validate(instance = ground_json, schema = schema)
        except jsonschema.exceptions.ValidationError as err:
            print("JSON data is invalid", err)
            print('Json format error, retrying...')    
            retries -= 1

            
    chat_history.append({"role": "assistant", "content": response_message})
    print(response_message)
    return response_message
